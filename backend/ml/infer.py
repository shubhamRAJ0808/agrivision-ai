"""
Loads the trained LeafDiseaseCNN checkpoint and exposes predict(image) for the API.
"""
import io
import json
import os

import torch
import torch.nn.functional as F
from PIL import Image

from .model import LeafDiseaseCNN
from .dataset import load_image_as_tensor

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(HERE, "checkpoints")
MODEL_PATH = os.path.join(CKPT_DIR, "disease_model.pt")
LABELS_PATH = os.path.join(CKPT_DIR, "labels.json")

_model = None
_labels = None

# Human-readable label + practical advice per class.
LABEL_INFO = {
    "Tomato___healthy": {
        "display": "Healthy",
        "advice": "No signs of disease detected. Maintain current watering and fertilizing schedule.",
    },
    "Tomato___Early_blight": {
        "display": "Early Blight",
        "advice": "Remove affected lower leaves, improve airflow, and apply a copper-based or chlorothalonil fungicide.",
    },
    "Tomato___Late_blight": {
        "display": "Late Blight",
        "advice": "Act quickly — late blight spreads fast in humid weather. Remove infected plants/leaves and apply a targeted fungicide (e.g. mancozeb).",
    },
    "Tomato___Leaf_Mold": {
        "display": "Leaf Mold",
        "advice": "Reduce humidity around plants, increase spacing/ventilation, and apply a fungicide labeled for leaf mold.",
    },
    "Other___not_tomato_leaf": {
        "display": "Not a tomato leaf / unclear photo",
        "advice": "This doesn't look like a close-up of a tomato leaf. Take a clear, well-lit photo of ONE tomato leaf "
                  "filling most of the frame (top side, in focus), then try again.",
    },
}

LOW_CONFIDENCE = 0.70   # below this the answer is flagged as uncertain


def _ensure_loaded():
    global _model, _labels
    if _model is not None:
        return
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        raise RuntimeError(
            "Trained model not found. Run `python3 ml/train.py` from the backend/ directory first."
        )
    with open(LABELS_PATH) as f:
        _labels = json.load(f)
    model = LeafDiseaseCNN(num_classes=len(_labels))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    _model = model


def predict(image_bytes: bytes) -> dict:
    """Runs real inference through the trained CNN on an uploaded image's bytes."""
    _ensure_loaded()
    img = Image.open(io.BytesIO(image_bytes))
    tensor = load_image_as_tensor(img, train=False).unsqueeze(0)  # 1x3xHxW

    with torch.no_grad():
        logits = _model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    top_idx = int(torch.argmax(probs).item())
    label = _labels[top_idx]
    confidence = float(probs[top_idx].item())
    info = LABEL_INFO.get(label, {"display": label, "advice": "Consult a local agronomist."})

    ranked = sorted(
        [{"label": LABEL_INFO.get(l, {}).get("display", l), "probability": round(float(p), 4)}
         for l, p in zip(_labels, probs.tolist())],
        key=lambda x: -x["probability"],
    )

    is_other = label == "Other___not_tomato_leaf"
    uncertain = confidence < LOW_CONFIDENCE
    # Severity is NOT measured by this model (that would need images labelled by infection stage), so we do not
    # invent one from the confidence score.
    if label == "Tomato___healthy":
        severity = "None"
    elif is_other:
        severity = "-"
    else:
        severity = "Not measured"

    warning = None
    if is_other:
        warning = "The model does not think this is a tomato leaf, so no disease is reported."
    elif uncertain:
        top2 = ranked[1]
        warning = (f"Low confidence: the model is unsure ({ranked[0]['label']} {confidence * 100:.0f}% vs "
                   f"{top2['label']} {top2['probability'] * 100:.0f}%). Retake the photo (one leaf, close-up, good light) "
                   "before acting on this.")

    return {
        "diagnosis": info["display"],
        "confidence": round(confidence * 100, 1),
        "severity": severity,
        "advice": info["advice"],
        "uncertain": uncertain and not is_other,
        "not_tomato": is_other,
        "warning": warning,
        "all_probabilities": ranked,
        "model": "LeafDiseaseCNN (trained from scratch: 4 tomato leaf classes + 'not a tomato leaf')",
    }
