"""Inference for the separate plant-species classifier (see species_train.py)."""
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
MODEL_PATH = os.path.join(CKPT_DIR, "species_model.pt")
LABELS_PATH = os.path.join(CKPT_DIR, "species_labels.json")
METRICS_PATH = os.path.join(CKPT_DIR, "species_metrics.json")
SIZE = 96
LOW_CONFIDENCE = 0.60

SPECIES_INFO = {
    "Alstonia Scholaris": ("Devil's tree (Saptaparni)", "Alstonia scholaris"),
    "Arjun": ("Arjuna", "Terminalia arjuna"),
    "Bael": ("Bael (Stone apple)", "Aegle marmelos"),
    "Basil": ("Basil (Tulsi)", "Ocimum tenuiflorum"),
    "Chinar": ("Chinar (Oriental plane)", "Platanus orientalis"),
    "Gauva": ("Guava", "Psidium guajava"),
    "Jamun": ("Jamun (Java plum)", "Syzygium cumini"),
    "Jatropha": ("Jatropha", "Jatropha curcas"),
    "Lemon": ("Lemon", "Citrus limon"),
    "Mango": ("Mango", "Mangifera indica"),
    "Pomegranate": ("Pomegranate", "Punica granatum"),
    "Pongamia Pinnata": ("Pongamia (Karanj)", "Pongamia pinnata"),
}

_model = None
_labels = None


def _ensure_loaded():
    global _model, _labels
    if _model is not None:
        return
    if not (os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH)):
        raise RuntimeError(
            "Species model not found. Train it with: python ml/species_train.py --data <dataset folder> "
            "(from the backend folder)."
        )
    with open(LABELS_PATH) as f:
        _labels = json.load(f)
    model = LeafDiseaseCNN(num_classes=len(_labels))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    _model = model


def _names(label):
    common, sci = SPECIES_INFO.get(label, (label, ""))
    return common, sci


def predict(image_bytes: bytes) -> dict:
    _ensure_loaded()
    img = Image.open(io.BytesIO(image_bytes))
    tensor = load_image_as_tensor(img, train=False, size=SIZE).unsqueeze(0)
    with torch.no_grad():
        probs = F.softmax(_model(tensor), dim=1)[0]

    order = torch.argsort(probs, descending=True).tolist()
    top = []
    for i in order[:3]:
        common, sci = _names(_labels[i])
        top.append({"species": common, "scientific_name": sci, "probability": round(float(probs[i]) * 100, 1)})

    best = top[0]
    confident = best["probability"] / 100 >= LOW_CONFIDENCE
    result = {
        "species": best["species"],
        "scientific_name": best["scientific_name"],
        "confidence": best["probability"],
        "confident": confident,
        "top3": top,
        "model": "Species CNN (trained from scratch on a 12-species leaf dataset)",
        "warning": None if confident else
        "Low confidence - this may be a plant outside the 12 species this model knows, "
        "or a photo that looks different from the training images.",
        "supported_species": [_names(l)[0] for l in _labels],
    }
    try:
        with open(METRICS_PATH) as f:
            m = json.load(f)
        result["accuracy_note"] = (
            f"Honest accuracy on a leaf it never saw: {round(m.get('new_leaf_accuracy', 0) * 100, 1)}% "
            "(dataset has only 1-2 leaves per species) - treat as a demo"
            if "new_leaf_accuracy" in m else None
        )
    except (OSError, ValueError):
        result["accuracy_note"] = None
    return result
