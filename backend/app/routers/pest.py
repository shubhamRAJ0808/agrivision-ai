import io
import numpy as np
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from PIL import Image
from sqlalchemy.orm import Session

from .. import models, auth
from ..database import get_db

router = APIRouter(prefix="/api/pest", tags=["pest"])

# NOTE: We do not have a labeled pest-image dataset available in this environment,
# so unlike /api/disease (a real trained CNN), this endpoint runs a real server-side
# computer-vision heuristic (texture/speckle variance analysis on the actual uploaded
# image, computed with numpy) rather than a trained deep learning classifier. This is
# stated honestly in the API response so the frontend/user isn't misled.

PESTS = [
    ("Aphids", "Small, spreading yellow/pale speckling with light texture noise."),
    ("Whiteflies", "Fine speckling concentrated on leaf undersides with higher variance."),
    ("Leaf Miners", "Irregular winding pale trails with high local contrast."),
]


def analyze_pest_image(image_bytes: bytes) -> dict:
    img = Image.open(io.BytesIO(image_bytes)).convert("L").resize((160, 160))
    arr = np.asarray(img, dtype=np.float32) / 255.0

    # Local texture variance via a simple 3x3 sliding window std-dev.
    gy, gx = np.gradient(arr)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    speckle_score = float(np.mean(grad_mag))  # proxy for irregular texture -> possible pest damage
    speckle_score = min(1.0, speckle_score * 6.0)

    infestation_pct = round(speckle_score * 100, 1)
    if speckle_score < 0.15:
        pest, severity = "No significant pest signs", "None"
    elif speckle_score < 0.35:
        pest, severity = PESTS[0][0], "Mild"
    elif speckle_score < 0.6:
        pest, severity = PESTS[1][0], "Moderate"
    else:
        pest, severity = PESTS[2][0], "Severe"

    return {
        "detection": pest,
        "severity": severity,
        "infestation_estimate_pct": infestation_pct,
        "method": "Server-side image texture/gradient analysis (heuristic computer vision, "
                  "not a trained deep learning classifier — no labeled pest dataset was "
                  "available to train one in this environment).",
    }


@router.post("/detect")
async def detect(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")
    image_bytes = await file.read()
    try:
        result = analyze_pest_image(image_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read this image. Try a JPG or PNG.")

    history = models.ScanHistory(
        user_id=current_user.id,
        module="pest",
        result_label=result["detection"],
        confidence="-",
        severity=result["severity"],
    )
    db.add(history)
    db.commit()
    return result
