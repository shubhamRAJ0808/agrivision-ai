import logging

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session

from .. import models, auth, config
from ..database import get_db
from ..gemini_client import generate, image_part, GeminiError
from ml import infer

log = logging.getLogger("agrivision.disease")

router = APIRouter(prefix="/api/disease", tags=["disease"])

GEMINI_VISION_PROMPT = (
    "Look at this tomato leaf photo. Does it show signs of disease "
    "(early blight, late blight, leaf mold) or does it look healthy? "
    "If it is not a tomato leaf, say so. Give a one or two sentence assessment."
)

MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB


def get_gemini_second_opinion(image_bytes: bytes, mime_type: str | None):
    """Optional extra opinion. Never raises: the CNN diagnosis must not depend on Gemini."""
    if not config.gemini_key():
        return None
    try:
        out = generate(
            [{"role": "user", "parts": [{"text": GEMINI_VISION_PROMPT}, image_part(image_bytes, mime_type)]}],
            temperature=0.2,
            max_output_tokens=400,
            timeout=30,
        )
        return {"source": f"Gemini ({out['model']}, general-purpose vision model)", "text": out["text"]}
    except GeminiError as e:
        log.warning("Second opinion skipped: %s", e.message)
        return None


@router.post("/diagnose")
async def diagnose(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    image_bytes = await file.read()
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large (max 10 MB).")

    try:
        result = infer.predict(image_bytes)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read this image. Try a JPG or PNG.")

    history = models.ScanHistory(
        user_id=current_user.id,
        module="disease",
        result_label=result["diagnosis"],
        confidence=f"{result['confidence']}%",
        severity=result["severity"],
    )
    db.add(history)
    db.commit()

    second_opinion = get_gemini_second_opinion(image_bytes, file.content_type)
    if second_opinion:
        result["second_opinion"] = second_opinion

    return result
