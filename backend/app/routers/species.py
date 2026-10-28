from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session

from .. import models, auth
from ..database import get_db
from ml import species_infer

router = APIRouter(prefix="/api/species", tags=["species"])

MAX_IMAGE_BYTES = 10 * 1024 * 1024


@router.post("/identify")
async def identify(
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
        result = species_infer.predict(image_bytes)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read this image. Try a JPG or PNG.")

    db.add(models.ScanHistory(
        user_id=current_user.id,
        module="species",
        result_label=result["species"],
        confidence=f"{result['confidence']}%",
        severity="-",
    ))
    db.commit()
    return result
