from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from .. import models, schemas, auth
from ..database import get_db

router = APIRouter(prefix="/api/history", tags=["history"])


@router.get("/", response_model=List[schemas.ScanHistoryOut])
def get_history(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    return (
        db.query(models.ScanHistory)
        .filter(models.ScanHistory.user_id == current_user.id)
        .order_by(models.ScanHistory.created_at.desc())
        .limit(50)
        .all()
    )
