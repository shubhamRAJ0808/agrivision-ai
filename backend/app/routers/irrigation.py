from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from .. import auth, models

router = APIRouter(prefix="/api/irrigation", tags=["irrigation"])


class IrrigationRequest(BaseModel):
    soil_moisture_pct: float = Field(35, ge=0, le=100)
    crop_stage: str = "vegetative"  # seedling, vegetative, flowering, maturity
    forecast_rain_mm_next_3d: float = 0
    avg_temp_c: float = 28
    soil_type: str = "loam"  # sandy, loam, clay


STAGE_WATER_NEED = {"seedling": 0.6, "vegetative": 1.0, "flowering": 1.3, "maturity": 0.5}
SOIL_RETENTION = {"sandy": 0.7, "loam": 1.0, "clay": 1.3}


@router.post("/recommend")
def recommend(payload: IrrigationRequest, current_user: models.User = Depends(auth.get_current_user)):
    stage_mult = STAGE_WATER_NEED.get(payload.crop_stage, 1.0)
    soil_mult = SOIL_RETENTION.get(payload.soil_type, 1.0)
    temp_boost = 1 + max(0, payload.avg_temp_c - 25) * 0.02

    base_mm = 25 * stage_mult * temp_boost / soil_mult
    moisture_deficit = max(0, 55 - payload.soil_moisture_pct) / 55
    recommended_mm = round(base_mm * (0.4 + moisture_deficit), 1)
    recommended_mm = max(0, recommended_mm - payload.forecast_rain_mm_next_3d * 0.7)

    if payload.soil_moisture_pct > 60 or payload.forecast_rain_mm_next_3d > 15:
        days_until = 4
        stress = "Low"
    elif payload.soil_moisture_pct > 35:
        days_until = 2
        stress = "Low" if payload.soil_moisture_pct > 45 else "Moderate"
    else:
        days_until = 0
        stress = "High"

    return {
        "recommended_water_mm": round(recommended_mm, 1),
        "next_irrigation_in_days": days_until,
        "water_stress": stress,
        "method": "Rule-based water-balance model using crop stage, soil type/retention, "
                  "temperature and forecast rainfall (server-side, real computation).",
    }
