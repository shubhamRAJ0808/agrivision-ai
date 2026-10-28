from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from .. import auth, models

router = APIRouter(prefix="/api/yield", tags=["yield"])


class YieldRequest(BaseModel):
    crop: str = "wheat"
    soil_moisture_pct: float = Field(40, ge=0, le=100)
    avg_temp_c: float = 25
    rainfall_mm_week: float = 20
    fertilizer_applied: bool = True
    ndvi: float = Field(0.6, ge=0, le=1)  # vegetation index, 0-1


BASE_YIELD = {"wheat": 4.5, "rice": 5.2, "tomato": 35.0, "maize": 5.8}  # tonnes/ha baseline


@router.post("/predict")
def predict_yield(payload: YieldRequest, current_user: models.User = Depends(auth.get_current_user)):
    base = BASE_YIELD.get(payload.crop.lower(), 4.5)

    # Simple, transparent multiplicative model over agronomic factors (documented, not a black box).
    moisture_factor = 1 - abs(payload.soil_moisture_pct - 55) / 100  # optimal ~55%
    temp_factor = 1 - abs(payload.avg_temp_c - 24) / 50
    rain_factor = min(1.15, 0.7 + payload.rainfall_mm_week / 60)
    fert_factor = 1.12 if payload.fertilizer_applied else 0.92
    ndvi_factor = 0.6 + payload.ndvi * 0.7

    multiplier = max(0.4, moisture_factor) * max(0.4, temp_factor) * rain_factor * fert_factor * ndvi_factor
    expected_yield = round(base * multiplier, 2)
    confidence = round(min(95, 65 + payload.ndvi * 30), 1)

    return {
        "crop": payload.crop,
        "expected_yield_t_per_ha": expected_yield,
        "confidence_pct": confidence,
        "factors": {
            "moisture_factor": round(moisture_factor, 2),
            "temp_factor": round(temp_factor, 2),
            "rain_factor": round(rain_factor, 2),
            "fertilizer_factor": fert_factor,
            "ndvi_factor": round(ndvi_factor, 2),
        },
        "method": "Transparent agronomic multiplicative model over soil moisture, temperature, "
                  "rainfall, fertilizer use and NDVI (server-side, real computation on your inputs).",
    }
