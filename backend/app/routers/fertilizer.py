from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from .. import auth, models

router = APIRouter(prefix="/api/fertilizer", tags=["fertilizer"])


class FertilizerRequest(BaseModel):
    soil_n_ppm: float = Field(20, ge=0)  # nitrogen
    soil_p_ppm: float = Field(15, ge=0)  # phosphorus
    soil_k_ppm: float = Field(120, ge=0)  # potassium
    crop_stage: str = "vegetative"
    crop: str = "wheat"


# Target nutrient levels (ppm) by crop stage — simplified agronomic reference ranges.
TARGETS = {
    "seedling": {"N": 30, "P": 20, "K": 100},
    "vegetative": {"N": 45, "P": 25, "K": 130},
    "flowering": {"N": 35, "P": 35, "K": 160},
    "maturity": {"N": 20, "P": 20, "K": 140},
}

# kg of product per ppm-deficit per hectare (simplified conversion factors).
UREA_KG_PER_PPM = 2.2       # Urea (46% N)
DAP_KG_PER_PPM = 2.0        # DAP (18-46-0), phosphorus source
MOP_KG_PER_PPM = 1.7        # Muriate of Potash (60% K)


@router.post("/recommend")
def recommend(payload: FertilizerRequest, current_user: models.User = Depends(auth.get_current_user)):
    target = TARGETS.get(payload.crop_stage, TARGETS["vegetative"])
    n_deficit = max(0, target["N"] - payload.soil_n_ppm)
    p_deficit = max(0, target["P"] - payload.soil_p_ppm)
    k_deficit = max(0, target["K"] - payload.soil_k_ppm)

    urea_kg = round(n_deficit * UREA_KG_PER_PPM, 1)
    dap_kg = round(p_deficit * DAP_KG_PER_PPM, 1)
    mop_kg = round(k_deficit * MOP_KG_PER_PPM, 1)

    recommendations = []
    if urea_kg > 0:
        recommendations.append({"product": "Urea", "amount_kg_per_ha": urea_kg, "nutrient": "Nitrogen"})
    if dap_kg > 0:
        recommendations.append({"product": "DAP", "amount_kg_per_ha": dap_kg, "nutrient": "Phosphorus"})
    if mop_kg > 0:
        recommendations.append({"product": "MOP (Potash)", "amount_kg_per_ha": mop_kg, "nutrient": "Potassium"})

    urgency = "Apply within 3 days" if (n_deficit + p_deficit + k_deficit) > 40 else "Apply within 7 days"
    if not recommendations:
        urgency = "No deficiency detected — soil nutrients are at target levels."

    return {
        "crop": payload.crop,
        "stage": payload.crop_stage,
        "recommendations": recommendations,
        "urgency": urgency,
        "deficits_ppm": {"N": round(n_deficit, 1), "P": round(p_deficit, 1), "K": round(k_deficit, 1)},
        "method": "Deficit-based recommendation against stage-specific NPK target ranges "
                  "(server-side, real computation on your soil-test inputs).",
    }
