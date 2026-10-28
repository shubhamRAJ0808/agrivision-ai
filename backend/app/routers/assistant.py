import re
import time
from typing import List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from .. import auth, models
from ..gemini_client import generate, GeminiError

router = APIRouter(prefix="/api/assistant", tags=["assistant"])


class Turn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    text: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: Optional[List[Turn]] = None  # previous turns, oldest first


SYSTEM_INSTRUCTION = (
    "You are AgriVision's Farm Assistant: an experienced, friendly agronomist who talks with "
    "farmers and gardeners. Answer ANY question related to crops, fields, orchards, kitchen "
    "gardens, soil, seeds, planting and harvest timing, irrigation and water management, "
    "fertilizers and nutrients, composting, plant diseases, pests, weeds, weather impact on "
    "crops, farm equipment, post-harvest storage, crop economics and selling. "
    "Style: conversational and practical, like a knowledgeable neighbour. Lead with the most "
    "likely explanation or the direct answer, then give clear next steps (short bullet lists "
    "are fine). Use metric units. When a diagnosis depends on missing details (crop, growth "
    "stage, symptoms, weather, recent sprays), give your best-guess possibilities AND ask 1-2 "
    "specific follow-up questions. Mention pesticide/fertilizer safety (label rates, protective "
    "gear, pre-harvest interval) when you recommend chemicals, and prefer low-cost organic or "
    "cultural options first where they work. Be honest about uncertainty. "
    "If a question is clearly unrelated to farming or plants, politely say you focus on farming "
    "and offer to help with a crop question. "
    "The app also has: Disease Diagnosis (photo of a tomato leaf, trained CNN), Pest Detection, "
    "Yield, Irrigation, Fertilizer, Market Price and Satellite modules. Point to them when useful, "
    "but still answer the question yourself. Never claim to have seen a photo - you only see text."
)

# ---------------------------------------------------------------------------
# Offline fallback: ONLY used when Gemini is unavailable. It is labeled as such
# in the response so the UI never pretends it came from an AI model.
# ---------------------------------------------------------------------------
KB = [
    (["yellow", "yellowing", "chlorosis"],
     "Yellowing leaves usually point to nitrogen deficiency, overwatering, or early-stage leaf rust. "
     "Check soil moisture first; if waterlogged, ease off irrigation for 2-3 days. If soil looks fine, "
     "try a nitrogen top-up via the Fertilizer module."),
    (["curl", "curling"],
     "Leaf curling is often heat/water stress or aphid feeding on leaf undersides. Check for insects; "
     "if none, increase irrigation slightly and check heat/wind exposure."),
    (["water", "irrigat", "watering"],
     "Use the Irrigation module for a recommendation based on soil moisture, crop stage and forecast rainfall."),
    (["harvest"],
     "Harvest timing depends on the crop: grain moisture below ~20% for cereals, or full color with slight "
     "give for produce. Tell me the crop for a more specific answer."),
    (["fertiliz", "fertilis", "npk", "urea"],
     "Use the Fertilizer module with your latest soil-test N/P/K numbers for a product + quantity recommendation."),
    (["price", "sell", "market", "mandi"],
     "Check the Market Price module for a forecast on your crop before deciding to hold or sell."),
    (["pest", "insect", "bug", "aphid", "worm", "caterpillar"],
     "Upload a field photo to the Pest Detection module. For mild pressure a neem-oil spray is usually "
     "enough; moderate-severe cases need a targeted insecticide."),
    (["disease", "rust", "blight", "fungus", "mold", "mould", "spots"],
     "Upload a leaf photo to the Disease Diagnosis module - it runs a trained CNN and returns the "
     "disease, confidence and severity directly."),
    (["rain", "weather", "forecast"],
     "Rain forecasts feed into the Irrigation module - enter the expected rainfall there."),
    (["hi", "hello", "hey", "namaste"],
     "Hi! Ask me about leaf symptoms, irrigation, fertilizer, pests, harvest timing, or market prices."),
]

FALLBACK = ("I don't have an offline answer for that. The full AI assistant (Gemini) is not available "
            "right now, so I can only answer a few basic topics: leaf color, watering, fertilizer, "
            "pests, harvest timing and prices.")


def rule_based_reply(message: str) -> str:
    words = set(re.findall(r"[a-z]+", message.lower()))
    text = message.lower()
    if re.match(r"^\s*(h+i+|he+y+|he+llo+|namaste|namaskar|hola)\b", text):
        return KB[-1][1]
    for keywords, reply in KB:
        for k in keywords:
            # whole-word match for short keywords (so "hi" doesn't match "which"), prefix match otherwise
            if (k in words) if len(k) <= 3 else (k in text):
                return reply
    return FALLBACK


def build_contents(payload: ChatRequest):
    contents = []
    for t in (payload.history or [])[-12:]:
        contents.append({"role": "user" if t.role == "user" else "model", "parts": [{"text": t.text}]})
    contents.append({"role": "user", "parts": [{"text": payload.message}]})
    # Gemini requires the first turn to be from the user
    while contents and contents[0]["role"] != "user":
        contents.pop(0)
    return contents


@router.post("/chat")
def chat(payload: ChatRequest, current_user: models.User = Depends(auth.get_current_user)):
    system = SYSTEM_INSTRUCTION
    if current_user.farm_name:
        system += f" The farmer's farm is called '{current_user.farm_name}'."

    try:
        result = generate(build_contents(payload), system_instruction=system)
        return {
            "reply": result["text"],
            "source": "gemini",
            "model": result["model"],
            "method": f"Google Gemini ({result['model']})",
        }
    except GeminiError as e:
        return {
            "reply": rule_based_reply(payload.message),
            "source": "rule-based-fallback",
            "error_code": e.code,
            "notice": f"AI assistant unavailable: {e.message}",
            "method": "Offline keyword rules (Gemini was NOT used for this answer).",
        }


_status_cache = {"t": 0.0, "value": None}


@router.get("/status")
def status(current_user: models.User = Depends(auth.get_current_user)):
    """Makes one tiny REAL call to Gemini so the UI can show whether the AI assistant is working (cached 60 s)."""
    now = time.time()
    if _status_cache["value"] and now - _status_cache["t"] < 60:
        return _status_cache["value"]
    try:
        out = generate([{"role": "user", "parts": [{"text": "Reply with the single word: ok"}]}],
                       temperature=0, max_output_tokens=16, timeout=20)
        value = {"ok": True, "model": out["model"]}
    except GeminiError as e:
        value = {"ok": False, "error_code": e.code, "message": e.message}
    _status_cache.update(t=now, value=value)
    return value
