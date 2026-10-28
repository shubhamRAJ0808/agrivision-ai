import datetime
import hashlib
import json
import logging
import math
import re
import time

from fastapi import APIRouter, Depends, Query

from .. import auth, models
from ..gemini_client import generate, GeminiError

log = logging.getLogger("agrivision.market")

router = APIRouter(prefix="/api/market", tags=["market"])

CACHE_TTL_SECONDS = 30 * 60  # reuse a live lookup for 30 min to save free-tier quota
_cache: dict = {}

# Only used when the live web lookup is unavailable. Illustrative, NOT real market data.
BASE_PRICES = {
    "wheat": 24.5, "rice": 32.0, "tomato": 18.0, "potato": 22.0, "onion": 25.0,
    "maize": 22.0, "soybean": 50.0, "cotton": 70.0, "sugarcane": 3.5, "mustard": 60.0,
}


def _seeded_wave(crop: str, day: int) -> float:
    """Deterministic seasonal wave used to PROJECT forward from a baseline price."""
    seed = int(hashlib.md5(crop.encode()).hexdigest(), 16) % 1000
    return math.sin((day + seed) / 9.0) * 0.03 + math.sin((day + seed) / 30.0) * 0.05


def build_projection(crop_key: str, base: float):
    forecasts = []
    for h in (7, 15, 30):
        drift = _seeded_wave(crop_key, h) + (h * 0.0015)
        forecasts.append({
            "horizon_days": h,
            "price": round(base * (1 + drift), 2),
            "pct_change": round(drift * 100, 1),
        })
    trend = forecasts[1]["pct_change"]  # 15-day outlook drives the advice
    recommendation = "Hold" if trend > 1.5 else ("Sell now" if trend < -1.5 else "Neutral — monitor")
    return forecasts, recommendation


def fetch_live_price(crop: str, region: str) -> dict:
    """Ask Gemini (with Google Search grounding) for today's price. Raises GeminiError or ValueError."""
    today = datetime.date.today().isoformat()
    prompt = (
        f"Today's date is {today}. Use Google Search to find the most recent wholesale / mandi / "
        f"market price of {crop} in {region}. Prefer official or well-known market-price sources. "
        "Convert it to a price PER KILOGRAM in the local currency (for India use INR; "
        "1 quintal = 100 kg). Reply with ONLY one JSON object, no markdown, in exactly this shape: "
        '{"price_per_kg": <number>, "currency": "INR", "market": "<market or site name>", '
        '"price_date": "<date of the price>", "note": "<one short sentence saying what this price is>"}. '
        'If you cannot find a reliable recent price, reply {"price_per_kg": null, "note": "<why>"}.'
    )
    out = generate(
        [{"role": "user", "parts": [{"text": prompt}]}],
        use_search=True,
        temperature=0.1,
        max_output_tokens=1024,
        timeout=45,
    )
    match = re.search(r"\{.*\}", out["text"], re.DOTALL)
    if not match:
        raise ValueError("The price lookup did not return structured data.")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        raise ValueError("The price lookup returned unreadable data.")

    price = data.get("price_per_kg")
    if not isinstance(price, (int, float)) or isinstance(price, bool) or not (0.05 <= price <= 20000):
        raise ValueError(data.get("note") or "No reliable current price was found for this crop/region.")

    currency = str(data.get("currency") or "INR")[:6].upper()
    return {
        "price": round(float(price), 2),
        "unit": f"{currency}/kg",
        "market": str(data.get("market") or "")[:80],
        "price_date": str(data.get("price_date") or "")[:40],
        "note": str(data.get("note") or "")[:200],
        "sources": out["sources"][:5],
        "model": out["model"],
    }


@router.get("/forecast")
def forecast(
    crop: str = Query("wheat", max_length=40),
    region: str = Query("India", max_length=60),
    current_user: models.User = Depends(auth.get_current_user),
):
    crop_key = crop.strip().lower()
    region = region.strip() or "India"
    cache_key = (crop_key, region.lower())

    cached = _cache.get(cache_key)
    live = cached[1] if cached and time.time() - cached[0] < CACHE_TTL_SECONDS else None
    notice = None

    if live is None:
        try:
            live = fetch_live_price(crop_key, region)
            _cache[cache_key] = (time.time(), live)
        except GeminiError as e:
            notice = f"Live price unavailable: {e.message}"
            log.warning("Live market lookup failed: %s", e.message)
        except ValueError as e:
            notice = f"Live price unavailable: {e}"
            log.warning("Live market lookup failed: %s", e)

    if live:
        base, unit = live["price"], live["unit"]
        forecasts, recommendation = build_projection(crop_key, base)
        price_note = " · ".join(x for x in (live["market"], live["price_date"], live["note"]) if x)
        return {
            "crop": crop_key,
            "region": region,
            "data_mode": "live",
            "current_price": base,
            "price_unit": unit,
            "price_note": price_note,
            "forecasts": forecasts,
            "recommendation": recommendation,
            "sources": live["sources"],
            "method": f"Current price: live web search via Gemini ({live['model']}) with Google Search. "
                      "7/15/30-day figures are a statistical projection from that price, "
                      "not verified future prices.",
        }

    base = BASE_PRICES.get(crop_key, 20.0)
    forecasts, recommendation = build_projection(crop_key, base)
    return {
        "crop": crop_key,
        "region": region,
        "data_mode": "simulated",
        "current_price": base,
        "price_unit": "INR/kg (illustrative)",
        "price_note": "",
        "forecasts": forecasts,
        "recommendation": recommendation,
        "sources": [],
        "notice": notice or "Live price lookup is not configured.",
        "method": "SIMULATED: illustrative baseline price with a seasonal projection. "
                  "Not real market data.",
    }
