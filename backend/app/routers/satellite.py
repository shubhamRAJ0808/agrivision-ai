import hashlib
import os
from fastapi import APIRouter, Depends
from .. import auth, models

router = APIRouter(prefix="/api/satellite", tags=["satellite"])

STATES = ["healthy", "stressed", "flooded"]
SATELLITE_API_KEY = os.environ.get("SATELLITE_API_KEY")


def get_simulated_scan(user_id: int):
    # Future provider integration: use SATELLITE_API_KEY with Sentinel Hub or
    # Planet to fetch imagery and derive zone states from real NDVI data.
    zones = []
    for i in range(24):
        h = int(hashlib.md5(f"{user_id}-{i}".encode()).hexdigest(), 16)
        state = STATES[h % 10 >= 8 and 2 or (h % 10 >= 5 and 1 or 0)]
        zones.append({"zone": i + 1, "state": state})

    counts = {s: sum(1 for z in zones if z["state"] == s) for s in STATES}
    total = len(zones)
    return {
        "zones": zones,
        "summary": {
            "healthy_pct": round(counts["healthy"] / total * 100, 1),
            "stressed_pct": round(counts["stressed"] / total * 100, 1),
            "flooded_pct": round(counts["flooded"] / total * 100, 1),
        },
        "method": "Simulated NDVI-style zone grid, deterministic per account (no live satellite "
                  "provider connected in this environment).",
    }


@router.get("/scan")
def scan(current_user: models.User = Depends(auth.get_current_user)):
    """
    Deterministic per-user 24-zone farm grid (seeded from user id so it's stable across
    requests rather than random). No live satellite feed is connected in this environment —
    in production this would call a provider (Sentinel Hub / Planet) and run NDVI scoring.
    """
    # When a provider is wired in, select it here when SATELLITE_API_KEY is set.
    return get_simulated_scan(current_user.id)
