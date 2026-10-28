# AgriVision AI

Full-stack smart-farming app: FastAPI + PyTorch backend, plain HTML/JS frontend served by the same server.

**Start here: [RUN_GUIDE.md](RUN_GUIDE.md)** (setup, Gemini key, run, tests, troubleshooting).
What changed in this version: [CHANGES.md](CHANGES.md).

## What is real vs. simulated

| Module | What it does |
|---|---|
| Disease Diagnosis | Trained CNN: Early Blight, Late Blight, Leaf Mold, Healthy **+ "Not a tomato leaf"** (rejects blank images and other plants). Retrained with random backgrounds/lighting so it no longer says "Late Blight" for everything. Held-out PlantVillage accuracy 86% (healthy 100%, leaf mold 93%, early/late blight ~64-69%); on the same leaves with new backgrounds/lighting 71%. **These are lab photos - no real field photos were available, so field accuracy is unknown and likely lower.** Results under 70% confidence are flagged as uncertain. "Severity" is not measured. Optional Gemini "second opinion" is shown separately. |
| Plant Species ID | Separate trained CNN, 12 species. **Caveat:** the dataset has only ~1–2 physical leaves per species, so accuracy on its own valid/test split is inflated. A stricter new-leaf test during training scored far lower — see `backend/ml/checkpoints/species_metrics.json` and the in-app note. Treat as a demo until trained on more varied photos. |
| Farm Assistant | Gemini (LLM) with conversation history. If Gemini is unavailable the chat says so and shows a basic offline answer — never passes it off as AI. |
| Market Price | Live price via Gemini + Google Search (needs the Gemini key), with source links. 7/15/30-day numbers are a projection from that price, not verified future prices. Falls back to a clearly labeled simulation. |
| Pest Detection | Image texture heuristic, **not** a trained model. |
| Yield / Irrigation / Fertilizer | Transparent formulas on your inputs, not machine learning. |
| Satellite | **Simulated** grid — a real feed needs a provider account (e.g. Sentinel Hub). |

## Layout
```
backend/  app/ (API, auth, gemini_client.py, routers)  ml/ (models, training, checkpoints)  tests/
frontend/ login.html signup.html app.html assets/
```
