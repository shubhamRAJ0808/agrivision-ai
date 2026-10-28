"""
Run this on YOUR computer, after putting your key in backend/.env, to prove Gemini really works:

    python tests/live_gemini_check.py

It makes real calls to Google and prints exactly what happened (key is never printed).
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app import config  # loads .env
from app.gemini_client import generate, GeminiError

print("Key found:", "YES" if config.gemini_key() else "NO  <-- add GEMINI_API_KEY to backend/.env")
print("Models to try:", config.gemini_models())
try:
    out = generate([{"role": "user", "parts": [{"text": "In two sentences: why do tomato leaves curl?"}]}],
                   system_instruction="You are a concise agronomist.")
    print("\n[1] Chat OK via", out["model"], "\n", out["text"])
except GeminiError as e:
    print("\n[1] Chat FAILED:", e.code, "-", e.message); sys.exit(1)

try:
    out = generate([{"role": "user", "parts": [{"text": "Current wheat price per kg in India today. One sentence."}]}], use_search=True)
    print("\n[2] Google Search grounding OK via", out["model"], "\n", out["text"], "\nSources:", [s["url"] for s in out["sources"]][:3])
except GeminiError as e:
    print("\n[2] Search grounding FAILED:", e.code, "-", e.message)
