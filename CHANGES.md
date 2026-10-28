# Changes made to the uploaded `agrivision-fullstack.zip`

Base: the project you uploaded (Copilot's first-round version). Only source files were compared
(`venv/` was ignored). The ChatGPT-edited version mentioned in your prompt (the one that shows
"Unable to reach the AgriVision API… port 8001") was not in the zip, so these changes are applied to
the zip's code, not to that edit.

## Fixed
| Problem | Cause | Fix |
|---|---|---|
| "Unable to reach the API" | Frontend hard-coded one backend address; opening via another port / Live Server / file:// broke every call | Backend now serves the frontend (`http://localhost:8000`); `api.js` auto-detects the backend and gives a clear message if it's down |
| Wrong password showed "session expired" | Any 401, including login failure, triggered logout | 401 handling skips `/api/auth/*` |
| Farm Assistant gave canned answers even with a key | Bare `except` swallowed every Gemini error and silently used keywords; model `gemini-2.0-flash` hard-coded; no history; key read once at import | New REST client `gemini_client.py`: tries several models, real error reasons, key read at call time, `.env` support, conversation history, broad agronomist prompt |
| Offline keyword "hi" matched inside words ("which") | Substring matching | Whole-word matching for short keywords |
| Chat/Gemini text inserted as raw HTML | `innerHTML` with unescaped text (XSS risk) | All dynamic text escaped; safe bold/bullet formatting |
| `requirements.txt` missing packages | `bcrypt`, `email-validator` not listed; deprecated `google-generativeai` listed | Updated |

## Added
* **Live market price**: Gemini + Google Search grounding (same key), source links, 30-min cache, labeled simulation fallback.
* **Plant Species ID** module (separate CNN, 12 species) — see accuracy caveat in the app and README.
* `backend/tests/` (pytest, mocked Gemini) and `tests/live_gemini_check.py` (real call).
* `.env.example`, `.gitignore`, `RUN_GUIDE.md`.

## Unchanged on purpose
Disease CNN architecture/weights/preprocessing, auth, history, pest/yield/irrigation/fertilizer logic, satellite (still simulated: a real satellite feed needs a provider account).


---
# Update 2 (after your screenshots)

## "Disease prediction always says Late Blight" - cause and fix
**Cause (measured, not guessed):** on clean lab photos the old model was fine (healthy 100 %, leaf mold 100 %), but it had
learned the *background and brightness* of the training photos. A plain white image scored "Late Blight 91 %", a solid green
image "Late Blight 100 %", a brightened healthy leaf "Late Blight 92 %", and your tomato-plant photo "Late Blight 98 %".
A softmax model must always pick one of its classes, and it had no "none of these".

**Fix (`ml/train_robust.py`):** leaves are cut out with PlantVillage's segmentation masks and pasted on random backgrounds,
with strong lighting/colour/blur/crop augmentation (gentle-to-hard curriculum), label smoothing, and a 5th class
"Not a tomato leaf" (other plants + blank backgrounds). Results on the same test battery:
white / green / noise images, Mango, Lemon, Basil, Jamun leaves -> "Not a tomato leaf"; brightened healthy leaf -> Healthy 98 %;
your plant photo -> "Late Blight 56 %, UNCERTAIN" with a retake warning (it is a whole plant, not a leaf close-up).
**Trade-off:** blight accuracy on lab photos fell from ~88 % to ~65 % because the old score was inflated by the background shortcut.
Healthy and leaf-mold stay 93-100 %. About 1 in 6 blight leaves is predicted Healthy - do not rely on this alone.
Also: "Severity" was derived from the confidence score, not measured, so it now shows "Not measured".

## "Farm Assistant can't answer" - what was wrong / what I added
I cannot see your machine, so the exact cause on your PC is unconfirmed. The screenshot shows the offline reply, which means
the server did not have a working key. The most likely reasons were (a) `.env` not read because `python-dotenv` was not
installed in your old venv, (b) the file saved as `.env.txt` or in the wrong folder, (c) the placeholder key. Changes:
`.env` is now parsed by built-in code (works without python-dotenv, handles Windows BOM/CRLF/quotes); the assistant shows a live
**AI connected / not connected + reason** line; `/api/health` reports which `.env` was loaded; the placeholder key is detected;
offline replies are labelled "Not an AI answer"; "hii/heyy" now match greetings; static files are served with no-cache.
