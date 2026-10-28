# AgriVision AI — Run Guide (Windows PowerShell)

## 1. One-time setup

```powershell
cd agrivision\backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks the activate script, run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## 2. Add your Gemini key (free: https://aistudio.google.com/apikey)

```powershell
copy .env.example .env
notepad .env
```

Replace `PASTE_YOUR_KEY_HERE` with your key and save. The server reads this file on start-up, so
you do **not** need `$env:GEMINI_API_KEY=...` any more (and it can't be forgotten when you open a new terminal).
Never share `.env` or put the key in the frontend.

## 3. Start everything — one command, one port

```powershell
cd agrivision\backend
python -m uvicorn app.main:app --port 8000
```

Now open **http://localhost:8000** in your browser (the backend serves the web pages too).
Sign up, log in, use the modules. Do **not** open the HTML files by double-clicking them.

If port 8000 is busy, use `--port 8001` and open `http://localhost:8001`.

## 4. Check that Gemini is really working

1. Open the **Farm Assistant**. The line at the top says either **"● AI connected (Gemini ...)"** or
   **"● AI not connected: <the real reason>"**. That line makes a real test call, so trust it.
2. Open http://localhost:8000/api/health and look at:
   * `"gemini_configured": true` - the server found a key
   * `"env_file_loaded"` - which `.env` file was read (`null` = no `.env` found!)
   * `"key_is_still_placeholder": true` - you left `PASTE_YOUR_KEY_HERE` in the file
3. Run the live test (real call to Google, never prints your key):
   ```powershell
   python tests/live_gemini_check.py
   ```
4. If an answer is NOT from Gemini the chat shows a yellow **"Not an AI answer"** box with the reason, above the basic offline reply.

**Most common reasons the assistant "can't answer":**

| What you see | Fix |
|---|---|
| `env_file_loaded: null` | The file must be named exactly **`.env`** and live in `agrivision\backend`. Notepad often saves `.env.txt` - in the Save dialog choose "All files". Check with `dir .env` |
| `still contains the placeholder` | Replace `PASTE_YOUR_KEY_HERE` with your real key and restart |
| `GEMINI_API_KEY is not set` | Restart the server after editing `.env` (it is read once at start-up) |
| `rejected the API key` | Key typed wrong/revoked/restricted -> create a new one at https://aistudio.google.com/apikey |
| `quota or rate limit` | Free-tier limit; wait a minute |
| `could not reach Google` | Internet, VPN, proxy or firewall is blocking this computer from `generativelanguage.googleapis.com` |
| `Model ... not available` or temporarily overloaded | Set `GEMINI_MODEL=gemini-3.8-flash` in `.env`, then restart the backend and retry |

If you change code or HTML and the browser still shows the old page, press **Ctrl+F5** (the server now sends no-cache headers, so this should be rare).

## 5. Automated tests

```powershell
pip install pytest httpx
python -m pytest tests -v
```

These use a *mocked* Gemini (they prove the code paths, not your key). Use step 4 for the real key.

## 6. Retraining models (optional — trained checkpoints are included)

```powershell
cd ml
python train_robust.py --tomato C:\path\PlantVillage-Dataset\raw --others C:\path\leaf_species_dataset   # disease model (robust version)
python train.py                                         # original simple trainer (needs ml\data\<class>\*.jpg)
python species_train.py --data C:\path\to\leaf_species_dataset   # 12-species model
```

## 7. Troubleshooting

* **"Cannot reach the AgriVision backend"** — the server is not running, or you are on a different port.
  Start it (step 3) and open the same address it prints.
* **Forgot a dependency** — activate the venv, then `pip install -r requirements.txt` again.
* **Camera doesn't open** — browsers only allow it on `http://localhost` or `https://`.
