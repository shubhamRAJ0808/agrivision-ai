import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import config  # noqa: F401  (loads .env before routers are imported)
from . import models
from .database import engine
from .routers import auth_router, disease, pest, yield_router, irrigation, fertilizer, market, satellite, assistant, history, species

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="AgriVision AI API", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your frontend origin in production
    allow_credentials=False,  # we use bearer tokens, not cookies
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router.router)
app.include_router(disease.router)
app.include_router(pest.router)
app.include_router(yield_router.router)
app.include_router(irrigation.router)
app.include_router(fertilizer.router)
app.include_router(market.router)
app.include_router(satellite.router)
app.include_router(assistant.router)
app.include_router(history.router)
app.include_router(species.router)


@app.get("/api/health")
def health():
    placeholder = (config.get_env("GEMINI_API_KEY") or "").upper().startswith("PASTE_YOUR")
    return {
        "status": "ok",
        "gemini_configured": bool(config.gemini_key()),
        "key_is_still_placeholder": placeholder,
        "env_file_loaded": config.ENV_FILE_LOADED,
        "env_files_checked": config.ENV_FILES_CHECKED,
    }


@app.middleware("http")
async def no_cache_for_frontend(request, call_next):
    """Stop browsers serving a stale app.html / api.js after you update the project."""
    response = await call_next(request)
    path = request.url.path
    if not path.startswith("/api/") and path.endswith((".html", ".js", ".css", "/")):
        response.headers["Cache-Control"] = "no-store"
    return response


# ---- Serve the frontend from the same server (one command, one port, no CORS issues) ----
FRONTEND_DIR = os.path.join(config.PROJECT_DIR, "frontend")

if os.path.isdir(FRONTEND_DIR):
    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse(url="/login.html")

    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
