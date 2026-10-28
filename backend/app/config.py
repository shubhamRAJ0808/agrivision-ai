"""
Central configuration. Loads a .env file (if present) so API keys don't have to be exported in
every new terminal. Works WITHOUT python-dotenv (a tiny built-in parser is used as a fallback), so a
missing package can never silently disable your key.

Search order:  backend/.env  ->  project-root/.env  ->  current working directory/.env
Values are read at call time via get_env().
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(HERE, ".."))
PROJECT_DIR = os.path.abspath(os.path.join(BACKEND_DIR, ".."))

ENV_FILE_LOADED = None   # path of the .env file that was actually read, for /api/health
ENV_FILES_CHECKED = []


def _parse_env_file(path):
    # utf-8-sig strips the BOM that Windows Notepad / PowerShell often add (a classic silent failure)
    with open(path, encoding="utf-8-sig") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.lower().startswith("export "):
                line = line[7:].strip()
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if value[:1] in "\"'" and value[-1:] == value[:1] and len(value) >= 2:
                value = value[1:-1]
            elif " #" in value:
                value = value.split(" #", 1)[0].strip()
            if key and key not in os.environ:   # real environment variables win over the file
                os.environ[key] = value


def _load():
    global ENV_FILE_LOADED
    for p in (os.path.join(BACKEND_DIR, ".env"), os.path.join(PROJECT_DIR, ".env"), os.path.join(os.getcwd(), ".env")):
        p = os.path.abspath(p)
        if p in ENV_FILES_CHECKED:
            continue
        ENV_FILES_CHECKED.append(p)
        if os.path.isfile(p):
            try:
                _parse_env_file(p)
                ENV_FILE_LOADED = ENV_FILE_LOADED or p
            except OSError:
                pass


_load()


def get_env(name: str, default=None):
    value = os.environ.get(name, default)
    if isinstance(value, str):
        value = value.strip().strip('"').strip("'").strip()
    return value or default


def gemini_key():
    key = get_env("GEMINI_API_KEY")
    if key and key.upper().startswith("PASTE_YOUR"):   # the placeholder from .env.example
        return None
    return key


def gemini_models():
    """Return the single configured Gemini model."""
    preferred = get_env("GEMINI_MODEL")
    return [preferred or "gemini-3.8-flash"]
