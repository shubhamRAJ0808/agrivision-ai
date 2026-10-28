"""
Minimal Gemini client over the public REST API (no SDK needed, so no version/deprecation issues).

    from .gemini_client import generate, GeminiError

Every failure raises GeminiError with a machine-readable .code and a human-readable message,
so callers can tell the user the REAL reason (bad key, quota, network...) instead of silently
pretending nothing happened. The API key is only ever sent to Google in a request header and
is never logged or returned to the browser.
"""
import base64
import logging
from typing import Optional

import requests

from . import config

log = logging.getLogger("agrivision.gemini")

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiError(Exception):
    """code: no_key | auth | quota | overloaded | model | network | timeout | blocked | empty | api"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _error_from_response(resp: requests.Response, model: str) -> GeminiError:
    try:
        body = resp.json()
        err = body.get("error", {}) if isinstance(body, dict) else {}
    except ValueError:
        err = {}
    status = err.get("status", "")
    detail = (err.get("message") or resp.text or "")[:300]

    if resp.status_code in (400, 401, 403) and (
        "API key" in detail or "API_KEY" in detail or status in ("UNAUTHENTICATED", "PERMISSION_DENIED")
    ):
        return GeminiError(
            "auth",
            "Gemini rejected the API key (invalid, expired, or not allowed). "
            "Create a new key at https://aistudio.google.com/apikey and restart the server.",
        )
    if resp.status_code == 429 or status == "RESOURCE_EXHAUSTED":
        return GeminiError(
            "quota",
            "Gemini free-tier quota or rate limit reached. Wait a minute and try again.",
        )
    if resp.status_code == 503 or status in ("UNAVAILABLE", "OVERLOADED"):
        return GeminiError(
            "overloaded",
            "Gemini is temporarily busy. The server will try another available model.",
        )
    if resp.status_code == 404 or (resp.status_code == 400 and "model" in detail.lower()):
        return GeminiError("model", f"Model '{model}' is not available for this key ({detail}).")
    return GeminiError("api", f"Gemini API error {resp.status_code}: {detail}")


def _extract(data: dict) -> dict:
    block = (data.get("promptFeedback") or {}).get("blockReason")
    if block:
        raise GeminiError("blocked", f"Gemini blocked this request ({block}).")

    candidates = data.get("candidates") or []
    if not candidates:
        raise GeminiError("empty", "Gemini returned no answer.")

    cand = candidates[0]
    parts = (cand.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
    if not text:
        raise GeminiError("empty", f"Gemini returned an empty answer (finish reason: {cand.get('finishReason')}).")

    sources, seen = [], set()
    chunks = (cand.get("groundingMetadata") or {}).get("groundingChunks") or []
    for ch in chunks:
        web = ch.get("web") or {}
        uri = web.get("uri")
        if uri and uri not in seen:
            seen.add(uri)
            sources.append({"title": web.get("title") or uri, "url": uri})

    return {"text": text, "sources": sources}


def generate(
    contents: list,
    system_instruction: Optional[str] = None,
    use_search: bool = False,
    temperature: float = 0.6,
    max_output_tokens: int = 2048,
    timeout: int = 40,
) -> dict:
    """
    contents: Gemini-format list, e.g. [{"role":"user","parts":[{"text":"hi"}]}]
    Returns {"text": str, "sources": [{"title","url"}], "model": str}
    """
    key = config.gemini_key()
    if not key:
        if (config.get_env("GEMINI_API_KEY") or "").upper().startswith("PASTE_YOUR"):
            raise GeminiError("no_key", "backend/.env still contains the placeholder PASTE_YOUR_KEY_HERE. "
                                        "Replace it with your real key, save, and restart the server.")
        where = (f"Found .env at {config.ENV_FILE_LOADED} but it has no GEMINI_API_KEY line."
                 if config.ENV_FILE_LOADED else "No .env file was found (looked in backend/ and the project folder).")
        raise GeminiError(
            "no_key",
            "GEMINI_API_KEY is not set on the server. " + where +
            " Create backend/.env with the line GEMINI_API_KEY=your_key, then restart the server.",
        )

    payload = {
        "contents": contents,
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_output_tokens},
    }
    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
    if use_search:
        payload["tools"] = [{"google_search": {}}]

    headers = {"x-goog-api-key": key, "Content-Type": "application/json"}

    for model in config.gemini_models():
        try:
            resp = requests.post(
                f"{API_ROOT}/{model}:generateContent", json=payload, headers=headers, timeout=timeout
            )
        except requests.Timeout:
            raise GeminiError("timeout", "Gemini took too long to respond. Please try again.")
        except requests.ConnectionError:
            raise GeminiError(
                "network",
                "The server could not reach Google's Gemini API. Check this computer's internet "
                "connection, proxy or firewall.",
            )
        except requests.RequestException as e:  # pragma: no cover
            raise GeminiError("network", f"Network problem talking to Gemini: {type(e).__name__}")

        if resp.status_code == 200:
            try:
                out = _extract(resp.json())
            except ValueError:
                raise GeminiError("api", "Gemini returned an unreadable response.")
            out["model"] = model
            return out

        err = _error_from_response(resp, model)
        log.warning("Gemini call failed on %s: %s (%s)", model, err.code, err.message)
        raise err

    raise GeminiError("model", "No Gemini model is configured.")


def image_part(image_bytes: bytes, mime_type: str) -> dict:
    return {"inline_data": {"mime_type": mime_type or "image/jpeg", "data": base64.b64encode(image_bytes).decode()}}
