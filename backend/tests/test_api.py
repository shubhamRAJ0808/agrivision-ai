"""
Automated checks for the AgriVision backend.  Run from the backend folder:

    python -m pytest tests -v

Gemini is MOCKED here (no real key/network needed), so these tests prove the request building,
response parsing, model fallback and error handling - they do NOT prove your live key works.
To test live Gemini, run:  python tests/live_gemini_check.py
"""
import glob
import json
import os
import sys
import tempfile
from unittest import mock

import pytest

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BACKEND)
os.environ["AGRIVISION_DB"] = os.path.join(tempfile.mkdtemp(), "test.db")
os.environ.pop("GEMINI_API_KEY", None)

from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)


class FakeResp:
    def __init__(self, status=200, body=None):
        self.status_code = status
        self._body = body if body is not None else {}
        self.text = json.dumps(self._body)

    def json(self):
        return self._body


def gemini_ok(text, sources=None):
    cand = {"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}
    if sources:
        cand["groundingMetadata"] = {"groundingChunks": [{"web": {"uri": u, "title": t}} for t, u in sources]}
    return FakeResp(200, {"candidates": [cand]})


def gemini_err(status, message, st=""):
    return FakeResp(status, {"error": {"code": status, "message": message, "status": st}})


@pytest.fixture(scope="module")
def auth_headers():
    r = client.post("/api/auth/signup", json={"name": "T", "email": "t@example.com", "password": "secret123", "farm_name": "Green Acres"})
    assert r.status_code == 200, r.text
    return {"Authorization": "Bearer " + r.json()["access_token"]}


@pytest.fixture
def with_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "FAKE-KEY-123")


def sample_image(kind="disease"):
    for pattern in (
        os.path.join(BACKEND, "ml", "data", "Tomato___Late_blight", "*"),
        "/home/claude/user_src/agrivision-fullstack/agrivision/backend/ml/data/Tomato___Late_blight/*",
    ) if kind == "disease" else (
        "/home/claude/species_raw/test/test/Mango/*",
    ):
        files = sorted(glob.glob(pattern))
        if files:
            return files[0]
    return None


# ---------------- basics ----------------
def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_frontend_is_served_from_backend():
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"].endswith("login.html")
    assert client.get("/login.html").status_code == 200
    assert client.get("/app.html").status_code == 200
    assert "resolveApiBase" in client.get("/assets/api.js").text


def test_wrong_password_is_401_with_message(auth_headers):
    r = client.post("/api/auth/login", json={"email": "t@example.com", "password": "nope"})
    assert r.status_code == 401 and "Incorrect" in r.json()["detail"]


def test_requires_auth():
    assert client.post("/api/assistant/chat", json={"message": "hi"}).status_code == 401


# ---------------- assistant ----------------
def test_assistant_no_key_is_honest_fallback(auth_headers):
    r = client.post("/api/assistant/chat", headers=auth_headers, json={"message": "my leaves turn yellow"})
    j = r.json()
    assert r.status_code == 200
    assert j["source"] == "rule-based-fallback" and j["error_code"] == "no_key"
    assert "GEMINI_API_KEY" in j["notice"] and "NOT used" in j["method"]


def test_assistant_uses_gemini_with_history_and_hides_key(auth_headers, with_key):
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured.update(url=url, payload=json, headers=headers)
        return gemini_ok("**Likely nitrogen deficiency.**\n* Check soil\n* Apply urea")

    hist = [{"role": "user", "text": "I grow wheat"}, {"role": "assistant", "text": "Great, what stage?"}]
    with mock.patch("app.gemini_client.requests.post", side_effect=fake_post):
        r = client.post("/api/assistant/chat", headers=auth_headers,
                        json={"message": "leaves are pale", "history": hist})
    j = r.json()
    assert j["source"] == "gemini" and "nitrogen" in j["reply"]
    # key goes only in the Google header, never in URL/body/response
    assert captured["headers"]["x-goog-api-key"] == "FAKE-KEY-123"
    assert "FAKE-KEY" not in captured["url"] and "FAKE-KEY" not in json.dumps(captured["payload"])
    assert "FAKE-KEY" not in r.text
    # conversation history + system prompt + farm name reach Gemini
    roles = [c["role"] for c in captured["payload"]["contents"]]
    assert roles == ["user", "model", "user"]
    assert "Green Acres" in captured["payload"]["systemInstruction"]["parts"][0]["text"]


@pytest.mark.parametrize("status,msg,st,code", [
    (400, "API key not valid. Please pass a valid API key.", "INVALID_ARGUMENT", "auth"),
    (403, "Permission denied", "PERMISSION_DENIED", "auth"),
    (429, "Quota exceeded", "RESOURCE_EXHAUSTED", "quota"),
    (500, "Internal error", "INTERNAL", "api"),
    (503, "Model overloaded", "UNAVAILABLE", "overloaded"),
])
def test_assistant_reports_real_gemini_errors(auth_headers, with_key, status, msg, st, code):
    with mock.patch("app.gemini_client.requests.post", return_value=gemini_err(status, msg, st)):
        j = client.post("/api/assistant/chat", headers=auth_headers, json={"message": "hello there"}).json()
    assert j["source"] == "rule-based-fallback" and j["error_code"] == code and j["notice"]


def test_assistant_network_and_timeout(auth_headers, with_key):
    import requests as rq
    with mock.patch("app.gemini_client.requests.post", side_effect=rq.ConnectionError()):
        assert client.post("/api/assistant/chat", headers=auth_headers, json={"message": "x y"}).json()["error_code"] == "network"
    with mock.patch("app.gemini_client.requests.post", side_effect=rq.Timeout()):
        assert client.post("/api/assistant/chat", headers=auth_headers, json={"message": "x y"}).json()["error_code"] == "timeout"


def test_configured_model_is_used_when_model_is_overloaded(auth_headers, with_key):
    calls = []

    def fake_post(url, **kw):
        calls.append(url)
        return gemini_err(503, "Model overloaded", "UNAVAILABLE")

    with mock.patch("app.gemini_client.requests.post", side_effect=fake_post):
        j = client.post("/api/assistant/chat", headers=auth_headers, json={"message": "hello"}).json()
    assert j["source"] == "rule-based-fallback" and j["error_code"] == "overloaded"
    assert len(calls) == 1 and calls[0].endswith("/gemini-3.8-flash:generateContent")


def test_offline_rules_do_not_match_inside_words(auth_headers):
    j = client.post("/api/assistant/chat", headers=auth_headers, json={"message": "which thing"}).json()
    assert "basic topics" in j["reply"]  # 'hi' must not match 'which'/'thing'


def test_assistant_validates_input(auth_headers):
    assert client.post("/api/assistant/chat", headers=auth_headers, json={"message": ""}).status_code == 422


# ---------------- market ----------------
def test_market_live_via_grounded_gemini(auth_headers, with_key):
    from app.routers import market
    market._cache.clear()
    body = '{"price_per_kg": 27.5, "currency": "INR", "market": "Azadpur Mandi", "price_date": "2026-10-27", "note": "Modal price"}'
    with mock.patch("app.gemini_client.requests.post", return_value=gemini_ok(body, [("Agmarknet", "https://agmarknet.gov.in/x")])) as m:
        r = client.get("/api/market/forecast?crop=wheat&region=Delhi", headers=auth_headers)
        j = r.json()
        assert j["data_mode"] == "live" and j["current_price"] == 27.5 and j["price_unit"] == "INR/kg"
        assert j["sources"][0]["url"].startswith("https://agmarknet")
        assert m.call_args.kwargs["json"]["tools"] == [{"google_search": {}}]
        # cached: second call must not hit Gemini again
        client.get("/api/market/forecast?crop=wheat&region=Delhi", headers=auth_headers)
        assert m.call_count == 1


def test_market_falls_back_to_labeled_simulation(auth_headers, with_key):
    from app.routers import market
    market._cache.clear()
    with mock.patch("app.gemini_client.requests.post", return_value=gemini_ok('{"price_per_kg": null, "note": "no data"}')):
        j = client.get("/api/market/forecast?crop=rice", headers=auth_headers).json()
    assert j["data_mode"] == "simulated" and "SIMULATED" in j["method"] and j["notice"]


def test_market_without_key_is_simulated_and_says_why(auth_headers):
    from app.routers import market
    market._cache.clear()
    j = client.get("/api/market/forecast?crop=maize", headers=auth_headers).json()
    assert j["data_mode"] == "simulated" and "GEMINI_API_KEY" in j["notice"]


def test_market_rejects_absurd_prices(auth_headers, with_key):
    from app.routers import market
    market._cache.clear()
    with mock.patch("app.gemini_client.requests.post", return_value=gemini_ok('{"price_per_kg": 9999999, "currency":"INR"}')):
        assert client.get("/api/market/forecast?crop=onion", headers=auth_headers).json()["data_mode"] == "simulated"


# ---------------- other modules ----------------
def test_satellite(auth_headers):
    r = client.get("/api/satellite/scan", headers=auth_headers)
    j = r.json()
    assert r.status_code == 200 and len(j["zones"]) == 24
    assert round(sum(j["summary"].values())) == 100


def test_yield_irrigation_fertilizer(auth_headers):
    assert client.post("/api/yield/predict", headers=auth_headers, json={"crop": "wheat"}).status_code == 200
    assert client.post("/api/irrigation/recommend", headers=auth_headers, json={}).status_code == 200
    assert client.post("/api/fertilizer/recommend", headers=auth_headers, json={}).status_code == 200


def test_disease_real_model(auth_headers):
    img = sample_image("disease")
    if not img:
        pytest.skip("no sample tomato image available")
    with open(img, "rb") as f:
        r = client.post("/api/disease/diagnose", headers=auth_headers, files={"file": ("leaf.jpg", f, "image/jpeg")})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["diagnosis"] == "Late Blight" and j["confidence"] > 50 and "second_opinion" not in j


def test_disease_rejects_non_image_and_corrupt(auth_headers):
    assert client.post("/api/disease/diagnose", headers=auth_headers, files={"file": ("a.txt", b"hi", "text/plain")}).status_code == 400
    assert client.post("/api/disease/diagnose", headers=auth_headers, files={"file": ("a.jpg", b"notanimage", "image/jpeg")}).status_code == 400


def test_disease_second_opinion_is_separate_and_optional(auth_headers, with_key):
    img = sample_image("disease")
    if not img:
        pytest.skip("no sample tomato image available")
    with mock.patch("app.gemini_client.requests.post", return_value=gemini_ok("Looks like late blight.")):
        with open(img, "rb") as f:
            j = client.post("/api/disease/diagnose", headers=auth_headers, files={"file": ("l.jpg", f, "image/jpeg")}).json()
    assert j["diagnosis"] == "Late Blight" and "late blight" in j["second_opinion"]["text"].lower()
    # Gemini failing must NOT break the CNN diagnosis
    with mock.patch("app.gemini_client.requests.post", return_value=gemini_err(429, "q", "RESOURCE_EXHAUSTED")):
        with open(img, "rb") as f:
            r = client.post("/api/disease/diagnose", headers=auth_headers, files={"file": ("l.jpg", f, "image/jpeg")})
    assert r.status_code == 200 and "second_opinion" not in r.json()


def test_species_identify(auth_headers):
    from ml import species_infer
    if not os.path.exists(species_infer.MODEL_PATH):
        pytest.skip("species model not trained yet")
    img = sample_image("species")
    if not img:
        pytest.skip("no sample species image available")
    with open(img, "rb") as f:
        r = client.post("/api/species/identify", headers=auth_headers, files={"file": ("m.jpg", f, "image/jpeg")})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["species"] == "Mango" and len(j["top3"]) == 3


def test_history_records_scans(auth_headers):
    rows = client.get("/api/history/", headers=auth_headers).json()
    assert any(x["module"] == "disease" for x in rows)


# ---------------- config / .env loading (works without python-dotenv) ----------------
def test_env_parser_handles_bom_quotes_comments_export(tmp_path, monkeypatch):
    from app import config
    f = tmp_path / ".env"
    f.write_bytes("﻿# comment\nexport FOO_A = \"quoted value\"\nFOO_B=plain # trailing\nFOO_C='single'\n".encode("utf-8"))
    for k in ("FOO_A", "FOO_B", "FOO_C"):
        monkeypatch.delenv(k, raising=False)
    config._parse_env_file(str(f))
    assert os.environ["FOO_A"] == "quoted value" and os.environ["FOO_B"] == "plain" and os.environ["FOO_C"] == "single"
    for k in ("FOO_A", "FOO_B", "FOO_C"):
        monkeypatch.delenv(k, raising=False)


def test_placeholder_key_is_treated_as_missing_with_clear_message(auth_headers, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "PASTE_YOUR_KEY_HERE")
    j = client.post("/api/assistant/chat", headers=auth_headers, json={"message": "hello"}).json()
    assert j["error_code"] == "no_key" and "placeholder" in j["notice"]
    h = client.get("/api/health").json()
    assert h["gemini_configured"] is False and h["key_is_still_placeholder"] is True


def test_assistant_status_endpoint(auth_headers, with_key):
    from app.routers import assistant
    assistant._status_cache.update(t=0.0, value=None)
    with mock.patch("app.gemini_client.requests.post", return_value=gemini_ok("ok")):
        j = client.get("/api/assistant/status", headers=auth_headers).json()
    assert j["ok"] is True and j["model"]
    assistant._status_cache.update(t=0.0, value=None)
    with mock.patch("app.gemini_client.requests.post", return_value=gemini_err(400, "API key not valid.", "INVALID_ARGUMENT")):
        j = client.get("/api/assistant/status", headers=auth_headers).json()
    assert j["ok"] is False and j["error_code"] == "auth"
    assistant._status_cache.update(t=0.0, value=None)


def test_offline_greeting_variants(auth_headers):
    for msg in ("hii", "hello", "heyy", "Hi!"):
        j = client.post("/api/assistant/chat", headers=auth_headers, json={"message": msg}).json()
        assert j["reply"].startswith("Hi!"), msg


def test_frontend_not_cached():
    assert client.get("/app.html").headers.get("cache-control") == "no-store"


# ---------------- disease model robustness ----------------
def _png(color):
    import io
    from PIL import Image
    b = io.BytesIO(); Image.new("RGB", (200, 200), color).save(b, "PNG"); return b.getvalue()


@pytest.mark.parametrize("color", [(255, 255, 255), (60, 160, 60), (0, 0, 0), (128, 128, 128)])
def test_blank_images_are_not_diagnosed_as_a_disease(auth_headers, color):
    r = client.post("/api/disease/diagnose", headers=auth_headers, files={"file": ("x.png", _png(color), "image/png")})
    j = r.json()
    assert r.status_code == 200
    assert j["diagnosis"] != "Late Blight", j          # the original bug: everything became Late Blight
    assert j["not_tomato"] or j["uncertain"], j        # must be rejected or flagged, never a confident disease


def test_severity_is_not_faked_from_confidence(auth_headers):
    img = sample_image("disease")
    if not img:
        pytest.skip("no sample")
    with open(img, "rb") as f:
        j = client.post("/api/disease/diagnose", headers=auth_headers, files={"file": ("l.jpg", f, "image/jpeg")}).json()
    assert j["severity"] in ("Not measured", "None", "-")
