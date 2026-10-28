// ---------------------------------------------------------------------------
// API base URL resolution
//
// Preferred setup: open the app from the backend itself (http://localhost:8000),
// because FastAPI serves this frontend too. Then API_BASE is "" (same origin)
// and there is no port mismatch or CORS to worry about.
//
// If you serve the frontend separately (Live Server, python -m http.server ...),
// we probe the usual backend addresses and use the first one that answers
// /api/health. You can also force one:  localStorage.setItem("agrivision_api", "http://localhost:8000")
// ---------------------------------------------------------------------------
const API_CANDIDATES = [
  "",                          // same origin (backend serving the frontend)
  "http://localhost:8000",
  "http://127.0.0.1:8000",
  "http://localhost:8001",
  "http://127.0.0.1:8001",
];

let _apiBasePromise = null;

async function _probe(base) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 2500);
  try {
    const res = await fetch(`${base}/api/health`, { signal: ctrl.signal });
    if (!res.ok) return false;
    const data = await res.json().catch(() => null);
    return !!(data && data.status === "ok");
  } catch (_) {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

function resolveApiBase(force = false) {
  if (_apiBasePromise && !force) return _apiBasePromise;
  _apiBasePromise = (async () => {
    let saved = null;
    try { saved = localStorage.getItem("agrivision_api"); } catch (_) {}
    const list = saved ? [saved, ...API_CANDIDATES] : API_CANDIDATES;
    for (const base of list) {
      if (await _probe(base)) return base;
    }
    return null;
  })();
  return _apiBasePromise;
}

// ---------------------------------------------------------------------------
// Session helpers
// ---------------------------------------------------------------------------
function getToken() {
  return localStorage.getItem("agrivision_token");
}
function setSession(token, user) {
  localStorage.setItem("agrivision_token", token);
  localStorage.setItem("agrivision_user", JSON.stringify(user));
}
function getUser() {
  try { return JSON.parse(localStorage.getItem("agrivision_user")); } catch { return null; }
}
function logout() {
  localStorage.removeItem("agrivision_token");
  localStorage.removeItem("agrivision_user");
  window.location.href = "login.html";
}
function requireAuth() {
  if (!getToken()) window.location.href = "login.html";
}

// ---------------------------------------------------------------------------
// Fetch wrapper
// ---------------------------------------------------------------------------
const BACKEND_DOWN_MSG =
  "Cannot reach the AgriVision backend. In the backend folder run:  " +
  "python -m uvicorn app.main:app --port 8000   then open http://localhost:8000";

async function apiFetch(path, options = {}) {
  const base = await resolveApiBase();
  if (base === null) {
    // Allow a retry on the next call (user may have just started the server).
    _apiBasePromise = null;
    throw new Error(BACKEND_DOWN_MSG);
  }

  const headers = { ...(options.headers || {}) };
  const token = getToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;
  if (!(options.body instanceof FormData) && options.body) {
    headers["Content-Type"] = "application/json";
  }

  let res;
  try {
    res = await fetch(`${base}${path}`, { ...options, headers });
  } catch (_) {
    _apiBasePromise = null;
    throw new Error(BACKEND_DOWN_MSG);
  }

  if (res.status === 401 && !path.startsWith("/api/auth/")) {
    logout();
    throw new Error("Session expired. Please log in again.");
  }

  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = data && data.detail;
    const msg = typeof detail === "string"
      ? detail
      : Array.isArray(detail) ? detail.map(d => d.msg).join("; ") : "Request failed";
    throw new Error(msg);
  }
  return data;
}
