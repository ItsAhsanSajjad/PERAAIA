from __future__ import annotations

import html
import os
import threading
import time
import uuid
from collections import deque
from urllib.parse import quote
from typing import List, Dict, Any, Optional, Tuple, Deque

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

# Phase 8 — auth layer.
from auth import (
    Identity,
    resolve_identity,
    is_protected_mode,
    auth_status_summary,
    AUTH_MODE,
)

from retriever import retrieve, retrieve_multi, rewrite_contextual_query
from answerer import answer_question
from query_intent import (
    classify_intent,
    llm_expand_vague_query,
    INTENT_OUT_OF_SCOPE,
    INTENT_VAGUE,
    INTENT_FOLLOWUP,
)
from log_config import setup_logging, get_logger, request_id_var
from openai_clients import has_api_key
from session_store import get_session_store, SessionTurn
from audit_trail import log_audit_entry
from context_state import anchor_query, extract_subject, extract_evidence_metadata
from smalltalk_intent import decide_smalltalk

# Auto-indexer (safe blue/green builds + atomic pointer switch)
from index_manager import SafeAutoIndexer, IndexManagerConfig

log = get_logger("pera.api")


# ============================================================
# Phase 7 — Hardening configuration (env-driven, safe defaults)
# ============================================================
# Single source of truth for all production-hardening knobs. Defaults
# are conservative for production; local development can relax them
# via env vars without code changes.

# CORS — comma-separated origin list, OR the literal value "*" for
# allow-all-origins (intentionally preserved per Phase 8 requirements).
#
# Configuration semantics:
#   CORS_ALLOW_ORIGINS="*"                       → allow ALL origins (dev or
#                                                  intentional production
#                                                  open API). Credentials are
#                                                  forced off in this mode
#                                                  because browsers reject
#                                                  the combination.
#   CORS_ALLOW_ORIGINS="https://a,https://b"     → strict allow-list.
#   CORS_ALLOW_ORIGINS=""                        → no CORSMiddleware mounted
#                                                  (most restrictive).
#
# The default in development is "*" (allow-all) for frictionless local
# iteration. The default in production is "" (no middleware) so a forgotten
# env var fails closed instead of leaking.
APP_VERSION = os.getenv("APP_VERSION", "2.3.0")
APP_ENV = os.getenv("APP_ENV", "development").strip().lower()  # "development" | "production"

_cors_default = "*" if APP_ENV != "production" else ""
_cors_raw = os.getenv("CORS_ALLOW_ORIGINS", _cors_default).strip()

# Detect the explicit allow-all directive. Treat "*" as a sentinel,
# NOT as an item in a list, so a misconfigured "*, https://a" is read
# as allow-all (matching the user-facing intent rather than silently
# constructing a broken list).
CORS_ALLOW_ALL = "*" in [s.strip() for s in _cors_raw.split(",")]
if CORS_ALLOW_ALL:
    CORS_ALLOW_ORIGINS: List[str] = ["*"]
else:
    CORS_ALLOW_ORIGINS = [o.strip() for o in _cors_raw.split(",") if o.strip()]

CORS_ALLOW_CREDENTIALS = (
    os.getenv("CORS_ALLOW_CREDENTIALS", "0").strip() == "1"
)

# Body / upload size guardrails.
# MAX_REQUEST_BYTES is a global cap enforced before the route runs.
# MAX_AUDIO_UPLOAD_BYTES is enforced inside /transcribe specifically.
MAX_REQUEST_BYTES = int(os.getenv("MAX_REQUEST_BYTES", str(2 * 1024 * 1024)))         # 2 MB
MAX_AUDIO_UPLOAD_BYTES = int(os.getenv("MAX_AUDIO_UPLOAD_BYTES", str(25 * 1024 * 1024)))  # 25 MB

# Rate limiting — simple in-process token bucket per client IP.
# Set RATE_LIMIT_ENABLED=0 to disable entirely (default ON).
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "1").strip() != "0"
RATE_LIMIT_ASK_PER_MIN = int(os.getenv("RATE_LIMIT_ASK_PER_MIN", "30"))
RATE_LIMIT_TRANSCRIBE_PER_MIN = int(os.getenv("RATE_LIMIT_TRANSCRIBE_PER_MIN", "20"))
RATE_LIMIT_BURST = int(os.getenv("RATE_LIMIT_BURST", "10"))
# Comma-separated CIDR-less prefixes whose requests bypass the limiter
# (defaults: localhost variants — never deny developer traffic by accident).
RATE_LIMIT_BYPASS_IPS = {
    s.strip() for s in os.getenv("RATE_LIMIT_BYPASS_IPS", "127.0.0.1,::1").split(",")
    if s.strip()
}

# Trusted proxy header name. When set, the limiter and audit trail
# read the client IP from this header (e.g. "X-Forwarded-For"). Empty
# disables proxy-aware extraction.
TRUSTED_FORWARDED_HEADER = os.getenv("TRUSTED_FORWARDED_HEADER", "").strip()


# ============================================================
# FastAPI app
# ============================================================
app = FastAPI(title="PERA AI Backend", version=APP_VERSION)

# CORS — env-driven. Three behaviors:
#   • CORS_ALLOW_ORIGINS="*"                  → allow ALL origins
#     (CORS_ALLOW_CREDENTIALS is force-disabled because browsers reject
#      `*` + credentials combination).
#   • CORS_ALLOW_ORIGINS="<csv list>"         → strict allow-list.
#   • CORS_ALLOW_ORIGINS=""                   → no middleware mounted
#     (production default, fails closed).
if CORS_ALLOW_ALL:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,  # forced off — wildcard + creds is illegal
        allow_methods=["GET", "POST", "DELETE", "PUT", "OPTIONS"],
        allow_headers=["*"],
    )
elif CORS_ALLOW_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ALLOW_ORIGINS,
        allow_credentials=CORS_ALLOW_CREDENTIALS,
        allow_methods=["GET", "POST", "DELETE", "PUT", "OPTIONS"],
        allow_headers=["*"],
    )

# Allow iframes (fix for browser blocking) + Phase 7 security headers.
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    if "X-Frame-Options" in response.headers:
        del response.headers["X-Frame-Options"]
    # CSP: keep frame-ancestors permissive in development (PDF viewer
    # iframes), tighten in production via APP_ENV.
    if APP_ENV == "production":
        response.headers.setdefault(
            "Content-Security-Policy",
            "frame-ancestors 'self'",
        )
    else:
        response.headers.setdefault("Content-Security-Policy", "frame-ancestors *")
    # Conservative defaults — safe for an API surface.
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("X-Permitted-Cross-Domain-Policies", "none")
    if APP_ENV == "production":
        # Browsers ignore HSTS over plain HTTP, so it's safe to send always.
        response.headers.setdefault(
            "Strict-Transport-Security",
            "max-age=31536000; includeSubDomains",
        )
    return response


# Phase 7 — Body size cap (declared Content-Length only).
# This cheaply rejects oversize bodies before the route runs. Per-route
# guards (e.g. /transcribe) still enforce stricter limits on the actual
# bytes read, in case a client lies about Content-Length.
@app.middleware("http")
async def enforce_max_body_size(request: Request, call_next):
    cl = request.headers.get("content-length")
    if cl:
        try:
            if int(cl) > MAX_REQUEST_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": "request_too_large",
                        "max_bytes": MAX_REQUEST_BYTES,
                        "received": int(cl),
                    },
                )
        except ValueError:
            pass
    return await call_next(request)


# ============================================================
# Phase 7 — Per-IP rate limiter (in-process, dependency-free)
# ============================================================
# Sliding-window counter per (client_ip, route_group). Two route
# groups are protected by default: "ask" (/api/ask, /ask, /ask_json,
# /transcribe is its own group). All other routes are unmetered.
#
# Limitations (intentional):
#   • In-process state — does NOT span workers. For a single-process
#     uvicorn deployment this is sufficient; for multi-worker, terminate
#     at a reverse proxy that has its own limiter.
#   • Per-IP only. No per-user tokens (auth is out of Phase 7 scope).
#   • Burst support is naive (uses the same window).
#
# To disable: set RATE_LIMIT_ENABLED=0.

_RATE_LOCK = threading.Lock()
_RATE_BUCKETS: Dict[Tuple[str, str], Deque[float]] = {}
_RATE_LAST_GC = [0.0]  # mutable holder for occasional cleanup

# Route → (group_name, requests_per_minute) lookup. Anything not
# listed here is not rate-limited.
_RATE_RULES: Dict[str, Tuple[str, int]] = {
    "/api/ask":    ("ask",        RATE_LIMIT_ASK_PER_MIN),
    "/ask":        ("ask",        RATE_LIMIT_ASK_PER_MIN),
    "/ask_json":   ("ask",        RATE_LIMIT_ASK_PER_MIN),
    "/transcribe": ("transcribe", RATE_LIMIT_TRANSCRIBE_PER_MIN),
}


def _client_ip(request: Request) -> str:
    """Resolve the client IP, honoring TRUSTED_FORWARDED_HEADER if set.
    Falls back to the socket peer; never raises."""
    if TRUSTED_FORWARDED_HEADER:
        v = request.headers.get(TRUSTED_FORWARDED_HEADER)
        if v:
            # X-Forwarded-For may contain a list; take the first.
            return v.split(",")[0].strip()
    try:
        return request.client.host if request.client else "unknown"
    except Exception:
        return "unknown"


def _gc_buckets(now: float) -> None:
    """Drop stale buckets every ~60s to keep memory bounded."""
    if now - _RATE_LAST_GC[0] < 60.0:
        return
    _RATE_LAST_GC[0] = now
    horizon = now - 120.0  # keep last 2 minutes for safety
    dead: List[Tuple[str, str]] = []
    for k, dq in _RATE_BUCKETS.items():
        # If every entry is older than the horizon, the bucket is dead.
        if not dq or dq[-1] < horizon:
            dead.append(k)
    for k in dead:
        _RATE_BUCKETS.pop(k, None)


def _check_rate(client_ip: str, group: str, rpm: int) -> Tuple[bool, int, int]:
    """Returns (allowed, remaining, retry_after_seconds).
    `rpm` is the steady-state per-minute cap; bursts up to
    rpm + RATE_LIMIT_BURST are tolerated within the window."""
    if rpm <= 0:
        return True, 0, 0
    window = 60.0
    cap = rpm + max(0, RATE_LIMIT_BURST)
    now = time.time()
    key = (client_ip, group)
    with _RATE_LOCK:
        dq = _RATE_BUCKETS.get(key)
        if dq is None:
            dq = deque()
            _RATE_BUCKETS[key] = dq
        # Drop expired timestamps.
        cutoff = now - window
        while dq and dq[0] < cutoff:
            dq.popleft()
        if len(dq) >= cap:
            oldest = dq[0]
            retry_after = max(1, int(window - (now - oldest)))
            return False, 0, retry_after
        dq.append(now)
        remaining = max(0, cap - len(dq))
        _gc_buckets(now)
        return True, remaining, 0


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if not RATE_LIMIT_ENABLED:
        return await call_next(request)
    rule = _RATE_RULES.get(request.url.path)
    if rule is None:
        return await call_next(request)
    ip = _client_ip(request)
    if ip in RATE_LIMIT_BYPASS_IPS:
        return await call_next(request)
    group, rpm = rule
    allowed, remaining, retry_after = _check_rate(ip, group, rpm)
    if not allowed:
        log.warning(
            "rate_limit: 429 ip=%s group=%s path=%s retry=%ds",
            ip, group, request.url.path, retry_after,
        )
        return JSONResponse(
            status_code=429,
            content={
                "error": "rate_limited",
                "group": group,
                "retry_after_seconds": retry_after,
            },
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": str(rpm + RATE_LIMIT_BURST),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Group": group,
            },
        )
    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(rpm + RATE_LIMIT_BURST)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    response.headers["X-RateLimit-Group"] = group
    return response


# ============================================================
# Phase 8 — Identity resolution middleware
# ============================================================
# Runs on every request, populates request.state.identity, but does
# NOT enforce. Enforcement is per-route via the require_identity /
# require_role dependencies. This way:
#   • /health and /ready stay open (no identity required).
#   • Audit log can still record an "anonymous" identity for unprotected
#     paths.
#   • Protected paths can short-circuit cleanly with a structured 401.
#
# The middleware never raises and never blocks a request.

@app.middleware("http")
async def identity_middleware(request: Request, call_next):
    headers = {k: v for k, v in request.headers.items()}
    identity, error = resolve_identity(headers)
    request.state.identity = identity
    request.state.identity_error = error  # consumed by require_identity
    return await call_next(request)


# ── Dependencies ──────────────────────────────────────────────
def get_identity(request: Request) -> Identity:
    """FastAPI dependency: returns the resolved Identity on request.state."""
    ident = getattr(request.state, "identity", None)
    if ident is None:
        # Defensive: middleware should always have populated this.
        from auth import _dev_anonymous_identity
        ident = _dev_anonymous_identity()
    return ident


def require_identity(request: Request) -> Identity:
    """FastAPI dependency: 401 when AUTH_MODE is protected and the
    request did not present a valid credential.

    In disabled (dev) mode, returns the anonymous identity and the
    request proceeds — local development stays usable.
    """
    err = getattr(request.state, "identity_error", None)
    ident = getattr(request.state, "identity", None) or get_identity(request)
    if is_protected_mode():
        if err or ident.anonymous:
            payload = err or {
                "error": "auth_required", "reason": "no_identity",
                "mode": AUTH_MODE,
            }
            log.warning(
                "auth: 401 mode=%s path=%s reason=%s",
                AUTH_MODE, request.url.path, payload.get("reason"),
            )
            raise HTTPException(status_code=401, detail=payload)
    return ident


def require_role(*required_roles: str):
    """Build a dependency that enforces at least one of `required_roles`.

    In disabled mode, role checks are a no-op (anonymous dev identity
    has no roles by default). Operators can still test role behavior by
    switching to bearer/JWT and providing the role list.
    """
    def _dep(request: Request) -> Identity:
        ident = require_identity(request)
        # In disabled mode, do not enforce roles — dev convenience.
        if not is_protected_mode():
            return ident
        if not any(ident.has_role(r) for r in required_roles):
            log.warning(
                "rbac: 403 sub=%s roles=%s required=%s path=%s",
                ident.subject, ident.roles, list(required_roles), request.url.path,
            )
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "forbidden",
                    "reason": "missing_role",
                    "required_any_of": list(required_roles),
                },
            )
        return ident
    return _dep


# ============================================================
# Static PDFs setup (serves real files at /assets/data/<filename>)
# ============================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(_SCRIPT_DIR, "assets", "data"))
app.mount("/assets/data", StaticFiles(directory=DATA_DIR), name="data")


# ============================================================
# Auto-indexer setup
# ============================================================
INDEXES_ROOT = os.getenv("INDEXES_ROOT", os.path.join("assets", "indexes")).replace("\\", "/")
ACTIVE_POINTER_PATH = os.getenv("INDEX_POINTER_PATH", os.path.join("assets", "indexes", "ACTIVE.json")).replace("\\", "/")

POLL_SECONDS = int(os.getenv("INDEX_POLL_SECONDS", "30"))
KEEP_LAST_N = int(os.getenv("INDEX_KEEP_LAST_N", "3"))
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "4500"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "350"))

indexer = SafeAutoIndexer(
    IndexManagerConfig(
        data_dir=DATA_DIR,
        indexes_root=INDEXES_ROOT,
        active_pointer_path=ACTIVE_POINTER_PATH,
        poll_seconds=POLL_SECONDS,
        keep_last_n=KEEP_LAST_N,
        chunk_max_chars=CHUNK_MAX_CHARS,
        chunk_overlap_chars=CHUNK_OVERLAP_CHARS,
    )
)

# ============================================================
# Request ID Middleware
# ============================================================
class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Accept forwarded request ID or generate a new one
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = rid
        request_id_var.set(rid)
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response

app.add_middleware(RequestIDMiddleware)


@app.on_event("startup")
def _startup():
    """Initialize logging and start background indexer."""
    setup_logging()
    log.info("PERA AI Backend starting up")
    indexer.start_background()
    # Pre-warm the per-PDF page-text cache used by reference resolution
    # so the first user query doesn't pay a multi-minute cold-start
    # cost on very large PDFs (hundreds of pages).
    try:
        from answerer import prewarm_pdf_cache
        prewarm_pdf_cache(DATA_DIR)
    except Exception as exc:
        log.warning("PDF cache prewarm dispatch failed: %s", exc)


# Mount the admin dashboard API. Defined after `indexer` and helpers so
# `admin_routes` can import them via the running app module.
from admin_routes import admin_router as _admin_router  # noqa: E402
app.include_router(_admin_router)


# ============================================================
# API models
# ============================================================
class QueryRequest(BaseModel):
    user_id: str
    message: str


class QueryResponse(BaseModel):
    user_id: str
    answer: str


class QueryResponseJSON(BaseModel):
    user_id: str
    answer: str
    references: List[Dict[str, Any]]


# ============================================================
# Helpers
# ============================================================
def _base_url() -> str:
    # accept both spellings
    return os.getenv("BASE_URL", os.getenv("Base_URL", "https://askpera.infinitysol.agency")).rstrip("/")


def _is_safe_under_assets_data(abs_path: str) -> bool:
    """Prevent path traversal to ensure the file is under the assets/data directory."""
    try:
        data_abs = os.path.abspath(DATA_DIR)
        file_abs = os.path.abspath(abs_path)
        return os.path.commonpath([data_abs, file_abs]) == data_abs
    except Exception:
        return False


def _extract_filename(ref_path: str, doc_name: str) -> str:
    """
    Extract filename from reference path/doc name.
    Ensures we only ever return a filename (no directories).
    """
    p = (ref_path or "").strip().replace("\\", "/")
    if p.startswith("/assets/data/") or p.startswith("assets/data/"):
        return os.path.basename(p)
    if p.lower().endswith(".pdf") or p.lower().endswith(".docx"):
        return os.path.basename(p)
    return os.path.basename((doc_name or "").strip())


def _safe_join_base_and_path(base_url: str, path: str) -> str:
    if not path:
        return ""
    p = path.strip().replace("\\", "/")
    if p.startswith("http://") or p.startswith("https://"):
        return p
    if not p.startswith("/"):
        p = "/" + p
    return f"{base_url}{p}"


def _build_assets_data_url(baseurl: str, filename: str) -> str:
    """Real static file path: https://<host>/assets/data/<filename>"""
    safe = quote(filename)
    return f"{baseurl}/assets/data/{safe}"


def _build_forced_download_url(baseurl: str, filename: str) -> str:
    """Optional forced-download endpoint (Content-Disposition: attachment)."""
    safe = quote(filename)
    return f"{baseurl}/download/{safe}"


def _compress_int_ranges(nums: List[int]) -> str:
    if not nums:
        return ""
    nums = sorted(set(int(x) for x in nums if isinstance(x, int) or str(x).isdigit()))
    if not nums:
        return ""
    out: List[str] = []
    start = prev = nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
        else:
            out.append(str(start) if start == prev else f"{start}–{prev}")
            start = prev = n
    out.append(str(start) if start == prev else f"{start}–{prev}")
    return ", ".join(out)


def _extract_pages(ref: Dict[str, Any]) -> List[int]:
    pages: List[int] = []
    ps = ref.get("page_start")
    pe = ref.get("page_end")
    try:
        if ps is not None:
            a = int(ps)
            b = int(pe) if pe is not None else a
            for p in range(min(a, b), max(a, b) + 1):
                pages.append(p)
    except Exception:
        pass
    return pages


def _extract_locs(ref: Dict[str, Any]) -> List[str]:
    locs: List[str] = []
    # include loc_start (and loc if present)
    for k in ("loc", "loc_start"):
        v = ref.get(k)
        if v is not None and str(v).strip():
            locs.append(str(v).strip())
    return sorted(set(locs))


def _apply_page_anchor_if_missing(open_url: str, pages: List[int]) -> str:
    if not open_url:
        return open_url
    if "#" in open_url:
        return open_url
    if pages:
        p = min(int(x) for x in pages if isinstance(x, int) or str(x).isdigit())
        return f"{open_url}#page={p}"
    return open_url


def _build_open_url_like_streamlit(baseurl: str, ref: Dict[str, Any]) -> str:
    """
    Match Streamlit behavior:
    - Prefer answerer-produced open_url (already includes #page= when available)
    - Else build base + ref.path + ref.url_hint
    - Else fallback to /assets/data/<filename>
    - Ensure #page= is applied if we have pages and no anchor
    """
    # 1) Prefer answerer open_url
    u = (ref.get("open_url") or "").strip()
    if u:
        pages = _extract_pages(ref)
        return _apply_page_anchor_if_missing(u, pages)

    # 2) base + path + url_hint
    path = (ref.get("path") or "").strip()
    url_hint = (ref.get("url_hint") or "").strip()
    if path:
        built = _safe_join_base_and_path(baseurl, path)
        u2 = f"{built}{url_hint}"
        pages = _extract_pages(ref)
        return _apply_page_anchor_if_missing(u2, pages)

    # 3) fallback to /assets/data/<filename>
    doc = (ref.get("document") or "Unknown document").strip()
    filename = _extract_filename("", doc)
    u3 = _build_assets_data_url(baseurl, filename)
    pages = _extract_pages(ref)
    return _apply_page_anchor_if_missing(u3, pages)


def _group_references_like_streamlit(refs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Group by (document, path) and aggregate pages/locs so output matches Streamlit style:
      Document
      Pages: ...
      Sections / Paragraphs: ...
    """
    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for r in refs or []:
        if not isinstance(r, dict):
            continue

        doc = (r.get("document") or r.get("doc_name") or "Unknown document").strip()
        path = (r.get("path") or r.get("public_path") or "").strip()

        # if no path provided, synthesize /assets/data/<filename>
        if not path:
            filename = _extract_filename(r.get("path", ""), doc)
            path = f"/assets/data/{filename}"

        key = (doc, path)
        g = grouped.get(key)
        if not g:
            g = {
                "document": doc,
                "path": path,
                "open_url": (r.get("open_url") or "").strip(),
                "url_hint": (r.get("url_hint") or "").strip(),
                "pages": set(),
                "locs": set(),
                "snippet": (r.get("snippet") or "").strip(),
            }
            grouped[key] = g

        # keep first snippet if already set; otherwise fill
        if not g.get("snippet") and (r.get("snippet") or "").strip():
            g["snippet"] = (r.get("snippet") or "").strip()

        # accumulate pages
        for p in _extract_pages(r):
            g["pages"].add(int(p))

        # accumulate locs
        for loc in _extract_locs(r):
            g["locs"].add(loc)

        # keep open_url/url_hint if missing
        if not g.get("open_url") and (r.get("open_url") or "").strip():
            g["open_url"] = (r.get("open_url") or "").strip()
        if not g.get("url_hint") and (r.get("url_hint") or "").strip():
            g["url_hint"] = (r.get("url_hint") or "").strip()

    out: List[Dict[str, Any]] = []
    for g in grouped.values():
        out.append(
            {
                "document": g["document"],
                "path": g["path"],
                "open_url": g.get("open_url", ""),
                "url_hint": g.get("url_hint", ""),
                "pages": sorted(list(g["pages"])),
                "locs": sorted(list(g["locs"])),
                "snippet": g.get("snippet", ""),
            }
        )

    out.sort(key=lambda x: x.get("document", ""))
    return out


# ============================================================
# Download endpoint (kept) - forces download
# ============================================================
@app.get("/download/{filename:path}")
def download_pdf(filename: str):
    """
    Forced download from assets/data with Content-Disposition: attachment.
    NOTE: References default to open_url (/assets/data...) for page anchors.
    """
    filename = (filename or "").strip()
    if not filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    abs_path = os.path.join(DATA_DIR, filename).replace("\\", "/")

    if not _is_safe_under_assets_data(abs_path):
        raise HTTPException(status_code=400, detail="Invalid path.")

    if not os.path.exists(abs_path) or not os.path.isfile(abs_path):
        raise HTTPException(status_code=404, detail="File not found.")

    return FileResponse(
        abs_path,
        media_type="application/pdf",
        filename=os.path.basename(abs_path),
        headers={"Content-Disposition": f'attachment; filename="{os.path.basename(abs_path)}"'},
    )


# ============================================================
# Main endpoint (HTML)
# FIX: Open links should jump to the correct page (Streamlit-like)
# ============================================================
@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest, identity: Identity = Depends(require_identity)):
    retrieval = retrieve(request.message)
    result = answer_question(request.message, retrieval)

    baseurl = _base_url()
    answer_text = result.get("answer", "") or ""

    # Group refs like Streamlit (aggregate pages/locs)
    raw_refs = result.get("references", []) or []
    grouped = _group_references_like_streamlit(raw_refs)

    parts: List[str] = []
    parts.append('<div style="font-family: Arial, sans-serif; line-height: 1.6; margin:0; padding:0;">')
    parts.append(f'<p style="margin:0; padding:0;">{answer_text}</p>')

    if grouped:
        parts.append('<hr style="margin:8px 0;" />')
        parts.append('<h3 style="margin:4px 0;">References</h3>')
        parts.append('<ol style="margin:0; padding-left:18px;">')

        for g in grouped:
            doc = (g.get("document") or "Unknown document").strip()
            snippet = (g.get("snippet") or "").strip()
            pages: List[int] = g.get("pages") or []
            locs: List[str] = g.get("locs") or []

            # Build Streamlit-like open URL: prefer open_url/path/url_hint and ensure #page=...
            open_url = _build_open_url_like_streamlit(
                baseurl,
                {
                    "document": doc,
                    "path": g.get("path"),
                    "open_url": g.get("open_url"),
                    "url_hint": g.get("url_hint"),
                    # Provide page_start/page_end from aggregated pages (for anchor)
                    "page_start": min(pages) if pages else None,
                    "page_end": max(pages) if pages else None,
                    "loc_start": (locs[0] if locs else None),
                },
            )

            meta_lines: List[str] = []
            if pages:
                meta_lines.append(f"Pages: {_compress_int_ranges(pages)}")
            if locs:
                joined = "; ".join(locs[:6])
                if len(locs) > 6:
                    joined += f"; +{len(locs)-6} more"
                meta_lines.append(f"Sections / Paragraphs: {joined}")

            # Phase 7 — HTML escape every interpolated value to prevent
            # XSS via document name, snippet, or attacker-controlled URL.
            # Meta lines are escaped before joining so the <br/> stays
            # literal but everything else is safe.
            esc_doc = html.escape(doc, quote=True)
            esc_snippet = html.escape(snippet, quote=True) if snippet else ""
            esc_meta_html = "<br/>".join(html.escape(m, quote=True) for m in meta_lines) if meta_lines else ""
            # URL: must be safe inside an href attribute. quote(safe=...)
            # leaves the structural URL characters (':', '/', '?', '#',
            # '&', '=') intact while encoding anything dangerous.
            safe_open_url = quote(open_url or "", safe=":/?#[]@!$&'()*+,;=%")
            # Belt-and-braces: also escape attribute-special chars in
            # case the URL already contained them.
            esc_href = html.escape(safe_open_url, quote=True)

            parts.append(
                f"""
                <li style="margin-bottom:10px;">
                  <a href="{esc_href}" target="_blank" rel="noopener noreferrer">{esc_doc}</a>
                  {("<div style='margin-top:3px; color:#6b7280; font-size:13px;'>" + esc_meta_html + "</div>") if esc_meta_html else ""}
                  {("<p style='margin:4px 0 0 0;'>" + esc_snippet + "</p>") if esc_snippet else ""}
                </li>
                """.strip()
            )

        parts.append("</ol>")

    parts.append("</div>")
    return QueryResponse(user_id=request.user_id, answer="\n".join(parts).strip())


# ============================================================
# JSON endpoint (for mobile / frontend)
# FIX: Provide Streamlit-like reference object (open_url with #page)
# ============================================================
@app.post("/ask_json", response_model=QueryResponseJSON)
def ask_question_json(request: QueryRequest, identity: Identity = Depends(require_identity)):
    retrieval = retrieve(request.message)
    result = answer_question(request.message, retrieval)

    baseurl = _base_url()

    raw_refs = result.get("references", []) or []
    grouped = _group_references_like_streamlit(raw_refs)

    refs_out: List[Dict[str, Any]] = []
    for g in grouped:
        doc = (g.get("document") or "Unknown document").strip()
        pages: List[int] = g.get("pages") or []
        locs: List[str] = g.get("locs") or []

        # Use the same open_url logic as Streamlit (page anchor)
        open_url = _build_open_url_like_streamlit(
            baseurl,
            {
                "document": doc,
                "path": g.get("path"),
                "open_url": g.get("open_url"),
                "url_hint": g.get("url_hint"),
                "page_start": min(pages) if pages else None,
                "page_end": max(pages) if pages else None,
                "loc_start": (locs[0] if locs else None),
            },
        )

        # Optional forced-download URL (use filename)
        filename = _extract_filename(g.get("path", ""), doc)
        download_url = _build_forced_download_url(baseurl, filename)

        refs_out.append(
            {
                "document": doc,
                "open_url": open_url,
                "download_url": download_url,
                "path": g.get("path"),
                "pages": pages,     # aggregated
                "locs": locs,       # aggregated
                "snippet": (g.get("snippet") or "").strip(),
            }
        )

    return QueryResponseJSON(
        user_id=request.user_id,
        answer=result.get("answer", "") or "",
        references=refs_out,
    )


# ============================================================
# Simple Chat API for Next.js frontend
# ============================================================
class SimpleChatRequest(BaseModel):
    question: str
    conversation_history: Optional[List[Dict[str, Any]]] = None
    session_id: Optional[str] = None  # Part 3: server-side session


class SimpleChatResponse(BaseModel):
    answer: str
    decision: str
    references: List[Dict[str, Any]]
    session_id: Optional[str] = None  # returned so client can persist
    grounding: Optional[Dict[str, Any]] = None  # Part 2 additive


@app.post("/api/ask", response_model=SimpleChatResponse)
def simple_ask(req: Request, request: SimpleChatRequest,
               identity: Identity = Depends(require_identity)):
    """Chat endpoint with session tracking, smalltalk bypass, entity anchoring, and audit trail."""
    question = request.question.strip()
    rid = getattr(req.state, "request_id", "")

    # ── 0. Session ────────────────────────────────────────────
    store = get_session_store()
    session = store.get_or_create(request.session_id)
    sid = session.session_id

    # ── 1. Smalltalk bypass ───────────────────────────────────
    st = decide_smalltalk(question)
    if st and st.is_greeting_only:
        log.info("Smalltalk bypass: '%s' -> deterministic response", question[:40])
        log_audit_entry(
            request_id=rid, session_id=sid, question=question,
            decision="smalltalk", answer_text=st.response,
            is_smalltalk=True,
        )
        return SimpleChatResponse(
            answer=st.response,
            decision="smalltalk",
            references=[],
            session_id=sid,
        )

    # If greeting + real question, use remainingquestion with ack prefix
    ack_prefix = ""
    if st and not st.is_greeting_only and st.remaining_question:
        ack_prefix = st.ack
        question = st.remaining_question
        log.info("Greeting stripped: ack='%s', question='%s'", ack_prefix, question[:60])

    # ── 2. Extract last Q/A from client history FIRST ───────────
    # (needed for entity anchoring when session_id is not sent)
    last_question_client = None
    last_answer_client = None
    if request.conversation_history:
        for msg in reversed(request.conversation_history):
            role = msg.get("role", "")
            content = (msg.get("content") or "").strip()
            if role == "assistant" and last_answer_client is None:
                last_answer_client = content
            elif role == "user" and last_question_client is None:
                last_question_client = content
            if last_question_client and last_answer_client:
                break

    # Build fallback subject from client history if session is empty
    fallback_subject = ""
    if not session.last_subject and last_answer_client:
        fallback_subject = extract_subject(last_answer_client)
        if fallback_subject:
            log.info("Using client-history subject fallback: '%s'", fallback_subject[:60])

    # ── 3. Entity anchoring (session + client fallback) ────────
    anchored_q, subject_used, was_anchored = anchor_query(
        question,
        last_subject=session.last_subject or fallback_subject,
        last_question=session.last_question or last_question_client or "",
        last_answer=session.last_answer or last_answer_client or "",
    )

    # Prefer server-side session for last Q/A if available
    lq = session.last_question or last_question_client or ""
    la = session.last_answer or last_answer_client or ""

    # ── 4. Rewrite query (handles Urdu, abbreviations, follow-ups) ──
    query_for_retrieval = rewrite_contextual_query(anchored_q, lq, la)
    rewrite_diff = ""
    if query_for_retrieval != question:
        rewrite_diff = f"{question} -> {query_for_retrieval}"
    log.info("Query: '%s' -> Anchored: '%s' -> Rewritten: '%s'",
             question[:60], anchored_q[:60], query_for_retrieval[:60])

    # ── 4b. Phase 5 — classify intent (deterministic, no LLM by default).
    # Runs on the rewritten query so abbreviations are resolved before
    # detection. The result is advisory: it shapes retrieval and prompts
    # but never overrides Phase 2 grounding/refusal logic.
    intent_obj = classify_intent(
        query_for_retrieval,
        has_followup_context=bool(was_anchored),
    )
    intent_payload = {
        "primary": intent_obj.primary,
        "is_multi_part": intent_obj.is_multi_part,
        "sub_queries": list(intent_obj.sub_queries or []),
        "is_vague": intent_obj.is_vague,
    }

    # Out-of-scope fast-path: skip retrieval entirely. The answerer also
    # has a fast-path for this case but it expects an empty retrieval;
    # we short-circuit here so we don't pay the FAISS round-trip.
    if intent_obj.primary == INTENT_OUT_OF_SCOPE:
        log.info("Intent=out_of_scope — skipping retrieval, returning refusal")
        retrieval = {"question": query_for_retrieval, "has_evidence": False, "evidence": []}
        result = answer_question(
            question_for_llm if False else question,  # original question for UX
            retrieval,
            conversation_history=None,
            intent=intent_payload,
        )
    else:
        # Build the list of retrieval queries:
        #   • multi-part → use sub_queries (decomposed deterministically)
        #   • vague     → use deterministic expansions; optional LLM upgrade
        #   • else      → single-query (zero behavioral change vs Phase 4)
        retrieval_queries = [query_for_retrieval]
        if intent_obj.is_multi_part and intent_obj.sub_queries:
            retrieval_queries = list(intent_obj.sub_queries)
            # Always keep the original full query as a hedge so we don't
            # lose context that lives only in the unsplit form.
            if query_for_retrieval not in retrieval_queries:
                retrieval_queries.append(query_for_retrieval)
        elif intent_obj.is_vague:
            retrieval_queries = list(intent_obj.expansions or [query_for_retrieval])
            llm_extra = llm_expand_vague_query(query_for_retrieval)
            if llm_extra:
                seen = {q.lower() for q in retrieval_queries}
                for e in llm_extra:
                    if e.lower() not in seen:
                        retrieval_queries.append(e)
                        seen.add(e.lower())

        log.info(
            "Retrieval plan: intent=%s queries=%d -> %s",
            intent_obj.primary, len(retrieval_queries),
            [q[:60] for q in retrieval_queries[:4]],
        )
        if len(retrieval_queries) <= 1:
            retrieval = retrieve(retrieval_queries[0])
        else:
            retrieval = retrieve_multi(retrieval_queries)

        # ── 5. Answer ─────────────────────────────────────────────
        # Only pass conversation_history for genuine follow-ups to prevent
        # prior wrong answers from contaminating standalone questions
        hist_for_llm = request.conversation_history if was_anchored else None
        # Use anchored_q (with resolved pronouns) for the LLM so position
        # filtering sees the actual role name (e.g. "salary of manager development"
        # instead of "salary of this?")
        question_for_llm = anchored_q if was_anchored else question
        result = answer_question(
            question_for_llm,
            retrieval,
            conversation_history=hist_for_llm,
            intent=intent_payload,
        )

    # ── 6. Extract metadata for session + audit ───────────────
    evidence_ids, doc_names = extract_evidence_metadata(retrieval)
    answer_text = result.get("answer", "")
    decision = result.get("decision", "answer")
    grounding = result.get("grounding")
    support_state = result.get("support_state", "")

    # Extract subject from the answer if we didn't anchor
    subject_from_answer = extract_subject(answer_text) if not subject_used else subject_used
    final_subject = subject_used or subject_from_answer

    # ── 7. Update session state ───────────────────────────────
    session.add_turn(SessionTurn(
        question=question,
        normalized_query=query_for_retrieval,
        answer_preview=answer_text[:300],
        decision=decision,
        evidence_ids=evidence_ids,
        doc_names=doc_names,
        subject_entity=final_subject,
    ))
    store.put(sid, session)

    # ── 8. Audit trail ────────────────────────────────────────
    # Build evidence text preview (the actual context sent to LLM)
    evidence_text = ""
    try:
        from answerer import format_evidence_for_llm
        evidence_text = format_evidence_for_llm(retrieval, question)
    except Exception:
        pass

    # Build grounding details dict for audit
    grounding_details = None
    if grounding:
        grounding_details = {
            "score": grounding.get("score"),
            "confidence": grounding.get("confidence"),
            "semantic_support": grounding.get("semantic_support"),
            "support_state": grounding.get("support_state", support_state),
            "unsupported_claims": grounding.get("unsupported_claims", [])[:3],
        }

    log_audit_entry(
        request_id=rid,
        session_id=sid,
        question=question,
        normalized_query=query_for_retrieval,
        decision=decision,
        answer_text=answer_text,
        evidence_ids=evidence_ids,
        doc_names=doc_names,
        references_count=len(result.get("references", [])),
        grounding_score=grounding.get("score") if grounding else None,
        grounding_confidence=grounding.get("confidence", "") if grounding else "",
        grounding_details=grounding_details,
        subject_entity=final_subject,
        evidence_text_preview=evidence_text,
        rewrite_diff=rewrite_diff,
        support_state=support_state,
        intent=intent_payload,
        client_ip=_client_ip(req),
        identity=identity.to_audit_dict() if identity else None,
    )

    # ── 9. Build response ─────────────────────────────────────
    final_answer = (ack_prefix + answer_text) if ack_prefix else answer_text

    return SimpleChatResponse(
        answer=final_answer,
        decision=decision,
        references=result.get("references", []),
        session_id=sid,
        grounding=grounding,
    )


# ============================================================
# NEW: Voice Transcription Endpoint
# ============================================================
from fastapi import File, UploadFile
from speech import transcribe_audio


class TranscribeResponse(BaseModel):
    text: str
    success: bool


# Phase 7 — /transcribe hardening
# Allow-list of accepted MIME prefixes. Whisper handles many formats;
# we accept anything that begins with audio/ plus a few common values
# that browsers send for WebM/Opus blobs and raw PCM uploads.
_ALLOWED_AUDIO_MIME_PREFIXES = ("audio/",)
_ALLOWED_AUDIO_MIME_VALUES = {
    "application/octet-stream",  # browsers often default to this for blobs
    "video/webm",                # MediaRecorder default on some browsers
}
_ALLOWED_AUDIO_EXTS = {
    ".wav", ".mp3", ".m4a", ".mp4", ".ogg", ".oga", ".opus",
    ".webm", ".flac", ".aac", ".amr",
}

# Magic-byte sniff: cheap defense against non-audio binaries pretending
# to be audio. Returns True when the first bytes match a known audio
# / WebM container signature.
def _looks_like_audio(b: bytes) -> bool:
    if not b or len(b) < 4:
        return False
    head = b[:12]
    # RIFF (WAV)
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return True
    # ID3 (MP3) or MP3 frame sync
    if head[:3] == b"ID3":
        return True
    if len(b) >= 2 and b[0] == 0xFF and (b[1] & 0xE0) == 0xE0:
        return True
    # OGG / OGM / Opus
    if head[:4] == b"OggS":
        return True
    # FLAC
    if head[:4] == b"fLaC":
        return True
    # ftyp box (M4A / MP4 audio)
    if len(b) >= 12 and head[4:8] == b"ftyp":
        return True
    # EBML (WebM / Matroska)
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return True
    # AMR
    if head[:5] == b"#!AMR":
        return True
    return False


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(audio: UploadFile = File(...),
                     identity: Identity = Depends(require_identity)):
    """Transcribe audio to text using Whisper.

    Phase 7 hardening:
      • Hard size cap (MAX_AUDIO_UPLOAD_BYTES).
      • Content-type allow-list (audio/*, octet-stream, video/webm).
      • Filename extension allow-list as a fallback signal.
      • Magic-byte sniff to reject non-audio binaries.
      • Clear, structured 4xx errors instead of silent success=False.
    """
    # 1. Content-type screen (cheap reject before reading body).
    ctype = (audio.content_type or "").lower().strip()
    if ctype and not (
        any(ctype.startswith(p) for p in _ALLOWED_AUDIO_MIME_PREFIXES)
        or ctype in _ALLOWED_AUDIO_MIME_VALUES
    ):
        log.warning("transcribe: rejected content-type=%r", ctype)
        raise HTTPException(
            status_code=415,
            detail={"error": "unsupported_media_type", "content_type": ctype},
        )

    # 2. Filename extension screen (advisory; some clients omit ctype).
    fname = (audio.filename or "").lower()
    if fname:
        _, ext = os.path.splitext(fname)
        if ext and ext not in _ALLOWED_AUDIO_EXTS:
            log.warning("transcribe: rejected extension=%r", ext)
            raise HTTPException(
                status_code=415,
                detail={"error": "unsupported_extension", "extension": ext},
            )

    # 3. Bounded body read. Read up to MAX+1 so we can detect overflow
    # without holding the whole oversize body in memory.
    try:
        cap = MAX_AUDIO_UPLOAD_BYTES
        audio_bytes = await audio.read(cap + 1)
    except Exception as e:
        log.error("transcribe: body read failed: %s", e)
        raise HTTPException(
            status_code=400,
            detail={"error": "body_read_failed"},
        )

    if audio_bytes is None or len(audio_bytes) == 0:
        raise HTTPException(
            status_code=400,
            detail={"error": "empty_audio"},
        )
    if len(audio_bytes) > cap:
        log.warning(
            "transcribe: payload too large (%d > %d)",
            len(audio_bytes), cap,
        )
        raise HTTPException(
            status_code=413,
            detail={"error": "audio_too_large", "max_bytes": cap},
        )

    # 4. Magic-byte sniff. Reject obvious non-audio binaries.
    if not _looks_like_audio(audio_bytes):
        log.warning(
            "transcribe: payload failed magic sniff (ctype=%s, ext=%s, len=%d)",
            ctype, fname, len(audio_bytes),
        )
        raise HTTPException(
            status_code=415,
            detail={"error": "not_recognized_as_audio"},
        )

    # 5. Hand off to Whisper. transcribe_audio never raises (returns
    # "⚠️…" on failure) so we surface that as success=False.
    try:
        text = transcribe_audio(audio_bytes)
        is_error = text.startswith("⚠️")
        return TranscribeResponse(
            text=text if not is_error else "",
            success=not is_error,
        )
    except Exception as e:
        log.error("Transcription endpoint error: %s", e, exc_info=True)
        return TranscribeResponse(text="", success=False)


# ============================================================
# NEW: PDF Serving Endpoint (for Next.js)
# ============================================================
from urllib.parse import unquote
import re

def _normalize_for_match(s: str) -> str:
    """Normalize filename for fuzzy matching - strips all special chars."""
    # Replace all types of dashes with hyphen
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")
    # Replace smart quotes
    s = s.replace("'", "'").replace("'", "'").replace('"', '"').replace('"', '"')
    # Remove extra spaces
    s = re.sub(r'\s+', ' ', s).strip()
    # Lowercase for comparison
    return s.lower()

@app.get("/pdf/{filename:path}")
def serve_pdf(filename: str):
    """Serve PDF files for the Next.js frontend."""
    try:
        # 1. First attempt: EXACT match (but URL decoded)
        # This handles 'PERA – FAQs.pdf' correctly if the fs has an en-dash
        decoded_name = unquote(filename).strip()
        filepath = os.path.join(DATA_DIR, decoded_name)

        found = False

        # Check exact existence
        if os.path.exists(filepath) and os.path.isfile(filepath):
            found = True

        # 2. Second attempt: Normalize dashes (en-dash/em-dash -> hyphen)
        if not found:
            normalized_name = decoded_name.replace("–", "-").replace("—", "-")
            filepath_norm = os.path.join(DATA_DIR, normalized_name)
            if os.path.exists(filepath_norm) and os.path.isfile(filepath_norm):
                filepath = filepath_norm
                found = True

        # 3. Third attempt: Try adding .pdf extension if missing
        if not found:
             # Try exact + .pdf
            filepath_ext = os.path.join(DATA_DIR, decoded_name + ".pdf")
            if os.path.exists(filepath_ext):
                filepath = filepath_ext
                found = True
            else:
                 # Try normalized + .pdf
                filepath_ext_norm = os.path.join(DATA_DIR, normalized_name + ".pdf")
                if os.path.exists(filepath_ext_norm):
                    filepath = filepath_ext_norm
                    found = True

        # 4. Fourth attempt: Smart fuzzy scan using normalized comparison
        if not found:
            try:
                # Normalize the target name (remove .pdf, normalize all special chars)
                target_norm = _normalize_for_match(decoded_name)
                if target_norm.endswith(".pdf"):
                    target_norm = target_norm[:-4]

                # Also try without any dashes/special chars at all
                target_stripped = re.sub(r'[^a-z0-9 ]', '', target_norm)

                for f in os.listdir(DATA_DIR):
                    f_norm = _normalize_for_match(f)
                    if f_norm.endswith(".pdf"):
                        f_norm = f_norm[:-4]
                    f_stripped = re.sub(r'[^a-z0-9 ]', '', f_norm)

                    # Match if normalized versions match OR stripped versions match
                    if f_norm == target_norm or f_stripped == target_stripped:
                        filepath = os.path.join(DATA_DIR, f)
                        found = True
                        log.debug("Fuzzy matched: %s -> %s", decoded_name, f)
                        break
            except Exception as e:
                log.warning("Error during fuzzy scan: %s", e)

        if not found:
             log.info("PDF Not Found: %s (decoded: %s)", filename, decoded_name)
             raise HTTPException(status_code=404, detail=f"PDF not found: {filename}")

        # Security check
        if not _is_safe_under_assets_data(filepath):
            log.warning("Access Denied for path: %s", filepath)
            raise HTTPException(status_code=403, detail="Access denied")
        
        return FileResponse(
            filepath,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'inline; filename="{os.path.basename(filepath)}"',
                "Content-Security-Policy": "frame-ancestors *",
                "X-Frame-Options": "ALLOWALL",
                "Access-Control-Allow-Origin": "*"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error("Internal server error serving PDF '%s': %s", filename, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error while serving file")


# ============================================================
# Health & Readiness Endpoints
# ============================================================
# Phase 7 — process start time, used in /health and /ready uptime field.
_PROCESS_STARTED_AT = time.time()


@app.get("/health")
def health_check():
    """Lightweight liveness probe — confirms the process is alive.
    Returns immediately without touching disk or network.
    """
    return {
        "status": "ok",
        "version": APP_VERSION,
        "env": APP_ENV,
        "uptime_seconds": int(time.time() - _PROCESS_STARTED_AT),
        "ts": int(time.time()),
        "auth": auth_status_summary(),
    }


@app.get("/ready")
def readiness_check():
    """Readiness probe — confirms the service can actually serve traffic.

    Returns 200 only when:
      • OpenAI API key is present
      • An active index pointer is set
      • The faiss.index and chunks.jsonl files for that pointer exist
      • The background indexer thread is alive (Phase 7)

    Returns 503 with structured `checks` so operators can see exactly
    which gate failed.
    """
    checks: Dict[str, Any] = {}
    active_dir: Optional[str] = None

    checks["openai_api_key"] = has_api_key()

    try:
        active_dir = indexer.pointer.read()
        checks["active_index"] = active_dir is not None
        checks["active_index_dir"] = active_dir or "none"
    except Exception as e:
        checks["active_index"] = False
        checks["active_index_error"] = str(e)

    if active_dir:
        import os as _os
        faiss_path = _os.path.join(active_dir, "faiss.index").replace("\\", "/")
        chunks_path = _os.path.join(active_dir, "chunks.jsonl").replace("\\", "/")
        checks["faiss_index_exists"] = _os.path.exists(faiss_path)
        checks["chunks_file_exists"] = _os.path.exists(chunks_path)
    else:
        checks["faiss_index_exists"] = False
        checks["chunks_file_exists"] = False

    # Phase 7 — verify the background indexer thread is alive. If it's
    # crashed silently, /ready should reflect that so the orchestrator
    # can recycle the pod.
    try:
        bg = getattr(indexer, "_thread", None)
        checks["indexer_thread_alive"] = bool(bg and bg.is_alive())
    except Exception as e:
        checks["indexer_thread_alive"] = False
        checks["indexer_thread_error"] = str(e)

    ready = all([
        checks.get("openai_api_key", False),
        checks.get("active_index", False),
        checks.get("faiss_index_exists", False),
        checks.get("chunks_file_exists", False),
        # Indexer thread is informational only — do NOT fail readiness on
        # it during the startup window where the thread may not have
        # been spawned yet. Operators can read the field directly.
    ])

    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "ready": ready,
            "version": APP_VERSION,
            "env": APP_ENV,
            "uptime_seconds": int(time.time() - _PROCESS_STARTED_AT),
            "ts": int(time.time()),
            "auth": auth_status_summary(),
            "checks": checks,
        },
    )
