"""
Admin authentication for the PERA AI admin dashboard.

Self-contained module — does not modify the existing auth.py used by the
public /api/ask flow. Provides:

- Hardcoded admin credentials (as specified by the project owner).
- HMAC-SHA256 signed session tokens (JWT-style, three base64url parts:
  header.payload.signature). Uses only the Python standard library so
  no additional dependency is required.
- ``issue_admin_token`` / ``verify_admin_token`` helpers.
- ``require_admin`` FastAPI dependency.
- A tiny per-IP rate limiter for the login endpoint.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

from fastapi import Header, HTTPException, Request

try:
    # python-dotenv is a project dependency; load .env so ADMIN_* env
    # vars are available regardless of import order. Idempotent + safe.
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover — dotenv optional at runtime
    pass

from log_config import get_logger

log = get_logger("pera.admin_auth")


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------
# Credentials are NEVER hardcoded. They are sourced from environment:
#
#   ADMIN_EMAIL          — the admin login email (plain string).
#   ADMIN_PASSWORD_HASH  — a PBKDF2-HMAC-SHA256 hash of the admin password,
#                          in the format produced by ``hash_password`` below:
#                          ``pbkdf2_sha256$<iterations>$<salt_b64>$<hash_b64>``.
#
# Generate a hash with: ``python scripts/generate_admin_password_hash.py``.
# If either variable is missing/malformed, admin login FAILS SAFELY (no
# fallback to any default credential). Values are never written to logs.
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "").strip()
ADMIN_PASSWORD_HASH = os.getenv("ADMIN_PASSWORD_HASH", "").strip()

# PBKDF2 parameters for newly generated hashes (verification reads the
# embedded iteration count, so changing this only affects new hashes).
_PBKDF2_ALGO = "pbkdf2_sha256"
_PBKDF2_ITERATIONS = 240_000

# 12 hours of session validity after login.
TOKEN_TTL_SECONDS = 12 * 60 * 60
TOKEN_ISSUER = "pera-admin"


# ---------------------------------------------------------------------------
# Signing key: generated once on first boot, persisted so admin sessions
# survive server restarts. Lives alongside the app binary; keep it out of
# source control via .gitignore.
# ---------------------------------------------------------------------------
_SECRET_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".admin_secret"
)


def _load_or_create_secret() -> bytes:
    try:
        if os.path.exists(_SECRET_FILE):
            with open(_SECRET_FILE, "rb") as f:
                data = f.read().strip()
            if len(data) >= 32:
                return data
    except OSError as exc:  # pragma: no cover — filesystem issue
        log.warning("Could not read admin secret file: %s", exc)
    secret = secrets.token_bytes(32)
    try:
        with open(_SECRET_FILE, "wb") as f:
            f.write(secret)
        try:
            os.chmod(_SECRET_FILE, 0o600)
        except OSError:
            pass
    except OSError as exc:  # pragma: no cover
        log.warning("Could not persist admin secret: %s", exc)
    return secret


_SECRET = _load_or_create_secret()


# ---------------------------------------------------------------------------
# Minimal HS256 token (JWT-compatible three-part base64url payload).
# ---------------------------------------------------------------------------
def _b64u_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64u_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def _sign(message: bytes) -> bytes:
    return hmac.new(_SECRET, message, hashlib.sha256).digest()


def issue_admin_token(email: str) -> Tuple[str, int]:
    """Return (token, exp_unix). 12h TTL."""
    now = int(time.time())
    exp = now + TOKEN_TTL_SECONDS
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": email,
        "role": "admin",
        "iat": now,
        "exp": exp,
        "iss": TOKEN_ISSUER,
    }
    header_b64 = _b64u_encode(json.dumps(header, separators=(",", ":")).encode())
    payload_b64 = _b64u_encode(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    sig_b64 = _b64u_encode(_sign(signing_input))
    return f"{header_b64}.{payload_b64}.{sig_b64}", exp


def verify_admin_token(token: str) -> Optional[Dict[str, Any]]:
    """Return the claims dict if the token is valid and unexpired, else None."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header_b64, payload_b64, sig_b64 = parts
        signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
        expected = _sign(signing_input)
        actual = _b64u_decode(sig_b64)
        if not hmac.compare_digest(expected, actual):
            return None
        payload_raw = _b64u_decode(payload_b64)
        claims = json.loads(payload_raw.decode("utf-8"))
    except Exception:
        return None
    if not isinstance(claims, dict):
        return None
    if claims.get("iss") != TOKEN_ISSUER:
        return None
    if claims.get("role") != "admin":
        return None
    exp = claims.get("exp")
    if not isinstance(exp, int) or exp < int(time.time()):
        return None
    return claims


# ---------------------------------------------------------------------------
# Password hashing (PBKDF2-HMAC-SHA256, Python stdlib — no extra dependency)
# ---------------------------------------------------------------------------
def hash_password(password: str, iterations: int = _PBKDF2_ITERATIONS) -> str:
    """Return a self-describing PBKDF2 hash string for ``password``.

    Format: ``pbkdf2_sha256$<iterations>$<salt_b64>$<hash_b64>``.
    A fresh random salt is generated each call. Used by the helper script
    to produce ``ADMIN_PASSWORD_HASH``; never stores the plaintext.
    """
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"{_PBKDF2_ALGO}${iterations}${_b64u_encode(salt)}${_b64u_encode(dk)}"


def _verify_password(password: str, stored_hash: str) -> bool:
    """Constant-time verify ``password`` against a stored PBKDF2 hash.

    Returns False on any malformed/unsupported hash rather than raising,
    so a misconfigured ADMIN_PASSWORD_HASH fails safely (login denied).
    """
    try:
        algo, iter_s, salt_b64, hash_b64 = stored_hash.split("$", 3)
        if algo != _PBKDF2_ALGO:
            return False
        iterations = int(iter_s)
        salt = _b64u_decode(salt_b64)
        expected = _b64u_decode(hash_b64)
    except Exception:
        return False
    dk = hashlib.pbkdf2_hmac("sha256", (password or "").encode("utf-8"), salt, iterations)
    return hmac.compare_digest(dk, expected)


# ---------------------------------------------------------------------------
# Credential check (constant-time, env-sourced, fail-safe)
# ---------------------------------------------------------------------------
def admin_auth_configured() -> bool:
    """True only when both ADMIN_EMAIL and ADMIN_PASSWORD_HASH are present."""
    return bool(ADMIN_EMAIL) and bool(ADMIN_PASSWORD_HASH)


def verify_credentials(email: str, password: str) -> bool:
    # Fail safely when not configured — never fall back to a default.
    if not admin_auth_configured():
        log.error(
            "Admin auth misconfigured: ADMIN_EMAIL and/or ADMIN_PASSWORD_HASH "
            "is not set. Login denied. Configure these env vars to enable login."
        )
        return False
    email_ok = hmac.compare_digest((email or "").strip(), ADMIN_EMAIL)
    pw_ok = _verify_password(password or "", ADMIN_PASSWORD_HASH)
    return email_ok and pw_ok


# ---------------------------------------------------------------------------
# Per-IP login rate limiter — 5 attempts / 60s window.
# ---------------------------------------------------------------------------
_RATE_WINDOW_SECONDS = 60
_RATE_MAX_ATTEMPTS = 5
_rate_buckets: Dict[str, Deque[float]] = {}


def check_rate_limit(ip: str) -> bool:
    now = time.time()
    bucket = _rate_buckets.setdefault(ip, deque())
    while bucket and now - bucket[0] > _RATE_WINDOW_SECONDS:
        bucket.popleft()
    if len(bucket) >= _RATE_MAX_ATTEMPTS:
        return False
    bucket.append(now)
    return True


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------
def require_admin(
    request: Request,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail={"error": "missing_token"})
    token = authorization.split(" ", 1)[1].strip()
    claims = verify_admin_token(token)
    if not claims:
        raise HTTPException(status_code=401, detail={"error": "invalid_token"})
    return claims
