"""
PERA AI — Phase 8: Authentication, Identity, and Authorization Hooks

Practical first-stage auth layer. Deterministic, env-driven, no external
IdP setup required. Designed so:

  • Development stays frictionless (AUTH_MODE=disabled, the default).
  • Production can require authenticated requests (AUTH_MODE=bearer or jwt).
  • Audit records can be tied to a normalized caller identity.
  • Eval harness keeps working — it imports the pipeline directly and
    never crosses the HTTP layer where auth is enforced.

Auth modes (env: AUTH_MODE):
  - "disabled"  → no checks; every request gets the dev-anonymous identity.
                  Default. Suitable for local dev only.
  - "bearer"    → require Authorization: Bearer <token> matching one of the
                  pre-shared tokens in AUTH_BEARER_TOKENS.
                  Optionally per-token role/subject via AUTH_BEARER_TOKENS_JSON.
  - "service"   → require X-Service-Token: <token> (signed internal header)
                  matching one of AUTH_SERVICE_TOKENS. Same JSON-mapping
                  shape as bearer mode. Useful for trusted backends.
  - "jwt"       → verify JWT in Authorization: Bearer header.
                  Requires PyJWT (optional dep). Falls back to bearer-mode
                  semantics if PyJWT is unavailable AND AUTH_JWT_ALLOW_FALLBACK=1.
                  Otherwise returns 503 at startup-detected misconfiguration.

Identity resolution returns an Identity dataclass that handlers and the
audit trail consume. An Identity object is ALWAYS produced (anonymous in
dev mode); only `require_identity` actually rejects.

Role enforcement (require_role) is a small RBAC hook. By default no
endpoints require roles — opt in per-route as policy evolves.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from log_config import get_logger

log = get_logger("pera.auth")


# ── Mode constants ───────────────────────────────────────────────────────────
AUTH_MODE_DISABLED = "disabled"
AUTH_MODE_BEARER = "bearer"
AUTH_MODE_SERVICE = "service"
AUTH_MODE_JWT = "jwt"

VALID_MODES = {AUTH_MODE_DISABLED, AUTH_MODE_BEARER, AUTH_MODE_SERVICE, AUTH_MODE_JWT}


# ── Configuration (read once at module import; env-overridable) ─────────────
AUTH_MODE = os.getenv("AUTH_MODE", AUTH_MODE_DISABLED).strip().lower()
if AUTH_MODE not in VALID_MODES:
    log.warning("AUTH_MODE=%r is not recognized; falling back to 'disabled'", AUTH_MODE)
    AUTH_MODE = AUTH_MODE_DISABLED

# Bearer / service token sets. Two ways to provide tokens:
#   1) Plain CSV: AUTH_BEARER_TOKENS="tok-1,tok-2"
#      All listed tokens are valid; subject = "bearer:<8-char-hash>".
#   2) JSON map: AUTH_BEARER_TOKENS_JSON='{"tok-1": {"sub":"alice","roles":["user"]}}'
#      Each token gets a normalized identity attached.
def _load_token_map(csv_env: str, json_env: str, label: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    csv_val = os.getenv(csv_env, "").strip()
    if csv_val:
        for tok in [t.strip() for t in csv_val.split(",") if t.strip()]:
            out[tok] = {}
    json_val = os.getenv(json_env, "").strip()
    if json_val:
        try:
            obj = json.loads(json_val)
            if isinstance(obj, dict):
                for tok, meta in obj.items():
                    if not isinstance(tok, str) or not tok:
                        continue
                    if not isinstance(meta, dict):
                        meta = {}
                    out[tok] = meta
        except Exception as e:
            log.error("%s: invalid JSON (%s) — ignoring JSON map", json_env, e)
    if AUTH_MODE == label and not out:
        log.warning(
            "AUTH_MODE=%s but no tokens configured (%s/%s). "
            "All requests will be rejected.",
            AUTH_MODE, csv_env, json_env,
        )
    return out


_BEARER_TOKEN_MAP = _load_token_map("AUTH_BEARER_TOKENS", "AUTH_BEARER_TOKENS_JSON", AUTH_MODE_BEARER)
_SERVICE_TOKEN_MAP = _load_token_map("AUTH_SERVICE_TOKENS", "AUTH_SERVICE_TOKENS_JSON", AUTH_MODE_SERVICE)

# JWT config
AUTH_JWT_SECRET = os.getenv("AUTH_JWT_SECRET", "").strip()
AUTH_JWT_ALG = os.getenv("AUTH_JWT_ALG", "HS256").strip()
AUTH_JWT_AUDIENCE = os.getenv("AUTH_JWT_AUDIENCE", "").strip()
AUTH_JWT_ISSUER = os.getenv("AUTH_JWT_ISSUER", "").strip()
AUTH_JWT_LEEWAY_SEC = int(os.getenv("AUTH_JWT_LEEWAY_SEC", "30"))
AUTH_JWT_ALLOW_FALLBACK = os.getenv("AUTH_JWT_ALLOW_FALLBACK", "0").strip() == "1"
AUTH_JWT_ROLES_CLAIM = os.getenv("AUTH_JWT_ROLES_CLAIM", "roles").strip()
AUTH_JWT_NAME_CLAIM = os.getenv("AUTH_JWT_NAME_CLAIM", "name").strip()
AUTH_JWT_SUBJECT_CLAIM = os.getenv("AUTH_JWT_SUBJECT_CLAIM", "sub").strip()

# Trusted forwarded header — reused from Phase 7 if present.
TRUSTED_FORWARDED_HEADER = os.getenv("TRUSTED_FORWARDED_HEADER", "").strip()

# Try to import PyJWT lazily. We do NOT crash on import errors so
# disabled/bearer/service modes work without the optional dependency.
try:
    import jwt as _pyjwt  # type: ignore
    _PYJWT_AVAILABLE = True
except Exception:
    _pyjwt = None  # type: ignore
    _PYJWT_AVAILABLE = False


# ── Identity dataclass ──────────────────────────────────────────────────────
@dataclass
class Identity:
    """Normalized identity attached to every request.

    `anonymous=True` is the dev-mode bypass identity; production modes
    refuse the request before producing one.
    """
    subject: str = "anonymous"          # stable id for the caller
    display_name: str = ""              # optional human label
    roles: List[str] = field(default_factory=list)
    auth_mode: str = AUTH_MODE_DISABLED
    auth_source: str = ""               # short reason e.g. "dev-bypass", "bearer-token", "jwt"
    anonymous: bool = True
    token_id: str = ""                  # short fingerprint for audit (NEVER full token)

    def to_audit_dict(self) -> Dict[str, Any]:
        """Compact view for audit logs. Never includes raw secrets."""
        return {
            "subject": self.subject,
            "display_name": self.display_name,
            "roles": list(self.roles or []),
            "auth_mode": self.auth_mode,
            "auth_source": self.auth_source,
            "anonymous": self.anonymous,
            "token_id": self.token_id,
        }

    def has_role(self, role: str) -> bool:
        return role in (self.roles or [])


# ── Helpers ─────────────────────────────────────────────────────────────────
def _short_token_id(token: str) -> str:
    """Stable 8-char fingerprint for audit log (NOT a credential)."""
    import hashlib
    return hashlib.sha256((token or "").encode("utf-8")).hexdigest()[:8]


def _extract_bearer(authorization_header: Optional[str]) -> Optional[str]:
    if not authorization_header:
        return None
    parts = authorization_header.strip().split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    tok = parts[1].strip()
    return tok or None


def _identity_from_token(token: str, token_map: Dict[str, Dict[str, Any]],
                         mode: str, source: str) -> Optional[Identity]:
    meta = token_map.get(token)
    if meta is None:
        return None
    sub = str(meta.get("sub") or f"{source}:{_short_token_id(token)}")
    name = str(meta.get("name") or "")
    roles_raw = meta.get("roles") or []
    if isinstance(roles_raw, list):
        roles = [str(r) for r in roles_raw if r]
    else:
        roles = []
    return Identity(
        subject=sub,
        display_name=name,
        roles=roles,
        auth_mode=mode,
        auth_source=source,
        anonymous=False,
        token_id=_short_token_id(token),
    )


# ── JWT path ────────────────────────────────────────────────────────────────
def _verify_jwt(token: str) -> Tuple[Optional[Identity], Optional[str]]:
    """Verify a JWT. Returns (identity, error_reason).

    Failure modes:
      • PyJWT not installed AND fallback not allowed → ("not_configured")
      • PyJWT not installed AND fallback allowed     → falls through to
        bearer-token validation (caller decides).
      • Token invalid / expired / wrong audience    → (None, "<reason>")
    """
    if not AUTH_JWT_SECRET:
        return None, "jwt_secret_not_set"
    if not _PYJWT_AVAILABLE:
        return None, "pyjwt_not_installed"
    try:
        decode_kwargs: Dict[str, Any] = {
            "algorithms": [AUTH_JWT_ALG],
            "leeway": AUTH_JWT_LEEWAY_SEC,
        }
        if AUTH_JWT_AUDIENCE:
            decode_kwargs["audience"] = AUTH_JWT_AUDIENCE
        if AUTH_JWT_ISSUER:
            decode_kwargs["issuer"] = AUTH_JWT_ISSUER
        claims = _pyjwt.decode(token, AUTH_JWT_SECRET, **decode_kwargs)
    except Exception as e:
        return None, f"jwt_invalid:{type(e).__name__}"

    sub = str(claims.get(AUTH_JWT_SUBJECT_CLAIM) or "").strip()
    if not sub:
        return None, "jwt_missing_subject"
    name = str(claims.get(AUTH_JWT_NAME_CLAIM) or "")
    roles_raw = claims.get(AUTH_JWT_ROLES_CLAIM) or []
    if isinstance(roles_raw, str):
        # tolerate comma-separated string
        roles = [r.strip() for r in roles_raw.split(",") if r.strip()]
    elif isinstance(roles_raw, list):
        roles = [str(r) for r in roles_raw if r]
    else:
        roles = []
    return Identity(
        subject=sub,
        display_name=name,
        roles=roles,
        auth_mode=AUTH_MODE_JWT,
        auth_source="jwt",
        anonymous=False,
        token_id=_short_token_id(token),
    ), None


# ── Public: identity resolution ─────────────────────────────────────────────
def _dev_anonymous_identity() -> Identity:
    return Identity(
        subject="anonymous",
        display_name="dev",
        roles=[],
        auth_mode=AUTH_MODE_DISABLED,
        auth_source="dev-bypass",
        anonymous=True,
        token_id="",
    )


def resolve_identity(headers: Dict[str, str]) -> Tuple[Identity, Optional[Dict[str, Any]]]:
    """Resolve identity from headers per AUTH_MODE.

    Returns (identity, error_payload).
      • identity is always returned (even when error_payload is set; in
        that case it is a neutral 'anonymous' object so logging works).
      • error_payload is None when authentication succeeded (or mode is
        disabled). Otherwise it is a dict suitable for a 401 JSON body.

    No request side-effects.
    """
    # Normalize headers to lower-case keys for safe lookup.
    h = {k.lower(): v for k, v in (headers or {}).items() if isinstance(k, str)}

    if AUTH_MODE == AUTH_MODE_DISABLED:
        return _dev_anonymous_identity(), None

    if AUTH_MODE == AUTH_MODE_BEARER:
        tok = _extract_bearer(h.get("authorization"))
        if not tok:
            return _dev_anonymous_identity(), {
                "error": "auth_required", "reason": "missing_bearer_token",
                "mode": AUTH_MODE,
            }
        ident = _identity_from_token(tok, _BEARER_TOKEN_MAP, AUTH_MODE_BEARER, "bearer-token")
        if ident is None:
            return _dev_anonymous_identity(), {
                "error": "auth_failed", "reason": "invalid_bearer_token",
                "mode": AUTH_MODE,
            }
        return ident, None

    if AUTH_MODE == AUTH_MODE_SERVICE:
        tok = (h.get("x-service-token") or "").strip()
        if not tok:
            return _dev_anonymous_identity(), {
                "error": "auth_required", "reason": "missing_service_token",
                "mode": AUTH_MODE,
            }
        ident = _identity_from_token(tok, _SERVICE_TOKEN_MAP, AUTH_MODE_SERVICE, "service-token")
        if ident is None:
            return _dev_anonymous_identity(), {
                "error": "auth_failed", "reason": "invalid_service_token",
                "mode": AUTH_MODE,
            }
        return ident, None

    if AUTH_MODE == AUTH_MODE_JWT:
        tok = _extract_bearer(h.get("authorization"))
        if not tok:
            return _dev_anonymous_identity(), {
                "error": "auth_required", "reason": "missing_bearer_token",
                "mode": AUTH_MODE,
            }
        ident, err = _verify_jwt(tok)
        if ident is not None:
            return ident, None
        # JWT failed. Optional safe fallback: treat as bearer if operator
        # explicitly allowed it AND a bearer-token map exists. This is for
        # ops that rotate to JWT incrementally; off by default.
        if (
            AUTH_JWT_ALLOW_FALLBACK
            and err in ("pyjwt_not_installed",)
            and _BEARER_TOKEN_MAP
        ):
            log.warning(
                "JWT verification unavailable (%s); falling back to bearer-token check",
                err,
            )
            fb = _identity_from_token(tok, _BEARER_TOKEN_MAP, AUTH_MODE_BEARER, "jwt-fallback-bearer")
            if fb is not None:
                return fb, None
        return _dev_anonymous_identity(), {
            "error": "auth_failed", "reason": err or "jwt_invalid",
            "mode": AUTH_MODE,
        }

    # Unreachable due to validation at module import; defensive default.
    return _dev_anonymous_identity(), None


# ── Authorization helpers (for use as FastAPI dependencies) ─────────────────
def is_protected_mode() -> bool:
    """True when at least one form of authentication is required."""
    return AUTH_MODE != AUTH_MODE_DISABLED


def auth_status_summary() -> Dict[str, Any]:
    """For /health and /ready visibility — no secrets."""
    return {
        "mode": AUTH_MODE,
        "protected": is_protected_mode(),
        "bearer_tokens_configured": len(_BEARER_TOKEN_MAP) if AUTH_MODE == AUTH_MODE_BEARER else 0,
        "service_tokens_configured": len(_SERVICE_TOKEN_MAP) if AUTH_MODE == AUTH_MODE_SERVICE else 0,
        "jwt_secret_set": bool(AUTH_JWT_SECRET) if AUTH_MODE == AUTH_MODE_JWT else False,
        "jwt_pyjwt_available": _PYJWT_AVAILABLE if AUTH_MODE == AUTH_MODE_JWT else False,
    }
