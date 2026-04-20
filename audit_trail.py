"""
PERA AI — Lightweight Answer Audit Trail

JSONL-based audit persistence for governance and debugging.
Each line captures the full question→answer lifecycle.

Privacy-aware: stores answer fingerprint (hash) instead of full text
unless AUDIT_STORE_FULL_TEXT is enabled.

Phase 7 hardening (additive, backward-compatible):
  • Per-entry HMAC signature (`sig`) when AUDIT_HMAC_SECRET is set.
    Detects offline tampering without external infra.
  • Hash-chain (`prev_hash` + `entry_hash`) so a deleted line is
    detectable at the next entry. Maintained per-day file.
  • `event_type` field for event-typed querying ("answer", "refuse",
    "smalltalk", "error").
  • Retention metadata (`retention_days`) per entry — informational,
    consumed by an external retention job.
  • Schema version bumped to 2; old lines read as version 1.

The hash chain uses SHA-256 over the canonical JSON of the entry
MINUS the volatile fields (`sig`, `entry_hash`, `prev_hash`). The
HMAC, when present, signs the same canonical bytes plus the
prev_hash for non-repudiation across the chain.
"""
from __future__ import annotations

import os
import hmac
import json
import time
import hashlib
import threading
from typing import Optional, Dict, Any, List

from log_config import get_logger

log = get_logger("pera.audit")

# ── Configuration ─────────────────────────────────────────────
AUDIT_DIR = os.getenv("AUDIT_DIR", "audit_logs")
AUDIT_ENABLED = os.getenv("AUDIT_ENABLED", "1").strip() != "0"
AUDIT_STORE_FULL_TEXT = os.getenv("AUDIT_STORE_FULL_TEXT", "0").strip() != "0"
AUDIT_MAX_ANSWER_CHARS = int(os.getenv("AUDIT_MAX_ANSWER_CHARS", "500"))

# Phase 7 — integrity / retention controls (all optional / safe defaults).
AUDIT_SCHEMA_VERSION = 2
AUDIT_HMAC_SECRET = os.getenv("AUDIT_HMAC_SECRET", "").encode("utf-8")
AUDIT_RETENTION_DAYS = int(os.getenv("AUDIT_RETENTION_DAYS", "365"))
# When false, hash-chain is still computed but prev_hash is read
# fresh on each write (slower under burst). When true (default),
# the in-memory tail-hash is reused so chain construction is O(1).
AUDIT_CHAIN_IN_MEMORY = os.getenv("AUDIT_CHAIN_IN_MEMORY", "1").strip() != "0"

# In-memory tail-hash per file path. Keyed by absolute path so the
# day-rollover correctly re-bootstraps from disk for the new file.
_chain_state: Dict[str, str] = {}

GENESIS_HASH = "0" * 64  # used as prev_hash for the first entry of a file


def _ensure_dir() -> str:
    os.makedirs(AUDIT_DIR, exist_ok=True)
    return AUDIT_DIR


def _answer_fingerprint(text: str) -> str:
    """SHA256 hash of the answer for privacy-bounded storage."""
    return hashlib.sha256((text or "").encode()).hexdigest()[:16]


def _current_log_path() -> str:
    """One JSONL file per day: audit_logs/audit_2026-03-12.jsonl"""
    day = time.strftime("%Y-%m-%d")
    return os.path.join(_ensure_dir(), f"audit_{day}.jsonl")


# ── Thread-safe writer ────────────────────────────────────────
_write_lock = threading.Lock()


# ── Phase 7: integrity helpers ────────────────────────────────

def _canonical_payload_bytes(entry: Dict[str, Any]) -> bytes:
    """Stable, sorted-key JSON encoding used as input to hashing.

    Volatile fields produced by signing (`entry_hash`, `prev_hash`,
    `sig`) are excluded so the hash covers only the entry content.
    """
    excluded = {"entry_hash", "prev_hash", "sig"}
    payload = {k: v for k, v in entry.items() if k not in excluded}
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _read_last_entry_hash(path: str) -> str:
    """Recover the last entry_hash from a partially-written file.

    Used when the in-memory chain state is missing (process restart or
    AUDIT_CHAIN_IN_MEMORY=0). Returns GENESIS_HASH if the file does
    not exist or contains no parseable entries.
    """
    if not os.path.exists(path):
        return GENESIS_HASH
    try:
        # Tail the file (cheap for small audit logs; if these grow huge
        # we should switch to an index sidecar).
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            chunk = 8192
            tail = b""
            while size > 0 and tail.count(b"\n") < 2:
                read = min(chunk, size)
                size -= read
                f.seek(size)
                tail = f.read(read) + tail
                if size == 0:
                    break
        last_line = tail.splitlines()[-1].decode("utf-8", errors="replace") if tail else ""
        if not last_line.strip():
            return GENESIS_HASH
        try:
            obj = json.loads(last_line)
            h = obj.get("entry_hash")
            if isinstance(h, str) and len(h) == 64:
                return h
        except json.JSONDecodeError:
            pass
    except Exception as e:
        log.warning("audit chain: could not recover tail hash from %s: %s", path, e)
    return GENESIS_HASH


def _compute_chain_and_signature(
    entry: Dict[str, Any], prev_hash: str
) -> Dict[str, str]:
    """Compute prev_hash, entry_hash, and (optional) HMAC signature.

    The entry_hash is SHA-256 of `prev_hash || canonical_payload`.
    The HMAC, when AUDIT_HMAC_SECRET is set, signs the same input.
    """
    payload = _canonical_payload_bytes(entry)
    h = hashlib.sha256()
    h.update(prev_hash.encode("ascii"))
    h.update(b"\n")
    h.update(payload)
    entry_hash = h.hexdigest()

    out = {"prev_hash": prev_hash, "entry_hash": entry_hash}
    if AUDIT_HMAC_SECRET:
        sig = hmac.new(
            AUDIT_HMAC_SECRET,
            prev_hash.encode("ascii") + b"\n" + payload,
            hashlib.sha256,
        ).hexdigest()
        out["sig"] = sig
    return out


def verify_audit_file(path: str) -> Dict[str, Any]:
    """Standalone integrity verifier — useful for ops/forensics.

    Walks the file line by line, recomputes the chain, and checks the
    HMAC if AUDIT_HMAC_SECRET is set. Returns:
      {"ok": bool, "lines": int, "first_break_line": int|None, "reason": str}

    This is not called from the request path; it is a tool for
    operators (`python -c "from audit_trail import verify_audit_file; print(verify_audit_file(path))"`).
    """
    if not os.path.exists(path):
        return {"ok": False, "lines": 0, "first_break_line": None, "reason": "missing"}
    prev = GENESIS_HASH
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for n, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                return {"ok": False, "lines": n, "first_break_line": n,
                        "reason": f"json_decode:{e}"}
            # Skip schema-v1 entries gracefully (no chain).
            if obj.get("schema_version") != AUDIT_SCHEMA_VERSION:
                # Reset chain from this point — can't verify legacy lines.
                prev = obj.get("entry_hash") or prev
                continue
            if obj.get("prev_hash") != prev:
                return {"ok": False, "lines": n, "first_break_line": n,
                        "reason": "prev_hash_mismatch"}
            payload = _canonical_payload_bytes(obj)
            h = hashlib.sha256()
            h.update(prev.encode("ascii"))
            h.update(b"\n")
            h.update(payload)
            expected = h.hexdigest()
            if obj.get("entry_hash") != expected:
                return {"ok": False, "lines": n, "first_break_line": n,
                        "reason": "entry_hash_mismatch"}
            if AUDIT_HMAC_SECRET and obj.get("sig"):
                expected_sig = hmac.new(
                    AUDIT_HMAC_SECRET,
                    prev.encode("ascii") + b"\n" + payload,
                    hashlib.sha256,
                ).hexdigest()
                if not hmac.compare_digest(obj["sig"], expected_sig):
                    return {"ok": False, "lines": n, "first_break_line": n,
                            "reason": "hmac_mismatch"}
            prev = obj["entry_hash"]
    return {"ok": True, "lines": n, "first_break_line": None, "reason": "ok"}


def log_audit_entry(
    *,
    request_id: str = "",
    session_id: str = "",
    question: str = "",
    normalized_query: str = "",
    decision: str = "",
    answer_text: str = "",
    evidence_ids: Optional[List[str]] = None,
    doc_names: Optional[List[str]] = None,
    references_count: int = 0,
    grounding_score: Optional[float] = None,
    grounding_confidence: str = "",
    grounding_details: Optional[Dict[str, Any]] = None,
    is_smalltalk: bool = False,
    is_error: bool = False,
    subject_entity: str = "",
    evidence_text_preview: str = "",
    rewrite_diff: str = "",
    reranker_scores: Optional[List[Dict[str, Any]]] = None,
    support_state: str = "",
    intent: Optional[Dict[str, Any]] = None,
    client_ip: str = "",
    event_type: str = "",
    identity: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Write one audit entry atomically.

    Phase 7 additions:
      • event_type: "answer" | "refuse" | "smalltalk" | "error"
        (auto-derived from decision/flags when not supplied).
      • client_ip: optional, useful for forensics + abuse triage.
      • Each entry carries schema_version, retention_days, prev_hash,
        entry_hash, and (when AUDIT_HMAC_SECRET is set) sig.
    """
    if not AUDIT_ENABLED:
        return

    AUDIT_MAX_EVIDENCE_CHARS = int(os.getenv("AUDIT_MAX_EVIDENCE_CHARS", "2000"))

    # Auto-derive event_type if not provided.
    if not event_type:
        if is_error:
            event_type = "error"
        elif is_smalltalk:
            event_type = "smalltalk"
        elif (decision or "").lower() == "refuse":
            event_type = "refuse"
        else:
            event_type = "answer"

    entry: Dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "event_type": event_type,
        "retention_days": AUDIT_RETENTION_DAYS,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "ts_unix": round(time.time(), 3),
        "request_id": request_id,
        "session_id": session_id,
        "client_ip": client_ip[:64] if client_ip else "",
        "question": question[:500],
        "normalized_query": normalized_query[:500] if normalized_query else "",
        "decision": decision,
        "references_count": references_count,
        "evidence_ids": (evidence_ids or [])[:10],
        "doc_names": (doc_names or [])[:5],
        "subject_entity": subject_entity,
        "support_state": support_state,
        "is_smalltalk": is_smalltalk,
        "is_error": is_error,
    }

    # Phase 8 — caller identity. Always present (anonymous in dev mode).
    # Field shape mirrors auth.Identity.to_audit_dict():
    #   subject, display_name, roles, auth_mode, auth_source, anonymous, token_id
    # The token_id is a short fingerprint, never the raw token.
    if identity:
        # Copy and clip to keep audit lines compact.
        roles_in = identity.get("roles") or []
        if isinstance(roles_in, list):
            roles = [str(r)[:64] for r in roles_in[:8]]
        else:
            roles = []
        entry["identity"] = {
            "subject": str(identity.get("subject") or "")[:128],
            "display_name": str(identity.get("display_name") or "")[:128],
            "roles": roles,
            "auth_mode": str(identity.get("auth_mode") or "")[:32],
            "auth_source": str(identity.get("auth_source") or "")[:32],
            "anonymous": bool(identity.get("anonymous", True)),
            "token_id": str(identity.get("token_id") or "")[:16],
        }
    else:
        # Maintain a stable shape so downstream consumers don't need
        # null checks. Marked anonymous so triage queries (e.g. "all
        # anonymous traffic in prod") work uniformly.
        entry["identity"] = {
            "subject": "anonymous",
            "display_name": "",
            "roles": [],
            "auth_mode": "",
            "auth_source": "",
            "anonymous": True,
            "token_id": "",
        }

    # Answer: either full text (bounded) or fingerprint
    if AUDIT_STORE_FULL_TEXT:
        entry["answer_preview"] = (answer_text or "")[:AUDIT_MAX_ANSWER_CHARS]
        # Evidence text preview — the actual chunks sent to LLM
        if evidence_text_preview:
            entry["evidence_text_preview"] = evidence_text_preview[:AUDIT_MAX_EVIDENCE_CHARS]
    else:
        entry["answer_hash"] = _answer_fingerprint(answer_text)
        entry["answer_len"] = len(answer_text or "")

    # Rewrite diff — shows original → rewritten query
    if rewrite_diff:
        entry["rewrite_diff"] = rewrite_diff[:500]

    # Reranker scores — top hits with blend scores
    if reranker_scores:
        entry["reranker_scores"] = reranker_scores[:10]

    # Grounding
    if grounding_score is not None:
        entry["grounding_score"] = round(grounding_score, 3)
        entry["grounding_confidence"] = grounding_confidence
    if grounding_details:
        entry["grounding_details"] = grounding_details

    # Phase 5 — intent routing trace
    if intent:
        entry["intent"] = {
            "primary": intent.get("primary", ""),
            "is_multi_part": bool(intent.get("is_multi_part", False)),
            "is_vague": bool(intent.get("is_vague", False)),
            # Cap sub-queries to keep audit lines compact.
            "sub_queries": [str(s)[:120] for s in (intent.get("sub_queries") or [])][:6],
        }

    if extra:
        entry["extra"] = extra

    try:
        path = _current_log_path()
        with _write_lock:
            # Phase 7 — chain construction. The lock serializes both
            # state lookup and file append so the chain stays consistent
            # under concurrent writers.
            if AUDIT_CHAIN_IN_MEMORY and path in _chain_state:
                prev_hash = _chain_state[path]
            else:
                prev_hash = _read_last_entry_hash(path)
                _chain_state[path] = prev_hash

            chain = _compute_chain_and_signature(entry, prev_hash)
            entry.update(chain)

            line = json.dumps(entry, ensure_ascii=False, default=str)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

            _chain_state[path] = chain["entry_hash"]

        log.debug(
            "Audit entry written: req=%s event=%s decision=%s",
            request_id, event_type, decision,
        )
    except Exception as e:
        log.error("Failed to write audit entry: %s", e)
