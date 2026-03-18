"""
PERA AI — Lightweight Answer Audit Trail

JSONL-based audit persistence for governance and debugging.
Each line captures the full question→answer lifecycle.

Privacy-aware: stores answer fingerprint (hash) instead of full text
unless AUDIT_STORE_FULL_TEXT is enabled.
"""
from __future__ import annotations

import os
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
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Write one audit entry atomically."""
    if not AUDIT_ENABLED:
        return

    AUDIT_MAX_EVIDENCE_CHARS = int(os.getenv("AUDIT_MAX_EVIDENCE_CHARS", "2000"))

    entry: Dict[str, Any] = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "ts_unix": round(time.time(), 3),
        "request_id": request_id,
        "session_id": session_id,
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

    if extra:
        entry["extra"] = extra

    try:
        path = _current_log_path()
        line = json.dumps(entry, ensure_ascii=False, default=str)
        with _write_lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        log.debug("Audit entry written: req=%s decision=%s", request_id, decision)
    except Exception as e:
        log.error("Failed to write audit entry: %s", e)
