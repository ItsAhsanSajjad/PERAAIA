"""
PERA AI — Bounded In-Memory Session Store

Provides server-side session state for follow-up context integrity.
Designed for future swap to Redis/DB without rewriting consumers.

Each session stores:
  - bounded recent turn history (question + answer + decision)
  - last subject/entity/role anchors
  - last evidence references
  - last normalized query
"""
from __future__ import annotations

import time
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from log_config import get_logger

log = get_logger("pera.session")

# ── Configuration ─────────────────────────────────────────────
MAX_SESSIONS = 2000             # evict oldest beyond this
MAX_TURNS_PER_SESSION = 10      # bounded history per session
SESSION_TTL_SECONDS = 3600      # 1 hour idle expiry


# ── Session Turn ──────────────────────────────────────────────
@dataclass
class SessionTurn:
    """One turn in the conversation."""
    question: str
    normalized_query: str = ""
    answer_preview: str = ""    # first 300 chars (privacy-bounded)
    decision: str = ""          # answer / refuse / error
    evidence_ids: List[str] = field(default_factory=list)
    doc_names: List[str] = field(default_factory=list)
    subject_entity: str = ""    # extracted subject/role/entity
    timestamp: float = field(default_factory=time.time)


# ── Session State ─────────────────────────────────────────────
@dataclass
class SessionState:
    """Server-side state for one session."""
    session_id: str
    turns: List[SessionTurn] = field(default_factory=list)
    last_subject: str = ""      # most recent anchored entity/role
    last_doc_names: List[str] = field(default_factory=list)
    last_evidence_ids: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    def add_turn(self, turn: SessionTurn) -> None:
        self.turns.append(turn)
        if len(self.turns) > MAX_TURNS_PER_SESSION:
            self.turns = self.turns[-MAX_TURNS_PER_SESSION:]
        self.last_active = time.time()

        # Update anchors
        if turn.subject_entity:
            self.last_subject = turn.subject_entity
        if turn.doc_names:
            self.last_doc_names = turn.doc_names[:5]
        if turn.evidence_ids:
            self.last_evidence_ids = turn.evidence_ids[:10]

    @property
    def last_question(self) -> str:
        return self.turns[-1].question if self.turns else ""

    @property
    def last_answer(self) -> str:
        return self.turns[-1].answer_preview if self.turns else ""

    @property
    def last_normalized_query(self) -> str:
        return self.turns[-1].normalized_query if self.turns else ""

    def is_expired(self) -> bool:
        return (time.time() - self.last_active) > SESSION_TTL_SECONDS


# ── Session Store ─────────────────────────────────────────────
class SessionStore:
    """
    Thread-safe bounded in-memory session store.
    
    Interface designed for future swap to Redis/DB:
        get(session_id) -> SessionState | None
        put(session_id, state) -> None
        get_or_create(session_id) -> SessionState
    """

    def __init__(self, max_sessions: int = MAX_SESSIONS):
        self._lock = threading.Lock()
        self._store: OrderedDict[str, SessionState] = OrderedDict()
        self._max = max_sessions

    def get(self, session_id: str) -> Optional[SessionState]:
        with self._lock:
            state = self._store.get(session_id)
            if state and state.is_expired():
                del self._store[session_id]
                log.debug("Session expired: %s", session_id)
                return None
            if state:
                # Move to end (LRU touch)
                self._store.move_to_end(session_id)
            return state

    def get_or_create(self, session_id: Optional[str] = None) -> SessionState:
        if not session_id:
            session_id = uuid.uuid4().hex[:16]

        with self._lock:
            existing = self._store.get(session_id)
            if existing and not existing.is_expired():
                self._store.move_to_end(session_id)
                return existing

            # Create new
            state = SessionState(session_id=session_id)
            self._store[session_id] = state
            self._store.move_to_end(session_id)
            log.debug("New session created: %s", session_id)

            # Evict if over capacity
            self._evict()
            return state

    def put(self, session_id: str, state: SessionState) -> None:
        with self._lock:
            self._store[session_id] = state
            self._store.move_to_end(session_id)
            self._evict()

    def _evict(self) -> None:
        """Evict oldest sessions if over capacity."""
        while len(self._store) > self._max:
            evicted_id, _ = self._store.popitem(last=False)
            log.debug("Session evicted (capacity): %s", evicted_id)


# ── Singleton ─────────────────────────────────────────────────
_store = SessionStore()


def get_session_store() -> SessionStore:
    return _store
