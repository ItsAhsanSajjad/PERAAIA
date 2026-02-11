import os
import json
import time
import sqlite3
import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

# Configure logging
logger = logging.getLogger(__name__)

def _now_ts() -> int:
    return int(time.time())

class MemoryStore(ABC):
    """Abstract base class for conversation memory storage."""
    
    @abstractmethod
    def load(self, user_id: str, conversation_id: str) -> Dict[str, Any]:
        """
        Load conversation history and state.
        Returns:
            {
                "messages": [... last 10 ...],
                "pinned_evidence": [... last 5 traces ...],
                "state": {...}
            }
        """
        pass

    @abstractmethod
    def append_user(self, user_id: str, conversation_id: str, content: str):
        """Add a user message."""
        pass

    @abstractmethod
    def append_assistant(self, user_id: str, conversation_id: str, content: str, evidence_trace: Optional[Dict[str, Any]] = None):
        """Add an assistant message and optional evidence trace."""
        pass
        
    @abstractmethod
    def update_state(self, user_id: str, conversation_id: str, updates: Dict[str, Any]):
        """Update the simple state dictionary (merge)."""
        pass

    @abstractmethod
    def clear(self, user_id: str, conversation_id: str):
        """Clear all memory for this conversation."""
        pass


class SQLiteMemoryStore(MemoryStore):
    """
    SQLite-backed memory store.
    Filesystem persistence without external services.
    """
    def __init__(self, db_path: str = "assets/memory.db"):
        self.db_path = db_path.replace("\\", "/")
        if os.path.dirname(self.db_path):
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path, timeout=30) as conn:
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        conversation_id TEXT,
                        role TEXT,
                        content TEXT,
                        created_at INTEGER
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_evidence (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        conversation_id TEXT,
                        evidence_json TEXT,
                        created_at INTEGER
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_state (
                        user_id TEXT,
                        conversation_id TEXT,
                        state_json TEXT,
                        updated_at INTEGER,
                        PRIMARY KEY (user_id, conversation_id)
                    )
                """)
                # Indexes for speed
                conn.execute("CREATE INDEX IF NOT EXISTS idx_msgs_conv ON conversation_messages(user_id, conversation_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_ev_conv ON conversation_evidence(user_id, conversation_id)")
            except Exception as e:
                logger.error(f"DB Init Error: {e}")
                raise
            # sqlite3 context manager handles commit/rollback but NOT close.
            # However, the 'with' block variable 'conn' will be closed explicitly? 
            # No, 'with sqlite3.connect' does NOT close. 
            # But the 'conn' variable scope ends. GC might close it. 
            # Adopting the user's rule: "Always close cursors and connections."
        conn.close()

    def load(self, user_id: str, conversation_id: str) -> Dict[str, Any]:
        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            conn.row_factory = sqlite3.Row # Optional, but good for robust access
            
            # Load messages (Limit 10, strictly ordered by time)
            cur = conn.execute("""
                SELECT role, content, created_at FROM conversation_messages
                WHERE user_id = ? AND conversation_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT 10
            """, (user_id, conversation_id))
            rows = cur.fetchall()
            messages = [{"role": r[0], "content": r[1], "timestamp": r[2]} for r in reversed(rows)]
            cur.close()

            # Load evidence (Limit 5)
            cur = conn.execute("""
                SELECT evidence_json FROM conversation_evidence
                WHERE user_id = ? AND conversation_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT 5
            """, (user_id, conversation_id))
            ev_rows = cur.fetchall()
            evidence = []
            for r in reversed(ev_rows):
                try:
                    evidence.append(json.loads(r[0]))
                except:
                    pass
            cur.close()

            # Load state
            cur = conn.execute("""
                SELECT state_json FROM conversation_state
                WHERE user_id = ? AND conversation_id = ?
            """, (user_id, conversation_id))
            row = cur.fetchone()
            state = {}
            if row:
                try:
                    state = json.loads(row[0])
                except:
                    pass
            cur.close()

            return {
                "messages": messages,
                "pinned_evidence": evidence,
                "state": state
            }
        finally:
            conn.close()

    def _trim_messages(self, conn, user_id: str, conversation_id: str):
        # Keep only last 10
        conn.execute("""
            DELETE FROM conversation_messages
            WHERE id NOT IN (
                SELECT id FROM conversation_messages
                WHERE user_id = ? AND conversation_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT 10
            ) AND user_id = ? AND conversation_id = ?
        """, (user_id, conversation_id, user_id, conversation_id))

    def _trim_evidence(self, conn, user_id: str, conversation_id: str):
        # Keep only last 5
        conn.execute("""
            DELETE FROM conversation_evidence
            WHERE id NOT IN (
                SELECT id FROM conversation_evidence
                WHERE user_id = ? AND conversation_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT 5
            ) AND user_id = ? AND conversation_id = ?
        """, (user_id, conversation_id, user_id, conversation_id))

    def append_user(self, user_id: str, conversation_id: str, content: str):
        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            with conn:
                conn.execute("""
                    INSERT INTO conversation_messages (user_id, conversation_id, role, content, created_at)
                    VALUES (?, ?, 'user', ?, ?)
                """, (user_id, conversation_id, content, _now_ts()))
                self._trim_messages(conn, user_id, conversation_id)
        finally:
            conn.close()

    def append_assistant(self, user_id: str, conversation_id: str, content: str, evidence_trace: Optional[Dict[str, Any]] = None):
        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            with conn:
                ts = _now_ts()
                conn.execute("""
                    INSERT INTO conversation_messages (user_id, conversation_id, role, content, created_at)
                    VALUES (?, ?, 'assistant', ?, ?)
                """, (user_id, conversation_id, content, ts))
                self._trim_messages(conn, user_id, conversation_id)

                if evidence_trace:
                    conn.execute("""
                        INSERT INTO conversation_evidence (user_id, conversation_id, evidence_json, created_at)
                        VALUES (?, ?, ?, ?)
                    """, (user_id, conversation_id, json.dumps(evidence_trace), ts))
                    self._trim_evidence(conn, user_id, conversation_id)
        finally:
            conn.close()

    def update_state(self, user_id: str, conversation_id: str, updates: Dict[str, Any]):
        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            with conn:
                # Read existing
                cur = conn.execute("SELECT state_json FROM conversation_state WHERE user_id=? AND conversation_id=?", (user_id, conversation_id))
                row = cur.fetchone()
                cur.close()

                current_state = {}
                if row:
                    try:
                        current_state = json.loads(row[0])
                    except:
                        pass
                
                # Merge
                current_state.update(updates)
                
                # Write back
                conn.execute("""
                    INSERT OR REPLACE INTO conversation_state (user_id, conversation_id, state_json, updated_at)
                    VALUES (?, ?, ?, ?)
                """, (user_id, conversation_id, json.dumps(current_state), _now_ts()))
        finally:
            conn.close()

    def clear(self, user_id: str, conversation_id: str):
        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            with conn:
                conn.execute("DELETE FROM conversation_messages WHERE user_id=? AND conversation_id=?", (user_id, conversation_id))
                conn.execute("DELETE FROM conversation_evidence WHERE user_id=? AND conversation_id=?", (user_id, conversation_id))
                conn.execute("DELETE FROM conversation_state WHERE user_id=? AND conversation_id=?", (user_id, conversation_id))
        finally:
            conn.close()


class RedisMemoryStore(MemoryStore):
    """
    Redis-backed memory store.
    Uses Lists for messages/evidence, String for state.
    TTL enforcement is built-in.
    """
    def __init__(self, redis_url: str):
        try:
            import redis
            self.r = redis.from_url(redis_url, decode_responses=True)
            self.ttl = int(os.getenv("MEMORY_TTL_SECONDS", "86400"))
        except ImportError:
            raise ImportError("redis-py not installed. Install with `pip install redis`.")

    def _k_msg(self, uid, cid): return f"pera:mem:msg:{uid}:{cid}"
    def _k_ev(self, uid, cid): return f"pera:mem:ev:{uid}:{cid}"
    def _k_st(self, uid, cid): return f"pera:mem:st:{uid}:{cid}"

    def load(self, user_id: str, conversation_id: str) -> Dict[str, Any]:
        # Messages: lrange 0 -1
        raw_msgs = self.r.lrange(self._k_msg(user_id, conversation_id), 0, -1)
        messages = [json.loads(m) for m in raw_msgs]
        
        # Evidence: lrange 0 -1
        raw_ev = self.r.lrange(self._k_ev(user_id, conversation_id), 0, -1)
        evidence = [json.loads(e) for e in raw_ev]
        
        # State: get
        raw_st = self.r.get(self._k_st(user_id, conversation_id))
        state = json.loads(raw_st) if raw_st else {}
        
        return {
            "messages": messages,
            "pinned_evidence": evidence,
            "state": state
        }

    def append_user(self, user_id: str, conversation_id: str, content: str):
        msg = {"role": "user", "content": content, "timestamp": _now_ts()}
        k = self._k_msg(user_id, conversation_id)
        self.r.rpush(k, json.dumps(msg))
        self.r.ltrim(k, -10, -1) # Keep last 10
        self.r.expire(k, self.ttl)

    def append_assistant(self, user_id: str, conversation_id: str, content: str, evidence_trace: Optional[Dict[str, Any]] = None):
        # Msg
        msg = {"role": "assistant", "content": content, "timestamp": _now_ts()}
        k_msg = self._k_msg(user_id, conversation_id)
        self.r.rpush(k_msg, json.dumps(msg))
        self.r.ltrim(k_msg, -10, -1)
        self.r.expire(k_msg, self.ttl)
        
        # Evidence
        if evidence_trace:
            k_ev = self._k_ev(user_id, conversation_id)
            self.r.rpush(k_ev, json.dumps(evidence_trace))
            self.r.ltrim(k_ev, -5, -1) # Keep last 5
            self.r.expire(k_ev, self.ttl)

    def update_state(self, user_id: str, conversation_id: str, updates: Dict[str, Any]):
        k_st = self._k_st(user_id, conversation_id)
        raw = self.r.get(k_st)
        current = json.loads(raw) if raw else {}
        current.update(updates)
        self.r.setex(k_st, self.ttl, json.dumps(current))

    def clear(self, user_id: str, conversation_id: str):
        self.r.delete(
            self._k_msg(user_id, conversation_id),
            self._k_ev(user_id, conversation_id),
            self._k_st(user_id, conversation_id)
        )


def get_memory_store() -> MemoryStore:
    """Factory to get the configured memory store."""
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        try:
            return RedisMemoryStore(redis_url)
        except Exception as e:
            print(f"[MemoryStore] REDIS_URL found but failed to init: {e}. Falling back to SQLite.")
    
    db_path = os.getenv("MEMORY_DB_PATH", "assets/memory.db")
    return SQLiteMemoryStore(db_path)
