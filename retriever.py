"""
PERA AI Retriever (Brain 2.0)
Simplified, robust semantic search without manual heuristic filtering.
(Updated: FAISS IDMap-safe + query normalization + safer keyword + dedupe)
"""
from __future__ import annotations

import os
import json
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from index_store import load_index_and_chunks, embed_texts

import numpy as np
import re as _re
from collections import defaultdict

load_dotenv()

# Configuration
TOP_K = int(os.getenv("RETRIEVER_TOP_K", "30"))
SIM_THRESHOLD = float(os.getenv("RETRIEVER_SIM_THRESHOLD", "0.14"))
LLM_REWRITE_MODEL = os.getenv("RETRIEVER_LLM_QUERY_REWRITE_MODEL", "gpt-4o-mini")

# Abbreviation -> full expansion (for embedding search quality)
_ABBREV_MAP_RAW = {
    "cto": "Chief Technology Officer",
    "dg": "Director General",
    "mgr": "Manager",
    "hr": "Human Resources",
    "it": "Information Technology",
    "eo": "Enforcement Officer",
    "io": "Investigation Officer",
    "sso": "System Support Officer",
    "tor": "Terms of Reference",
    "jd": "Job Description",
    "sr": "Service Rules",
    "Schedule-I": "Organizational Structure",
    "Schedule-II": "Appointment & Conditions of Service",
    "Schedule-III": "Special Pay Package PERA (SPPP)",
    "Schedule-IV": "Rules / Regulations Adopted by the Authority",
    "Schedule-V": "Transfer and Posting",
    "Schedule-VI": "Special Allowance and Benefits",
    "sppp": "Special Pay Package PERA",
    "faqs": "Frequently Asked Questions",
}

def _norm_key(s: str) -> str:
    # Keep alphanumerics; makes "Schedule-I" -> "schedulei"
    return _re.sub(r"[^a-z0-9]+", "", (s or "").lower())

# Normalize abbrev keys once so Schedule-I etc. work
_ABBREV_MAP = {_norm_key(k): v for k, v in _ABBREV_MAP_RAW.items()}

# Smart Context Expansion Keywords
# If query contains these, we fetch adjacent pages (±RADIUS) to capture tables/schedules
_EXPANSION_KEYWORDS = {
    "salary", "pay", "allowance", "benefit", "scale", "sppp", "grade", "compensation",
    "detail", "full", "sab kuch", "batao", "explain", "structure",
    # Roman Urdu / misspellings
    "salay", "tankhwah", "tankha", "kitni", "payscale", "pay scale",
    "maaash", "maash",
}
_EXPANSION_RADIUS = 3  # Fetch ±3 pages for salary/detail queries


def _normalize_vec(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v) + 1e-12)
    return v / n


def _expand_abbreviations(query: str) -> str:
    """Expand known abbreviations in-place for better embedding matches."""
    words = (query or "").split()
    expanded = []
    for w in words:
        key = _norm_key(w)
        if key in _ABBREV_MAP:
            expanded.append(_ABBREV_MAP[key])
        else:
            expanded.append(w)
    return " ".join(expanded)


def _build_id_map(rows: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """
    FAISS IndexIDMap2 returns vector IDs (the 'id' field in chunks.jsonl).
    Build id -> row map for active rows only.
    """
    m: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        if not r.get("active", True):
            continue
        rid = r.get("id")
        if rid is None:
            continue
        try:
            m[int(rid)] = r
        except Exception:
            continue
    return m


def _get_page_map_by_id(id_map: Dict[int, Dict[str, Any]]) -> Dict[Tuple[str, int], List[int]]:
    """
    Build map of (doc_name, page) -> list of chunk IDs (NOT list indices).
    """
    m: Dict[Tuple[str, int], List[int]] = defaultdict(list)
    for cid, r in id_map.items():
        doc = r.get("doc_name", "Unknown")
        page = r.get("loc_start")
        if isinstance(page, int):
            m[(doc, page)].append(cid)
    return m


_client = None

def get_client():
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        )
    return _client


# =============================================================================
# Active index pointer
# =============================================================================
class ActiveIndexPointer:
    def __init__(self, pointer_path: str = "assets/indexes/ACTIVE.json"):
        self.pointer_path = (pointer_path or "").replace("\\", "/")

    def read_raw(self) -> Optional[str]:
        if not self.pointer_path or not os.path.exists(self.pointer_path):
            return None
        try:
            with open(self.pointer_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            p = (data.get("active_index_dir") or "").strip()
            return p.replace("\\", "/") if p else None
        except Exception:
            return None

_ACTIVE_POINTER = ActiveIndexPointer(os.getenv("INDEX_POINTER_PATH", "assets/indexes/ACTIVE.json"))

def _resolve_index_dir(index_dir: Optional[str]) -> str:
    if index_dir and os.path.isdir(index_dir):
        return index_dir

    ptr = _ACTIVE_POINTER.read_raw()
    if ptr and os.path.isdir(ptr):
        return ptr

    return "assets/index"


# =============================================================================
# Main Retrieval Logic
# =============================================================================
def retrieve(question: str, index_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Semantic search + smart expansion + keyword fallback (IDMap-safe).
    """
    resolved_dir = _resolve_index_dir(index_dir)
    idx, rows = load_index_and_chunks(resolved_dir)

    empty_result = {
        "question": question,
        "has_evidence": False,
        "evidence": []
    }

    if idx is None or not rows:
        print(f"[Retriever] No index found at {resolved_dir}")
        return empty_result

    # Build ID map (CRITICAL for IndexIDMap2 correctness)
    id_map = _build_id_map(rows)
    if not id_map:
        print("[Retriever] No active rows available in chunks.jsonl")
        return empty_result

    # 1) Expand abbreviations + embed query
    expanded_q = _expand_abbreviations(question)
    if expanded_q != question:
        print(f"[Retriever] Expanded: '{question}' -> '{expanded_q}'")

    try:
        print(f"[Retriever] Embedding query: '{expanded_q}'...")
        qv = embed_texts([expanded_q])[0]
        qv = _normalize_vec(qv)  # CRITICAL: match normalized index vectors (cosine sim stability)
        print(f"[Retriever] Embedding done. Shape: {qv.shape}")
    except Exception as e:
        print(f"[Retriever] Embedding failed: {e}")
        return empty_result

    # 2) Search FAISS (IndexIDMap2 returns IDs, not list positions)
    try:
        print(f"[Retriever] Searching FAISS with TOP_K={TOP_K}...")
        D, I = idx.search(qv.reshape(1, -1), TOP_K)
        print(f"[Retriever] Search done. Found {len(I[0])} hits.")
    except Exception as e:
        print(f"[Retriever] FAISS search failed: {e}")
        return empty_result

    base_ids: List[int] = []
    base_id_set = set()
    for x in I[0]:
        try:
            xi = int(x)
        except Exception:
            continue
        if xi < 0:
            continue
        if xi in id_map:  # ignore stale IDs
            base_ids.append(xi)
            base_id_set.add(xi)

    # --- Smart Page Expansion Logic (ID-based) ---
    should_expand = any(k in (question or "").lower() for k in _EXPANSION_KEYWORDS)
    expanded_ids = set()

    if should_expand and base_ids:
        print("[Retriever] Smart Expansion Triggered (Salary/Detail context)")
        page_map = _get_page_map_by_id(id_map)

        # For top 10 FAISS hits, fetch neighbor pages ±RADIUS
        for rank, (score, cid) in enumerate(zip(D[0], base_ids)):
            if rank >= 10:
                break

            r = id_map.get(cid)
            if not r:
                continue

            doc = r.get("doc_name")
            page = r.get("loc_start")

            if isinstance(page, int) and doc:
                for offset in range(-_EXPANSION_RADIUS, _EXPANSION_RADIUS + 1):
                    if offset == 0:
                        continue
                    p = page + offset
                    for neighbor_id in page_map.get((doc, p), []):
                        if neighbor_id not in base_id_set:
                            expanded_ids.add(neighbor_id)

    print(f"[Retriever] Added {len(expanded_ids)} context chunks.")

    # --- Hybrid Search: Keyword fallback (ID-based + safer matching) ---
    # Run keyword scan, but with safer token matching to reduce false positives.
    keyword_hits: Dict[int, float] = {}

    try:
        q_clean = _re.sub(r"[^\w\s]", " ", (expanded_q or "").lower())
        _stop = {
            "kya", "hai", "kon", "kaun", "ki", "ka", "ke", "se", "ko", "ne", "ye", "yeh",
            "what", "who", "is", "the", "a", "an", "of", "in", "for", "and", "how", "where",
            "when", "which", "does", "was", "are", "kia", "hain", "mein", "par", "say",
        }
        q_words = [w for w in q_clean.split() if w not in _stop and len(w) > 1]

        # Limit phrase to avoid noisy giant phrases
        full_phrase = " ".join(q_words[:12]).strip()
        phrase_enabled = len(q_words) > 1 and len(full_phrase) >= 6

        # Token regex supports English + Urdu range; keeps numbers too
        token_re = _re.compile(r"[a-z0-9\u0600-\u06FF]+", _re.IGNORECASE)

        for cid, r in id_map.items():
            txt = (r.get("text") or "").lower()
            if not txt:
                continue

            # Tokenize once per chunk
            tokens = set(token_re.findall(txt))

            match_count = sum(1 for w in q_words if w in tokens)
            if not q_words:
                continue

            is_phrase_match = phrase_enabled and (full_phrase in txt)

            score = 0.0
            if is_phrase_match:
                score = 0.72
            else:
                ratio = match_count / max(1, len(q_words))
                if ratio == 1.0:
                    score = 0.64
                elif ratio >= 0.75 and len(q_words) >= 3:
                    score = 0.60
                elif ratio >= 0.5 and len(q_words) >= 2:
                    score = 0.55

            if score >= 0.55:
                prev = keyword_hits.get(cid, 0.0)
                if score > prev:
                    keyword_hits[cid] = score

    except Exception as e:
        print(f"[Retriever] Keyword fallback error: {e}")

    # Map FAISS ID -> semantic score
    faiss_scores: Dict[int, float] = {}
    for score, cid in zip(D[0], I[0]):
        try:
            cii = int(cid)
            if cii in id_map:
                faiss_scores[cii] = float(score)
        except Exception:
            continue

    # Final keyword list with a small semantic boost
    final_keyword_list: List[Tuple[int, float]] = []
    for cid, ks in keyword_hits.items():
        base_score = float(ks)
        if cid in faiss_scores:
            base_score += float(faiss_scores[cid]) * 0.10
        final_keyword_list.append((cid, base_score))

    final_keyword_list.sort(key=lambda x: x[1], reverse=True)
    final_keyword_list = final_keyword_list[:10]  # prevent flooding

    # 3) Format results (Grouped by Document) with score filtering + dedupe
    docs_map: Dict[str, Dict[str, Any]] = {}

    def _ensure_doc(doc_name: str, initial_score: float) -> Dict[str, Any]:
        if doc_name not in docs_map:
            docs_map[doc_name] = {
                "doc_name": doc_name,
                "max_score": float(initial_score),
                "hits": [],
                "_seen": set(),  # internal dedupe
            }
        return docs_map[doc_name]

    def _process_hit(chunk_id: int, score_val: float, is_context: bool = False) -> None:
        r = id_map.get(int(chunk_id))
        if not r:
            return

        final_score = float(score_val)

        # Skip below threshold ONLY if not smart context
        if not is_context and final_score < SIM_THRESHOLD:
            return

        doc_name = r.get("doc_name", "Unknown")
        text = r.get("text", "") or ""
        page = r.get("loc_start", "?")
        public_path = r.get("public_path", f"/assets/data/{doc_name}")

        doc_group = _ensure_doc(doc_name, final_score)

        # Update max score only from non-context hits
        if (not is_context) and final_score > float(doc_group["max_score"]):
            doc_group["max_score"] = final_score

        # Dedupe same (page + text hash prefix)
        sig = (str(page), text[:200])
        if sig in doc_group["_seen"]:
            return
        doc_group["_seen"].add(sig)

        doc_group["hits"].append({
            "text": text,
            "score": final_score,
            "page_start": page,
            "public_path": public_path,
            "_is_smart_context": is_context
        })

    # A) FAISS hits (IDs)
    for score, cid in zip(D[0], I[0]):
        try:
            cii = int(cid)
        except Exception:
            continue
        if cii < 0:
            continue
        _process_hit(cii, float(score), is_context=False)

    # B) Expanded neighbor IDs (context)
    for cid in expanded_ids:
        _process_hit(int(cid), SIM_THRESHOLD, is_context=True)

    # C) Keyword hits (IDs)
    for cid, sc in final_keyword_list:
        _process_hit(int(cid), float(sc), is_context=False)

    evidence = list(docs_map.values())
    # Remove internal dedupe tracker
    for d in evidence:
        if "_seen" in d:
            del d["_seen"]

    evidence.sort(key=lambda x: float(x.get("max_score", 0)), reverse=True)

    return {
        "question": question,
        "has_evidence": len(evidence) > 0,
        "evidence": evidence
    }


# =============================================================================
# Query Contextualizer (Memory)
# =============================================================================
def rewrite_contextual_query(current_query: str, last_question: str, last_answer: str) -> str:
    """
    Rewrite follow-up questions to be standalone using LLM.
    """
    should_rewrite = os.getenv("RETRIEVER_LLM_QUERY_REWRITE_ALWAYS", "0") != "0"

    if not last_question and not should_rewrite:
        return current_query

    if len(current_query) < 4 and current_query.lower() in ["ok", "thanks", "theek", "sahi"]:
        return current_query

    system_prompt = (
        "You are a query rewriter for a RAG system.\n"
        "Your task: Rewrite the user query to be a standalone, semantically rich search query.\n"
        "Rules:\n"
        "1. Expand abbreviations (e.g. 'CTO' -> 'Chief Technology Officer', 'DG' -> 'Director General').\n"
        "2. Map broad terms to specific document phrasing (e.g. 'powers' -> 'powers, functions, responsibilities').\n"
        "3. Urdu/Hindi: Preserve direction of action and correct subject/object.\n"
        "4. Resolve pronouns using History if available.\n"
        "5. Keep final query in English for best match with document corpus.\n"
        "6. OUTPUT ONLY THE REWRITTEN QUERY."
    )

    user_prompt = (
        f"History: {last_question or 'None'}\n"
        f"Answer Context: {(last_answer or '')[:200]}...\n"
        f"Current Follow-up: {current_query}\n"
        "Rewritten Query:"
    )

    try:
        client = get_client()
        response = client.chat.completions.create(
            model=LLM_REWRITE_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Retriever] Rewrite failed: {e}")
        return current_query
