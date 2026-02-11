"""
PERA AI Retriever (Brain 2.0 + Hybrid Search + Query Normalization)
Simplified, robust semantic search with BM25-lite keyword scoring.
"""
from __future__ import annotations

import os
import json
from typing import List, Dict, Any, Optional
import numpy as np

from dotenv import load_dotenv
from openai import OpenAI
from index_store import load_index_and_chunks, embed_texts

load_dotenv()

# Configuration
TOP_K = int(os.getenv("RETRIEVER_TOP_K", "30"))
SIM_THRESHOLD = float(os.getenv("RETRIEVER_SIM_THRESHOLD", "0.14"))
LLM_REWRITE_MODEL = os.getenv("RETRIEVER_LLM_QUERY_REWRITE_MODEL", "gpt-4o-mini")

# Abbreviation -> full expansion (for embedding search quality)
_ABBREV_MAP = {
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
    "sppp": "Special Pay Package PERA",
    "lms": "Learning Management System",
    "faqs": "Frequently Asked Questions",
}

# Smart context expansion keywords

import re as _re
import math as _math
from collections import defaultdict, Counter


def _json_safe(x):
    """Cast numpy scalars to native Python types for JSON/Pydantic serialization."""
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x

# ── Query normalization: fix common typos / phonetic misspellings ─────────────
_QUERY_TYPO_MAP = {
    "sehdule": "schedule", "shedule": "schedule", "schdule": "schedule",
    "scheduel": "schedule", "schedul": "schedule",
    "sa;lary": "salary", "salray": "salary", "salry": "salary",
    "sallary": "salary", "slary": "salary",
    "pera": "PERA", "perra": "PERA",
    "sppp1": "SPPP-1", "sppp2": "SPPP-2", "sppp3": "SPPP-3",
    "sppp4": "SPPP-4", "sppp5": "SPPP-5", "sppp6": "SPPP-6",
    "sppp7": "SPPP-7", "sppp8": "SPPP-8", "sppp9": "SPPP-9",
    "sppp10": "SPPP-10",
    "bps14": "BPS-14", "bps16": "BPS-16", "bps17": "BPS-17",
    "bps18": "BPS-18", "bps19": "BPS-19", "bps20": "BPS-20",
}

# ── Keyword boosts: if query contains these, inject them as extra search terms
_KEYWORD_BOOSTS = {
    "schedule 1": ["Schedule-I", "Description of Offences and Fines", "offences", "penalties"],
    "schedule 3": ["Schedule-III", "Special Pay Package", "SPPP", "minimum", "maximum"],
    "schedule i": ["Schedule-I", "Description of Offences and Fines"],
    "schedule iii": ["Schedule-III", "Special Pay Package", "SPPP"],
    "sppp": ["Special Pay Package", "Schedule-III", "SPPP", "minimum", "maximum"],
    "sppp-1": ["SPPP-1", "Special Pay Package", "Schedule-III"],
    "salary table": ["Schedule-III", "pay package", "SPPP", "BPS", "minimum", "maximum"],
    "pay scale": ["Schedule-III", "Special Pay Package", "BPS", "pay scale"],
    "pay package": ["Special Pay Package", "SPPP", "Schedule-III"],
    "what is pera": ["Punjab Enforcement Regulatory Authority", "established", "functions", "objectives"],
    "pera kia hai": ["Punjab Enforcement Regulatory Authority", "established", "functions"],
    "pera kya hai": ["Punjab Enforcement Regulatory Authority", "established", "functions"],
}

def _normalize_query(query: str) -> str:
    """Fix common typos and phonetic misspellings."""
    words = query.split()
    normalized = []
    for w in words:
        key = w.lower().strip()
        if key in _QUERY_TYPO_MAP:
            normalized.append(_QUERY_TYPO_MAP[key])
        else:
            normalized.append(w)
    return " ".join(normalized)

def _get_keyword_boosts(query: str) -> List[str]:
    """Return extra keyword boost terms for the query."""
    q_lower = query.lower()
    boosts = []
    for trigger, terms in _KEYWORD_BOOSTS.items():
        if trigger in q_lower:
            boosts.extend(terms)
    return list(set(boosts))


# ── Table Registry: index-aware deterministic lookup ──────────────────────────

# Routes: keyword triggers -> lambda that matches chunk text
_TABLE_ROUTES = [
    # Schedule-I (offences / fines table)
    {
        "triggers": ["schedule 1", "schedule-i", "schedule i", "sehdule 1",
                      "schedule-1", "offences table", "fines table"],
        "match": lambda t: ("schedule-i" in t or "schedule - i" in t or
                            ("offence" in t and "fine" in t) or
                            ("offences" in t and "penalties" in t)),
        "label": "Schedule-I",
    },
    # Schedule-III / SPPP salary table
    {
        "triggers": ["schedule 3", "schedule-iii", "schedule iii", "sehdule 3",
                      "schedule-3", "sppp salary", "sppp-1 salary", "sppp1 salary",
                      "sppp table", "special pay package", "sppp-1", "sppp 1"],
        "match": lambda t: ("sppp" in t and ("minimum" in t or "maximum" in t or
                            "350" in t or "schedule-iii" in t)),
        "label": "Schedule-III/SPPP",
    },
]


class TableRegistry:
    """
    Index-aware table registry.  Tracks which index_dir it was built for
    and auto-rebuilds when the active index changes (hot-swap safe).
    """
    def __init__(self):
        self._registry: Dict[str, list] = {}
        self._built_for_index: Optional[str] = None
        self._chunks_count: int = 0

    # ── public ────────────────────────────────────────────────────────────
    @property
    def index_dir(self) -> Optional[str]:
        return self._built_for_index

    @property
    def ready(self) -> bool:
        return bool(self._registry)

    @property
    def chunks_count(self) -> int:
        return self._chunks_count

    def build(self, chunks: list, index_dir: str) -> Dict[str, list]:
        """
        Build (or rebuild) the registry from *chunks*.
        Caches result; subsequent calls with the same index_dir are no-ops.
        """
        norm = (index_dir or "").replace("\\", "/").rstrip("/")
        if self._built_for_index == norm and self._registry:
            return self._registry

        registry: Dict[str, list] = {}
        for route in _TABLE_ROUTES:
            label = route["label"]
            registry[label] = []
            for ci, chunk in enumerate(chunks):
                text_lower = (chunk.get("text") or "").lower()
                if route["match"](text_lower):
                    registry[label].append(ci)

        self._registry = registry
        self._built_for_index = norm
        self._chunks_count = len(chunks)
        for label, indices in registry.items():
            if indices:
                print(f"[TableRegistry] {label}: {len(indices)} chunks (index={norm})")
        return registry

    def invalidate(self):
        """Force rebuild on next call."""
        self._registry.clear()
        self._built_for_index = None
        self._chunks_count = 0

    def lookup(self, question: str, chunks: list,
               index_dir: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """
        Deterministic table lookup — returns evidence list or None.
        Auto-rebuilds registry if index_dir changed.
        """
        if index_dir:
            self.build(chunks, index_dir)

        q_lower = _normalize_query(question).lower()
        for route in _TABLE_ROUTES:
            for trigger in route["triggers"]:
                if trigger in q_lower:
                    label = route["label"]
                    indices = self._registry.get(label, [])
                    if not indices:
                        continue
                    print(f"[TableLookup] Matched '{trigger}' -> {label} ({len(indices)} chunks)")
                    from collections import defaultdict
                    docs_map = defaultdict(lambda: {"doc_name": "", "max_score": 1.0, "hits": []})
                    for ci in indices:
                        chunk = chunks[ci]
                        doc_name = chunk.get("doc_name", "Unknown")
                        docs_map[doc_name]["doc_name"] = doc_name
                        docs_map[doc_name]["hits"].append({
                            "text": chunk.get("text", ""),
                            "score": 1.0,
                            "page_start": chunk.get("loc_start", 0),
                            "page_end": chunk.get("loc_end", 0),
                            "chunk_index": ci,
                            "_is_table_lookup": True,
                        })
                    return list(docs_map.values())
        return None

    def debug_info(self) -> Dict[str, Any]:
        """Return debug snapshot."""
        return {
            "index_dir": self._built_for_index,
            "ready": self.ready,
            "chunks_count": self._chunks_count,
            "labels": {k: len(v) for k, v in self._registry.items()},
        }


# Module-level singleton
_table_registry = TableRegistry()


def table_lookup(question: str, chunks: list,
                 index_dir: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
    """Convenience wrapper around the singleton registry."""
    return _table_registry.lookup(question, chunks, index_dir=index_dir)


def get_table_registry() -> TableRegistry:
    """Return the singleton for debug / index-swap hooks."""
    return _table_registry

# Smart Context Expansion Keywords
# If query contains these, we fetch adjacent pages (+-1) to capture tables/schedules
_EXPANSION_KEYWORDS = {
    "salary", "pay", "allowance", "benefit", "scale", "sppp", "grade", "compensation",
    "detail", "full", "sab kuch", "batao", "explain", "structure",
    "schedule", "appendix", "annex", "table",
    # Roman Urdu / misspellings
    "salay", "tankhwah", "tankha", "kitni", "payscale", "pay scale",
    "maaash", "maash", "salary",
}
_EXPANSION_RADIUS = 3  # Fetch ±3 pages for salary/detail queries

# ── MMR (Maximal Marginal Relevance) re-ranking ─────────────────────────────
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.7"))   # 0=full diversity, 1=full relevance
MMR_TOP_K = int(os.getenv("MMR_TOP_K", "20"))         # final hits after MMR

def _mmr_rerank(
    query_vec: np.ndarray,
    candidate_indices: List[int],
    candidate_scores: List[float],
    chunks: List[Dict[str, Any]],
    embed_fn=None,
    lam: float = MMR_LAMBDA,
    top_k: int = MMR_TOP_K,
) -> List[int]:
    """
    Maximal Marginal Relevance re-ranking.
    Returns up to top_k indices from candidate_indices, balancing
    relevance (to query) and diversity (among selected).
    Uses text-based Jaccard similarity as a lightweight proxy when
    chunk embeddings are not cached.
    """
    if len(candidate_indices) <= top_k:
        return list(candidate_indices)

    # Build text-token sets for fast Jaccard
    from collections import Counter
    _tok_cache: Dict[int, set] = {}
    def _tokens(ci: int) -> set:
        if ci not in _tok_cache:
            text = (chunks[ci].get("text") or "").lower()
            _tok_cache[ci] = set(text.split())
        return _tok_cache[ci]

    def _jaccard(a: int, b: int) -> float:
        ta, tb = _tokens(a), _tokens(b)
        inter = len(ta & tb)
        union = len(ta | tb)
        return inter / union if union else 0.0

    # Pair each candidate with its relevance score
    scored = list(zip(candidate_indices, candidate_scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    selected: List[int] = [scored[0][0]]
    remaining = [(ci, s) for ci, s in scored[1:]]

    for _ in range(min(top_k - 1, len(remaining))):
        if not remaining:
            break
        best_idx = -1
        best_mmr = -1e9
        for ri, (ci, rel) in enumerate(remaining):
            max_sim = max(_jaccard(ci, si) for si in selected)
            mmr_score = lam * rel - (1 - lam) * max_sim
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = ri
        sel_ci, _ = remaining.pop(best_idx)
        selected.append(sel_ci)

    return selected

def _get_page_map(chunks: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    """Build a map of (doc_name, page) -> list of chunk indices."""
    m = defaultdict(list)
    for i, c in enumerate(chunks):
        doc = c.get("doc_name", "Unknown")
        # heuristic: loc_start is usually page num for PDFs
        page = c.get("loc_start") 
        if isinstance(page, int):
             m[(doc, page)].append(i)
    return m

def _expand_abbreviations(query: str) -> str:
    """Expand known abbreviations in-place for better embedding matches."""
    words = query.split()
    expanded = []
    for w in words:
        key = _re.sub(r'[^a-zA-Z]', '', w).lower()
        if key in _ABBREV_MAP:
            expanded.append(_ABBREV_MAP[key])
        else:
            expanded.append(w)
    return " ".join(expanded)

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
        
    # Fallback
    return "assets/index"

# =============================================================================
# Main Retrieval Logic
# =============================================================================
def retrieve(question: str, index_dir: Optional[str] = None, intent: Optional[str] = None, pinned_chunks: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Hybrid search: vector (FAISS) + BM25-lite keyword scoring + pinned chunk boosting.
    intent: optional intent string (e.g. 'DOC_LOOKUP') to adjust thresholds.
    pinned_chunks: list of dicts {chunk_id, doc_name} from previous turns to boost.
    """
    resolved_dir = _resolve_index_dir(index_dir)
    idx, chunks = load_index_and_chunks(resolved_dir)
    
    # Determine effective threshold based on intent
    is_doc_lookup = (intent == "DOC_LOOKUP")
    effective_threshold = 0.08 if is_doc_lookup else SIM_THRESHOLD
    
    empty_result = {
        "question": question,
        "has_evidence": False,
        "evidence": []
    }

    if idx is None or not chunks:
        print(f"[Retriever] No index found at {resolved_dir}")
        return empty_result

    # 0. Normalize query (fix typos)
    question = _normalize_query(question)

    # 0a. DETERMINISTIC TABLE LOOKUP (bypass FAISS for schedule/SPPP)
    table_evidence = table_lookup(question, chunks, index_dir=resolved_dir)
    if table_evidence:
        print(f"[Retriever] Table lookup returned {len(table_evidence)} doc groups")
        return {
            "question": question,
            "has_evidence": True,
            "evidence": table_evidence,
            "_table_lookup": True,
            "_index_dir": resolved_dir,
        }

    boost_terms = _get_keyword_boosts(question)
    if boost_terms:
        print(f"[Retriever] Keyword boosts: {boost_terms[:5]}")

    # 1. Expand abbreviations + Embed query
    expanded_q = _expand_abbreviations(question)
    if expanded_q != question:
        print(f"[Retriever] Expanded: '{question}' -> '{expanded_q}'")
    try:
        print(f"[Retriever] Embedding query: '{expanded_q}'...")
        query_vec = embed_texts([expanded_q])[0]
        print(f"[Retriever] Embedding done. Shape: {query_vec.shape}")
    except Exception as e:
        print(f"[Retriever] Embedding failed: {e}")
        return empty_result

    # 2. Search FAISS
    # We fetch TOP_K chunks. Hierarchical chunks already have role context.
    try:
        print(f"[Retriever] Searching FAISS with TOP_K={TOP_K}...")
        D, I = idx.search(query_vec.reshape(1, -1), TOP_K)
        print(f"[Retriever] Search done. Found {len(I[0])} hits.")
    except Exception as e:
        print(f"[Retriever] FAISS search failed: {e}")
        return empty_result

    # --- Smart Page Expansion Logic ---
    # intended for salary/tables often disconnected from role definition
    should_expand = any(k in question.lower() for k in _EXPANSION_KEYWORDS)
    expanded_hits_indices = set()
    
    if should_expand:
        print("[Retriever] Smart Expansion Triggered (Salary/Detail context)")
        page_map = _get_page_map(chunks)
        # For top 10 hits, fetch neighbor pages ±RADIUS
        for rank, (score, doc_idx) in enumerate(zip(D[0], I[0])):
            doc_idx = int(doc_idx)
            if rank >= 10: break 
            if doc_idx < 0 or doc_idx >= len(chunks): continue
            
            c = chunks[doc_idx]
            doc = c.get("doc_name")
            page = c.get("loc_start")
            
            if isinstance(page, int):
                # Fetch neighbors: page-RADIUS to page+RADIUS
                for offset in range(-_EXPANSION_RADIUS, _EXPANSION_RADIUS + 1):
                    if offset == 0: continue  # skip self
                    p = page + offset
                    if (doc, p) in page_map:
                        for neighbor_idx in page_map[(doc, p)]:
                            if neighbor_idx not in I[0]: 
                                expanded_hits_indices.add(neighbor_idx)
    
    print(f"[Retriever] Added {len(expanded_hits_indices)} context chunks.")

    # --- Hybrid Search: Keyword Fallback for Names/Entities ---
    # FAISS semantic search fails on proper names or specific roles. 
    # We ALWAYS run a keyword scan to find exact matches and inject them.
    top_faiss_score = float(D[0][0]) if len(D[0]) > 0 else 0
    keyword_hits = {}  # Dict {index: score}

    # --- Pinned Boost Logic (Memory) ---
    # If we have pinned chunks from recent history, boost them significantly so they appear
    # at the top if they are even remotely relevant.
    # We add them to keyword_hits with a synthetic high score.
    if pinned_chunks:
        print(f"[Retriever] Boosting {len(pinned_chunks)} pinned chunks from history")
        for pc in pinned_chunks:
             try:
                 cid = int(pc.get("chunk_id", -1))
                 if cid >= 0 and cid < len(chunks):
                     # Boost score: enough to pass thresholds but maybe not beat a perfect vector match
                     # unless it's very relevant. 
                     # Let's give it a baseline of 0.50 (above most thresholds).
                     # If it matches keywords, it will go higher.
                     keyword_hits[cid] = 0.55 
             except:
                 pass
    
    # Run keyword search regardless of FAISS score (Hybrid Search)
    if True: 
        # Extract meaningful words from query (skip common Urdu/English stopwords)
        import re
        q_lower = re.sub(r'[^\w\s]', '', question.lower())  # Strip punctuation
        _stop = {"kya", "hai", "kon", "kaun", "ki", "ka", "ke", "se", "ko", "ne", "ye", "yeh",
                 "what", "who", "is", "the", "a", "an", "of", "in", "for", "and", "how", "where",
                 "when", "which", "does", "was", "are", "kia", "hain", "mein", "par", "say"}
        q_words = [w for w in q_lower.split() if w not in _stop and len(w) > 1]
        
        if len(q_words) >= 1:
            # Also add boost terms as extra keywords for matching
            all_search_words = list(q_words)
            if boost_terms:
                all_search_words.extend([b.lower() for b in boost_terms])
                all_search_words = list(set(all_search_words))
            
            # Check overlap for every chunk
            for ci, chunk in enumerate(chunks):
                text_lower = chunk.get("text", "").lower()
                
                # count how many search words are in text
                match_count = sum(1 for w in all_search_words if w in text_lower)
                
                # Check for full phrase match (bonus)
                full_phrase = " ".join(q_words)
                is_phrase_match = (full_phrase in text_lower) if len(q_words) > 1 else False
                
                # BM25-lite: token overlap ratio
                ratio = match_count / max(len(all_search_words), 1)
                
                score = 0.0
                if is_phrase_match:
                    score = 0.75  # Super high confidence for exact phrase
                elif ratio == 1.0:
                    score = 0.65  # All words present
                elif ratio >= 0.75 and len(all_search_words) >= 3:
                    score = 0.60  # Most words present
                elif ratio >= 0.5 and len(all_search_words) >= 2:
                    score = 0.55  # Half words present
                elif ratio >= 0.3 and is_doc_lookup:
                    score = 0.45  # Lower bar for DOC_LOOKUP intent
                
                # Only add if score is significant
                if score >= (0.40 if is_doc_lookup else 0.55):
                    if ci not in keyword_hits:
                        keyword_hits[ci] = score
                    else:
                        keyword_hits[ci] = max(keyword_hits[ci], score)

            # Remove already-found indices from FAISS results to avoid duplication logic
            # (though _process_hit handles duplication by doc, we pass index)
            # Actually, `keyword_hits` is now a dict {idx: score}. 
            # We should filter out indices that are already in I[0] BUT 
            # if our keyword score is higher, we might want to update it?
            # For simplicity, let's just process them. _process_hit will update max_score 
            # for the document if valid.



    # 3. Format results (Grouped by Document) with score filtering
    docs_map = {}
    filtered_count = 0
    
    # Helper to process a chunk index with a score
    def _process_hit(idx_val, score_val, is_context=False):
        idx_val = int(idx_val)              # SEV-0: FAISS returns numpy.int64
        if idx_val < 0 or idx_val >= len(chunks):
            return
        
        final_score = float(score_val)      # SEV-0: FAISS returns numpy.float32
        # Skip below threshold ONLY if not smart context
        if not is_context and final_score < effective_threshold:
            return

        chunk = chunks[idx_val]
        text = chunk.get("text", "")
        doc_name = chunk.get("doc_name", "Unknown")

        if doc_name not in docs_map:
            docs_map[doc_name] = {
                "doc_name": doc_name,
                "max_score": final_score, # Context chunks might have lower score, but we usually update this with max
                "hits": []
            }
        
        # Update max score for the doc group (context chunks don't bump max score to avoid ranking irrelevant docs high)
        if not is_context and final_score > docs_map[doc_name]["max_score"]:
            docs_map[doc_name]["max_score"] = final_score

        # Add hit (enriched with chunk_id and tier for extract pipeline)
        tier = "table" if chunk.get("_is_table_lookup") else (
            "keyword" if is_context else "vector"
        )
        docs_map[doc_name]["hits"].append({
            "text": text,
            "score": final_score,
            "chunk_id": int(idx_val),                            # SEV-0: always native int
            "page_start": _json_safe(chunk.get("loc_start", "?")),
            "page_end": _json_safe(chunk.get("loc_end", chunk.get("loc_start", "?"))),
            "public_path": chunk.get("public_path", f"/assets/data/{doc_name}"),
            "tier": tier,
            "_is_smart_context": is_context
        })

    # A. Process FAISS matches
    for rank, (score, doc_idx) in enumerate(zip(D[0], I[0])):
        _process_hit(int(doc_idx), float(score), is_context=False)
        
    # B. Process Expanded matches (give them a synthetic score just below the threshold so they appear at bottom of doc group?)
    # Actually score doesn't matter for filtering if we flag them. We can give them SIM_THRESHOLD to be safe.
    for idx_val in expanded_hits_indices:
        _process_hit(idx_val, SIM_THRESHOLD, is_context=True)

    # C. Process Keyword fallback matches (give them a boosted score since they're exact text matches)
    # Add semantic boost if available to preserve ranking among keyword hits
    final_keyword_list = []
    try:
        # Map index -> semantic score from FAISS results (if any)
        # CAST to int because FAISS returns numpy int64 which might not hash same as int in some envs
        faiss_scores = {int(idx): float(score) for idx, score in zip(I[0], D[0])}
        
        for idx_val, score_val in keyword_hits.items():
            base_score = float(score_val)
            # Boost if it also has a high semantic score (Base + 10% of Semantic)
            if idx_val in faiss_scores:
                base_score += (faiss_scores[idx_val] * 0.1)
            final_keyword_list.append((idx_val, base_score))
            
        # Sort by score descending and take top 10 to prevent context flooding
        final_keyword_list.sort(key=lambda x: x[1], reverse=True)
        final_keyword_list = final_keyword_list[:10]
    except Exception as e:
        print(f"[Retriever] Hybrid scoring error: {e}")
        # Fallback to just using keyword hits as is (raw score)
        final_keyword_list = [(k, v) for k, v in keyword_hits.items()]
    
    for idx_val, score_val in final_keyword_list:
        _process_hit(idx_val, score_val, is_context=False)

    # D. For DOC_LOOKUP: also ensure boost-term chunks are included even if FAISS missed them
    if is_doc_lookup and boost_terms:
        boost_lower = [b.lower() for b in boost_terms]
        for ci, chunk in enumerate(chunks):
            if ci in docs_map:  # already processed
                continue
            text_lower = chunk.get("text", "").lower()
            boost_overlap = sum(1 for b in boost_lower if b in text_lower)
            if boost_overlap >= 2:  # at least 2 boost terms match
                _process_hit(ci, 0.40, is_context=True)  # inject with modest score

    # E. MMR diversity: collect all unique chunk indices, re-rank, rebuild
    all_hit_indices = []
    all_hit_scores = []
    _hit_data = {}  # chunk_id -> hit dict + doc_name
    for doc_name, dg in docs_map.items():
        for hit in dg["hits"]:
            ci = hit.get("chunk_id")
            if ci is not None and ci not in _hit_data:
                all_hit_indices.append(ci)
                all_hit_scores.append(hit["score"])
                _hit_data[ci] = (doc_name, hit)

    if len(all_hit_indices) > MMR_TOP_K:
        try:
            mmr_selected = _mmr_rerank(
                query_vec, all_hit_indices, all_hit_scores, chunks,
                lam=MMR_LAMBDA, top_k=MMR_TOP_K,
            )
            mmr_set = set(mmr_selected)
            print(f"[Retriever] MMR: {len(all_hit_indices)} -> {len(mmr_set)} diverse hits")
            # Rebuild docs_map with only MMR-selected hits
            new_docs_map = {}
            for ci in mmr_selected:
                if ci in _hit_data:
                    doc_name, hit = _hit_data[ci]
                    if doc_name not in new_docs_map:
                        new_docs_map[doc_name] = {
                            "doc_name": doc_name,
                            "max_score": hit["score"],
                            "hits": [],
                        }
                    new_docs_map[doc_name]["hits"].append(hit)
                    if hit["score"] > new_docs_map[doc_name]["max_score"]:
                        new_docs_map[doc_name]["max_score"] = hit["score"]
            docs_map = new_docs_map
        except Exception as e:
            print(f"[Retriever] MMR failed, using raw results: {e}")

    # Convert to list and SORT by max_score descending
    evidence = list(docs_map.values())
    evidence.sort(key=lambda x: x["max_score"], reverse=True)

    return {
        "question": question,
        "has_evidence": len(evidence) > 0,
        "evidence": evidence,
        "_index_dir": resolved_dir,
    }

# =============================================================================
# Query Contextualizer (Memory)
# =============================================================================
# =============================================================================
# Query Contextualizer (Memory)
# =============================================================================
def rewrite_contextual_query(current_query: str, history_block: str) -> str:
    """
    Rewrite follow-up questions to be standalone using LLM + constraints.
    history_block: Text block of last 5 turns (User: ... \n Assistant: ...)
    """
    if not history_block or not current_query.strip():
        return current_query

    # 0. Fast Heuristics: Don't rewrite if query is very short/greeting
    q_lower = current_query.lower().strip()
    if len(q_lower) < 5 and q_lower in ["hello", "hi", "salam", "ok", "thanks"]:
        return current_query

    # 1. Heuristic Trigger: Only rewrite if we detect ambiguity or explicitly asked
    # Pronouns/Follow-up markers
    triggers = [
        "usne", "unhone", "iska", "iski", "uske", "woh", "ye", "yeh", "it", "this", "that", 
        "he", "she", "they", "them", "previous", "uphar", "uupar", "above", "pehle",
        "and", "or", "phip", "fir", "what about", "kya ye", "kya wo"
    ]
    # Check if ANY trigger word is in the query (simple token match)
    tokens = q_lower.split()
    has_trigger = any(t in tokens for t in triggers)
    
    # Always rewrite if env var is set, otherwise rely on trigger
    force_rewrite = os.getenv("RETRIEVER_ALWAYS_REWRITE", "0") == "1"
    
    if not has_trigger and not force_rewrite:
        # If no obvious pronoun, maybe it's self-contained? 
        # But for 'salary' it might be 'salary of *previous job*', so context matters.
        # Let's be conservative: if it's longer (MIN_WORDS) assume it's standalone?
        # Actually safer to skip rewrite to avoid hallucinating context if not needed.
        return current_query

    print(f"[Retriever] Rewriting query due to context trigger followed by history block length: {len(history_block)}")

    system_prompt = (
        "You are a query rewriter for a RAG system.\n"
        "Your task: Rewrite the user **Current Follow-up** to be a standalone, semantically rich search query, "
        "resolving any pronouns or references using the **Conversation History**.\n\n"
        "Rules:\n"
        "1. **Resolve References**: 'uske' -> 'DG's', 'iska' -> 'Leave Policy', 'it' -> 'Medical Allowance'.\n"
        "2. **Maintain Intent**: Do NOT change the question type (e.g. asking for salary vs powers).\n"
        "3. **Urdu/Hindi**: Translate carefully. Preserve direction of action.\n"
        "4. **Expand Abbreviations**: 'CTO' -> 'Chief Technology Officer', 'DG' -> 'Director General'.\n"
        "5. **Context**: Use the history to fill in missing entities.\n"
        "6. **Output**: ONLY the rewritten query text. No quotes, no preamble.\n"
    )
    
    user_prompt = (
        f"Conversation History:\n{history_block}\n\n"
        f"Current Follow-up: {current_query}\n\n"
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
        rewritten = response.choices[0].message.content.strip()
        # Fallback: if empty, return original
        return rewritten if rewritten else current_query
    except Exception as e:
        print(f"[Retriever] Rewrite failed: {e}")
        return current_query
