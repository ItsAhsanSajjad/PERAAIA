"""
PERA AI Retriever (Brain 2.0)
Simplified, robust semantic search without manual heuristic filtering.
"""
from __future__ import annotations

import os
import json
from typing import List, Dict, Any, Optional

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
    # Typos for "schedule" to ensure normalization works
    "schedudle": "schedule",
    "scheduel": "schedule",
    "scedule": "schedule",
    "sehdule": "schedule",
    "skedule": "schedule",
    "schdule": "schedule",
    "shcedule": "schedule",
    "shadule": "schedule",
    "schedul": "schedule",
}

# Smart context expansion keywords

import re as _re
from collections import defaultdict

# Smart Context Expansion Keywords
# If query contains these, we fetch adjacent pages (±1) to capture tables/schedules
_EXPANSION_KEYWORDS = {
    "salary", "pay", "allowance", "benefit", "scale", "sppp", "grade", "compensation",
    "detail", "full", "sab kuch", "batao", "explain", "structure",
    # Roman Urdu / misspellings
    "salay", "tankhwah", "tankha", "kitni", "payscale", "pay scale",
    "maaash", "maash", "salary",
}
_EXPANSION_RADIUS = 3  # Fetch ±3 pages for salary/detail queries

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
    """Expand known abbreviations and normalize schedule numbers."""
    words = query.split()
    expanded = []
    for w in words:
        key = _re.sub(r'[^a-zA-Z]', '', w).lower()
        if key in _ABBREV_MAP:
            expanded.append(_ABBREV_MAP[key])
        else:
            expanded.append(w)
    result = " ".join(expanded)
    
    # Schedule number normalization: "schedule 3" -> "Schedule-III"
    # Replace the Arabic numeral with the Roman numeral format used in PERA documents
    _arabic_to_roman = {"1": "I", "2": "II", "3": "III", "4": "IV", "5": "V",
                        "6": "VI", "7": "VII", "8": "VIII", "9": "IX", "10": "X"}
    m = _re.search(r'\bschedule[\s\-]*(\d+)\b', result, _re.IGNORECASE)
    if m:
        num = m.group(1)
        roman = _arabic_to_roman.get(num)
        if roman:
            # Replace "schedule 3" with "Schedule-III" in the main query
            result = result[:m.start()] + f"Schedule-{roman}" + result[m.end():]
    
    return result

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
def retrieve(question: str, index_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Pure semantic search. No manual filtering.
    """
    resolved_dir = _resolve_index_dir(index_dir)
    idx, chunks = load_index_and_chunks(resolved_dir)
    
    empty_result = {
        "question": question,
        "has_evidence": False,
        "evidence": []
    }

    if idx is None or not chunks:
        print(f"[Retriever] No index found at {resolved_dir}")
        return empty_result

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
    
    # Run keyword search regardless of FAISS score (Hybrid Search)
    if True: 
        # Extract meaningful words from EXPANDED query (includes abbreviation expansions)
        import re
        q_lower = re.sub(r'[^\w\s\-]', '', expanded_q.lower())  # Use expanded_q; preserve hyphens
        _stop = {"kya", "hai", "kon", "kaun", "ki", "ka", "ke", "se", "ko", "ne", "ye", "yeh",
                 "what", "who", "is", "the", "a", "an", "of", "in", "for", "and", "how", "where",
                 "when", "which", "does", "was", "are", "kia", "hain", "mein", "par", "say",
                 "hota", "hoti", "hote", "karta", "karte", "karo", "bata", "batao", "wala", "wali"}
        q_words = [w for w in q_lower.split() if w not in _stop and len(w) > 1]
        
        if len(q_words) >= 1:
            # Check overlap for every chunk
            for ci, chunk in enumerate(chunks):
                text_lower = chunk.get("text", "").lower()
                
                # count how many q_words are in text
                match_count = sum(1 for w in q_words if w in text_lower)
                
                # Check for full phrase match (bonus)
                full_phrase = " ".join(q_words)
                is_phrase_match = (full_phrase in text_lower) if len(q_words) > 1 else False
                
                score = 0.0
                if is_phrase_match:
                    score = 0.75  # Super high confidence for exact phrase
                elif len(q_words) >= 1:
                    ratio = match_count / len(q_words)
                    if ratio == 1.0:
                        score = 0.65  # All words present
                    elif ratio >= 0.75 and len(q_words) >= 3:
                        score = 0.60  # Most words present
                    elif ratio >= 0.5 and len(q_words) >= 2:
                        score = 0.55  # Half works present (only for multi-word queries)
                
                # Only add if score is significant and better than what FAISS found (roughly)
                if score >= 0.55:
                    # Store tuple (index, score)
                    # We use a dict to store max score for each index
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
        if idx_val < 0 or idx_val >= len(chunks):
            return
        
        final_score = float(score_val)
        # Skip below threshold ONLY if not smart context
        if not is_context and final_score < SIM_THRESHOLD:
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

        # Add hit
        docs_map[doc_name]["hits"].append({
            "text": text,
            "score": final_score,
            "page_start": chunk.get("loc_start", "?"),
            "public_path": chunk.get("public_path", f"/assets/data/{doc_name}"),
            "_is_smart_context": is_context
        })

    # A. Process FAISS matches
    for rank, (score, doc_idx) in enumerate(zip(D[0], I[0])):
        _process_hit(doc_idx, score, is_context=False)
        
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

    # D. SPPP Salary Table Enrichment
    # When query is about salary/pay, ensure the SPPP breakup table is included.
    # Problem: Role chunk (e.g. "Android Dev = SPPP-4") is on page 355,
    # but salary table ("SPPP-4 minimum/maximum pay") is on page 156/401.
    # Expansion (±3 pages) can't bridge that gap, so we inject it explicitly.
    _salary_kw = {"salary", "pay", "allowance", "benefit", "tankhwah", "maash",
                  "kitni", "compensation", "scale", "sppp", "number", "amount"}
    q_salary_check = question.lower()
    is_salary_q = any(kw in q_salary_check for kw in _salary_kw)
    
    if is_salary_q:
        # Check if SPPP table is already in results
        has_sppp_table = any(
            "Minimum Pay" in hit.get("text", "") and "Maximum Pay" in hit.get("text", "")
            for doc_data in docs_map.values()
            for hit in doc_data.get("hits", [])
        )
        
        if not has_sppp_table:
            injected = 0
            for ci, chunk in enumerate(chunks):
                text = chunk.get("text", "")
                if ("Minimum Pay" in text and "Maximum Pay" in text and "SPPP" in text):
                    _process_hit(ci, 0.55, is_context=False)  # Inject as regular hit (not context)
                    injected += 1
            if injected:
                print(f"[Retriever] Injected {injected} SPPP salary table chunk(s)")

    # Convert to list and SORT by max_score descending
    evidence = list(docs_map.values())
    evidence.sort(key=lambda x: x["max_score"], reverse=True)

    return {
        "question": question,
        "has_evidence": len(evidence) > 0,
        "evidence": evidence
    }

# =============================================================================
# Lightweight Typo Corrector (for standalone first queries)
# =============================================================================
_COMMON_TYPOS = {
    "genral": "general", "genreal": "general", "gneral": "general",
    "directer": "director", "diretor": "director", "drector": "director",
    "suppor": "support", "suport": "support", "suppert": "support",
    "salarey": "salary", "salaray": "salary", "salery": "salary",
    "offcer": "officer", "oficer": "officer", "offier": "officer",
    "manger": "manager", "mangaer": "manager", "maneger": "manager",
    "enforcment": "enforcement", "enforcemnt": "enforcement",
    "moniter": "monitor", "monitering": "monitoring", "montioring": "monitoring",
    "additonal": "additional", "aditional": "additional",
    "asistant": "assistant", "assistent": "assistant",
    "benifit": "benefit", "benfit": "benefit", "benifits": "benefits",
    "allowence": "allowance", "alowance": "allowance",
    "compansation": "compensation", "compenstion": "compensation",
    "responsibilties": "responsibilities", "responsibiliy": "responsibility",
    "qualificaton": "qualification", "qualifcation": "qualification",
    "experiance": "experience", "experienc": "experience",
    "regulaton": "regulation", "regulaion": "regulation",
    "notificaton": "notification", "noification": "notification",
    "suppo": "support",
}

def _fix_common_typos(query: str) -> str:
    """Fix common typos without using LLM — fast and doesn't over-expand."""
    words = query.split()
    fixed = []
    for w in words:
        w_lower = w.lower()
        if w_lower in _COMMON_TYPOS:
            # Preserve original case pattern
            replacement = _COMMON_TYPOS[w_lower]
            if w[0].isupper():
                replacement = replacement.capitalize()
            fixed.append(replacement)
        else:
            fixed.append(w)
    result = " ".join(fixed)
    if result != query:
        print(f"[Retriever] Typo fix: '{query}' -> '{result}'")
    return result

# =============================================================================
# Query Contextualizer (Memory)
# =============================================================================
def rewrite_contextual_query(current_query: str, last_question: str, last_answer: str) -> str:
    """
    Rewrite follow-up questions to be standalone using LLM.
    Only rewrites when there IS conversation history — first messages pass through unchanged
    because the embedding model handles typos/abbreviations well, and LLM rewriting
    over-expands terms (e.g. 'PERA' -> 'Punjab Enforcement and Regulation Authority') which hurts retrieval.
    """
    # Only use LLM rewrite for follow-ups (with history)
    # For standalone queries, just fix typos — LLM rewrite over-expands terms
    if not last_question:
        return _fix_common_typos(current_query)
        
    # If the user is just saying "thanks" or "ok", don't rewrite
    if len(current_query) < 4 and current_query.lower() in ["ok", "thanks", "theek", "sahi"]:
        return current_query

    system_prompt = (
        "You are a query rewriter for a RAG system.\n"
        "Your task: Rewrite the user query to be a standalone, semantically rich search query.\n"
        "Rules:\n"
        "1. Expand abbreviations (e.g. 'CTO' -> 'Chief Technology Officer', 'DG' -> 'Director General', 'EPO' -> 'Emergency Prohibition Order'). IMPORTANT: Do NOT expand 'PERA' — keep it as 'PERA'.\n"
        "2. Map broad terms to specific document sections:\n"
        "   - 'powers' -> 'powers, functions, and responsibilities'\n"
        "   - 'salary' / 'pay' / 'tankhwah' -> 'pay and allowances SPPP Schedule'\n"
        "   - 'in numbers' / 'kitni' / 'amount' -> 'minimum pay maximum pay PKR fuel limit SPPP'\n"
        "   - 'members of board' / 'pera board' -> 'Constitution of the Authority'\n"
        "   - 'appeal against EPO' -> 'representation against EPO' (Note: Tribunal appeal remains 'appeal')\n"
        "   - 'encroachment' -> 'removal of encroachment' or 'unauthorized occupation'\n"
        "3. **Urdu/Hindi**: Translate carefully. 'kis ko' means 'whom' (object). 'kon' means 'who' (subject). Preserve the direction of action (e.g. 'CTO kis ko fire kr skta hai' -> 'Who can the Chief Technology Officer terminate?').\n"
        "4. Resolve pronouns using History if available.\n"
        "5. Keep the language (English) for the final query to match document content.\n"
        "6. OUTPUT ONLY THE REWRITTEN QUERY. No quotes."
    )
    
    # Clean markdown from answer context for better rewriting
    clean_answer = (last_answer or '')
    clean_answer = _re.sub(r'\*\*', '', clean_answer)       # Bold markers
    clean_answer = _re.sub(r'\[(\d+)\]', '', clean_answer)  # Citation refs [1][2]
    clean_answer = _re.sub(r'^[-•]\s*', '', clean_answer, flags=_re.MULTILINE)  # Bullet points
    clean_answer = clean_answer.strip()

    user_prompt = (
        f"History: {last_question or 'None'}\n"
        f"Answer Context: {clean_answer[:300]}...\n"
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
        rewritten = response.choices[0].message.content.strip()

        # Hard-coded post-processing: ensure pay/salary queries include SPPP terms
        lower = rewritten.lower()
        pay_keywords = ['pay', 'salary', 'allowances', 'tankhwah', 'compensation', 'remuneration']
        number_keywords = ['number', 'numerical', 'kitni', 'kitna', 'amount', 'figure']
        if any(kw in lower for kw in pay_keywords):
            if 'sppp' not in lower:
                rewritten = rewritten + " SPPP Schedule"
            if any(kw in lower for kw in number_keywords) and 'minimum' not in lower:
                rewritten = rewritten + " minimum pay maximum pay PKR fuel limit"

        return rewritten
    except Exception as e:
        print(f"[Retriever] Rewrite failed: {e}")
        return current_query
