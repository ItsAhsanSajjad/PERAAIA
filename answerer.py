"""
PERA AI Answerer (Brain 2.0)
"ChatGPT on our data" - Pure LLM Synthesis.
"""
from __future__ import annotations

import os
from typing import List, Dict, Any

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gpt-4o-mini")

# Evidence quality thresholds (from .env)
ANSWER_MIN_TOP_SCORE = float(os.getenv("ANSWER_MIN_TOP_SCORE", "0.28"))
HIT_MIN_SCORE = float(os.getenv("HIT_MIN_SCORE", "0.26"))
MAX_HITS_PER_DOC = int(os.getenv("MAX_HITS_PER_DOC_FOR_PROMPT", "15"))
MAX_DOCS = int(os.getenv("MAX_DOCS_FOR_PROMPT", "6"))
MAX_EVIDENCE_CHARS = int(os.getenv("MAX_EVIDENCE_CHARS", "24000"))

_client = None

def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    return _client

# =============================================================================
# Context Formatting
# =============================================================================
def format_evidence_for_llm(retrieval: Dict[str, Any], question: str = "") -> str:
    """
    Format retrieved chunks into a clean context block.
    Applies score filtering and caps to prevent context overflow.
    Sorts hits by relevance to the query subject to avoid important chunks being cut off.
    """
    if not retrieval.get("has_evidence"):
        return ""
    
    evidence_list = retrieval.get("evidence", [])
    context_parts = []
    total_chars = 0
    
    # Extract subject keywords from question for relevance sorting
    q_lower = question.lower() if question else ""
    # Expand known abbreviations for better subject matching
    _ABBREV = {"cto": "chief technology officer", "dg": "director general", 
               "hr": "human resources", "it": "information technology",
               "adg": "additional director general", "eo": "enforcement officer"}
    expanded_q = q_lower
    for abbr, full in _ABBREV.items():
        if abbr in q_lower.split():
            expanded_q = expanded_q.replace(abbr, full)
    # Subject words to check in chunk text (filter out generic query words)
    _stop = {"what", "which", "where", "when", "does", "that", "this", "with",
             "from", "about", "have", "been", "will", "shall", "their", "these",
             "salary", "scale", "detail", "full", "explain", "the", "for", "and", "how"}
    _subject_words = [w for w in expanded_q.split() if len(w) > 2 and w not in _stop]
    
    docs_used = 0
    for doc_group in evidence_list:
        # Skip entire doc if max_score is below threshold
        if doc_group.get("max_score", 0) < ANSWER_MIN_TOP_SCORE:
            continue
        
        if docs_used >= MAX_DOCS:
            break
        
        doc_name = doc_group.get("doc_name", "Unknown Document")
        hits = doc_group.get("hits", [])
        
        # Sort hits: chunks containing subject keywords FIRST, then by score
        def _hit_relevance(h):
            text_lower = (h.get("text") or "").lower()
            # Count how many subject words appear in the chunk
            subject_match = sum(1 for w in _subject_words if w in text_lower)
            score = h.get("score", 0)
            # Primary sort: subject match count (descending)
            # Secondary sort: score (descending)
            return (-subject_match, -score)
        
        sorted_hits = sorted(hits, key=_hit_relevance)
        
        hits_used = 0
        for hit in sorted_hits:
            # Skip low-score hits (UNLESS they are smart context expansion)
            is_context = hit.get("_is_smart_context", False)
            if not is_context and hit.get("score", 0) < HIT_MIN_SCORE:
                continue
            
            if hits_used >= MAX_HITS_PER_DOC:
                break
                
            text = (hit.get("text") or "").strip()
            page = hit.get("page_start", "?")
            
            part = f"Source: {doc_name} (Page {page})\nContent: {text}"
            part_len = len(part)
            
            # Cap total evidence chars
            if total_chars + part_len > MAX_EVIDENCE_CHARS:
                break
            
            context_parts.append(part)
            total_chars += part_len
            hits_used += 1
        
        if hits_used > 0:
            docs_used += 1
        
        if total_chars >= MAX_EVIDENCE_CHARS:
            break


    return "\n\n---\n\n".join(context_parts)

def extract_references_simple(retrieval: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract reference links for the UI.
    Only includes docs/hits that pass score thresholds (same as LLM context).
    """
    refs = []
    seen = set()
    base_url = os.getenv("BASE_URL", "https://ask.pera.gop.pk").rstrip("/")
    
    evidence_list = retrieval.get("evidence", [])
    docs_used = 0
    for doc_group in evidence_list:
        # Skip docs below quality threshold
        if doc_group.get("max_score", 0) < ANSWER_MIN_TOP_SCORE:
            continue
        
        if docs_used >= MAX_DOCS:
            break
        
        doc_name = doc_group.get("doc_name", "Document")
        hits_added = 0
        
        for hit in doc_group.get("hits", []):
            # Skip low-score hits (allow smart context through)
            is_context = hit.get("_is_smart_context", False)
            if not is_context and hit.get("score", 0) < HIT_MIN_SCORE:
                continue
            
            if hits_added >= 2:  # Max 2 refs per doc
                break
            
            page = hit.get("page_start", 1)
            path = hit.get("public_path", "")
            text = (hit.get("text") or "")[:200]
            
            # Key for deduplication
            key = f"{doc_name}_{page}"
            if key in seen:
                continue
            seen.add(key)
            
            url = f"{base_url}{path}#page={page}" if path else f"{base_url}/assets/data/{doc_name}#page={page}"
            
            refs.append({
                "document": doc_name,
                "page_start": page,
                "open_url": url,
                "snippet": text,
            })
            hits_added += 1
        
        if hits_added > 0:
            docs_used += 1
            
    return refs

# =============================================================================
# Creator Question Detection (Code-level, not LLM-dependent)
# =============================================================================
_CREATOR_RESPONSE = "I was developed by **Muhammad Ahsan Sajjad**, Lead AI under the supervision of the CTO of PERA."

def _is_creator_question(question: str) -> bool:
    """Detect if user asks about the chatbot's creator (not PERA the org)."""
    q = question.lower()
    # Must contain a specific 'who made' phrase (not just 'developer' which is a job title)
    maker_phrases = [
        "kisne banaya", "kis ne banaya", "kisnyu bnaya", "kisny bnaya",
        "who made", "who created", "who developed", "who built",
        "tumhe banaya", "tumhe bnaya", "aapko banaya", "aapko bnaya",
        "ye banaya", "yeh banaya", "is ko banaya",
        "developed by whom", "created by whom", "made by whom",
    ]
    has_maker = any(phrase in q for phrase in maker_phrases)
    if not has_maker:
        return False
    # If references PERA the org -> NOT a creator question
    if "pera" in q and not any(w in q for w in ["pera ai", "pera bot", "pera chatbot", "pera assistant"]):
        return False
    # If references 'you/bot/AI' -> definitely creator question
    bot_words = ["you", "tum", "aap", "tumhe", "aapko", "bot", "chatbot",
                 "ye", "yeh", "is"]
    if any(w in q.split() for w in bot_words):
        return True
    # Generic 'kisne banaya' without context -> assume about the bot
    return True

# =============================================================================
# Main Answer Function (System Prompt Driven)
# =============================================================================
def answer_question(
    current_question: str,
    retrieval: Dict[str, Any],
    conversation_history: List[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Generate answer using System Prompt + Context.
    No manual regex rules.
    """
    client = get_client()
    
    # 0. Creator question intercept (deterministic, no LLM needed)
    if _is_creator_question(current_question):
        return {
            "answer": _CREATOR_RESPONSE,
            "references": [],
            "decision": "answer"
        }
    
    # 1. Build Context
    context_str = format_evidence_for_llm(retrieval, question=current_question)
    if not context_str:
        return {
            "answer": "I'm sorry, I couldn't find any information about that in the PERA documents.",
            "references": [],
            "decision": "refuse"
        }

    # 2. Define System Persona
    system_prompt = (
        "You are the PERA AI Assistant, an expert on Punjab Economic Research Institute (PERA) regulations.\n"
        "Your goal is to answer the user's question accurately based ONLY on the provided Context.\n\n"
        "Directives:\n"
        "1. **Truthfulness**: Answer ONLY from the Context.\n"
        "2. If the user asks about \"powers\", \"functions\", or \"duties\", treat these as synonymous unless a specific distinction is required. If statutory powers are not found, describe the \"Areas of Responsibilities\".\n"
        "3. **Inference**: If specific powers (like firing/termination) are not explicitly stated for a role, look for generic \"Competent Authority\" rules or Service Rules in the context and infer based on the role's seniority (e.g. Head of Department).\n"
        "4. If the answer is still not found, state that specific details are not available but general rules may apply.\n"
        "5. **Persona**: Be professional, helpful, and concise. Use bullet points for lists (like powers, duties).\n"
        "6. **Language**: Reply in the same language as the user (English, Urdu, or Roman Urdu).\n"
        "7. **Formatting**: Use Markdown. Bold key terms.\n\n"
        "Context:\n"
        f"{context_str}"
    )

    # 3. Construct Messages
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add recent history (last 2 turns) for conversation flow
    if conversation_history:
        # Filter for valid roles
        valid_history = [m for m in conversation_history if m.get("role") in ("user", "assistant")]
        messages.extend(valid_history[-4:])
    
    # Add current question
    messages.append({"role": "user", "content": current_question})

    # 4. Call LLM
    try:
        response = client.chat.completions.create(
            model=ANSWER_MODEL,
            messages=messages,
            temperature=0.3, # Low temp for factual accuracy
        )
        answer_text = response.choices[0].message.content
        
        # Post-processing checks
        lower_ans = answer_text.lower()

        # Suppress references when LLM says info is not available
        _NO_INFO_PHRASES = [
            "not available in the provided context",
            "not explicitly mentioned",
            "not found in the provided",
            "i couldn't find",
            "i could not find",
            "no information available",
            "specific details are not available",
            "not mentioned in the context",
        ]
        if any(phrase in lower_ans for phrase in _NO_INFO_PHRASES):
            return {
                "answer": answer_text,
                "references": [],
                "decision": "refuse"
            }

        return {
            "answer": answer_text,
            "references": extract_references_simple(retrieval),
            "decision": "answer"
        }
        
    except Exception as e:
        print(f"[Answerer] LLM call failed: {e}")
        return {
            "answer": "I encountered an error while processing your request. Please try again.",
            "references": [],
            "decision": "error"
        }
