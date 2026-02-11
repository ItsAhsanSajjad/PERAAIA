"""
PERA AI Answerer (Brain 2.0)
"ChatGPT on our data" - Pure LLM Synthesis.
"""
from __future__ import annotations

import os
import re
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
MAX_DOCS = int(os.getenv("MAX_DOCS_FOR_PROMPT", "10"))
MAX_EVIDENCE_CHARS = int(os.getenv("MAX_EVIDENCE_CHARS", "28000"))

_client = None

def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    return _client

# =============================================================================
# Context Formatting
# =============================================================================
# =============================================================================
# Context Formatting & Ranking
# =============================================================================

def _get_ranked_chunks(retrieval: Dict[str, Any], question: str) -> List[Dict[str, Any]]:
    """
    Process retrieval results to produce a flat, ranked list of chunks.
    Shared logic for both LLM context and UI references to ensure alignment.
    """
    if not retrieval.get("has_evidence"):
        return []
        
    evidence_list = retrieval.get("evidence", [])
    
    # Extract subject keywords from question for relevance sorting
    q_lower = question.lower() if question else ""
    # Use the retriever's full expansion (abbreviations + schedule normalization)
    from retriever import _expand_abbreviations as _expand
    expanded_q = _expand(q_lower)
            
    # Subject words to check in chunk text (filter out generic query words)
    _stop = {"what", "which", "where", "when", "does", "that", "this", "with",
             "from", "about", "have", "been", "will", "shall", "their", "these",
             "salary", "scale", "detail", "full", "explain", "the", "for", "and", "how",
             "are", "is", "can", "may", "should"}
    _subject_words = [w for w in expanded_q.split() if len(w) > 2 and w not in _stop]
    
    ranked_chunks = []
    
    for doc_group in evidence_list:
        # Skip entire doc if max_score is below threshold
        if doc_group.get("max_score", 0) < ANSWER_MIN_TOP_SCORE:
            continue
            
        doc_name = doc_group.get("doc_name", "Unknown Document")
        hits = doc_group.get("hits", [])
        
        for hit in hits:
            # Skip low-score hits (UNLESS they are smart context expansion)
            is_context = hit.get("_is_smart_context", False)
            if not is_context and hit.get("score", 0) < HIT_MIN_SCORE:
                continue
                
            text = (hit.get("text") or "").strip()
            text_lower = text.lower()
            
            # Helper score: count subject matches
            subject_match = sum(1 for w in _subject_words if w in text_lower)
            
            # BOOST: Schedule header position — if query has a specific schedule number
            # (e.g. "schedule-iii"), boost chunks where it appears in the title/header
            import re as _re_ans
            sched_match = _re_ans.search(r'(schedule-[ivx]+)', expanded_q)
            if sched_match:
                sched_term = sched_match.group(1)
                # Check if schedule term appears in the first 80 chars (header position)
                if sched_term in text_lower[:80]:
                    subject_match += 3  # Strong boost for title match
            
            # BOOST: Salary table gets priority for salary queries
            # Broad keyword set including Urdu/typo variants
            _pay_query_words = {"salary", "salarey", "salaray", "pay", "allowance", "benefit",
                                "tankhwah", "tankha", "scale", "sppp", "compensation", "kitni",
                                "maash", "number", "amount", "ktna", "kitna", "kitne"}
            if any(pw in q_lower for pw in _pay_query_words):
                # Match BOTH header chunks AND data row chunks
                is_sppp_table = (
                    ("minimum pay" in text_lower and "maximum pay" in text_lower) or
                    ("sppp" in text_lower and re.search(r'\d{2,3},\d{3}', text))
                )
                if is_sppp_table:
                    subject_match += 5  # Ensure salary table ranks near top
            
            # BOOST: Name queries — push chunks with personal names to top
            _name_query_words = {"name", "nam", "naam", "kon", "kaun", "who"}
            if any(nw in q_lower.split() for nw in _name_query_words):
                # Chunks containing capitalized names like "Capt. Farrukh" or "Mr. Khan"
                if re.search(r'(?:Capt|Mr|Dr|Mrs|Ms|Col|Brig|Gen)\.?\s+[A-Z][a-z]', text):
                    subject_match += 3  # Push personal name chunks above role descriptions
            
            # DE-BOOST: Position descriptions for procedure/legal queries
            # "Position Title:" chunks mention arrest/powers/duties as job duties,
            # burying the actual Act sections with legal procedures
            _procedure_query = {"arrest", "happen", "procedure", "warrant", "magistrate",
                               "penalty", "offence", "comply", "hearing", "appeal",
                               "notice", "serve", "confiscat", "seiz", "search",
                               "encroach", "nuisance", "epo", "fir"}
            if any(pw in q_lower for pw in _procedure_query):
                if text.startswith("Position Title:") or "Purpose of the Position:" in text:
                    subject_match -= 2  # Push job descriptions below Act sections
                # BOOST chunks with actual section references
                if re.search(r'(?:section|sect\.|s\.) ?\d+', text_lower):
                    subject_match += 2
            
            ranked_chunks.append({
                "doc_name": doc_name,
                "text": text,
                "score": hit.get("score", 0),
                "page": hit.get("page_start", "?"),
                "public_path": hit.get("public_path", ""),
                "subject_match": subject_match,
                "_is_smart_context": is_context
            })
            
    # Global Sort:
    # 1. Subject match count (highest first)
    # 2. Score (highest first)
    # This mixes chunks from different docs if a "lower" doc has a better specific chunk
    ranked_chunks.sort(key=lambda x: (-x["subject_match"], -x["score"]))
    
    # KEYWORD INJECTION: For "name/who" queries, FAISS often misses name chunks
    # because names appear in meeting minutes with NO semantic overlap to "name of X"
    _name_query_words = {"name", "nam", "naam", "kon", "kaun", "who"}
    if any(nw in q_lower.split() for nw in _name_query_words):
        # Check if existing ranked chunks already have a good name match
        has_name = any(re.search(r'(?:Capt|Mr|Dr|Mrs|Ms|Col|Brig|Gen)\.?\s+[A-Z][a-z]', c.get("text","")) 
                      for c in ranked_chunks[:10])
        if not has_name:
            # Scan ALL chunks from the index for keyword matches
            from index_store import load_index_and_chunks
            _, all_chunks = load_index_and_chunks()
            
            # Find role-related subject words (e.g., "director", "general", "cto")
            role_words = [w for w in _subject_words if w not in _name_query_words]
            
            injected = []
            for ci, chunk in enumerate(all_chunks):
                text = chunk.get("text", "") or ""
                text_lower = text.lower()
                # Must contain the role keyword AND a proper name
                has_role = any(rw in text_lower for rw in role_words)
                has_proper_name = bool(re.search(r'(?:Capt|Mr|Dr|Mrs|Ms|Col|Brig|Gen)\.?\s+[A-Z][a-z]', text))
                if has_role and has_proper_name and len(text) > 50:
                    injected.append({
                        "doc_name": chunk.get("doc_name", "Unknown"),
                        "text": text,
                        "score": 0.600,  # Synthetic score 
                        "page": chunk.get("loc_start", "?"),
                        "public_path": chunk.get("public_path", ""),
                        "subject_match": 10,  # Very high to ensure top ranking
                        "_is_smart_context": False,
                        "_keyword_injected": True,
                    })
            
            if injected:
                # Take top 3 keyword-injected chunks, deduplicate
                injected = injected[:3]
                ranked_chunks = injected + ranked_chunks
    
    return ranked_chunks

def format_evidence_for_llm(ranked_chunks: List[Dict[str, Any]]) -> str:
    """Format ranked chunks into context string with [ID] citations."""
    context_parts = []
    total_chars = 0
    
    for i, chunk in enumerate(ranked_chunks):
        # Cap limits
        if i >= MAX_DOCS * 3: # Allow more chunks since they are hits not docs
            break
            
        ref_id = i + 1
        part = f"[{ref_id}] Source: {chunk['doc_name']} (Page {chunk['page']})\nContent: {chunk['text']}"
        part_len = len(part)
        
        if total_chars + part_len > MAX_EVIDENCE_CHARS:
            break
            
        context_parts.append(part)
        total_chars += part_len
        
    return "\n\n---\n\n".join(context_parts)

def extract_references_simple(ranked_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract reference objects matching the rankings.
    
    IMPORTANT: ref_id numbering MUST match format_evidence_for_llm's [1],[2],[3]...
    labeling exactly. Both iterate over ranked_chunks in the same order with the
    same MAX_DOCS*3 limit. No dedup here — dedup happens AFTER citation filtering.
    """
    refs = []
    base_url = os.getenv("BASE_URL", "https://ask.pera.gop.pk").rstrip("/")
    
    for i, chunk in enumerate(ranked_chunks):
        if i >= MAX_DOCS * 3:
            break
        
        ref_id = i + 1  # Sequential, matches format_evidence_for_llm [1],[2]...
        path = chunk["public_path"]
        page = chunk["page"]
        doc_name = chunk["doc_name"]
        
        url = f"{base_url}{path}#page={page}" if path else f"{base_url}/assets/data/{doc_name}#page={page}"
        
        refs.append({
            "id": ref_id,
            "document": doc_name,
            "page_start": page,
            "open_url": url,
            "snippet": chunk["text"][:200],
            "score": chunk["score"]
        })
        
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

def _is_greeting(question: str) -> bool:
    """Detect if the query is a simple greeting."""
    q = question.lower().strip()
    greetings = [
        "hi", "hello", "hey",
        "salam", "assalam o alaikum", "aoa",
        "greetings", "slam", "slm", "ssa"
    ]
    if q in greetings:
        return True
    # Check for short greetings like "hi there", "hello bot"
    for g in greetings:
        if q.startswith(g + " ") and len(q.split()) <= 4:
            return True
    return False

def _get_identity_answer(question: str) -> str | None:
    """Detect if the query is about the bot's name/identity and return appropriate answer.
    
    IMPORTANT: Must NOT trigger for role queries like 'CTO nam kia hai' or 'DG ka naam'.
    Only triggers when the query is specifically about the BOT's identity.
    """
    q = question.lower().strip()
    
    # If query mentions a PERA role, it's NOT about the bot — skip identity
    _role_keywords = {"cto", "dg", "director", "manager", "sdeo", "head", "officer",
                      "chairman", "chairperson", "secretary", "deputy", "assistant",
                      "monitoring", "legal", "finance", "admin", "hr", "procurement"}
    q_words = set(q.split())
    if q_words & _role_keywords:
        return None  # Query is about a PERA role's name, not the bot
    
    # Roman Urdu / Urdu phrases (only match if the query is ABOUT the bot)
    urdu_phrases = [
        "tera naam", "tumhara naam", "ap ka naam", "aap ka naam", 
        "tum kon", "aap kon", "tera nam", "tumhara nam",
    ]
    # Short standalone queries like just "nam kia hai" or "naam kya hai" (no role prefix)
    standalone_name_qs = ["nam kia hai", "naam kya hai", "name kia hai", "nam kya hai"]
    
    if any(p in q for p in urdu_phrases):
         return "Mera naam PERA AI Assistant hai. Main Punjab Enforcement and Regulatory Authority (PERA) ke regulations par madad karne ke liye yahan hoon. Aapko kis cheez ki madad chahiye?"

    # Only match standalone name questions if there's no subject prefix
    if q in standalone_name_qs or (len(q.split()) <= 4 and any(p in q for p in standalone_name_qs)):
         return "Mera naam PERA AI Assistant hai. Main Punjab Enforcement and Regulatory Authority (PERA) ke regulations par madad karne ke liye yahan hoon. Aapko kis cheez ki madad chahiye?"

    # English phrases
    if (("your name" in q) or ("who are you" in q)) and not (q_words & _role_keywords):
        return "I am the PERA AI Assistant, designed to help you with PERA regulations and acts. How can I assist you today?"
        
    return None

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
    
    # 0. Check for Greeting (only for first message - skip if in conversation)
    has_history = conversation_history and len(conversation_history) > 0
    if not has_history and _is_greeting(current_question):
        return {
            "answer": "Hello! I am the PERA AI Assistant. How can I help you with PERA regulations, acts, or notifications today?",
            "references": [],
            "decision": "answer"
        }
    
    # 0a. Check for Identity (Name) — skip if user is in an active conversation
    # (prevents "nam kia hai" follow-up about DG from being caught as bot identity q)
    if not has_history:
        identity_ans = _get_identity_answer(current_question)
        if identity_ans:
            return {
                "answer": identity_ans,
                "references": [],
                "decision": "answer"
            }
    client = get_client()
    
    # 0. Creator question intercept (deterministic, no LLM needed)
    if _is_creator_question(current_question):
        return {
            "answer": _CREATOR_RESPONSE,
            "references": [],
            "decision": "answer"
        }
    
    # 1. Build Ranked Context
    ranked_chunks = _get_ranked_chunks(retrieval, current_question)
    context_str = format_evidence_for_llm(ranked_chunks)
    
    if not context_str:
        return {
            "answer": "I'm sorry, I couldn't find any information about that in the PERA documents.",
            "references": [],
            "decision": "refuse"
        }

    # 2. Define System Persona
    first_directive = (
        "You are the PERA AI Assistant, an expert on Punjab Enforcement and Regulatory Authority (PERA) regulations.\n"
        "Your PRIMARY goal is to answer the user's question helpfully and accurately using the provided Context.\n\n"
        "Directives:\n"
        "1. **ANSWER FIRST**: Always try your best to answer from the Context. If the user asks about 'powers' and the Context has 'responsibilities' or 'duties' for that role — that IS the answer, present it. Never refuse when related information exists.\n"
        "2. **Citations**: Cite sources using [1], [2] notation matching context chunk numbers. Place citations at end of relevant sentences.\n"
        "3. **Truthfulness**: Answer ONLY from the Context. Do NOT fabricate, guess, or invent information not present in the Context. If asked 'what is X' and the Context does not mention X at all, you MUST refuse. Never guess meanings of abbreviations.\n"
        "4. **Conflict Resolution**: If documents conflict, prioritize the 'PERA Act'. EPO timeline is 15 days.\n"
        "5. **Role Specificity**: For specific role queries, treat 'powers', 'responsibilities', 'duties', and 'kaam' as equivalent. List what the documents say about that role.\n"
        "6. **Confidentiality**: Whistle-blower identity must NOT be disclosed without written consent.\n"
        "7. **Persona**: Be professional, helpful, concise. Use bullet points.\n"
        "8. **Language**: Reply in the **SAME language** as user's query (English, Urdu, or Roman Urdu). Never use Hindi script.\n"
        "9. **Formatting**: Use Markdown. Bold key terms.\n"
        "10. **Pay Cross-Reference**: When asked about a role's pay in numbers, cross-reference between chunks (role SPPP level + SPPP salary table).\n"
        "11. **Identity (Developer ONLY)**: ONLY if asked who CREATED/BUILT/DEVELOPED this AI (words: creator, developer, banaya, father, Ahsan), say it was developed by **Muhammad Ahsan Sajjad**. He is NOT a PERA employee. Do NOT confuse him with CTO or any PERA role. For PERA role names, answer ONLY from Context.\n"
        "12. **Salary Table Reading**: For SPPP salaries, use 'Minimum Pay' and 'Maximum Pay' columns from the SPPP Breakup table.\n"
        "13. **Refusal**: If the Context does NOT contain information relevant to the user's question, say: "
        "'Yeh information PERA documents mein available nahi hai. Mazeed madad ke liye PERA se contact karein.'\n\n"
        "Context:\n"
        f"{context_str}"
    )
    system_prompt = first_directive

    # 3. Construct Messages
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add recent history (last 2 turns) for conversation flow
    if conversation_history:
        # Filter for valid roles
        valid_history = [m for m in conversation_history if m.get("role") in ("user", "assistant")]
        # Skip poisoned history: if most recent assistant messages are refusals,
        # the LLM will pattern-follow and refuse new queries too
        _refusal_markers = ["available nahi hai", "digital mind is a little tired", 
                           "couldn't find any information", "PERA se contact karein"]
        recent = valid_history[-4:]
        assistant_msgs = [m for m in recent if m.get("role") == "assistant"]
        refusal_count = sum(1 for m in assistant_msgs 
                          if any(rm in (m.get("content") or "").lower() for rm in _refusal_markers))
        # Only include history if it's mostly successful answers (not refusals)
        if refusal_count < len(assistant_msgs):
            messages.extend(recent)
    
    # Add current question — use expanded form so LLM can match abbreviations to context
    from retriever import _expand_abbreviations
    user_q = _expand_abbreviations(current_question)
    messages.append({"role": "user", "content": user_q})

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
            "not found in the provided",
            "i couldn't find",
            "i could not find",
            "no information available",
            "not mentioned in the context",
            "my digital mind is a little tired",
            "available nahi hai",
            "documents mein available nahi",
            "information is not available",
        ]

        # Check for Identity Answer (suppress references but keep as answer)
        if "muhammad ahsan sajjad" in lower_ans:
             return {
                "answer": answer_text,
                "references": [],
                "decision": "answer"
            }

        if any(phrase in lower_ans for phrase in _NO_INFO_PHRASES):
            return {
                "answer": answer_text,
                "references": [],
                "decision": "refuse"
            }

        # Filter references: only include chunks the LLM actually cited
        all_refs = extract_references_simple(ranked_chunks)
        import re as _re_ans
        cited_ids = set(int(m) for m in _re_ans.findall(r'\[(\d+)\]', answer_text))
        
        if cited_ids:
            filtered_refs = [r for r in all_refs if r.get("id") in cited_ids]
        else:
            # LLM didn't cite anything explicitly — show top 3 as fallback
            filtered_refs = all_refs[:3]
        
        # Dedup: remove duplicate page references (keep first occurrence)
        seen_pages = set()
        deduped_refs = []
        for ref in filtered_refs:
            page_key = f"{ref['document']}_{ref['page_start']}"
            if page_key not in seen_pages:
                seen_pages.add(page_key)
                deduped_refs.append(ref)
        # Reassign clean sequential IDs for frontend display
        for idx, ref in enumerate(deduped_refs):
            ref["id"] = idx + 1
        
        return {
            "answer": answer_text,
            "references": deduped_refs,
            "decision": "answer"
        }
        
    except Exception as e:
        print(f"[Answerer] LLM call failed: {e}")
        return {
            "answer": "I encountered an error while processing your request. Please try again.",
            "references": [],
            "decision": "error"
        }
