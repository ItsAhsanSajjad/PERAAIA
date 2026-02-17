"""
PERA AI Answerer (Brain 2.0)
"ChatGPT on our data" - Pure LLM Synthesis.
"""
from __future__ import annotations

import os
import re
from typing import List, Dict, Any, Optional

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

_client: Optional[OpenAI] = None


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

    IMPORTANT CHANGE:
    - Avoid "Source:" / "Content:" labels (models tend to copy them into the answer).
    - Wrap evidence in tags to discourage copying into final answer.
    """
    if not retrieval.get("has_evidence"):
        return ""

    evidence_list = retrieval.get("evidence", [])
    context_parts: List[str] = []
    total_chars = 0

    # Extract subject keywords from question for relevance sorting
    q_lower = question.lower() if question else ""
    _ABBREV = {
        "cto": "chief technology officer",
        "dg": "director general",
        "hr": "human resources",
        "it": "information technology",
        "adg": "additional director general",
        "eo": "enforcement officer",
    }

    expanded_q = q_lower
    for abbr, full in _ABBREV.items():
        if abbr in q_lower.split():
            expanded_q = expanded_q.replace(abbr, full)

    _stop = {
        "what", "which", "where", "when", "does", "that", "this", "with",
        "from", "about", "have", "been", "will", "shall", "their", "these",
        "salary", "scale", "detail", "full", "explain", "the", "for", "and", "how"
    }
    _subject_words = [w for w in expanded_q.split() if len(w) > 2 and w not in _stop]

    docs_used = 0
    for doc_group in evidence_list:
        if doc_group.get("max_score", 0) < ANSWER_MIN_TOP_SCORE:
            continue
        if docs_used >= MAX_DOCS:
            break

        doc_name = (doc_group.get("doc_name", "Unknown Document") or "Unknown Document").strip()
        hits = doc_group.get("hits", [])

        def _hit_relevance(h: Dict[str, Any]):
            text_lower = (h.get("text") or "").lower()
            subject_match = sum(1 for w in _subject_words if w in text_lower)
            score = h.get("score", 0)
            return (-subject_match, -score)

        sorted_hits = sorted(hits, key=_hit_relevance)

        hits_used = 0
        for hit in sorted_hits:
            is_context = hit.get("_is_smart_context", False)
            if not is_context and hit.get("score", 0) < HIT_MIN_SCORE:
                continue
            if hits_used >= MAX_HITS_PER_DOC:
                break

            text = (hit.get("text") or "").strip()
            page = hit.get("page_start", "?")

            # Evidence block (tagged)
            # NOTE: keep doc/page for grounding, but tags discourage copying.
            safe_doc = doc_name.replace("<", "").replace(">", "").replace('"', "").replace("'", "")
            part = (
                f"<evidence doc=\"{safe_doc}\" page=\"{page}\">\n"
                f"{text}\n"
                f"</evidence>"
            )

            if total_chars + len(part) > MAX_EVIDENCE_CHARS:
                break

            context_parts.append(part)
            total_chars += len(part)
            hits_used += 1

        if hits_used > 0:
            docs_used += 1

        if total_chars >= MAX_EVIDENCE_CHARS:
            break

    return "\n\n".join(context_parts)


def extract_references_simple(retrieval: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract reference links for the UI.
    Only includes docs/hits that pass score thresholds (same as LLM context).
    """
    refs: List[Dict[str, Any]] = []
    seen = set()
    base_url = os.getenv("BASE_URL", "https://ask.pera.gop.pk").rstrip("/")

    evidence_list = retrieval.get("evidence", [])
    docs_used = 0
    for doc_group in evidence_list:
        if doc_group.get("max_score", 0) < ANSWER_MIN_TOP_SCORE:
            continue
        if docs_used >= MAX_DOCS:
            break

        doc_name = doc_group.get("doc_name", "Document")
        hits_added = 0

        for hit in doc_group.get("hits", []):
            is_context = hit.get("_is_smart_context", False)
            if not is_context and hit.get("score", 0) < HIT_MIN_SCORE:
                continue
            if hits_added >= 2:
                break

            page = hit.get("page_start", 1)
            path = hit.get("public_path", "")
            text = (hit.get("text") or "")[:200]

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
    q = question.lower()
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
    if "pera" in q and not any(w in q for w in ["pera ai", "pera bot", "pera chatbot", "pera assistant"]):
        return False
    return True


# =============================================================================
# NEW: Detect if user explicitly asked for sources/pages/links
# =============================================================================
def _user_wants_references(question: str) -> bool:
    q = (question or "").lower()
    triggers = [
        "source", "sources", "reference", "references", "citation", "citations",
        "document", "pdf", "file", "link", "open url",
        "page", "page number", "kis page", "konse page", "konsi file", "konsa document",
        "hawala", "ref", "proof",
    ]
    return any(t in q for t in triggers)


# =============================================================================
# Strip references inside model answer (because UI shows references separately)
# =============================================================================
# inline blocks: (Source: ...), [Source: ...], (References: ...), etc.
_INLINE_CITATION_BLOCK_RE = re.compile(
    r"(\(|\[)\s*(sources?|references?|citations?)\s*:\s*.*?(\)|\])",
    re.IGNORECASE | re.DOTALL
)

# lines like: "Source: ...", "- Source: ...", "* References: ...", "• Citations: ..."
_CITATION_LINE_RE = re.compile(
    r"^\s*([-*•]\s*)?(sources?|references?|citations?)\s*:\s*.*$",
    re.IGNORECASE | re.MULTILINE
)

# remove whole trailing section starting with headings like "### References" / "## Sources" / "**Citations**"
_TRAILING_REF_SECTION_RE = re.compile(
    r"(?is)\n\s*(#{1,6}\s*)?(\*\*)?\s*(sources?|references?|citations?)\s*(\*\*)?\s*:?\s*\n.*$"
)

# remove link-ish lines if model prints them
_LINKISH_LINE_RE = re.compile(
    r"^\s*([-*•]\s*)?.*(https?://|/assets/|#page=|open_url)\S.*$",
    re.IGNORECASE | re.MULTILINE
)


def _strip_answer_references(answer_text: str) -> str:
    if not answer_text:
        return answer_text

    txt = answer_text

    # Remove entire trailing "References/Sources/Citations" section
    txt = re.sub(_TRAILING_REF_SECTION_RE, "", txt)

    # Remove inline citation blocks
    txt = re.sub(_INLINE_CITATION_BLOCK_RE, "", txt)

    # Remove citation lines
    txt = re.sub(_CITATION_LINE_RE, "", txt)

    # Remove link-ish lines
    txt = re.sub(_LINKISH_LINE_RE, "", txt)

    # Cleanup
    txt = re.sub(r"[ \t]+\n", "\n", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    txt = re.sub(r"[ \t]{2,}", " ", txt)

    return txt.strip()


# =============================================================================
# Main Answer Function
# =============================================================================
def answer_question(
    current_question: str,
    retrieval: Dict[str, Any],
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    client = get_client()

    # 0. Creator question intercept
    if _is_creator_question(current_question):
        return {"answer": _CREATOR_RESPONSE, "references": [], "decision": "answer"}

    # 1. Build Context
    context_str = format_evidence_for_llm(retrieval, question=current_question)
    if not context_str:
        return {
            "answer": "I'm sorry, I couldn't find any information about that in the PERA documents.",
            "references": [],
            "decision": "refuse"
        }

    # 2. System Persona (STRICT: never write references in answer)
    system_prompt = (
        "You are the PERA AI Assistant for Punjab Enforcement and Regulatory Authority (PERA). "
        "You operate in a high-stakes government environment where credibility and clarity are essential.\n\n"

        "NON-NEGOTIABLE RULES\n"
        "1) Answer using ONLY the provided Context. Do not use external knowledge.\n"
        "2) Do not invent facts (powers, procedures, numbers, dates, thresholds, authorities).\n"
        "3) Do NOT infer authority from seniority or job title unless explicitly stated in Context.\n\n"

        "IMPORTANT: DO NOT WRITE REFERENCES IN THE ANSWER TEXT\n"
        "- The UI will show references separately below the answer.\n"
        "- Do NOT output: 'Source:', 'Sources:', 'References:', 'Citations:', document names, page numbers, links, or footnotes.\n"
        "- Do NOT add any 'References' section at the end.\n"
        "- Only mention document/page if the USER explicitly asks for it.\n\n"

        "INTERPRETATION\n"
        "4) Treat 'powers', 'functions', and 'duties' as synonyms unless the Context explicitly distinguishes them.\n"
        "5) For authority questions (termination, sealing, fines, arrest, delegation): look for explicit role authority, "
        "'Competent Authority'/delegation clauses, or the procedure. If not explicit, do not infer.\n"
        "6) If the Context contains conflicting statements, present both neutrally.\n\n"

        "REPUTATION-SAFE FALLBACK\n"
        "7) If the exact answer is not explicitly defined, do NOT say only 'not found.' Use this structure:\n"
        "   A) 'The provided PERA official documents do not explicitly define <X>.'\n"
        "   B) Give 2–5 closest relevant points grounded in the Context.\n"
        "   C) Ask 1–2 targeted clarification questions.\n"
        "   D) Suggest a refined query phrasing using PERA terms.\n\n"

        "STYLE\n"
        "8) Be professional, composed, and concise. Use Markdown headings and bullet points.\n"
        "9) Reply in the same language as the user (English, Urdu, Roman Urdu).\n\n"

        "CONTEXT (do not quote or reproduce tags):\n"
        f"{context_str}"
    )

    # 3. Construct Messages
    messages = [{"role": "system", "content": system_prompt}]

    if conversation_history:
        valid_history = [m for m in conversation_history if m.get("role") in ("user", "assistant")]
        messages.extend(valid_history[-4:])

    messages.append({"role": "user", "content": current_question})

    # 4. Call LLM
    try:
        response = client.chat.completions.create(
            model=ANSWER_MODEL,
            messages=messages,
            temperature=0.3,
        )
        answer_text = response.choices[0].message.content or ""

        # ✅ Strip references ONLY if user did NOT explicitly ask for them
        if not _user_wants_references(current_question):
            answer_text = _strip_answer_references(answer_text)

        lower_ans = answer_text.lower()
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
            return {"answer": answer_text, "references": [], "decision": "refuse"}

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
