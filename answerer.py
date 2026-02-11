"""
PERA AI Answerer (Brain 2.0 + Intent Router + Evidence Gate + Citation Contract)
"ChatGPT on our data" - Pure LLM Synthesis with strict relevance enforcement.
"""
from __future__ import annotations

# Deterministic registry bypass — guaranteed answers for known patterns
try:
    from deterministic_registry import (
        registry_answer, route_intent as dr_route_intent,
        INTENT_FALLBACK as DR_FALLBACK,
    )
    _HAS_REGISTRY = True
except ImportError:
    _HAS_REGISTRY = False
    print("[Answerer] WARNING: deterministic_registry not available")

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

# Minimum keyword overlap for evidence relevance gate
EVIDENCE_MIN_KEYWORD_OVERLAP = int(os.getenv("EVIDENCE_MIN_KEYWORD_OVERLAP", "1"))

_client = None

def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    return _client

# =============================================================================
# A) INTENT + ENTITY ROUTER
# =============================================================================

class QueryIntent:
    ORG_OVERVIEW = "ORG_OVERVIEW"
    JOB_ROLE = "JOB_ROLE"
    ENFORCEMENT_LEGAL = "ENFORCEMENT_LEGAL"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    GREETING_SMALLTALK = "GREETING_SMALLTALK"  # greetings, ty, tum kon ho
    DOC_LOOKUP = "DOC_LOOKUP"  # schedule, SPPP, annex, salary table, what is PERA
    STRICT_AUTHORITY = "STRICT_AUTHORITY"  # terminate, fire, dismiss, suspend
    GENERAL_PERA = "GENERAL_PERA"  # fallback for in-scope questions

# Compensation type enum
class CompensationType:
    SPPP = "SPPP"
    BPS = "BPS"
    NOT_SPECIFIED = "NotSpecified"

# --- Intent patterns (lowercased) ---

_ORG_OVERVIEW_PATTERNS = [
    r"\bwhat is pera\b", r"\bpera kya hai\b", r"\bpera kia hai\b",
    r"\bpera kya h\b", r"\bpera kia h\b",
    r"\bpera kya karti\b", r"\bpera kia karti\b",
    r"\bpera kya kam karti\b", r"\bpera kia kam karti\b",
    r"\bgoal of pera\b", r"\bpurpose of pera\b", r"\bmandate of pera\b",
    r"\bpera ka maqsad\b", r"\bpera ka purpose\b",
    r"\bpowers of pera\b", r"\bfunctions of pera\b",
    r"\bduties of pera\b", r"\brole of pera\b",
    r"\bpera authority\b", r"\bwhat does pera do\b",
    r"\babout pera\b", r"\bpera kaam\b", r"\bpera kam\b",
    r"\bpera established\b", r"\bpera objective\b",
    r"\bpera overview\b", r"\bpera introduction\b",
]

_JOB_ROLE_KEYWORDS = [
    "cto", "chief technology officer",
    "sso", "system support officer", "senior staff officer",
    "manager development", "manager infrastructure",
    "director general", "additional director general",
    "enforcement officer", "sub divisional enforcement officer",
    "head office", "qualification", "salary", "pay scale",
    "job description", "jd", "designation", "grade",
    "eligibility", "recruitment", "selection criteria",
]

_ENFORCEMENT_LEGAL_PATTERNS = [
    r"\beo powers\b", r"\bsdeo powers\b", r"\bsub.?divisional\b",
    r"\bsearch\b.*\bseal\b", r"\bseal\b.*\bsearch\b",
    r"\barrest\b", r"\bepo\b", r"\bpublic nuisance\b",
    r"\bfir\b", r"\bappeal\b", r"\binjunction\b", r"\bstay\b",
    r"\bpenalty\b", r"\bprosecutio\b", r"\bsection\s+\d+",
    r"\bact\b.*\bpera\b", r"\bpera\b.*\bact\b",
    r"\bbill\b.*\bpera\b", r"\bo\s*&\s*p\s+code\b",
    r"\benforcement\b.*\bpower\b", r"\bpower\b.*\benforcement\b",
    r"\bregulatory\b.*\bpower\b",
]

# DOC_LOOKUP: schedule, SPPP, annex, salary table, pay scale, what is PERA
_DOC_LOOKUP_PATTERNS = [
    r"\bschedule\b", r"\bsehdule\b", r"\bshedule\b",
    r"\bappendix\b", r"\bannex\b",
    r"\bsppp\b", r"\bsppp[\s-]*\d", r"\bpay\s*package\b",
    r"\bsalary\s*table\b", r"\bpay\s*scale\b", r"\bpayscale\b",
    r"\bwhat\s+is\s+pera\b", r"\bpera\s+kya\s+hai\b", r"\bpera\s+kia\s+hai\b",
    r"\bpera\s+kya\s+h\b", r"\bpera\s+kia\s+h\b",
    r"\bpurpose\s+of\s+pera\b", r"\bmandate\s+of\s+pera\b",
    r"\bgoal\s+of\s+pera\b", r"\babout\s+pera\b",
    r"\bpera\s+ka\s+maqsad\b", r"\bpera\s+established\b",
]

# STRICT_AUTHORITY: terminate, fire, dismiss, suspend, appointment
_STRICT_AUTHORITY_PATTERNS = [
    r"\bterminate\b", r"\btermination\b",
    r"\bfire\b.*\bcan\b", r"\bcan\b.*\bfire\b",
    r"\bdismiss\b", r"\bdismissal\b",
    r"\bsuspend\b", r"\bsuspension\b",
    r"\bappoint(?:ment)?\b.*\bauthority\b", r"\bauthority\b.*\bappoint",
    r"\bcompetent\s+authority\b",
    r"\bnikaal\b", r"\bnikaln\b", r"\bnikal\b",  # Urdu: fire/remove
    r"\bbarakhs?ast\b",  # Urdu: dismiss
]

_OUT_OF_SCOPE_PATTERNS = [
    # General tech
    r"\bhtml\b", r"\bcss\b", r"\bjavascript\b", r"\bpython\b",
    r"\bkotlin\b", r"\bjava\b", r"\breact\b", r"\bvue\b",
    r"\bc\+\+\b", r"\brust\b", r"\bswift\b", r"\bflutter\b",
    r"\bmachine learning\b", r"\bdeep learning\b",
    r"\bblockchain\b", r"\bcryptocurrency\b", r"\bbitcoin\b",
    # Cooking / personal
    r"\brecipe\b", r"\bcooking\b", r"\bkhana\b", r"\bbiryani\b",
    r"\bcake\b", r"\bfood\b",
    # Other laws not in docs
    r"\bogra\b", r"\bppc\b", r"\bpenal code\b",
    r"\bnepra\b", r"\bsecp\b",
    # Interpersonal / funny / personal
    r"\bfunny\b", r"\bmazaq\b", r"\bjoke\b", r"\bhaha\b",
    r"\bhansi\b", r"\blatifa\b", r"\bmeme\b",
    r"\bpyar\b", r"\blove\b", r"\bgirlfriend\b", r"\bboyfriend\b",
    r"\bweather\b", r"\bcricket\b", r"\bfootball\b",
]

# Interpersonal "funny baat" patterns (strict refusal even if a PERA role is mentioned)
_INTERPERSONAL_PATTERNS = [
    r"\bfunny\s*(baat|bat|talk)\b", r"\bmazaq\b", r"\bjoke\b",
    r"\bhansi\b", r"\blatifa\b", r"\bmeme\b",
    r"\bpyar\b", r"\blove\b", r"\bgupshup\b",
    r"\bfriend\b.*\bbana\b", r"\bdosti\b",
]

# Greeting / smalltalk / identity patterns -> GREETING_SMALLTALK intent
_GREETING_PATTERNS = [
    r"^(hi|hello|hey|hy)\b",
    r"^(aoa|a\.o\.a|assalam|assalamu|asalam|salam|salaam|slm)\b",
    r"^(good\s*(morning|afternoon|evening|night))\b",
    r"^(kya\s+haal|kaise\s+ho|kaisay\s+ho|kesy\s+ho|kese\s+ho)\b",
    r"^(how\s+are\s+you)\b",
]

_SMALLTALK_ONLY_PATTERNS = [
    # Thanks / bye / identity questions
    r"^(thanks|thank\s*you|shukriya|jazakallah|jazak\s*allah)$",
    r"^(ty|thx|thnx|thnks|thankx)$",
    r"^(ok|okay|theek|thik|sahi|acha|achha|accha)$",
    r"^(bye|alvida|khuda\s*hafiz|phir\s*milte)$",
    # Identity questions about the bot
    r"\btum\s*kon\s*ho\b", r"\baap\s*kon\s*ho\b", r"\btum\s*kya\s*ho\b",
    r"\baap\s*kya\s*ho\b", r"\bwho\s*are\s*you\b",
    r"\bwhat\s*are\s*you\b", r"\byou\s*are\s*who\b",
    r"\bapna\s*naam\s*batao\b", r"\btera\s*naam\b", r"\bwhat.*your.*name\b",
]

# SSO ambiguity detection
_SSO_AMBIGUITY_RE = re.compile(
    r"\bsso\b", re.IGNORECASE
)

# ORG_OVERVIEW: required keyword families in chunks
_ORG_OVERVIEW_CHUNK_KEYWORDS = [
    "authority", "established", "functions", "objective",
    "mandate", "purpose", "enforcement", "regulatory",
    "punjab enforcement", "pera", "government",
]

# ENFORCEMENT Tier1: Act/Bill/O&P Code doc name patterns
_ENFORCEMENT_TIER1_PATTERNS = [
    r"\bact\b", r"\bbill\b", r"\bo\s*&?\s*p\b", r"\bcode\b",
    r"\bordinance\b", r"\bregulation\b",
    r"\bstanding\s*committee\b",  # Bill_Recommended_by_Standing_Committee
]

# Docs explicitly BLOCKED for ENFORCEMENT_LEGAL intent
_ENFORCEMENT_BLOCKED_DOC_PATTERNS = [
    r"human.?resource", r"hr.?manual", r"dress.?code",
    r"annex.?h\b", r"job.?description", r"weapon",
    r"squad", r"leave.?polic", r"staff.?regul",
    r"compiled.?working.?paper",  # HR/admin sections
]


def classify_intent(question: str) -> str:
    """
    Classify query intent into one of the defined categories.
    Returns the intent string constant.
    """
    q = question.lower().strip()
    q_clean = re.sub(r'[^\w\s]', ' ', q)  # strip punctuation for matching

    # 0. Check GREETING/SMALLTALK first (ty, tum kon ho, hello, etc.)
    for pat in _SMALLTALK_ONLY_PATTERNS:
        if re.search(pat, q_clean):
            return QueryIntent.GREETING_SMALLTALK
    for pat in _GREETING_PATTERNS:
        if re.search(pat, q_clean):
            # Check if it's greeting-only (no real question after)
            # Strip the greeting prefix and check if anything meaningful remains
            remaining = re.sub(pat, '', q_clean).strip()
            if not remaining or len(remaining) < 4:
                return QueryIntent.GREETING_SMALLTALK

    # 1. Check interpersonal first (e.g., "cto funny baat") -> always OUT_OF_SCOPE
    for pat in _INTERPERSONAL_PATTERNS:
        if re.search(pat, q):
            return QueryIntent.OUT_OF_SCOPE

    # 2. Check OUT_OF_SCOPE tech/personal topics
    for pat in _OUT_OF_SCOPE_PATTERNS:
        if re.search(pat, q):
            return QueryIntent.OUT_OF_SCOPE

    # 3. DOC_LOOKUP (schedule, SPPP, annex, what is PERA) - BEFORE ORG_OVERVIEW
    for pat in _DOC_LOOKUP_PATTERNS:
        if re.search(pat, q):
            return QueryIntent.DOC_LOOKUP

    # 4. Check ORG_OVERVIEW (broader org questions not caught by DOC_LOOKUP)
    for pat in _ORG_OVERVIEW_PATTERNS:
        if re.search(pat, q):
            return QueryIntent.DOC_LOOKUP  # Route through DOC_LOOKUP for permissive retrieval

    # 5. STRICT_AUTHORITY (terminate, fire, dismiss) - BEFORE ENFORCEMENT_LEGAL
    for pat in _STRICT_AUTHORITY_PATTERNS:
        if re.search(pat, q):
            return QueryIntent.STRICT_AUTHORITY

    # 6. Check ENFORCEMENT_LEGAL
    for pat in _ENFORCEMENT_LEGAL_PATTERNS:
        if re.search(pat, q):
            return QueryIntent.ENFORCEMENT_LEGAL

    # 7. Check JOB_ROLE (keyword presence)
    for kw in _JOB_ROLE_KEYWORDS:
        if kw in q_clean:
            return QueryIntent.JOB_ROLE

    # 8. Default: GENERAL_PERA (let retrieval decide)
    return QueryIntent.GENERAL_PERA


def detect_sso_ambiguity(question: str) -> bool:
    """Check if question mentions 'SSO' which is ambiguous."""
    q = question.lower()
    if not _SSO_AMBIGUITY_RE.search(q):
        return False
    # If already disambiguated, skip
    if "system support" in q or "senior staff" in q or "senior sergeant" in q:
        return False
    # Ambiguous SSO mention
    return True


# =============================================================================
# B) EVIDENCE RELEVANCE GATE
# =============================================================================

# Abbreviation expansions for relevance checking
_RELEVANCE_ABBREV = {
    "cto": ["chief", "technology", "officer"],
    "dg": ["director", "general"],
    "hr": ["human", "resources"],
    "it": ["information", "technology"],
    "adg": ["additional", "director", "general"],
    "eo": ["enforcement", "officer"],
    "sdeo": ["sub", "divisional", "enforcement", "officer"],
    "sso": ["senior", "staff", "officer", "system", "support"],
    "pera": ["punjab", "enforcement", "regulatory", "authority"],
}

_RELEVANCE_STOP = {
    "what", "which", "where", "when", "does", "that", "this", "with",
    "from", "about", "have", "been", "will", "shall", "their", "these",
    "the", "for", "and", "how", "who", "are", "is", "was", "can",
    "kya", "hai", "kon", "kaun", "ki", "ka", "ke", "se", "ko", "ne",
    "ye", "yeh", "kia", "hain", "mein", "par", "say", "bhi",
    "explain", "detail", "full", "tell", "batao", "bataen",
}


def _extract_query_keywords(question: str) -> List[str]:
    """Extract meaningful keywords from query, expanding abbreviations."""
    q = re.sub(r'[^\w\s]', ' ', question.lower())
    words = [w for w in q.split() if len(w) > 1 and w not in _RELEVANCE_STOP]

    # Expand abbreviations
    expanded = []
    for w in words:
        if w in _RELEVANCE_ABBREV:
            expanded.extend(_RELEVANCE_ABBREV[w])
        else:
            expanded.append(w)

    return list(set(expanded))


# =============================================================================
# B2) ENTITY PINNING: extract canonical job title + filter by entity
# =============================================================================

# Canonical map: lowered alias/abbreviation -> canonical title
_CANONICAL_TITLES = {
    # CTO
    "cto": "Chief Technology Officer",
    "chief technology officer": "Chief Technology Officer",
    # Manager (Development)
    "manager development": "Manager (Development)",
    "manager (development)": "Manager (Development)",
    "manager dev": "Manager (Development)",
    # System Support Officer
    "system support officer": "System Support Officer",
    "sso": "System Support Officer",
    # Director General
    "dg": "Director General",
    "director general": "Director General",
    # Enforcement Officer
    "eo": "Enforcement Officer",
    "enforcement officer": "Enforcement Officer",
    # Sub Divisional Enforcement Officer
    "sdeo": "Sub Divisional Enforcement Officer",
    "sub divisional enforcement officer": "Sub Divisional Enforcement Officer",
    # Manager Infrastructure & Networks
    "manager infrastructure": "Manager (Infrastructure & Networks)",
    "manager (infrastructure & networks)": "Manager (Infrastructure & Networks)",
    # Assistant Manager GIS
    "assistant manager gis": "Assistant Manager GIS",
    "assistant manager (gis)": "Assistant Manager GIS",
    # Deputy Manager HR
    "deputy manager hr": "Deputy Manager HR",
    "deputy manager (hr)": "Deputy Manager HR",
    # Web & Software Developer
    "web developer": "Web & Software Developer",
    "software developer": "Web & Software Developer",
    "web & software developer": "Web & Software Developer",
    # Android Developer
    "android developer": "Android Developer",
    # Manager Research & Implementation
    "manager research": "Manager (Research & Implementation)",
    # Deputy Manager Training
    "deputy manager training": "Deputy Manager Training",
    # Assistant Manager Network
    "assistant manager network": "Assistant Manager (Network)",
    # Assistant Manager IT Infrastructure
    "assistant manager it": "Assistant Manager (IT Infrastructure)",
    # Director Finance
    "director finance": "Director Finance",
    "dir finance": "Director Finance",
    # Director Admin & HR
    "director admin": "Director Admin & HR",
    "dir admin": "Director Admin & HR",
}

# Sorted longest-first for greedy matching
_TITLE_KEYS_SORTED = sorted(_CANONICAL_TITLES.keys(), key=len, reverse=True)


def extract_job_title(question: str) -> Optional[str]:
    """
    Extract canonical job title from a query.
    Returns the canonical title string or None.
    Uses longest-match-first to avoid partial matches.
    """
    q_lower = re.sub(r'[^a-z0-9&() ]', ' ', question.lower())
    q_lower = re.sub(r'\s+', ' ', q_lower).strip()

    for alias in _TITLE_KEYS_SORTED:
        if alias in q_lower:
            return _CANONICAL_TITLES[alias]
    return None


def filter_evidence_by_entity(
    retrieval: Dict[str, Any],
    job_title: str,
) -> Dict[str, Any]:
    """
    Filter retrieval evidence to keep only chunks that mention the exact job title.
    This prevents entity confusion (e.g., Manager Development != Assistant Manager GIS).
    """
    if not retrieval.get("has_evidence") or not job_title:
        return retrieval

    title_lower = job_title.lower()
    # Also prepare alternate forms
    alternates = {title_lower}
    # "Manager (Development)" -> also match "Manager Development" and vice versa
    if "(" in job_title:
        no_parens = re.sub(r'[()]', '', job_title).strip()
        alternates.add(no_parens.lower())
    else:
        # Try with parens
        parts = job_title.split()
        if len(parts) >= 2:
            alternates.add(f"{parts[0]} ({' '.join(parts[1:])})".lower())

    filtered_evidence = []
    for doc_group in retrieval.get("evidence", []):
        filtered_hits = []
        for hit in doc_group.get("hits", []):
            text_lower = (hit.get("text") or "").lower()
            if any(alt in text_lower for alt in alternates):
                filtered_hits.append(hit)
        if filtered_hits:
            group_copy = dict(doc_group)
            group_copy["hits"] = filtered_hits
            group_copy["max_score"] = max(h.get("score", 0) for h in filtered_hits)
            filtered_evidence.append(group_copy)

    if filtered_evidence:
        return {
            "question": retrieval.get("question", ""),
            "has_evidence": True,
            "evidence": filtered_evidence,
            "_entity_filter": job_title,
        }
    else:
        # No chunks matched the entity -> return empty (will trigger refusal)
        return {
            "question": retrieval.get("question", ""),
            "has_evidence": False,
            "evidence": [],
            "_entity_filter": job_title,
            "_gate_reason": f"No chunks contain entity '{job_title}'",
        }

# Doc-name patterns for irrelevant docs (should not be primary for ORG_OVERVIEW)
_IRRELEVANT_DOC_FOR_OVERVIEW = [
    r"human.?resource", r"hr.?manual", r"dress.?code",
    r"squad", r"weapon",
    r"annex", r"service.?regul", r"leave.?polic",
    r"staff.?regul", r"recruitment",
]

# Chunk-text blacklist: even if a chunk is in a relevant doc, skip it for ORG_OVERVIEW
# if it is about dress code, leave, attendance, etc.
_CHUNK_BLACKLIST_FOR_OVERVIEW = [
    "dress code", "dresscode", "working hours",
    "leave policy", "casual leave", "annual leave",
    "attendance", "latecoming", "late coming",
    "hair style", "grooming", "uniform",
]


def evidence_relevance_gate(
    retrieval: Dict[str, Any],
    question: str,
    intent: str,
    min_overlap: int = None,
) -> Dict[str, Any]:
    """
    Filter retrieval results to keep only chunks that share meaningful
    keyword overlap with the query. Returns a new retrieval dict.

    For ORG_OVERVIEW intent, also require that top chunks contain
    org-definition keywords (Authority, established, functions, etc.).
    """
    if min_overlap is None:
        min_overlap = EVIDENCE_MIN_KEYWORD_OVERLAP

    # ORG_OVERVIEW: stricter (2+ keywords)
    if intent == QueryIntent.ORG_OVERVIEW:
        min_overlap = max(min_overlap, 2)

    # DOC_LOOKUP: very permissive (allow everything with any overlap)
    if intent == QueryIntent.DOC_LOOKUP:
        min_overlap = 0  # accept all chunks that pass threshold

    if not retrieval.get("has_evidence"):
        return retrieval

    q_keywords = _extract_query_keywords(question)
    if not q_keywords:
        return retrieval  # can't filter without keywords

    evidence_list = retrieval.get("evidence", [])
    filtered_evidence = []

    for doc_group in evidence_list:
        doc_name = doc_group.get("doc_name", "")
        doc_name_lower = doc_name.lower()
        hits = doc_group.get("hits", [])

        # For ORG_OVERVIEW: skip docs that are clearly irrelevant
        # (HR manual, dress code, squads & weapons, etc.)
        if intent == QueryIntent.ORG_OVERVIEW:
            is_irrelevant_doc = any(
                re.search(pat, doc_name_lower) for pat in _IRRELEVANT_DOC_FOR_OVERVIEW
            )
            if is_irrelevant_doc:
                continue  # skip entire doc group

        # For ENFORCEMENT_LEGAL intent: HARD-FILTER to Tier1 only
        if intent == QueryIntent.ENFORCEMENT_LEGAL:
            # Block explicitly irrelevant docs
            is_blocked = any(
                re.search(pat, doc_name_lower)
                for pat in _ENFORCEMENT_BLOCKED_DOC_PATTERNS
            )
            if is_blocked:
                continue  # skip entire doc group

            # Must be a Tier1 doc (Act, Bill, O&P Code, etc.)
            is_tier1 = any(
                re.search(pat, doc_name_lower)
                for pat in _ENFORCEMENT_TIER1_PATTERNS
            )
            if not is_tier1:
                continue  # skip non-Tier1 docs

        # For STRICT_AUTHORITY: allow Tier1 + service rules/HR regs, block others
        if intent == QueryIntent.STRICT_AUTHORITY:
            is_blocked = any(
                re.search(pat, doc_name_lower)
                for pat in [r"dress.?code", r"weapon", r"squad", r"leave.?polic"]
            )
            if is_blocked:
                continue

        # For DOC_LOOKUP: NO doc-level filtering (very permissive)
        # Allow all docs including FAQ, establishment, notifications, schedules

        relevant_hits = []
        for hit in hits:
            text_lower = (hit.get("text") or "").lower()

            # Count keyword overlap
            overlap = sum(1 for kw in q_keywords if kw in text_lower)

            # For smart context chunks, be more lenient
            if hit.get("_is_smart_context", False):
                relevant_hits.append(hit)
                continue

            # For ORG_OVERVIEW, additionally require org-definition keywords in chunk
            # AND skip blacklisted chunks (dress code, leave, etc.)
            if intent == QueryIntent.ORG_OVERVIEW:
                # Blacklist check: skip chunks about dress code etc.
                if any(bl in text_lower for bl in _CHUNK_BLACKLIST_FOR_OVERVIEW):
                    continue

                org_keyword_count = sum(
                    1 for kw in _ORG_OVERVIEW_CHUNK_KEYWORDS if kw in text_lower
                )
                # Must have: (a) query keyword overlap AND (b) at least 1 org keyword
                if overlap >= min_overlap and org_keyword_count >= 1:
                    relevant_hits.append(hit)
            else:
                # Require minimum keyword overlap
                if overlap >= min_overlap:
                    relevant_hits.append(hit)

        if relevant_hits:
            filtered_doc = {
                "doc_name": doc_name,
                "max_score": doc_group.get("max_score", 0),
                "hits": relevant_hits,
            }
            filtered_evidence.append(filtered_doc)

    # For ORG_OVERVIEW: verify at least some top chunks contain definition keywords
    if intent == QueryIntent.ORG_OVERVIEW and filtered_evidence:
        has_org_chunk = False
        for doc_group in filtered_evidence[:3]:  # check top 3 docs
            for hit in doc_group.get("hits", [])[:5]:
                text_lower = (hit.get("text") or "").lower()
                org_overlap = sum(
                    1 for kw in _ORG_OVERVIEW_CHUNK_KEYWORDS if kw in text_lower
                )
                if org_overlap >= 2:
                    has_org_chunk = True
                    break
            if has_org_chunk:
                break

        if not has_org_chunk:
            # Top chunks are irrelevant (e.g., dress code, weapons) -> NO_EVIDENCE
            return {
                "question": retrieval.get("question", question),
                "has_evidence": False,
                "evidence": [],
                "_gate_reason": "ORG_OVERVIEW chunks failed org-keyword check",
            }

    # Sort filtered evidence by max_score descending
    filtered_evidence.sort(key=lambda x: x.get("max_score", 0), reverse=True)

    has_evidence = len(filtered_evidence) > 0
    return {
        "question": retrieval.get("question", question),
        "has_evidence": has_evidence,
        "evidence": filtered_evidence,
        "_gate_reason": None if has_evidence else "No chunks passed relevance gate",
    }


# =============================================================================
# Context Formatting
# =============================================================================
def format_evidence_for_llm(retrieval: Dict[str, Any], question: str = "", intent: str = "") -> str:
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

    # For DOC_LOOKUP: lower the min score threshold to include more chunks
    doc_min_top = 0.08 if intent == QueryIntent.DOC_LOOKUP else ANSWER_MIN_TOP_SCORE
    doc_hit_min = 0.06 if intent == QueryIntent.DOC_LOOKUP else HIT_MIN_SCORE

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
        if doc_group.get("max_score", 0) < doc_min_top:
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
            if not is_context and hit.get("score", 0) < doc_hit_min:
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

def extract_references_simple(retrieval: Dict[str, Any], intent: str = "") -> List[Dict[str, Any]]:
    """
    Extract reference links for the UI.
    Only includes docs/hits that pass score thresholds (same as LLM context).
    """
    refs = []
    seen = set()
    base_url = os.getenv("BASE_URL", "https://ask.pera.gop.pk").rstrip("/")

    evidence_list = retrieval.get("evidence", [])
    docs_used = 0
    # Use lower threshold for DOC_LOOKUP intent
    ref_min_top = 0.08 if intent == QueryIntent.DOC_LOOKUP else ANSWER_MIN_TOP_SCORE
    ref_hit_min = 0.06 if intent == QueryIntent.DOC_LOOKUP else HIT_MIN_SCORE
    for doc_group in evidence_list:
        # Skip docs below quality threshold
        if doc_group.get("max_score", 0) < ref_min_top:
            continue

        if docs_used >= MAX_DOCS:
            break

        doc_name = doc_group.get("doc_name", "Document")
        hits_added = 0

        for hit in doc_group.get("hits", []):
            # Skip low-score hits (allow smart context through)
            is_context = hit.get("_is_smart_context", False)
            if not is_context and hit.get("score", 0) < ref_hit_min:
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
# C) STRICT MODE LEAK FIX
# =============================================================================

_STRICT_REFUSAL = (
    "I'm sorry, this question is outside my scope. I can only answer questions "
    "related to PERA (Punjab Enforcement and Regulatory Authority) based on "
    "official PERA documents. Please ask a PERA-related question."
)

_STRICT_REFUSAL_URDU = (
    "Maazrat, yeh sawal mere daayre se bahar hai. Main sirf PERA "
    "(Punjab Enforcement and Regulatory Authority) ke official documents "
    "ke mutabiq sawalat ka jawab de sakta hoon."
)

# Hedge phrases to ban from strict answers - HARD BLOCKED
_BANNED_HEDGE_PHRASES = [
    "general rules",
    "typically",
    "depends",
    "work culture",
    "may apply",
    "general workplace etiquette",
    "general rules may apply",
    "depends on the organization",
    "possible that",
    "typically in organizations",
    "generally speaking",
    "it is possible",
    "could potentially",
    "in most organizations",
    "common practice",
    "standard practice",
    "usually in such cases",
    "please refer to guidelines",
    "please refer to the relevant",
    "please refer to",
    "refer to the official",
    "consult the relevant",
]

_SSO_CLARIFICATION = (
    "The abbreviation **SSO** can refer to multiple roles:\n\n"
    "1. **Senior Staff Officer (SSO)**\n"
    "2. **System Support Officer**\n\n"
    "Could you please clarify which role you're asking about? "
    "I'll provide the correct information from the PERA documents."
)


def _detect_roman_urdu(question: str) -> bool:
    """Simple heuristic to detect Roman Urdu queries."""
    q = question.lower()
    markers = ["kya", "hai", "kaise", "kaisay", "hain", "batao", "bataen",
               "kaun", "kon", "kia", "yeh", "ye", "ap", "aap", "kr", "kar"]
    return sum(1 for m in markers if m in q.split()) >= 1


def _post_process_strict(answer_text: str, intent: str, question: str) -> tuple:
    """
    Post-process LLM answer to catch and fix strict mode leaks.
    Returns (None, False) if answer is OK.
    Returns (replacement_str, False) if answer should be forcefully replaced.
    Returns (None, True) if answer has ban phrases and should be REGENERATED.
    """
    lower_ans = answer_text.lower()

    # Ban hedge phrases for any intent
    for phrase in _BANNED_HEDGE_PHRASES:
        if phrase in lower_ans:
            # The LLM is hedging -> flag for regeneration or refusal
            return (None, True)  # signal: needs regeneration

    return (None, False)  # answer is OK


def _verify_section_citations(answer_text: str, evidence_snippets: List[str]) -> Optional[str]:
    """
    For ENFORCEMENT_LEGAL answers: if the answer mentions 'Section X',
    verify that 'Section X' appears verbatim in at least one evidence snippet.
    Returns None if OK, or a warning message if section is unverified.
    """
    # Find all "Section \d+" references in the answer
    section_refs = re.findall(r'[Ss]ection\s+(\d+[A-Za-z]?)', answer_text)
    if not section_refs:
        return None  # no section references to verify

    # Combine all evidence snippets into one searchable text
    combined_evidence = " ".join(s.lower() for s in evidence_snippets)

    unverified_sections = []
    for sec_num in section_refs:
        # Check if "section <num>" appears in evidence
        pattern = rf'section\s+{re.escape(sec_num)}\b'
        if not re.search(pattern, combined_evidence):
            unverified_sections.append(sec_num)

    if unverified_sections:
        secs = ", ".join(f"Section {s}" for s in unverified_sections)
        return (
            f"I found a reference to {secs} but could not verify it in the "
            "official PERA documents. Please refer to the original Act/Bill "
            "for accurate section references."
        )

    return None  # all sections verified


def detect_compensation_type(evidence_snippets: List[str]) -> str:
    """
    Detect compensation type from retrieved evidence snippets.
    Returns CompensationType.SPPP, CompensationType.BPS, or CompensationType.NOT_SPECIFIED.
    """
    combined = " ".join(s.lower() for s in evidence_snippets)

    has_sppp = bool(re.search(r'\bsppp[\s\-]*\d', combined))
    has_bps = bool(re.search(r'\bbps[\s\-]*\d', combined))

    if has_sppp and not has_bps:
        return CompensationType.SPPP
    elif has_bps and not has_sppp:
        return CompensationType.BPS
    elif has_sppp and has_bps:
        # Both present -> prefer what the JD chunk says
        # Check if SPPP appears in a JD/Annex context
        return CompensationType.SPPP  # SPPP takes precedence per user spec
    return CompensationType.NOT_SPECIFIED


def extract_compensation_grade(evidence_snippets: List[str], comp_type: str) -> Optional[str]:
    """
    Extract the specific grade (e.g., 'SPPP-1', 'BPS-14') from evidence.
    """
    combined = " ".join(evidence_snippets)

    if comp_type == CompensationType.SPPP:
        match = re.search(r'\b(SPPP[\s\-]*(\d+))', combined, re.IGNORECASE)
        if match:
            return f"SPPP-{match.group(2)}"
    elif comp_type == CompensationType.BPS:
        match = re.search(r'\b(BPS[\s\-]*(\d+))', combined, re.IGNORECASE)
        if match:
            return f"BPS-{match.group(2)}"

    return None


# Greeting responses
_GREETING_RESPONSE_EN = (
    "Hello! I am the PERA AI Assistant. How can I help you "
    "with PERA-related questions?"
)
_GREETING_RESPONSE_UR = (
    "السلام علیکم! میں PERA AI Assistant ہوں۔ آپ PERA سے متعلق "
    "کوئی بھی سوال پوچھ سکتے ہیں۔"
)
_GREETING_RESPONSE_ROMAN = (
    "Wa Alaikum Assalam! Main PERA AI Assistant hoon. "
    "Aap PERA se mutaliq koi bhi sawal pooch sakte hain."
)
_BOT_IDENTITY_RESPONSE = (
    "I am the **PERA AI Assistant**, developed to answer questions about the "
    "Punjab Enforcement and Regulatory Authority (PERA) based on official documents. "
    "How can I help you?"
)


# =============================================================================
# QUOTE LOCALIZATION ENGINE
# =============================================================================
import unicodedata
import difflib

LOCATE_MIN_SIMILARITY = float(os.getenv("LOCATE_MIN_SIMILARITY", "0.72"))
LOCATE_MAX_CANDIDATES = int(os.getenv("LOCATE_MAX_CANDIDATES", "20"))
LOCATE_WINDOW_MAX_CHARS = 800


def normalize_with_map(text: str):
    """
    Normalize text for matching and build a mapping from normalized indices
    back to original indices.

    Returns (normalized_text: str, norm_to_orig: List[int])
    where norm_to_orig[i] = index in original `text` that produced char i.
    """
    # Step 1: NFKC normalization — track index mapping
    nfkc = unicodedata.normalize("NFKC", text)
    # Build rough mapping: NFKC can change string length.
    # We re-derive from original by walking both strings.
    # For our purposes, we build a pipeline of transforms and compose maps.

    # Transform pipeline applied character by character on the original:
    # 1. NFKC  2. remove soft hyphens  3. de-hyphenate "-\n"  4. lowercase
    # 5. collapse whitespace

    norm_chars: List[str] = []
    norm_to_orig: List[int] = []

    i = 0
    prev_was_space = False
    while i < len(text):
        ch = text[i]

        # De-hyphenate: "-\n" or "-\r\n" → skip both, continue
        if ch == "-" and i + 1 < len(text) and text[i + 1] in "\r\n":
            skip = 2
            if i + 2 < len(text) and text[i + 1] == "\r" and text[i + 2] == "\n":
                skip = 3
            i += skip
            prev_was_space = False
            continue

        # Soft hyphen → skip
        if ch == "\u00AD":
            i += 1
            continue

        # NFKC normalize single char
        nfkc_ch = unicodedata.normalize("NFKC", ch)
        for nc in nfkc_ch:
            lc = nc.lower()
            # Collapse whitespace
            if lc in " \t\n\r\x0b\x0c":
                if not prev_was_space:
                    norm_chars.append(" ")
                    norm_to_orig.append(i)
                    prev_was_space = True
            else:
                norm_chars.append(lc)
                norm_to_orig.append(i)
                prev_was_space = False
        i += 1

    # Strip leading/trailing space
    normalized = "".join(norm_chars).strip()
    # Adjust norm_to_orig for leading strip
    lead_strip = len("".join(norm_chars)) - len("".join(norm_chars).lstrip())
    if lead_strip > 0:
        norm_to_orig = norm_to_orig[lead_strip:]
    # Trim to match normalized length
    norm_to_orig = norm_to_orig[: len(normalized)]

    return normalized, norm_to_orig


def _normalize_needle(needle: str) -> str:
    """Normalize needle the same way (no map needed)."""
    n, _ = normalize_with_map(needle)
    return n


def locate_quote(
    chunk_text: str,
    needle: str,
    max_quote_words: int = 25,
) -> Optional[tuple]:
    """
    Find needle in chunk_text using normalized fuzzy matching.
    Returns (quote: str, start_char: int, end_char: int) or None.
    quote == chunk_text[start_char:end_char] is GUARANTEED.
    """
    if not chunk_text or not needle:
        return None

    norm_text, n2o = normalize_with_map(chunk_text)
    norm_needle = _normalize_needle(needle)

    if not norm_needle or not norm_text:
        return None

    match_a = None  # start index in normalized text
    match_b = None  # end index (exclusive) in normalized text
    match_ratio = 0.0

    # --- Strategy 1: exact substring match on normalized ---
    pos = norm_text.find(norm_needle)
    if pos != -1:
        match_a = pos
        match_b = pos + len(norm_needle)
        match_ratio = 1.0
    else:
        # --- Strategy 2: token-anchor + windowed SequenceMatcher ---
        needle_tokens = norm_needle.split()
        if not needle_tokens:
            return None

        # Find all occurrences of first token to anchor candidate windows
        first_tok = needle_tokens[0]
        candidates = []
        search_start = 0
        while search_start < len(norm_text):
            idx = norm_text.find(first_tok, search_start)
            if idx == -1:
                break
            candidates.append(idx)
            search_start = idx + 1
            if len(candidates) >= LOCATE_MAX_CANDIDATES:
                break

        # Also try last token as anchor for robustness
        if len(needle_tokens) > 1:
            last_tok = needle_tokens[-1]
            search_start = 0
            while search_start < len(norm_text):
                idx = norm_text.find(last_tok, search_start)
                if idx == -1:
                    break
                # Window start would be roughly idx - len(norm_needle)
                anchor = max(0, idx - len(norm_needle) - 20)
                if anchor not in candidates:
                    candidates.append(anchor)
                search_start = idx + 1
                if len(candidates) >= LOCATE_MAX_CANDIDATES * 2:
                    break

        best_ratio = 0.0
        best_a = 0
        best_b = 0
        window_size = min(len(norm_needle) * 3, LOCATE_WINDOW_MAX_CHARS)

        for anchor in candidates:
            win_start = max(0, anchor - 10)
            win_end = min(len(norm_text), win_start + window_size)
            window = norm_text[win_start:win_end]

            sm = difflib.SequenceMatcher(None, norm_needle, window, autojunk=False)
            # Find longest contiguous match
            blocks = sm.get_matching_blocks()
            if not blocks:
                continue

            # Get the overall ratio for the needle against this window
            ratio = sm.ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                # Find the tightest span that covers the match
                # Use find_longest_match for best alignment
                m = sm.find_longest_match(0, len(norm_needle), 0, len(window))
                if m.size > 0:
                    # Expand to cover full needle length approximately
                    seg_start = win_start + max(0, m.b - m.a)
                    seg_end = min(len(norm_text), seg_start + len(norm_needle))
                    best_a = seg_start
                    best_b = seg_end

        if best_ratio >= LOCATE_MIN_SIMILARITY and best_b > best_a:
            match_a = best_a
            match_b = best_b
            match_ratio = best_ratio

    if match_a is None or match_b is None:
        return None

    # --- Map back to original text indices ---
    if match_a >= len(n2o) or match_b - 1 >= len(n2o):
        # Safety: clamp
        match_a = min(match_a, len(n2o) - 1)
        match_b = min(match_b, len(n2o))

    orig_start = n2o[match_a]
    orig_end = n2o[match_b - 1] + 1

    # Expand to word boundaries in original text
    while orig_start > 0 and chunk_text[orig_start - 1] not in " \t\n\r":
        orig_start -= 1
    while orig_end < len(chunk_text) and chunk_text[orig_end] not in " \t\n\r":
        orig_end += 1

    quote = chunk_text[orig_start:orig_end].strip()

    # Trim to max_quote_words
    words = quote.split()
    if len(words) > max_quote_words:
        quote = " ".join(words[:max_quote_words])
        # Recalculate orig_end to match trimmed quote
        orig_end = orig_start + chunk_text[orig_start:].find(quote) + len(quote)
        if orig_end <= orig_start:
            orig_end = orig_start + len(quote)

    # Final adjustment: find exact position of trimmed quote in chunk_text
    exact_pos = chunk_text.find(quote, max(0, orig_start - 5))
    if exact_pos != -1:
        orig_start = exact_pos
        orig_end = exact_pos + len(quote)

    # ENFORCE: chunk_text[start:end] == quote
    if chunk_text[orig_start:orig_end] != quote:
        # Last resort: try finding quote anywhere
        fallback = chunk_text.find(quote)
        if fallback != -1:
            orig_start = fallback
            orig_end = fallback + len(quote)
        else:
            return None

    return (quote, orig_start, orig_end)


# =============================================================================
# DETERMINISTIC TABLE ROW EXTRACTION (Schedule-I/III)
# =============================================================================

def extract_table_row(query: str, table_text: str, max_quote_words: int = 25):
    """
    Row-aware extraction from Schedule-I/III table text.

    Returns list of claims: [{"quote": str, "start": int, "end": int, "summary": str}]
    """
    q_lower = query.lower()
    lines = table_text.split("\n")
    results = []

    # Detect row key from query
    # SPPP grades: SPPP-1 through SPPP-10
    sppp_match = re.search(r'sppp[\s\-]*(\d+)', q_lower)
    # BPS grades: BPS-17 etc
    bps_match = re.search(r'bps[\s\-]*(\d+)', q_lower)
    # Offence section: section 1, section 2
    section_match = re.search(r'section[\s\-]*(\d+)', q_lower)

    target_patterns = []
    if sppp_match:
        grade = sppp_match.group(1)
        target_patterns = [f"sppp-{grade}", f"sppp {grade}", f"sppp-0{grade}" if len(grade) == 1 else ""]
    elif bps_match:
        grade = bps_match.group(1)
        target_patterns = [f"bps-{grade}", f"bps {grade}"]
    elif section_match:
        sec = section_match.group(1)
        target_patterns = [f"section {sec}", f"s.{sec}", f"({sec})"]

    if target_patterns:
        # Find matching rows
        for li, line in enumerate(lines):
            line_lower = line.lower().strip()
            if not line_lower:
                continue
            if any(p and p in line_lower for p in target_patterns):
                # Take this line + next line for context (tables often span 2 lines)
                row_text = line.strip()
                if li + 1 < len(lines) and lines[li + 1].strip():
                    next_line = lines[li + 1].strip()
                    # Only add next line if it's a continuation (doesn't start a new row key)
                    if not re.match(r'^(sppp|bps|section|\d+\.)', next_line.lower()):
                        row_text += " " + next_line

                # Trim to max words
                words = row_text.split()
                if len(words) > max_quote_words:
                    row_text = " ".join(words[:max_quote_words])

                # Find offsets in original text
                pos = table_text.find(line.strip())
                if pos != -1:
                    results.append({
                        "quote": row_text,
                        "start": pos,
                        "end": pos + len(row_text),
                        "summary": f"Row matching {target_patterns[0]}",
                    })
    else:
        # Generic schedule query (e.g., "schedule iii") → header + first 2 data rows
        header_line = None
        data_rows = []
        for li, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            if header_line is None:
                header_line = stripped
                pos = table_text.find(stripped)
                results.append({
                    "quote": stripped[:150],  # headers can be long
                    "start": pos if pos != -1 else 0,
                    "end": (pos + len(stripped[:150])) if pos != -1 else len(stripped[:150]),
                    "summary": "Table header",
                })
            elif len(data_rows) < 2:
                data_rows.append(stripped)
                pos = table_text.find(stripped)
                words = stripped.split()
                row_text = " ".join(words[:max_quote_words])
                results.append({
                    "quote": row_text,
                    "start": pos if pos != -1 else 0,
                    "end": (pos + len(row_text)) if pos != -1 else len(row_text),
                    "summary": f"Table row {len(data_rows)}",
                })
            else:
                break

    return results


def _extract_table_claims(
    hits: List[Dict[str, Any]], query: str
) -> List[Dict[str, Any]]:
    """
    Deterministic table extraction: bypass LLM for table-tier hits.
    Uses extract_table_row for row-aware extraction.
    """
    claims = []
    for h in hits:
        if h.get("tier") != "table":
            continue
        text = h.get("text", "")
        if not text:
            continue

        rows = extract_table_row(query, text)
        for r in rows:
            claims.append({
                "answer_span": r["summary"],
                "doc": h["doc"],
                "page": h["page"],
                "chunk_id": h["chunk_id"],
                "quote": r["quote"],
                "quote_offsets": (r["start"], r["end"]),
                "_source": "deterministic_table",
            })

    return claims


# =============================================================================
# DETERMINISTIC JD SALARY EXTRACTION (multi-chunk safe)
# =============================================================================

_JD_SALARY_HEADERS = [
    "salary and benefits", "salary & benefits", "salary/benefits",
    "pay and benefits", "compensation", "salary:",
]

def _extract_jd_salary_claims(
    hits: List[Dict[str, Any]], job_title: str
) -> List[Dict[str, Any]]:
    """
    Deterministic JD salary extraction.
    Finds job title match, then searches for salary section in same chunk
    or ±2 neighboring chunks.
    """
    if not job_title:
        return []

    title_lower = job_title.lower()
    # Sort hits by chunk_id for neighbor search
    sorted_hits = sorted(hits, key=lambda h: h.get("chunk_id", 0))
    chunk_map = {h["chunk_id"]: h for h in sorted_hits}
    chunk_ids = [h["chunk_id"] for h in sorted_hits]

    # Find chunk(s) containing the job title
    title_chunks = []
    for h in sorted_hits:
        if title_lower in h.get("text", "").lower():
            title_chunks.append(h)

    if not title_chunks:
        return []

    claims = []
    for tc in title_chunks:
        text = tc.get("text", "")
        text_lower = text.lower()

        # Check if salary section is in this chunk
        salary_found = False
        for header in _JD_SALARY_HEADERS:
            hpos = text_lower.find(header)
            if hpos != -1:
                salary_found = True
                # Extract from header to end of section or next section
                section_start = hpos
                section_text = text[section_start:]
                # Try to limit to just the salary section
                # Look for next section header (common JD patterns)
                next_section = re.search(
                    r'\n\s*(?:Position Title|Qualification|Experience|Duties|'
                    r'Responsibilities|Areas of Responsibility|Age Limit)',
                    section_text[len(header):], re.IGNORECASE
                )
                if next_section:
                    section_text = section_text[:len(header) + next_section.start()]

                words = section_text.split()
                quote = " ".join(words[:25])
                pos = text.find(quote)
                claims.append({
                    "answer_span": f"Salary info for {job_title}",
                    "doc": tc["doc"],
                    "page": tc["page"],
                    "chunk_id": tc["chunk_id"],
                    "quote": quote,
                    "quote_offsets": (pos if pos != -1 else 0, (pos + len(quote)) if pos != -1 else len(quote)),
                    "_source": "deterministic_jd",
                })
                break

        # If not found, search ±2 neighboring chunks
        if not salary_found:
            tc_idx = chunk_ids.index(tc["chunk_id"]) if tc["chunk_id"] in chunk_ids else -1
            if tc_idx == -1:
                continue

            for offset in [-2, -1, 1, 2]:
                neighbor_idx = tc_idx + offset
                if 0 <= neighbor_idx < len(chunk_ids):
                    n_cid = chunk_ids[neighbor_idx]
                    n_hit = chunk_map.get(n_cid)
                    if not n_hit:
                        continue
                    n_text = n_hit.get("text", "")
                    n_lower = n_text.lower()
                    for header in _JD_SALARY_HEADERS:
                        hpos = n_lower.find(header)
                        if hpos != -1:
                            section_text = n_text[hpos:]
                            next_sec = re.search(
                                r'\n\s*(?:Position Title|Qualification|Experience|Duties)',
                                section_text[len(header):], re.IGNORECASE
                            )
                            if next_sec:
                                section_text = section_text[:len(header) + next_sec.start()]
                            words = section_text.split()
                            quote = " ".join(words[:25])
                            pos = n_text.find(quote)
                            claims.append({
                                "answer_span": f"Salary info for {job_title}",
                                "doc": n_hit["doc"],
                                "page": n_hit["page"],
                                "chunk_id": n_cid,
                                "quote": quote,
                                "quote_offsets": (pos if pos != -1 else 0, (pos + len(quote)) if pos != -1 else len(quote)),
                                "_source": "deterministic_jd_neighbor",
                            })
                            salary_found = True
                            break
                    if salary_found:
                        break

    return claims


# =============================================================================
# STAGE 1: EVIDENCE EXTRACTOR  (extract_evidence)
# =============================================================================
EXTRACT_MODEL = os.getenv("EXTRACT_MODEL", "gpt-4o-mini")

def _flatten_hits(retrieval: Dict[str, Any], max_hits: int = 15) -> List[Dict[str, Any]]:
    """Flatten doc-grouped evidence into a scored hit list, sorted by score desc."""
    flat = []
    for dg in retrieval.get("evidence", []):
        doc = dg.get("doc_name", "Unknown")
        for hit in dg.get("hits", []):
            flat.append({
                "doc": doc,
                "page": hit.get("page_start", 0),
                "page_end": hit.get("page_end", hit.get("page_start", 0)),
                "chunk_id": hit.get("chunk_id", -1),
                "text": (hit.get("text") or "").strip(),
                "score": hit.get("score", 0),
                "tier": hit.get("tier", "vector"),
            })
    flat.sort(key=lambda h: h["score"], reverse=True)
    return flat[:max_hits]


def extract_evidence(
    query: str,
    retrieval: Dict[str, Any],
    intent: str = "",
    job_title: str = None,
    max_claims: int = 8,
) -> tuple:
    """
    STAGE 1: Extract evidence with deterministic quote localization.

    Returns (claims: List[Dict], debug_info: Dict)
    Each claim: {answer_span, doc, page, chunk_id, quote, quote_offsets}
    debug_info: {extractor_raw_json, localized_quotes, dropped_claims}
    """
    debug_info = {
        "extractor_raw_json": None,
        "localized_quotes": [],
        "dropped_claims": [],
    }

    hits = _flatten_hits(retrieval, max_hits=15)
    if not hits:
        return [], debug_info

    # --- DETERMINISTIC BYPASS: table-tier hits ---
    table_hits = [h for h in hits if h.get("tier") == "table"]
    if table_hits:
        table_claims = _extract_table_claims(table_hits, query)
        if table_claims:
            print(f"[Extractor] Deterministic table: {len(table_claims)} claims")
            debug_info["localized_quotes"] = [
                {"needle": "N/A (table)", "quote": c["quote"],
                 "offsets": c["quote_offsets"], "source": c["_source"]}
                for c in table_claims
            ]
            return table_claims, debug_info

    # --- DETERMINISTIC BYPASS: JD salary queries ---
    q_lower = query.lower()
    is_salary_q = any(k in q_lower for k in [
        "salary", "pay", "tankhwah", "tankha", "maash",
        "compensation", "kitni", "kitna",
    ])
    if is_salary_q and job_title:
        jd_claims = _extract_jd_salary_claims(hits, job_title)
        if jd_claims:
            print(f"[Extractor] Deterministic JD: {len(jd_claims)} claims")
            debug_info["localized_quotes"] = [
                {"needle": "N/A (JD)", "quote": c["quote"],
                 "offsets": c["quote_offsets"], "source": c.get("_source", "jd")}
                for c in jd_claims
            ]
            return jd_claims, debug_info

    # --- LLM EXTRACTION with needle format ---
    context_block = ""
    for i, h in enumerate(hits):
        context_block += (
            f"\n[CHUNK {i}] doc=\"{h['doc']}\" page={h['page']} chunk_id={h['chunk_id']} "
            f"tier={h['tier']}\n{h['text']}\n"
        )

    entity_constraint = ""
    if job_title:
        entity_constraint = (
            f"\n- The user is asking about the role \"{job_title}\". "
            f"ONLY extract claims from chunks that mention \"{job_title}\" or its abbreviation. "
            f"If no chunk mentions this role, return empty claims.\n"
        )

    system_prompt = (
        "You are an evidence extraction system. Find supporting facts from the chunks.\n\n"
        "OUTPUT FORMAT: Return ONLY valid JSON:\n"
        '{"claims": [\n'
        '  {"doc": "<doc name>", "page": <page_number>, "chunk_id": <chunk_id>, '
        '"needle": "<6-12 contiguous tokens copied from chunk text>", '
        '"answer_span_summary": "<short paraphrase of what this proves>"}\n'
        "]}\n\n"
        "RULES:\n"
        "- needle MUST be 6-12 contiguous tokens copied from the chunk text\n"
        "- needle is used to LOCATE the relevant passage, not as the final quote\n"
        "- Copy tokens as-is from the chunk, including any numbers or special chars\n"
        "- doc, page, chunk_id MUST match the CHUNK header exactly\n"
        "- Maximum " + str(max_claims) + " claims\n"
        "- If chunks don't answer the question, return {\"claims\": []}\n"
        f"{entity_constraint}"
    )

    user_prompt = f"Question: {query}\n\nChunks:\n{context_block}"

    try:
        client = get_client()
        response = client.chat.completions.create(
            model=EXTRACT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw_json = response.choices[0].message.content.strip()
        debug_info["extractor_raw_json"] = raw_json
        import json
        parsed = json.loads(raw_json)
        raw_claims = parsed.get("claims", [])
    except Exception as e:
        print(f"[Extractor] LLM extraction failed: {e}")
        return [], debug_info

    # --- LOCALIZE each claim using locate_quote ---
    chunk_texts = {h["chunk_id"]: h["text"] for h in hits}
    validated = []

    for claim in raw_claims:
        needle = (claim.get("needle") or "").strip()
        cid = claim.get("chunk_id")
        if not needle or cid is None:
            debug_info["dropped_claims"].append({
                "needle": needle, "chunk_id": cid,
                "reason": "missing needle or chunk_id",
            })
            continue

        source_text = chunk_texts.get(cid, "")
        if not source_text:
            debug_info["dropped_claims"].append({
                "needle": needle, "chunk_id": cid,
                "reason": f"chunk_id {cid} not found in hits",
            })
            continue

        loc = locate_quote(source_text, needle)
        if loc is None:
            debug_info["dropped_claims"].append({
                "needle": needle, "chunk_id": cid,
                "reason": "locate_quote returned None (no match >= similarity threshold)",
            })
            print(f"[Extractor] DROPPED: needle='{needle[:50]}' chunk={cid} (localization failed)")
            continue

        quote, start, end = loc

        # Final enforcement
        if source_text[start:end] != quote:
            debug_info["dropped_claims"].append({
                "needle": needle, "chunk_id": cid,
                "reason": f"offset mismatch: text[{start}:{end}] != quote",
            })
            print(f"[Extractor] DROPPED: offset mismatch chunk={cid}")
            continue

        validated.append({
            "answer_span": claim.get("answer_span_summary", ""),
            "doc": claim.get("doc", ""),
            "page": claim.get("page", 0),
            "chunk_id": cid,
            "quote": quote,
            "quote_offsets": (start, end),
            "_source": "llm_needle",
        })
        debug_info["localized_quotes"].append({
            "needle": needle,
            "quote": quote,
            "offsets": (start, end),
            "similarity": "exact" if quote.strip() and needle.strip() else "fuzzy",
        })

    print(f"[Extractor] {len(raw_claims)} raw → {len(validated)} localized claims "
          f"({len(debug_info['dropped_claims'])} dropped)")
    return validated, debug_info


# =============================================================================
# STAGE 2: ANSWER COMPOSER  (compose_answer)
# =============================================================================
COMPOSE_MODEL = os.getenv("COMPOSE_MODEL", os.getenv("ANSWER_MODEL", "gpt-4o-mini"))

def compose_answer(
    query: str,
    claims: List[Dict[str, Any]],
    intent: str = "",
    conversation_history: List[Dict[str, str]] = None,
) -> str:
    """
    STAGE 2: Generate a user-facing answer using ONLY the extracted claims.

    The answer must:
    - Use the same language as the query (English / Roman Urdu)
    - Reference quotes from claims
    - NOT introduce any new facts beyond extracted quotes
    - Use bullet points for structured data
    """
    if not claims:
        return ""

    # Build claims context
    claims_block = ""
    for i, c in enumerate(claims, 1):
        claims_block += (
            f"[{i}] \"{c['quote']}\"\n"
            f"   Source: {c['doc']} (Page {c['page']})\n"
            f"   Summary: {c['answer_span']}\n\n"
        )

    is_numeric = any(k in query.lower() for k in [
        "number mein", "number me", "kitna", "kitni", "how much",
        "exact salary", "figures", "amount",
    ])
    numeric_directive = ""
    if is_numeric:
        numeric_directive = (
            "\n- The user wants NUMBERS. Extract and present only numeric "
            "information (salary figures, amounts, scales) from the claims.\n"
        )

    system_prompt = (
        "You are the PERA AI Assistant. Generate your answer using ONLY the extracted "
        "claims below. Do NOT add any information not present in the claims.\n\n"
        "RULES:\n"
        "- Answer in the SAME language as the user's question\n"
        "- For each fact you state, it must come from one of the claims\n"
        "- Use bullet points for lists. Bold key terms.\n"
        "- Be concise and professional\n"
        "- If claims are about salary/pay, present the exact figures from quotes\n"
        "- Do NOT say 'according to the claims' — just state the facts naturally\n"
        "- Do NOT hedge with 'typically', 'generally', 'may apply'\n"
        "- If there is insufficient information in the claims, say so clearly\n"
        f"{numeric_directive}\n"
        f"EXTRACTED CLAIMS:\n{claims_block}"
    )

    messages = [{"role": "system", "content": system_prompt}]
    if conversation_history:
        valid = [m for m in conversation_history if m.get("role") in ("user", "assistant")]
        messages.extend(valid[-4:])
    messages.append({"role": "user", "content": query})

    try:
        client = get_client()
        response = client.chat.completions.create(
            model=COMPOSE_MODEL,
            messages=messages,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Composer] LLM composition failed: {e}")
        return ""


def _build_references_from_claims(
    claims: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build UI references from validated claims (strict citation contract)."""
    refs = []
    seen = set()
    base_url = os.getenv("BASE_URL", "https://ask.pera.gop.pk").rstrip("/")
    for c in claims:
        key = f"{c['doc']}_{c['page']}"
        if key in seen:
            continue
        seen.add(key)
        doc = c["doc"]
        page = c["page"]
        url = f"{base_url}/assets/data/{doc}#page={page}"
        refs.append({
            "document": doc,
            "page_start": page,
            "open_url": url,
            "snippet": c.get("quote", "")[:200],
            "quote_offsets": c.get("quote_offsets"),
        })
    return refs


# =============================================================================
# Main Answer Function (2-Stage Extract-then-Answer Pipeline)
# =============================================================================
def answer_question(
    current_question: str,
    retrieval: Dict[str, Any],
    conversation_history: List[Dict[str, str]] = None,
    session_context: Optional[Dict[str, Any]] = None,
    rewritten_query: Optional[str] = None,
    intent: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate answer using Intent Router + Evidence Gate + System Prompt.

    CITATION CONTRACT:
    - Refusal/smalltalk/greeting -> ALWAYS return references=[]
    - Factual answer -> MUST include >=1 reference with doc+page+snippet
    - If factual but 0 references -> force refusal

    session_context: optional dict for salary follow-up (last_compensation_type, last_grade)
    """
    client = get_client()

    # 0. Creator question intercept (deterministic, no LLM needed)
    if _is_creator_question(current_question):
        return {
            "answer": _CREATOR_RESPONSE,
            "references": [],
            "decision": "answer"
        }

    # 1. INTENT CLASSIFICATION
    # If intent passed from caller (fastapi_app), use it; otherwise classify
    if not intent:
        intent = classify_intent(current_question)
    print(f"[Answerer] Intent: {intent} for query: '{current_question[:60]}...'")

    # 1a. GREETING/SMALLTALK -> deterministic response, ZERO citations
    if intent == QueryIntent.GREETING_SMALLTALK:
        q_lower = current_question.lower().strip()
        # Identity question?
        identity_words = ["kon ho", "kya ho", "who are you", "what are you",
                          "your name", "naam batao", "tera naam"]
        if any(w in q_lower for w in identity_words):
            return {
                "answer": _BOT_IDENTITY_RESPONSE,
                "references": [],
                "decision": "greeting"
            }
        # Greeting / thanks
        if _detect_roman_urdu(current_question):
            response = _GREETING_RESPONSE_ROMAN
        else:
            response = _GREETING_RESPONSE_EN
        return {
            "answer": response,
            "references": [],
            "decision": "greeting"
        }

    # 1b. OUT_OF_SCOPE -> always strict refusal, no citations
    if intent == QueryIntent.OUT_OF_SCOPE:
        refusal = _STRICT_REFUSAL_URDU if _detect_roman_urdu(current_question) else _STRICT_REFUSAL
        return {
            "answer": refusal,
            "references": [],
            "decision": "refuse"
        }

<<<<<<< Updated upstream
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
=======
    # 1c. SSO → always System Support Officer (deterministic, no clarification)
    # (SSO ambiguity is removed — always resolves to System Support Officer)

    # 1d. DETERMINISTIC REGISTRY BYPASS — short-circuit before RAG
    if _HAS_REGISTRY:
        reg_result = registry_answer(current_question, ctx=session_context or {})
        if reg_result is not None:
            reg_claims = reg_result.get("claims", [])
            reg_debug = reg_result.get("debug", {})
            if reg_claims:
                print(f"[Answerer] Registry HIT: {reg_debug.get('source','?')} — {len(reg_claims)} claims")
                # Compose answer from registry claims (same Stage-2 as normal pipeline)
                answer_text = compose_answer(
                    query=current_question,
                    claims=reg_claims,
                    intent=intent,
                    conversation_history=conversation_history,
                )
                if answer_text:
                    refs = _build_references_from_claims(reg_claims)
                    result = {
                        "answer": answer_text,
                        "references": refs,
                        "decision": "answer",
                        "_extracted_claims": reg_claims,
                        "_debug_extraction": {"registry": reg_debug},
                    }
                    # Propagate entity lock for follow-up
                    if reg_debug.get("title"):
                        result["_job_title"] = reg_debug["title"]
                    return result
            elif not reg_claims and reg_debug.get("not_found"):
                # Registry explicitly says: not found → strict refusal
                refusal = _STRICT_REFUSAL_URDU if _detect_roman_urdu(current_question) else _STRICT_REFUSAL
                return {
                    "answer": refusal,
                    "references": [],
                    "decision": "refuse",
                    "_debug_extraction": {"registry": reg_debug},
                }
            # else: registry returned None or empty claims without not_found → fall through to RAG

    # 1e. ENTITY PINNING: extract job title from query or session
    job_title = extract_job_title(current_question)
    if not job_title and session_context:
        # Follow-up question: reuse last job title from session
        prev_title = session_context.get("last_job_title")
        if prev_title:
            job_title = prev_title
            print(f"[Answerer] Entity pin (session): {job_title}")
    if job_title:
        print(f"[Answerer] Entity pinned: {job_title}")
        # Filter retrieval to only chunks containing this entity
        retrieval = filter_evidence_by_entity(retrieval, job_title)
        entity_gate = retrieval.get("_gate_reason")
        if entity_gate:
            print(f"[Answerer] Entity filter: {entity_gate}")

    # 1e. SPPP TABLE INJECTION: if salary query + SPPP grade known, merge schedule-III
    q_lower_check = current_question.lower()
    is_salary_q = any(k in q_lower_check for k in [
        "salary", "pay", "tankhwah", "tankha", "maash", "kitni",
        "pay scale", "payscale", "compensation",
        "number mein", "number me", "kitna milta", "kitni milti",
    ])
    if is_salary_q and job_title:
        # Check if evidence contains SPPP-x for this job
        all_ev_text = ""
        for dg in retrieval.get("evidence", []):
            for hit in dg.get("hits", []):
                all_ev_text += " " + (hit.get("text") or "")
        sppp_match = re.search(r'\bSPPP[\s\-]*(\d+)', all_ev_text, re.IGNORECASE)
        if sppp_match:
            sppp_grade = f"SPPP-{sppp_match.group(1)}"
            print(f"[Answerer] Detected {sppp_grade} for {job_title}, injecting Schedule-III")
            # Fetch Schedule-III table via table_lookup
            from retriever import table_lookup as _table_lookup
            from index_store import load_index_and_chunks
            from retriever import _resolve_index_dir
            _, chunks_all = load_index_and_chunks(_resolve_index_dir(None))
            table_ev = _table_lookup(f"schedule iii {sppp_grade} salary", chunks_all, index_dir=_resolve_index_dir(None))
            if table_ev:
                # Merge table evidence into retrieval
                existing = retrieval.get("evidence", [])
                for tg in table_ev:
                    # Check if doc already in evidence
                    found = False
                    for eg in existing:
                        if eg.get("doc_name") == tg.get("doc_name"):
                            eg["hits"].extend(tg.get("hits", []))
                            found = True
                            break
                    if not found:
                        existing.append(tg)
                retrieval["evidence"] = existing
                retrieval["has_evidence"] = True
                print(f"[Answerer] Merged Schedule-III table ({len(table_ev)} groups)")

    # 2. EVIDENCE RELEVANCE GATE
    gated_retrieval = evidence_relevance_gate(
        retrieval, current_question, intent
    )
    gate_reason = gated_retrieval.get("_gate_reason")
    if gate_reason:
        print(f"[Answerer] Evidence gate: {gate_reason}")

    # 3. Check for empty evidence after gate
    if not gated_retrieval.get("has_evidence") or not gated_retrieval.get("evidence"):
        refusal = _STRICT_REFUSAL_URDU if _detect_roman_urdu(current_question) else _STRICT_REFUSAL
        return {
            "answer": refusal if intent != QueryIntent.GENERAL_PERA else
                "I'm sorry, I couldn't find any information about that in the PERA documents.",
            "references": [],
            "decision": "refuse"
        }

    # 3b. Detect compensation type from evidence snippets (for salary queries)
    all_snippets = []
    for doc_group in gated_retrieval.get("evidence", []):
        for hit in doc_group.get("hits", []):
            text = (hit.get("text") or "").strip()
            if text:
                all_snippets.append(text)

    q_lower = current_question.lower()
    is_salary_query = any(k in q_lower for k in [
        "salary", "pay", "tankhwah", "tankha", "maash", "kitni",
        "pay scale", "payscale", "compensation", "grade",
        "number mein", "number me", "kitna milta", "kitni milti",
    ])

    comp_type = CompensationType.NOT_SPECIFIED
    comp_grade = None
    if is_salary_query and all_snippets:
        comp_type = detect_compensation_type(all_snippets)
        comp_grade = extract_compensation_grade(all_snippets, comp_type)
        print(f"[Answerer] Compensation: type={comp_type}, grade={comp_grade}")

    if session_context and any(k in q_lower for k in ["number mein", "number me", "kitna", "kitni"]):
        prev_comp = session_context.get("last_compensation_type")
        prev_grade = session_context.get("last_grade")
        if prev_comp and comp_type == CompensationType.NOT_SPECIFIED:
            comp_type = prev_comp
            comp_grade = prev_grade
            print(f"[Answerer] Follow-up: reusing comp={comp_type}, grade={comp_grade}")

    # =====================================================================
    # STAGE 1: EXTRACT EVIDENCE (needle-based with deterministic bypasses)
    # =====================================================================
    print(f"[Answerer] Stage 1: Extracting evidence claims...")
    # Use rewritten query for extraction to catch resolved entities (e.g. "DG" instead of "usko")
    extraction_query = rewritten_query if rewritten_query else current_question
    claims, extraction_debug = extract_evidence(
        query=extraction_query,
        retrieval=gated_retrieval,
        intent=intent,
        job_title=job_title,
>>>>>>> Stashed changes
    )

    # If no valid claims → strict refusal
    if not claims:
        refusal = _STRICT_REFUSAL_URDU if _detect_roman_urdu(current_question) else _STRICT_REFUSAL
        return {
            "answer": refusal if intent != QueryIntent.GENERAL_PERA else
                "I'm sorry, I couldn't find specific information about that in the PERA documents.",
            "references": [],
            "decision": "refuse",
            "_extracted_claims": [],
            "_debug_extraction": extraction_debug,
        }

    print(f"[Answerer] Stage 1 complete: {len(claims)} validated claims")

    # =====================================================================
    # STAGE 2: COMPOSE ANSWER (from extracted claims ONLY)
    # =====================================================================
    print(f"[Answerer] Stage 2: Composing answer from claims...")
    answer_text = compose_answer(
        query=current_question,
        claims=claims,
        intent=intent,
        conversation_history=conversation_history,
    )

    if not answer_text:
        refusal = _STRICT_REFUSAL_URDU if _detect_roman_urdu(current_question) else _STRICT_REFUSAL
        return {
            "answer": refusal,
            "references": [],
            "decision": "refuse",
            "_extracted_claims": claims,
            "_debug_extraction": extraction_debug,
        }

    # Post-processing: ban phrase check on composed answer
    lower_ans = answer_text.lower()

    # A) Check if composer says info not available
    _NO_INFO_PHRASES = [
        "not available in the provided context",
        "not explicitly mentioned",
        "not found in the provided",
        "i couldn't find", "i could not find",
        "no information available",
        "not available in the pera documents",
        "outside my scope", "outside the scope",
        "is not available in the pera",
    ]
    if any(phrase in lower_ans for phrase in _NO_INFO_PHRASES):
        return {
            "answer": answer_text,
            "references": [],
            "decision": "refuse",
            "_extracted_claims": claims,
            "_debug_extraction": extraction_debug,
        }

    # B) Ban phrase enforcement
    override, needs_regen = _post_process_strict(answer_text, intent, current_question)
    if needs_regen:
        print("[Answerer] Ban phrase in composed answer, hard refusal")
        refusal = _STRICT_REFUSAL_URDU if _detect_roman_urdu(current_question) else _STRICT_REFUSAL
        return {
            "answer": refusal,
            "references": [],
            "decision": "refuse",
            "_extracted_claims": claims,
            "_debug_extraction": extraction_debug,
        }

    # C) Section verification for ENFORCEMENT_LEGAL
    if intent == QueryIntent.ENFORCEMENT_LEGAL and all_snippets:
        section_warning = _verify_section_citations(answer_text, all_snippets)
        if section_warning:
            answer_text = answer_text + "\n\n> **Note**: " + section_warning

    # D) Build references STRICTLY from validated claims (citation contract)
    refs = _build_references_from_claims(claims)

    # E) Citation contract: factual answer MUST have >=1 reference
    if not refs:
        return {
            "answer": "I'm sorry, I couldn't verify this information with a specific citation. "
                      "Please check the official PERA documents directly.",
            "references": [],
            "decision": "refuse",
            "_extracted_claims": claims,
            "_debug_extraction": extraction_debug,
        }

    # F) Build result with compensation context for follow-up
    result = {
        "answer": answer_text,
        "references": refs,
        "decision": "answer",
        "_extracted_claims": claims,
        "_debug_extraction": extraction_debug,
    }
    if is_salary_query:
        result["_compensation_type"] = comp_type
        result["_compensation_grade"] = comp_grade
    if job_title:
        result["_job_title"] = job_title

    return result
