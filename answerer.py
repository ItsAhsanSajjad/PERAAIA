"""
PERA AI Answerer (v3: clean support-state model)

4 answer states: supported, partially_supported, unsupported, conflicting
No more contradictory "not explicitly defined" + correct answer.
"""
from __future__ import annotations

import os
import re
from typing import List, Dict, Any, Optional

from openai_clients import get_chat_client, ANSWER_MODEL
from grounding import verify_grounding, GroundingResult
from log_config import get_logger

log = get_logger("pera.answerer")

# Evidence quality thresholds
ANSWER_MIN_TOP_SCORE = float(os.getenv("ANSWER_MIN_TOP_SCORE", "0.15"))
HIT_MIN_SCORE = float(os.getenv("HIT_MIN_SCORE", "0.12"))
MAX_HITS_PER_DOC = int(os.getenv("MAX_HITS_PER_DOC_FOR_PROMPT", "6"))
MAX_DOCS = int(os.getenv("MAX_DOCS_FOR_PROMPT", "5"))
MAX_EVIDENCE_CHARS = int(os.getenv("MAX_EVIDENCE_CHARS", "18000"))


# =============================================================================
# Position Title Detection for Evidence Filtering
# =============================================================================
_POSITION_TITLE_RE = re.compile(
    r"Position\s+Title\s*:\s*-?\s*(.+?)(?:\s*Report\s+To\s*:|\n)",
    re.IGNORECASE
)

# Extended pattern for REFERENCE extraction only (also matches [Role:] format)
_REF_POSITION_TITLE_RE = re.compile(
    r"(?:Position\s+Title\s*:\s*-?\s*(.+?)(?:\s*Report\s+To\s*:|\n)"
    r"|\[Role:\s*(.+?)\])",
    re.IGNORECASE
)

# Stop words that should NOT be part of a role name
_ROLE_STOP_WORDS = {
    # English stop words
    "salary", "pay", "benefit", "benefits", "allowance", "appointment",
    "scale", "package", "sppp", "compensation", "grade", "detail",
    "details", "responsibilities", "duties", "powers", "functions",
    "qualification", "experience", "of", "the", "in", "at", "for",
    "what", "is", "are", "about", "tell", "me", "explain",
    "how", "much", "does", "do", "earn", "get", "paid", "make",
    "structure", "reporting", "report", "reports", "who",
    # Urdu / Roman Urdu stop words
    "ki", "ka", "ke", "ko", "kya", "hai", "hain", "hy", "hen",
    "aur", "ya", "mein", "se", "par", "pe", "ye", "yeh", "woh",
    "nahi", "nhi", "na", "ho", "tha", "thi", "the",
    "batao", "btao", "bataen", "bataiye", "batayein", "btaen",
    "kitni", "kitna", "kitne", "kahan", "kaun", "konsi", "kaunsi",
    "wala", "wali", "wale", "hota", "hoti", "hote",
    "kaise", "kaisa", "kesi", "kaisi",
    "abhi", "kab", "jab", "tab",
}

# Urdu particles that should be stripped before role detection
_URDU_PARTICLES = re.compile(
    r"\b(?:ki|ka|ke|ko|kya|hai|hain|hy|hen|aur|ya|mein|se|par|pe|ye|yeh|woh|"
    r"nahi|nhi|na|ho|tha|thi|batao|btao|bataen|bataiye|batayein|btaen|"
    r"kitni|kitna|kitne|kahan|kaun|konsi|kaunsi|wala|wali|wale|"
    r"hota|hoti|hote|kaise|kaisa|kesi|kaisi|abhi|kab|jab|tab)\b",
    re.IGNORECASE
)

def _normalize_query_for_role(question: str) -> str:
    """Strip Urdu/Roman Urdu particles from query before role detection.
    'SSO ki salary kya hai' → 'SSO salary'
    'manager development ka pay scale kya hai' → 'manager development pay scale'
    """
    q = _URDU_PARTICLES.sub(' ', question or '')
    return re.sub(r'\s+', ' ', q).strip()

def _extract_position_title(text: str) -> str:
    m = _POSITION_TITLE_RE.search(text or "")
    return m.group(1).strip().lower() if m else ""

def _extract_ref_position_title(text: str) -> str:
    """Like _extract_position_title but also matches [Role:] format.
    Used ONLY for reference filtering, not evidence assembly."""
    m = _REF_POSITION_TITLE_RE.search(text or "")
    if not m:
        return ""
    title = (m.group(1) or m.group(2) or "").strip()
    return title.lower()

# Pattern 1: PREFIX SPEC — "manager development", "director monitoring", "assistant manager HR"
_ROLE_PREFIX_RE = re.compile(
    r"\b((?:senior\s+|assistant\s+|deputy\s+)?(?:manager|director|head|coordinator|superintendent))"
    r"\s+([\w&]+(?:\s+[\w&]+)?)",
    re.IGNORECASE
)

# Pattern 1b: BARE PREFIX — "deputy director", "director general", "assistant director"
# These are complete role names without a specialization word
_BARE_ROLE_RE = re.compile(
    r"\b((?:deputy|assistant|additional|joint|regional)\s+(?:director|manager|head)(?:\s+general)?)"
    r"(?:\s+|$)",
    re.IGNORECASE
)

# Pattern 2: SPEC ROLE — "system support officer", "enforcement officer", "data entry operator"
_ROLE_SUFFIX_RE = re.compile(
    r"\b([\w]+(?:\s+[\w]+)?)\s+(officer|operator|sergeant|analyst|developer|writer|administrator)\b",
    re.IGNORECASE
)

# Full title patterns from PERA docs: "Manager (Development)", "Assistant Director (Admin & HR)"
_FULL_TITLE_RE = re.compile(
    r"\b((?:senior\s+|assistant\s+|deputy\s+)?(?:manager|director|head))\s*\(\s*([^)]+)\s*\)",
    re.IGNORECASE
)

def _detect_target_role(question: str) -> str:
    """
    Detect the target role/position from a user's question.
    Handles multiple naming patterns used in PERA:
    - PREFIX SPEC: "manager development salary" → "manager development"
    - SPEC ROLE: "system support officer salary" → "system support officer"
    - FULL TITLE: "Manager (Development)" → "manager development"
    """
    # First strip Urdu particles, then normalize English
    q = _normalize_query_for_role(question or "")
    q = re.sub(r"[''\u2019]s\b", "", q)
    q_lower = q.lower()
    
    # Try full parenthesized title first: "Manager (Development)"
    m = _FULL_TITLE_RE.search(q)
    if m:
        prefix = m.group(1).strip().lower()
        spec = m.group(2).strip().lower()
        return f"{prefix} {spec}"
    
    # Try PREFIX SPEC pattern: "manager development salary"
    m = _ROLE_PREFIX_RE.search(q)
    if m:
        prefix = m.group(1).strip().lower()
        suffix_raw = m.group(2).strip().lower()
        # Strip stop words from the end
        suffix_words = suffix_raw.split()
        cleaned = []
        for w in suffix_words:
            if w in _ROLE_STOP_WORDS:
                break
            cleaned.append(w)
        if cleaned:
            return (prefix + " " + " ".join(cleaned)).strip()
        # PREFIX SPEC failed (spec was all stop words like 'salary')
        # Fall through to BARE PREFIX check below
    
    # Try BARE PREFIX pattern: "deputy director salary" → "deputy director"
    m = _BARE_ROLE_RE.search(q)
    if m:
        return m.group(1).strip().lower()
    
    # Try SPEC ROLE pattern: "system support officer salary", "enforcement officer"
    m = _ROLE_SUFFIX_RE.search(q)
    if m:
        spec_raw = m.group(1).strip().lower()
        role_word = m.group(2).strip().lower()
        # Strip leading stop words from spec
        spec_words = spec_raw.split()
        cleaned = [w for w in spec_words if w not in _ROLE_STOP_WORDS]
        if cleaned:
            return " ".join(cleaned) + " " + role_word
        return role_word
    
    return ""


# =============================================================================
# Context Formatting (v2: position-title-aware)
# =============================================================================
def format_evidence_for_llm(retrieval: Dict[str, Any], question: str = "") -> str:
    """
    Format retrieved chunks into a clean context block.
    Uses position-title-aware scoring to prevent wrong-entity evidence.
    """
    if not retrieval.get("has_evidence"):
        return ""

    evidence_list = retrieval.get("evidence", [])
    context_parts: List[str] = []
    total_chars = 0

    q_lower = question.lower() if question else ""
    _ABBREV = {
        "cto": "chief technology officer",
        "dg": "director general",
        "ddg": "deputy director general",
        "adg": "additional director general",
        "hr": "human resources",
        "it": "information technology",
        "eo": "enforcement officer",
        "io": "investigation officer",
        "sso": "system support officer",
        "sdeo": "sub divisional enforcement officer",
        "deo": "data entry operator",
        "mgr": "manager",
        "dba": "database administrator",
        "se": "software engineer",
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

    # Detect target role/position for entity-aware filtering
    # Use expanded_q so abbreviations like SSO are properly resolved
    target_role = _detect_target_role(expanded_q) or _detect_target_role(question)
    if target_role:
        log.info("Position-aware filtering: target='%s'", target_role)

    # Flatten all hits across doc groups for unified position-aware scoring
    all_hits = []
    for doc_group in evidence_list:
        doc_name = (doc_group.get("doc_name", "Unknown Document") or "Unknown Document").strip()
        for hit in doc_group.get("hits", []):
            hit["_doc_name"] = doc_name
            all_hits.append(hit)

    # Score each hit with position-title awareness
    scored_hits = []
    for hit in all_hits:
        text = (hit.get("text") or "")
        text_lower = text.lower()
        base_score = float(hit.get("_blend", hit.get("score", 0)))
        is_context = hit.get("_is_smart_context", False)

        if not is_context and base_score < HIT_MIN_SCORE:
            continue

        subject_match = sum(1 for w in _subject_words if w in text_lower)

        # Position-title matching: boost exact, DROP wrong positions entirely
        title_bonus = 0.0
        if target_role:
            chunk_title = _extract_position_title(text)
            if chunk_title:
                target_norm = re.sub(r"[^a-z0-9\s]", "", target_role)
                chunk_norm = re.sub(r"[^a-z0-9\s]", "", chunk_title)
                if target_norm in chunk_norm or chunk_norm in target_norm:
                    title_bonus = 10.0  # Exact position match
                else:
                    # Multi-word matching: check if the SPECIFIC words from the
                    # target role appear in the chunk title.
                    # Generic role words are excluded (they appear in many positions).
                    _generic_role_words = {
                        "manager", "officer", "director", "assistant", "deputy",
                        "chief", "head", "senior", "coordinator", "superintendent",
                        "operator", "sergeant", "analyst", "developer", "writer",
                    }
                    target_words = target_norm.split()
                    # Specific words = non-generic words with len > 2
                    specific_words = [w for w in target_words
                                      if w not in _generic_role_words and len(w) > 2]
                    if specific_words:
                        # How many specific words match?
                        matches = sum(1 for w in specific_words if w in chunk_norm)
                        if matches == len(specific_words):
                            title_bonus = 5.0   # All specific words match
                        elif matches > 0:
                            title_bonus = 2.0   # Partial match
                        else:
                            # HARD DROP — wrong position chunk
                            log.debug("Dropping wrong-position chunk: '%s' (wanted '%s')",
                                     chunk_title[:40], target_role)
                            continue
                    else:
                        # Only generic words (e.g. just "officer") — can't filter
                        # Don't drop, just don't boost
                        title_bonus = 0.0

        # Section-heading boost: DISABLED (caused scoring order changes)
        # heading_bonus commented out to preserve proven ranking
        heading_bonus = 0.0

        combined = subject_match + title_bonus + heading_bonus + base_score
        scored_hits.append((combined, hit))

    # FALLBACK: If position filter dropped ALL position-titled chunks,
    # re-scan and keep top hits by subject_match + base_score only.
    # This prevents zero-evidence scenarios when the target role exists
    # in the docs but the position filter was too aggressive.
    if target_role and not scored_hits:
        log.warning("Position filter dropped all hits for '%s' — using fallback", target_role)
        for hit in all_hits:
            text = (hit.get("text") or "")
            text_lower = text.lower()
            base_score = float(hit.get("_blend", hit.get("score", 0)))
            is_context = hit.get("_is_smart_context", False)
            if not is_context and base_score < HIT_MIN_SCORE:
                continue
            subject_match = sum(1 for w in _subject_words if w in text_lower)
            combined = subject_match + base_score
            scored_hits.append((combined, hit))

    scored_hits.sort(key=lambda x: x[0], reverse=True)

    # Content dedup is available but disabled to preserve proven scoring order.
    # Enable by uncommenting when evidence limits are expanded in future.
    # (dedup code preserved below for future use)

    # Assemble evidence respecting per-doc limits
    doc_hit_counts: Dict[str, int] = {}
    docs_used_set: set = set()

    for _score, hit in scored_hits:
        doc_name = hit["_doc_name"]

        if len(docs_used_set) >= MAX_DOCS and doc_name not in docs_used_set:
            continue
        doc_count = doc_hit_counts.get(doc_name, 0)
        if doc_count >= MAX_HITS_PER_DOC:
            continue

        text = (hit.get("text") or "").strip()
        page = hit.get("page_start", "?")
        safe_doc = doc_name.replace("<", "").replace(">", "").replace('"', "").replace("'", "")
        eid = hit.get("evidence_id", "")
        part = (
            f'<evidence doc="{safe_doc}" page="{page}" eid="{eid}">\n'
            f"{text}\n"
            f"</evidence>"
        )

        if total_chars + len(part) > MAX_EVIDENCE_CHARS:
            break

        context_parts.append(part)
        total_chars += len(part)
        doc_hit_counts[doc_name] = doc_count + 1
        docs_used_set.add(doc_name)
        # Track hit for reference extraction
        hit["_used_for_evidence"] = True

        if total_chars >= MAX_EVIDENCE_CHARS:
            break

    evidence_ids = []
    for part in context_parts:
        m = re.search(r'eid="([^"]+)"', part)
        if m:
            evidence_ids.append(m.group(1))
    if evidence_ids:
        log.info("Evidence assembled: %d chunks, %d chars, eids=%s",
                 len(context_parts), total_chars, evidence_ids[:10])

    # --- Salary-Bridge: inject salary-table chunks when position chunks lack salary data ---
    # This fixes the case where position description chunks (e.g., "Head Monitoring")
    # are retrieved but the salary value is on a different page (Schedule-III / SPPP table).
    _SALARY_QUERY_WORDS = {"salary", "pay", "scale", "bps", "sppp", "compensation", "allowance",
                           "kitni", "kya", "btao", "tankhwah", "benefits"}
    is_salary_query = any(w in q_lower.split() for w in _SALARY_QUERY_WORDS)

    if is_salary_query and target_role and context_parts:
        # Check if existing evidence already contains salary values
        assembled_text = " ".join(context_parts).lower()
        has_salary_data = bool(re.search(
            r"(?:sppp[-\s]*\d|bps[-\s]*\d|bs[-\s]*\d|pay\s+(?:scale|package)|"
            r"salary\s+and\s+benefits.*(?:sppp|bps|bs[-\s])|"
            r"minimum\s+pay\s+per\s+month|maximum\s+pay)",
            assembled_text
        ))

        if not has_salary_data:
            log.info("Salary-bridge: position chunks found for '%s' but NO salary data — "
                     "injecting supplementary salary evidence", target_role)

            # Scan ALL scored_hits (including ones not yet assembled) for salary-relevant chunks
            salary_supplements = []
            for _score, hit in scored_hits:
                if hit.get("_used_for_evidence"):
                    continue  # Already in evidence
                text = (hit.get("text") or "").lower()
                # Look for chunks with SPPP/BPS salary data or Schedule-III tables
                if re.search(r"(?:sppp[-\s]*\d|bps[-\s]*\d|salary\s+and\s+benefits|"
                             r"schedule[-\s]*iii|pay\s+(?:scale|package)|"
                             r"minimum\s+pay|maximum\s+pay)", text):
                    salary_supplements.append(hit)
                    if len(salary_supplements) >= 3:
                        break

            # Also scan all_hits for Schedule-III/salary chapter chunks
            # that might not be in scored_hits due to position filtering
            if len(salary_supplements) < 2:
                for hit in all_hits:
                    if hit.get("_used_for_evidence"):
                        continue
                    text = (hit.get("text") or "").lower()
                    if re.search(r"(?:schedule[-\s]*iii|chapter\s+vi.*salary|"
                                 r"salary\s+structure|pay\s+scales)", text):
                        if hit not in salary_supplements:
                            salary_supplements.append(hit)
                            if len(salary_supplements) >= 3:
                                break

            for hit in salary_supplements:
                text = (hit.get("text") or "").strip()
                page = hit.get("page_start", "?")
                doc_name = hit.get("_doc_name", "Unknown")
                safe_doc = doc_name.replace("<", "").replace(">", "").replace('"', "").replace("'", "")
                eid = hit.get("evidence_id", "")
                part = (
                    f'<evidence doc="{safe_doc}" page="{page}" eid="{eid}" '
                    f'role="salary-supplement">\n'
                    f"{text}\n"
                    f"</evidence>"
                )
                if total_chars + len(part) > MAX_EVIDENCE_CHARS + 4000:
                    break  # Allow up to 4K extra for salary supplements
                context_parts.append(part)
                total_chars += len(part)
                hit["_used_for_evidence"] = True
                log.info("Salary-bridge injected: %s p.%s (%d chars)", safe_doc[:30], page, len(part))

    return "\n\n".join(context_parts)


def extract_references_simple(retrieval: Dict[str, Any], question: str = "") -> List[Dict[str, Any]]:
    """Extract reference links ONLY from chunks that were actually used for the LLM answer,
    filtered for relevance to the query topic."""
    refs: List[Dict[str, Any]] = []
    seen = set()
    base_url = os.getenv("BASE_URL", "https://ask.pera.gop.pk").rstrip("/")

    evidence_list = retrieval.get("evidence", [])

    # Detect target role for relevance filtering
    target_role = _detect_target_role(question)

    # Build subject keywords for relevance scoring
    _stop = {
        "what", "which", "where", "when", "does", "that", "this", "with",
        "from", "about", "have", "been", "will", "shall", "their", "these",
        "the", "for", "and", "how", "tell", "me", "explain", "describe",
        "give", "show", "detail", "details", "full", "salary", "pay",
        "scale", "benefit", "appointment", "who", "pera",
    }
    q_words = set(w.lower() for w in (question or "").split()
                  if len(w) > 2 and w.lower() not in _stop)

    # Collect ONLY hits that were marked as used by format_evidence_for_llm
    used_hits = []
    for doc_group in evidence_list:
        doc_name = doc_group.get("doc_name", "Document")
        for hit in doc_group.get("hits", []):
            if hit.get("_used_for_evidence"):
                score = float(hit.get("_blend", hit.get("score", 0)))
                used_hits.append((score, doc_name, hit))

    # Fallback: if no marks, use high-score hits
    if not used_hits:
        for doc_group in evidence_list:
            doc_name = doc_group.get("doc_name", "Document")
            for hit in doc_group.get("hits", []):
                score = float(hit.get("_blend", hit.get("score", 0)))
                if score >= HIT_MIN_SCORE:
                    used_hits.append((score, doc_name, hit))

    # Score hits for REFERENCE relevance (different from evidence scoring)
    scored_refs = []
    for base_score, doc_name, hit in used_hits:
        text = (hit.get("text") or "").lower()
        ref_score = base_score

        # Boost/Drop: snippet position title vs target role
        if target_role:
            target_norm = re.sub(r"[^a-z0-9\s]", "", target_role)
            chunk_title = _extract_ref_position_title(hit.get("text") or "")
            if chunk_title:
                chunk_norm = re.sub(r"[^a-z0-9\s]", "", chunk_title)
                if target_norm in chunk_norm or chunk_norm in target_norm:
                    ref_score += 5.0  # Exact position match
                else:
                    # Check significant words
                    _generic = {"manager", "officer", "director", "assistant", "deputy",
                               "chief", "head", "senior"}
                    sig = [w for w in target_norm.split() if w not in _generic and len(w) > 2]
                    if sig and all(w in chunk_norm for w in sig):
                        ref_score += 3.0  # Significant words match
                    elif sig and not any(w in chunk_norm for w in sig):
                        # HARD DROP: chunk has a Position Title for a DIFFERENT role
                        continue
            else:
                # No Position Title in chunk — check if target role words in body
                if target_norm in text:
                    ref_score += 3.0
                # No penalty for chunks without Position Title (may be salary tables)

        # Boost: snippet mentions query subject words
        word_hits = sum(1 for w in q_words if w in text)
        ref_score += word_hits * 0.5

        scored_refs.append((ref_score, doc_name, hit))

    scored_refs.sort(key=lambda x: x[0], reverse=True)

    # Take top refs (max 4), with per-doc limits
    docs_used = set()
    doc_ref_counts: Dict[str, int] = {}
    max_refs = 4
    max_refs_per_doc = 2

    for _score, doc_name, hit in scored_refs:
        if len(refs) >= max_refs:
            break
        if len(docs_used) >= MAX_DOCS and doc_name not in docs_used:
            continue
        if doc_ref_counts.get(doc_name, 0) >= max_refs_per_doc:
            continue

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
        doc_ref_counts[doc_name] = doc_ref_counts.get(doc_name, 0) + 1
        docs_used.add(doc_name)

    return refs


# =============================================================================
# Creator Question Detection
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
# Detect if user explicitly asked for sources/pages/links
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
# Strip references inside model answer (UI shows references separately)
# =============================================================================
_INLINE_CITATION_BLOCK_RE = re.compile(
    r"(\(|\[)\s*(sources?|references?|citations?)\s*:\s*.*?(\)|\])",
    re.IGNORECASE | re.DOTALL
)

_CITATION_LINE_RE = re.compile(
    r"^\s*([-*•]\s*)?(sources?|references?|citations?)\s*:\s*.*$",
    re.IGNORECASE | re.MULTILINE
)

_TRAILING_REF_SECTION_RE = re.compile(
    r"(?is)\n\s*(#{1,6}\s*)?(\*\*)?\s*(sources?|references?|citations?)\s*(\*\*)?\s*:?\s*\n.*$"
)

_LINKISH_LINE_RE = re.compile(
    r"^\s*([-*•]\s*)?.*(https?://|/assets/|#page=|open_url)\S.*$",
    re.IGNORECASE | re.MULTILINE
)


def _strip_answer_references(answer_text: str) -> str:
    if not answer_text:
        return answer_text

    txt = answer_text
    txt = re.sub(_TRAILING_REF_SECTION_RE, "", txt)
    txt = re.sub(_INLINE_CITATION_BLOCK_RE, "", txt)
    txt = re.sub(_CITATION_LINE_RE, "", txt)
    txt = re.sub(_LINKISH_LINE_RE, "", txt)

    txt = re.sub(r"[ \t]+\n", "\n", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    txt = re.sub(r"[ \t]{2,}", " ", txt)

    return txt.strip()


# =============================================================================
# Support-State Classification
# =============================================================================
def _classify_support_state(
    grounding: GroundingResult,
    answer_text: str,
) -> str:
    """
    Classify the answer into one of 4 support states:
      - "supported": answer is directly supported by evidence
      - "partially_supported": answer is compositionally supported (multiple clauses/pages)
      - "unsupported": evidence does not support the answer
      - "conflicting": evidence contains conflicting information

    This drives wording, decision, and audit metadata.
    """
    # Check for ACTUAL content conflict — not just score similarity.
    # "Conflict risk" from grounding only means multiple docs scored similarly,
    # which is normal and NOT an indicator of conflicting content.
    # Only classify as "conflicting" if semantic check explicitly flagged conflicts
    # in the answer text itself (e.g. LLM noted conflicting provisions).
    answer_lower = answer_text.lower()
    has_conflict_language = any(phrase in answer_lower for phrase in [
        "conflicting", "contradict", "differ on this",
        "inconsistent", "two different", "opposing provisions",
    ])
    if has_conflict_language:
        return "conflicting"

    # Semantic support provides the most reliable signal
    sem = grounding.semantic_support

    if sem == "full":
        return "supported"
    elif sem == "combined":
        return "partially_supported"
    elif sem == "partial":
        # Partial = some support exists. Only refuse if score is very low.
        if grounding.score >= 0.35:
            return "partially_supported"
        else:
            return "unsupported"
    elif sem == "none":
        # Even 'none' from semantic judge should check evidence quality.
        # If evidence quality is decent, the semantic judge may be too strict.
        if grounding.score >= 0.35:
            log.info("Semantic=none but evidence score=%.3f — downgrading to partially_supported", grounding.score)
            return "partially_supported"
        return "unsupported"

    # When semantic check was not run, use score-based classification
    if grounding.score >= 0.65:
        return "supported"
    elif grounding.score >= 0.35:
        return "partially_supported"
    elif grounding.score < 0.25:
        return "unsupported"
    else:
        return "partially_supported"


# =============================================================================
# Self-Refusal Stripping (catches ALL LLM-generated refusal language)
# =============================================================================
_CONTRADICTORY_DISCLAIMER_PATTERNS = [
    # "The provided PERA documents do not explicitly define/mention/state X"
    re.compile(r"^.*?(?:the\s+)?provided\s+(?:PERA\s+)?(?:official\s+)?(?:documents?|context)\s+do(?:es)?\s+not\s+(?:explicitly\s+)?(?:define|mention|state|specify|address|contain|detail|cover).*?[.\n]", re.IGNORECASE | re.MULTILINE),
    # "This/it is not explicitly mentioned/defined/detailed"
    re.compile(r"^.*?(?:this|it|the\s+position)\s+(?:is|are)\s+not\s+(?:explicitly\s+)?(?:mentioned|defined|stated|specified|detailed|covered|available|found).*?[.\n]", re.IGNORECASE | re.MULTILINE),
    # "not explicitly/specifically/directly defined in the provided/available/PERA..."
    re.compile(r"^.*?not\s+(?:explicitly|specifically|directly)\s+(?:defined|mentioned|stated|addressed|covered|detailed|found|available)\s+in\s+(?:the\s+)?(?:provided|available|PERA|given).*?[.\n]", re.IGNORECASE | re.MULTILINE),
    # "no specific/explicit mention/definition/provision"
    re.compile(r"^.*?(?:no\s+specific|no\s+explicit|no\s+direct)\s+(?:mention|definition|provision|clause|detail|information).*?[.\n]", re.IGNORECASE | re.MULTILINE),
    # "do not contain information" / "does not contain"
    re.compile(r"^.*?(?:do(?:es)?\s+not\s+contain|cannot\s+find|could\s+not\s+find|unable\s+to\s+find)\s+(?:information|details?|data).*?[.\n]", re.IGNORECASE | re.MULTILINE),
    # "insufficient information" / "not available"
    re.compile(r"^.*?(?:insufficient|inadequate)\s+(?:information|details?|evidence|data).*?[.\n]", re.IGNORECASE | re.MULTILINE),
    # "If you need specific information about X, please clarify"
    re.compile(r"^.*?(?:if you need|please\s+(?:clarify|provide|specify)).*?(?:specific|additional|more)\s+(?:information|context|details?).*?[.\n]", re.IGNORECASE | re.MULTILINE),
    # "However, if you are referring to..." type hedging
    re.compile(r"^However,\s+(?:if\s+you\s+are\s+referring|this\s+is\s+not).*?[.\n]", re.IGNORECASE | re.MULTILINE),
]


def _strip_contradictory_disclaimers(answer_text: str) -> str:
    """
    Remove refusal-style disclaimers from answers that are actually supported.
    Only called for 'supported' and 'partially_supported' states.
    """
    txt = answer_text
    for pat in _CONTRADICTORY_DISCLAIMER_PATTERNS:
        txt = pat.sub("", txt)

    # Clean up leading whitespace/newlines after stripping
    txt = re.sub(r"^\s*\n+", "", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()


# =============================================================================
# Wording Notes per Support State (appended at end, not bold)
# =============================================================================
_PARTIAL_SUPPORT_NOTE = (
    "\n\nNote: This answer is derived from the relevant PERA provisions and is not stated "
    "as a single standalone clause in the documents."
)

_CONFLICTING_NOTE = (
    "\n\nNote: The PERA documents contain provisions that may differ on this matter. "
    "The relevant positions are presented above."
)


def _apply_support_state_wording(answer_text: str, support_state: str) -> str:
    """
    Apply clean, professional wording based on support state.
    - supported: answer as-is, no notes
    - partially_supported: answer first, plain note at end
    - conflicting: answer first, plain conflict note at end
    - unsupported: replaced entirely (caller handles this)
    """
    if support_state == "supported":
        # Clean answer, no notes
        return _strip_contradictory_disclaimers(answer_text)

    elif support_state == "partially_supported":
        cleaned = _strip_contradictory_disclaimers(answer_text)
        return cleaned + _PARTIAL_SUPPORT_NOTE

    elif support_state == "conflicting":
        cleaned = _strip_contradictory_disclaimers(answer_text)
        return cleaned + _CONFLICTING_NOTE

    # unsupported — caller should not reach here
    return answer_text


# =============================================================================
# Main Answer Function
# =============================================================================
def answer_question(
    current_question: str,
    retrieval: Dict[str, Any],
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    client = get_chat_client()

    # 0. Creator question intercept
    if _is_creator_question(current_question):
        return {"answer": _CREATOR_RESPONSE, "references": [], "decision": "answer",
                "support_state": "supported"}

    # 0b. Short-query expansion: "pera?" → "What is PERA?", "SSO?" → "Tell me about System Support Officer"
    # Ultra-short queries confuse the LLM into refusing them as "too vague"
    q_stripped = re.sub(r'[?!.\s]+$', '', current_question).strip().lower()
    if len(q_stripped.split()) <= 2:
        _SHORT_EXPANSIONS = {
            "pera": "What is PERA (Punjab Enforcement and Regulatory Authority)?",
        }
        # Check direct mapping first
        if q_stripped in _SHORT_EXPANSIONS:
            current_question = _SHORT_EXPANSIONS[q_stripped]
            log.info("Short-query expanded: '%s' -> '%s'", q_stripped, current_question)
        else:
            # For abbreviations like "SSO?", "CTO?", expand to "Tell me about {expanded}"
            _ABBREV_SHORT = {
                "cto": "Chief Technology Officer", "dg": "Director General",
                "ddg": "Deputy Director General", "adg": "Additional Director General",
                "dd": "Deputy Director", "eo": "Enforcement Officer",
                "io": "Investigation Officer", "sso": "System Support Officer",
                "sdeo": "Sub Divisional Enforcement Officer", "deo": "Data Entry Operator",
                "dba": "Database Administrator", "se": "Software Engineer",
            }
            if q_stripped in _ABBREV_SHORT:
                current_question = f"Tell me about the {_ABBREV_SHORT[q_stripped]} position at PERA"
                log.info("Short-query expanded (abbrev): '%s' -> '%s'", q_stripped, current_question)

    # 1. Build Context
    context_str = format_evidence_for_llm(retrieval, question=current_question)
    if not context_str:
        return {
            "answer": "I'm sorry, I couldn't find any information about that in the PERA documents.",
            "references": [],
            "decision": "refuse",
            "support_state": "unsupported"
        }

    # 1b. Pre-generation evidence quality gate — REMOVED.
    # If FAISS found evidence and it passed into context, the LLM should always try.
    # The only refuse point is empty context (no evidence at all).
    evidence_list = retrieval.get("evidence", [])

    # 2. System Prompt (v5: balanced — accurate with coverage)
    system_prompt = (
        "You are the PERA AI Assistant for Punjab Enforcement and Regulatory Authority (PERA). "
        "You specialize in answering questions about PERA's regulations, structure, roles, operations, and policies.\n\n"

        "CORE RULES\n"
        "1) Answer using ONLY the provided Context. Do not use external knowledge.\n"
        "2) Do not invent facts (powers, procedures, numbers, dates, thresholds, authorities).\n"
        "3) Do NOT infer authority from seniority or job title unless explicitly stated in Context.\n\n"

        "ANSWER APPROACH\n"
        "4) The Context below contains pre-selected relevant evidence. Your PRIMARY JOB is to extract "
        "and present information from it. ALWAYS give a substantive answer when evidence is present.\n"
        "5) If the answer requires combining information from multiple evidence sections, clauses, "
        "or pages, do so. This is normal for regulatory documents where salary tables, appointment "
        "rules, and job descriptions are on different pages.\n"
        "6) SALARY/PAY RULES: If the Context contains salary, pay scale, SPPP, BPS, or appointment "
        "references for the asked position, STATE THE EXACT VALUES directly (e.g., 'SPPP-3', "
        "'BPS-17', 'BS-20'). Do NOT say 'specific details not provided' when the values ARE in Context.\n"
        "7) If the Context contains conflicting statements, present both positions neutrally.\n"
        "8) If the question is completely unrelated to PERA (weather, sports, cooking), "
        "say the question is outside PERA's scope.\n\n"

        "CRITICAL: AVOID UNNECESSARY REFUSALS\n"
        "9) NEVER say 'not explicitly defined', 'not explicitly mentioned', "
        "'not found in the documents', 'insufficient information', or "
        "'cannot provide specific details' if the Context has ANY relevant content.\n"
        "10) If the Context only partially covers the topic, answer what IS available "
        "and clearly state what specific aspect is not covered.\n"
        "11) When salary/pay data exists in the Context, ALWAYS include it in your answer. "
        "The salary data may appear in a DIFFERENT evidence section than the job description — "
        "scan ALL evidence sections for the relevant pay scale information.\n\n"

        "INTERPRETATION\n"
        "12) Treat 'powers', 'functions', and 'duties' as synonyms unless Context distinguishes them.\n"
        "13) For roles: if Context describes a position's purpose, responsibilities, qualifications, "
        "salary, or reporting structure, present ALL of it as the answer.\n\n"

        "REFERENCES — DO NOT INCLUDE IN ANSWER\n"
        "14) The UI shows references separately. Do NOT output Source/References/page numbers.\n"
        "15) Only mention document/page if the USER explicitly asks for it.\n\n"

        "STYLE\n"
        "16) Professional, composed, concise. Use Markdown formatting.\n"
        "17) ALWAYS answer in English, regardless of the language the user asked in. "
        "Use professional, clear English. If the user asked in Urdu or Roman Urdu, "
        "still provide the answer in English for maximum accuracy.\n"
        "18) When evidence says 'Salary and Benefits: Pay & Benefits equivalent to BPS-XX' or "
        "'as explained in column 5 of Schedule-II', interpret this as: the salary IS the BPS/SPPP "
        "scale mentioned. State it directly as the answer.\n\n"

        "COMPLETENESS (CRITICAL)\n"
        "19) EXHAUSTIVE EXTRACTION: If the Context contains a LIST of items, responsibilities, "
        "conditions, qualifications, or steps — include ALL of them in your answer. Do NOT "
        "truncate or summarize lists. Present every item.\n"
        "20) MULTI-SECTION SYNTHESIS: Information about a role or topic may be spread across MULTIPLE "
        "evidence sections. Scan ALL evidence sections and combine: job description + salary + "
        "qualifications + appointment + reporting structure + KPIs. Present the COMPLETE picture.\n"
        "21) SECTION COVERAGE: If the Context contains section headings like 'Purpose of the Position', "
        "'Key Responsibilities', 'Qualification/Experience', 'Salary and Benefits', or 'Appointment' — "
        "extract and present ALL content under each relevant heading.\n"
        "22) MULTI-PART QUESTIONS: If the user asks about multiple aspects (e.g., 'salary and duties'), "
        "address EVERY aspect separately. Do not skip any part of the question.\n"
        "23) ENTITY-SALARY LINKAGE: When a position description chunk says 'SPPP-X' or 'BPS-XX' "
        "or 'as mentioned in Schedule-III', and a separate evidence section contains the Schedule-III "
        "salary table with SPPP pay ranges — LINK them. State the SPPP level from the position "
        "chunk AND the corresponding pay range from the salary table. Do NOT say 'salary not found' "
        "when the SPPP/BPS level IS mentioned in the position evidence.\n\n"

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
            temperature=0.0,
        )
        answer_text = response.choices[0].message.content or ""

        # --- Conditional Multi-Pass Refinement ---
        # Triggers when evidence is rich but answer is short, indicating missed information.
        # Now includes salary queries since salary-bridge ensures correct context.
        evidence_section_count = context_str.count("<evidence ")
        answer_word_count = len(answer_text.split())
        if evidence_section_count >= 5 and answer_word_count < 100:
            log.info("Refinement triggered: %d evidence sections but only %d words in answer",
                     evidence_section_count, answer_word_count)
            refinement_prompt = (
                "You previously answered the question below. However, the context contains "
                f"{evidence_section_count} evidence sections and your answer may be incomplete.\n\n"
                "TASK: Review your answer against ALL the context sections below. "
                "If ANY relevant information from the context is missing from your answer, "
                "produce an EXPANDED answer that includes ALL missing details. "
                "Do NOT just repeat your previous answer — ADD the missing information.\n"
                "If your answer is already complete, return it as-is.\n\n"
                f"ORIGINAL QUESTION: {current_question}\n\n"
                f"YOUR PREVIOUS ANSWER:\n{answer_text}\n\n"
                "CONTEXT (same as before):\n"
                f"{context_str}"
            )
            try:
                refine_response = client.chat.completions.create(
                    model=ANSWER_MODEL,
                    messages=[{"role": "system", "content": refinement_prompt}],
                    temperature=0.0,
                )
                refined = refine_response.choices[0].message.content or ""
                # Only use refinement if it's meaningfully longer
                if len(refined.split()) > answer_word_count + 20:
                    log.info("Refinement expanded answer: %d → %d words",
                             answer_word_count, len(refined.split()))
                    answer_text = refined
                else:
                    log.info("Refinement did not expand answer — keeping original")
            except Exception as e:
                log.warning("Refinement pass failed: %s — using initial answer", e)

        # Strip references ONLY if user did NOT explicitly ask for them
        if not _user_wants_references(current_question):
            answer_text = _strip_answer_references(answer_text)

        # 5. Post-generation grounding verification (AUDIT ONLY — never refuses)
        grounding = verify_grounding(
            answer_text=answer_text,
            evidence_list=evidence_list,
            context_str=context_str,
            question=current_question,
        )

        # 6. Classify support state (for audit and wording — NOT for refusal)
        support_state = _classify_support_state(grounding, answer_text)
        log.info("Support state: %s (grounding score=%.3f, semantic=%s)",
                 support_state, grounding.score, grounding.semantic_support or "n/a")

        refs = extract_references_simple(retrieval, question=current_question)

        # 7. CRITICAL DESIGN RULE:
        # If evidence was found and the LLM generated an answer, ALWAYS show it.
        # Grounding is audit metadata, NOT a kill switch.
        # The only refusal point is the pre-generation gate above (no evidence at all).
        #
        # For a government chatbot, refusing when evidence exists is worse than
        # showing a potentially imperfect answer with references the user can verify.

        # Apply wording based on support state (never replaces answer with refusal)
        if support_state == "unsupported":
            # Even for "unsupported" grounding, show the LLM's answer with a note
            support_state = "partially_supported"
            log.info("Grounding flagged unsupported but evidence exists — showing answer anyway")

        final_answer = _apply_support_state_wording(answer_text, support_state)

        decision = "answer"

        result = {
            "answer": final_answer,
            "references": refs,
            "decision": decision,
            "support_state": support_state,
            "grounding": {
                "confidence": grounding.confidence,
                "score": grounding.score,
                "semantic_support": grounding.semantic_support,
                "support_state": support_state,
            },
        }

        if grounding.confidence == "low":
            log.info("Low grounding confidence (%.3f) — answer shown with note", grounding.score)

        return result

    except Exception as e:
        log.error("LLM call failed: %s", e, exc_info=True)
        return {
            "answer": "I encountered an error while processing your request. Please try again.",
            "references": [],
            "decision": "error",
            "support_state": "error"
        }
