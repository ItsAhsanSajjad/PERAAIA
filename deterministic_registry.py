"""
Deterministic Registry — Guaranteed content lookup bypassing FAISS.
Coverage Lock: every known-in-doc query routes here before RAG fallback.
Exports: route_intent(), registry_answer()
"""

import re, os, json
from typing import Optional, Dict, List, Tuple, Any

# ─── Intent constants ─────────────────────────────────────────────
INTENT_SCHEDULE_LOOKUP = "SCHEDULE_LOOKUP"
INTENT_JD_LOOKUP       = "JD_LOOKUP"
INTENT_ACT_SECTION     = "ACT_SECTION"
INTENT_ACT_DEFINITION  = "ACT_DEFINITION"   # "act mean", "authority mean"
INTENT_PERA_DEFINITION = "PERA_DEFINITION"
INTENT_PERA_COMMENCEMENT = "PERA_COMMENCEMENT"
INTENT_NAMED_PERSON    = "NAMED_PERSON"
INTENT_FALLBACK        = "FALLBACK"

# ─── Schedule regex aliases (typo-tolerant) ───────────────────────
_SCHED_NUM = {
    "1": "1", "i": "1", "one": "1",
    "2": "2", "ii": "2", "two": "2",
    "3": "3", "iii": "3", "three": "3",
    "4": "4", "iv": "4", "four": "4",
    "5": "5", "v": "5", "five": "5",
}

_SCHEDULE_RE = re.compile(
    r"(?:schedule|sehdule|schdule|sched|shdule|schedual|schedl|shcedule|schudule|shedule)"
    r"[\s\-_]*(1|2|3|4|5|i{1,3}|iv|v|one|two|three|four|five)\b",
    re.IGNORECASE,
)

# ─── SPPP salary patterns ────────────────────────────────────────
_SPPP_SALARY_RE = re.compile(
    r"sppp[\s\-]?([1-6])\b.*?"
    r"(?:salary|pay|scale|tankha|tankhwah|maash|kitni|kitna|compensation|number)",
    re.IGNORECASE,
)

# ─── JD role aliases (expanded) ──────────────────────────────────
_TITLE_ALIASES: Dict[str, str] = {
    # SSO
    "sso":                        "System Support Officer",
    "system support officer":     "System Support Officer",
    "system support":             "System Support Officer",
    # CTO
    "cto":                        "Chief Technology Officer",
    "chief technology officer":   "Chief Technology Officer",
    "chief technology":           "Chief Technology Officer",
    # Android
    "android":                    "Android Developer",
    "android developer":          "Android Developer",
    "android dev":                "Android Developer",
    # Manager (Development)
    "manager development":        "Manager (Development)",
    "manager (development)":      "Manager (Development)",
    "manager dev":                "Manager (Development)",
    # DG (JD aspects only — name queries caught earlier)
    "director general":           "Director General",
    "dg":                         "Director General",
    # Additional roles (extend as needed):
    "graphic designer":           "Graphic Designer",
    "executive assistant":        "Executive Assistant",
    "hearing officer":            "Hearing Officer",
    "enforcement officer":        "Enforcement Officer",
    "investigation officer":      "Investigation Officer",
    "sergeant":                   "Sergeant",
}

# ─── Act section — expanded regex ─────────────────────────────────
# Matches: "section 10", "sec 10", "s. 10", "s10", "Section 10(1)"
_ACT_SECTION_RE = re.compile(
    r"\b(?:section|sec|s\.?)\s*(\d+)(?:\s*\(\d+\))?\b",
    re.IGNORECASE,
)

# ─── Act section keyword map (TC lookup) ──────────────────────────
_ACT_SECTION_KEYWORDS = {
    "short title":            ["section 1", "Short title"],
    "extend":                 ["section 1", "extend"],
    "commencement":           ["section 1", "come into force"],
    "come into force":        ["section 1", "come into force"],
    "definition":             ["section 2", "definitions"],
    "definitions":            ["section 2", "definitions"],
    "establishment":          ["section 3", "establish"],
    "establishment of pera":  ["section 3", "establish"],
    "chairperson":            ["section 4", "Chairperson"],
    "selection panel":        ["section 5", "Selection Panel"],
    "meeting":                ["section 6", "meetings"],
    "authority power":        ["section 10", "powers and functions"],
    "authority function":     ["section 10", "powers and functions"],
    "powers and functions":   ["section 10", "powers and functions"],
    "powers of authority":    ["section 10", "powers and functions"],
    "board":                  ["section 12", "Board"],
    "hearing officer":        ["section 16", "Hearing Officer"],
    "enforcement station":    ["section 19", "Enforcement Station"],
    "requisition":            ["section 20", "requisition"],
    "enforcement officer":    ["section 22", "Enforcement Officer"],
    "investigation officer":  ["section 25", "Investigation Officer"],
    "sergeant":               ["section 26", "Sergeant"],
    "enforcement squad":      ["section 28", "Enforcement Squad"],
    "arrest":                 ["section 31", "arrest"],
    "procedure after arrest": ["section 32", "after arrest"],
    "public nuisance":        ["section 34", "public nuisance"],
    "nuisance":               ["section 34", "public nuisance"],
    "epo":                    ["section 35", "EPO"],
    "encroachment":           ["section 37", "encroachment"],
    "moveable encroachment":  ["section 38", "moveable encroachment"],
    "immovable encroachment": ["section 39", "immovable"],
    "notice":                 ["section 40", "notice"],
    "penalty":                ["section 41", "penalty"],
    "fine":                   ["section 41", "fine"],
    "penalties":              ["section 41", "penalty"],
    "state property":         ["section 42", "state property"],
    "bar of jurisdiction":    ["section 50", "jurisdiction"],
    "representation":         ["section 51", "representation"],
    "whistle":                ["section 59", "whistle"],
    "whistleblower":          ["section 59", "whistle"],
    "fund":                   ["section 69", "fund"],
    "delegation":             ["section 74", "delegation"],
    "rules":                  ["section 78", "rules"],
    "regulations":            ["section 79", "regulations"],
    "scheduled laws":         ["SCHEDULE", "scheduled laws"],
}

# ─── PERA definition regex (expanded) ─────────────────────────────
_PERA_DEF_RE = re.compile(
    r"(?:"
    r"what\s+is\s+pera|pera\s+kia\s+(?:hy|hai|he|h)|pera\s+kya\s+(?:hai|he|h|hy)"
    r"|pera\s+definition|pera\s+meaning|pera\s+ka\s+matlab"
    r"|pera\s+hai|pera\s+kia\s+kam"
    r"|punjab\s+enforcement.*?(?:kia|kya|what)"
    r"|goal\s+of\s+pera|purpose\s+of\s+pera"
    r"|pera\s+k[iy]a\s+krt[ia]"
    r")",
    re.IGNORECASE,
)

# ─── PERA commencement regex ──────────────────────────────────────
_PERA_WHEN_RE = re.compile(
    r"(?:"
    r"pera\s+k[ab]b?\s+ban[ia]"
    r"|when\s+(?:was\s+)?pera\s+establish"
    r"|pera\s+establish"
    r"|pera\s+commencement"
    r"|pera\s+k[ab]b\s+shuru"
    r"|when\s+(?:did\s+)?pera\s+(?:come|start|begin)"
    r"|pera\s+ban[ia]\s+k[ab]b"
    r")",
    re.IGNORECASE,
)

# ─── Named person regex ──────────────────────────────────────────
_NAMED_PERSON_RE = re.compile(
    r"(?:"
    r"salma\s*butt"
    r"|dg\s+ka\s+nam|dg\s+name|dg\s+kon|dg\s+kaun"
    r"|who\s+is\s+(?:the\s+)?(?:current\s+)?dg"
    r"|director\s+general\s+(?:name|nam|kon|kaun)"
    r"|who\s+is\s+(?:the\s+)?director\s+general"
    r"|dg\s+ka\s+nam\s+kia"
    r")",
    re.IGNORECASE,
)

# ─── DG reporting regex ──────────────────────────────────────────
_DG_REPORT_RE = re.compile(
    r"(?:"
    r"dg\s+(?:kisko|kis\s+ko)\s+report"
    r"|dg\s+report(?:ing)?\s+(?:hy|hai|to|structure)"
    r"|director\s+general\s+report"
    r")",
    re.IGNORECASE,
)

# ─── Act definition regex ("act mean", "authority mean") ──────────
_ACT_DEF_RE = re.compile(
    r"\b(?:act|authority|authorized\s*officer|complainant|board|hearing\s*officer"
    r"|enforcement\s*officer|investigation\s*officer|scheduled\s*law"
    r"|encroachment|public\s*nuisance|state\s*property"
    r")\s*(?:mean|means|ka\s+matlab|kia\s+(?:hai|hy|he|h)|ki\s+definition"
    r"|define|defined)\b",
    re.IGNORECASE,
)

# Extract which term user wants defined
_ACT_DEF_TERM_RE = re.compile(
    r"\b(act|authority|authorized\s*officer|complainant|board|hearing\s*officer"
    r"|enforcement\s*officer|investigation\s*officer|scheduled\s*law"
    r"|encroachment|public\s*nuisance|state\s*property"
    r")\s*(?:mean|means|ka\s+matlab|kia|ki\s+definition|define|defined)\b",
    re.IGNORECASE,
)


# ═════════════════════════════════════════════════════════════════
#  Schedule-III data (canonical, verified from docs)
# ═════════════════════════════════════════════════════════════════
SCHEDULE_III_ROWS = [
    {"sppp": "SPPP-1", "min": "350,082", "max": "1,000,082", "fuel": "300"},
    {"sppp": "SPPP-2", "min": "296,324", "max": "746,324",   "fuel": "250"},
    {"sppp": "SPPP-3", "min": "250,051", "max": "500,051",   "fuel": "200"},
    {"sppp": "SPPP-4", "min": "115,561", "max": "250,561",   "fuel": "150"},
    {"sppp": "SPPP-5", "min": "90,219",  "max": "200,219",   "fuel": "100"},
]

SCHEDULE_III_TEXT = (
    "Schedule-III Special Pay Package PERA (SPPP) Summary\n"
    "Sr. No. | SPPP | Minimum Pay per Month (PKR) | Maximum Pay per Month (PKR) | Fuel Limit (Petrol Liters/Month)\n"
)
for _i, _r in enumerate(SCHEDULE_III_ROWS, 1):
    SCHEDULE_III_TEXT += f"{_i} | {_r['sppp']} | {_r['min']} | {_r['max']} | {_r['fuel']}\n"
SCHEDULE_III_TEXT += "6 | Interns | As decided by the Contractual Employment Committee\n"

SCHEDULE_III_DOC = "THE PUNJAB ENFORCEMENT AND REGULATORY AUTHORITY - APPOINTMENT AND CONDITIONS OF SERVICE FOR CONTRACTUAL EMPLOYEES - AMENDED REGULATIONS 2025.pdf"

# ═════════════════════════════════════════════════════════════════
#  JD Registry  (keyed by canonical title)
# ═════════════════════════════════════════════════════════════════
_JD_SALARY_MAP: Dict[str, Dict[str, Any]] = {
    "System Support Officer": {
        "sppp": "SPPP-5",
        "salary_text": "SPPP-5 + any other benefit admissible to SPPP-5 as mentioned in Schedule-III",
        "salary_doc": SCHEDULE_III_DOC,
    },
    "Chief Technology Officer": {
        "sppp": "SPPP-3",
        "salary_text": "SPPP-3 + any other benefit admissible to SPPP-3 as mentioned in Schedule-III",
        "salary_doc": SCHEDULE_III_DOC,
    },
    "Android Developer": {
        "sppp": "SPPP-4",
        "salary_text": "SPPP-4 + any other benefit admissible to SPPP-4 as mentioned in Schedule-III",
        "salary_doc": SCHEDULE_III_DOC,
    },
    "Manager (Development)": {
        "sppp": "SPPP-3",
        "salary_text": "SPPP-3 + any other benefit admissible to SPPP-3 as mentioned in Schedule-III",
        "salary_doc": SCHEDULE_III_DOC,
    },
}


# ═════════════════════════════════════════════════════════════════
#  INTENT ROUTER
# ═════════════════════════════════════════════════════════════════
def route_intent(query: str) -> Tuple[str, dict]:
    """
    Return (intent_type, metadata).
    Ordered so the most specific patterns win first.
    """
    q = query.strip()
    ql = q.lower()

    # 1) Schedule lookup (typo-tolerant)
    m = _SCHEDULE_RE.search(ql)
    if m:
        raw_num = m.group(1).lower()
        num = _SCHED_NUM.get(raw_num, raw_num)
        return INTENT_SCHEDULE_LOOKUP, {"schedule_num": num}

    # 1b) SPPP salary → Schedule-III with specific row
    sppp_m = _SPPP_SALARY_RE.search(ql)
    if sppp_m:
        return INTENT_SCHEDULE_LOOKUP, {"schedule_num": "3", "sppp_row": sppp_m.group(1)}

    # 2) Named person queries (before JD to catch "DG name" early)
    if _NAMED_PERSON_RE.search(ql):
        return INTENT_NAMED_PERSON, {"sub": "name"}

    # 2b) DG reporting
    if _DG_REPORT_RE.search(ql):
        return INTENT_NAMED_PERSON, {"sub": "reporting"}

    # 3) PERA commencement (before definition — more specific)
    if _PERA_WHEN_RE.search(ql):
        return INTENT_PERA_COMMENCEMENT, {}

    # 4) PERA definition
    if _PERA_DEF_RE.search(ql):
        return INTENT_PERA_DEFINITION, {}

    # 5) Act term definition ("act mean in authority", "authority means kya")
    adm = _ACT_DEF_TERM_RE.search(ql)
    if adm:
        term = adm.group(1).strip().lower()
        return INTENT_ACT_DEFINITION, {"term": term}

    # 6) JD lookup — resolve title from aliases
    resolved_title = _resolve_job_title(ql)
    if resolved_title:
        # DG role/salary queries go to JD intent
        aspect = _detect_jd_aspect(ql)
        return INTENT_JD_LOOKUP, {"job_title": resolved_title, "aspect": aspect}

    # 7) Act section (numeric or keyword)
    sec_match = _ACT_SECTION_RE.search(ql)
    if sec_match:
        return INTENT_ACT_SECTION, {"section_num": sec_match.group(1)}

    for kw, meta_val in _ACT_SECTION_KEYWORDS.items():
        if kw in ql:
            return INTENT_ACT_SECTION, {"keyword": kw, "search_terms": meta_val}

    # 8) Fallback
    return INTENT_FALLBACK, {}


def _resolve_job_title(ql: str) -> Optional[str]:
    """Resolve a job title from query text using alias dictionary."""
    # Longest-first so "chief technology officer" matches before "cto"
    for alias in sorted(_TITLE_ALIASES.keys(), key=len, reverse=True):
        if alias in ql:
            return _TITLE_ALIASES[alias]
    return None


def _detect_jd_aspect(ql: str) -> str:
    """Detect what aspect of the JD the user is asking about."""
    salary_kw = ["salary", "pay", "scale", "tankhwah", "tankha", "maash",
                 "compensation", "kitni", "kitna", "number mein", "number me",
                 "number", "numbers", "rakam", "paisa", "rupees", "pkr"]
    for kw in salary_kw:
        if kw in ql:
            return "salary"
    role_kw = ["role", "responsibilities", "responsibility", "kaam", "zimmedar",
               "duty", "duties", "function", "functions", "power", "powers",
               "work", "karta", "krti", "kia karta", "kia krti"]
    for kw in role_kw:
        if kw in ql:
            return "role"
    qual_kw = ["qualification", "education", "degree", "experience", "eligibility"]
    for kw in qual_kw:
        if kw in ql:
            return "qualification"
    report_kw = ["report", "reports to", "boss", "supervisor", "under",
                 "kisko report", "kis ko report"]
    for kw in report_kw:
        if kw in ql:
            return "reports_to"
    return "general"


# ═════════════════════════════════════════════════════════════════
#  CHUNK-BASED REGISTRY (loads chunks at startup)
# ═════════════════════════════════════════════════════════════════
_CHUNKS: List[dict] = []
_CHUNKS_LOADED = False


def _ensure_chunks():
    """Load chunks once."""
    global _CHUNKS, _CHUNKS_LOADED
    if _CHUNKS_LOADED:
        return
    try:
        from index_store import load_index_and_chunks
        from retriever import _resolve_index_dir
        _, _CHUNKS = load_index_and_chunks(_resolve_index_dir(None))
        _CHUNKS_LOADED = True
        print(f"[Registry] Loaded {len(_CHUNKS)} chunks")
    except Exception as e:
        print(f"[Registry] WARNING: could not load chunks: {e}")
        _CHUNKS_LOADED = True  # don't retry


def _get_chunk(idx: int) -> dict:
    """Get a chunk by index."""
    _ensure_chunks()
    if 0 <= idx < len(_CHUNKS):
        return _CHUNKS[idx]
    return {}


def _search_chunks_keyword(terms: List[str], max_results: int = 5) -> List[Tuple[int, dict]]:
    """Search chunks by keyword matching (all terms must be present)."""
    _ensure_chunks()
    results = []
    for i, c in enumerate(_CHUNKS):
        text_lower = (c.get("text", "") or "").lower()
        if all(t.lower() in text_lower for t in terms):
            results.append((i, c))
            if len(results) >= max_results:
                break
    return results


def _search_chunks_any(terms: List[str], max_results: int = 5) -> List[Tuple[int, dict]]:
    """Search chunks where ANY term matches (OR logic). Scored by match count."""
    _ensure_chunks()
    scored = []
    for i, c in enumerate(_CHUNKS):
        text_lower = (c.get("text", "") or "").lower()
        ct = sum(1 for t in terms if t.lower() in text_lower)
        if ct > 0:
            scored.append((ct, i, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [(i, c) for _, i, c in scored[:max_results]]


# ═════════════════════════════════════════════════════════════════
#  CLAIM BUILDER
# ═════════════════════════════════════════════════════════════════

def _make_claim(quote: str, doc: str, page: int, chunk_id: int,
                answer_span: str, source: str = "registry",
                page_end: int = None, tier: str = "registry") -> dict:
    """Create a claim dict consistent with pipeline expectations."""
    start = 0
    end = len(quote)
    if isinstance(page, int) and page < 1:
        page = 1
    if page_end is None:
        page_end = page
    return {
        "answer_span": answer_span,
        "doc": doc,
        "page": page,
        "page_end": page_end,
        "chunk_id": int(chunk_id) if chunk_id is not None else 0,
        "quote": quote,
        "quote_offsets": [start, end],
        "_source": source,
        "tier": tier,
    }


def _chunk_claim(idx: int, c: dict, answer_span: str, source: str,
                 max_quote: int = 500) -> dict:
    """Shorthand: build claim from a chunk tuple (idx, chunk_dict)."""
    text = c.get("text", "")
    quote = text[:max_quote].strip() if len(text) > max_quote else text.strip()
    return _make_claim(
        quote=quote,
        doc=c.get("doc_name", ""),
        page=c.get("loc_start", 1),
        page_end=c.get("loc_end", c.get("loc_start", 1)),
        chunk_id=idx,
        answer_span=answer_span,
        source=source,
    )


def _sched3_chunk_meta() -> Tuple[int, int, int, str]:
    """Dynamic lookup for Schedule-III chunk → (chunk_id, page, page_end, doc)."""
    hits = _search_chunks_keyword(["Schedule-III", "SPPP", "Minimum"], max_results=1)
    if hits:
        cidx, c = hits[0]
        return cidx, c.get("loc_start", 1), c.get("loc_end", c.get("loc_start", 1)), c.get("doc_name", SCHEDULE_III_DOC)
    return 0, 1, 1, SCHEDULE_III_DOC


# ═════════════════════════════════════════════════════════════════
#  DETERMINISTIC ANSWER GENERATORS
# ═════════════════════════════════════════════════════════════════

def _answer_schedule(meta: dict) -> Optional[dict]:
    """Deterministic answer for schedule queries."""
    num = meta.get("schedule_num", "3")
    sppp_row = meta.get("sppp_row")

    if num == "3":
        if sppp_row and sppp_row.isdigit() and 1 <= int(sppp_row) <= 5:
            row = SCHEDULE_III_ROWS[int(sppp_row) - 1]
            quote = (
                f"Schedule-III Special Pay Package PERA (SPPP) Summary\n"
                f"{row['sppp']} | Minimum Pay per Month (PKR): {row['min']} | "
                f"Maximum Pay per Month (PKR): {row['max']} | "
                f"Fuel Limit: {row['fuel']} Liters/Month"
            )
            span = "Schedule-III SPPP pay table (" + row["sppp"] + ")"
        else:
            quote = SCHEDULE_III_TEXT.strip()
            span = "Schedule-III SPPP pay table"

        cidx, pg, pg_end, doc = _sched3_chunk_meta()
        claims = [_make_claim(quote=quote, doc=doc, page=pg, page_end=pg_end,
                              chunk_id=cidx, answer_span=span, source="schedule_registry")]
        debug = {"source": "schedule_registry", "schedule": "III"}
        if sppp_row:
            debug["sppp_row"] = sppp_row
        return {"claims": claims, "debug": debug}

    # Schedule I / II / IV / V — keyword search in chunks
    label_map = {
        "1": ("Schedule-I", "organizational structure"),
        "2": ("Schedule-II", "Terms"),
        "4": ("Schedule-IV", "Promotion"),
        "5": ("Schedule-V", "Transfer"),
    }
    if num in label_map:
        sched_label, extra_kw = label_map[num]
        hits = _search_chunks_keyword([sched_label, extra_kw], max_results=3)
        if not hits:
            hits = _search_chunks_keyword([sched_label], max_results=3)
        if hits:
            claims = [_chunk_claim(idx, c, f"{sched_label} content", "schedule_registry")
                      for idx, c in hits[:2]]
            return {"claims": claims, "debug": {"source": "schedule_registry", "schedule": num}}

    return None


def _answer_jd(meta: dict, query: str) -> Optional[dict]:
    """Deterministic answer for JD queries."""
    title = meta.get("job_title", "")
    aspect = meta.get("aspect", "general")
    ql = query.lower()

    if not title:
        return None

    # Find JD chunks for this title
    search_title = title.lower().replace("(", "").replace(")", "")
    jd_hits = _search_chunks_keyword(["Position Title", search_title], max_results=5)

    if not jd_hits:
        short_title = title.split("(")[0].strip().lower()
        jd_hits = _search_chunks_keyword(["Position Title", short_title], max_results=5)

    if not jd_hits:
        return None

    claims = []

    if aspect == "salary" or "number" in ql or "kitni" in ql or "kitna" in ql:
        # Salary query: get JD salary text + Schedule-III row
        salary_info = _JD_SALARY_MAP.get(title)
        if salary_info:
            sppp = salary_info["sppp"]
            salary_text = salary_info["salary_text"]

            # Find the salary chunk
            salary_hits = _search_chunks_keyword([sppp, "Schedule-III"], max_results=5)
            for idx, c in salary_hits:
                text = c.get("text", "")
                if "Salary and Benefits" in text or sppp in text:
                    quote = text.strip()[:300].strip()
                    claims.append(_make_claim(
                        quote=quote, doc=c.get("doc_name", ""),
                        page=c.get("loc_start", 1),
                        page_end=c.get("loc_end", c.get("loc_start", 1)),
                        chunk_id=idx,
                        answer_span=f"{title} salary: {salary_text}",
                        source="jd_registry",
                    ))
                    break

            # Auto-add Schedule-III row for exact SPPP numbers
            sppp_num = int(sppp.split("-")[1])
            if 1 <= sppp_num <= 5:
                row = SCHEDULE_III_ROWS[sppp_num - 1]
                sched_quote = (
                    f"{row['sppp']} | Minimum: {row['min']} PKR | "
                    f"Maximum: {row['max']} PKR | Fuel: {row['fuel']} Liters/Month"
                )
                cidx, pg, pg_end, doc = _sched3_chunk_meta()
                claims.append(_make_claim(
                    quote=sched_quote, doc=doc, page=pg, page_end=pg_end,
                    chunk_id=cidx,
                    answer_span=f"Schedule-III {sppp}: min {row['min']}, max {row['max']} PKR/month",
                    source="schedule_registry",
                ))
        else:
            # No SPPP mapping — check JD chunks for BPS or salary text
            for idx, c in jd_hits:
                text = (c.get("text", "") or "").lower()
                if "bps" in text or "salary" in text or "pay" in text:
                    chunk_text = c.get("text", "")
                    claims.append(_chunk_claim(idx, c, f"{title} salary information", "jd_registry", 400))
                    break

    elif aspect == "role":
        for idx, c in jd_hits:
            text = c.get("text", "")
            if any(kw in text for kw in ["Areas of Responsibilities", "Purpose of the Position",
                                          "Responsibilities", "Functions", "Duties"]):
                claims.append(_chunk_claim(idx, c, f"{title} role and responsibilities", "jd_registry", 600))
                break
        # If no specific responsibilities chunk found, use the first JD chunk
        if not claims and jd_hits:
            idx, c = jd_hits[0]
            claims.append(_chunk_claim(idx, c, f"{title} job description", "jd_registry", 600))

    elif aspect == "qualification":
        for idx, c in jd_hits:
            text = c.get("text", "")
            if "Qualification" in text:
                claims.append(_chunk_claim(idx, c, f"{title} qualification and experience", "jd_registry"))
                break

    elif aspect == "reports_to":
        for idx, c in jd_hits:
            text = c.get("text", "")
            if "Report" in text:
                lines = text.split("\n")
                report_line = ""
                for line in lines:
                    if "Report" in line and "To" in line:
                        report_line = line.strip()
                        break
                quote = report_line if report_line else text[:300].strip()
                claims.append(_make_claim(
                    quote=quote, doc=c.get("doc_name", ""),
                    page=c.get("loc_start", 1),
                    page_end=c.get("loc_end", c.get("loc_start", 1)),
                    chunk_id=idx,
                    answer_span=f"{title} reports to",
                    source="jd_registry",
                ))
                break

    else:
        # General: return full JD
        for idx, c in jd_hits:
            text = c.get("text", "")
            if "Position Title" in text:
                claims.append(_chunk_claim(idx, c, f"{title} job description", "jd_registry", 600))
                break

    if claims:
        return {
            "claims": claims,
            "debug": {"source": "jd_registry", "title": title, "aspect": aspect},
        }
    return None


def _answer_act_section(meta: dict, query: str) -> Optional[dict]:
    """Deterministic answer for Act section queries."""
    section_num = meta.get("section_num")
    keyword = meta.get("keyword", "")
    search_terms = meta.get("search_terms", [])

    if section_num:
        # Try "Section X"
        hits = _search_chunks_keyword([f"section {section_num}"], max_results=5)
        if not hits:
            # Try "X. " (often used in headings)
            hits = _search_chunks_keyword([f"{section_num}. "], max_results=5)
        
        # Prefer Bill document
        bill_hits = [(i, c) for i, c in hits if "Bill" in c.get("doc_name", "")]
        if bill_hits:
            hits = bill_hits
    elif search_terms:
        hits = _search_chunks_keyword(search_terms, max_results=5)
        bill_hits = [(i, c) for i, c in hits if "Bill" in c.get("doc_name", "")]
        if bill_hits:
            hits = bill_hits
    else:
        return None

    if not hits:
        return None

    claims = [_chunk_claim(idx, c, "PERA Act section reference", "act_registry")
              for idx, c in hits[:2]]

    return {
        "claims": claims,
        "debug": {"source": "act_registry", "section": section_num or keyword},
    }


def _answer_act_definition(meta: dict) -> Optional[dict]:
    """Answer 'act mean in authority', 'authority mean kya' etc."""
    term = meta.get("term", "act")

    # Search for the term in definitions section
    hits = _search_chunks_keyword(["definitions", term], max_results=5)
    bill_hits = [(i, c) for i, c in hits if "Bill" in c.get("doc_name", "")]
    if bill_hits:
        hits = bill_hits

    if not hits:
        # Try broader search
        hits = _search_chunks_keyword([f'"{term}" means'], max_results=3)
        if not hits:
            hits = _search_chunks_keyword([term, "means"], max_results=3)

    if not hits:
        return None

    claims = [_chunk_claim(idx, c, f"Definition of '{term}' in PERA Act", "act_registry")
              for idx, c in hits[:2]]

    return {
        "claims": claims,
        "debug": {"source": "act_registry", "sub": "definition", "term": term},
    }


def _answer_pera_definition(meta: dict = None) -> Optional[dict]:
    """Answer 'what is PERA' / 'pera kia hai'."""
    hits = _search_chunks_keyword(
        ["Punjab Enforcement and Regulatory Authority", "establishment"],
        max_results=3,
    )
    if not hits:
        hits = _search_chunks_keyword(["Punjab Enforcement", "Authority", "Bill"], max_results=3)
    bill_hits = [(i, c) for i, c in hits if "Bill" in c.get("doc_name", "")]
    if bill_hits:
        hits = bill_hits

    if hits:
        idx, c = hits[0]
        return {
            "claims": [_chunk_claim(idx, c, "PERA is the Punjab Enforcement and Regulatory Authority", "act_registry")],
            "debug": {"source": "act_registry", "sub": "definition"},
        }
    return None


def _answer_pera_commencement() -> Optional[dict]:
    """Answer 'pera kb bani' / 'when established'. Returns Section 1(3) text only."""
    # Try to find commencement / come into force text
    hits = _search_chunks_keyword(["come into force", "Punjab Enforcement"], max_results=3)
    if not hits:
        hits = _search_chunks_keyword(["Commencement", "PERA"], max_results=3)
    if not hits:
        hits = _search_chunks_keyword(["section 1", "come into force"], max_results=3)

    bill_hits = [(i, c) for i, c in hits if "Bill" in c.get("doc_name", "")]
    if bill_hits:
        hits = bill_hits

    if hits:
        idx, c = hits[0]
        return {
            "claims": [_chunk_claim(idx, c, "PERA establishment/commencement information", "act_registry", 400)],
            "debug": {"source": "act_registry", "sub": "commencement"},
        }
    return None


def _answer_named_person(query: str, meta: dict = None) -> Optional[dict]:
    """Answer named person queries ONLY if found in chunks."""
    ql = query.lower()
    sub = (meta or {}).get("sub", "name")

    # DG reporting structure
    if sub == "reporting":
        hits = _search_chunks_keyword(["Director General", "Reports To"], max_results=5)
        if not hits:
            hits = _search_chunks_keyword(["Director General", "Authority"], max_results=5)
        if hits:
            idx, c = hits[0]
            return {
                "claims": [_chunk_claim(idx, c, "DG PERA reporting structure", "named_entity", 400)],
                "debug": {"source": "named_entity", "person": "DG reporting"},
            }
        return {"claims": [], "debug": {"source": "named_entity", "person": "DG reporting", "not_found": True}}

    # Salma Butt
    if "salma butt" in ql or "salma" in ql:
        hits = _search_chunks_keyword(["Salma Butt"], max_results=3)
        if hits:
            idx, c = hits[0]
            return {
                "claims": [_chunk_claim(idx, c, "Salma Butt is mentioned in PERA documents", "named_entity", 400)],
                "debug": {"source": "named_entity", "person": "Salma Butt"},
            }
        return {"claims": [], "debug": {"source": "named_entity", "person": "Salma Butt", "not_found": True}}

    # DG name — STRICT: only answer if explicit personal name found in chunk next to "Director General"
    hits = _search_chunks_keyword(["Director General"], max_results=10)
    name_indicators = ["Mr.", "Ms.", "Dr.", "Capt.", "Col.", "Brig.", "General"]
    for idx, c in hits:
        text = c.get("text", "")
        if any(ind in text for ind in name_indicators):
            return {
                "claims": [_chunk_claim(idx, c, "Director General information", "named_entity", 400)],
                "debug": {"source": "named_entity", "person": "DG"},
            }
    # No name found — strict refusal
    return {"claims": [], "debug": {"source": "named_entity", "person": "DG name", "not_found": True}}


# ═════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════

def registry_answer(query: str, ctx: dict = None) -> Optional[dict]:
    """
    Main entry: attempts deterministic answer.
    Returns None if fallback to normal RAG is needed.
    Returns {"claims": [...], "debug": {...}} if deterministic answer found.
    """
    intent, meta = route_intent(query)
    print(f"[Registry] intent={intent} meta={meta}")

    if intent == INTENT_SCHEDULE_LOOKUP:
        return _answer_schedule(meta)

    elif intent == INTENT_JD_LOOKUP:
        # Follow-up "number mein" with entity lock
        if ctx and ctx.get("last_job_title") and meta.get("aspect") == "salary":
            if not _resolve_job_title(query.lower()):
                meta["job_title"] = ctx["last_job_title"]
        return _answer_jd(meta, query)

    elif intent == INTENT_ACT_SECTION:
        return _answer_act_section(meta, query)

    elif intent == INTENT_ACT_DEFINITION:
        return _answer_act_definition(meta)

    elif intent == INTENT_PERA_DEFINITION:
        return _answer_pera_definition(meta)

    elif intent == INTENT_PERA_COMMENCEMENT:
        return _answer_pera_commencement()

    elif intent == INTENT_NAMED_PERSON:
        return _answer_named_person(query, meta)

    return None  # FALLBACK


def get_sppp_row(sppp_label: str) -> Optional[dict]:
    """Get Schedule-III row for a given SPPP label (e.g., 'SPPP-5')."""
    for row in SCHEDULE_III_ROWS:
        if row["sppp"].lower() == sppp_label.lower():
            return row
    return None
