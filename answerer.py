from __future__ import annotations

import os
import re
import json
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import quote

from dotenv import load_dotenv
from openai import OpenAI

from smalltalk_intent import decide_smalltalk

load_dotenv()

# ============================================================
# Hard requirement: exact refusal sentence (ONLY for true empty/unrelated)
# ============================================================
REFUSAL_TEXT = "There is no information available to this question."

# ============================================================
# Models / limits
# ============================================================
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gpt-4.1-mini")
VERIFIER_MODEL = os.getenv("VERIFIER_MODEL", ANSWER_MODEL)

MAX_EVIDENCE_CHARS = int(os.getenv("MAX_EVIDENCE_CHARS", "24000"))

# IMPORTANT: permissive; use "clarify" instead of refusing
MIN_EVIDENCE_CHARS_TO_ANSWER = int(os.getenv("MIN_EVIDENCE_CHARS_TO_ANSWER", "120"))

# Gates (keep permissive; verifier enforces correctness)
HIT_MIN_SCORE = float(os.getenv("HIT_MIN_SCORE", "0.35"))
HIT_STRONG_SCORE_BYPASS = float(os.getenv("HIT_STRONG_SCORE_BYPASS", "0.55"))
HIT_MEDIUM_SCORE_KEEP = float(os.getenv("HIT_MEDIUM_SCORE_KEEP", "0.48"))

MAX_HITS_PER_DOC_FOR_PROMPT = int(os.getenv("MAX_HITS_PER_DOC_FOR_PROMPT", "4"))
MAX_DOCS_FOR_PROMPT = int(os.getenv("MAX_DOCS_FOR_PROMPT", "4"))
MAX_REFS_RETURNED = int(os.getenv("MAX_REFS_RETURNED", "8"))

REF_SNIPPET_CHARS = int(os.getenv("REF_SNIPPET_CHARS", "360"))

# accept both env var spellings (your app.py uses Base_URL)
BASE_URL = os.getenv("BASE_URL", os.getenv("Base_URL", "")).strip().rstrip("/")

# Evidence junk filters (keep conservative)
MIN_EVIDENCE_CHARS = int(os.getenv("ANSWER_MIN_EVIDENCE_CHARS", "60"))
MIN_EVIDENCE_WORDS = int(os.getenv("ANSWER_MIN_EVIDENCE_WORDS", "8"))
MIN_EVIDENCE_LETTERS = int(os.getenv("ANSWER_MIN_EVIDENCE_LETTERS", "25"))

DEBUG_ANSWERER = os.getenv("DEBUG_ANSWERER", "0").strip() == "1"


# ============================================================
# Client
# ============================================================
def _client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is missing. Ensure .env is present and loaded.")
    return OpenAI(api_key=key)


# ============================================================
# Response helpers: refuse / clarify
# ============================================================
def _refuse(debug: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "decision": "refuse",
        "answer": REFUSAL_TEXT,
        "references": [],
        "used_chunk_ids": [],
    }
    if DEBUG_ANSWERER and debug:
        out["debug"] = debug
    return out


def _clarify(
    message: str,
    references: Optional[List[Dict[str, Any]]] = None,
    used_ids: Optional[List[int]] = None,
    debug: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "decision": "clarify",
        "answer": (message or "").strip() or "Please clarify your question.",
        "references": references or [],
        "used_chunk_ids": used_ids or [],
    }
    if DEBUG_ANSWERER and debug:
        out["debug"] = debug
    return out


# ============================================================
# Smalltalk handling (robust)
# ============================================================
def _handle_smalltalk(question: str) -> Optional[Dict[str, Any]]:
    try:
        decision = decide_smalltalk(question or "")
    except Exception:
        return None

    # object style
    if hasattr(decision, "is_greeting_only"):
        if getattr(decision, "is_greeting_only"):
            resp = (getattr(decision, "response", None) or "Hello!").strip()
            return {"decision": "answer", "answer": resp, "references": [], "used_chunk_ids": []}

    # dict style
    if isinstance(decision, dict):
        if decision.get("is_smalltalk") or decision.get("smalltalk") or decision.get("is_greeting_only"):
            resp = (decision.get("response") or "Hello!").strip()
            return {"decision": "answer", "answer": resp, "references": [], "used_chunk_ids": []}

    # tuple style
    if isinstance(decision, tuple) and len(decision) >= 2:
        if bool(decision[0]):
            resp = (decision[1] or "Hello!").strip()
            return {"decision": "answer", "answer": resp, "references": [], "used_chunk_ids": []}

    return None


# ============================================================
# Deterministic cleanup helpers
# ============================================================
_BRACKET_CIT_RE = re.compile(r"\[[^\]]+\]")

def _strip_inline_citations(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return t
    t = _BRACKET_CIT_RE.sub("", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    return t.strip()

def _normalize_ws(text: str) -> str:
    t = (text or "").replace("\u00ad", "")  # soft hyphen
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ============================================================
# Minimal language handling (English / Urdu / Roman Urdu)
# ============================================================
_ROMAN_URDU_HINTS = {
    "kya", "ky", "ka", "ki", "ke", "mein", "me", "wala", "wali", "walay",
    "kaise", "kesy", "kesay", "batao", "batain", "btao", "mujhe", "hum", "ap",
    "aap", "kr", "kro", "karen", "krna", "krne", "hona", "hai", "hain", "tha",
    "thi", "thay", "q", "kyun", "qk", "plz", "please"
}

def _detect_lang_simple(q: str) -> str:
    if re.search(r"[\u0600-\u06FF]", q or ""):
        return "ur"
    # roman urdu heuristic: many roman-urdu markers, but not heavy English punctuation
    tokens = re.findall(r"[a-zA-Z']+", (q or "").lower())
    if tokens:
        hit = sum(1 for t in tokens if t in _ROMAN_URDU_HINTS)
        if hit >= 2:
            return "roman_ur"
    return "en"

def _clarify_text(lang: str, hint: str = "") -> str:
    hint = (hint or "").strip()
    if lang == "ur":
        base = "براہِ کرم اپنا سوال تھوڑا واضح کریں تاکہ میں درست جواب دے سکوں۔"
        if hint:
            return f"{base}\n\nوضاحت: {hint}"
        return base

    if lang == "roman_ur":
        base = "Meharbani kar ke apna sawal thora wazeh kar dein taa ke main documents se sahi jawab de sakun."
        if hint:
            return f"{base}\n\nWazahat chahiye: {hint}"
        return base

    base = "Please clarify your question so I can answer accurately from the documents."
    if hint:
        return f"{base}\n\nClarification needed: {hint}"
    return base


# ============================================================
# Evidence quality: drop obvious junk only (keep permissive)
# ============================================================
_PAGE_GARBAGE_RE = re.compile(r"^\s*page\s*\d+\s*(of\s*\d+)?\s*$", re.I)
_ONLY_NUM_PUNCT_RE = re.compile(r"^[\s0-9\-–—_.,:;|/\\()]+$")

def _count_letters(s: str) -> int:
    return len(re.findall(r"[A-Za-z\u0600-\u06FF]", s or ""))

def _count_words(s: str) -> int:
    return len(re.findall(r"[A-Za-z\u0600-\u06FF]{2,}", s or ""))

def _is_low_signal_chunk(txt: str) -> bool:
    t = (txt or "").strip()
    if not t:
        return True
    if _PAGE_GARBAGE_RE.match(t):
        return True
    if len(t) <= 12 and _ONLY_NUM_PUNCT_RE.match(t):
        return True
    if len(t) < MIN_EVIDENCE_CHARS and _count_words(t) < 3:
        return True
    letters = _count_letters(t)
    words = _count_words(t)
    if letters < MIN_EVIDENCE_LETTERS and words < MIN_EVIDENCE_WORDS:
        tl = t.lower()
        if "pera" in tl or "authority" in tl or "پيرا" in tl:
            return False
        return True
    return False


# ============================================================
# Paths / URLs (deterministic, never LLM-generated)
# ============================================================
def _safe_default_url_path(doc_name: str) -> str:
    dn = (doc_name or "").strip()
    if not dn:
        return "/assets/data"
    return f"/assets/data/{dn}".replace("\\", "/")

def _normalize_public_path(path_or_url: str, doc_name: str) -> str:
    p = (path_or_url or "").strip().replace("\\", "/")

    # If retriever already supplied canonical /assets/data/...
    if p.startswith("/assets/data/"):
        return p
    if p.startswith("assets/data/"):
        return "/" + p
    if "/assets/data/" in p:
        tail = p.split("/assets/data/", 1)[1]
        return "/assets/data/" + tail

    # If a full URL is passed, try to recover /assets/data/... part
    if p.startswith("http://") or p.startswith("https://"):
        if "/assets/data/" in p:
            tail = p.split("/assets/data/", 1)[1]
            return "/assets/data/" + tail
        # otherwise fall back to doc_name
        return _safe_default_url_path(doc_name)

    # If looks like filename
    if p.lower().endswith(".pdf") or p.lower().endswith(".docx"):
        filename = p.split("/")[-1]
        return f"/assets/data/{filename}"

    return _safe_default_url_path(doc_name)

def _file_type_from_doc(doc_name: str, public_path: str) -> str:
    dn = (doc_name or "").lower().strip()
    pp = (public_path or "").lower().strip()
    if dn.endswith(".pdf") or pp.endswith(".pdf"):
        return "pdf"
    if dn.endswith(".docx") or pp.endswith(".docx"):
        return "docx"
    return "file"

def _make_snippet(text: str) -> str:
    t = _normalize_ws(text or "")
    if len(t) <= REF_SNIPPET_CHARS:
        return t
    return t[:REF_SNIPPET_CHARS].rstrip() + "…"

def _build_open_url(public_path: str, url_hint: str) -> str:
    if not BASE_URL:
        return f"{public_path}{url_hint or ''}"
    return f"{BASE_URL}{public_path}{url_hint or ''}"

def _build_download_url(doc_name: str) -> str:
    filename = (doc_name or "").strip()
    if not filename:
        return ""
    if not BASE_URL:
        return f"/download/{quote(filename)}"
    return f"{BASE_URL}/download/{quote(filename)}"


# ============================================================
# Evidence filtering (permissive, stable)
# ============================================================
def _filter_evidence(question: str, evidence_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for d in evidence_docs:
        hits = d.get("hits", []) or []
        if not hits:
            continue

        hits2 = [h for h in hits if not _is_low_signal_chunk((h.get("text") or "").strip())]
        if not hits2:
            continue

        hits2.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)

        strong = [h for h in hits2 if float(h.get("score", 0.0) or 0.0) >= HIT_STRONG_SCORE_BYPASS]
        if strong:
            kept = strong[:MAX_HITS_PER_DOC_FOR_PROMPT]
        else:
            def _rank(h: Dict[str, Any]) -> Tuple[float, int]:
                sc = float(h.get("score", 0.0) or 0.0)
                try:
                    ov = int(h.get("overlap_all", h.get("overlap", 0)) or 0)
                except Exception:
                    ov = 0
                return (sc, ov)
            hits2.sort(key=_rank, reverse=True)
            kept = hits2[:MAX_HITS_PER_DOC_FOR_PROMPT]

        d2 = dict(d)
        d2["hits"] = kept
        filtered.append(d2)

    return filtered[:MAX_DOCS_FOR_PROMPT]


# ============================================================
# Build evidence prompt AND keep deterministic evidence_id mapping
# ============================================================
_EVIDENCE_HEADER_RE = re.compile(
    r"\[EVIDENCE_ID:\s*(\d+)\s*\|\s*CHUNK_ID:\s*([0-9NAna]+)\s*\|",
    re.I
)

def _format_loc_for_prompt(hit: Dict[str, Any]) -> str:
    doc = hit.get("doc_name", "Unknown document")
    loc_kind = hit.get("loc_kind")
    loc_start = hit.get("loc_start")
    if loc_kind == "page":
        return f"{doc} — p. {loc_start}"
    return f"{doc} — {loc_start}"

def _hit_chunk_id(hit: Dict[str, Any]) -> Optional[int]:
    for k in ("id", "chunk_id"):
        v = hit.get(k)
        try:
            if v is None:
                continue
            return int(v)
        except Exception:
            continue
    return None

def _truncate_evidence_blocks_with_used_hits(
    evidence_docs: List[Dict[str, Any]]
) -> Tuple[str, List[Dict[str, Any]], Dict[int, int]]:
    """
    Returns:
      evidence_text,
      included_hits (in order),
      evidence_id_to_chunk_id mapping
    """
    chunks: List[str] = []
    included_hits: List[Dict[str, Any]] = []
    evidence_id_to_chunk: Dict[int, int] = {}
    total = 0

    for d in evidence_docs[:MAX_DOCS_FOR_PROMPT]:
        doc_name = d.get("doc_name", "Unknown document")
        doc_rank = d.get("doc_rank", 0)
        hits = d.get("hits", []) or []

        header = f"\n\n=== DOCUMENT: {doc_name} (rank={doc_rank}) ===\n"
        if total + len(header) > MAX_EVIDENCE_CHARS:
            break
        chunks.append(header)
        total += len(header)

        kept = 0
        for h in hits:
            if kept >= MAX_HITS_PER_DOC_FOR_PROMPT:
                break

            text = (h.get("text") or "").strip()
            if not text or _is_low_signal_chunk(text):
                continue

            cid = _hit_chunk_id(h)
            loc = _format_loc_for_prompt(h)

            evidence_id = len(included_hits) + 1
            block = (
                f"\n[EVIDENCE_ID: {evidence_id} | CHUNK_ID: {cid if cid is not None else 'NA'} | {loc}]\n"
                f"{text}\n"
            )

            if total + len(block) > MAX_EVIDENCE_CHARS:
                break

            chunks.append(block)
            total += len(block)
            included_hits.append(h)

            if cid is not None:
                evidence_id_to_chunk[evidence_id] = int(cid)

            kept += 1

        if total >= MAX_EVIDENCE_CHARS:
            break

    evidence_text = "".join(chunks).strip()
    return evidence_text, included_hits, evidence_id_to_chunk


# ============================================================
# References from USED hits only (deterministic)
# ============================================================
def _make_reference(hit: Dict[str, Any]) -> Dict[str, Any]:
    doc = hit.get("doc_name", "Unknown document")
    raw_path = (hit.get("public_path") or hit.get("path") or "").strip()
    public_path = _normalize_public_path(raw_path, doc)

    loc_kind = hit.get("loc_kind")
    loc_start = hit.get("loc_start")
    loc_end = hit.get("loc_end")

    snippet = _make_snippet(hit.get("text") or "")

    url_hint = ""
    if loc_kind == "page" and loc_start is not None:
        url_hint = f"#page={loc_start}"

    open_url = _build_open_url(public_path, url_hint)
    download_url = _build_download_url(doc)

    ref: Dict[str, Any] = {
        "document": doc,
        "path": public_path,
        "open_url": open_url,
        "download_url": download_url,
        "file_type": _file_type_from_doc(doc, public_path),
        "loc_kind": loc_kind,
        "loc_start": loc_start,
        "loc_end": loc_end,
        "snippet": snippet,
        "url_hint": url_hint,
    }

    if loc_kind == "page" and loc_start is not None:
        ref["page_start"] = loc_start
        ref["page_end"] = loc_start if loc_end is None else loc_end
    else:
        ref["loc"] = str(loc_start) if loc_start is not None else ""

    return ref

def _build_references_from_used_hits(used_hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    seen = set()

    for h in used_hits:
        doc = h.get("doc_name", "Unknown document")
        loc_kind = h.get("loc_kind")
        loc_start = h.get("loc_start")
        key = (doc, loc_kind, str(loc_start))

        if key in seen:
            continue
        seen.add(key)

        refs.append(_make_reference(h))
        if len(refs) >= MAX_REFS_RETURNED:
            break

    return refs


# ============================================================
# Robust JSON extraction (fixes major production failures)
# ============================================================
def _extract_first_valid_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Scans for balanced {...} blocks and returns the first one that parses.
    Handles model outputs that contain multiple JSON blocks or extra braces.
    """
    if not text:
        return None

    s = text.strip()

    # quick path: if whole content is json
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # balanced brace scan
    start_positions = [m.start() for m in re.finditer(r"\{", s)]
    for start in start_positions:
        depth = 0
        for i in range(start, len(s)):
            ch = s[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = s[start:i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        break  # move to next start
        # continue with next start
    return None


# ============================================================
# Verifier: clause-level verification (reduces false negatives)
# ============================================================
def _verify_supported_clauses(
    client: OpenAI,
    question: str,
    draft: str,
    evidence_text: str,
    lang: str
) -> Tuple[str, bool]:
    """
    Ask the verifier to keep only explicitly supported CLAUSES (not whole sentences).
    This dramatically reduces "all dropped" outcomes on long sentences.
    """
    d = (draft or "").strip()
    if not d:
        return "", False

    system = (
        "You are a STRICT verifier for a government document-grounded system.\n"
        "Given QUESTION, EVIDENCE, and DRAFT_ANSWER:\n"
        "- Rewrite the answer into short bullet clauses.\n"
        "- Keep ONLY clauses that are explicitly supported by the evidence.\n"
        "- Remove any clause that adds any new fact not in evidence.\n"
        "- Do NOT use outside knowledge.\n"
        "Return ONLY valid JSON: {\"supported_clauses\":[\"...\",\"...\"]}\n"
        "Clauses must be short and factual."
    )

    # keep same language
    lang_rule = "Write clauses in Urdu." if lang == "ur" else ("Write clauses in Roman Urdu." if lang == "roman_ur" else "Write clauses in English.")
    system = system + "\n" + lang_rule

    payload = {"question": question, "evidence": evidence_text, "draft_answer": d}

    try:
        raw = client.chat.completions.create(
            model=VERIFIER_MODEL,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        ).choices[0].message.content.strip()
    except Exception:
        return "", False

    data = _extract_first_valid_json_object(raw)
    if not data:
        return "", False

    clauses = data.get("supported_clauses", []) or []
    if not isinstance(clauses, list):
        return "", False

    cleaned = []
    for c in clauses:
        if isinstance(c, str):
            cc = _strip_inline_citations(c).strip()
            if cc:
                cleaned.append(cc)

    if not cleaned:
        return "", False

    # join clauses into a compact answer
    # keep style: bullet-like but single block
    if lang == "ur":
        out = "۔ ".join(cleaned).strip()
        if not out.endswith("۔"):
            out += "۔"
        return out, True

    out = "; ".join(cleaned).strip()
    return out, True


# ============================================================
# used ids parsing helpers
# ============================================================
def _parse_int_list(x: Any) -> List[int]:
    out: List[int] = []
    if x is None:
        return out
    if isinstance(x, list):
        items = x
    elif isinstance(x, str):
        # allow "1,2,3" or "1 2 3"
        items = re.split(r"[,\s]+", x.strip())
    else:
        items = [x]

    for it in items:
        try:
            if it is None:
                continue
            out.append(int(str(it).strip()))
        except Exception:
            continue
    # stable dedup
    out2: List[int] = []
    seen = set()
    for n in out:
        if n in seen:
            continue
        seen.add(n)
        out2.append(n)
    return out2


# ============================================================
# Main API
# ============================================================
def answer_question(question: str, retrieval: Dict[str, Any]) -> Dict[str, Any]:
    q = (question or "").strip()
    if not q:
        return _refuse({"reason": "empty_question"} if DEBUG_ANSWERER else None)

    lang = _detect_lang_simple(q)

    small = _handle_smalltalk(q)
    if small:
        return small

    # True "no evidence" => ONLY case for REFUSAL_TEXT
    if not retrieval or not retrieval.get("has_evidence"):
        return _refuse({"reason": "no_retrieval_or_has_evidence_false"} if DEBUG_ANSWERER else None)

    evidence_docs = retrieval.get("evidence", []) or []
    if not evidence_docs:
        return _refuse({"reason": "no_evidence_docs"} if DEBUG_ANSWERER else None)

    evidence_docs = _filter_evidence(q, evidence_docs)
    if not evidence_docs:
        return _clarify(
            _clarify_text(lang, "I found documents, but the retrieved snippets were too low-signal to answer. Please rephrase or specify the exact role/section you mean."),
            references=[],
            used_ids=[],
            debug={"reason": "filtered_evidence_empty"} if DEBUG_ANSWERER else None,
        )

    evidence_text, included_hits, evidence_id_to_chunk = _truncate_evidence_blocks_with_used_hits(evidence_docs)

    if not evidence_text or len(evidence_text) < MIN_EVIDENCE_CHARS_TO_ANSWER:
        refs = _build_references_from_used_hits(included_hits) if included_hits else []
        used_ids = [cid for cid in (_hit_chunk_id(h) for h in included_hits) if cid is not None]
        return _clarify(
            _clarify_text(lang, "The available excerpts are too short/partial. Tell me the exact role, section name, or page range you want."),
            references=refs[: max(1, min(3, len(refs)))],
            used_ids=used_ids[: max(1, min(6, len(used_ids)))],
            debug={"reason": "evidence_too_small", "evidence_chars": len(evidence_text)} if DEBUG_ANSWERER else None,
        )

    client = _client()

    # Ask for structured JSON so we can bind references to used hits deterministically
    system = (
        "You are the official AI assistant for PERA (Punjab Enforcement and Regulatory Authority).\n"
        "You MUST follow these rules:\n"
        "1) Use ONLY the evidence blocks.\n"
        "2) Do NOT guess or use outside knowledge.\n"
        "3) If the evidence does NOT explicitly answer the question, choose decision='clarify' and ask ONE precise clarification question.\n"
        "4) Use decision='refuse' ONLY if the evidence is empty or completely unrelated (rare).\n"
        "5) Return ONLY valid JSON in this exact schema:\n"
        "   {\"decision\":\"answer|clarify|refuse\",\"answer\":\"...\",\"used_chunk_ids\":[...],\"used_evidence_ids\":[...]} \n"
        "   - used_chunk_ids: list of CHUNK_ID values\n"
        "   - used_evidence_ids: list of EVIDENCE_ID values (if easier)\n"
        "6) If decision='refuse', answer MUST be exactly: "
        f"\"{REFUSAL_TEXT}\"\n"
        "7) If decision='answer', keep it concise, formal, and auditable.\n"
        "8) NEVER output a person's name unless it appears exactly in the evidence.\n"
        "9) NEVER output a number unless it appears exactly in the evidence.\n"
        "10) Do NOT include citations/brackets/page numbers in the answer text.\n"
        "11) Write in the same language as the user question (English, Urdu, or Roman Urdu).\n"
    )

    user = (
        f"QUESTION:\n{q}\n\n"
        f"EVIDENCE:\n{evidence_text}\n\n"
        "Task:\n"
        "- If evidence explicitly answers: decision='answer' and include used_chunk_ids and/or used_evidence_ids.\n"
        "- If evidence is partial/ambiguous: decision='clarify' and ask ONE targeted follow-up question.\n"
        "- If unrelated/empty: decision='refuse' and answer exactly the refusal sentence.\n"
    )

    raw = client.chat.completions.create(
        model=ANSWER_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    ).choices[0].message.content.strip()

    data = _extract_first_valid_json_object(raw)
    if not data:
        refs = _build_references_from_used_hits(included_hits)
        used_ids = [cid for cid in (_hit_chunk_id(h) for h in included_hits) if cid is not None]
        return _clarify(
            _clarify_text(lang, "I found relevant documents but could not produce a reliable structured answer. Please rephrase or specify the exact role/section."),
            references=refs[:3],
            used_ids=used_ids[:6],
            debug={"reason": "model_output_not_json", "raw_head": raw[:240]} if DEBUG_ANSWERER else None,
        )

    decision = (data.get("decision") or "").strip().lower()
    answer = (data.get("answer") or "").strip()

    used_chunk_ids = _parse_int_list(data.get("used_chunk_ids"))
    used_evidence_ids = _parse_int_list(data.get("used_evidence_ids"))

    # If model provided evidence_ids, map them to chunk_ids
    for eid in used_evidence_ids:
        cid = evidence_id_to_chunk.get(int(eid))
        if cid is not None and cid not in used_chunk_ids:
            used_chunk_ids.append(int(cid))

    # Stable dedup
    used_ids: List[int] = []
    seen = set()
    for n in used_chunk_ids:
        if n in seen:
            continue
        seen.add(n)
        used_ids.append(n)

    # Map used IDs to included hits
    id_to_hit: Dict[int, Dict[str, Any]] = {}
    for h in included_hits:
        cid = _hit_chunk_id(h)
        if cid is None:
            continue
        id_to_hit[int(cid)] = h

    used_hits: List[Dict[str, Any]] = []
    for cid in used_ids:
        h = id_to_hit.get(int(cid))
        if h:
            used_hits.append(h)

    # If model didn't specify usable ids, fall back deterministically to top hits
    if not used_hits:
        used_hits = included_hits[: max(1, min(3, len(included_hits)))]
        used_ids = [cid for cid in (_hit_chunk_id(h) for h in used_hits) if cid is not None]

    # decision handling
    if decision == "refuse":
        if answer.strip() == REFUSAL_TEXT:
            # Only allow refuse if truly empty/unrelated.
            # Since retrieval.has_evidence was true, this should be rare; keep as-is to match policy.
            return _refuse({"reason": "model_refuse"} if DEBUG_ANSWERER else None)
        return _clarify(
            _clarify_text(lang, "The documents retrieved do not explicitly answer this as written. Which exact role/section or timeframe do you mean?"),
            references=_build_references_from_used_hits(used_hits)[:3],
            used_ids=used_ids[:6],
            debug={"reason": "model_refuse_but_answer_not_exact"} if DEBUG_ANSWERER else None,
        )

    if decision == "clarify":
        answer = _strip_inline_citations(answer)
        if not answer:
            answer = _clarify_text(lang, "Which exact role/section or page range should I use?")
        refs = _build_references_from_used_hits(used_hits)
        return _clarify(
            answer,
            references=refs[:3],
            used_ids=used_ids[:6],
            debug={"reason": "model_clarify"} if DEBUG_ANSWERER else None
        )

    # decision == "answer" (or unknown -> treat as answer attempt)
    answer = _strip_inline_citations(answer)
    if not answer or answer.strip() == REFUSAL_TEXT:
        refs = _build_references_from_used_hits(used_hits)
        return _clarify(
            _clarify_text(lang, "I found relevant excerpts but they don’t explicitly resolve your question. Please specify the exact role/title or section."),
            references=refs[:3],
            used_ids=used_ids[:6],
            debug={"reason": "answer_empty_or_refusal"} if DEBUG_ANSWERER else None,
        )

    # Verifier pass: if it can't support, degrade to clarify (NOT refusal)
    verified, ok = _verify_supported_clauses(client, q, answer, evidence_text, lang)
    verified = _strip_inline_citations(verified)

    if not ok or not verified:
        refs = _build_references_from_used_hits(used_hits)
        return _clarify(
            _clarify_text(lang, "The retrieved evidence is related but does not explicitly support a full answer. Please clarify the exact clause/role you mean."),
            references=refs[:3],
            used_ids=used_ids[:6],
            debug={"reason": "verifier_could_not_support"} if DEBUG_ANSWERER else None,
        )

    refs = _build_references_from_used_hits(used_hits)
    out: Dict[str, Any] = {
        "decision": "answer",
        "answer": verified.strip(),
        "references": refs,
        "used_chunk_ids": used_ids,
    }
    if DEBUG_ANSWERER:
        out["debug"] = {
            "decision_in": decision,
            "included_hits": len(included_hits),
            "used_hits": len(used_hits),
            "refs": len(refs),
            "evidence_chars": len(evidence_text),
            "mapped_evidence_ids": used_evidence_ids,
        }
    return out
