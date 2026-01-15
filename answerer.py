from __future__ import annotations

import os
import re
from typing import Dict, Any, List

from dotenv import load_dotenv
from openai import OpenAI

from smalltalk_intent import decide_smalltalk

load_dotenv()

# Fixed refusal sentence (exact)
REFUSAL_TEXT = "There is no information available to this question."

ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gpt-4.1-mini")
MAX_EVIDENCE_CHARS = int(os.getenv("MAX_EVIDENCE_CHARS", "24000"))

# Gates
ANSWER_MIN_TOP_SCORE = float(os.getenv("ANSWER_MIN_TOP_SCORE", "0.45"))
HIT_MIN_SCORE = float(os.getenv("HIT_MIN_SCORE", "0.40"))

# ✅ NEW: if semantic score is strong, allow evidence even with low lexical overlap
HIT_STRONG_SCORE_BYPASS = float(os.getenv("HIT_STRONG_SCORE_BYPASS", "0.62"))

MAX_HITS_PER_DOC_FOR_PROMPT = int(os.getenv("MAX_HITS_PER_DOC_FOR_PROMPT", "4"))
MAX_DOCS_FOR_PROMPT = int(os.getenv("MAX_DOCS_FOR_PROMPT", "4"))
MAX_REFS_RETURNED = int(os.getenv("MAX_REFS_RETURNED", "8"))

# Reference snippet size
REF_SNIPPET_CHARS = int(os.getenv("REF_SNIPPET_CHARS", "360"))

# -----------------------------
# Client
# -----------------------------
def _client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is missing. Ensure .env is present and loaded.")
    return OpenAI(api_key=key)


def _refuse() -> Dict[str, Any]:
    return {"answer": REFUSAL_TEXT, "references": []}


# -----------------------------
# Deterministic cleanup helpers
# -----------------------------
_BRACKET_CIT_RE = re.compile(r"\[[^\]]+\]")  # e.g. [Doc — p. 1]


def _strip_inline_citations(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return t
    t = _BRACKET_CIT_RE.sub("", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    return t.strip()


# -----------------------------
# ✅ Query normalization for lexical scoring
# -----------------------------
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "or",
    "in", "on", "at", "for", "from", "by", "with", "about", "tell", "me",
    "who", "what", "when", "where", "why", "how", "please",
}

# ✅ Deterministic abbreviation expansions (helps overlap detection)
_ABBREV_MAP = {
    "cto": "chief technology officer",
    "tor": "terms of reference",
    "tors": "terms of reference",
    "dg": "director general",
    "hr": "human resource",
    "it": "information technology",
    "ppra": "punjab procurement regulatory authority",
    "ipo": "initial public offering",
}


def _expand_abbreviations(q: str) -> str:
    s = (q or "").lower()
    # replace whole words only
    for k, v in _ABBREV_MAP.items():
        s = re.sub(rf"\b{re.escape(k)}\b", v, s)
    return s


def _tokenize(s: str) -> List[str]:
    s = _expand_abbreviations(s or "")
    # keep english + digits + urdu block
    s = re.sub(r"[^a-z0-9\u0600-\u06FF\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    toks = []
    for t in s.split(" "):
        if len(t) < 3:
            continue
        if t in _STOPWORDS:
            continue
        toks.append(t)
    return toks


def _keyword_overlap(question: str, text: str) -> int:
    q = set(_tokenize(question))
    if not q:
        return 0
    t = set(_tokenize(text))
    return len(q.intersection(t))


def _safe_default_path(doc_name: str) -> str:
    doc_name = (doc_name or "").strip()
    if not doc_name:
        return ""
    return os.path.join("assets", "data", doc_name).replace("\\", "/")


def _file_type_from_path(path: str) -> str:
    p = (path or "").lower()
    if p.endswith(".pdf"):
        return "pdf"
    if p.endswith(".docx"):
        return "docx"
    return "file"


def _make_snippet(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) <= REF_SNIPPET_CHARS:
        return t
    return t[: REF_SNIPPET_CHARS].rstrip() + "…"


# -----------------------------
# Evidence prompt building
# -----------------------------
def _format_loc_for_prompt(hit: Dict[str, Any]) -> str:
    doc = hit.get("doc_name", "Unknown document")
    loc_kind = hit.get("loc_kind")
    loc_start = hit.get("loc_start")
    if loc_kind == "page":
        return f"{doc} — p. {loc_start}"
    return f"{doc} — {loc_start}"


def _truncate_evidence_blocks(evidence_docs: List[Dict[str, Any]]) -> str:
    chunks: List[str] = []
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

        for h in hits[:MAX_HITS_PER_DOC_FOR_PROMPT]:
            loc = _format_loc_for_prompt(h)
            text = (h.get("text") or "").strip()
            if not text:
                continue

            block = f"\n[{loc}]\n{text}\n"
            if total + len(block) > MAX_EVIDENCE_CHARS:
                break
            chunks.append(block)
            total += len(block)

        if total >= MAX_EVIDENCE_CHARS:
            break

    return "".join(chunks).strip()


# -----------------------------
# ✅ Evidence filtering (SYSTEMATIC FIX)
# -----------------------------
def _filter_evidence(question: str, evidence_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep ONLY high-confidence hits.
    Systematic fix:
    - Prefer retriever-provided overlap if present.
    - Allow strong semantic hits even if lexical overlap is 0 (prevents false refusals).
    """
    filtered: List[Dict[str, Any]] = []

    for d in evidence_docs:
        hits = d.get("hits", []) or []
        good_hits: List[Dict[str, Any]] = []

        for h in hits:
            score = float(h.get("score", 0.0) or 0.0)
            if score < HIT_MIN_SCORE:
                continue

            txt = (h.get("text") or "").strip()
            if not txt:
                continue

            # ✅ Prefer retriever overlap (already tuned with variants/intent)
            retr_overlap = h.get("overlap")
            if retr_overlap is not None:
                try:
                    retr_overlap = int(retr_overlap)
                except Exception:
                    retr_overlap = None

            # ✅ If overlap exists and is >=1, accept
            if retr_overlap is not None and retr_overlap >= 1:
                good_hits.append(h)
                continue

            # Otherwise compute our own lexical overlap (abbrev-expanded)
            lex_ov = _keyword_overlap(question, txt)

            # ✅ If lexical overlap exists, accept
            if lex_ov > 0:
                good_hits.append(h)
                continue

            # ✅ Strong semantic bypass (prevents “info exists but refused”)
            if score >= HIT_STRONG_SCORE_BYPASS:
                good_hits.append(h)
                continue

        if good_hits:
            good_hits.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
            d2 = dict(d)
            d2["hits"] = good_hits[:MAX_HITS_PER_DOC_FOR_PROMPT]
            filtered.append(d2)

    return filtered[:MAX_DOCS_FOR_PROMPT]


# -----------------------------
# References (rich objects)
# -----------------------------
def _make_reference(hit: Dict[str, Any]) -> Dict[str, Any]:
    doc = hit.get("doc_name", "Unknown document")

    path = (hit.get("path") or "").strip() or _safe_default_path(doc)
    path = path.replace("\\", "/")

    loc_kind = hit.get("loc_kind")
    loc_start = hit.get("loc_start")
    loc_end = hit.get("loc_end")

    snippet = _make_snippet(hit.get("text") or "")

    ref: Dict[str, Any] = {
        "document": doc,
        "path": path,
        "file_type": _file_type_from_path(path),
        "loc_kind": loc_kind,
        "loc_start": loc_start,
        "loc_end": loc_end,
        "snippet": snippet,
    }

    if loc_kind == "page" and loc_start is not None:
        ref["page_start"] = loc_start
        ref["page_end"] = loc_start if loc_end is None else loc_end
        ref["url_hint"] = f"#page={loc_start}"
    else:
        ref["loc"] = str(loc_start) if loc_start is not None else ""
        ref["url_hint"] = ""

    return ref


def _build_references_from_filtered(evidence_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    refs: List[Dict[str, Any]] = []

    for d in evidence_docs[:MAX_DOCS_FOR_PROMPT]:
        for h in (d.get("hits") or [])[:MAX_HITS_PER_DOC_FOR_PROMPT]:
            doc = h.get("doc_name", "Unknown document")
            loc_kind = h.get("loc_kind")
            loc_start = h.get("loc_start")

            key = (doc, loc_kind, str(loc_start))
            if key in seen:
                continue
            seen.add(key)

            refs.append(_make_reference(h))
            if len(refs) >= MAX_REFS_RETURNED:
                return refs

    return refs


# -----------------------------
# Main answering API
# -----------------------------
def answer_question(question: str, retrieval: Dict[str, Any]) -> Dict[str, Any]:
    """
    Output format:
    - Answer text only (NO inline citations)
    - References returned only in 'references' list for UI
    - Greeting/smalltalk handled deterministically with no retrieval/refs
    """

    decision = decide_smalltalk(question or "")
    if decision and decision.is_greeting_only:
        return {"answer": decision.response, "references": []}

    if not retrieval or not retrieval.get("has_evidence"):
        return _refuse()

    evidence_docs = retrieval.get("evidence", []) or []
    if not evidence_docs:
        return _refuse()

    # ✅ Filter out irrelevant evidence (now recall-safe)
    evidence_docs = _filter_evidence(question, evidence_docs)
    if not evidence_docs:
        return _refuse()

    # Hard gate: top hit score AFTER filtering
    try:
        top_hit = (evidence_docs[0].get("hits") or [{}])[0]
        top_score = float(top_hit.get("score", 0.0) or 0.0)
    except Exception:
        top_score = 0.0

    if top_score < ANSWER_MIN_TOP_SCORE:
        return _refuse()

    evidence_text = _truncate_evidence_blocks(evidence_docs)
    if not evidence_text:
        return _refuse()

    system = (
        "You are a strict document-grounded assistant.\n"
        "RULES:\n"
        "1) Use ONLY the evidence blocks.\n"
        "2) Do NOT guess.\n"
        "3) If not explicitly supported, output exactly:\n"
        f"{REFUSAL_TEXT}\n"
        "4) Prefer the latest/highest-ranked document.\n"
        "5) If documents conflict, do NOT merge; describe each version.\n"
        "6) IMPORTANT: Do NOT include citations, brackets, page numbers, or references in the answer text.\n"
        "7) Keep the answer concise and professional.\n"
    )

    user = (
        f"QUESTION:\n{question}\n\n"
        f"EVIDENCE:\n{evidence_text}\n\n"
        "TASK:\n"
        "Write the best supported answer.\n"
        "If conflict exists, mention it clearly and concisely.\n"
        f"If unsupported, output exactly: {REFUSAL_TEXT}\n"
    )

    client = _client()
    draft = client.chat.completions.create(
        model=ANSWER_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    ).choices[0].message.content.strip()

    if not draft or draft.strip() == REFUSAL_TEXT:
        return _refuse()

    draft_clean = _strip_inline_citations(draft)
    if not draft_clean or draft_clean.strip() == REFUSAL_TEXT:
        return _refuse()

    verifier_system = (
        "You are a strict verifier.\n"
        "Check every sentence is directly supported by the evidence.\n"
        "If ANY sentence is not supported, respond with exactly:\n"
        f"{REFUSAL_TEXT}\n"
        "Otherwise return the draft unchanged.\n"
        "Also: the draft must NOT include any bracketed citations like [ ... ].\n"
    )

    verifier_user = (
        f"DRAFT ANSWER:\n{draft_clean}\n\n"
        f"EVIDENCE:\n{evidence_text}\n"
    )

    verified = client.chat.completions.create(
        model=ANSWER_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": verifier_system},
            {"role": "user", "content": verifier_user},
        ],
    ).choices[0].message.content.strip()

    if not verified or verified.strip() == REFUSAL_TEXT:
        return _refuse()

    verified_clean = _strip_inline_citations(verified)
    if not verified_clean or verified_clean.strip() == REFUSAL_TEXT:
        return _refuse()

    refs = _build_references_from_filtered(evidence_docs)
    return {"answer": verified_clean, "references": refs}
