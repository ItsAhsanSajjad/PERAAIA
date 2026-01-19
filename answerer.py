from __future__ import annotations

import os
import re
from typing import Dict, Any, List
from urllib.parse import quote

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

# if semantic score is strong, allow evidence even with low lexical overlap
HIT_STRONG_SCORE_BYPASS = float(os.getenv("HIT_STRONG_SCORE_BYPASS", "0.62"))

MAX_HITS_PER_DOC_FOR_PROMPT = int(os.getenv("MAX_HITS_PER_DOC_FOR_PROMPT", "4"))
MAX_DOCS_FOR_PROMPT = int(os.getenv("MAX_DOCS_FOR_PROMPT", "4"))
MAX_REFS_RETURNED = int(os.getenv("MAX_REFS_RETURNED", "8"))

# Reference snippet size
REF_SNIPPET_CHARS = int(os.getenv("REF_SNIPPET_CHARS", "360"))

# Base URL for full links (used by API + mobile clients)
BASE_URL = os.getenv("Base_URL", "https://askpera.infinitysol.agency").rstrip("/")


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
# Query normalization for lexical scoring
# -----------------------------
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "or",
    "in", "on", "at", "for", "from", "by", "with", "about", "tell", "me",
    "who", "what", "when", "where", "why", "how", "please",
}

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
    for k, v in _ABBREV_MAP.items():
        s = re.sub(rf"\b{re.escape(k)}\b", v, s)
    return s


def _tokenize(s: str) -> List[str]:
    s = _expand_abbreviations(s or "")
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


# -----------------------------
# Paths / URLs
# -----------------------------
def _safe_default_url_path(doc_name: str) -> str:
    """
    Public path for FastAPI static mount:
      /assets/data/<filename>
    """
    dn = (doc_name or "").strip()
    if not dn:
        return "/assets/data"
    return f"/assets/data/{dn}".replace("\\", "/")


def _normalize_public_path(path_or_url: str, doc_name: str) -> str:
    """
    Normalize any stored path into /assets/data/<file>.pdf
    """
    p = (path_or_url or "").strip().replace("\\", "/")

    # If already correct public path
    if p.startswith("/assets/data/"):
        return p

    # If filesystem-ish relative
    if p.startswith("assets/data/"):
        return "/" + p

    # If it's something like ".../assets/data/<file>"
    if "/assets/data/" in p:
        tail = p.split("/assets/data/", 1)[1]
        return "/assets/data/" + tail

    # If it ends in a filename, use doc_name filename
    if p.lower().endswith(".pdf"):
        filename = p.split("/")[-1]
        return f"/assets/data/{filename}"

    # fallback
    return _safe_default_url_path(doc_name)


def _file_type_from_doc(doc_name: str, public_path: str) -> str:
    """
    We only want PDFs now. If something old leaks in, label it but still generate safe links.
    """
    dn = (doc_name or "").lower().strip()
    pp = (public_path or "").lower().strip()
    if dn.endswith(".pdf") or pp.endswith(".pdf"):
        return "pdf"
    if dn.endswith(".docx") or pp.endswith(".docx"):
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


def _build_open_url(public_path: str, url_hint: str) -> str:
    """
    Full open URL for web/mobile.
    """
    return f"{BASE_URL}{public_path}{url_hint or ''}"


def _build_download_url(doc_name: str) -> str:
    """
    Uses FastAPI endpoint:
      /download/{filename}
    """
    filename = (doc_name or "").strip()
    # ensure safe URL encoding for spaces/special chars
    return f"{BASE_URL}/download/{quote(filename)}"


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
# Evidence filtering
# -----------------------------
def _filter_evidence(question: str, evidence_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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

            retr_overlap = h.get("overlap")
            if retr_overlap is not None:
                try:
                    retr_overlap = int(retr_overlap)
                except Exception:
                    retr_overlap = None

            if retr_overlap is not None and retr_overlap >= 1:
                good_hits.append(h)
                continue

            lex_ov = _keyword_overlap(question, txt)
            if lex_ov > 0:
                good_hits.append(h)
                continue

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

    # ✅ hard safety: we only want to expose PDFs now
    # If doc_name is .docx somehow, still build paths but mark file_type
    raw_path = (hit.get("path") or "").strip()
    public_path = _normalize_public_path(raw_path, doc)

    loc_kind = hit.get("loc_kind")
    loc_start = hit.get("loc_start")
    loc_end = hit.get("loc_end")

    snippet = _make_snippet(hit.get("text") or "")

    # url_hint supports page linking for PDFs
    url_hint = ""
    if loc_kind == "page" and loc_start is not None:
        url_hint = f"#page={loc_start}"

    open_url = _build_open_url(public_path, url_hint)
    download_url = _build_download_url(doc)

    ref: Dict[str, Any] = {
        "document": doc,

        # ✅ path remains URL path for static mount (backward compatible)
        "path": public_path,

        # ✅ new fields for mobile/devs
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
    decision = decide_smalltalk(question or "")
    if decision and decision.is_greeting_only:
        return {"answer": decision.response, "references": []}

    if not retrieval or not retrieval.get("has_evidence"):
        return _refuse()

    evidence_docs = retrieval.get("evidence", []) or []
    if not evidence_docs:
        return _refuse()

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
