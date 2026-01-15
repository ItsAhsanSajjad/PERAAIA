from __future__ import annotations

import os
import re
import time
import difflib
import hashlib
from dataclasses import dataclass
from typing import Optional, Tuple, List, Set, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

# =============================================================================
# PERA AI Assistant (Document-Only, ChatGPT-style)
# - English / Urdu / Roman Urdu ONLY
# - Strict grounding: require file_search citations; otherwise refuse
# - Adds: Document Name + Page (PDF) / Section Anchor (DOCX) references
#   by mapping OpenAI "file_citation.quote" back to local files in assets/data/
# =============================================================================

# ---------------- ENV / CLIENT ----------------
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("API key not found! Make sure .env file exists with OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

ASSISTANT_ID = os.getenv("ASSISTANT_ID", "asst_Ojp1LRjdwWbpj5mam6EMxYkB")

ASSISTANT_TIMEOUT_SECONDS = int(os.getenv("ASSISTANT_TIMEOUT_SECONDS", "120"))
ASSISTANT_RETRY_TIMEOUT_SECONDS = int(os.getenv("ASSISTANT_RETRY_TIMEOUT_SECONDS", "60"))
ASSISTANT_POLL_INTERVAL = float(os.getenv("ASSISTANT_POLL_INTERVAL", "0.35"))

DEBUG_ASSISTANT = os.getenv("DEBUG_ASSISTANT", "0").strip() in ("1", "true", "True", "yes", "YES")

# ---------------- LOCAL DOCS ROOT ----------------
# Your screenshot indicates documents live under: assets/data/
ASSETS_DATA_DIR = os.getenv("ASSETS_DATA_DIR", "assets/data").replace("\\", "/")

# ---------------- LANGUAGE + SMALLTALK ----------------
_SMALLTALK_PATTERNS = (
    r"^\s*(hi|hello|hey)\b",
    r"^\s*(aoa|a\.o\.a|assalam|assalamu|asalam|salam|salaam|slm)\b",
    r"^\s*(thanks|thank you|shukriya|jazakallah)\b",
    r"^\s*(good\s+morning|good\s+afternoon|good\s+evening)\b",
    r"^\s*(ap\s+kaise|aap\s+kaise|kya\s+haal|kya\s+hal)\b",
)

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")  # Hindi/Devanagari script
URDU_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")


def _looks_like_smalltalk(text: str) -> bool:
    t = text.strip().lower()
    return any(re.search(p, t) for p in _SMALLTALK_PATTERNS)


def _detect_language(text: str) -> str:
    """
    Returns:
      - "english" | "urdu" | "roman_urdu" | "unsupported"
    """
    if DEVANAGARI_RE.search(text):
        return "unsupported"

    if URDU_ARABIC_RE.search(text):
        return "urdu"

    t = text.lower()
    roman_markers = [
        "ap", "aap", "kya", "kaise", "kaisay", "kesy", "kese", "hain", "hai",
        "matlab", "batain", "batao", "kahan", "kab", "kyun", "halaat", "haal"
    ]
    if any(re.search(rf"\b{m}\b", t) for m in roman_markers):
        return "roman_urdu"

    return "english"


def _reply_style_instruction(lang: str) -> str:
    if lang == "urdu":
        return "Respond in Urdu."
    if lang == "roman_urdu":
        return "Respond in Roman Urdu."
    return "Respond in English."


def _unsupported_language_message() -> str:
    return (
        "I currently support only English, Urdu, and Roman Urdu. "
        "Please ask the same question in one of these languages."
    )


def _smalltalk_response(lang: str, user_text: str = "") -> str:
    t_raw = (user_text or "").strip()
    t = t_raw.lower()

    t_norm = re.sub(r"[^a-z\u0600-\u06FF\s]", " ", t)
    t_norm = re.sub(r"\s+", " ", t_norm).strip()

    has_urdu_script = bool(URDU_ARABIC_RE.search(t_raw))

    salam_pattern = r"\b(a\.?o\.?a|ass?al[a]?m(u)?(\s+o)?\s+alaikum|ass?alamu\s+alaikum|asalam|salaam|salam|slm)\b"
    user_used_salam = bool(re.search(salam_pattern, t_norm))

    how_are_you_pattern = r"\b(kaise|kaisi|kaisay|kese|kesy|kya\s+haal|haal\s+chaal|hal)\b"
    has_how_are_you = bool(re.search(how_are_you_pattern, t_norm)) or bool(re.search(r"(کیسے|کیسی|کیا حال)", t_raw))

    if user_used_salam:
        if has_urdu_script or lang == "urdu":
            if has_how_are_you:
                return "وعلیکم السلام! میں ٹھیک ہوں۔ آپ دستاویز یا PERA کے بارے میں کیا جاننا چاہتے ہیں؟"
            return "وعلیکم السلام! آپ کیسے ہیں؟ آپ دستاویز یا PERA کے بارے میں کیا جاننا چاہتے ہیں؟"
        else:
            if has_how_are_you:
                return "Wa Alaikum Assalam! Main theek hoon. Aap document/PERA ke bare mein kya poochna chahte hain?"
            return "Wa Alaikum Assalam! Aap kaise hain? Aap document/PERA ke bare mein kya poochna chahte hain?"

    if has_urdu_script or lang == "urdu":
        if has_how_are_you:
            return "سلام! میں ٹھیک ہوں۔ آپ دستاویز یا PERA کے بارے میں کیا جاننا چاہتے ہیں؟"
        return "سلام! آپ دستاویز یا PERA کے بارے میں کیا جاننا چاہتے ہیں؟"

    if lang == "roman_urdu":
        if has_how_are_you:
            return "Hello! Main theek hoon. Aap document/PERA ke bare mein kya poochna chahte hain?"
        return "Hello! Aap document/PERA ke bare mein kya poochna chahte hain?"

    return "Hi! How can I help you with the document / PERA?"


# ---------------- GREETING + QUESTION SPLIT ----------------
def _split_greeting_and_question(text: str) -> Tuple[bool, str, str]:
    raw = (text or "").strip()
    if not raw:
        return False, "", ""

    greeting_prefix_re = re.compile(
        r"^(?P<greet>"
        r"(hi|hello|hey)"
        r"|a\.?o\.?a"
        r"|slm"
        r"|salaam|salam"
        r"|ass?al[a]?m(u)?(\s+o)?\s+alaikum"
        r"|ass?alamu\s+alaikum"
        r"|assalam"
        r")\b",
        re.IGNORECASE
    )

    m = greeting_prefix_re.search(raw)
    if not m:
        return False, "", raw

    remaining = raw[m.end():].lstrip(" ,:-—–\n\t")

    has_question_signal = (
        ("?" in remaining)
        or bool(re.search(r"\b(what|who|when|where|why|how|define|explain|tell|meaning|full\s+form|abbreviation)\b", remaining.lower()))
        or bool(re.search(r"\b(kya|kaise|kaisay|kese|kesy|matlab|tafseel|batao|batain|samjhao)\b", remaining.lower()))
        or bool(URDU_ARABIC_RE.search(remaining))
        or bool(DEVANAGARI_RE.search(remaining))
    )

    if not remaining or not has_question_signal:
        return True, "", ""

    greet_token = (m.group("greet") or "").lower()
    used_salam = any(x in greet_token for x in ["aoa", "salam", "salaam", "assal", "slm"])
    has_urdu_script = bool(URDU_ARABIC_RE.search(raw))

    if used_salam:
        ack = "وعلیکم السلام! " if has_urdu_script else "Wa Alaikum Assalam! "
    else:
        ack = "Hi! "

    return True, ack, remaining


# ---------------- PDF TEXT INDEX (local; used for silent mapping only) ----------------
def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def _scan_assets_docs() -> List[str]:
    """
    Auto-discover PDF and DOCX in assets/data (no manual registration).
    """
    root = ASSETS_DATA_DIR
    out: List[str] = []
    if not os.path.isdir(root):
        return out

    for name in os.listdir(root):
        p = os.path.join(root, name).replace("\\", "/")
        if not os.path.isfile(p):
            continue
        low = name.lower()
        if low.endswith(".pdf") or low.endswith(".docx"):
            out.append(p)
    return sorted(out)


def _candidate_pdf_paths() -> List[str]:
    # Keep your earlier fallbacks but also include assets/data auto-discovery
    env_one = os.getenv("BOOK_PDF_PATH", "").strip()
    env_many = os.getenv("BOOK_PDF_PATHS", "").strip()

    paths: List[str] = []
    if env_many:
        paths.extend([p.strip() for p in env_many.split(",") if p.strip()])
    if env_one:
        paths.insert(0, env_one)

    # Auto-discovered docs (only PDFs for this "intent mapping index")
    for p in _scan_assets_docs():
        if p.lower().endswith(".pdf"):
            paths.append(p)

    # Old fallbacks
    paths.extend([
        "data/books/book.pdf",
        "data/book.pdf",
    ])

    seen = set()
    existing = []
    for p in paths:
        p = p.replace("\\", "/")
        if p in seen:
            continue
        seen.add(p)
        if os.path.exists(p):
            existing.append(p)
    return existing


PDF_PATHS = _candidate_pdf_paths()


class PDFTextIndex:
    def __init__(self, pdf_paths: List[str]):
        self.pdf_paths = pdf_paths
        self._loaded = False
        self._text = ""
        self._vocab: Set[str] = set()

    def load(self) -> None:
        if self._loaded:
            return
        if not self.pdf_paths:
            self._loaded = True
            return

        raw_all = []
        for path in self.pdf_paths:
            raw_all.append(self._extract_text_from_pdf(path))
        raw = "\n".join([t for t in raw_all if t])

        self._text = _normalize(raw)

        tokens = re.findall(r"[A-Za-z]{2,}|[\u0600-\u06FF]{2,}", raw)
        vocab = set()
        for tok in tokens:
            tt = _normalize(tok)
            if 3 <= len(tt) <= 30:
                vocab.add(tt)
        self._vocab = vocab
        self._loaded = True

    def has_text(self) -> bool:
        self.load()
        return bool(self._text)

    def contains_term(self, term: str) -> bool:
        self.load()
        t = _normalize(term)
        if not t or not self._text:
            return False

        if re.fullmatch(r"[a-z0-9_\-]+", t):
            return re.search(rf"\b{re.escape(t)}\b", self._text) is not None
        return t in self._text

    def has_pera(self) -> bool:
        self.load()
        return "pera" in self._vocab or self.contains_term("PERA")

    def best_vocab_match(self, term: str, cutoff: float = 0.86) -> Optional[str]:
        self.load()
        t = _normalize(term)
        if not t or len(t) < 3 or not self._vocab:
            return None
        if t in self._vocab:
            return t
        matches = difflib.get_close_matches(t, list(self._vocab), n=1, cutoff=cutoff)
        return matches[0] if matches else None

    @staticmethod
    def _extract_text_from_pdf(path: str) -> str:
        try:
            from pypdf import PdfReader
            reader = PdfReader(path)
            parts = []
            for page in reader.pages:
                parts.append(page.extract_text() or "")
            return "\n".join(parts)
        except Exception:
            return ""


PDF_INDEX = PDFTextIndex(PDF_PATHS)


# ---------------- CHATGPT-STYLE SILENT INTENT REFINEMENT ----------------
_STOPWORDS = {
    "what", "who", "when", "where", "why", "how", "is", "are", "was", "were",
    "the", "a", "an", "of", "in", "on", "at", "for", "to", "from", "and", "or",
    "please", "tell", "me", "about", "explain", "define", "meaning", "full", "form",
    "abbreviation", "composition", "authority", "role", "function", "powers", "duties",
    "chapter", "section", "act", "rules", "law"
}


def _extract_candidate_terms(q: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,20}", q or "")
    out: List[str] = []
    seen = set()
    for t in tokens:
        tl = t.lower()
        if tl in _STOPWORDS:
            continue
        if tl in seen:
            continue
        seen.add(tl)
        out.append(t)
    return out


def _looks_pera_related(q: str) -> bool:
    t = (q or "").lower()
    cues = [
        "pera", "authority", "act", "section", "chapter",
        "composition", "powers", "functions", "appointment",
        "chairperson", "member", "director", "secretary",
        "regulatory", "enforcement", "punjab"
    ]
    return any(c in t for c in cues) or bool(URDU_ARABIC_RE.search(q or ""))


def _silently_refine_question(q: str) -> Tuple[str, Dict[str, str]]:
    q = re.sub(r"\s+", " ", (q or "")).strip()
    corrections: Dict[str, str] = {}

    if not PDF_INDEX.has_text():
        return q, corrections

    refined = q

    for cand in _extract_candidate_terms(q):
        if PDF_INDEX.contains_term(cand):
            continue

        best = PDF_INDEX.best_vocab_match(cand, cutoff=0.86)
        if best and best != _normalize(cand):
            corrections[cand] = best
            refined = re.sub(rf"\b{re.escape(cand)}\b", best, refined, flags=re.IGNORECASE)

    if PDF_INDEX.has_pera() and _looks_pera_related(refined):
        m = re.search(r"\b(in|of|for)\s+([A-Za-z]{3,12})\b", refined, flags=re.IGNORECASE)
        if m:
            token = m.group(2)
            if token and (not PDF_INDEX.contains_term(token)):
                if token.lower() not in _STOPWORDS:
                    corrections[token] = "pera"
                    refined = re.sub(rf"\b{re.escape(token)}\b", "pera", refined, flags=re.IGNORECASE)

    if PDF_INDEX.has_pera() and _looks_pera_related(refined):
        if not re.search(r"\bpera\b", refined, flags=re.IGNORECASE):
            refined = refined + " (PERA)"

    return refined, corrections


def _sanitize_answer_text(answer: str, corrections: Dict[str, str]) -> str:
    if not answer or not corrections:
        return answer or ""

    out = answer
    for wrong, correct in corrections.items():
        if not wrong or not correct:
            continue
        out = re.sub(
            rf"\b{re.escape(wrong)}\b",
            correct.upper() if correct == "pera" else correct,
            out,
            flags=re.IGNORECASE
        )

    if PDF_INDEX.has_pera():
        if re.search(r"\bJERA\b", out) and not PDF_INDEX.contains_term("JERA"):
            out = re.sub(r"\bJERA\b", "PERA", out)

    return out


# ---------------- MESSAGES ----------------
def _no_grounded_answer(lang: str) -> str:
    # English: your strict refusal sentence
    if lang == "urdu":
        return "فراہم کردہ دستاویزات میں اس سوال سے متعلق کوئی معلومات موجود نہیں ہیں۔"
    if lang == "roman_urdu":
        return "Provided document(s) mein is sawal se related koi maloomat maujood nahin hai."
    return "There is no information related to this question in the provided document(s)."


def _timeout_message(lang: str) -> str:
    if lang == "urdu":
        return "جواب دینے میں معمول سے زیادہ وقت لگ رہا ہے۔ براہِ کرم دوبارہ کوشش کریں یا سوال کو مختصر کریں۔"
    if lang == "roman_urdu":
        return "Response mein zyada time lag raha hai. Please dobara try karein ya sawal thora short karein."
    return "It’s taking longer than usual to respond. Please try again or shorten the question."


# =============================================================================
# Citation -> (Document Name + Page/Section) mapping
# =============================================================================

@dataclass
class CitationHit:
    file_id: str
    quote: str


# Simple caches to avoid repeated heavy extraction
_DOCNAME_CACHE: Dict[str, str] = {}             # file_id -> filename
_PDF_PAGE_TEXT_CACHE: Dict[str, List[str]] = {} # local_path -> [page_text]
_DOCX_PARA_CACHE: Dict[str, List[Dict[str, Any]]] = {}  # local_path -> [{"i":int,"heading":str,"text":str}]


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _list_local_docs() -> List[str]:
    return _scan_assets_docs()


def _find_local_file_by_name(filename: str) -> Optional[str]:
    """
    Match by exact filename (case-insensitive) inside assets/data.
    """
    if not filename:
        return None
    target = filename.lower()
    for p in _list_local_docs():
        if os.path.basename(p).lower() == target:
            return p
    return None


def _get_filename_for_file_id(file_id: str) -> Optional[str]:
    """
    OpenAI Files API: resolve file_id to filename.
    Cache results.
    """
    if not file_id:
        return None
    if file_id in _DOCNAME_CACHE:
        return _DOCNAME_CACHE[file_id]
    try:
        f = client.files.retrieve(file_id)
        name = getattr(f, "filename", None) or getattr(f, "name", None)
        if name:
            _DOCNAME_CACHE[file_id] = name
            return name
    except Exception:
        return None
    return None


def _load_pdf_pages(path: str) -> List[str]:
    if path in _PDF_PAGE_TEXT_CACHE:
        return _PDF_PAGE_TEXT_CACHE[path]
    pages: List[str] = []
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        for pg in reader.pages:
            pages.append(pg.extract_text() or "")
    except Exception:
        pages = []
    _PDF_PAGE_TEXT_CACHE[path] = pages
    return pages


def _load_docx_paragraphs(path: str) -> List[Dict[str, Any]]:
    """
    Returns list of paragraphs with heading context.
    Each item: {"i": 1-based index, "heading": "<nearest heading>", "text": "..."}
    """
    if path in _DOCX_PARA_CACHE:
        return _DOCX_PARA_CACHE[path]

    paras: List[Dict[str, Any]] = []
    try:
        from docx import Document
        doc = Document(path)
        current_heading = ""
        i = 0
        for p in doc.paragraphs:
            txt = (p.text or "").strip()
            if not txt:
                continue
            style = (p.style.name or "").lower() if p.style else ""
            if "heading" in style:
                current_heading = txt
                continue
            i += 1
            paras.append({"i": i, "heading": current_heading, "text": txt})
    except Exception:
        paras = []

    _DOCX_PARA_CACHE[path] = paras
    return paras


def _tokenize_light(s: str) -> List[str]:
    s = _normalize(s)
    # keep words/numbers; ignore very short tokens
    toks = re.findall(r"[a-z0-9\u0600-\u06FF]{3,}", s)
    return toks[:200]  # cap


def _best_page_for_quote(pages: List[str], quote: str) -> Optional[int]:
    """
    Returns 1-based page number where quote likely appears.
    Uses token overlap + substring checks (fast and robust enough).
    """
    if not pages or not quote:
        return None

    qn = _normalize(quote)
    if len(qn) < 20:
        # too short; risk false matches
        return None

    qtoks = set(_tokenize_light(quote))
    if not qtoks:
        return None

    best_score = 0.0
    best_idx = None

    for idx0, ptxt in enumerate(pages):
        pn = _normalize(ptxt)
        if not pn:
            continue

        # strong match: substring
        if qn[:120] in pn:
            return idx0 + 1

        ptoks = set(_tokenize_light(ptxt))
        if not ptoks:
            continue

        # overlap score
        inter = len(qtoks & ptoks)
        score = inter / max(1, min(len(qtoks), 40))

        if score > best_score:
            best_score = score
            best_idx = idx0 + 1

    # conservative threshold to avoid wrong page numbers
    if best_score >= 0.35:
        return best_idx
    return None


def _best_docx_anchor_for_quote(paras: List[Dict[str, Any]], quote: str) -> Optional[str]:
    """
    Returns a stable "section/page equivalent" anchor for DOCX:
    Section: <Heading> (Paragraphs a–b)
    """
    if not paras or not quote:
        return None

    qn = _normalize(quote)
    if len(qn) < 20:
        return None

    qtoks = set(_tokenize_light(quote))
    if not qtoks:
        return None

    best_score = 0.0
    best_para_i = None

    for item in paras:
        txt = item.get("text", "")
        tn = _normalize(txt)
        if not tn:
            continue

        # substring quick check
        if qn[:120] in tn:
            best_para_i = item["i"]
            best_score = 1.0
            break

        ttoks = set(_tokenize_light(txt))
        inter = len(qtoks & ttoks)
        score = inter / max(1, min(len(qtoks), 30))
        if score > best_score:
            best_score = score
            best_para_i = item["i"]

    if best_para_i is None or best_score < 0.30:
        return None

    # Build a small window for citation (para range)
    start_i = max(1, best_para_i - 3)
    end_i = best_para_i + 3

    # Find heading near the best paragraph
    heading = ""
    # scan around best index in cached list by position
    for item in paras:
        if item["i"] == best_para_i:
            heading = item.get("heading", "") or ""
            break

    if heading:
        return f'Section: "{heading}" (Paragraphs {start_i}–{end_i})'
    return f"Paragraphs {start_i}–{end_i}"


def _references_from_citations(citations: List[CitationHit]) -> List[Dict[str, Any]]:
    """
    Convert OpenAI file citations (file_id + quote) into references with:
    - document name
    - page numbers (PDF) or section anchors (DOCX)
    """
    refs: List[Dict[str, Any]] = []
    seen_keys: Set[str] = set()

    for c in citations:
        filename = _get_filename_for_file_id(c.file_id) or "Unknown document"
        local_path = _find_local_file_by_name(filename)

        # Always include at least document name even if local mapping fails
        if not local_path:
            key = f"{filename}|unknown"
            if key not in seen_keys:
                seen_keys.add(key)
                refs.append({"document": filename})
            continue

        low = filename.lower()

        if low.endswith(".pdf"):
            pages = _load_pdf_pages(local_path)
            page = _best_page_for_quote(pages, c.quote)
            if page is not None:
                key = f"{filename}|p{page}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    refs.append({"document": filename, "page_start": page, "page_end": page})
            else:
                key = f"{filename}|unknown"
                if key not in seen_keys:
                    seen_keys.add(key)
                    refs.append({"document": filename})
            continue

        if low.endswith(".docx"):
            paras = _load_docx_paragraphs(local_path)
            anchor = _best_docx_anchor_for_quote(paras, c.quote)
            if anchor:
                key = f"{filename}|{anchor}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    refs.append({"document": filename, "loc": anchor})
            else:
                key = f"{filename}|unknown"
                if key not in seen_keys:
                    seen_keys.add(key)
                    refs.append({"document": filename})
            continue

        # unknown type
        key = f"{filename}|unknown"
        if key not in seen_keys:
            seen_keys.add(key)
            refs.append({"document": filename})

    return refs


# ---------------- ASSISTANT RESPONSE PARSING ----------------
def _extract_assistant_text_and_citations(msg) -> Tuple[str, List[CitationHit]]:
    text_parts: List[str] = []
    citations: List[CitationHit] = []

    content_blocks = getattr(msg, "content", None) or []
    for block in content_blocks:
        block_text = getattr(block, "text", None)
        if not block_text:
            continue

        value = getattr(block_text, "value", "")
        if value:
            text_parts.append(value)

        annotations = getattr(block_text, "annotations", None) or []
        for ann in annotations:
            ann_type = getattr(ann, "type", None)
            if ann_type != "file_citation":
                continue
            fc = getattr(ann, "file_citation", None)
            if not fc:
                continue
            file_id = getattr(fc, "file_id", None) or ""
            quote = getattr(fc, "quote", None) or ""
            if file_id and quote:
                citations.append(CitationHit(file_id=file_id, quote=quote))

    return "\n".join(text_parts).strip(), citations


# ---------------- RETRY PROMPT ----------------
def _rewrite_query_for_retrieval(q: str, lang: str) -> str:
    q0 = (q or "").strip()

    q1 = re.sub(r"\b(what|who|is|are|the|a|an|please|tell|me|about|explain|define)\b", " ", q0, flags=re.I)
    q1 = re.sub(r"\b(kya|hai|hain|ka|ki|ke|mein|me|se|batao|batain|please)\b", " ", q1, flags=re.I)
    q1 = re.sub(r"\s+", " ", q1).strip()
    if len(q1) < 6:
        q1 = q0

    return (
        "You are PERA AI Assistant.\n"
        "Use ONLY the provided documents via file_search.\n"
        "Do NOT invent or expand acronyms/terms unless they appear in the documents AND are cited.\n"
        "Answer in a concise, confident summary (max 6 lines).\n"
        "If not found in documents, say you cannot confirm from the documents.\n"
        f"{_reply_style_instruction(lang)}\n"
        f"Question: {q1}"
    )


# ---------------- PUBLIC API ----------------
def get_or_create_thread_id(existing_thread_id: Optional[str] = None) -> str:
    if existing_thread_id:
        return existing_thread_id
    thread = client.beta.threads.create()
    return thread.id


# ---------------- INTERNAL RUN ----------------
def _run_assistant(thread_id: str, lang: str, timeout_seconds: int) -> Tuple[str, List[CitationHit], bool]:
    grounding_instructions = (
        "You are PERA AI Assistant.\n"
        "Answer using ONLY the provided PERA documents via file_search.\n"
        "Do not guess or invent.\n"
        "Do NOT invent or expand acronyms/terms unless the exact term appears in the documents AND is cited.\n"
        "If the documents do not contain the needed information, say so clearly.\n"
        "Every factual claim must be supported by file_search citations.\n"
        "Style: concise, confident, human-like summary. Max 6 lines.\n"
        f"{_reply_style_instruction(lang)}\n"
    )

    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=ASSISTANT_ID,
        additional_instructions=grounding_instructions,
    )

    start = time.time()

    while True:
        run_status = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)

        if run_status.status == "completed":
            break

        if run_status.status in ("failed", "cancelled", "expired"):
            return "", [], False

        if time.time() - start > timeout_seconds:
            return "", [], True

        time.sleep(ASSISTANT_POLL_INTERVAL)

    messages = client.beta.threads.messages.list(thread_id=thread_id)

    for msg in messages.data:
        if msg.role != "assistant":
            continue
        assistant_text, citations = _extract_assistant_text_and_citations(msg)
        return assistant_text or "", citations, False

    return "", [], False


# ---------------- MAIN FUNCTION ----------------
def ask_assistant(question: str, thread_id: str) -> Dict[str, Any]:
    """
    Returns a dict (for app.py renderer):
      {
        "answer": "...",
        "references": [{"document":"...", "page_start":1, "page_end":1} | {"document":"...", "loc":"..."}]
      }
    """
    if not question or not question.strip():
        return {"answer": "Please enter a question.", "references": []}

    # If user wrote in unsupported script (Hindi/Devanagari etc.), stop immediately
    lang_raw = _detect_language(question)
    if lang_raw == "unsupported":
        return {"answer": _unsupported_language_message(), "references": []}

    # Greeting + question split
    has_greet, greet_ack, remaining_q = _split_greeting_and_question(question)

    # Greeting-only
    if _looks_like_smalltalk(question) and not remaining_q:
        lang_greet = _detect_language(question)
        if lang_greet == "unsupported":
            return {"answer": _unsupported_language_message(), "references": []}
        return {"answer": _smalltalk_response(lang_greet, question), "references": []}

    # Determine actual question
    question_for_qa = remaining_q.strip() if remaining_q else question.strip()

    lang = _detect_language(question_for_qa)
    if lang == "unsupported":
        return {"answer": _unsupported_language_message(), "references": []}

    # Pure smalltalk without a real question
    if _looks_like_smalltalk(question_for_qa) and not re.search(r"[?]", question_for_qa):
        return {"answer": _smalltalk_response(lang, question_for_qa), "references": []}

    # Silent refinement + corrections map (used to sanitize output)
    refined_question, corrections = _silently_refine_question(question_for_qa)

    if DEBUG_ASSISTANT:
        print("\n[DEBUG] user_q:", question_for_qa)
        print("[DEBUG] refined_q:", refined_question)
        print("[DEBUG] corrections:", corrections)
        print("[DEBUG] assets/data docs:", _list_local_docs())

    # 1) First run
    client.beta.threads.messages.create(thread_id=thread_id, role="user", content=refined_question)

    text, citations, timed_out = _run_assistant(thread_id, lang, timeout_seconds=ASSISTANT_TIMEOUT_SECONDS)
    if timed_out:
        return {"answer": _timeout_message(lang), "references": []}

    if text and len(citations) > 0:
        clean = _sanitize_answer_text(text, corrections)
        answer = (greet_ack + clean) if greet_ack else clean
        refs = _references_from_citations(citations)
        return {"answer": answer, "references": refs}

    # 2) One retry with stronger retrieval prompt
    retry_prompt = _rewrite_query_for_retrieval(refined_question, lang)
    client.beta.threads.messages.create(thread_id=thread_id, role="user", content=retry_prompt)

    text2, citations2, timed_out2 = _run_assistant(thread_id, lang, timeout_seconds=ASSISTANT_RETRY_TIMEOUT_SECONDS)
    if timed_out2:
        return {"answer": _timeout_message(lang), "references": []}

    if text2 and len(citations2) > 0:
        clean2 = _sanitize_answer_text(text2, corrections)
        answer2 = (greet_ack + clean2) if greet_ack else clean2
        refs2 = _references_from_citations(citations2)
        return {"answer": answer2, "references": refs2}

    # No grounded answer => refuse safely
    return {"answer": _no_grounded_answer(lang), "references": []}
