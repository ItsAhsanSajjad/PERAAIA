from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class ExtractedUnit:
    """
    A traceable extraction unit that can later be chunked.
    - PDF  => one unit per page
    - DOCX => one unit per section (heading) or paragraph-range block
    """
    doc_name: str
    source_type: str              # "pdf" | "docx"
    loc_kind: str                 # "page" | "section" | "paragraphs"
    loc_start: Any                # int page number, or str anchor
    loc_end: Any                  # int page number, or str anchor
    text: str

    # optional metadata
    path: Optional[str] = None
    doc_rank: int = 0


# -----------------------------
# Helpers
# -----------------------------
SUPPORTED_EXTS = (".pdf", ".docx")

_NUL_RE = re.compile(r"\x00+")
# common garbage from PDF extraction
_PAGE_NUM_RE = re.compile(r"^\s*(page\s*)?\d+\s*(of\s*\d+)?\s*$", re.I)

# Keep Urdu/Arabic block characters; do not destroy them during cleaning
_MULTI_NEWLINES_RE = re.compile(r"\n{4,}")

# Bullet patterns
_BULLET_RE = re.compile(r"^\s*([•\-\u2022]|\d+[\)\.]|[A-Za-z][\)\.])\s+")
# Detect tabular alignment (raw, before whitespace collapsing)
_TABLE_LIKE_RE = re.compile(r"(\t+|\s{2,})")

# Hyphenation: join "regula-" + "tory" (very common in PDFs)
_HYPHEN_END_RE = re.compile(r".*[\w\u0600-\u06FF]-$")


def _clean_text_general(s: str) -> str:
    """
    General cleaning after structure decisions are already made.
    IMPORTANT: we do NOT collapse multiple spaces here globally
    because tables may have been converted into " | " already.
    """
    s = s or ""
    s = _NUL_RE.sub(" ", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _MULTI_NEWLINES_RE.sub("\n\n\n", s)
    # Trim each line but preserve line boundaries
    s = "\n".join([ln.strip() for ln in s.split("\n")])
    return s.strip()


def _is_heading_style(style_name: str) -> bool:
    if not style_name:
        return False
    sn = style_name.strip().lower()
    return sn.startswith("heading")


def discover_documents(data_dir: str = "assets/data") -> List[str]:
    data_dir = data_dir.replace("\\", "/")
    if not os.path.isdir(data_dir):
        return []
    out: List[str] = []
    for name in os.listdir(data_dir):
        p = os.path.join(data_dir, name).replace("\\", "/")
        if not os.path.isfile(p):
            continue
        low = name.lower()
        if low.endswith(SUPPORTED_EXTS):
            out.append(p)
    return sorted(out)


# -----------------------------
# PDF extraction quality helpers
# -----------------------------
def _pdf_lines_raw(text: str) -> List[str]:
    """
    Split PDF raw text into lines WITHOUT collapsing multiple spaces.
    This is critical for table detection.
    """
    t = text or ""
    t = _NUL_RE.sub(" ", t)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    # do NOT collapse spaces here
    lines = [ln.strip("\n") for ln in t.split("\n")]
    lines = [ln.strip() for ln in lines if ln and ln.strip()]
    return lines


def _normalize_line_for_header_footer(line: str) -> str:
    """
    Normalize digits so "Page 12" ~= "Page 03".
    Also normalize whitespace so positional repetition can be detected.
    """
    l = (line or "").strip().lower()
    l = re.sub(r"\d+", "0", l)
    l = re.sub(r"\s+", " ", l).strip()
    return l


def _is_header_footer_candidate(line: str) -> bool:
    """
    Avoid over-stripping: only consider short / low-density lines.
    """
    s = (line or "").strip()
    if not s:
        return False
    if len(s) > 120:
        return False
    # page number lines are always candidates
    if _PAGE_NUM_RE.match(s):
        return True
    # low-content lines: mostly punctuation/digits
    letters = len(re.findall(r"[A-Za-z\u0600-\u06FF]", s))
    if letters <= 6:
        return True
    # common boilerplate header/footer signals
    sl = s.lower()
    if "punjab" in sl and ("authority" in sl or "regulatory" in sl or "enforcement" in sl):
        return True
    return False


def _detect_repeated_header_footer(page_lines: List[List[str]], min_pages: int = 3) -> Dict[str, set]:
    """
    Find repeated first/last lines across many pages (headers/footers).
    Returns {"header": set(lines), "footer": set(lines)} of normalized strings.
    Conservative: only lines that look like headers/footers are eligible.
    """
    if len(page_lines) < min_pages:
        return {"header": set(), "footer": set()}

    first_counts: Dict[str, int] = {}
    last_counts: Dict[str, int] = {}

    eligible_pages = 0
    for lines in page_lines:
        if not lines:
            continue
        eligible_pages += 1

        # consider first 2 lines, last 2 lines (only if candidate)
        for ln in lines[:2]:
            if not _is_header_footer_candidate(ln):
                continue
            k = _normalize_line_for_header_footer(ln)
            if k:
                first_counts[k] = first_counts.get(k, 0) + 1

        for ln in lines[-2:]:
            if not _is_header_footer_candidate(ln):
                continue
            k = _normalize_line_for_header_footer(ln)
            if k:
                last_counts[k] = last_counts.get(k, 0) + 1

    if eligible_pages < min_pages:
        return {"header": set(), "footer": set()}

    threshold = max(2, int(0.60 * eligible_pages))  # more conservative than 50%
    header = {k for k, c in first_counts.items() if c >= threshold}
    footer = {k for k, c in last_counts.items() if c >= threshold}
    return {"header": header, "footer": footer}


def _strip_headers_footers(lines: List[str], hf: Dict[str, set]) -> List[str]:
    if not lines:
        return lines
    out: List[str] = []
    for ln in lines:
        s = (ln or "").strip()
        if not s:
            continue
        norm = _normalize_line_for_header_footer(s)
        if norm in hf.get("header", set()) or norm in hf.get("footer", set()):
            continue
        if _PAGE_NUM_RE.match(s):
            continue
        out.append(s)
    return out


def _looks_like_table_row(raw_line: str) -> bool:
    """
    Detect table-like lines before whitespace collapse:
    - contains multiple spaces or tabs
    - has enough content
    """
    s = (raw_line or "").rstrip()
    if len(s) < 20:
        return False
    if _TABLE_LIKE_RE.search(s) is None:
        return False
    # avoid treating normal sentences as tables
    if s.count("  ") >= 1 or "\t" in s:
        # needs multiple tokens
        tokens = re.findall(r"[A-Za-z\u0600-\u06FF0-9]{2,}", s)
        return len(tokens) >= 3
    return False


def _normalize_table_row(raw_line: str) -> str:
    """
    Normalize table-like spacing into a stable delimiter.
    This makes retrieval + grounding stronger than losing column structure.
    """
    s = (raw_line or "").strip()
    s = re.sub(r"\t+", "  ", s)
    # Convert 2+ spaces into a visible column delimiter
    s = re.sub(r"\s{2,}", " | ", s).strip()
    return s


def _join_pdf_lines(lines: List[str]) -> str:
    """
    Join PDF-extracted lines into a cleaner text:
    - keep bullets / table-like rows as new lines
    - merge narrative lines into paragraphs
    - fix hyphenated line breaks
    """
    if not lines:
        return ""

    merged: List[str] = []
    buf: List[str] = []

    def flush_buf() -> None:
        nonlocal buf
        if not buf:
            return
        merged.append(" ".join(buf).strip())
        buf = []

    prev_line_raw: Optional[str] = None

    for ln in lines:
        raw = (ln or "").strip()
        if not raw:
            flush_buf()
            prev_line_raw = None
            continue

        # bullets and tables preserved as their own lines
        if _BULLET_RE.search(raw) is not None:
            flush_buf()
            merged.append(raw)
            prev_line_raw = raw
            continue

        if _looks_like_table_row(raw):
            flush_buf()
            merged.append(_normalize_table_row(raw))
            prev_line_raw = raw
            continue

        # Hyphenation fix: previous buffer last token ends with '-' and current begins with a word
        if buf:
            prev = buf[-1]
            if _HYPHEN_END_RE.match(prev):
                # join without space: "regula-" + "tory" => "regulatory"
                buf[-1] = prev[:-1] + raw
                prev_line_raw = raw
                continue

        # Preserve sectioning after colon
        if buf and buf[-1].endswith(":"):
            flush_buf()
            buf.append(raw)
            prev_line_raw = raw
            continue

        buf.append(raw)
        prev_line_raw = raw

    flush_buf()

    text = "\n".join(merged)
    return _clean_text_general(text)


# -----------------------------
# PDF Extraction
# -----------------------------
def extract_pdf_units(pdf_path: str) -> List[ExtractedUnit]:
    """
    Extract PDF page-by-page.
    - conservative repeated header/footer removal
    - table preservation
    - hyphenation repair
    """
    units: List[ExtractedUnit] = []
    pdf_path = (pdf_path or "").replace("\\", "/")
    doc_name = os.path.basename(pdf_path)

    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        pages = reader.pages
    except Exception as e:
        print(f"Error extracting {pdf_path}: {e}")
        return units

    # first pass: raw page lines for header/footer detection
    raw_lines_by_page: List[List[str]] = []
    for page in pages:
        try:
            raw = page.extract_text() or ""
        except Exception:
            raw = ""
        raw_lines_by_page.append(_pdf_lines_raw(raw))

    hf = _detect_repeated_header_footer(raw_lines_by_page)

    # second pass: build units
    for i, lines in enumerate(raw_lines_by_page):
        page_no = i + 1

        lines2 = _strip_headers_footers(lines, hf)

        # fallback: if we stripped too much, use original lines
        if len(lines2) < max(3, int(0.25 * len(lines))):
            lines2 = lines

        text = _join_pdf_lines(lines2)
        text = _clean_text_general(text)

        # keep only meaningful pages
        if not text:
            continue

        units.append(
            ExtractedUnit(
                doc_name=doc_name,
                source_type="pdf",
                loc_kind="page",
                loc_start=page_no,
                loc_end=page_no,
                text=text,
                path=pdf_path,
            )
        )
    return units


# -----------------------------
# DOCX Extraction
# -----------------------------
def extract_docx_units(
    docx_path: str,
    min_chars_per_unit: int = 800,
    max_chars_per_unit: int = 6000
) -> List[ExtractedUnit]:
    """
    Extract DOCX into stable units using headings.
    """
    units: List[ExtractedUnit] = []
    docx_path = (docx_path or "").replace("\\", "/")
    doc_name = os.path.basename(docx_path)

    try:
        from docx import Document
        doc = Document(docx_path)
    except Exception:
        return units

    paras: List[Dict[str, Any]] = []
    para_idx = 0
    current_heading = ""

    for p in doc.paragraphs:
        txt = (p.text or "").strip()
        if not txt:
            continue

        style_name = ""
        try:
            style_name = p.style.name if p.style else ""
        except Exception:
            style_name = ""

        if _is_heading_style(style_name):
            current_heading = txt
            continue

        para_idx += 1
        paras.append({
            "i": para_idx,
            "heading": current_heading,
            "text": txt
        })

    if not paras:
        return units

    has_any_heading = any(p["heading"] for p in paras)

    if has_any_heading:
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for p in paras:
            h = p["heading"] or "Untitled"
            groups.setdefault(h, []).append(p)

        for heading, items in groups.items():
            _emit_docx_group_as_units(
                units=units,
                doc_name=doc_name,
                docx_path=docx_path,
                heading=heading,
                items=items,
                min_chars=min_chars_per_unit,
                max_chars=max_chars_per_unit
            )
        return units

    _emit_docx_paragraph_blocks(
        units=units,
        doc_name=doc_name,
        docx_path=docx_path,
        items=paras,
        min_chars=min_chars_per_unit,
        max_chars=max_chars_per_unit
    )
    return units


def _emit_docx_group_as_units(
    units: List[ExtractedUnit],
    doc_name: str,
    docx_path: str,
    heading: str,
    items: List[Dict[str, Any]],
    min_chars: int,
    max_chars: int
) -> None:
    buffer: List[str] = []
    start_i = None
    end_i = None
    char_count = 0

    for p in items:
        txt = p["text"]
        i = p["i"]

        if start_i is None:
            start_i = i
        end_i = i

        buffer.append(txt)
        char_count += len(txt) + 1

        if char_count >= max_chars:
            text = _clean_text_general("\n".join(buffer))
            if text:
                anchor = f'Section: "{heading}" (Paragraphs {start_i}–{end_i})'
                units.append(
                    ExtractedUnit(
                        doc_name=doc_name,
                        source_type="docx",
                        loc_kind="section",
                        loc_start=anchor,
                        loc_end=anchor,
                        text=text,
                        path=docx_path,
                    )
                )
            buffer = []
            start_i = None
            end_i = None
            char_count = 0

    if buffer:
        text = _clean_text_general("\n".join(buffer))
        if text:
            anchor = f'Section: "{heading}" (Paragraphs {start_i}–{end_i})'
            units.append(
                ExtractedUnit(
                    doc_name=doc_name,
                    source_type="docx",
                    loc_kind="section",
                    loc_start=anchor,
                    loc_end=anchor,
                    text=text,
                    path=docx_path,
                )
            )


def _emit_docx_paragraph_blocks(
    units: List[ExtractedUnit],
    doc_name: str,
    docx_path: str,
    items: List[Dict[str, Any]],
    min_chars: int,
    max_chars: int
) -> None:
    buffer: List[str] = []
    start_i = None
    end_i = None
    char_count = 0

    for p in items:
        txt = p["text"]
        i = p["i"]

        if start_i is None:
            start_i = i
        end_i = i

        buffer.append(txt)
        char_count += len(txt) + 1

        if char_count >= max_chars:
            text = _clean_text_general("\n".join(buffer))
            if text:
                anchor = f"Paragraphs {start_i}–{end_i}"
                units.append(
                    ExtractedUnit(
                        doc_name=doc_name,
                        source_type="docx",
                        loc_kind="paragraphs",
                        loc_start=anchor,
                        loc_end=anchor,
                        text=text,
                        path=docx_path,
                    )
                )
            buffer = []
            start_i = None
            end_i = None
            char_count = 0

    if buffer:
        text = _clean_text_general("\n".join(buffer))
        if text:
            anchor = f"Paragraphs {start_i}–{end_i}"
            units.append(
                ExtractedUnit(
                    doc_name=doc_name,
                    source_type="docx",
                    loc_kind="paragraphs",
                    loc_start=anchor,
                    loc_end=anchor,
                    text=text,
                    path=docx_path,
                )
            )


# -----------------------------
# Unified interface (PDF + DOCX)
# -----------------------------
def extract_units_from_file(path: str) -> List[ExtractedUnit]:
    p = (path or "").replace("\\", "/")
    low = p.lower()
    if low.endswith(".pdf"):
        return extract_pdf_units(p)
    if low.endswith(".docx"):
        return extract_docx_units(p)
    return []


def extract_units_from_files(paths: List[str]) -> List[ExtractedUnit]:
    all_units: List[ExtractedUnit] = []
    for p in paths:
        all_units.extend(extract_units_from_file(p))
    return all_units
