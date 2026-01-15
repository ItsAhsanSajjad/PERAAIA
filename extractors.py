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
    - PDF => one unit per page
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
    doc_rank: int = 0             # optional; can be filled by registry later


# -----------------------------
# Helpers
# -----------------------------

SUPPORTED_EXTS = (".pdf", ".docx")


def _clean_text(s: str) -> str:
    s = s or ""
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _is_heading_style(style_name: str) -> bool:
    if not style_name:
        return False
    sn = style_name.strip().lower()
    # "Heading 1", "Heading 2", etc.
    return sn.startswith("heading")


def _safe_int(x, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def discover_documents(data_dir: str = "assets/data") -> List[str]:
    """
    Utility: list PDF/DOCX files in assets/data. (Used in testing)
    """
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
# PDF Extraction
# -----------------------------

def extract_pdf_units(pdf_path: str) -> List[ExtractedUnit]:
    """
    Extract PDF page-by-page.
    Each unit includes:
      doc_name, type="pdf", page=1..N, text
    """
    units: List[ExtractedUnit] = []
    doc_name = os.path.basename(pdf_path)

    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        pages = reader.pages
    except Exception as e:
        # If extraction fails, return empty list (safe)
        return units

    for i, page in enumerate(pages):
        page_no = i + 1  # 1-indexed page numbers
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""

        txt = _clean_text(txt)
        if not txt:
            continue

        units.append(
            ExtractedUnit(
                doc_name=doc_name,
                source_type="pdf",
                loc_kind="page",
                loc_start=page_no,
                loc_end=page_no,
                text=txt,
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
    Extract DOCX into stable "section/page equivalent" units.

    Strategy:
      - Use Heading 1/2/3 as section boundaries.
      - If no headings, create blocks by paragraph ranges.
      - Always keep paragraph indexes for traceability.

    Output units:
      source_type="docx"
      loc_kind="section" or "paragraphs"
      loc_start/loc_end: anchor strings like:
        Section: "Leave Policy" (Paragraphs 45–82)
        Paragraphs 120–155
    """
    units: List[ExtractedUnit] = []
    doc_name = os.path.basename(docx_path)

    try:
        from docx import Document
        doc = Document(docx_path)
    except Exception:
        return units

    # Gather paragraphs with style + index
    paras: List[Dict[str, Any]] = []
    para_idx = 0  # count non-empty text paragraphs only
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

        # If heading, update current heading and continue (do not include heading text itself as content)
        if _is_heading_style(style_name):
            current_heading = txt
            continue

        para_idx += 1
        paras.append({
            "i": para_idx,               # 1-based index of content paragraphs
            "heading": current_heading,  # nearest preceding heading
            "text": txt
        })

    if not paras:
        return units

    # If headings exist, group by heading
    has_any_heading = any(p["heading"] for p in paras)

    if has_any_heading:
        # Group paragraphs by heading value
        # Some paragraphs may have heading="" (before any heading). We'll treat that as "Untitled"
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for p in paras:
            h = p["heading"] or "Untitled"
            groups.setdefault(h, []).append(p)

        # For each group, split into blocks if too large
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

    # If no headings, create blocks by paragraph range
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
    """
    Takes a list of paragraphs under one heading, emits 1+ units.
    """
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

        # flush if too large
        if char_count >= max_chars:
            text = _clean_text("\n".join(buffer))
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

    # flush remaining (if meaningful)
    if buffer:
        text = _clean_text("\n".join(buffer))
        if text and len(text) >= min_chars:
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
        elif text:
            # even if small, still include to avoid missing content
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
    """
    Emits paragraph-range blocks when no headings exist.
    """
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
            text = _clean_text("\n".join(buffer))
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
        text = _clean_text("\n".join(buffer))
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
    """
    Detect file type and extract units.
    """
    low = (path or "").lower()
    if low.endswith(".pdf"):
        return extract_pdf_units(path)
    if low.endswith(".docx"):
        return extract_docx_units(path)
    return []


def extract_units_from_files(paths: List[str]) -> List[ExtractedUnit]:
    """
    Extract units from multiple files.
    """
    all_units: List[ExtractedUnit] = []
    for p in paths:
        all_units.extend(extract_units_from_file(p))
    return all_units
