from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Any, Optional

from extractors import ExtractedUnit


# -----------------------------
# Chunk structure
# -----------------------------

@dataclass
class Chunk:
    doc_name: str
    doc_rank: int
    source_type: str         # "pdf" | "docx"
    loc_kind: str            # "page" | "section" | "paragraphs"
    loc_start: Any
    loc_end: Any
    chunk_text: str
    path: Optional[str] = None


# -----------------------------
# Utilities
# -----------------------------

def _clean_text(s: str) -> str:
    s = s or ""
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _parse_book_rank(filename: str) -> int:
    """
    book1, book2, ... bookN => higher number = newer/higher priority.
    If no match => rank 0.
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    m = re.search(r"\bbook\s*([0-9]+)\b", base, flags=re.IGNORECASE)
    if not m:
        return 0
    try:
        return int(m.group(1))
    except Exception:
        return 0


def _split_into_paragraphs(text: str) -> List[str]:
    """
    Splits on blank lines first; fallback to line splits if needed.
    """
    t = _clean_text(text)
    if not t:
        return []

    parts = [p.strip() for p in re.split(r"\n\s*\n", t) if p.strip()]
    if len(parts) >= 2:
        return parts

    parts2 = [p.strip() for p in t.split("\n") if p.strip()]
    if len(parts2) >= 2:
        return parts2

    return [t]


def _chunk_by_char_budget(parts: List[str], max_chars: int, overlap_chars: int) -> List[str]:
    """
    Creates chunks up to max_chars. Adds a small overlap between chunks.
    overlap_chars is approximated overlap, not token-based.
    """
    if not parts:
        return []

    chunks: List[str] = []
    buf: List[str] = []
    size = 0

    def flush():
        nonlocal buf, size
        if not buf:
            return
        chunk = _clean_text("\n\n".join(buf))
        if chunk:
            chunks.append(chunk)
        buf = []
        size = 0

    for p in parts:
        p = p.strip()
        if not p:
            continue

        # If a single paragraph is huge, hard-split it
        if len(p) > max_chars:
            flush()
            start = 0
            while start < len(p):
                end = min(len(p), start + max_chars)
                chunks.append(_clean_text(p[start:end]))
                start = end - overlap_chars if overlap_chars > 0 else end
            continue

        if size + len(p) + 2 > max_chars and buf:
            flush()

        buf.append(p)
        size += len(p) + 2

    flush()

    # Lightweight overlap: prefix tail of previous chunk into next
    if overlap_chars > 0 and len(chunks) > 1:
        out: List[str] = []
        for i, c in enumerate(chunks):
            if i == 0:
                out.append(c)
                continue
            prev = chunks[i - 1]
            overlap = prev[-overlap_chars:]
            out.append(_clean_text(overlap + "\n\n" + c))
        return out

    return chunks


# -----------------------------
# Main chunking API
# -----------------------------

def chunk_units(
    units: List[ExtractedUnit],
    max_chars: int = 4500,
    overlap_chars: int = 350,
    min_chunk_chars: int = 200
) -> List[Chunk]:
    """
    Converts extracted units into chunks while preserving traceability.

    Rules enforced:
      - PDF: one unit per page, chunk stays within page (never mix pages)
      - DOCX: one unit per section/paragraph-range, chunk stays within that unit

    IMPORTANT FIX:
      - Do NOT discard short-but-meaningful units (e.g., short notifications).
        If the whole unit is shorter than min_chunk_chars, keep it as ONE chunk.
    """
    out: List[Chunk] = []

    for u in units:
        txt = _clean_text(u.text)
        if not txt:
            continue

        # Determine rank if not pre-filled
        rank = u.doc_rank if getattr(u, "doc_rank", 0) else _parse_book_rank(u.doc_name)

        # ✅ If unit text itself is short, keep it as a single chunk (do not drop)
        if len(txt) < min_chunk_chars:
            out.append(
                Chunk(
                    doc_name=u.doc_name,
                    doc_rank=rank,
                    source_type=u.source_type,
                    loc_kind=u.loc_kind,
                    loc_start=u.loc_start,
                    loc_end=u.loc_end,
                    chunk_text=txt,
                    path=u.path
                )
            )
            continue

        parts = _split_into_paragraphs(txt)
        chunk_texts = _chunk_by_char_budget(parts, max_chars=max_chars, overlap_chars=overlap_chars)

        for ctext in chunk_texts:
            ctext = _clean_text(ctext)
            if not ctext:
                continue

            # ✅ Keep short chunks if they are the only chunk produced from a unit
            # (prevents dropping small tail chunks)
            if len(ctext) < min_chunk_chars and len(chunk_texts) > 1:
                continue

            out.append(
                Chunk(
                    doc_name=u.doc_name,
                    doc_rank=rank,
                    source_type=u.source_type,
                    loc_kind=u.loc_kind,
                    loc_start=u.loc_start,
                    loc_end=u.loc_end,
                    chunk_text=ctext,
                    path=u.path
                )
            )

    return out
