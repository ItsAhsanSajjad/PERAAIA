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
_WS_RE = re.compile(r"[ \t]+")
_NUL_RE = re.compile(r"\x00+")

# Keep unicode (Urdu) and punctuation. Only normalize whitespace.
def _clean_text(s: str) -> str:
    s = s or ""
    s = _NUL_RE.sub(" ", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # IMPORTANT: collapse horizontal whitespace, but preserve newlines (structure)
    s = _WS_RE.sub(" ", s)
    s = re.sub(r"\n{4,}", "\n\n\n", s)
    s = "\n".join([ln.strip() for ln in s.split("\n")])
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


# --- structural heuristics: detect tables/lists/headings ---
_BULLET_RE = re.compile(r"^\s*([•\-\u2022]|\d+[\)\.]|[a-zA-Z][\)\.])\s+")
# UPDATED: extractor now normalizes table spacing into " | "
_PIPE_TABLE_RE = re.compile(r"\s\|\s")  # " | " delimiter
# fallback if something still contains tabs
_TAB_TABLE_RE = re.compile(r"\t+")

_HEADING_RE = re.compile(r"^\s*(schedule|annex|annexure|appendix|chapter|section)\b", re.I)

def _looks_like_table_line(line: str) -> bool:
    if not line:
        return False
    s = line.strip()
    if len(s) < 18:
        return False
    # Strong signal: pipe-delimited columns from updated extractor
    if _PIPE_TABLE_RE.search(s):
        # require at least 2 separators to reduce false positives
        return s.count("|") >= 2
    # fallback: tabs suggest columns
    if _TAB_TABLE_RE.search(s):
        return True
    return False

def _looks_like_list_line(line: str) -> bool:
    if not line:
        return False
    return _BULLET_RE.search(line) is not None

def _looks_like_table_or_list(line: str) -> bool:
    return _looks_like_list_line(line) or _looks_like_table_line(line)

def _is_heading(line: str) -> bool:
    if not line:
        return False
    s = line.strip()
    if _HEADING_RE.search(s):
        return True
    # short ALL-CAPS headings
    letters = re.sub(r"[^A-Za-z]+", "", s)
    if 4 <= len(letters) <= 40 and letters.isupper():
        return True
    return False


def _split_into_blocks(text: str) -> List[str]:
    """
    Production chunking blocks:
    - Split on blank lines
    - Start new block on headings
    - Keep tables/lists as their own blocks and do not merge into narrative text
    - Prevent mode mixing: narrative vs list/table
    """
    t = _clean_text(text)
    if not t:
        return []

    lines = t.split("\n")
    blocks: List[str] = []
    buf: List[str] = []

    def flush():
        nonlocal buf
        if not buf:
            return
        b = _clean_text("\n".join(buf))
        if b:
            blocks.append(b)
        buf = []

    def last_is_structured() -> bool:
        if not buf:
            return False
        return _looks_like_table_or_list(buf[-1])

    for ln in lines:
        raw = (ln or "").strip()

        # blank line = boundary
        if not raw:
            flush()
            continue

        # headings start a new block
        if _is_heading(raw):
            flush()
            buf.append(raw)
            continue

        structured = _looks_like_table_or_list(raw)

        # If switching between narrative <-> structured, flush to avoid mixing
        if buf:
            prev_structured = last_is_structured()
            if structured != prev_structured:
                flush()

        buf.append(raw)

    flush()

    return blocks if blocks else [t]


def _trim_overlap_to_boundary(tail: str) -> str:
    """
    Make overlap start at a clean boundary to avoid mid-word stitching.
    We drop leading partial tokens until first whitespace/punctuation boundary.
    """
    s = (tail or "").strip()
    if not s:
        return ""

    # If it starts mid-word, remove the partial word prefix
    # Example: "ulatory Authority ..." -> remove "ulatory"
    if re.match(r"^[A-Za-z\u0600-\u06FF0-9]+", s):
        # find first boundary (space or punctuation) AFTER some chars
        m = re.search(r"[\s\.,;:\)\]\}!\?]", s)
        if m and m.start() < 20:
            s = s[m.start():].lstrip()

    # keep it reasonably short and clean
    return s.strip()


def _chunk_by_char_budget(blocks: List[str], max_chars: int, overlap_chars: int) -> List[str]:
    """
    Creates chunks up to max_chars.
    Overlap is applied as a tail snippet of previous chunk.

    Hardening:
    - Never allow overlap >= max_chars
    - Safe splitting for huge blocks
    - Cap total chunks globally
    """
    if not blocks:
        return []

    max_chars = max(500, int(max_chars or 0))
    overlap_chars = max(0, int(overlap_chars or 0))
    if overlap_chars >= max_chars:
        overlap_chars = max(0, max_chars // 5)

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

    GLOBAL_MAX_CHUNKS = 20000

    for b in blocks:
        b = (b or "").strip()
        if not b:
            continue

        # Huge block safeguard
        if len(b) > max_chars:
            flush()

            step = max(200, max_chars - overlap_chars)
            approx = int(len(b) / max_chars) + 5
            max_parts = max(80, min(3000, approx))

            start = 0
            parts_made = 0
            while start < len(b) and parts_made < max_parts:
                end = min(len(b), start + max_chars)
                part = _clean_text(b[start:end])
                if part:
                    chunks.append(part)
                    if len(chunks) >= GLOBAL_MAX_CHUNKS:
                        return chunks
                parts_made += 1
                if end >= len(b):
                    break
                start = start + step

            if start < len(b):
                tail = _clean_text(b[-max_chars:])
                if tail:
                    chunks.append(tail)
                    if len(chunks) >= GLOBAL_MAX_CHUNKS:
                        return chunks
            continue

        if size + len(b) + 2 > max_chars and buf:
            flush()

        buf.append(b)
        size += len(b) + 2

        if len(chunks) >= GLOBAL_MAX_CHUNKS:
            flush()
            return chunks

    flush()

    # Overlap: add tail of previous chunk (clean boundary)
    if overlap_chars > 0 and len(chunks) > 1:
        out: List[str] = []
        cap = max(80, min(overlap_chars, 500))  # keep overlap reasonable

        for i, c in enumerate(chunks):
            if i == 0:
                out.append(c)
                continue

            prev = chunks[i - 1]
            tail = prev[-cap:]
            tail = _trim_overlap_to_boundary(tail)

            if tail:
                stitched = _clean_text(tail + "\n\n" + c)
            else:
                stitched = _clean_text(c)

            out.append(stitched)
        return out

    return chunks


def _force_keep_chunk(ctext: str) -> bool:
    """
    Some chunks must be kept even if short, because they answer common questions.
    """
    t = (ctext or "").lower()
    if "schedule" in t or "annex" in t or "annexure" in t or "appendix" in t:
        return True
    if "punjab enforcement and regulatory authority" in t or re.search(r"\bpera\b", t):
        if len(t) < 500:
            return True
    if "chief technology officer" in t or re.search(r"\bcto\b", t):
        return True
    if "terms of reference" in t or re.search(r"\btor\b", t):
        return True
    return False


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

    Guarantees:
      - PDF units are page-scoped (never mix pages)
      - DOCX units are unit-scoped (never mix sections/ranges)

    Safety:
      - Keep short but high-value chunks (Schedule/Annex/definitions/role titles)
      - Drop short tails only when clearly low-signal
    """
    out: List[Chunk] = []

    for u in units:
        txt = _clean_text(getattr(u, "text", "") or "")
        if not txt:
            continue

        rank = getattr(u, "doc_rank", 0) or _parse_book_rank(getattr(u, "doc_name", ""))

        # If unit itself is short, keep it as one chunk
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
                    path=getattr(u, "path", None),
                )
            )
            continue

        blocks = _split_into_blocks(txt)
        chunk_texts = _chunk_by_char_budget(blocks, max_chars=max_chars, overlap_chars=overlap_chars)

        if not chunk_texts:
            chunk_texts = [txt]

        for i, ctext in enumerate(chunk_texts):
            ctext = _clean_text(ctext)
            if not ctext:
                continue

            if len(ctext) < min_chunk_chars:
                if len(chunk_texts) == 1:
                    pass
                elif _force_keep_chunk(ctext):
                    pass
                elif i == len(chunk_texts) - 1:
                    # allow last tail only if it has enough real words
                    if len(re.findall(r"[A-Za-z\u0600-\u06FF]{3,}", ctext)) >= 10:
                        pass
                    else:
                        continue
                else:
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
                    path=getattr(u, "path", None),
                )
            )

    return out
