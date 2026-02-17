from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Any, Optional, Tuple

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
# Config / Debug
# -----------------------------
DEBUG_CHUNKER = os.getenv("CHUNKER_DEBUG", "0").strip() != "0"


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
    # collapse horizontal whitespace, preserve newlines (structure)
    s = _WS_RE.sub(" ", s)
    # cap runaway blank lines but preserve structure
    s = re.sub(r"\n{5,}", "\n\n\n\n", s)
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


# -----------------------------
# Structural heuristics: tables/lists/headings
# -----------------------------
_BULLET_RE = re.compile(r"^\s*(?:[•\-\u2022]|\d+[\)\.]|[a-zA-Z][\)\.])\s+")
_PIPE_TABLE_RE = re.compile(r"\s\|\s")   # " | " delimiter
_TAB_TABLE_RE = re.compile(r"\t+")

def _looks_like_table_line(line: str) -> bool:
    if not line:
        return False
    s = line.strip()
    if len(s) < 18:
        return False
    if _PIPE_TABLE_RE.search(s):
        return s.count("|") >= 2
    if _TAB_TABLE_RE.search(s):
        return True
    return False

def _looks_like_list_line(line: str) -> bool:
    if not line:
        return False
    return _BULLET_RE.search(line) is not None

def _looks_like_table_or_list(line: str) -> bool:
    return _looks_like_list_line(line) or _looks_like_table_line(line)


# -----------------------------
# Heading detection (hardened)
# -----------------------------
# Captures: Schedule I / Schedule-I / Schedule 1, Annexure A, Appendix B,
# Chapter 2, Section 12, Rule 3, Regulation 4, Part II, Table 1
_HEADING_RE = re.compile(
    r"^\s*(?:"
    r"(schedule|annex(?:ure)?|appendix|chapter|section|rule|regulation|part|table)\s*"
    r"([\-–—]?\s*[A-Za-z0-9IVXLC]+)?"
    r")\b"
    r".*$",
    re.IGNORECASE
)

# All-caps short headings (e.g., "POWERS", "DUTIES", "ELIGIBILITY")
def _is_all_caps_heading(s: str) -> bool:
    s = (s or "").strip()
    if len(s) < 4 or len(s) > 60:
        return False
    letters = re.sub(r"[^A-Za-z]+", "", s)
    if not (4 <= len(letters) <= 40):
        return False
    return letters.isupper()

def _is_heading(line: str) -> bool:
    if not line:
        return False
    s = line.strip()
    if _HEADING_RE.match(s):
        return True
    if _is_all_caps_heading(s):
        return True
    return False


# -----------------------------
# Role heading detection (safer)
# -----------------------------
# Avoid false positives: don’t treat long sentences as roles.
_MAX_ROLE_LEN = 80

# Exclusion patterns (lines that look like metadata fields, not headings)
_ROLE_EXCLUDE_RE = re.compile(
    r"^\s*(report\s*to|reporting\s*to|reports\s*to|department|location|grade|scale|pay|"
    r"job\s*summary|summary|objective|purpose|education|qualification|experience|"
    r"responsibilit|duties|skills|competenc|note|remarks)\b",
    re.IGNORECASE
)

# "Position Title:" "Job Title:" "Role:"
_ROLE_PREFIX_RE = re.compile(r"^\s*(position\s*title|job\s*title|role)\s*[:\-]+\s*(.*)$", re.IGNORECASE)

# Role titles patterns (expandable)
_ROLE_HEADING_RE = re.compile(
    r"^\s*(?:"
    r"chief\s+technology\s+officer|"
    r"chief\s+executive\s+officer|"
    r"director\s+general|"
    r"additional\s+director\s+general|"
    r"deputy\s+director(?:\s+general)?|"
    r"assistant\s+director|"
    r"director\s+\([^)]+\)|"
    r"manager\s*\([^)]+\)|"
    r"project\s+manager|"
    r"hr\s+manager|"
    r"it\s+manager|"
    r"enforcement\s+officer|"
    r"sub[- ]?divisional\s+enforcement\s+officer|"
    r"investigation\s+officer|"
    r"inspection\s+officer|"
    r"system\s+support\s+officer|"
    r"database\s+administrator|"
    r"system\s+administrator|"
    r"network\s+administrator|"
    r"software\s+developer|"
    r"android\s+developer|"
    r"[A-Za-z][A-Za-z\s&]+\s+(officer|developer|manager|director|engineer|specialist|administrator|coordinator)"
    r")\s*$",
    re.IGNORECASE
)

def _is_role_heading(line: str) -> Optional[str]:
    """
    Detect if a line is a role heading.
    Returns the role title if detected, None otherwise.
    """
    if not line:
        return None
    s = line.strip()

    if not s or len(s) > _MAX_ROLE_LEN:
        return None

    # Exclude common metadata / section labels
    if _ROLE_EXCLUDE_RE.match(s):
        return None

    # Handle "Position Title: - Role"
    m = _ROLE_PREFIX_RE.match(s)
    if m:
        s = (m.group(2) or "").strip()
        s = re.sub(r"^[-•\:\s]+", "", s).strip()
        if not s:
            return None

    # Ensure it looks like a title (not a sentence)
    # Too many punctuation marks -> likely not a clean title
    if sum(1 for ch in s if ch in ".;,:") >= 2:
        return None

    if _ROLE_HEADING_RE.match(s):
        return s

    return None


# -----------------------------
# Block splitting with role context
# -----------------------------
def _split_into_blocks_with_context(text: str) -> List[Tuple[Optional[str], str]]:
    """
    Splits into blocks while tracking role headings.
    Returns list of (role_context, block_text) tuples.

    Rules:
      - Blank lines split blocks
      - Headings start new blocks
      - Do not mix narrative with tables/lists
      - Role headings update current role context
      - IMPORTANT: role heading lines are NOT included as content blocks
        (we inject role context separately to avoid duplication)
    """
    t = _clean_text(text)
    if not t:
        return []

    lines = t.split("\n")
    blocks: List[Tuple[Optional[str], str]] = []
    buf: List[str] = []
    current_role: Optional[str] = None

    def flush():
        nonlocal buf
        if not buf:
            return
        b = _clean_text("\n".join(buf))
        if b:
            blocks.append((current_role, b))
        buf = []

    def last_is_structured() -> bool:
        if not buf:
            return False
        return _looks_like_table_or_list(buf[-1])

    for ln in lines:
        raw = (ln or "").strip()

        if not raw:
            flush()
            continue

        role_match = _is_role_heading(raw)
        if role_match:
            if DEBUG_CHUNKER:
                print(f"[Chunker] Role heading detected: {role_match}")
            flush()
            current_role = role_match
            # DO NOT add the role heading itself to buf (avoid duplication)
            continue

        if _is_heading(raw):
            flush()
            buf.append(raw)
            continue

        structured = _looks_like_table_or_list(raw)
        if buf:
            prev_structured = last_is_structured()
            if structured != prev_structured:
                flush()

        buf.append(raw)

    flush()
    return blocks


def _trim_overlap_to_boundary(tail: str) -> str:
    """
    Make overlap start at a clean boundary to avoid mid-word stitching.
    """
    s = (tail or "").strip()
    if not s:
        return ""

    # Remove leading partial token if overlap begins mid-token
    m = re.search(r"[\s\.,;:\)\]\}!\?]", s)
    if m and m.start() < 20:
        s = s[m.start():].lstrip()

    return s.strip()


def _chunk_by_char_budget(blocks: List[str], max_chars: int, overlap_chars: int) -> List[str]:
    """
    Creates chunks up to max_chars.
    Overlap is applied as a tail snippet of previous chunk.

    Hardening:
      - overlap < max_chars
      - safe splitting for huge blocks
      - global chunk cap
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
            start = 0
            parts_made = 0
            max_parts = max(80, min(3000, int(len(b) / max_chars) + 10))

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
                start += step
            continue

        if size + len(b) + 2 > max_chars and buf:
            flush()

        buf.append(b)
        size += len(b) + 2

        if len(chunks) >= GLOBAL_MAX_CHUNKS:
            flush()
            return chunks

    flush()

    # Overlap stitching
    if overlap_chars > 0 and len(chunks) > 1:
        out: List[str] = []
        cap = max(80, min(overlap_chars, 500))

        for i, c in enumerate(chunks):
            if i == 0:
                out.append(c)
                continue

            prev = chunks[i - 1]
            tail = _trim_overlap_to_boundary(prev[-cap:])
            if tail:
                out.append(_clean_text(tail + "\n\n" + c))
            else:
                out.append(_clean_text(c))
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

        # Split into (role_context, block_text)
        blocks_with_ctx = _split_into_blocks_with_context(txt)
        if not blocks_with_ctx:
            continue

        # Group blocks by role context to prevent role bleeding
        grouped: List[Tuple[Optional[str], List[str]]] = []
        current_ctx = blocks_with_ctx[0][0]
        current_texts: List[str] = []

        for ctx, btxt in blocks_with_ctx:
            if ctx != current_ctx:
                if current_texts:
                    grouped.append((current_ctx, current_texts))
                current_ctx = ctx
                current_texts = []
            current_texts.append(btxt)

        if current_texts:
            grouped.append((current_ctx, current_texts))

        # Chunk each group independently
        for ctx, texts in grouped:
            chunk_texts = _chunk_by_char_budget(texts, max_chars=max_chars, overlap_chars=overlap_chars)
            if not chunk_texts and len(texts) == 1:
                chunk_texts = texts

            for i, ctext in enumerate(chunk_texts):
                ctext = _clean_text(ctext)
                if not ctext:
                    continue

                # short chunk handling
                if len(ctext) < min_chunk_chars:
                    if len(chunk_texts) == 1:
                        pass
                    elif _force_keep_chunk(ctext):
                        pass
                    elif i == len(chunk_texts) - 1:
                        # keep last short tail only if meaningful
                        if len(re.findall(r"[A-Za-z\u0600-\u06FF]{3,}", ctext)) >= 10:
                            pass
                        else:
                            continue
                    else:
                        continue

                # Role context injection (only once, avoid duplication)
                final_text = ctext
                if ctx and f"[role:" not in ctext.lower():
                    final_text = f"[Role: {ctx}]\n{ctext}"

                out.append(
                    Chunk(
                        doc_name=u.doc_name,
                        doc_rank=rank,
                        source_type=u.source_type,
                        loc_kind=u.loc_kind,
                        loc_start=u.loc_start,
                        loc_end=u.loc_end,
                        chunk_text=final_text,
                        path=getattr(u, "path", None),
                    )
                )

    return out
