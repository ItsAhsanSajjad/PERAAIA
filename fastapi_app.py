from __future__ import annotations

import os
from urllib.parse import quote
from typing import List, Dict, Any, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from retriever import retrieve
from answerer import answer_question

# Auto-indexer (safe blue/green builds + atomic pointer switch)
from index_manager import SafeAutoIndexer, IndexManagerConfig


app = FastAPI()

# -----------------------------
# Static PDFs
# -----------------------------
DATA_DIR = os.getenv("DATA_DIR", os.path.join("assets", "data")).replace("\\", "/")
app.mount(
    "/assets/data",
    StaticFiles(directory=DATA_DIR),
    name="data"
)

# -----------------------------
#  Auto-indexer setup
# -----------------------------
INDEXES_ROOT = os.getenv("INDEXES_ROOT", os.path.join("assets", "indexes")).replace("\\", "/")
ACTIVE_POINTER_PATH = os.getenv("INDEX_POINTER_PATH", os.path.join("assets", "indexes", "ACTIVE.json")).replace("\\", "/")

POLL_SECONDS = int(os.getenv("INDEX_POLL_SECONDS", "30"))
KEEP_LAST_N = int(os.getenv("INDEX_KEEP_LAST_N", "3"))

CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "4500"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "350"))

indexer = SafeAutoIndexer(
    IndexManagerConfig(
        data_dir=DATA_DIR,
        indexes_root=INDEXES_ROOT,
        active_pointer_path=ACTIVE_POINTER_PATH,
        poll_seconds=POLL_SECONDS,
        keep_last_n=KEEP_LAST_N,
        chunk_max_chars=CHUNK_MAX_CHARS,
        chunk_overlap_chars=CHUNK_OVERLAP_CHARS,
    )
)


@app.on_event("startup")
def _startup_indexer():
    """
     Starts background indexing:
    - bootstraps an initial index if ACTIVE pointer missing
    - then polls for new/changed PDFs and rebuilds safely
    """
    indexer.start_background()


# -----------------------------
# API models
# -----------------------------
class QueryRequest(BaseModel):
    user_id: str
    message: str


class QueryResponse(BaseModel):
    user_id: str
    answer: str


class QueryResponseJSON(BaseModel):
    user_id: str
    answer: str
    references: List[Dict[str, Any]]


# -----------------------------
# Helpers
# -----------------------------
def _is_safe_under_assets_data(abs_path: str) -> bool:
    """Prevent path traversal: ensure abs_path is under DATA_DIR."""
    try:
        data_abs = os.path.abspath(DATA_DIR)
        file_abs = os.path.abspath(abs_path)
        return os.path.commonpath([data_abs, file_abs]) == data_abs
    except Exception:
        return False


def _extract_filename(ref_path: str, doc_name: str) -> str:
    """
    Given a reference path (maybe '/assets/data/x.pdf' or 'assets/data/x.pdf' or ''),
    return a safe filename to use in /download/{filename}.
    """
    p = (ref_path or "").strip().replace("\\", "/")

    if p.startswith("/assets/data/") or p.startswith("assets/data/"):
        return os.path.basename(p)

    if p.lower().endswith(".pdf"):
        return os.path.basename(p)

    # Fallback to doc_name (which should already be the filename)
    return os.path.basename((doc_name or "").strip())


def _dedupe_references(refs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate repeated references so the same file doesn't appear multiple times
    just because multiple chunks were used.
    """
    seen = set()
    out: List[Dict[str, Any]] = []
    for r in refs or []:
        if not isinstance(r, dict):
            continue
        doc = (r.get("document") or "Unknown document").strip()
        filename = _extract_filename(r.get("path", ""), doc)
        key = (doc, filename)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _build_download_url(baseurl: str, filename: str) -> str:
    """
    IMPORTANT: URL-encode filenames so spaces/unicode/() don't cause 403/404.
    """
    safe = quote(filename)
    return f"{baseurl}/download/{safe}"


# -----------------------------
#  Download endpoint (forces download)
# -----------------------------
@app.get("/download/{filename:path}")
def download_pdf(filename: str):
    """
    Download a PDF from assets/data with Content-Disposition: attachment.
    Example:
      /download/PERA%20Special%20Allowaance%20Advice%2019-01-2026.pdf
    """
    filename = (filename or "").strip()
    if not filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    abs_path = os.path.join(DATA_DIR, filename).replace("\\", "/")

    if not _is_safe_under_assets_data(abs_path):
        raise HTTPException(status_code=400, detail="Invalid path.")

    if not os.path.exists(abs_path) or not os.path.isfile(abs_path):
        raise HTTPException(status_code=404, detail="File not found.")

    return FileResponse(
        abs_path,
        media_type="application/pdf",
        filename=os.path.basename(abs_path),
        headers={"Content-Disposition": f'attachment; filename="{os.path.basename(abs_path)}"'},
    )


# -----------------------------
# Main endpoint (HTML)  ONLY DOWNLOAD LINKS
# -----------------------------
@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    retrieval = retrieve(request.message)
    result = answer_question(request.message, retrieval)

    baseurl = os.getenv("Base_URL", "https://askpera.infinitysol.agency").rstrip("/")

    answer_text = result.get("answer", "") or ""
    references = _dedupe_references(result.get("references", []) or [])

    parts: List[str] = []
    parts.append('<div style="font-family: Arial, sans-serif; line-height: 1.6; margin:0; padding:0;">')
    parts.append(f'<p style="margin:0; padding:0;">{answer_text}</p>')

    if references:
        parts.append('<hr style="margin:8px 0;" />')
        parts.append('<h3 style="margin:4px 0;">References</h3>')
        parts.append('<ol style="margin:0; padding-left:18px;">')

        for ref in references:
            doc = (ref.get("document") or "Unknown document").strip()
            snippet = (ref.get("snippet") or "").strip()

            filename = _extract_filename(ref.get("path", ""), doc)
            download_href = _build_download_url(baseurl, filename)

            parts.append(
                f"""
                <li style="margin-bottom:6px;">
                  <a href="{download_href}" download>{doc}</a>
                  <p style="margin:2px 0;">{snippet}</p>
                </li>
                """.strip()
            )

        parts.append("</ol>")

    parts.append("</div>")

    return QueryResponse(user_id=request.user_id, answer="\n".join(parts).strip())


# -----------------------------
# JSON endpoint (for mobile) ONLY DOWNLOAD LINKS
# -----------------------------
@app.post("/ask_json", response_model=QueryResponseJSON)
def ask_question_json(request: QueryRequest):
    retrieval = retrieve(request.message)
    result = answer_question(request.message, retrieval)

    baseurl = os.getenv("Base_URL", "https://askpera.infinitysol.agency").rstrip("/")

    references = _dedupe_references(result.get("references", []) or [])

    refs_out: List[Dict[str, Any]] = []
    for ref in references:
        doc = (ref.get("document") or "Unknown document").strip()
        snippet = (ref.get("snippet") or "").strip()

        filename = _extract_filename(ref.get("path", ""), doc)
        download_url = _build_download_url(baseurl, filename)

        refs_out.append({
            "document": doc,
            "snippet": snippet,
            "download_url": download_url,

            # keep optional metadata for future UI improvements
            "page_start": ref.get("page_start"),
            "page_end": ref.get("page_end"),
            "loc_kind": ref.get("loc_kind"),
            "loc_start": ref.get("loc_start"),
            "loc_end": ref.get("loc_end"),
        })

    return QueryResponseJSON(
        user_id=request.user_id,
        answer=result.get("answer", "") or "",
        references=refs_out
    )
