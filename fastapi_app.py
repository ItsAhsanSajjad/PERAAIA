from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from retriever import retrieve
from answerer import answer_question

# ✅ Auto-indexer (safe blue/green builds + atomic pointer switch)
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
# ✅ Auto-indexer setup
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
    ✅ Starts background indexing:
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
    references: list


# -----------------------------
# Helpers
# -----------------------------
def _is_safe_under_assets_data(abs_path: str) -> bool:
    """
    Prevent path traversal: ensure abs_path is under DATA_DIR.
    """
    try:
        data_abs = os.path.abspath(DATA_DIR)
        file_abs = os.path.abspath(abs_path)
        return os.path.commonpath([data_abs, file_abs]) == data_abs
    except Exception:
        return False


def _to_assets_data_rel(doc_name: str) -> str:
    """
    Convert doc_name into 'assets/data/<doc_name>' relative path.
    """
    doc_name = (doc_name or "").strip()
    return os.path.join("assets", "data", doc_name).replace("\\", "/")


# -----------------------------
# ✅ Download endpoint (forces download with headers)
# -----------------------------
@app.get("/download/{filename:path}")
def download_pdf(filename: str):
    """
    Download a PDF from assets/data with Content-Disposition: attachment.
    Example:
      /download/PERA%20Special%20Allowaance%20Advice%2019-01-2026.pdf
    """
    filename = (filename or "").strip()

    # Only allow direct file names under DATA_DIR
    abs_path = os.path.join(DATA_DIR, filename).replace("\\", "/")

    if not _is_safe_under_assets_data(abs_path):
        raise HTTPException(status_code=400, detail="Invalid path.")

    if not os.path.exists(abs_path) or not os.path.isfile(abs_path):
        raise HTTPException(status_code=404, detail="File not found.")

    # Force download
    return FileResponse(
        abs_path,
        media_type="application/pdf",
        filename=os.path.basename(abs_path),
        headers={"Content-Disposition": f'attachment; filename="{os.path.basename(abs_path)}"'}
    )


# -----------------------------
# Main endpoint (HTML)
# -----------------------------
@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    retrieval = retrieve(request.message)
    result = answer_question(request.message, retrieval)

    baseurl = os.getenv("Base_URL", "https://askpera.infinitysol.agency").rstrip("/")

    html_answer = f"""
    <div style="font-family: Arial, sans-serif; line-height: 1.6; margin:0; padding:0;">
      <p style="margin:0; padding:0;">{result['answer']}</p>
    """

    references = result.get("references", []) or []
    if references:
        html_answer += """
        <hr style="margin:8px 0;" />
        <h3 style="margin:4px 0;">References</h3>
        <ol style="margin:0; padding-left:18px;">
        """

        for ref in references:
            doc = ref.get("document", "Unknown document")
            snippet = ref.get("snippet", "") or ""

            # Your answerer stores a relative path like assets/data/<file>.pdf
            # We normalize it for open link:
            path = (ref.get("path") or "").strip()
            if not path:
                path = _to_assets_data_rel(doc)

            # Ensure starts with / for web
            open_path = "/" + path.lstrip("/")

            url_hint = (ref.get("url_hint") or "").strip()

            open_href = f"{baseurl}{open_path}{url_hint}"

            # Download uses our download endpoint with filename only
            # Extract filename from path:
            filename = os.path.basename(path)
            download_href = f"{baseurl}/download/{filename}"

            html_answer += f"""
            <li style="margin-bottom:10px;">
              <div style="margin-bottom:4px;"><strong>{doc}</strong></div>
              <div style="margin-bottom:4px;">
                <a href="{open_href}" target="_blank" rel="noopener noreferrer">Open</a>
                &nbsp;|&nbsp;
                <a href="{download_href}">Download</a>
              </div>
              <p style="margin:2px 0;">{snippet}</p>
            </li>
            """

        html_answer += "</ol>"

    html_answer = html_answer.rstrip("</p>\n    </div>") + "</p></div>"
    return QueryResponse(user_id=request.user_id, answer=html_answer.strip())


# -----------------------------
# ✅ JSON endpoint (recommended for mobile)
# -----------------------------
@app.post("/ask_json", response_model=QueryResponseJSON)
def ask_question_json(request: QueryRequest):
    retrieval = retrieve(request.message)
    result = answer_question(request.message, retrieval)

    baseurl = os.getenv("Base_URL", "https://askpera.infinitysol.agency").rstrip("/")

    refs_out = []
    for ref in (result.get("references") or []):
        doc = ref.get("document", "Unknown document")
        snippet = ref.get("snippet", "") or ""

        path = (ref.get("path") or "").strip()
        if not path:
            path = _to_assets_data_rel(doc)

        url_hint = (ref.get("url_hint") or "").strip()

        open_url = f"{baseurl}/" + path.lstrip("/") + url_hint
        filename = os.path.basename(path)
        download_url = f"{baseurl}/download/{filename}"

        refs_out.append({
            "document": doc,
            "snippet": snippet,
            "path": path,
            "url_hint": url_hint,
            "open_url": open_url,
            "download_url": download_url,
            "page_start": ref.get("page_start"),
            "page_end": ref.get("page_end"),
            "loc_kind": ref.get("loc_kind"),
            "loc_start": ref.get("loc_start"),
            "loc_end": ref.get("loc_end"),
        })

    return QueryResponseJSON(
        user_id=request.user_id,
        answer=result.get("answer", ""),
        references=refs_out
    )
