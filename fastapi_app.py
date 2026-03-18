from __future__ import annotations

import os
import uuid
from urllib.parse import quote
from typing import List, Dict, Any, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

from retriever import retrieve, rewrite_contextual_query
from answerer import answer_question
from log_config import setup_logging, get_logger, request_id_var
from openai_clients import has_api_key
from session_store import get_session_store, SessionTurn
from audit_trail import log_audit_entry
from context_state import anchor_query, extract_subject, extract_evidence_metadata
from smalltalk_intent import decide_smalltalk

# Auto-indexer (safe blue/green builds + atomic pointer switch)
from index_manager import SafeAutoIndexer, IndexManagerConfig

log = get_logger("pera.api")


# ============================================================
# FastAPI app
# ============================================================
app = FastAPI(title="PERA AI Backend", version="2.1.0")

# Keep your permissive CORS behavior
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Allow iframes (fix for browser blocking)
@app.middleware("http")
async def add_iframe_headers(request, call_next):
    response = await call_next(request)
    # X-Frame-Options is obsolete and ALLOWALL is invalid. 
    # Use CSP frame-ancestors instead.
    if "X-Frame-Options" in response.headers:
        del response.headers["X-Frame-Options"]
    response.headers["Content-Security-Policy"] = "frame-ancestors *"
    return response


# ============================================================
# Static PDFs setup (serves real files at /assets/data/<filename>)
# ============================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(_SCRIPT_DIR, "assets", "data"))
app.mount("/assets/data", StaticFiles(directory=DATA_DIR), name="data")


# ============================================================
# Auto-indexer setup
# ============================================================
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

# ============================================================
# Request ID Middleware
# ============================================================
class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Accept forwarded request ID or generate a new one
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = rid
        request_id_var.set(rid)
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response

app.add_middleware(RequestIDMiddleware)


@app.on_event("startup")
def _startup():
    """Initialize logging and start background indexer."""
    setup_logging()
    log.info("PERA AI Backend starting up")
    indexer.start_background()


# ============================================================
# API models
# ============================================================
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


# ============================================================
# Helpers
# ============================================================
def _base_url() -> str:
    # accept both spellings
    return os.getenv("BASE_URL", os.getenv("Base_URL", "https://askpera.infinitysol.agency")).rstrip("/")


def _is_safe_under_assets_data(abs_path: str) -> bool:
    """Prevent path traversal to ensure the file is under the assets/data directory."""
    try:
        data_abs = os.path.abspath(DATA_DIR)
        file_abs = os.path.abspath(abs_path)
        return os.path.commonpath([data_abs, file_abs]) == data_abs
    except Exception:
        return False


def _extract_filename(ref_path: str, doc_name: str) -> str:
    """
    Extract filename from reference path/doc name.
    Ensures we only ever return a filename (no directories).
    """
    p = (ref_path or "").strip().replace("\\", "/")
    if p.startswith("/assets/data/") or p.startswith("assets/data/"):
        return os.path.basename(p)
    if p.lower().endswith(".pdf") or p.lower().endswith(".docx"):
        return os.path.basename(p)
    return os.path.basename((doc_name or "").strip())


def _safe_join_base_and_path(base_url: str, path: str) -> str:
    if not path:
        return ""
    p = path.strip().replace("\\", "/")
    if p.startswith("http://") or p.startswith("https://"):
        return p
    if not p.startswith("/"):
        p = "/" + p
    return f"{base_url}{p}"


def _build_assets_data_url(baseurl: str, filename: str) -> str:
    """Real static file path: https://<host>/assets/data/<filename>"""
    safe = quote(filename)
    return f"{baseurl}/assets/data/{safe}"


def _build_forced_download_url(baseurl: str, filename: str) -> str:
    """Optional forced-download endpoint (Content-Disposition: attachment)."""
    safe = quote(filename)
    return f"{baseurl}/download/{safe}"


def _compress_int_ranges(nums: List[int]) -> str:
    if not nums:
        return ""
    nums = sorted(set(int(x) for x in nums if isinstance(x, int) or str(x).isdigit()))
    if not nums:
        return ""
    out: List[str] = []
    start = prev = nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
        else:
            out.append(str(start) if start == prev else f"{start}–{prev}")
            start = prev = n
    out.append(str(start) if start == prev else f"{start}–{prev}")
    return ", ".join(out)


def _extract_pages(ref: Dict[str, Any]) -> List[int]:
    pages: List[int] = []
    ps = ref.get("page_start")
    pe = ref.get("page_end")
    try:
        if ps is not None:
            a = int(ps)
            b = int(pe) if pe is not None else a
            for p in range(min(a, b), max(a, b) + 1):
                pages.append(p)
    except Exception:
        pass
    return pages


def _extract_locs(ref: Dict[str, Any]) -> List[str]:
    locs: List[str] = []
    # include loc_start (and loc if present)
    for k in ("loc", "loc_start"):
        v = ref.get(k)
        if v is not None and str(v).strip():
            locs.append(str(v).strip())
    return sorted(set(locs))


def _apply_page_anchor_if_missing(open_url: str, pages: List[int]) -> str:
    if not open_url:
        return open_url
    if "#" in open_url:
        return open_url
    if pages:
        p = min(int(x) for x in pages if isinstance(x, int) or str(x).isdigit())
        return f"{open_url}#page={p}"
    return open_url


def _build_open_url_like_streamlit(baseurl: str, ref: Dict[str, Any]) -> str:
    """
    Match Streamlit behavior:
    - Prefer answerer-produced open_url (already includes #page= when available)
    - Else build base + ref.path + ref.url_hint
    - Else fallback to /assets/data/<filename>
    - Ensure #page= is applied if we have pages and no anchor
    """
    # 1) Prefer answerer open_url
    u = (ref.get("open_url") or "").strip()
    if u:
        pages = _extract_pages(ref)
        return _apply_page_anchor_if_missing(u, pages)

    # 2) base + path + url_hint
    path = (ref.get("path") or "").strip()
    url_hint = (ref.get("url_hint") or "").strip()
    if path:
        built = _safe_join_base_and_path(baseurl, path)
        u2 = f"{built}{url_hint}"
        pages = _extract_pages(ref)
        return _apply_page_anchor_if_missing(u2, pages)

    # 3) fallback to /assets/data/<filename>
    doc = (ref.get("document") or "Unknown document").strip()
    filename = _extract_filename("", doc)
    u3 = _build_assets_data_url(baseurl, filename)
    pages = _extract_pages(ref)
    return _apply_page_anchor_if_missing(u3, pages)


def _group_references_like_streamlit(refs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Group by (document, path) and aggregate pages/locs so output matches Streamlit style:
      Document
      Pages: ...
      Sections / Paragraphs: ...
    """
    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for r in refs or []:
        if not isinstance(r, dict):
            continue

        doc = (r.get("document") or r.get("doc_name") or "Unknown document").strip()
        path = (r.get("path") or r.get("public_path") or "").strip()

        # if no path provided, synthesize /assets/data/<filename>
        if not path:
            filename = _extract_filename(r.get("path", ""), doc)
            path = f"/assets/data/{filename}"

        key = (doc, path)
        g = grouped.get(key)
        if not g:
            g = {
                "document": doc,
                "path": path,
                "open_url": (r.get("open_url") or "").strip(),
                "url_hint": (r.get("url_hint") or "").strip(),
                "pages": set(),
                "locs": set(),
                "snippet": (r.get("snippet") or "").strip(),
            }
            grouped[key] = g

        # keep first snippet if already set; otherwise fill
        if not g.get("snippet") and (r.get("snippet") or "").strip():
            g["snippet"] = (r.get("snippet") or "").strip()

        # accumulate pages
        for p in _extract_pages(r):
            g["pages"].add(int(p))

        # accumulate locs
        for loc in _extract_locs(r):
            g["locs"].add(loc)

        # keep open_url/url_hint if missing
        if not g.get("open_url") and (r.get("open_url") or "").strip():
            g["open_url"] = (r.get("open_url") or "").strip()
        if not g.get("url_hint") and (r.get("url_hint") or "").strip():
            g["url_hint"] = (r.get("url_hint") or "").strip()

    out: List[Dict[str, Any]] = []
    for g in grouped.values():
        out.append(
            {
                "document": g["document"],
                "path": g["path"],
                "open_url": g.get("open_url", ""),
                "url_hint": g.get("url_hint", ""),
                "pages": sorted(list(g["pages"])),
                "locs": sorted(list(g["locs"])),
                "snippet": g.get("snippet", ""),
            }
        )

    out.sort(key=lambda x: x.get("document", ""))
    return out


# ============================================================
# Download endpoint (kept) - forces download
# ============================================================
@app.get("/download/{filename:path}")
def download_pdf(filename: str):
    """
    Forced download from assets/data with Content-Disposition: attachment.
    NOTE: References default to open_url (/assets/data...) for page anchors.
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


# ============================================================
# Main endpoint (HTML)
# FIX: Open links should jump to the correct page (Streamlit-like)
# ============================================================
@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    retrieval = retrieve(request.message)
    result = answer_question(request.message, retrieval)

    baseurl = _base_url()
    answer_text = result.get("answer", "") or ""

    # Group refs like Streamlit (aggregate pages/locs)
    raw_refs = result.get("references", []) or []
    grouped = _group_references_like_streamlit(raw_refs)

    parts: List[str] = []
    parts.append('<div style="font-family: Arial, sans-serif; line-height: 1.6; margin:0; padding:0;">')
    parts.append(f'<p style="margin:0; padding:0;">{answer_text}</p>')

    if grouped:
        parts.append('<hr style="margin:8px 0;" />')
        parts.append('<h3 style="margin:4px 0;">References</h3>')
        parts.append('<ol style="margin:0; padding-left:18px;">')

        for g in grouped:
            doc = (g.get("document") or "Unknown document").strip()
            snippet = (g.get("snippet") or "").strip()
            pages: List[int] = g.get("pages") or []
            locs: List[str] = g.get("locs") or []

            # Build Streamlit-like open URL: prefer open_url/path/url_hint and ensure #page=...
            open_url = _build_open_url_like_streamlit(
                baseurl,
                {
                    "document": doc,
                    "path": g.get("path"),
                    "open_url": g.get("open_url"),
                    "url_hint": g.get("url_hint"),
                    # Provide page_start/page_end from aggregated pages (for anchor)
                    "page_start": min(pages) if pages else None,
                    "page_end": max(pages) if pages else None,
                    "loc_start": (locs[0] if locs else None),
                },
            )

            meta_lines: List[str] = []
            if pages:
                meta_lines.append(f"Pages: {_compress_int_ranges(pages)}")
            if locs:
                joined = "; ".join(locs[:6])
                if len(locs) > 6:
                    joined += f"; +{len(locs)-6} more"
                meta_lines.append(f"Sections / Paragraphs: {joined}")

            meta_html = ""
            if meta_lines:
                meta_html = "<br/>".join(meta_lines)

            parts.append(
                f"""
                <li style="margin-bottom:10px;">
                  <a href="{open_url}" target="_blank" rel="noopener noreferrer">{doc}</a>
                  {("<div style='margin-top:3px; color:#6b7280; font-size:13px;'>" + meta_html + "</div>") if meta_html else ""}
                  {("<p style='margin:4px 0 0 0;'>" + snippet + "</p>") if snippet else ""}
                </li>
                """.strip()
            )

        parts.append("</ol>")

    parts.append("</div>")
    return QueryResponse(user_id=request.user_id, answer="\n".join(parts).strip())


# ============================================================
# JSON endpoint (for mobile / frontend)
# FIX: Provide Streamlit-like reference object (open_url with #page)
# ============================================================
@app.post("/ask_json", response_model=QueryResponseJSON)
def ask_question_json(request: QueryRequest):
    retrieval = retrieve(request.message)
    result = answer_question(request.message, retrieval)

    baseurl = _base_url()

    raw_refs = result.get("references", []) or []
    grouped = _group_references_like_streamlit(raw_refs)

    refs_out: List[Dict[str, Any]] = []
    for g in grouped:
        doc = (g.get("document") or "Unknown document").strip()
        pages: List[int] = g.get("pages") or []
        locs: List[str] = g.get("locs") or []

        # Use the same open_url logic as Streamlit (page anchor)
        open_url = _build_open_url_like_streamlit(
            baseurl,
            {
                "document": doc,
                "path": g.get("path"),
                "open_url": g.get("open_url"),
                "url_hint": g.get("url_hint"),
                "page_start": min(pages) if pages else None,
                "page_end": max(pages) if pages else None,
                "loc_start": (locs[0] if locs else None),
            },
        )

        # Optional forced-download URL (use filename)
        filename = _extract_filename(g.get("path", ""), doc)
        download_url = _build_forced_download_url(baseurl, filename)

        refs_out.append(
            {
                "document": doc,
                "open_url": open_url,
                "download_url": download_url,
                "path": g.get("path"),
                "pages": pages,     # aggregated
                "locs": locs,       # aggregated
                "snippet": (g.get("snippet") or "").strip(),
            }
        )

    return QueryResponseJSON(
        user_id=request.user_id,
        answer=result.get("answer", "") or "",
        references=refs_out,
    )


# ============================================================
# Simple Chat API for Next.js frontend
# ============================================================
class SimpleChatRequest(BaseModel):
    question: str
    conversation_history: Optional[List[Dict[str, Any]]] = None
    session_id: Optional[str] = None  # Part 3: server-side session


class SimpleChatResponse(BaseModel):
    answer: str
    decision: str
    references: List[Dict[str, Any]]
    session_id: Optional[str] = None  # returned so client can persist
    grounding: Optional[Dict[str, Any]] = None  # Part 2 additive


@app.post("/api/ask", response_model=SimpleChatResponse)
def simple_ask(req: Request, request: SimpleChatRequest):
    """Chat endpoint with session tracking, smalltalk bypass, entity anchoring, and audit trail."""
    question = request.question.strip()
    rid = getattr(req.state, "request_id", "")

    # ── 0. Session ────────────────────────────────────────────
    store = get_session_store()
    session = store.get_or_create(request.session_id)
    sid = session.session_id

    # ── 1. Smalltalk bypass ───────────────────────────────────
    st = decide_smalltalk(question)
    if st and st.is_greeting_only:
        log.info("Smalltalk bypass: '%s' -> deterministic response", question[:40])
        log_audit_entry(
            request_id=rid, session_id=sid, question=question,
            decision="smalltalk", answer_text=st.response,
            is_smalltalk=True,
        )
        return SimpleChatResponse(
            answer=st.response,
            decision="smalltalk",
            references=[],
            session_id=sid,
        )

    # If greeting + real question, use remainingquestion with ack prefix
    ack_prefix = ""
    if st and not st.is_greeting_only and st.remaining_question:
        ack_prefix = st.ack
        question = st.remaining_question
        log.info("Greeting stripped: ack='%s', question='%s'", ack_prefix, question[:60])

    # ── 2. Extract last Q/A from client history FIRST ───────────
    # (needed for entity anchoring when session_id is not sent)
    last_question_client = None
    last_answer_client = None
    if request.conversation_history:
        for msg in reversed(request.conversation_history):
            role = msg.get("role", "")
            content = (msg.get("content") or "").strip()
            if role == "assistant" and last_answer_client is None:
                last_answer_client = content
            elif role == "user" and last_question_client is None:
                last_question_client = content
            if last_question_client and last_answer_client:
                break

    # Build fallback subject from client history if session is empty
    fallback_subject = ""
    if not session.last_subject and last_answer_client:
        fallback_subject = extract_subject(last_answer_client)
        if fallback_subject:
            log.info("Using client-history subject fallback: '%s'", fallback_subject[:60])

    # ── 3. Entity anchoring (session + client fallback) ────────
    anchored_q, subject_used, was_anchored = anchor_query(
        question,
        last_subject=session.last_subject or fallback_subject,
        last_question=session.last_question or last_question_client or "",
        last_answer=session.last_answer or last_answer_client or "",
    )

    # Prefer server-side session for last Q/A if available
    lq = session.last_question or last_question_client or ""
    la = session.last_answer or last_answer_client or ""

    # ── 4. Rewrite query (handles Urdu, abbreviations, follow-ups) ──
    query_for_retrieval = rewrite_contextual_query(anchored_q, lq, la)
    rewrite_diff = ""
    if query_for_retrieval != question:
        rewrite_diff = f"{question} -> {query_for_retrieval}"
    log.info("Query: '%s' -> Anchored: '%s' -> Rewritten: '%s'",
             question[:60], anchored_q[:60], query_for_retrieval[:60])

    # ── 5. Retrieve + Answer ──────────────────────────────────
    retrieval = retrieve(query_for_retrieval)
    # Only pass conversation_history for genuine follow-ups to prevent
    # prior wrong answers from contaminating standalone questions
    hist_for_llm = request.conversation_history if was_anchored else None
    # Use anchored_q (with resolved pronouns) for the LLM so position
    # filtering sees the actual role name (e.g. "salary of manager development"
    # instead of "salary of this?")
    question_for_llm = anchored_q if was_anchored else question
    result = answer_question(
        question_for_llm,
        retrieval,
        conversation_history=hist_for_llm,
    )

    # ── 6. Extract metadata for session + audit ───────────────
    evidence_ids, doc_names = extract_evidence_metadata(retrieval)
    answer_text = result.get("answer", "")
    decision = result.get("decision", "answer")
    grounding = result.get("grounding")
    support_state = result.get("support_state", "")

    # Extract subject from the answer if we didn't anchor
    subject_from_answer = extract_subject(answer_text) if not subject_used else subject_used
    final_subject = subject_used or subject_from_answer

    # ── 7. Update session state ───────────────────────────────
    session.add_turn(SessionTurn(
        question=question,
        normalized_query=query_for_retrieval,
        answer_preview=answer_text[:300],
        decision=decision,
        evidence_ids=evidence_ids,
        doc_names=doc_names,
        subject_entity=final_subject,
    ))
    store.put(sid, session)

    # ── 8. Audit trail ────────────────────────────────────────
    # Build evidence text preview (the actual context sent to LLM)
    evidence_text = ""
    try:
        from answerer import format_evidence_for_llm
        evidence_text = format_evidence_for_llm(retrieval, question)
    except Exception:
        pass

    # Build grounding details dict for audit
    grounding_details = None
    if grounding:
        grounding_details = {
            "score": grounding.get("score"),
            "confidence": grounding.get("confidence"),
            "semantic_support": grounding.get("semantic_support"),
            "support_state": grounding.get("support_state", support_state),
            "unsupported_claims": grounding.get("unsupported_claims", [])[:3],
        }

    log_audit_entry(
        request_id=rid,
        session_id=sid,
        question=question,
        normalized_query=query_for_retrieval,
        decision=decision,
        answer_text=answer_text,
        evidence_ids=evidence_ids,
        doc_names=doc_names,
        references_count=len(result.get("references", [])),
        grounding_score=grounding.get("score") if grounding else None,
        grounding_confidence=grounding.get("confidence", "") if grounding else "",
        grounding_details=grounding_details,
        subject_entity=final_subject,
        evidence_text_preview=evidence_text,
        rewrite_diff=rewrite_diff,
        support_state=support_state,
    )

    # ── 9. Build response ─────────────────────────────────────
    final_answer = (ack_prefix + answer_text) if ack_prefix else answer_text

    return SimpleChatResponse(
        answer=final_answer,
        decision=decision,
        references=result.get("references", []),
        session_id=sid,
        grounding=grounding,
    )


# ============================================================
# NEW: Voice Transcription Endpoint
# ============================================================
from fastapi import File, UploadFile
from speech import transcribe_audio


class TranscribeResponse(BaseModel):
    text: str
    success: bool


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(audio: UploadFile = File(...)):
    """Transcribe audio to text using Whisper."""
    try:
        audio_bytes = await audio.read()
        text = transcribe_audio(audio_bytes)
        
        # Check if it's an error message
        is_error = text.startswith("⚠️")
        
        return TranscribeResponse(
            text=text if not is_error else "",
            success=not is_error
        )
    except Exception as e:
        log.error("Transcription endpoint error: %s", e, exc_info=True)
        return TranscribeResponse(text="", success=False)


# ============================================================
# NEW: PDF Serving Endpoint (for Next.js)
# ============================================================
from urllib.parse import unquote
import re

def _normalize_for_match(s: str) -> str:
    """Normalize filename for fuzzy matching - strips all special chars."""
    # Replace all types of dashes with hyphen
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")
    # Replace smart quotes
    s = s.replace("'", "'").replace("'", "'").replace('"', '"').replace('"', '"')
    # Remove extra spaces
    s = re.sub(r'\s+', ' ', s).strip()
    # Lowercase for comparison
    return s.lower()

@app.get("/pdf/{filename:path}")
def serve_pdf(filename: str):
    """Serve PDF files for the Next.js frontend."""
    try:
        # 1. First attempt: EXACT match (but URL decoded)
        # This handles 'PERA – FAQs.pdf' correctly if the fs has an en-dash
        decoded_name = unquote(filename).strip()
        filepath = os.path.join(DATA_DIR, decoded_name)
        
        found = False
        
        # Check exact existence
        if os.path.exists(filepath) and os.path.isfile(filepath):
            found = True
        
        # 2. Second attempt: Normalize dashes (en-dash/em-dash -> hyphen)
        if not found:
            normalized_name = decoded_name.replace("–", "-").replace("—", "-")
            filepath_norm = os.path.join(DATA_DIR, normalized_name)
            if os.path.exists(filepath_norm) and os.path.isfile(filepath_norm):
                filepath = filepath_norm
                found = True
        
        # 3. Third attempt: Try adding .pdf extension if missing
        if not found:
             # Try exact + .pdf
            filepath_ext = os.path.join(DATA_DIR, decoded_name + ".pdf")
            if os.path.exists(filepath_ext):
                filepath = filepath_ext
                found = True
            else:
                 # Try normalized + .pdf
                filepath_ext_norm = os.path.join(DATA_DIR, normalized_name + ".pdf")
                if os.path.exists(filepath_ext_norm):
                    filepath = filepath_ext_norm
                    found = True
        
        # 4. Fourth attempt: Smart fuzzy scan using normalized comparison
        if not found:
            try:
                # Normalize the target name (remove .pdf, normalize all special chars)
                target_norm = _normalize_for_match(decoded_name)
                if target_norm.endswith(".pdf"):
                    target_norm = target_norm[:-4]
                
                # Also try without any dashes/special chars at all
                target_stripped = re.sub(r'[^a-z0-9 ]', '', target_norm)
                    
                for f in os.listdir(DATA_DIR):
                    f_norm = _normalize_for_match(f)
                    if f_norm.endswith(".pdf"):
                        f_norm = f_norm[:-4]
                    f_stripped = re.sub(r'[^a-z0-9 ]', '', f_norm)
                    
                    # Match if normalized versions match OR stripped versions match
                    if f_norm == target_norm or f_stripped == target_stripped:
                        filepath = os.path.join(DATA_DIR, f)
                        found = True
                        log.debug("Fuzzy matched: %s -> %s", decoded_name, f)
                        break
            except Exception as e:
                log.warning("Error during fuzzy scan: %s", e)
        
        if not found:
             log.info("PDF Not Found: %s (decoded: %s)", filename, decoded_name)
             raise HTTPException(status_code=404, detail=f"PDF not found: {filename}")

        # Security check
        if not _is_safe_under_assets_data(filepath):
            log.warning("Access Denied for path: %s", filepath)
            raise HTTPException(status_code=403, detail="Access denied")
        
        return FileResponse(
            filepath,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'inline; filename="{os.path.basename(filepath)}"',
                "Content-Security-Policy": "frame-ancestors *",
                "X-Frame-Options": "ALLOWALL",
                "Access-Control-Allow-Origin": "*"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error("Internal server error serving PDF '%s': %s", filename, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error while serving file")


# ============================================================
# Health & Readiness Endpoints
# ============================================================
@app.get("/health")
def health_check():
    """Lightweight liveness probe — confirms the process is alive."""
    return {"status": "ok"}


@app.get("/ready")
def readiness_check():
    """Readiness probe — confirms the service can serve traffic."""
    checks: Dict[str, Any] = {}

    # 1. OpenAI API key present
    checks["openai_api_key"] = has_api_key()

    # 2. Active index pointer exists and is valid
    try:
        active_dir = indexer.pointer.read()
        checks["active_index"] = active_dir is not None
        checks["active_index_dir"] = active_dir or "none"
    except Exception as e:
        checks["active_index"] = False
        checks["active_index_error"] = str(e)

    # 3. FAISS index loadable (lightweight — just checks file exists)
    if active_dir:
        import os as _os
        faiss_path = _os.path.join(active_dir, "faiss.index").replace("\\", "/")
        chunks_path = _os.path.join(active_dir, "chunks.jsonl").replace("\\", "/")
        checks["faiss_index_exists"] = _os.path.exists(faiss_path)
        checks["chunks_file_exists"] = _os.path.exists(chunks_path)
    else:
        checks["faiss_index_exists"] = False
        checks["chunks_file_exists"] = False

    ready = all([
        checks.get("openai_api_key", False),
        checks.get("active_index", False),
        checks.get("faiss_index_exists", False),
        checks.get("chunks_file_exists", False),
    ])

    return JSONResponse(
        status_code=200 if ready else 503,
        content={"ready": ready, "checks": checks}
    )
