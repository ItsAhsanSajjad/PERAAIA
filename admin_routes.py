"""
Admin API routes for the PERA AI dashboard.

All endpoints live under ``/api/admin/*`` and require a valid Bearer token
issued by ``POST /api/admin/login``. See ``admin_auth.py`` for the token
format and credential storage.

Endpoints:

* ``POST   /api/admin/login``              — exchange email + password for a JWT.
* ``GET    /api/admin/documents``          — list current PDFs in assets/data with indexed status.
* ``POST   /api/admin/documents``          — upload a new PDF; triggers an index rebuild.
* ``DELETE /api/admin/documents/{name}``   — remove a PDF; triggers an index rebuild.
* ``GET    /api/admin/index-status``       — current active index build metadata.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional
from urllib.parse import unquote

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from admin_auth import (
    check_rate_limit,
    issue_admin_token,
    require_admin,
    verify_credentials,
)
from doc_registry import scan_assets_data
from log_config import get_logger

log = get_logger("pera.admin_routes")

admin_router = APIRouter(prefix="/api/admin", tags=["admin"])

_MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB


# ---------------------------------------------------------------------------
# Lazy accessors — avoid circular imports with fastapi_app
# ---------------------------------------------------------------------------
def _get_indexer():
    import fastapi_app as _app
    return _app.indexer


def _get_data_dir() -> str:
    import fastapi_app as _app
    return _app.DATA_DIR


def _get_active_pointer_path() -> str:
    import fastapi_app as _app
    return _app.ACTIVE_POINTER_PATH


def _is_safe_path(abs_path: str) -> bool:
    import fastapi_app as _app
    return _app._is_safe_under_assets_data(abs_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9 ._()\-]")


def _sanitize_filename(name: str) -> str:
    """Strip path components and unsafe characters, keep extension."""
    base = os.path.basename(name or "").strip()
    base = _SAFE_NAME_RE.sub("_", base)
    # Prevent leading dots (hidden file) and empty after sanitization.
    base = base.lstrip(". ")
    return base


def _unique_target_path(data_dir: str, filename: str) -> str:
    """Return a non-colliding path under data_dir for the given filename."""
    target = os.path.join(data_dir, filename)
    if not os.path.exists(target):
        return target
    stem, ext = os.path.splitext(filename)
    for i in range(1, 1000):
        cand = os.path.join(data_dir, f"{stem}-{i}{ext}")
        if not os.path.exists(cand):
            return cand
    raise HTTPException(status_code=409, detail={"error": "filename_collision"})


def _load_active_manifest() -> Dict[str, Any]:
    """Read the manifest of the currently-active index build."""
    pointer_path = _get_active_pointer_path()
    try:
        with open(pointer_path, "r", encoding="utf-8") as f:
            pointer = json.load(f)
        build_dir = (pointer.get("active_index_dir") or "").replace("\\", "/")
        manifest_path = os.path.join(build_dir, "manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        manifest["_build_dir"] = build_dir
        return manifest
    except Exception:
        return {}


def _indexed_filenames_from_manifest(manifest: Dict[str, Any]) -> set:
    """Extract the set of filenames that are currently indexed.

    The manifest stores ``files`` as a dict keyed by filename with a
    descriptor value. Older builds may store it as a list of dicts — we
    handle both.
    """
    files = manifest.get("files") or {}
    names: set = set()
    if isinstance(files, dict):
        for key, entry in files.items():
            if isinstance(entry, dict):
                fn = entry.get("filename") or key or os.path.basename(entry.get("path", ""))
            else:
                fn = key
            if fn:
                names.add(fn)
    elif isinstance(files, list):
        for entry in files:
            if isinstance(entry, dict):
                fn = entry.get("filename") or os.path.basename(entry.get("path", ""))
                if fn:
                    names.add(fn)
    return names


def _manifest_doc_count(manifest: Dict[str, Any]) -> int:
    files = manifest.get("files") or {}
    if isinstance(files, (dict, list)):
        return len(files)
    return 0


def _manifest_built_at(manifest: Dict[str, Any]) -> int:
    """Return a best-effort unix timestamp for when the active build was made.

    Order: explicit top-level keys, then manifest file mtime.
    """
    for k in ("built_at", "created_at", "timestamp"):
        v = manifest.get(k)
        if isinstance(v, (int, float)) and v > 0:
            return int(v)
    build_dir = manifest.get("_build_dir") or ""
    mf = os.path.join(build_dir, "manifest.json")
    try:
        return int(os.path.getmtime(mf))
    except OSError:
        return 0


def _count_chunks_in_active_build(manifest: Dict[str, Any]) -> int:
    build_dir = manifest.get("_build_dir") or ""
    chunks_path = os.path.join(build_dir, "chunks.jsonl")
    if not os.path.exists(chunks_path):
        return 0
    try:
        # Count active rows only (rows may have active=False flag).
        active = 0
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if row.get("active", True):
                    active += 1
        return active
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class LoginBody(BaseModel):
    email: str
    password: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@admin_router.post("/login")
def admin_login(body: LoginBody, request: Request) -> Dict[str, Any]:
    client_ip = (
        request.client.host
        if request.client else request.headers.get("x-forwarded-for", "unknown")
    )
    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail={"error": "rate_limited", "retry_after_seconds": 60},
        )
    if not verify_credentials(body.email, body.password):
        # Intentionally opaque to prevent user-enumeration.
        log.info("Admin login rejected for ip=%s", client_ip)
        raise HTTPException(
            status_code=401, detail={"error": "invalid_credentials"}
        )
    token, exp = issue_admin_token(body.email.strip())
    log.info("Admin login granted for ip=%s", client_ip)
    return {"token": token, "expires_at": exp}


@admin_router.get("/documents")
def list_documents(claims: Dict[str, Any] = Depends(require_admin)) -> List[Dict[str, Any]]:
    data_dir = _get_data_dir()
    try:
        entries = scan_assets_data(data_dir)
    except Exception as exc:
        log.error("scan_assets_data failed: %s", exc)
        entries = []
    manifest = _load_active_manifest()
    indexed = _indexed_filenames_from_manifest(manifest)
    out: List[Dict[str, Any]] = []
    for e in entries:
        out.append({
            "filename": e.get("filename"),
            "size": int(e.get("size", 0)),
            "mtime": int(e.get("mtime", 0)),
            "authority": int(e.get("authority", 0)),
            "rank": int(e.get("rank", 0)),
            "indexed": e.get("filename") in indexed,
        })
    return out


@admin_router.post("/documents")
async def upload_document(
    file: UploadFile = File(...),
    claims: Dict[str, Any] = Depends(require_admin),
) -> Dict[str, Any]:
    # Basic content-type check (trust-but-verify with magic bytes below).
    if file.content_type and file.content_type not in (
        "application/pdf",
        "application/octet-stream",  # some browsers omit the MIME
    ):
        raise HTTPException(
            status_code=400,
            detail={"error": "unsupported_content_type", "got": file.content_type},
        )
    raw_name = file.filename or ""
    if not raw_name.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail={"error": "only_pdf_allowed"}
        )
    safe = _sanitize_filename(raw_name)
    if not safe.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail={"error": "only_pdf_allowed"}
        )

    data_dir = _get_data_dir()
    os.makedirs(data_dir, exist_ok=True)
    target_path = os.path.join(data_dir, safe)
    if os.path.exists(target_path):
        raise HTTPException(
            status_code=409,
            detail={"error": "duplicate_filename", "filename": safe},
        )
    target_abs = os.path.abspath(target_path)
    if not _is_safe_path(target_abs):
        raise HTTPException(status_code=400, detail={"error": "unsafe_path"})

    # Stream to disk with size cap + magic-byte check on first read.
    total = 0
    first_chunk = True
    try:
        with open(target_abs, "wb") as out_f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                if first_chunk:
                    if not chunk.startswith(b"%PDF-"):
                        out_f.close()
                        try:
                            os.remove(target_abs)
                        except OSError:
                            pass
                        raise HTTPException(
                            status_code=400,
                            detail={"error": "not_a_pdf"},
                        )
                    first_chunk = False
                total += len(chunk)
                if total > _MAX_UPLOAD_BYTES:
                    out_f.close()
                    try:
                        os.remove(target_abs)
                    except OSError:
                        pass
                    raise HTTPException(
                        status_code=413,
                        detail={"error": "file_too_large", "limit_bytes": _MAX_UPLOAD_BYTES},
                    )
                out_f.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        log.error("Upload failed for %s: %s", safe, exc)
        try:
            os.remove(target_abs)
        except OSError:
            pass
        raise HTTPException(status_code=500, detail={"error": "upload_failed"})

    log.info("Admin uploaded PDF: %s (%d bytes)", safe, total)

    # Trigger indexing — blocking call, atomic blue/green swap inside.
    started = time.time()
    try:
        indexer = _get_indexer()
        result = indexer.ensure_index_once() or {}
    except Exception as exc:
        log.error("Index rebuild after upload failed: %s", exc)
        result = {"changed": False, "error": str(exc)}
    duration_ms = int((time.time() - started) * 1000)
    return {
        "filename": os.path.basename(target_abs),
        "size": total,
        "duration_ms": duration_ms,
        "index": result,
    }


@admin_router.delete("/documents/{filename:path}")
def delete_document(
    filename: str,
    claims: Dict[str, Any] = Depends(require_admin),
) -> Dict[str, Any]:
    name = unquote(filename or "").strip()
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise HTTPException(status_code=400, detail={"error": "bad_filename"})
    data_dir = _get_data_dir()
    target_abs = os.path.abspath(os.path.join(data_dir, name))
    if not _is_safe_path(target_abs):
        raise HTTPException(status_code=400, detail={"error": "unsafe_path"})
    if not os.path.exists(target_abs):
        raise HTTPException(status_code=404, detail={"error": "not_found"})
    try:
        os.remove(target_abs)
    except OSError as exc:
        log.error("Delete failed for %s: %s", name, exc)
        raise HTTPException(status_code=500, detail={"error": "delete_failed"})

    log.info("Admin deleted PDF: %s", name)

    started = time.time()
    try:
        indexer = _get_indexer()
        result = indexer.ensure_index_once() or {}
    except Exception as exc:
        log.error("Index rebuild after delete failed: %s", exc)
        result = {"changed": False, "error": str(exc)}
    duration_ms = int((time.time() - started) * 1000)
    return {
        "filename": name,
        "duration_ms": duration_ms,
        "index": result,
    }


@admin_router.get("/index-status")
def index_status(claims: Dict[str, Any] = Depends(require_admin)) -> Dict[str, Any]:
    manifest = _load_active_manifest()
    build_dir = manifest.get("_build_dir") or ""
    return {
        "active_index_dir": build_dir,
        "built_at": _manifest_built_at(manifest),
        "doc_count": _manifest_doc_count(manifest),
        "chunk_count": _count_chunks_in_active_build(manifest),
    }
