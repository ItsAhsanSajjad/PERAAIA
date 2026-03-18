"""
PERA AI — Centralized OpenAI Client & Config

Single source of truth for API keys, base URLs, model names,
and lazy-init clients used across answerer, retriever, index_store, and speech.
"""
from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ─── Configuration (all from .env) ───────────────────────────────────────────

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()

# Model names
ANSWER_MODEL: str = os.getenv("ANSWER_MODEL", "gpt-4o-mini").strip()
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip()
LLM_REWRITE_MODEL: str = os.getenv("RETRIEVER_LLM_QUERY_REWRITE_MODEL", "gpt-4o-mini").strip()
TRANSCRIBE_MODEL: str = os.getenv("TRANSCRIBE_MODEL", "whisper-1").strip()

# ─── Validation ──────────────────────────────────────────────────────────────

def require_api_key() -> str:
    """Raise RuntimeError if OPENAI_API_KEY is missing."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing. Ensure .env is present and loaded.")
    return OPENAI_API_KEY


def has_api_key() -> bool:
    """Non-throwing check for readiness probes."""
    return bool(OPENAI_API_KEY)


# ─── Lazy-init Clients ──────────────────────────────────────────────────────
# The "chat/rewrite" client uses OPENAI_BASE_URL (may point to a gateway).
# The "transcription" client always uses the default OpenAI API (Whisper).

_chat_client: Optional[OpenAI] = None
_transcription_client: Optional[OpenAI] = None


def get_chat_client() -> OpenAI:
    """
    Shared client for chat completions and embeddings.
    Uses OPENAI_BASE_URL which may point to a gateway.
    """
    global _chat_client
    if _chat_client is None:
        _chat_client = OpenAI(
            api_key=require_api_key(),
            base_url=OPENAI_BASE_URL or "https://api.openai.com/v1",
        )
    return _chat_client


def get_transcription_client() -> OpenAI:
    """
    Client for Whisper transcription.
    Always uses the standard OpenAI API (not the gateway base_url).
    """
    global _transcription_client
    if _transcription_client is None:
        _transcription_client = OpenAI(api_key=require_api_key())
    return _transcription_client
