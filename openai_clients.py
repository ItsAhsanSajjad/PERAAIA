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

# Phase 4: dedicated grounding judge model.
# Defaults to ANSWER_MODEL so existing deployments behave identically out
# of the box. Set GROUNDING_JUDGE_MODEL to a stronger model (for example
# "gpt-4o") to escape self-judging bias — the generator no longer judges
# its own output. Optionally set GROUNDING_JUDGE_BASE_URL and
# GROUNDING_JUDGE_API_KEY if the judge lives behind a different gateway
# or account; otherwise the same chat client is reused.
GROUNDING_JUDGE_MODEL: str = os.getenv(
    "GROUNDING_JUDGE_MODEL", ANSWER_MODEL
).strip()
GROUNDING_JUDGE_BASE_URL: str = os.getenv("GROUNDING_JUDGE_BASE_URL", "").strip()
GROUNDING_JUDGE_API_KEY: str = os.getenv("GROUNDING_JUDGE_API_KEY", "").strip()

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
        _transcription_client = OpenAI(
            api_key=require_api_key(),
            timeout=45.0,
            max_retries=1,
        )
    return _transcription_client


# Phase 4 — Grounding judge client.
# Returns a separate OpenAI client when the operator configured a
# distinct base URL OR API key for the judge model. Otherwise reuses
# the main chat client (zero behavioral change). Never raises: callers
# pass-through to the chat client on any failure path.

_judge_client: Optional[OpenAI] = None


def get_grounding_judge_client() -> OpenAI:
    """Return the client to use for grounding-judge calls.

    Resolution order:
      1. If GROUNDING_JUDGE_API_KEY or GROUNDING_JUDGE_BASE_URL is set,
         build (once) a dedicated client with those values.
      2. Otherwise return the shared chat client. This preserves the
         current behavior when no operator override is supplied.

    Safe fallback: if the dedicated client cannot be constructed
    (missing key, etc.), the function returns the shared chat client.
    """
    global _judge_client
    use_dedicated = bool(GROUNDING_JUDGE_API_KEY or GROUNDING_JUDGE_BASE_URL)
    if not use_dedicated:
        return get_chat_client()
    if _judge_client is None:
        try:
            api_key = GROUNDING_JUDGE_API_KEY or require_api_key()
            base_url = (
                GROUNDING_JUDGE_BASE_URL
                or OPENAI_BASE_URL
                or "https://api.openai.com/v1"
            )
            _judge_client = OpenAI(api_key=api_key, base_url=base_url)
        except Exception:
            return get_chat_client()
    return _judge_client
