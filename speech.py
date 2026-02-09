from __future__ import annotations

import os
import io
import tempfile
import subprocess
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def _client() -> OpenAI:
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is missing. Ensure .env is present and loaded.")
    return OpenAI(api_key=key)


def _guess_ext(audio_bytes: bytes) -> str:
    """
    Best-effort magic header detection.
    """
    if not audio_bytes:
        return ".bin"

    head = audio_bytes[:16]

    # WAV: "RIFF....WAVE"
    if head[:4] == b"RIFF" and b"WAVE" in head:
        return ".wav"

    # MP3: "ID3" or MPEG frame sync 0xFF 0xFB
    if head[:3] == b"ID3" or (len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0):
        return ".mp3"

    # OGG: "OggS"
    if head[:4] == b"OggS":
        return ".ogg"

    # WEBM/Matroska: 1A 45 DF A3
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return ".webm"

    # M4A/MP4 often: "....ftyp"
    if len(head) >= 12 and head[4:8] == b"ftyp":
        return ".m4a"

    return ".bin"


def _ffmpeg_exists() -> bool:
    try:
        p = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        return p.returncode == 0
    except Exception:
        return False


def _convert_to_wav(in_path: str) -> Optional[str]:
    """
    Convert input audio to WAV using ffmpeg (if installed).
    Returns wav path or None if conversion fails.
    """
    if not _ffmpeg_exists():
        return None

    out_path = in_path + ".wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-i", in_path,
        "-ac", "1",
        "-ar", "16000",
        out_path
    ]

    try:
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            return None
        if not os.path.exists(out_path) or os.path.getsize(out_path) < 1024:
            return None
        return out_path
    except Exception:
        return None


def transcribe_audio(audio_bytes: bytes, model: str = None) -> str:
    """
    Safe transcription:
    - Never raises to Streamlit
    - Returns readable error text if unsupported/corrupt
    """
    if not audio_bytes or len(audio_bytes) < 800:
        return "⚠️ I could not read the audio (empty/too short). Please record again."

    model = model or (os.getenv("TRANSCRIBE_MODEL") or "whisper-1")

    ext = _guess_ext(audio_bytes)

    try:
        with tempfile.TemporaryDirectory() as td:
            raw_path = os.path.join(td, f"audio{ext}")
            with open(raw_path, "wb") as f:
                f.write(audio_bytes)

            # If file is unknown/webm, try converting to wav for maximum compatibility
            use_path = raw_path
            if ext in (".webm", ".bin", ".m4a", ".ogg"):
                wav = _convert_to_wav(raw_path)
                if wav:
                    use_path = wav

            # Call OpenAI transcription
            client = _client()
            with open(use_path, "rb") as f:
                result = client.audio.transcriptions.create(
                    model=model,
                    file=f
                )

            text = (getattr(result, "text", None) or "").strip()
            if not text:
                return "⚠️ I could not transcribe the audio. Please try again with clearer speech."
            return text

    except Exception as e:
        # Never crash the app
        msg = str(e)
        if "corrupted" in msg.lower() or "unsupported" in msg.lower() or "invalid_value" in msg.lower():
            return "⚠️ Audio format not supported. Please record again. (If this continues, install FFmpeg for conversion.)"
        return f"⚠️ Voice transcription failed: {msg}"
