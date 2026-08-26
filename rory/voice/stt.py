"""STT Protocol + Sarvam Saaras."""
from __future__ import annotations

from typing import Protocol

import httpx

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"


class STT(Protocol):
    def transcribe(self, wav_bytes: bytes) -> str:
        ...


class SarvamSTT:
    def __init__(self, api_key: str, model: str = "saaras:v3") -> None:
        self._api_key = api_key
        self._model = model

    def transcribe(self, wav_bytes: bytes) -> str:
        response = httpx.post(
            SARVAM_STT_URL,
            headers={"api-subscription-key": self._api_key},
            data={"model": self._model, "language_code": "unknown"},
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json().get("transcript", "")


def is_usable(transcript: str) -> bool:
    """Empty or whitespace-only transcripts must not reach the LLM — this is
    the check that keeps silence, noise, or a dropped recording from turning
    into a confusing empty-input turn."""
    return bool(transcript and transcript.strip())
