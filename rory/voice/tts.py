"""TTS Protocol + Sarvam Bulbul + a local espeak-ng fallback + a disk cache.

Cache key is sha256(text + voice), where "voice" identifies the exact
engine+speaker that would produce the audio — so switching providers or
speakers can never return stale audio for matching text, and repeated dev
runs (and repeated phrases in normal use) cost nothing.
"""
from __future__ import annotations

import base64
import hashlib
import re
import subprocess
from pathlib import Path
from typing import Protocol

import httpx

SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
CACHE_DIR = Path("data/tts_cache")

_MARKDOWN_PATTERNS = [
    (re.compile(r"\*\*\*(.+?)\*\*\*"), r"\1"),  # ***bold italic***
    (re.compile(r"\*\*(.+?)\*\*"), r"\1"),  # **bold**
    (re.compile(r"\*(.+?)\*"), r"\1"),  # *italic*
    (re.compile(r"__(.+?)__"), r"\1"),  # __bold__
    (re.compile(r"(?<!\w)_(.+?)_(?!\w)"), r"\1"),  # _italic_
    (re.compile(r"`(.+?)`"), r"\1"),  # `code`
    (re.compile(r"^#{1,6}\s+", re.MULTILINE), ""),  # # headers
    (re.compile(r"^[-*+]\s+", re.MULTILINE), ""),  # - bullet markers
]


def strip_markdown(text: str) -> str:
    """The LLM writes replies with markdown emphasis meant for visual
    display (e.g. "**CUSTOS**"). A TTS engine has no notion of markdown and
    reads the literal characters — "asterisk asterisk CUSTOS asterisk
    asterisk" — so this strips common markdown syntax before synthesis.
    Applied here rather than relying on a prompt instruction: an LLM
    avoiding markdown "most of the time" isn't good enough when the failure
    mode is audibly reading out punctuation."""
    cleaned = text
    for pattern, repl in _MARKDOWN_PATTERNS:
        cleaned = pattern.sub(repl, cleaned)
    return cleaned


class TTS(Protocol):
    def synthesize(self, text: str) -> bytes:
        """Return WAV bytes."""
        ...


class SarvamCreditExhausted(Exception):
    """Raised on HTTP 402/403 — no credit left, or the key can't bill."""


class SarvamTTS:
    def __init__(
        self,
        api_key: str,
        speaker: str = "anushka",
        model: str = "bulbul:v2",
        language_code: str = "en-IN",
    ) -> None:
        self._api_key = api_key
        self._speaker = speaker
        self._model = model
        self._language_code = language_code

    def synthesize(self, text: str) -> bytes:
        response = httpx.post(
            SARVAM_TTS_URL,
            headers={"api-subscription-key": self._api_key},
            json={
                "text": text,
                "language_code": self._language_code,
                "speaker": self._speaker,
                "model": self._model,
            },
            timeout=60.0,
        )
        if response.status_code in (402, 403):
            raise SarvamCreditExhausted(f"Sarvam TTS returned {response.status_code}")
        response.raise_for_status()
        audio_b64 = response.json()["audios"][0]
        return base64.b64decode(audio_b64)


class LocalTTS:
    """espeak-ng, invoked as an argv list with shell=False — same subprocess
    discipline as tools/desktop.py. Requires the espeak-ng system package.

    espeak-ng's base English voices are male; "+fN" is its formant-shift
    modifier for a female-sounding variant of the same voice (N picks which
    formant preset, 1-4 — f3 is a reasonable default, no separate voice pack
    needed).
    """

    def __init__(self, voice: str = "en-us+f3") -> None:
        self._voice = voice

    def synthesize(self, text: str) -> bytes:
        result = subprocess.run(
            ["espeak-ng", "--stdout", "-v", self._voice, text],
            shell=False,
            capture_output=True,
            check=True,
        )
        return result.stdout


class FallbackTTS:
    """Tries the primary engine; falls back to local rather than losing the
    turn's spoken answer when Sarvam itself is the problem: credit
    exhaustion, a server error, or the connection timing out. Observed all
    three during development against the real API — this isn't
    speculative. A 4xx other than 402/403 (e.g. a malformed request) is a
    bug worth surfacing, not masking, so it re-raises."""

    def __init__(self, primary: TTS, fallback: TTS) -> None:
        self._primary = primary
        self._fallback = fallback

    def synthesize(self, text: str) -> bytes:
        try:
            return self._primary.synthesize(text)
        except SarvamCreditExhausted:
            return self._fallback.synthesize(text)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code >= 500:
                return self._fallback.synthesize(text)
            raise
        except httpx.TransportError:
            return self._fallback.synthesize(text)


class CachedTTS:
    def __init__(self, engine: TTS, voice: str, cache_dir: Path = CACHE_DIR) -> None:
        self._engine = engine
        self._voice = voice
        self._cache_dir = cache_dir

    def synthesize(self, text: str) -> bytes:
        text = strip_markdown(text)
        key = hashlib.sha256(f"{text}|{self._voice}".encode("utf-8")).hexdigest()
        path = self._cache_dir / f"{key}.wav"
        if path.exists():
            return path.read_bytes()

        audio = self._engine.synthesize(text)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(audio)
        return audio


def build_tts(settings) -> TTS:
    local = LocalTTS(voice=settings.tts_local_voice)
    if settings.tts_engine == "local":
        return CachedTTS(local, voice=f"local:{settings.tts_local_voice}")

    sarvam = SarvamTTS(api_key=settings.sarvam_api_key, speaker=settings.tts_voice)
    engine = FallbackTTS(primary=sarvam, fallback=local)
    return CachedTTS(engine, voice=f"sarvam:{settings.tts_voice}")
