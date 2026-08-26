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

# Measured against the live API: successful calls are fast and barely scale
# with length — 0.95s for 25 chars, 2.12s for 105, 2.13s for 300 — while a
# failing one always hangs for ~30.3s before returning a 500 (Sarvam's own
# server-side timeout). The two populations are cleanly separated, so a
# short timeout costs nothing on success and converts a 30s dead wait into a
# fast retry. 5s leaves >2x headroom over the slowest success observed.
SARVAM_TIMEOUT_S = 5.0

# Failures are transient and independent (observed 4/6 failing in one burst,
# then succeeding immediately on the next call), so a retry is far more
# likely to get the real voice than to waste time. This exists because the
# fallback — espeak-ng — sounds audibly much worse; retrying is what keeps
# a temporary Sarvam wobble from degrading the voice for a whole turn.
SARVAM_ATTEMPTS = 3

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
        speaker: str = "shreya",
        model: str = "bulbul:v3",
        language_code: str = "en-IN",
        timeout: float = SARVAM_TIMEOUT_S,
    ) -> None:
        self._api_key = api_key
        self._speaker = speaker
        self._model = model
        self._language_code = language_code
        self._timeout = timeout

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
            timeout=self._timeout,
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
    """Retries the primary engine on transient failures, then falls back to
    local rather than losing the turn's spoken answer.

    Retrying before falling back matters because the two engines are not
    equivalent: the fallback (espeak-ng) is a formant synthesizer that
    sounds audibly far worse than Sarvam's neural voice. Since Sarvam's
    failures were measured to be transient and its successes fast (~1.1s),
    another attempt usually costs little and recovers the good voice.

    Credit exhaustion is not retried — that's a persistent condition, so
    retrying only burns the user's time. A 4xx other than 402/403 (e.g. a
    malformed request) is a bug worth surfacing, not masking, so it
    re-raises immediately.

    `last_engine` records which engine actually produced the audio, so a
    caller can trace whether the user is hearing the real voice or the
    degraded one — otherwise "why does it sound bad?" is invisible.
    """

    def __init__(self, primary: TTS, fallback: TTS, attempts: int = SARVAM_ATTEMPTS) -> None:
        self._primary = primary
        self._fallback = fallback
        self._attempts = attempts
        self.last_engine: str | None = None

    def synthesize(self, text: str) -> bytes:
        for attempt in range(self._attempts):
            try:
                audio = self._primary.synthesize(text)
                self.last_engine = "primary"
                return audio
            except SarvamCreditExhausted:
                break
            except httpx.ConnectError:
                # The network is down. Retrying cannot succeed and only
                # delays the local voice that works offline — fail fast.
                # Checked before TransportError, which it subclasses.
                break
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code < 500:
                    raise
            except httpx.TransportError:
                pass

        self.last_engine = "fallback"
        return self._fallback.synthesize(text)


class CachedTTS:
    def __init__(self, engine: TTS, voice: str, cache_dir: Path = CACHE_DIR) -> None:
        self._engine = engine
        self._voice = voice
        self._cache_dir = cache_dir
        self.last_engine: str | None = None

    def synthesize(self, text: str) -> bytes:
        text = strip_markdown(text)
        key = hashlib.sha256(f"{text}|{self._voice}".encode("utf-8")).hexdigest()
        path = self._cache_dir / f"{key}.wav"
        if path.exists():
            self.last_engine = "cache"
            return path.read_bytes()

        audio = self._engine.synthesize(text)
        # Forwarded so a caller sees which engine really spoke, through the
        # cache wrapper rather than only from the layer that knows.
        self.last_engine = getattr(self._engine, "last_engine", None)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(audio)
        return audio


def build_tts(settings) -> TTS:
    local = LocalTTS(voice=settings.tts_local_voice)
    if settings.tts_engine == "local":
        return CachedTTS(local, voice=f"local:{settings.tts_local_voice}")

    sarvam = SarvamTTS(
        api_key=settings.sarvam_api_key,
        speaker=settings.tts_voice,
        model=settings.tts_model,
    )
    engine = FallbackTTS(primary=sarvam, fallback=local)
    # Model is part of the cache identity: the same speaker name can sound
    # different across models, and switching must not replay stale audio.
    return CachedTTS(engine, voice=f"sarvam:{settings.tts_model}:{settings.tts_voice}")
