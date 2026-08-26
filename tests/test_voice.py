import subprocess

import httpx
import numpy as np
import pytest

from rory.voice import audio, stt, tts


def test_wav_round_trip_preserves_samples_and_rate():
    samples = np.array([0, 1000, -1000, 32000, -32000], dtype=np.int16)

    wav_bytes = audio.to_wav_bytes(samples, sample_rate=16000)
    decoded, rate = audio.from_wav_bytes(wav_bytes)

    assert rate == 16000
    assert np.array_equal(decoded, samples)


def test_record_then_play_round_trip_preserves_format():
    """Simulates what Recorder.stop() produces (resample to 16kHz, encode to
    WAV) and what play() consumes (decode WAV) — the actual device calls
    aren't exercised, but the format contract between them is."""
    one_second_at_44100 = np.sin(np.linspace(0, 440 * 2 * np.pi, 44100)).astype(np.float64)
    native_samples = (one_second_at_44100 * 10000).astype(np.int16)

    resampled = audio.resample(native_samples, from_rate=44100, to_rate=audio.SAMPLE_RATE)
    wav_bytes = audio.to_wav_bytes(resampled, audio.SAMPLE_RATE)
    decoded, rate = audio.from_wav_bytes(wav_bytes)

    assert rate == audio.SAMPLE_RATE
    # ~1 second of audio at the target rate, resampling arithmetic aside.
    assert abs(len(decoded) - audio.SAMPLE_RATE) <= 1
    assert decoded.dtype == np.int16


def test_resample_is_a_noop_at_matching_rates():
    samples = np.array([1, 2, 3], dtype=np.int16)

    assert np.array_equal(audio.resample(samples, 16000, 16000), samples)


def test_resample_handles_empty_input():
    empty = np.zeros((0,), dtype=np.int16)

    assert len(audio.resample(empty, 44100, 16000)) == 0


def test_is_usable_rejects_empty_and_whitespace_transcripts():
    assert stt.is_usable("") is False
    assert stt.is_usable("   ") is False
    assert stt.is_usable("hello") is True


class CountingTTS:
    def __init__(self, audio_bytes: bytes = b"fake-wav") -> None:
        self.calls = 0
        self._audio_bytes = audio_bytes

    def synthesize(self, text: str) -> bytes:
        self.calls += 1
        return self._audio_bytes


def test_strip_markdown_removes_bold_italic_code_and_headers():
    assert tts.strip_markdown("**CUSTOS**") == "CUSTOS"
    assert tts.strip_markdown("*emphasis*") == "emphasis"
    assert tts.strip_markdown("***very bold***") == "very bold"
    assert tts.strip_markdown("__CUSTOS__") == "CUSTOS"
    assert tts.strip_markdown("use `search_notes` for that") == "use search_notes for that"
    assert tts.strip_markdown("# Heading\nBody text") == "Heading\nBody text"
    assert tts.strip_markdown("- one\n- two") == "one\ntwo"


def test_strip_markdown_leaves_ordinary_text_untouched():
    text = "Relay uses exponential backoff for retries, 5 attempts total."
    assert tts.strip_markdown(text) == text


def test_cached_tts_strips_markdown_before_synthesizing_and_before_hashing(tmp_path):
    underlying = CountingTTS()
    cached = tts.CachedTTS(underlying, voice="local:default", cache_dir=tmp_path)

    cached.synthesize("**CUSTOS** is a platform")
    # A second call whose markdown-stripped form is identical must still hit
    # the cache — the key is computed from the cleaned text.
    cached.synthesize("CUSTOS is a platform")

    assert underlying.calls == 1


def test_cached_tts_does_not_recall_the_engine_on_a_hit(tmp_path):
    underlying = CountingTTS()
    cached = tts.CachedTTS(underlying, voice="local:default", cache_dir=tmp_path)

    first = cached.synthesize("hello there")
    second = cached.synthesize("hello there")

    assert underlying.calls == 1
    assert first == second == b"fake-wav"


def test_cache_key_includes_voice_so_switching_voices_does_not_collide(tmp_path):
    underlying = CountingTTS()
    voice_a = tts.CachedTTS(underlying, voice="sarvam:anushka", cache_dir=tmp_path)
    voice_b = tts.CachedTTS(underlying, voice="sarvam:manisha", cache_dir=tmp_path)

    voice_a.synthesize("hello there")
    voice_b.synthesize("hello there")

    assert underlying.calls == 2


def test_local_tts_invokes_espeak_as_an_argv_list_never_a_shell(monkeypatch):
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(argv, 0, stdout=b"RIFF....WAVE")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = tts.LocalTTS(voice="en-us+f3").synthesize("hello; rm -rf /")

    assert isinstance(captured["argv"], list)
    assert captured["kwargs"]["shell"] is False
    assert "hello; rm -rf /" in captured["argv"]
    assert "en-us+f3" in captured["argv"]
    assert result == b"RIFF....WAVE"


def test_fallback_tts_switches_to_local_when_sarvam_credit_is_exhausted():
    class ExhaustedSarvam:
        def synthesize(self, text: str) -> bytes:
            raise tts.SarvamCreditExhausted("402")

    local = CountingTTS(b"local-audio")
    fallback = tts.FallbackTTS(primary=ExhaustedSarvam(), fallback=local)

    result = fallback.synthesize("hello there")

    assert result == b"local-audio"
    assert local.calls == 1


def test_fallback_tts_does_not_touch_local_when_primary_succeeds():
    primary = CountingTTS(b"sarvam-audio")
    local = CountingTTS(b"local-audio")
    fallback = tts.FallbackTTS(primary=primary, fallback=local)

    result = fallback.synthesize("hello there")

    assert result == b"sarvam-audio"
    assert local.calls == 0


def test_fallback_tts_switches_to_local_on_a_sarvam_server_error():
    request = httpx.Request("POST", "https://api.sarvam.ai/text-to-speech")
    response = httpx.Response(500, request=request)

    class FlakyServer:
        def synthesize(self, text: str) -> bytes:
            raise httpx.HTTPStatusError("500", request=request, response=response)

    local = CountingTTS(b"local-audio")
    fallback = tts.FallbackTTS(primary=FlakyServer(), fallback=local)

    assert fallback.synthesize("hello there") == b"local-audio"
    assert local.calls == 1


def test_fallback_tts_reraises_a_client_error_other_than_credit_exhaustion():
    request = httpx.Request("POST", "https://api.sarvam.ai/text-to-speech")
    response = httpx.Response(400, request=request)

    class BadRequest:
        def synthesize(self, text: str) -> bytes:
            raise httpx.HTTPStatusError("400", request=request, response=response)

    local = CountingTTS(b"local-audio")
    fallback = tts.FallbackTTS(primary=BadRequest(), fallback=local)

    with pytest.raises(httpx.HTTPStatusError):
        fallback.synthesize("hello there")
    assert local.calls == 0


def test_fallback_tts_switches_to_local_on_a_timeout():
    class TimingOut:
        def synthesize(self, text: str) -> bytes:
            raise httpx.ReadTimeout("timed out")

    local = CountingTTS(b"local-audio")
    fallback = tts.FallbackTTS(primary=TimingOut(), fallback=local)

    assert fallback.synthesize("hello there") == b"local-audio"
    assert local.calls == 1


@pytest.mark.live
def test_sarvam_tts_real_api_call():
    from rory.config import settings

    client = tts.SarvamTTS(api_key=settings.sarvam_api_key)
    audio_bytes = client.synthesize("Hello from Rory.")

    assert audio_bytes[:4] == b"RIFF"


@pytest.mark.live
def test_sarvam_stt_real_api_call():
    from rory.config import settings

    tts_client = tts.SarvamTTS(api_key=settings.sarvam_api_key)
    wav_bytes = tts_client.synthesize("Testing speech to text.")

    stt_client = stt.SarvamSTT(api_key=settings.sarvam_api_key)
    transcript = stt_client.transcribe(wav_bytes)

    assert stt.is_usable(transcript)


class NeverCalledCore:
    def handle_text(self, text: str):
        raise AssertionError("handle_text must not be called for an unusable transcript")


class EmptySTT:
    def transcribe(self, wav_bytes: bytes) -> str:
        return "   "


def test_empty_transcript_short_circuits_before_any_llm_call(capsys):
    from rory.cli import run_voice_turn

    run_voice_turn(EmptySTT(), NeverCalledCore(), tts=None, wav_bytes=b"whatever")

    assert "couldn't hear" in capsys.readouterr().out.lower()


def test_usable_transcript_is_shown_before_the_answer(capsys):
    from rory.cli import run_voice_turn
    from rory.core import Reply

    class HeardSTT:
        def transcribe(self, wav_bytes: bytes) -> str:
            return "what does Relay use for retries?"

    class StubCore:
        def handle_text(self, text: str) -> Reply:
            assert text == "what does Relay use for retries?"
            return Reply(turn_id="t1", text="Exponential backoff.")

    run_voice_turn(HeardSTT(), StubCore(), tts=CountingTTS(), wav_bytes=b"whatever")

    out = capsys.readouterr().out
    heard_line = out.index("heard: what does Relay use for retries?")
    answer_line = out.index("Exponential backoff.")
    assert heard_line < answer_line
