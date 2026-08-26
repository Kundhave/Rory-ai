"""The failure matrix, tested deterministically — no network, no API cost.

Every row here must produce a sentence a user could act on: never a stack
trace, never a fabricated success. The two anti-fabrication rules (unverified
tool results, empty retrieval) get the most coverage because they are the
ones where a wrong answer is indistinguishable from a right one.
"""
import httpx
import psutil
import pytest

from rory.core import Reply, RoryCore
from rory.llm import _is_retryable
from rory.tools.registry import dispatch
from rory.voice import tts as tts_module
from rory.voice.stt import is_usable
from tests.fakes import FakeLLM, calls, says


# --- STT returns empty or garbage -> no LLM call, prompt retry ---------------

def test_empty_transcript_never_reaches_the_llm(capsys):
    from rory.cli import run_voice_turn

    class EmptySTT:
        def transcribe(self, wav_bytes: bytes) -> str:
            return "   "

    class NeverCalled:
        def handle_text(self, text: str):
            raise AssertionError("the LLM must not be called for an empty transcript")

    run_voice_turn(EmptySTT(), NeverCalled(), tts=None, wav_bytes=b"x")

    assert "couldn't hear" in capsys.readouterr().out.lower()


@pytest.mark.parametrize("garbage", ["", "   ", "\n\t "])
def test_garbage_transcripts_are_rejected(garbage):
    assert is_usable(garbage) is False


# --- LLM 429 / 5xx / timeout -> one retry, then honest failure ---------------

def test_rate_limit_and_server_errors_are_retryable_but_a_dead_network_is_not():
    from google.genai import errors

    request = httpx.Request("POST", "https://example.invalid")
    assert _is_retryable(errors.ClientError(429, {"error": {"message": "quota"}})) is True
    assert _is_retryable(errors.ServerError(503, {"error": {"message": "unavailable"}})) is True
    assert _is_retryable(httpx.ReadTimeout("timed out")) is True
    # Fail fast: retrying a down network just makes the user wait longer.
    assert _is_retryable(httpx.ConnectError("unreachable", request=request)) is False
    # A genuine 400 is a bug to surface, not something to paper over.
    assert _is_retryable(errors.ClientError(400, {"error": {"message": "bad request"}})) is False


def test_an_llm_failure_becomes_an_honest_reply_not_an_exception():
    class Exhausted:
        def generate(self, messages, system=None, tools=None):
            raise RuntimeError("429 RESOURCE_EXHAUSTED")

    reply = RoryCore(Exhausted()).handle_text("hi")

    assert reply.error is not None
    assert "429" in reply.error
    assert reply.text == ""  # nothing fabricated to fill the gap


# --- retrieval returns nothing -> say it is not in the notes -----------------

def test_empty_retrieval_is_handed_to_the_model_as_an_empty_result_set(monkeypatch):
    from rory.rag import retrieve

    monkeypatch.setattr(retrieve, "load_chunks", lambda: [])
    monkeypatch.setattr(retrieve, "load_vectors", lambda: [])
    monkeypatch.setattr(retrieve, "search", lambda *a, **k: [])

    result = dispatch("search_notes", {"query": "something absent"})

    assert result["ok"] is True
    assert result["results"] == []


def test_the_model_sees_an_empty_result_set_before_answering(monkeypatch):
    from rory.rag import retrieve

    monkeypatch.setattr(retrieve, "load_chunks", lambda: [])
    monkeypatch.setattr(retrieve, "load_vectors", lambda: [])
    monkeypatch.setattr(retrieve, "search", lambda *a, **k: [])

    fake = FakeLLM([
        calls(("search_notes", {"query": "my exam result"})),
        says("That isn't in your notes."),
    ])
    reply = RoryCore(fake).handle_text("what was my exam result?")

    assert reply.text == "That isn't in your notes."
    assert '"results": []' in fake.calls[1][-1]["content"]


def test_grounding_rules_cover_empty_and_weak_retrieval():
    from rory.agent.prompts import build_system_prompt

    prompt = build_system_prompt()
    assert "empty `results` list" in prompt
    assert "weak" in prompt          # hedging rule for low scores
    assert "general knowledge" in prompt  # outside-the-KB rule


# --- tool raises -> {ok: false} envelope, model explains --------------------

def test_a_raising_tool_becomes_an_envelope_the_model_can_explain(monkeypatch):
    monkeypatch.setattr(psutil, "process_iter", lambda attrs: 1 / 0)

    result = dispatch("check_app_running", {"app": "browser"})

    assert result["ok"] is False
    assert "ZeroDivisionError" in result["error"]


def test_an_unknown_app_is_rejected_before_any_process_starts(monkeypatch):
    import subprocess

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("must not launch"))

    result = dispatch("open_app", {"app": "spotify"})

    assert result["ok"] is False
    assert "not one of" in result["error"]


# --- app not installed -> report honestly, do not claim success -------------

def test_a_missing_binary_is_reported_as_failure_not_success(monkeypatch):
    import subprocess

    def missing(*args, **kwargs):
        raise FileNotFoundError("no such binary")

    monkeypatch.setattr(subprocess, "Popen", missing)

    result = dispatch("open_app", {"app": "browser"})

    assert result["ok"] is False
    assert "could not launch" in result["detail"]


def test_an_already_running_app_is_success_and_worded_without_overclaiming(monkeypatch):
    import subprocess

    from rory.tools import desktop

    class DelegatedToExistingInstance:
        """Most apps hand off to a running instance and exit 0 immediately."""

        def __init__(self, argv, **kwargs):
            pass

        def poll(self):
            return 0

    monkeypatch.setattr(subprocess, "Popen", DelegatedToExistingInstance)
    monkeypatch.setattr(desktop.time, "sleep", lambda _: None)

    result = dispatch("open_app", {"app": "browser"})

    assert result["ok"] is True  # not an error
    assert "launched" not in result["detail"]  # would overstate a fresh start


# --- process check fails -> verified=false ---------------------------------

def test_an_unverifiable_process_check_reports_neither_yes_nor_no():
    result = dispatch("check_app_running", {"app": "chatgpt"})

    assert result["verified"] is False
    assert result["running"] is None  # no value the model could read as an answer


def test_a_partial_scan_does_not_claim_the_app_is_absent(monkeypatch):
    class Unreadable:
        @property
        def info(self):
            raise psutil.AccessDenied()

    monkeypatch.setattr(psutil, "process_iter", lambda attrs: [Unreadable()])

    result = dispatch("check_app_running", {"app": "browser"})

    assert result["running"] is False
    assert result["verified"] is False


def test_the_model_is_given_the_unverified_flag_before_answering():
    fake = FakeLLM([
        calls(("check_app_running", {"app": "chatgpt"})),
        says("I couldn't tell whether ChatGPT is running."),
    ])
    reply = RoryCore(fake).handle_text("is chatgpt open?")

    fed_back = fake.calls[1][-1]["content"]
    assert '"verified": false' in fed_back
    assert '"running": null' in fed_back
    assert reply.text == "I couldn't tell whether ChatGPT is running."


# --- TTS fails -> show text, never lose the answer -------------------------

def test_a_tts_failure_never_costs_the_user_the_answer(capsys):
    from rory.cli import speak

    class BrokenTTS:
        def synthesize(self, text: str) -> bytes:
            raise ConnectionError("network down")

    speak(BrokenTTS(), Reply(turn_id="t1", text="Exponential backoff."))

    assert "voice unavailable" in capsys.readouterr().out


def test_sarvam_credit_exhaustion_falls_back_to_local_audio():
    class Exhausted:
        def synthesize(self, text: str) -> bytes:
            raise tts_module.SarvamCreditExhausted("402")

    class Local:
        def synthesize(self, text: str) -> bytes:
            return b"local-audio"

    engine = tts_module.FallbackTTS(primary=Exhausted(), fallback=Local())

    assert engine.synthesize("hello") == b"local-audio"
    assert engine.last_engine == "fallback"


def test_a_dead_network_does_not_burn_three_tts_attempts():
    request = httpx.Request("POST", "https://api.sarvam.ai/text-to-speech")

    class Unreachable:
        def __init__(self) -> None:
            self.calls = 0

        def synthesize(self, text: str) -> bytes:
            self.calls += 1
            raise httpx.ConnectError("unreachable", request=request)

    class Local:
        def synthesize(self, text: str) -> bytes:
            return b"local-audio"

    primary = Unreachable()
    engine = tts_module.FallbackTTS(primary=primary, fallback=Local(), attempts=3)

    assert engine.synthesize("hello") == b"local-audio"
    assert primary.calls == 1  # fail fast, not three round trips to nowhere


# --- audio device busy -> error naming the actual device -------------------

def test_a_busy_microphone_names_the_device(monkeypatch):
    import sounddevice as sd

    from rory.voice.audio import Recorder

    monkeypatch.setattr(
        sd, "query_devices",
        lambda device, kind: {"default_samplerate": 44100.0, "name": "HDA Intel PCH: ALC257 Analog"},
    )

    def busy(**kwargs):
        raise sd.PortAudioError("Device unavailable")

    monkeypatch.setattr(sd, "InputStream", busy)

    recorder = Recorder()
    with pytest.raises(RuntimeError) as excinfo:
        recorder.start()

    assert "HDA Intel PCH: ALC257 Analog" in str(excinfo.value)


def test_a_microphone_failure_in_the_widget_becomes_an_error_state_not_a_crash():
    from rory.ui.widget import State, VoiceWorker

    class BusyMic:
        def start(self) -> None:
            raise RuntimeError("microphone 'ALC257 Analog' unavailable: Device unavailable")

    worker = VoiceWorker(
        core=None, tts=None, stt=None,
        recorder_factory=BusyMic, player=lambda audio: None,
    )
    events = []
    worker.state_changed.connect(lambda state, detail: events.append((state, detail)))

    worker.start_listening()

    assert events[-1][0] == State.ERROR.value
    assert "ALC257 Analog" in events[-1][1]
