"""State-transition tests for VoiceWorker. No QApplication.exec(), no
threads, no rendering — signals are connected with plain Python callables and
invoked directly on the test's own thread (Qt uses a direct, synchronous
call when sender and receiver share a thread)."""
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from rory.core import Reply
from rory.ui import widget as widget_module
from rory.ui.widget import State, VoiceWorker, _StatusButton

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(scope="module", autouse=True)
def qapp():
    # A QObject's signal/slot machinery needs a QCoreApplication instance to
    # exist, even though these tests never call exec().
    app = QApplication.instance() or QApplication([])
    yield app


class FakeRecorder:
    def __init__(self, wav_bytes: bytes = b"fake-wav") -> None:
        self.started = False
        self.stopped = False
        self._wav_bytes = wav_bytes

    def start(self) -> None:
        self.started = True

    def stop(self) -> bytes:
        self.stopped = True
        return self._wav_bytes


class BrokenRecorder:
    def start(self) -> None:
        raise OSError("device busy")


class FakeSTT:
    def __init__(self, transcript: str = "what does Relay use for retries?") -> None:
        self.transcript = transcript

    def transcribe(self, wav_bytes: bytes) -> str:
        return self.transcript


class FakeCore:
    def __init__(self, reply: Reply) -> None:
        self._reply = reply
        self.calls: list[str] = []

    def handle_text(self, text: str) -> Reply:
        self.calls.append(text)
        return self._reply


class FakeTTS:
    def __init__(self, should_fail: bool = False) -> None:
        self.should_fail = should_fail
        self.calls = 0

    def synthesize(self, text: str) -> bytes:
        self.calls += 1
        if self.should_fail:
            raise ConnectionError("network down")
        return b"fake-audio"


def _collect_states(worker: VoiceWorker) -> list[tuple[str, str]]:
    events: list[tuple[str, str]] = []
    worker.state_changed.connect(lambda state, detail: events.append((state, detail)))
    return events


def test_normal_turn_transitions_listening_processing_speaking_idle():
    recorder = FakeRecorder()
    core = FakeCore(Reply(turn_id="t1", text="Exponential backoff."))
    tts = FakeTTS()
    worker = VoiceWorker(core, tts, FakeSTT(), recorder_factory=lambda: recorder, player=lambda audio: None)

    states = _collect_states(worker)
    transcripts: list[str] = []
    worker.transcript_ready.connect(transcripts.append)
    replies: list[str] = []
    worker.reply_ready.connect(replies.append)

    worker.start_listening()
    worker.stop_and_process()

    assert [s for s, _ in states] == [
        State.LISTENING.value,
        State.PROCESSING.value,
        State.SPEAKING.value,
        State.IDLE.value,
    ]
    assert all(detail == "" for _, detail in states)
    assert transcripts == ["what does Relay use for retries?"]
    assert replies == ["Exponential backoff."]
    assert recorder.started and recorder.stopped
    assert core.calls == ["what does Relay use for retries?"]


def test_failed_recording_start_goes_straight_to_error_with_the_real_message():
    worker = VoiceWorker(
        FakeCore(Reply(turn_id="t1", text="")),
        FakeTTS(),
        FakeSTT(),
        recorder_factory=BrokenRecorder,
        player=lambda audio: None,
    )
    states = _collect_states(worker)

    worker.start_listening()

    assert states == [(State.ERROR.value, "device busy")]


def test_unusable_transcript_returns_to_idle_without_calling_core():
    recorder = FakeRecorder()
    core = FakeCore(Reply(turn_id="t1", text="should not be reached"))
    worker = VoiceWorker(core, FakeTTS(), FakeSTT(transcript="   "), recorder_factory=lambda: recorder, player=lambda a: None)

    states = _collect_states(worker)
    transcripts: list[str] = []
    worker.transcript_ready.connect(transcripts.append)

    worker.start_listening()
    worker.stop_and_process()

    assert [s for s, _ in states] == [State.LISTENING.value, State.PROCESSING.value, State.IDLE.value]
    assert transcripts == [""]
    assert core.calls == []


def test_core_error_surfaces_the_real_error_message():
    recorder = FakeRecorder()
    core = FakeCore(Reply(turn_id="t1", text="", error="429 RESOURCE_EXHAUSTED"))
    worker = VoiceWorker(core, FakeTTS(), FakeSTT(), recorder_factory=lambda: recorder, player=lambda a: None)

    states = _collect_states(worker)
    replies: list[str] = []
    worker.reply_ready.connect(replies.append)

    worker.start_listening()
    worker.stop_and_process()

    assert states[-1] == (State.ERROR.value, "429 RESOURCE_EXHAUSTED")
    assert replies == []  # never reached — the error short-circuits before reply_ready


def test_tts_failure_surfaces_as_error_but_reply_text_was_already_delivered():
    recorder = FakeRecorder()
    core = FakeCore(Reply(turn_id="t1", text="Exponential backoff."))
    tts = FakeTTS(should_fail=True)
    worker = VoiceWorker(core, tts, FakeSTT(), recorder_factory=lambda: recorder, player=lambda a: None)

    states = _collect_states(worker)
    replies: list[str] = []
    worker.reply_ready.connect(replies.append)

    worker.start_listening()
    worker.stop_and_process()

    # The answer was delivered before TTS was ever attempted...
    assert replies == ["Exponential backoff."]
    # ...so a TTS failure afterward is real, but must not erase that answer.
    assert states[-1] == (State.ERROR.value, "voice unavailable: network down")


def test_extra_stop_call_without_an_active_recording_is_a_noop():
    worker = VoiceWorker(FakeCore(Reply(turn_id="t1", text="x")), FakeTTS(), FakeSTT(), player=lambda a: None)
    states = _collect_states(worker)

    worker.stop_and_process()

    assert states == []


def test_a_hanging_tts_call_times_out_instead_of_blocking_forever(monkeypatch):
    # A real hang past even SarvamTTS's own 60s httpx timeout was observed
    # in practice — this proves the bound actually kicks in, using a tiny
    # timeout so the test itself stays fast rather than actually waiting.
    monkeypatch.setattr(widget_module, "TTS_TIMEOUT_S", 0.05)

    class HangingTTS:
        def synthesize(self, text: str) -> bytes:
            time.sleep(5)
            return b"never gets here in time"

    recorder = FakeRecorder()
    core = FakeCore(Reply(turn_id="t1", text="Exponential backoff."))
    worker = VoiceWorker(core, HangingTTS(), FakeSTT(), recorder_factory=lambda: recorder, player=lambda a: None)
    states = _collect_states(worker)

    worker.start_listening()
    worker.stop_and_process()

    assert states[-1][0] == State.ERROR.value
    assert "voice unavailable" in states[-1][1]


def test_status_button_paints_without_raising_in_every_state():
    # A real crash was caused by a Qt API misuse (QPen's constructor doesn't
    # accept a "cap" kwarg in PySide6) that only surfaced inside paintEvent —
    # invisible to the pure state-machine tests above, which never render
    # anything. Qt swallows exceptions raised inside paintEvent overrides
    # (logs to stderr, doesn't propagate), so calling _paint() directly
    # matters here — going through paintEvent()/repaint() would have passed
    # even with the broken code, exactly how this bug slipped through once.
    from PySide6.QtGui import QPainter, QPixmap

    button = _StatusButton()
    pixmap = QPixmap(_StatusButton.SIZE, _StatusButton.SIZE)
    for state in State:
        button.set_state(state)
        painter = QPainter(pixmap)
        button._paint(painter)
        painter.end()
