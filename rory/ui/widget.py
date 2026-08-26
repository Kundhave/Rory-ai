"""PySide6 desktop widget — a third adapter (alongside cli.py) over
RoryCore.handle_text. Nothing here reaches into core/agent/tools/rag/voice
beyond calling their existing public interfaces.

Threading model: the Qt main thread owns painting and click handling only.
Recording, STT, RoryCore.handle_text, and TTS all run on VoiceWorker, which
lives on its own QThread. The two directions cross threads exclusively
through Qt signals (request_start/request_stop going in, state_changed/
transcript_ready/reply_ready coming back) — Qt marshals a signal emission
into a thread-safe queued call automatically whenever the emitting and
receiving QObjects live on different threads. Nothing here ever calls a
worker method directly, and the worker never touches a widget.
"""
from __future__ import annotations

import os
import socket
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path

from PySide6.QtCore import QObject, QSocketNotifier, Qt, QThread, Signal, Slot
from PySide6.QtGui import QAction
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtWidgets import QApplication, QFrame, QLabel, QMenu, QVBoxLayout, QWidget

from rory.config import settings
from rory.core import RoryCore
from rory.llm import GeminiLLM
from rory.trace import Trace
from rory.voice.audio import Recorder, play
from rory.voice.stt import SarvamSTT, is_usable
from rory.voice.tts import build_tts

# An external trigger (e.g. a compositor keybind) writes any datagram here to
# toggle listening. Global hotkey libraries mostly don't work under Wayland,
# so Rory doesn't try to grab one itself — it just listens.
SOCKET_PATH = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "rory.sock"

# A real hang past even SarvamTTS's own 60s httpx timeout was observed
# during development — this is the hard ceiling that guarantees a turn
# always resolves to IDLE or ERROR rather than freezing the worker (and the
# whole app, since a stuck worker also blocks Quit — see RoryWidget.quit).
TTS_TIMEOUT_S = 45.0


class State(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ERROR = "error"


_STATE_COLORS = {
    State.IDLE: "#7f8c8d",
    State.LISTENING: "#e74c3c",
    State.PROCESSING: "#f39c12",
    State.SPEAKING: "#2ecc71",
    State.ERROR: "#c0392b",
}


class VoiceWorker(QObject):
    """Owns everything that touches a device or the network. Lives on its
    own QThread; every result crosses back to the main thread as a signal."""

    state_changed = Signal(str, str)  # State.value, detail (e.g. the actual error text)
    transcript_ready = Signal(str)
    reply_ready = Signal(str)

    def __init__(self, core: RoryCore, tts, stt, recorder_factory=Recorder, player=play) -> None:
        super().__init__()
        self._core = core
        self._tts = tts
        self._stt = stt
        self._recorder_factory = recorder_factory
        self._player = player
        self._recorder = None

    @Slot()
    def start_listening(self) -> None:
        try:
            self._recorder = self._recorder_factory()
            self._recorder.start()
            self.state_changed.emit(State.LISTENING.value, "")
        except Exception as exc:
            self.state_changed.emit(State.ERROR.value, str(exc))

    @Slot()
    def stop_and_process(self) -> None:
        if self._recorder is None:
            return
        # No turn_id exists yet — RoryCore mints its own inside handle_text —
        # so this stage gets its own, same seam as cli.py's run_voice_turn.
        trace = Trace(str(uuid.uuid4()))
        try:
            wav_bytes = self._recorder.stop()
            self._recorder = None
            self.state_changed.emit(State.PROCESSING.value, "")

            started = time.monotonic()
            transcript = self._stt.transcribe(wav_bytes)
            trace.event("stt", (time.monotonic() - started) * 1000, chars=trace.text_or_len(transcript))

            if not is_usable(transcript):
                self.transcript_ready.emit("")
                self.state_changed.emit(State.IDLE.value, "")
                return
            self.transcript_ready.emit(transcript)

            reply = self._core.handle_text(transcript)
            if reply.error:
                self.state_changed.emit(State.ERROR.value, reply.error)
                return
            self.reply_ready.emit(reply.text)

            if reply.text.strip():
                self.state_changed.emit(State.SPEAKING.value, "")
                try:
                    tts_started = time.monotonic()
                    # Bounded regardless of what the TTS engine itself does
                    # internally. A real hang was observed here in practice —
                    # the process outlived even SarvamTTS's own 60s httpx
                    # timeout and had to be force-killed — so this call must
                    # never be allowed to block the worker (and therefore
                    # never appear to freeze the whole app) beyond a fixed
                    # ceiling. A timed-out call's underlying thread is
                    # abandoned, not killed (Python can't force-stop a
                    # thread) — an acceptable leak next to freezing the UI.
                    # shutdown(wait=False): waiting here would block on the
                    # very thread this timeout exists to stop waiting for.
                    pool = ThreadPoolExecutor(max_workers=1)
                    try:
                        audio = pool.submit(self._tts.synthesize, reply.text).result(timeout=TTS_TIMEOUT_S)
                    finally:
                        pool.shutdown(wait=False)
                    trace.event("tts", (time.monotonic() - tts_started) * 1000, chars=len(reply.text))
                    self._player(audio)
                except Exception as exc:
                    # The reply text already went out via reply_ready — a
                    # TTS failure is real and worth surfacing, but must not
                    # hide the answer already delivered.
                    self.state_changed.emit(State.ERROR.value, f"voice unavailable: {exc}")
                    return

            self.state_changed.emit(State.IDLE.value, "")
        except Exception as exc:
            self.state_changed.emit(State.ERROR.value, str(exc))


SVG_PATH = Path("assets/images/widget.svg")
IMAGE_SIZE = 140


class RoryStickyWidget(QWidget):
    """The always-visible desktop widget: the user's own SVG artwork,
    unmodified, inside a state-colored border, plus a small text panel for
    the transcript/reply. Stays open for Rory's whole run rather than
    hiding after a turn.

    Wayland gives no way for an application to truly pin itself always-on-
    top or embed into the desktop layer — see docs/DECISIONS.md ADR-012 and
    its follow-up. This window stays open indefinitely; "always on top" is
    left to a compositor window rule the user adds themselves (README.md
    has the exact Hyprland line, matched by this window's title, "Rory").
    """

    toggled = Signal()
    quit_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Rory")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedWidth(IMAGE_SIZE)

        self._image_frame = QFrame()
        self._image_frame.setFixedSize(IMAGE_SIZE, IMAGE_SIZE)
        self._set_border(State.IDLE)

        svg = QSvgWidget(str(SVG_PATH), self._image_frame)
        svg.setFixedSize(IMAGE_SIZE - 8, IMAGE_SIZE - 8)
        svg.move(4, 4)

        self._transcript = QLabel("")
        self._transcript.setWordWrap(True)
        self._transcript.setMaximumWidth(IMAGE_SIZE)
        self._transcript.setStyleSheet(_TEXT_PANEL_STYLE)
        self._transcript.hide()

        self._reply = QLabel("")
        self._reply.setWordWrap(True)
        self._reply.setMaximumWidth(IMAGE_SIZE)
        self._reply.setStyleSheet(_TEXT_PANEL_STYLE)
        self._reply.hide()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 8)
        layout.addWidget(self._image_frame, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self._transcript)
        layout.addWidget(self._reply)

        self._position_default()

    def _position_default(self) -> None:
        # Best-effort only: under Wayland/Hyprland this move() was confirmed
        # to have no effect during development (position stayed wherever
        # Hyprland's tiling/floating layout put it regardless). Kept because
        # it's harmless and does work on X11 — see README.md for what
        # actually controls placement here.
        self.adjustSize()
        screen = QApplication.primaryScreen().availableGeometry()
        self.move(screen.right() - self.width() - 40, screen.top() + 40)

    def _set_border(self, state: State) -> None:
        self._image_frame.setStyleSheet(
            f"border: 4px solid {_STATE_COLORS[state]}; border-radius: 16px; background: transparent;"
        )

    def mousePressEvent(self, event) -> None:  # noqa: N802 — Qt override
        if event.button() == Qt.MouseButton.LeftButton:
            self.toggled.emit()

    def contextMenuEvent(self, event) -> None:  # noqa: N802 — Qt override
        menu = QMenu(self)
        quit_action = QAction("Quit", menu)
        quit_action.triggered.connect(self.quit_requested.emit)
        menu.addAction(quit_action)
        menu.exec(event.globalPos())

    def closeEvent(self, event) -> None:  # noqa: N802 — Qt override
        self.quit_requested.emit()
        event.accept()

    def set_state(self, state: State, detail: str) -> None:
        self._set_border(state)
        self.setToolTip(state.value + (f": {detail}" if detail else ""))

    def set_transcript(self, text: str) -> None:
        self._transcript.setText(f"heard: {text}" if text else "couldn't hear that clearly — try again")
        self._transcript.show()

    def set_reply(self, text: str) -> None:
        self._reply.setText(text)
        self._reply.show()


_TEXT_PANEL_STYLE = (
    "color: white; background: rgba(0, 0, 0, 170); padding: 6px; "
    "border-radius: 6px; margin-top: 4px;"
)


def bind_trigger_socket(on_trigger) -> tuple[socket.socket, QSocketNotifier]:
    """Any datagram received on SOCKET_PATH calls on_trigger() — same effect
    as a tray click. Read on the main thread's Qt event loop via
    QSocketNotifier, not polled, so it costs nothing when idle."""
    if SOCKET_PATH.exists():
        SOCKET_PATH.unlink()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    sock.bind(str(SOCKET_PATH))
    sock.setblocking(False)

    notifier = QSocketNotifier(sock.fileno(), QSocketNotifier.Type.Read)

    def _on_activated(_fd: int) -> None:
        try:
            sock.recv(64)
        except OSError:
            return
        on_trigger()

    notifier.activated.connect(_on_activated)
    return sock, notifier


def unbind_trigger_socket(sock: socket.socket) -> None:
    sock.close()
    if SOCKET_PATH.exists():
        SOCKET_PATH.unlink()


class RoryWidget(QObject):
    request_start = Signal()
    request_stop = Signal()

    def __init__(self) -> None:
        super().__init__()
        llm = GeminiLLM(api_key=settings.gemini_api_key, model=settings.gemini_model)
        core = RoryCore(llm)
        tts = build_tts(settings)
        stt = SarvamSTT(api_key=settings.sarvam_api_key)

        self._state = State.IDLE
        self.sticky = RoryStickyWidget()
        self.sticky.toggled.connect(self.toggle)
        self.sticky.quit_requested.connect(self.quit)

        self._thread = QThread()
        self.worker = VoiceWorker(core, tts, stt)
        self.worker.moveToThread(self._thread)
        self._thread.start()

        self.request_start.connect(self.worker.start_listening)
        self.request_stop.connect(self.worker.stop_and_process)
        self.worker.state_changed.connect(self._on_state_changed)
        self.worker.transcript_ready.connect(self.sticky.set_transcript)
        self.worker.reply_ready.connect(self.sticky.set_reply)

        self._trigger_socket, self._trigger_notifier = bind_trigger_socket(self.toggle)

        self.sticky.show()

    def toggle(self) -> None:
        if self._state in (State.IDLE, State.ERROR):
            self.request_start.emit()
        elif self._state == State.LISTENING:
            self.request_stop.emit()
        # PROCESSING/SPEAKING: a turn is already running, ignore extra clicks.

    def _on_state_changed(self, state_value: str, detail: str) -> None:
        state = State(state_value)
        self._state = state
        self.sticky.set_state(state, detail)

    def quit(self) -> None:
        # self._thread.wait() here would block the main thread — and
        # therefore the Wayland event loop — for as long as the worker's
        # currently running slot takes to return. That's exactly the freeze
        # a stuck TTS call caused in practice (see TTS_TIMEOUT_S). Quitting
        # must stay non-blocking on the main thread regardless of whether
        # the worker is mid-turn.
        unbind_trigger_socket(self._trigger_socket)
        self._thread.finished.connect(QApplication.instance().quit)
        self._thread.quit()


def main() -> None:
    app = QApplication(sys.argv)
    widget = RoryWidget()  # noqa: F841 — must stay alive for the app's lifetime
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
