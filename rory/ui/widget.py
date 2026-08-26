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

import numpy as np
from PySide6.QtCore import QObject, QSocketNotifier, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtSvg import QSvgRenderer
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


# The status button's color per state. IDLE is the requested light pink;
# PROCESSING stays pink too since the spinner (not a color change) is what
# communicates "loading" — see _StatusButton.
_STATE_COLORS = {
    State.IDLE: "#f6a9c9",
    State.LISTENING: "#e74c3c",
    State.PROCESSING: "#f6a9c9",
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
_RENDER_RESOLUTION = 1024  # matches the SVG's own viewBox


def _cropped_artwork(max_dimension: int) -> tuple[QPixmap, float]:
    """The SVG canvas has a lot of transparent margin around the actual
    drawing (confirmed: a 1024x1024 canvas holding artwork occupying only a
    few hundred pixels of it) — without cropping, the widget shows a tiny
    image inside a mostly-empty frame. This finds the bounding box of the
    non-transparent pixels and crops to it, so the frame hugs the actual
    artwork. Returns the scaled pixmap and its aspect ratio (width/height),
    so the caller can size a frame that matches instead of forcing a square.
    """
    renderer = QSvgRenderer(str(SVG_PATH))
    full = QImage(_RENDER_RESOLUTION, _RENDER_RESOLUTION, QImage.Format.Format_ARGB32)
    full.fill(0)
    painter = QPainter(full)
    renderer.render(painter)
    painter.end()

    buf = full.constBits()
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(full.height(), full.width(), 4)
    alpha = arr[:, :, 3]
    ys, xs = np.where(alpha > 10)

    if len(xs) == 0:
        cropped = full
    else:
        pad = 8
        x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad, full.width())
        y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad, full.height())
        cropped = full.copy(x0, y0, x1 - x0, y1 - y0)

    aspect = cropped.width() / cropped.height()
    pixmap = QPixmap.fromImage(cropped).scaled(
        max_dimension,
        max_dimension,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    return pixmap, aspect


class _StatusButton(QWidget):
    """The click target: a small round badge overlaid on the cat's corner.
    Its color and icon are the state indicator now (the border rectangle
    this replaced didn't fit "just the cat, no square" well). Drawn, not
    loaded — no icon assets needed for a handful of simple glyphs."""

    SIZE = 32
    _SPIN_INTERVAL_MS = 60
    _SPIN_STEP_DEG = 30

    clicked = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._state = State.IDLE
        self._spin_angle = 0
        self._spin_timer = QTimer(self)
        self._spin_timer.setInterval(self._SPIN_INTERVAL_MS)
        self._spin_timer.timeout.connect(self._advance_spin)

    def set_state(self, state: State) -> None:
        self._state = state
        if state in (State.PROCESSING, State.SPEAKING):
            if not self._spin_timer.isActive():
                self._spin_timer.start()
        else:
            self._spin_timer.stop()
        self.update()

    def _advance_spin(self) -> None:
        self._spin_angle = (self._spin_angle + self._SPIN_STEP_DEG) % 360
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802 — Qt override
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()

    def paintEvent(self, event) -> None:  # noqa: N802 — Qt override
        # Qt swallows exceptions raised inside a paintEvent override (they
        # print to stderr rather than propagate) — real drawing logic lives
        # in _paint() below instead so tests calling it directly actually
        # see a raised exception rather than silence. This gap is exactly
        # how a real QPen API-misuse crash slipped past every prior test.
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._paint(painter)

    def _paint(self, painter: QPainter) -> None:
        painter.setBrush(QColor(_STATE_COLORS[self._state]))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(1, 1, self.SIZE - 2, self.SIZE - 2)

        if self._state in (State.PROCESSING, State.SPEAKING):
            # A loading spinner: a short rotating arc.
            spin_pen = QPen(QColor("white"), 3)
            spin_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(spin_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            rect = self.rect().adjusted(8, 8, -8, -8)
            painter.drawArc(rect, self._spin_angle * 16, 110 * 16)
        elif self._state == State.ERROR:
            font = painter.font()
            font.setBold(True)
            font.setPointSize(14)
            painter.setFont(font)
            painter.setPen(QColor("white"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "!")
        else:
            # A simple mic glyph: a rounded capsule + a small stand line.
            # Solid white when idle; the button's red fill during LISTENING
            # is what actually signals "recording."
            painter.setBrush(QColor("white"))
            painter.setPen(Qt.PenStyle.NoPen)
            capsule = self.rect().adjusted(12, 6, -12, -15)
            painter.drawRoundedRect(capsule, 4, 4)
            painter.setPen(QPen(QColor("white"), 2))
            mid_x = self.SIZE // 2
            painter.drawLine(mid_x, self.SIZE - 11, mid_x, self.SIZE - 7)
            painter.drawLine(mid_x - 4, self.SIZE - 7, mid_x + 4, self.SIZE - 7)


class RoryStickyWidget(QWidget):
    """The always-visible desktop widget: the user's own SVG artwork,
    unmodified, with a small round status button overlaid on its corner as
    the click target, plus a small text panel for the transcript/reply.
    Stays open for Rory's whole run rather than hiding after a turn.

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

        pixmap, aspect = _cropped_artwork(IMAGE_SIZE - 8)
        # Frame hugs the artwork's actual proportions (portrait, in this
        # image) rather than forcing a square with visible empty margin.
        if aspect >= 1:
            frame_w, frame_h = IMAGE_SIZE, round(IMAGE_SIZE / aspect)
        else:
            frame_w, frame_h = round(IMAGE_SIZE * aspect), IMAGE_SIZE

        self._image_frame = QFrame()
        self._image_frame.setFixedSize(frame_w, frame_h)
        self._image_frame.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._image_frame.setAutoFillBackground(False)
        self._image_frame.setStyleSheet("background: transparent; border: none;")

        image_label = QLabel(self._image_frame)
        image_label.setPixmap(pixmap)
        image_label.setFixedSize(frame_w, frame_h)
        image_label.move(0, 0)
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_label.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        image_label.setAutoFillBackground(False)
        image_label.setStyleSheet("background: transparent;")

        # Overlaid as a badge on the cat's bottom-right corner, overlapping
        # slightly rather than sitting fully outside the artwork.
        self._status_button = _StatusButton(self._image_frame)
        self._status_button.move(
            frame_w - _StatusButton.SIZE + 4,
            frame_h - _StatusButton.SIZE + 4,
        )
        self._status_button.raise_()
        self._status_button.clicked.connect(self.toggled)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._image_frame, alignment=Qt.AlignmentFlag.AlignHCenter)

        self._position_default()

    def _position_default(self) -> None:
        # Best-effort only: under Wayland/Hyprland this move() was confirmed
        # to have no effect during development (position stayed wherever
        # Hyprland's tiling/floating layout put it regardless). Kept because
        # it's harmless and does work on X11 — see README.md for what
        # actually controls placement here.
        self.adjustSize()
        screen = QApplication.primaryScreen().availableGeometry()
        self.move(screen.right() - self.width() - 40, screen.bottom() - self.height() - 40)

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
        # detail (the actual error text, when there is one) still surfaces
        # via the tooltip — not shown as always-visible text on the widget,
        # per request, but not silently discarded either.
        self._status_button.set_state(state)
        self.setToolTip(state.value + (f": {detail}" if detail else ""))


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
        # transcript_ready/reply_ready are intentionally not connected to
        # anything here — the widget shows state only (button color/icon +
        # tooltip), not the transcript or reply text, per request. The
        # worker still emits them regardless; nothing about VoiceWorker
        # changed to accommodate this, only what the UI does with them.

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
