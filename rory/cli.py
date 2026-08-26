"""Text REPL — the primary development interface for Rory's core.

Runs the same RoryCore.handle_text loop that voice and GUI adapters will use,
with no audio devices or Qt event loop involved. Type 'exit' or Ctrl-D to quit.
Press Enter on an empty line to start recording; press Enter again to stop.
"""
from __future__ import annotations

import time
import uuid

from rory.config import settings
from rory.core import RoryCore, Reply
from rory.llm import GeminiLLM
from rory.trace import Trace
from rory.voice.audio import Recorder, play
from rory.voice.stt import STT, SarvamSTT, is_usable
from rory.voice.tts import TTS, build_tts


def main() -> None:
    llm = GeminiLLM(api_key=settings.gemini_api_key, model=settings.gemini_model)
    core = RoryCore(llm)
    tts = build_tts(settings)
    stt = SarvamSTT(api_key=settings.sarvam_api_key)

    print("Rory CLI. Type 'exit' to quit. Press Enter on an empty line to talk.")
    while True:
        try:
            text = input("> ").strip()
        except EOFError:
            print()
            break

        if text.lower() in {"exit", "quit"}:
            break

        if not text:
            # Constructed on first use, not at startup, so a machine with no
            # working input device can still run the CLI in text mode. A busy
            # or missing microphone must print a sentence, not a stack trace.
            try:
                recorder = Recorder()
                print("[recording... press Enter to stop]")
                recorder.start()
                input()
                wav_bytes = recorder.stop()
            except Exception as exc:
                print(f"[microphone unavailable: {exc}]")
                continue
            run_voice_turn(stt, core, tts, wav_bytes)
            continue

        reply = core.handle_text(text)
        if reply.error:
            print(f"[error] {reply.error}")
            continue

        # Text is printed before speaking so the answer is never lost even if
        # TTS fails.
        print(reply.text)
        speak(tts, reply)


def run_voice_turn(stt: STT, core: RoryCore, tts: TTS, wav_bytes: bytes) -> None:
    # This turn has no turn_id yet — RoryCore mints its own once handle_text
    # is called — so the STT stage gets its own id here rather than one
    # shared with the LLM/tool/TTS stages that follow.
    trace = Trace(str(uuid.uuid4()))
    started = time.monotonic()
    try:
        transcript = stt.transcribe(wav_bytes)
    except Exception as exc:
        # Speech recognition is a network call; when it fails the user gets a
        # sentence explaining why, never a traceback.
        trace.event("stt_error", (time.monotonic() - started) * 1000, error=str(exc))
        print(f"[couldn't reach speech recognition: {exc}]")
        return
    trace.event("stt", (time.monotonic() - started) * 1000, chars=trace.text_or_len(transcript))

    # Speech recognition mishearing a proper noun is the most common failure
    # in this system. Showing the raw transcript is the cheapest possible
    # fix: the user sees immediately if "Relay" became "railay" instead of
    # silently getting an answer to the wrong question.
    if not is_usable(transcript):
        print("[couldn't hear that clearly — try again]")
        return
    print(f"heard: {transcript}")

    reply = core.handle_text(transcript)
    if reply.error:
        print(f"[error] {reply.error}")
        return

    print(reply.text)
    speak(tts, reply)


def speak(tts: TTS, reply: Reply) -> None:
    if not reply.text.strip():
        return

    trace = Trace(reply.turn_id)
    started = time.monotonic()
    try:
        audio = tts.synthesize(reply.text)
        trace.event(
            "tts",
            (time.monotonic() - started) * 1000,
            chars=len(reply.text),
            engine=getattr(tts, "last_engine", None),
        )
        play(audio)
    except Exception as exc:
        trace.event("tts_error", (time.monotonic() - started) * 1000, error=str(exc))
        print(f"[voice unavailable: {exc}]")


if __name__ == "__main__":
    main()
