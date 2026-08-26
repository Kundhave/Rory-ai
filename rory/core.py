"""RoryCore facade. handle_text(str) -> Reply is the entire system boundary
that adapters (CLI, voice, GUI) talk to. Nothing audio- or UI-shaped lives here."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from rory.agent import loop
from rory.agent.prompts import build_system_prompt
from rory.llm import LLM, Message
from rory.trace import Trace

# Sliding window over turns. Bounded memory, O(1) per turn. Failure mode: once
# the conversation exceeds this many messages, the oldest ones silently drop
# out of context — a fact mentioned early on can stop being "remembered" with
# no error or warning. Acceptable for V1; a real summarising memory is a later
# feature, not a fix to bolt on here.
MAX_HISTORY_MESSAGES = 20


@dataclass
class Reply:
    turn_id: str
    text: str
    error: str | None = None


class RoryCore:
    def __init__(self, llm: LLM) -> None:
        self._llm = llm
        self._history: list[Message] = []

    def handle_text(self, text: str) -> Reply:
        turn_id = str(uuid.uuid4())
        trace = Trace(turn_id)
        turn_start = time.monotonic()

        try:
            self._history.append({"role": "user", "content": text})
            self._trim_history()

            trace.event("turn_start", 0.0, prompt_chars=trace.text_or_len(text))

            # The loop appends its own tool-call and tool-result messages, so
            # the next turn sees what was actually run.
            answer = loop.run(self._llm, self._history, build_system_prompt(), trace)

            self._history.append({"role": "assistant", "content": answer})
            self._trim_history()

            trace.event("turn_complete", (time.monotonic() - turn_start) * 1000)
            return Reply(turn_id=turn_id, text=answer)

        except Exception as exc:
            trace.event("turn_error", (time.monotonic() - turn_start) * 1000, error=str(exc))
            return Reply(turn_id=turn_id, text="", error=str(exc))

    def _trim_history(self) -> None:
        if len(self._history) > MAX_HISTORY_MESSAGES:
            self._history = self._history[-MAX_HISTORY_MESSAGES:]
