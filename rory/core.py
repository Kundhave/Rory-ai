"""RoryCore facade. handle_text(str) -> Reply is the entire system boundary
that adapters (CLI, voice, GUI) talk to. Nothing audio- or UI-shaped lives here."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

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

            llm_start = time.monotonic()
            response = self._llm.generate(self._history, system=build_system_prompt())
            trace.event(
                "llm_generate",
                (time.monotonic() - llm_start) * 1000,
                prompt_chars=trace.text_or_len(text),
                completion_chars=len(response.text),
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            )

            self._history.append({"role": "assistant", "content": response.text})
            self._trim_history()

            trace.event("turn_complete", (time.monotonic() - turn_start) * 1000)
            return Reply(turn_id=turn_id, text=response.text)

        except Exception as exc:
            trace.event("turn_error", (time.monotonic() - turn_start) * 1000, error=str(exc))
            return Reply(turn_id=turn_id, text="", error=str(exc))

    def _trim_history(self) -> None:
        if len(self._history) > MAX_HISTORY_MESSAGES:
            self._history = self._history[-MAX_HISTORY_MESSAGES:]
