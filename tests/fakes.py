"""Deterministic, offline fakes for anything touching the agent loop."""
from __future__ import annotations

from rory.llm import LLMResponse, Message, TokenUsage


class FakeLLM:
    """Replays scripted responses, or echoes the last user message if none are left."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses or [])
        self.calls: list[list[Message]] = []

    def generate(self, messages: list[Message], system: str | None = None) -> LLMResponse:
        self.calls.append(list(messages))
        text = self._responses.pop(0) if self._responses else messages[-1]["content"]
        return LLMResponse(text=text, tool_calls=[], usage=TokenUsage(1, 1, 2))
