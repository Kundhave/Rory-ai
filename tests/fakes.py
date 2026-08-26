"""Deterministic, offline fakes for anything touching the agent loop."""
from __future__ import annotations

from rory.llm import LLMResponse, Message, TokenUsage, ToolCall


def calls(*named_args: tuple[str, dict]) -> LLMResponse:
    """A scripted turn where the model asks for tools instead of answering."""
    return LLMResponse(
        text="",
        tool_calls=[ToolCall(name=name, arguments=args) for name, args in named_args],
        usage=TokenUsage(1, 1, 2),
    )


def says(text: str) -> LLMResponse:
    return LLMResponse(text=text, tool_calls=[], usage=TokenUsage(1, 1, 2))


class FakeLLM:
    """Replays a scripted list of responses. A bare string is shorthand for
    `says(...)`. Once the script runs out it echoes the last message, so a test
    that over-runs its script fails on the assertion rather than an IndexError."""

    def __init__(self, responses: list[LLMResponse | str] | None = None) -> None:
        self._responses = [says(r) if isinstance(r, str) else r for r in (responses or [])]
        self.calls: list[list[Message]] = []
        self.tools_seen: list[list[dict]] = []

    def generate(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        self.tools_seen.append(tools or [])
        if self._responses:
            return self._responses.pop(0)
        return says(messages[-1]["content"])
