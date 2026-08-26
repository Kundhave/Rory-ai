"""LLM Protocol and the Gemini implementation.

The protocol normalises exactly three things every provider gives us: text,
tool calls, and token usage. Tool calls are unused until Feature 2, but the
shape is defined now so the agent loop can be built against a stable
interface without a breaking change later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, TypedDict

from google import genai
from google.genai import types


class Message(TypedDict):
    role: str  # "user" or "assistant"
    content: str


@dataclass
class ToolCall:
    name: str
    arguments: dict


@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=lambda: TokenUsage(0, 0, 0))


class LLM(Protocol):
    def generate(self, messages: list[Message], system: str | None = None) -> LLMResponse:
        ...


_ROLE_MAP = {"user": "user", "assistant": "model"}


class GeminiLLM:
    def __init__(self, api_key: str, model: str) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def generate(self, messages: list[Message], system: str | None = None) -> LLMResponse:
        contents = [
            types.Content(role=_ROLE_MAP[m["role"]], parts=[types.Part.from_text(text=m["content"])])
            for m in messages
        ]
        config = types.GenerateContentConfig(system_instruction=system) if system else None
        response = self._client.models.generate_content(
            model=self._model, contents=contents, config=config
        )
        usage = response.usage_metadata
        return LLMResponse(
            text=response.text or "",
            tool_calls=[],
            usage=TokenUsage(
                prompt_tokens=usage.prompt_token_count or 0,
                completion_tokens=usage.candidates_token_count or 0,
                total_tokens=usage.total_token_count or 0,
            ),
        )
