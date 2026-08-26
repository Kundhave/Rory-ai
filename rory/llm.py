"""LLM Protocol and the Gemini implementation.

The protocol normalises exactly three things every provider gives us: text,
tool calls, and token usage.

Tool schemas go in as plain JSON-schema dicts; provider-specific packaging
stays inside the implementation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol, TypedDict

import httpx
from google import genai
from google.genai import errors, types

# One retry, not a retry storm. A 429 or a 5xx is usually transient, and the
# free tier's per-minute quota in particular clears quickly — but a second
# failure means something is actually wrong, and making the user wait through
# a third attempt buys nothing over telling them honestly.
LLM_ATTEMPTS = 2
LLM_BACKOFF_S = 2.0


def _is_retryable(exc: Exception) -> bool:
    """A connection error means the network is down — retrying just makes the
    user wait longer for the same failure, so it fails fast. A timeout or a
    server-side error is worth exactly one more try."""
    if isinstance(exc, httpx.ConnectError):
        return False
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, errors.ServerError):
        return True
    if isinstance(exc, errors.ClientError):
        return exc.code == 429
    return False


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
    def generate(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        ...


_ROLE_MAP = {"user": "user", "assistant": "model"}


def _declaration(schema: dict) -> dict:
    # Gemini rejects a parameters block with no properties, so omit it entirely
    # for zero-argument tools.
    declaration = {"name": schema["name"], "description": schema["description"]}
    if schema["parameters"]["properties"]:
        declaration["parameters"] = schema["parameters"]
    return declaration


class GeminiLLM:
    def __init__(self, api_key: str, model: str) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def generate(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        for _ in range(LLM_ATTEMPTS - 1):
            try:
                return self._generate_once(messages, system, tools)
            except Exception as exc:
                if not _is_retryable(exc):
                    raise
                time.sleep(LLM_BACKOFF_S)
        # Final attempt: whatever it raises is what the caller sees, which
        # RoryCore turns into an honest Reply(error=...).
        return self._generate_once(messages, system, tools)

    def _generate_once(
        self,
        messages: list[Message],
        system: str | None,
        tools: list[dict] | None,
    ) -> LLMResponse:
        contents = [
            types.Content(role=_ROLE_MAP[m["role"]], parts=[types.Part.from_text(text=m["content"])])
            for m in messages
        ]
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(function_declarations=[_declaration(t) for t in tools])] if tools else None,
            # We dispatch tools ourselves through the registry; the SDK must
            # never invoke anything on its own.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        response = self._client.models.generate_content(
            model=self._model, contents=contents, config=config
        )
        parts = response.candidates[0].content.parts or [] if response.candidates else []
        usage = response.usage_metadata
        return LLMResponse(
            text="".join(part.text for part in parts if part.text),
            tool_calls=[
                ToolCall(name=part.function_call.name, arguments=dict(part.function_call.args or {}))
                for part in parts
                if part.function_call
            ],
            usage=TokenUsage(
                prompt_tokens=usage.prompt_token_count or 0,
                completion_tokens=usage.candidates_token_count or 0,
                total_tokens=usage.total_token_count or 0,
            ),
        )
