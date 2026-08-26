"""LLM Protocol and the Gemini implementation.

The protocol normalises exactly three things every provider gives us: text,
tool calls, and token usage.

Tool schemas go in as plain JSON-schema dicts; provider-specific packaging
stays inside the implementation.
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
