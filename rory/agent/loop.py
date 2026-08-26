"""The tool-calling loop: generate -> dispatch -> feed results back -> repeat.

Tool exchanges are written into the ordinary message history as text rather
than a provider-native tool-message type. That keeps `Message` a plain
{role, content} pair, so the loop works against any provider the Protocol
covers without a second message format to normalise.
"""
from __future__ import annotations

import json
import time

from rory.llm import LLM, Message, ToolCall
from rory.rag import retrieve  # noqa: F401 — importing registers search_notes
from rory.tools import clock, desktop  # noqa: F401 — importing registers the tools
from rory.tools.registry import dispatch, schemas, truncate_result
from rory.trace import Trace

# A confused model must not loop forever. Four is enough for the deepest real
# sequence here (check -> open -> re-check -> answer).
MAX_ITERATIONS = 4

CAP_REACHED_TEXT = (
    "I kept going back and forth on that and stopped rather than spin. "
    "Could you try asking a different way?"
)


def run(
    llm: LLM,
    messages: list[Message],
    system: str,
    trace: Trace,
    tools: list[dict] | None = None,
) -> str:
    tools = schemas() if tools is None else tools
    for _ in range(MAX_ITERATIONS):
        started = time.monotonic()
        response = llm.generate(messages, system=system, tools=tools)
        trace.event(
            "llm_generate",
            (time.monotonic() - started) * 1000,
            completion_chars=len(response.text),
            tool_calls=[call.name for call in response.tool_calls],
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )

        if not response.tool_calls:
            return response.text

        messages.append({"role": "assistant", "content": _render_calls(response.tool_calls)})
        messages.append({"role": "user", "content": _run_calls(response.tool_calls, trace)})

    return CAP_REACHED_TEXT


def _run_calls(calls: list[ToolCall], trace: Trace) -> str:
    lines = []
    for call in calls:
        started = time.monotonic()
        result = dispatch(call.name, call.arguments)
        trace.event(
            "tool_dispatch",
            (time.monotonic() - started) * 1000,
            tool=call.name,
            ok=result.get("ok"),
            verified=result.get("verified"),
        )
        lines.append(f"{call.name} -> {truncate_result(result)}")
    return "TOOL RESULTS\n" + "\n".join(lines)


def _render_calls(calls: list[ToolCall]) -> str:
    rendered = "\n".join(f"{call.name}({json.dumps(call.arguments)})" for call in calls)
    return "TOOL CALLS\n" + rendered
