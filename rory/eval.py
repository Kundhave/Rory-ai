"""Golden-question harness. Runnable as:

    python -m rory.eval            RAG mode: search_notes retrieves on demand
    python -m rory.eval --stuff    the whole knowledge base pasted into the
                                    system prompt instead, search_notes withheld

Both modes make real Gemini calls — this is not part of the pytest suite and
costs real API quota (free-tier limits are small and vary by model: daily
caps on some, per-minute caps on others). Use --model/--limit/--delay to
control cost and pacing; see README.md for recorded results.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from rory.agent import loop
from rory.agent.prompts import build_system_prompt
from rory.config import settings
from rory.llm import GeminiLLM
from rory.rag.ingest import KNOWLEDGE_DIR
from rory.tools.registry import schemas
from rory.trace import Trace

GOLDEN_PATH = Path("tests/golden.yaml")


class RateLimitedLLM:
    """Paces every generate() call, not just every case — a single case can
    make two or more calls back-to-back (tool call, then the final answer),
    which blows through a free-tier per-minute limit even with a delay
    between cases."""

    def __init__(self, llm, delay_s: float) -> None:
        self._llm = llm
        self._delay_s = delay_s
        self._first_call = True

    def generate(self, *args, **kwargs):
        if not self._first_call:
            time.sleep(self._delay_s)
        self._first_call = False
        return self._llm.generate(*args, **kwargs)


@dataclass
class CaseResult:
    id: str
    ok: bool
    reasons: list[str] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)
    retrieved_sources: list[str] = field(default_factory=list)
    answer: str = ""
    elapsed_s: float = 0.0


def load_cases(path: Path = GOLDEN_PATH) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def stuffed_system_prompt(knowledge_dir: Path = KNOWLEDGE_DIR) -> str:
    docs = sorted(p for p in knowledge_dir.glob("*.md") if p.name != "README.md")
    body = "\n\n".join(f"### {p.name}\n{p.read_text(encoding='utf-8')}" for p in docs)
    return f"{build_system_prompt()}\n\n## Full knowledge base\n{body}"


def run_case(case: dict, llm, system: str, tools: list[dict], stuff_mode: bool = False) -> CaseResult:
    trace = Trace(turn_id=f"eval-{case['id']}")
    messages = [{"role": "user", "content": case["utterance"]}]

    started = time.monotonic()
    try:
        answer = loop.run(llm, messages, system, trace, tools=tools)
    except Exception as exc:
        # A batch of 20+ free-tier calls will occasionally hit a transient
        # 503/429 from Google's side. One flaky case shouldn't cost every
        # result gathered so far in the same run.
        return CaseResult(id=case["id"], ok=False, reasons=[f"API error: {exc}"],
                           elapsed_s=time.monotonic() - started)
    elapsed = time.monotonic() - started

    tool_calls: list[str] = []
    retrieved_sources: list[str] = []
    for message in messages:
        if message["role"] == "assistant" and message["content"].startswith("TOOL CALLS"):
            tool_calls.extend(line.split("(")[0] for line in message["content"].splitlines()[1:])
        if "search_notes ->" in message["content"] and '"source"' in message["content"]:
            for match in re.finditer(r'"source":\s*"([^"]+)"', message["content"]):
                retrieved_sources.append(match.group(1))

    reasons = []
    expect_tool = case.get("expect_tool")
    # In --stuff mode, search_notes is withheld on purpose (the KB is already
    # in the prompt), so a case that normally expects search_notes to fire
    # isn't a routing failure here — it's the point of the comparison. Every
    # other expected tool (get_datetime, or no tool) still applies unchanged.
    skip_retrieval_routing = stuff_mode and expect_tool == "search_notes"

    if not skip_retrieval_routing:
        if expect_tool is None:
            if tool_calls:
                reasons.append(f"expected no tool call, got {tool_calls}")
        elif expect_tool not in tool_calls:
            reasons.append(f"expected tool {expect_tool!r}, got {tool_calls}")

        for source in case.get("expect_source", []):
            if source not in retrieved_sources:
                reasons.append(f"expected {source!r} in top-3 retrieved sources, got {retrieved_sources}")

    for needle in case.get("must_contain", []):
        if needle.lower() not in answer.lower():
            reasons.append(f"answer missing required text {needle!r}")

    for needle in case.get("must_not_contain", []):
        if needle.lower() in answer.lower():
            reasons.append(f"answer contains forbidden text {needle!r}")

    if case.get("must_refuse") and not _looks_like_refusal(answer):
        reasons.append("answer does not read as an uncertainty/refusal")

    return CaseResult(
        id=case["id"],
        ok=not reasons,
        reasons=reasons,
        tool_calls=tool_calls,
        retrieved_sources=retrieved_sources,
        answer=answer,
        elapsed_s=elapsed,
    )


_REFUSAL_MARKERS = ("don't have", "do not have", "not in my notes", "no information",
                     "can't find", "cannot find", "not sure", "don't know", "no note")


def _looks_like_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


def print_table(results: list[CaseResult], label: str) -> None:
    print(f"\n=== {label} ===")
    for r in results:
        status = "PASS" if r.ok else "FAIL"
        print(f"  [{status}] {r.id:28s} ({r.elapsed_s:5.2f}s)")
        if not r.ok:
            for reason in r.reasons:
                print(f"           - {reason}")

    passed = sum(r.ok for r in results)
    retrieval_cases = [r for r in results if r.retrieved_sources or "search_notes" in r.tool_calls]
    print(f"\n  {passed}/{len(results)} passed")
    if retrieval_cases:
        print(f"  recall@3 tool-fired on {len(retrieval_cases)} retrieval case(s)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stuff", action="store_true", help="paste the whole KB into the prompt instead of retrieving")
    parser.add_argument("--model", default=settings.gemini_model, help="override GEMINI_MODEL for this run")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N cases (quota control)")
    parser.add_argument("--delay", type=float, default=4.0, help="seconds between every LLM call (free-tier per-minute limits)")
    args = parser.parse_args()

    cases = load_cases()
    if args.limit:
        cases = cases[: args.limit]

    llm = RateLimitedLLM(GeminiLLM(api_key=settings.gemini_api_key, model=args.model), args.delay)
    all_tools = schemas()
    if args.stuff:
        system = stuffed_system_prompt()
        tools = [t for t in all_tools if t["name"] != "search_notes"]
        label = f"--stuff ({args.model})"
    else:
        system = build_system_prompt()
        tools = all_tools
        label = f"RAG ({args.model})"

    results = [run_case(case, llm, system, tools, stuff_mode=args.stuff) for case in cases]
    print_table(results, label)

    if not all(r.ok for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
