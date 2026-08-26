import json
import subprocess
import sys
from pathlib import Path

from rory.core import RoryCore
from tests.fakes import FakeLLM


def test_handle_text_returns_reply():
    core = RoryCore(FakeLLM(["Hello there."]))

    reply = core.handle_text("hi")

    assert reply.error is None
    assert reply.text == "Hello there."
    assert reply.turn_id


def test_second_turn_can_reference_earlier_fact():
    fake = FakeLLM(["Nice to meet you, Kundhave.", "Your name is Kundhave."])
    core = RoryCore(fake)

    core.handle_text("My name is Kundhave.")
    reply = core.handle_text("What's my name?")

    assert reply.text == "Your name is Kundhave."
    # The second call's message history must still contain the first turn.
    second_call_messages = fake.calls[1]
    assert any("Kundhave" in m["content"] for m in second_call_messages)


def test_llm_failure_becomes_error_reply_not_exception():
    class BrokenLLM:
        def generate(self, messages, system=None, tools=None):
            raise RuntimeError("upstream exploded")

    core = RoryCore(BrokenLLM())

    reply = core.handle_text("hi")

    assert reply.error == "upstream exploded"
    assert reply.text == ""


def test_trace_output_is_one_json_object_per_line():
    core = RoryCore(FakeLLM(["ok"]))

    core.handle_text("hi")

    trace_path = Path("logs/trace.jsonl")
    lines = trace_path.read_text().strip().splitlines()
    assert len(lines) >= 1
    for line in lines:
        record = json.loads(line)
        assert "turn_id" in record
        assert "stage" in record
        assert "elapsed_ms" in record


def test_missing_api_key_fails_at_import_with_clear_message(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent
    env = {k: v for k, v in __import__("os").environ.items() if k != "GEMINI_API_KEY"}
    env["PYTHONPATH"] = str(repo_root)
    result = subprocess.run(
        [sys.executable, "-c", "import rory.config"],
        env=env,
        capture_output=True,
        text=True,
        cwd=tmp_path,  # no .env here, unlike the repo root
    )

    assert result.returncode != 0
    assert "gemini_api_key" in result.stderr.lower()
