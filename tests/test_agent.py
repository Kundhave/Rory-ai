import json

import psutil

from rory.agent.loop import CAP_REACHED_TEXT, MAX_ITERATIONS
from rory.agent.prompts import build_system_prompt
from rory.core import RoryCore
from tests.fakes import FakeLLM, calls, says


def test_turn_without_tool_calls_returns_the_text():
    core = RoryCore(FakeLLM(["Nothing to look up."]))

    assert core.handle_text("hi").text == "Nothing to look up."


def test_tools_are_offered_to_the_llm():
    fake = FakeLLM(["hi"])
    RoryCore(fake).handle_text("hi")

    offered = {tool["name"] for tool in fake.tools_seen[0]}
    assert offered == {"get_datetime", "open_app", "check_app_running", "search_notes"}


def test_tool_result_is_fed_back_before_the_final_answer():
    fake = FakeLLM([calls(("get_datetime", {})), says("It's Wednesday.")])
    core = RoryCore(fake)

    reply = core.handle_text("what day is it?")

    assert reply.text == "It's Wednesday."
    # The second LLM call must have seen the tool's actual output.
    fed_back = fake.calls[1][-1]["content"]
    assert "TOOL RESULTS" in fed_back
    assert "day_of_week" in fed_back


def test_unverified_check_is_what_the_model_is_given(monkeypatch):
    monkeypatch.setattr(psutil, "process_iter", lambda attrs: (_ for _ in ()).throw(psutil.AccessDenied()))

    uncertain = "I wasn't able to check whether the browser is running."
    fake = FakeLLM([calls(("check_app_running", {"app": "browser"})), says(uncertain)])

    reply = RoryCore(fake).handle_text("is my browser open?")

    # The real assertion: the loop hands the model an honest unverified result
    # rather than a bare running=false it could mistake for an answer. A fake
    # LLM cannot prove the model *complies* — that is the grounding rule's job,
    # asserted separately below and exercised for real by a live test.
    fed_back = json.loads(fake.calls[1][-1]["content"].split(" -> ", 1)[1])
    assert fed_back["verified"] is False
    assert fed_back["running"] is None

    assert reply.text == uncertain


def test_grounding_rules_tell_the_model_not_to_answer_yes_or_no_when_unverified():
    prompt = build_system_prompt()

    assert "verified: false" in prompt
    assert "Do not answer" in prompt and "yes" in prompt


def test_loop_stops_at_the_iteration_cap():
    # Always asks for another tool, never answers.
    fake = FakeLLM([calls(("get_datetime", {}))] * (MAX_ITERATIONS + 3))

    reply = RoryCore(fake).handle_text("loop forever")

    assert len(fake.calls) == MAX_ITERATIONS
    assert reply.text == CAP_REACHED_TEXT
    assert reply.error is None


def test_a_tool_exception_does_not_crash_the_turn(monkeypatch):
    monkeypatch.setattr(psutil, "process_iter", lambda attrs: 1 / 0)

    fake = FakeLLM([
        calls(("check_app_running", {"app": "browser"})),
        says("Something went wrong checking that."),
    ])

    reply = RoryCore(fake).handle_text("is my browser open?")

    assert reply.error is None
    assert reply.text == "Something went wrong checking that."
    assert "ZeroDivisionError" in fake.calls[1][-1]["content"]


def test_invalid_tool_arguments_come_back_as_a_result_the_model_can_explain():
    fake = FakeLLM([
        calls(("open_app", {"app": "spotify"})),
        says("I can't open Spotify — it isn't one of the apps I'm allowed to launch."),
    ])

    reply = RoryCore(fake).handle_text("open spotify")

    assert reply.error is None
    assert "not one of" in fake.calls[1][-1]["content"]


def test_tool_exchange_stays_in_history_for_the_next_turn():
    fake = FakeLLM([calls(("get_datetime", {})), says("It's Wednesday."), says("Yes, Wednesday.")])
    core = RoryCore(fake)

    core.handle_text("what day is it?")
    core.handle_text("are you sure?")

    assert any("TOOL RESULTS" in message["content"] for message in fake.calls[2])


def test_leaked_tool_call_syntax_is_never_spoken_and_the_model_self_corrects():
    # Observed for real: Gemini sometimes writes a tool call out as plain
    # text (e.g. "default_api.search_notes(query=...)") instead of using
    # the structured function-calling channel. That text never arrives as
    # a real ToolCall, so without this check it looks exactly like an
    # ordinary answer and gets spoken to the user verbatim.
    fake = FakeLLM([
        says('default_api.search_notes(query="kavin niranjana mitali")'),
        says("Kavin, Niranjana, and Mitali are mentioned in your notes."),
    ])

    reply = RoryCore(fake).handle_text("tell me about kavin, niranjana and mitali")

    assert reply.text == "Kavin, Niranjana, and Mitali are mentioned in your notes."
    # The model was given a chance to self-correct, using one iteration.
    assert len(fake.calls) == 2
    corrective_message = fake.calls[1][-1]["content"]
    assert "wasn't a real answer" in corrective_message


def test_a_model_that_keeps_leaking_tool_call_syntax_hits_the_cap_not_the_user():
    fake = FakeLLM([says('search_notes(query="x")')] * (MAX_ITERATIONS + 2))

    reply = RoryCore(fake).handle_text("tell me about kavin")

    assert len(fake.calls) == MAX_ITERATIONS
    assert reply.text == CAP_REACHED_TEXT


def test_a_real_answer_that_happens_to_mention_a_tool_name_is_not_flagged():
    # The detector matches call syntax specifically ("name(") — not just the
    # bare word — so a normal sentence never gets mistaken for a leak.
    fake = FakeLLM([says("I used search_notes to check, and Relay uses exponential backoff.")])

    reply = RoryCore(fake).handle_text("what does relay use for retries")

    assert reply.text == "I used search_notes to check, and Relay uses exponential backoff."
    assert len(fake.calls) == 1
