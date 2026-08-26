"""THE WHITELIST.

APPS is the entire set of things Rory can launch. Each value holds a hardcoded
argv list; model output is only ever a *key* into this dict, never any part of
the command. `AppName` is derived from the keys, so the tool schema's enum and
the whitelist cannot drift apart — adding an app is one dict entry.
"""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from typing import Literal

import psutil

from rory.tools.registry import tool

# Long enough to catch a binary that dies on startup, short enough that a voice
# turn does not feel stalled.
_STARTUP_GRACE_SECONDS = 0.4


@dataclass(frozen=True)
class App:
    argv: list[str]
    # The process name to match when checking if it runs. None means this app
    # cannot be distinguished from another by process name alone — the check
    # reports unverified rather than guessing.
    process: str | None


APPS: dict[str, App] = {
    "browser": App(["chromium"], "chromium"),
    "terminal": App(["alacritty"], "alacritty"),
    "files": App(["nautilus"], "nautilus"),
    "editor": App(["code"], "code"),
    "chatgpt": App(["chromium", "--app=https://chatgpt.com"], None),
}

AppName = Literal[*APPS]


@tool
def check_app_running(app: AppName) -> dict:
    """Check whether a desktop application is currently running. Returns
    `verified` indicating whether the check itself could be trusted."""
    target = APPS[app].process
    if target is None:
        return {
            "ok": False,
            "running": None,
            "verified": False,
            "detail": f"{app} shares a process with another app; cannot identify it reliably",
        }

    scanned_everything = True
    try:
        for process in psutil.process_iter(["name"]):
            try:
                if process.info["name"] == target:
                    return {"ok": True, "running": True, "verified": True,
                            "detail": f"{app} is running"}
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                scanned_everything = False
    except psutil.Error as exc:
        return {"ok": False, "running": None, "verified": False,
                "detail": f"could not check {app}: {type(exc).__name__}"}

    detail = f"{app} does not appear to be running"
    if not scanned_everything:
        detail += ", but some processes could not be inspected"
    return {"ok": True, "running": False, "verified": scanned_everything, "detail": detail}


@tool
def open_app(app: AppName) -> dict:
    """Launch a desktop application. Only use this when the user asks to open,
    start, or launch something."""
    argv = APPS[app].argv
    try:
        process = subprocess.Popen(
            argv,
            shell=False,
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, PermissionError) as exc:
        return {"ok": False, "detail": f"could not launch {app}: {type(exc).__name__}"}

    # A missing shared library or bad flag shows up as an immediate non-zero
    # exit. Reporting success without this check would be fabricating an action.
    time.sleep(_STARTUP_GRACE_SECONDS)
    code = process.poll()
    if code is not None and code != 0:
        return {"ok": False, "detail": f"{app} exited immediately with code {code}"}
    return {"ok": True, "detail": f"launched {app}"}
