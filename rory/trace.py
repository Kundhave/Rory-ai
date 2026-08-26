"""Structured JSONL turn logging.

Defaults to metadata only (lengths, counts, elapsed ms) — never message content.
A turn's raw text can contain anything the user typed or the LLM produced, and
trace files are meant to be safe to grep, ship in a bug report, or leave lying
around in logs/. RORY_TRACE_VERBOSE opts into full text for local debugging.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from rory.config import settings

_LOG_PATH = Path("logs/trace.jsonl")


class Trace:
    """One instance per turn. Each event() call appends one JSON line."""

    def __init__(self, turn_id: str, path: Path = _LOG_PATH) -> None:
        self._turn_id = turn_id
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, stage: str, elapsed_ms: float, **meta: Any) -> None:
        record = {
            "turn_id": self._turn_id,
            "stage": stage,
            "elapsed_ms": round(elapsed_ms, 2),
            **meta,
        }
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def text_or_len(self, text: str) -> str | int:
        """Redact text to its length unless RORY_TRACE_VERBOSE is set."""
        return text if settings.rory_trace_verbose else len(text)
