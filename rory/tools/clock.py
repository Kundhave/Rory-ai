from __future__ import annotations

from datetime import datetime

from rory.tools.registry import tool


@tool
def get_datetime() -> dict:
    """Get the current local date and time. Use this for any question about what
    the date is, what day of the week it is, or what time it is right now."""
    now = datetime.now().astimezone()
    return {
        "ok": True,
        "iso": now.isoformat(),
        "human": now.strftime("%A, %d %B %Y at %I:%M %p"),
        "timezone": now.tzname(),
        "day_of_week": now.strftime("%A"),
    }
