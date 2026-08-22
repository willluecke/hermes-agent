#!/usr/bin/env python3
"""Dependency-free local system time tool."""

import json
from datetime import datetime

from tools.registry import registry


def current_time_tool() -> str:
    """Return the current local time as structured JSON."""
    try:
        now = datetime.now().astimezone()
        return json.dumps(
            {
                "iso": now.isoformat(),
                "timezone": str(now.tzinfo) if now.tzinfo else "unknown",
                "utc_offset": now.strftime("%z"),
                "unix": int(now.timestamp()),
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S"),
                "weekday": now.strftime("%A"),
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        return json.dumps({"error": f"Failed to read system time: {exc}"})


def check_current_time_requirements() -> bool:
    return True


CURRENT_TIME_SCHEMA = {
    "name": "current_time",
    "description": (
        "Get the current local date and time directly from the system clock. "
        "Returns an ISO 8601 timestamp, timezone, UTC offset, Unix timestamp, "
        "date, time, and weekday."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


registry.register(
    name="current_time",
    toolset="current_time",
    schema=CURRENT_TIME_SCHEMA,
    handler=lambda args, **kwargs: current_time_tool(),
    check_fn=check_current_time_requirements,
    emoji="🕒",
)
