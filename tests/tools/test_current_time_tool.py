"""Tests for the local current-time tool."""

import json
import time
from datetime import datetime

from tools.current_time_tool import (
    CURRENT_TIME_SCHEMA,
    check_current_time_requirements,
    current_time_tool,
)


def test_current_time_returns_consistent_timezone_aware_fields():
    before = int(time.time())
    result = json.loads(current_time_tool())
    after = int(time.time())

    assert "error" not in result
    assert set(result) == {
        "iso",
        "timezone",
        "utc_offset",
        "unix",
        "date",
        "time",
        "weekday",
    }
    parsed = datetime.fromisoformat(result["iso"])
    assert parsed.tzinfo is not None
    assert parsed.strftime("%Y-%m-%d") == result["date"]
    assert before - 2 <= result["unix"] <= after + 2


def test_current_time_schema_and_requirements():
    assert check_current_time_requirements() is True
    assert CURRENT_TIME_SCHEMA["name"] == "current_time"
    assert CURRENT_TIME_SCHEMA["parameters"] == {
        "type": "object",
        "properties": {},
        "required": [],
    }


def test_current_time_is_registered_and_exposed_to_server_toolsets():
    from tools.registry import registry
    import toolsets

    entry = registry.get_entry("current_time")
    assert entry is not None
    assert "iso" in json.loads(entry.handler({}))
    assert "current_time" in toolsets._HERMES_CORE_TOOLS
    assert "current_time" in toolsets.TOOLSETS["hermes-api-server"]["tools"]
    assert toolsets.TOOLSETS["current_time"]["tools"] == ["current_time"]
