"""The acceptance_criteria tool answers in the todo tool's shape so the verify judge can use it."""

from __future__ import annotations

import json

from tools import acceptance_criteria_tool as module
from tools.registry import registry


def test_registered_in_the_typesafe_toolset_without_env_requirements():
    from toolsets import _HERMES_CORE_TOOLS, TOOLSETS

    entry = registry.get_entry("acceptance_criteria")
    assert entry is not None
    assert registry.get_toolset_for_tool("acceptance_criteria") == "typesafe"
    assert not getattr(entry, "requires_env", None)
    assert "acceptance_criteria" in _HERMES_CORE_TOOLS
    assert "acceptance_criteria" in TOOLSETS["typesafe"]["tools"]


def test_returns_criteria_as_in_progress_todos_in_order():
    result = json.loads(module.acceptance_criteria({"criteria": ["The --json flag prints valid JSON", "  ", "Existing tests still pass"]}))
    assert result["todos"] == [
        {"id": "1", "content": "The --json flag prints valid JSON", "status": "in_progress"},
        {"id": "2", "content": "Existing tests still pass", "status": "in_progress"},
    ]
    assert "2 acceptance criteria registered" in result["note"]


def test_accepts_todo_shaped_items_and_bounds_length_and_count():
    long = "x" * 1_000
    result = json.loads(module.acceptance_criteria({"criteria": [{"content": long}] + [f"c{n}" for n in range(20)]}))
    assert len(result["todos"]) == module.MAX_ITEMS
    assert len(result["todos"][0]["content"]) == module.MAX_CHARS


def test_refuses_empty_or_malformed_input():
    assert "error" in json.loads(module.acceptance_criteria({"criteria": []}))
    assert "error" in json.loads(module.acceptance_criteria({"criteria": ["", None]}))
    assert "error" in json.loads(module.acceptance_criteria({}))


def test_records_a_status_per_item_so_progress_can_be_marked():
    result = json.loads(module.acceptance_criteria({"criteria": [
        {"content": "Flag parses", "status": "completed"},
        {"content": "Tests pass", "status": "PENDING"},
        {"content": "Docs updated", "status": "bogus"},
        "Changelog entry",
    ]}))
    assert [(item["content"], item["status"]) for item in result["todos"]] == [
        ("Flag parses", "completed"), ("Tests pass", "pending"), ("Docs updated", "in_progress"), ("Changelog entry", "in_progress"),
    ]
    schema = module.ACCEPTANCE_CRITERIA_SCHEMA["parameters"]["properties"]["criteria"]["items"]
    assert schema["anyOf"][1]["properties"]["status"]["enum"] == ["pending", "in_progress", "completed", "cancelled"]
    assert "status" in module.ACCEPTANCE_CRITERIA_SCHEMA["description"]


def test_retire_and_clear_need_a_reason_and_are_echoed_for_the_plugin():
    assert "needs a reason" in json.loads(module.acceptance_criteria({"retire": [{"content": "Docs updated"}]}))["error"]
    assert "needs a reason" in json.loads(module.acceptance_criteria({"clear": {}}))["error"]
    assert "needs a reason" in json.loads(module.acceptance_criteria({"criteria": [{"content": "Docs updated", "status": "cancelled"}]}))["error"]
    result = json.loads(module.acceptance_criteria({"criteria": [], "retire": [{"content": "Docs updated", "reason": "user said skip the docs"}, {"id": "p2", "reason": "superseded"}]}))
    assert result["todos"] == [] and result["retire"] == [{"target": "Docs updated", "reason": "user said skip the docs"}, {"target": "p2", "reason": "superseded"}]
    assert "retired with a stated reason" in result["note"]
    result = json.loads(module.acceptance_criteria({"clear": {"reason": "user dropped the redesign"}, "criteria": ["Flat header renders"]}))
    assert result["clear"] == {"reason": "user dropped the redesign"} and [item["content"] for item in result["todos"]] == ["Flat header renders"]
    result = json.loads(module.acceptance_criteria({"criteria": [{"content": "Docs updated", "status": "cancelled", "reason": "out of scope now"}, "Tests pass"]}))
    assert result["todos"][0] == {"id": "1", "content": "Docs updated", "status": "cancelled", "reason": "out of scope now"}
    assert "error" in json.loads(module.acceptance_criteria({"retire": "Docs updated"}))
    schema = module.ACCEPTANCE_CRITERIA_SCHEMA["parameters"]
    assert schema["required"] == [] and set(schema["properties"]) == {"criteria", "retire", "clear"}
    assert "never retire silently" in module.ACCEPTANCE_CRITERIA_SCHEMA["description"]
