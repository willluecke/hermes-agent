"""The report_results tool: structured result claims the verify judge checks against the evidence ledger."""

from __future__ import annotations

import json

from tools import report_results_tool as module
from tools.registry import registry


def test_registered_in_the_typesafe_toolset_without_env_requirements():
    from toolsets import _HERMES_CORE_TOOLS, TOOLSETS

    entry = registry.get_entry("report_results")
    assert entry is not None
    assert registry.get_toolset_for_tool("report_results") == "typesafe"
    assert not getattr(entry, "requires_env", None)
    assert "report_results" in _HERMES_CORE_TOOLS
    assert "report_results" in TOOLSETS["typesafe"]["tools"]


def test_returns_a_manifest_with_ids_in_order_and_defaults():
    result = json.loads(module.report_results({"results": [
        {"claim": "Plugin suite passes", "evidence": ["c7"], "criterion": "1"},
        {"claim": "98 passed", "evidence": "c7", "predicate": "count", "expected": {"passed": 98, "failed": 0}},
        {"claim": "Deployed", "evidence": ["c9"], "predicate": "contains", "expected": {"text": "success"}},
    ]}))
    assert result["manifest"] == [
        {"id": "r1", "criterion": "1", "claim": "Plugin suite passes", "evidence": ["c7"], "predicate": "passed", "expected": {}},
        {"id": "r2", "criterion": "", "claim": "98 passed", "evidence": ["c7"], "predicate": "count", "expected": {"passed": 98, "failed": 0}},
        {"id": "r3", "criterion": "", "claim": "Deployed", "evidence": ["c9"], "predicate": "contains", "expected": {"text": "success"}},
    ]
    assert "3 result claims registered" in result["note"] and "skipped" not in result


def test_an_empty_list_means_no_result_is_claimed():
    result = json.loads(module.report_results({"results": []}))
    assert result["manifest"] == [] and "claims no check result" in result["note"]


def test_malformed_items_are_named_and_the_rest_kept():
    result = json.loads(module.report_results({"results": [
        "not an object",
        {"claim": "", "evidence": ["c1"]},
        {"claim": "bad predicate", "evidence": ["c1"], "predicate": "vibes"},
        {"claim": "count without counts", "evidence": ["c1"], "predicate": "count"},
        {"claim": "contains without text", "evidence": ["c1"], "predicate": "contains"},
        {"claim": "fine", "evidence": ["c1"]},
    ]}))
    assert [item["claim"] for item in result["manifest"]] == ["fine"]
    assert len(result["skipped"]) == 5 and "item 3: predicate must be one of" in result["skipped"][2]
    assert "error" in json.loads(module.report_results({"results": [{"claim": "", "evidence": []}]}))
    assert "error" in json.loads(module.report_results({}))
    assert "error" in json.loads(module.report_results({"results": "c1"}))


def test_bounds_on_items_claim_length_and_evidence():
    result = json.loads(module.report_results({"results": [{"claim": "x" * 1_000, "evidence": [f"c{i}" for i in range(20)]}] * 30}))
    assert len(result["manifest"]) == module.MAX_ITEMS
    assert len(result["manifest"][0]["claim"]) == module.MAX_CLAIM_CHARS
    assert len(result["manifest"][0]["evidence"]) == module.MAX_EVIDENCE
    schema = module.REPORT_RESULTS_SCHEMA["parameters"]["properties"]["results"]["items"]
    assert schema["properties"]["predicate"]["enum"] == list(module.PREDICATES)
    assert schema["required"] == ["claim", "evidence"]
    assert "re-run by the gate" in module.REPORT_RESULTS_SCHEMA["description"] and "wrapper scripts" in module.REPORT_RESULTS_SCHEMA["description"]
