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
        {"claim": "Plugin suite passes", "evidence": ["pytest -q tests/plugins"], "criterion": "1"},
        {"claim": "98 passed", "evidence": "pytest -q tests/plugins", "predicate": "count", "expected": {"passed": 98, "failed": 0}},
        {"claim": "Deployed", "evidence": ["./deploy.sh staging"], "predicate": "contains", "expected": {"text": "success"}},
    ]}))
    assert result["manifest"] == [
        {"id": "r1", "criterion": "1", "claim": "Plugin suite passes", "evidence": ["pytest -q tests/plugins"], "predicate": "passed", "expected": {}},
        {"id": "r2", "criterion": "", "claim": "98 passed", "evidence": ["pytest -q tests/plugins"], "predicate": "count", "expected": {"passed": 98, "failed": 0}},
        {"id": "r3", "criterion": "", "claim": "Deployed", "evidence": ["./deploy.sh staging"], "predicate": "contains", "expected": {"text": "success"}},
    ]
    assert "3 result claims registered" in result["note"] and "newest run" in result["note"]
    assert "skipped" not in result and "warnings" not in result


def test_a_cited_row_id_is_registered_but_flagged_at_once():
    # The model is never shown row ids; a cited one is a guess, said so in the tool's own reply on every lane.
    result = json.loads(module.report_results({"results": [
        {"claim": "MCP tests pass", "evidence": ["c5"]},
        {"claim": "Suite passes", "evidence": ["npm test", "PC12"]},
    ]}))
    assert [item["evidence"] for item in result["manifest"]] == [["c5"], ["npm test", "PC12"]]
    assert result["warnings"] == [
        "item 1 cites c5: row ids are never shown to you and are not accepted; cite the command as you ran it",
        "item 2 cites PC12: row ids are never shown to you and are not accepted; cite the command as you ran it",
    ]


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
    description = module.REPORT_RESULTS_SCHEMA["description"]
    assert "re-run by the gate" in description and "wrapper scripts" in description
    assert "as you ran it" in description and "newest run" in description and "Never cite a row id" in description
    assert "shown to you in the drift" not in description, "the ids were never shown; the description must not say they were"
    assert "Not row ids" in schema["properties"]["evidence"]["description"]
    long = json.loads(module.report_results({"results": [{"claim": "heredoc", "evidence": ["x" * 5_000], "predicate": "ran"}]}))
    assert len(long["manifest"][0]["evidence"][0]) == module.MAX_CITED_CHARS
