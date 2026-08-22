"""Tests for the append-only governed decision ledger."""

import json

from tools.decision_log_tool import decision_log_tool


def _authority_config():
    return {
        "model": {
            "provider": "openai-codex",
            "default": "gpt-5.6-sol",
            "openai_runtime": "codex_app_server",
            "openai_runtime_require_exact": True,
        },
        "agent": {"reasoning_effort": "xhigh"},
    }


def test_record_writes_exact_authority_and_list_skips_malformed(
    monkeypatch, tmp_path
):
    ledger = tmp_path / "decisions.jsonl"
    monkeypatch.setenv("HERMES_DECISION_LEDGER", str(ledger))
    monkeypatch.setattr("hermes_cli.config.load_config", _authority_config)

    recorded = json.loads(
        decision_log_tool(
            "record",
            decision="Use the exact Codex subscription runtime.",
            rationale="Durable judgment must use the selected authority.",
            evidence=["model/list exposes gpt-5.6-sol with xhigh"],
            confidence="high",
            reversibility="reversible",
            scope="persistent-agent",
            alternatives=["API-billed orchestration"],
            dissent=["Subscription limits can interrupt service"],
        )
    )
    assert recorded["ok"] is True
    assert recorded["entry"]["authority"] == {
        "provider": "openai-codex",
        "model": "gpt-5.6-sol",
        "effort": "xhigh",
        "runtime": "codex_app_server",
    }
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write("not-json\n")
    listed = json.loads(decision_log_tool("list", scope="persistent-agent"))
    assert [entry["id"] for entry in listed["entries"]] == [
        recorded["entry"]["id"]
    ]


def test_record_rejects_non_exact_authority(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_DECISION_LEDGER", str(tmp_path / "decisions.jsonl"))
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "model": {"provider": "anthropic", "default": "claude-opus-5"},
            "agent": {"reasoning_effort": "xhigh"},
        },
    )

    result = json.loads(
        decision_log_tool(
            "record",
            decision="Should fail.",
            rationale="Wrong authority.",
            evidence=["test"],
            confidence="high",
            reversibility="reversible",
            scope="test",
        )
    )
    assert result["ok"] is False
    assert "exact OpenAI/Codex" in result["error"]


def test_decision_log_is_registered_and_exposed():
    from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS
    from tools.registry import registry
    import toolsets

    assert registry.get_entry("decision_log") is not None
    assert "decision_log" in toolsets._HERMES_CORE_TOOLS
    assert "decision_log" in toolsets.TOOLSETS["hermes-api-server"]["tools"]
    assert toolsets.TOOLSETS["decision_log"]["tools"] == ["decision_log"]
    assert "decision_log" in EXPOSED_TOOLS
