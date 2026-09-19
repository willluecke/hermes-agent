"""POST /v1/hooks/claude: a managed Claude Code process reports one tool call."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_cli import claude_hooks

pytestmark = pytest.mark.asyncio


def _app() -> web.Application:
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "api-key"}))
    app = web.Application()
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/hooks/claude", adapter._handle_claude_hook)
    return app


class _Agent:
    session_id = "sess-a"
    _current_turn_id = "turn-1"


@pytest.fixture
def bound():
    claude_hooks._bindings.clear()
    agent = _Agent()
    token = claude_hooks.new_hook_token()
    claude_hooks.bind_hook_token(token, session_id="sess-a", agent=agent)
    yield token, agent
    claude_hooks._bindings.clear()


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


PRE = {"event": "PreToolUse", "tool_name": "Bash", "tool_input": {"command": "rm -rf build", "description": "clean"}, "tool_use_id": "toolu_1"}
POST = {
    "event": "PostToolUse", "tool_name": "Bash", "tool_input": {"command": "pytest -q", "description": "tests"},
    "tool_response": {"stdout": "1 failed, 3 passed", "stderr": "", "interrupted": False}, "tool_use_id": "toolu_2", "duration_ms": 25,
    "cwd": "/work",
}


async def test_only_a_bound_hook_token_is_accepted(bound):
    token, _agent = bound
    async with TestClient(TestServer(_app())) as cli:
        assert (await cli.post("/v1/hooks/claude", json=PRE)).status == 401
        assert (await cli.post("/v1/hooks/claude", json=PRE, headers=_bearer("api-key"))).status == 401, "the API key is not a hook token"
        assert (await cli.post("/v1/hooks/claude", json=PRE, headers=_bearer(token + "x"))).status == 401
        assert (await cli.post("/v1/hooks/claude", data="not json", headers=_bearer(token))).status == 400
        resp = await cli.post("/v1/hooks/claude", json=PRE, headers=_bearer(token))
        assert resp.status == 200 and await resp.json() == {"action": "pass"}


async def test_pre_tool_use_returns_the_plugin_block_under_hermes_tool_names(bound, monkeypatch):
    token, _agent = bound
    seen = {}

    def directive(name, args, **kwargs):
        seen.update(name=name, args=args, **kwargs)
        return "block", "Hold: confirm with the user."

    monkeypatch.setattr("hermes_cli.plugins.get_pre_tool_call_directive", directive)
    async with TestClient(TestServer(_app())) as cli:
        resp = await cli.post("/v1/hooks/claude", json=PRE, headers=_bearer(token))
        assert resp.status == 200 and await resp.json() == {"action": "block", "message": "Hold: confirm with the user."}
    assert seen["name"] == "terminal" and seen["args"] == {"command": "rm -rf build"}
    assert seen["session_id"] == "sess-a" and seen["turn_id"] == "turn-1" and seen["tool_call_id"] == "toolu_1"

    monkeypatch.setattr("hermes_cli.plugins.get_pre_tool_call_directive", lambda *a, **k: ("approve", "needs a human"))
    async with TestClient(TestServer(_app())) as cli:
        resp = await cli.post("/v1/hooks/claude", json=PRE, headers=_bearer(token))
        assert await resp.json() == {"action": "pass"}, "an approve directive is not escalated on this lane"


async def test_post_tool_use_feeds_the_observers_live_and_returns_their_steer(bound, monkeypatch):
    token, _agent = bound
    calls = []
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: name == "post_tool_call")

    def invoke(name, **kwargs):
        calls.append((name, kwargs))
        return [{"message": "Drift check: back to criterion 1."}] if len(calls) == 1 else []

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", invoke)
    async with TestClient(TestServer(_app())) as cli:
        resp = await cli.post("/v1/hooks/claude", json=POST, headers=_bearer(token))
        assert resp.status == 200 and await resp.json() == {"message": "Drift check: back to criterion 1."}
        resp = await cli.post("/v1/hooks/claude", json={**POST, "tool_name": "Read", "tool_input": {"file_path": "/tmp/a.py"}, "tool_response": {"type": "text", "file": {"content": "x"}}}, headers=_bearer(token))
        assert await resp.json() == {}
    (name, first), (_name, second) = calls
    assert name == "post_tool_call"
    assert first["tool_name"] == "terminal" and first["args"] == {"command": "pytest -q"} and first["result"] == "1 failed, 3 passed"
    assert first["steerable"] is True and first["duration_ms"] == 25 and first["tool_call_id"] == "toolu_2"
    assert first["session_id"] == "sess-a" and first["turn_id"] == "turn-1"
    assert first["cwd"] == "/work", "the evidence ledger keys its workspace digest and re-runs on the call's directory"
    assert second["tool_name"] == "read_file" and "x" in second["result"]
    assert claude_hooks.live_calls(token) == 2, "the parity replay will know the observers already saw these"


async def test_a_dispatch_failure_lets_the_call_proceed(bound, monkeypatch):
    token, _agent = bound

    def boom(*args, **kwargs):
        raise RuntimeError("plugin exploded")

    monkeypatch.setattr("hermes_cli.plugins.get_pre_tool_call_directive", boom)
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: True)
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", boom)
    async with TestClient(TestServer(_app())) as cli:
        assert await (await cli.post("/v1/hooks/claude", json=PRE, headers=_bearer(token))).json() == {"action": "pass"}
        assert await (await cli.post("/v1/hooks/claude", json=POST, headers=_bearer(token))).json() == {}


async def test_post_tool_use_runs_the_real_criteria_fidelity_check_and_returns_its_note(bound, monkeypatch, tmp_path):
    """A criteria registration from a managed Claude process reaches the preflight plugin under its Hermes name, and the plugin's note rides the hook response back to the model."""
    token, _agent = bound
    plugin_file = Path(__file__).resolve().parents[2] / "plugins" / "system-one-preflight" / "__init__.py"
    spec = importlib.util.spec_from_file_location("system_one_preflight_hook_endpoint_test", plugin_file)
    preflight = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(preflight)
    settings = {"mode": "feedback", "log_path": str(tmp_path / "preflight.jsonl"), "tuning": "off", "tuning_state_path": str(tmp_path / "tuning.json"), "drift_check": "off"}
    monkeypatch.setattr(preflight, "_settings_reader", lambda key, default=None: settings.get(key, default))

    def jev(state, questions, timeout=None):
        answers = {key: {"type": "noul", "noul": 0.1 if key == "entails_2" else 0.9} for key in questions}
        answers["coverage"] = {"type": "noul", "noul": 0.2}
        return {"model": "jev-1.13.0", "answers": answers}

    monkeypatch.setattr(preflight, "_ask", jev)
    preflight._session_scope["sess-a"] = ["Add a --json flag to the exporter"]
    preflight.reset_drift("sess-a")
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: name == "post_tool_call")
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", lambda name, **kwargs: [preflight.on_post_tool_call(**kwargs)])
    registered = json.dumps({"todos": [
        {"id": "1", "content": "The --json flag prints valid JSON", "status": "in_progress"},
        {"id": "2", "content": "The README gains a section on exporters", "status": "in_progress"},
    ], "note": "2 acceptance criteria registered."})
    body = {
        "event": "PostToolUse", "tool_name": "mcp__hermes-tools__acceptance_criteria",
        "tool_input": {"criteria": ["The --json flag prints valid JSON", "The README gains a section on exporters"]},
        # Claude Code hands an MCP tool's result back as the SDK's {"result": "<json>"} wrapper.
        "tool_response": [{"type": "text", "text": json.dumps({"result": registered})}], "tool_use_id": "toolu_3", "duration_ms": 5,
    }
    async with TestClient(TestServer(_app())) as cli:
        resp = await cli.post("/v1/hooks/claude", json=body, headers=_bearer(token))
        assert resp.status == 200
        message = (await resp.json())["message"]
    assert "The README gains a section on exporters" in message and "will not be judged" in message
    assert "may not cover everything the request asks for (P(cover) = 0.20)" in message
    assert preflight._session_excluded["sess-a"] == ["2"]
    [record] = [json.loads(line) for line in (tmp_path / "preflight.jsonl").read_text().splitlines() if '"fidelity"' in line]
    assert record["excluded"] == ["2"] and record["steer"] is True and record["replay"] is False


async def test_post_tool_use_feeds_the_real_evidence_ledger_with_the_calls_directory(bound, monkeypatch, tmp_path):
    """A Bash call from a managed Claude process becomes a ledger row (exit unknown, cwd from the payload), and a report_results call becomes the manifest."""
    token, _agent = bound
    plugin_file = Path(__file__).resolve().parents[2] / "plugins" / "system-one-preflight" / "__init__.py"
    spec = importlib.util.spec_from_file_location("system_one_preflight_hook_endpoint_ledger_test", plugin_file)
    preflight = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(preflight)
    settings = {
        "mode": "feedback", "log_path": str(tmp_path / "preflight.jsonl"), "tuning": "off", "tuning_state_path": str(tmp_path / "tuning.json"),
        "drift_check": "off", "fidelity_check": "off", "ledger_dir": str(tmp_path / "evidence"),
    }
    monkeypatch.setattr(preflight, "_settings_reader", lambda key, default=None: settings.get(key, default))
    preflight.reset_drift("sess-a")
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: name == "post_tool_call")
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", lambda name, **kwargs: [preflight.on_post_tool_call(**kwargs)])
    manifest = json.dumps({"manifest": [{"id": "r1", "criterion": "1", "claim": "suite passes", "evidence": ["c1"], "predicate": "passed", "expected": {}}], "note": "1 result claim registered."})
    async with TestClient(TestServer(_app())) as cli:
        resp = await cli.post("/v1/hooks/claude", json={**POST, "cwd": str(tmp_path)}, headers=_bearer(token))
        assert resp.status == 200 and await resp.json() == {}
        resp = await cli.post("/v1/hooks/claude", json={
            "event": "PostToolUse", "tool_name": "mcp__hermes-tools__report_results", "tool_input": {"results": [{"claim": "suite passes", "evidence": ["c1"]}]},
            "tool_response": [{"type": "text", "text": json.dumps({"result": manifest})}], "tool_use_id": "toolu_4", "duration_ms": 5, "cwd": str(tmp_path),
        }, headers=_bearer(token))
        assert resp.status == 200
    [row] = preflight.ledger_state("sess-a")["rows"]
    assert row["id"] == "c1" and row["command"] == "pytest -q" and row["cwd"] == str(tmp_path)
    assert row["exit"] is None and row["status"] == "fail" and row["counts"] == {"failed": 1, "passed": 3}, "no exit code on this lane; the runner's own summary decides"
    assert Path(row["file"]).read_text() == "1 failed, 3 passed"
    assert [item["claim"] for item in preflight._session_manifest["sess-a"]] == ["suite passes"]
