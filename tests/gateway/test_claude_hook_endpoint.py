"""POST /v1/hooks/claude: a managed Claude Code process reports one tool call."""

from __future__ import annotations

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
