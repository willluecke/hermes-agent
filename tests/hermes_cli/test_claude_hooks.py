"""The Claude Code hook channel: the token registry and the hook script."""

from __future__ import annotations

import io
import json
import os
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from hermes_cli import claude_hook, claude_hooks


class _Agent:
    session_id = "sess-a"
    _current_turn_id = "turn-1"


@pytest.fixture(autouse=True)
def _clean_registry():
    claude_hooks._bindings.clear()
    claude_hooks.set_hook_endpoint(None)
    yield
    claude_hooks._bindings.clear()
    claude_hooks.set_hook_endpoint(None)


def test_endpoint_url_uses_loopback_for_wildcard_listeners():
    assert claude_hooks.hook_endpoint_for("0.0.0.0", 8642) == "http://127.0.0.1:8642/v1/hooks/claude"
    assert claude_hooks.hook_endpoint_for("::", 8642) == "http://127.0.0.1:8642/v1/hooks/claude"
    assert claude_hooks.hook_endpoint_for("", 1) == "http://127.0.0.1:1/v1/hooks/claude"
    assert claude_hooks.hook_endpoint_for("localhost", 2) == "http://localhost:2/v1/hooks/claude"
    assert claude_hooks.hook_endpoint_for("::1", 3) == "http://[::1]:3/v1/hooks/claude"


def test_bindings_resolve_to_the_live_agent_and_its_current_turn():
    agent = _Agent()
    token = claude_hooks.new_hook_token()
    assert claude_hooks.resolve_hook_token(token) is None
    claude_hooks.bind_hook_token(token, session_id="sess-a", agent=agent)
    binding = claude_hooks.resolve_hook_token(token)
    assert binding["session_id"] == "sess-a" and binding["agent"] is agent and binding["turn_id"] == "turn-1"
    assert binding["token"] == token
    assert claude_hooks.note_live_call(token) == 1 and claude_hooks.note_live_call(token) == 2
    later = _Agent()
    later._current_turn_id = "turn-2"
    claude_hooks.bind_hook_token(token, session_id="sess-a", agent=later)
    assert claude_hooks.resolve_hook_token(token)["turn_id"] == "turn-2"
    assert claude_hooks.live_calls(token) == 2, "rebinding keeps the count"
    claude_hooks.reset_live_calls(token)
    assert claude_hooks.live_calls(token) == 0
    claude_hooks.unbind_hook_token(token)
    assert claude_hooks.resolve_hook_token(token) is None
    assert claude_hooks.note_live_call(token) == 0


def test_a_dead_agent_retires_its_token():
    token = claude_hooks.new_hook_token()
    claude_hooks.bind_hook_token(token, session_id="s", agent=_Agent())
    assert claude_hooks.resolve_hook_token(token) is None, "the weak reference is all that holds it"


def test_settings_wire_both_hooks_to_the_script_and_the_environment_carries_the_endpoint():
    settings = claude_hooks.hook_settings(python="/venv/bin/python")
    assert set(settings["hooks"]) == {"PreToolUse", "PostToolUse"}
    [entry] = settings["hooks"]["PreToolUse"]
    [hook] = entry["hooks"]
    assert hook["type"] == "command" and hook["timeout"] == claude_hooks.HOOK_TIMEOUT_SECONDS
    assert hook["command"] == f'"/venv/bin/python" "{claude_hooks.hook_script_path()}"'
    assert claude_hooks.hook_script_path().exists()
    assert "matcher" not in entry, "every tool call is reported"
    assert claude_hooks.hook_environment("tok") == {}, "no endpoint, no hooks"
    claude_hooks.set_hook_endpoint("http://127.0.0.1:8642/v1/hooks/claude")
    assert claude_hooks.hook_environment("tok") == {"HERMES_HOOK_URL": "http://127.0.0.1:8642/v1/hooks/claude", "HERMES_HOOK_TOKEN": "tok"}
    assert claude_hooks.hook_environment("") == {}


# ---------------------------------------------------------------------------
# The hook script against a stand-in gateway
# ---------------------------------------------------------------------------

class _Gateway(BaseHTTPRequestHandler):
    answers = {}
    seen = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).seen.append((self.path, self.headers.get("Authorization"), body))
        status, answer = type(self).answers.get(body.get("event"), (200, {}))
        payload = json.dumps(answer).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture
def gateway():
    _Gateway.answers = {}
    _Gateway.seen = []
    server = HTTPServer(("127.0.0.1", 0), _Gateway)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1/hooks/claude"
    server.shutdown()
    server.server_close()


PRE = {
    "session_id": "9e86", "transcript_path": "/x", "cwd": "/tmp", "hook_event_name": "PreToolUse",
    "tool_name": "Bash", "tool_input": {"command": "rm -rf build", "description": "clean"}, "tool_use_id": "toolu_1",
}
POST = {**PRE, "hook_event_name": "PostToolUse", "tool_response": {"stdout": "hi", "stderr": "", "interrupted": False}, "duration_ms": 25}


def _run(payload, url, token="tok"):
    out, err = io.StringIO(), io.StringIO()
    code = claude_hook.main(stdin=io.StringIO(json.dumps(payload)), stdout=out, stderr=err, environ={"HERMES_HOOK_URL": url, "HERMES_HOOK_TOKEN": token})
    return code, out.getvalue(), err.getvalue()


def test_a_pre_block_exits_2_with_the_message_on_stderr(gateway):
    _Gateway.answers = {"PreToolUse": (200, {"action": "block", "message": "Hold: confirm with the user first."})}
    code, out, err = _run(PRE, gateway)
    assert (code, out, err) == (2, "", "Hold: confirm with the user first.\n")
    [(path, auth, body)] = _Gateway.seen
    assert path == "/v1/hooks/claude" and auth == "Bearer tok"
    assert body == {
        "event": "PreToolUse", "tool_name": "Bash", "tool_input": {"command": "rm -rf build", "description": "clean"},
        "tool_response": None, "tool_use_id": "toolu_1", "claude_session_id": "9e86", "cwd": "/tmp",
    }


def test_a_post_message_is_the_json_block_decision_claude_puts_in_front_of_the_model(gateway):
    _Gateway.answers = {"PostToolUse": (200, {"message": "Drift check: back to criterion 1."})}
    code, out, err = _run(POST, gateway)
    assert code == 0 and err == ""
    assert json.loads(out) == {"decision": "block", "reason": "Drift check: back to criterion 1."}
    assert _Gateway.seen[-1][2]["tool_response"] == {"stdout": "hi", "stderr": "", "interrupted": False}


def test_everything_else_lets_the_call_proceed_silently(gateway):
    assert _run(PRE, gateway) == (0, "", ""), "a pass answer"
    assert _run(POST, gateway) == (0, "", ""), "an empty post answer"
    _Gateway.answers = {"PreToolUse": (500, {"error": "boom"})}
    assert _run(PRE, gateway) == (0, "", ""), "a failing gateway"
    assert _run({**PRE, "hook_event_name": "Notification"}, gateway) == (0, "", ""), "an event the gateway does not take"
    calls = len(_Gateway.seen)
    out, err = io.StringIO(), io.StringIO()
    assert claude_hook.main(stdin=io.StringIO(json.dumps(PRE)), stdout=out, stderr=err, environ={}) == 0
    assert len(_Gateway.seen) == calls, "no endpoint in the environment: nothing is sent"
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        closed_port = probe.getsockname()[1]
    assert _run(PRE, f"http://127.0.0.1:{closed_port}/v1/hooks/claude") == (0, "", ""), "an unreachable gateway"
    assert claude_hook.main(stdin=io.StringIO("not json"), stdout=out, stderr=err, environ={"HERMES_HOOK_URL": gateway, "HERMES_HOOK_TOKEN": "t"}) == 0


def test_large_tool_responses_are_clipped_before_they_leave_the_process():
    body = claude_hook.build_request({**POST, "tool_response": {"stdout": "x" * 100_000}})
    assert len(body["tool_response"]["stdout"]) < 20_000
    body = claude_hook.build_request({**POST, "tool_response": "y" * 100_000})
    assert len(body["tool_response"]) == claude_hook.MAX_RESPONSE_CHARS + 1


def test_the_script_runs_as_a_command_from_stdin(gateway):
    _Gateway.answers = {"PreToolUse": (200, {"action": "block", "message": "Blocked by the drift check."})}
    env = {**os.environ, "HERMES_HOOK_URL": gateway, "HERMES_HOOK_TOKEN": "tok"}
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, str(claude_hooks.hook_script_path())],
        input=json.dumps(PRE), env=env, capture_output=True, text=True, cwd=str(Path("/tmp")), timeout=30,
    )
    assert completed.returncode == 2 and completed.stderr == "Blocked by the drift check.\n" and completed.stdout == ""
