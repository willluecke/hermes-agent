import json
from pathlib import Path

import pytest

import toolsets
from tools import talos_tool as talos
from tools.registry import registry


@pytest.fixture
def talos_server(monkeypatch, tmp_path):
    """A loopback stand-in for Talos that records the requests it receives."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading

    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _reply(self, status, body):
            raw = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            seen.append(("GET", self.path, self.headers.get("Authorization"), None))
            if self.headers.get("Authorization") != "Bearer " + "t" * 40:
                return self._reply(401, {"error": "Sign in through Hermes to open this operator view."})
            if self.path == "/api/state":
                return self._reply(200, {"at": 1, "timezone": "UTC", "health": {"stale": False}, "models": {},
                    "projects": [{"id": "p", "repo": "/r", "mode": "paused", "priority": 1, "version": 1, "body": {"name": "P", "objective": "O", "secret_blob": "x"}}],
                    "tasks": [{"id": "t", "project_id": "p", "state": "QUEUED", "waiting": None, "detail": None, "generation": 0, "updated": 1, "version": 1, "progress": {}, "body": {"contract": {"artifact": "A", "obligations": [{"id": "OB"}]}}}],
                    "decisions": [], "commitments": [], "routines": [], "budgets": [], "events": [{"kind": "scheduler.tick"}]})
            if self.path == "/api/live":
                return self._reply(200, {"at": 1, "items": [], "needs_you": 0})
            if self.path == "/api/tasks/t":
                return self._reply(200, {"task": {"id": "t"}, "findings": [], "reviews": [], "verifications": []})
            return self._reply(404, {"error": "Not found"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            seen.append(("POST", self.path, self.headers.get("Authorization"), payload))
            if self.path == "/api/decisions/answer":
                return self._reply(409, {"error": "Decision version is stale"})
            if self.path == "/api/projects/save":
                return self._reply(200, {"id": payload["id"], "version": 2})
            return self._reply(404, {"error": "Unknown operator action"})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    token = tmp_path / "dashboard.token"
    token.write_text("t" * 40 + "\n")
    monkeypatch.setenv("HERMES_TALOS_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("HERMES_TALOS_TOKEN_FILE", str(token))
    yield seen, token
    server.shutdown()
    server.server_close()


def test_state_is_compacted_and_carries_the_operator_token(talos_server):
    seen, _ = talos_server
    result = json.loads(talos.talos_tool("state"))
    assert result["ok"] is True
    assert seen[-1][1] == "/api/state" and seen[-1][2] == "Bearer " + "t" * 40
    project = result["result"]["projects"][0]
    assert project["name"] == "P" and "secret_blob" not in project and "body" not in project
    task = result["result"]["tasks"][0]
    assert task["artifact"] == "A" and "body" not in task
    assert "events" not in result["result"]


def test_task_and_live_reads(talos_server):
    assert json.loads(talos.talos_tool("task", id="t"))["result"]["task"]["id"] == "t"
    assert json.loads(talos.talos_tool("live"))["result"]["needs_you"] == 0
    bad = json.loads(talos.talos_tool("task", id="../x"))
    assert bad["ok"] is False and "task id" in bad["error"]


def test_writes_validate_required_fields_and_relay_refusals(talos_server):
    seen, _ = talos_server
    missing = json.loads(talos.talos_tool("decision_answer", id="d"))
    assert missing["ok"] is False and "version" in missing["error"] and "answer" in missing["error"]
    assert not any(call[0] == "POST" for call in seen), "an incomplete write never reaches Talos"
    stale = json.loads(talos.talos_tool("decision_answer", id="d", version=1, answer={"action": "approve", "reason": "ok"}))
    assert stale["ok"] is False and "409" in stale["error"] and "stale" in stale["error"]
    saved = json.loads(talos.talos_tool("project_save", id="p", repo="/r", mode="active", priority=70, body={"objective": "O"}))
    assert saved["ok"] is True and saved["result"]["version"] == 2
    assert seen[-1][3]["priority"] == 70


def test_missing_token_or_unreachable_runtime_is_an_error_not_a_guess(talos_server, monkeypatch):
    _, token = talos_server
    token.unlink()
    result = json.loads(talos.talos_tool("state"))
    assert result["ok"] is False and "token" in result["error"]
    token.write_text("t" * 40)
    monkeypatch.setenv("HERMES_TALOS_URL", "http://127.0.0.1:1")
    result = json.loads(talos.talos_tool("live"))
    assert result["ok"] is False and "unreachable" in result["error"]


def test_registered_in_registry_and_toolsets():
    assert registry.get_entry("talos") is not None
    assert "talos" in toolsets.TOOLSETS["talos"]["tools"]
    from agent.transports import hermes_tools_mcp_server as mcp
    assert "talos" in mcp.EXPOSED_TOOLS


def test_handler_maps_verb_onto_the_upstream_action_field(talos_server):
    seen, _ = talos_server
    talos._handler({"action": "commitment_action", "id": "c", "verb": "done", "evidence": "shipped"})
    assert seen[-1][1] == "/api/commitments/action"
    assert seen[-1][3] == {"id": "c", "action": "done", "evidence": "shipped"}
