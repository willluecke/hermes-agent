import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import toolsets
from tools import hermes_conversation_tool as convo
from tools.registry import registry


@pytest.fixture
def sync_server(monkeypatch):
    """A loopback stand-in for hermes-sync with one saved conversation."""
    state = {"conversation": {
        "id": "c_regwatch", "title": "Make $1 million", "project": "reg-watch", "model": "claude-code::claude-opus-5-5",
        "effort": "default", "harness": "hermes", "updatedAt": 5,
        "messages": [{"role": "user", "content": "hi", "id": "m1"}, {"role": "assistant", "content": "hello", "id": "m2"}],
    }, "other": {"id": "c_other", "title": "Other", "project": "talos", "model": "m", "effort": "high", "updatedAt": 9, "messages": []}}
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
            seen.append(("GET", self.path, self.headers.get("Authorization")))
            if self.headers.get("Authorization") != "Bearer sync-key":
                return self._reply(401, {"error": "unauthorized"})
            if self.path == "/state":
                return self._reply(200, {"conversations": {"c_regwatch": state["conversation"], "c_other": state["other"]}})
            if self.path == "/conversations/c_regwatch":
                return self._reply(200, {"conversation": state["conversation"], "window": {}, "activityCounts": {}})
            return self._reply(404, {"error": "not found"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            seen.append(("POST", self.path, payload))
            if self.path == "/queued-turns":
                # The store anchors the user message and the assistant slot; the
                # reply fills in once the resumed session answers.
                state["conversation"]["messages"] += [
                    {"role": "user", "content": payload["text"], "clientTurnId": payload["id"], "id": payload["id"] + "_user"},
                    {"role": "assistant", "content": "Crawl fix lands after the 10:13 run.", "runId": "run_1", "terminalState": "completed"},
                ]
                return self._reply(200, {"turn": {"id": payload["id"], "state": "queued"}, "turns": []})
            return self._reply(404, {"error": "unknown"})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("HERMES_SYNC_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("HERMES_SYNC_KEY", "sync-key")
    monkeypatch.setattr(convo, "POLL_SECONDS", 0.01)
    yield seen, state
    server.shutdown()
    server.server_close()


def test_list_and_read_use_the_sync_key_and_trim_to_summaries(sync_server):
    seen, _ = sync_server
    listed = json.loads(convo.hermes_conversation_tool("list", project="reg-watch"))
    assert listed["ok"] is True and [c["id"] for c in listed["result"]["conversations"]] == ["c_regwatch"]
    assert seen[-1][2] == "Bearer sync-key"
    everything = json.loads(convo.hermes_conversation_tool("list"))
    assert [c["id"] for c in everything["result"]["conversations"]] == ["c_other", "c_regwatch"], "newest first"
    read = json.loads(convo.hermes_conversation_tool("read", id="c_regwatch", tail=1))
    assert read["result"]["tail"] == [{"role": "assistant", "content": "hello", "id": "m2"}]
    assert read["result"]["project"] == "reg-watch"


def test_send_queues_a_turn_with_the_saved_runtime_and_waits_for_the_reply(sync_server):
    seen, state = sync_server
    result = json.loads(convo.hermes_conversation_tool("send", id="c_regwatch", text="status?", actor="Talos", wait_seconds=5))
    assert result["ok"] is True
    post = next(call for call in seen if call[0] == "POST")
    assert post[1] == "/queued-turns"
    assert post[2]["conversationId"] == "c_regwatch" and post[2]["text"] == "[Talos] status?"
    assert post[2]["runtime"] == {"model": "claude-code::claude-opus-5-5", "effort": "default", "project": "reg-watch"}
    assert post[2]["explicitlyReleased"] is True and post[2]["id"].startswith("agent_")
    assert result["result"]["status"] == "answered"
    assert result["result"]["reply"] == "Crawl fix lands after the 10:13 run."


def test_send_refuses_empty_text_and_unknown_conversations(sync_server):
    assert json.loads(convo.hermes_conversation_tool("send", id="c_regwatch", text="  "))["ok"] is False
    missing = json.loads(convo.hermes_conversation_tool("send", id="c_nope", text="x"))
    assert missing["ok"] is False and "404" in missing["error"]
    unknown = json.loads(convo.hermes_conversation_tool("frobnicate"))
    assert unknown["ok"] is False and "Unknown action" in unknown["error"]


def test_unreachable_sync_is_an_error_not_a_guess(monkeypatch):
    monkeypatch.setenv("HERMES_SYNC_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("HERMES_SYNC_KEY", "k")
    result = json.loads(convo.hermes_conversation_tool("list"))
    assert result["ok"] is False and "unreachable" in result["error"]


def test_registered_everywhere():
    assert registry.get_entry("hermes_conversation") is not None
    assert "hermes_conversation" in toolsets.TOOLSETS["hermes_conversation"]["tools"]
    from agent.transports import hermes_tools_mcp_server as mcp
    assert "hermes_conversation" in mcp.EXPOSED_TOOLS
