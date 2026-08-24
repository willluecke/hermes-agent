"""Tests for CodexAppServerSession — drive turns through a mock client.

The session adapter has the most complex behavior of the three new modules:
notification draining, server-request handling (approvals), interrupt,
deadline timeouts. These tests pin all of that without spawning real codex.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch
from typing import Any, Optional

import pytest

import agent.transports.codex_app_server_session as session_mod
from agent.transports.codex_app_server_session import (
    CodexAppServerSession,
    _ServerRequestRouting,
    _approval_choice_to_codex_decision,
    _prepare_turn_input_items,
)
from agent.transports.codex_app_server import CodexAppServerError


class FakeClient:
    """Stand-in for CodexAppServerClient that records calls and lets the test
    drive the notification / server-request streams synchronously."""

    def __init__(self, *, codex_bin: str = "codex", codex_home=None) -> None:
        self.codex_bin = codex_bin
        self.codex_home = codex_home
        self.requests: list[tuple[str, dict]] = []
        self.notifications_responses: list[dict] = []
        self.responses: list[tuple[Any, dict]] = []
        self.error_responses: list[tuple[Any, int, str]] = []
        self._initialized = False
        self._closed = False
        self._notifications: list[dict] = []
        self._server_requests: list[dict] = []
        self._request_handler = None  # Optional[Callable[[str, dict], dict]]

    # API matching CodexAppServerClient
    def initialize(self, **kwargs):
        self._initialized = True
        return {"userAgent": "fake/0.0.0", "codexHome": "/tmp",
                "platformOs": "linux", "platformFamily": "unix"}

    def request(self, method: str, params: Optional[dict] = None, timeout: float = 30.0):
        self.requests.append((method, params or {}))
        if self._request_handler is not None:
            return self._request_handler(method, params or {})
        # Sensible defaults for protocol methods used by the session
        if method == "thread/start":
            return {"thread": {"id": "thread-fake-001"},
                    "activePermissionProfile": {"id": "workspace-write"}}
        if method == "thread/resume":
            return {"thread": {"id": (params or {})["threadId"]}}
        if method == "turn/start":
            return {"turn": {"id": "turn-fake-001"}}
        if method == "turn/interrupt":
            return {}
        if method == "turn/steer":
            return {"turnId": (params or {}).get("expectedTurnId")}
        return {}

    def notify(self, method: str, params=None):
        pass

    def respond(self, request_id, result):
        self.responses.append((request_id, result))

    def respond_error(self, request_id, code, message, data=None):
        self.error_responses.append((request_id, code, message))

    def take_notification(self, timeout: float = 0.0):
        if self._notifications:
            return self._notifications.pop(0)
        # Honor a tiny sleep so the loop doesn't hot-spin; the real client
        # blocks on a queue. For tests we want determinism.
        if timeout > 0:
            time.sleep(min(timeout, 0.001))
        return None

    def take_server_request(self, timeout: float = 0.0):
        if self._server_requests:
            return self._server_requests.pop(0)
        return None

    def close(self):
        self._closed = True

    def is_alive(self) -> bool:
        # Fake is "alive" until close() is called; tests that want a dead
        # subprocess can patch this attribute or call close() directly.
        return not self._closed

    def stderr_tail(self, n: int = 20):
        return list(getattr(self, "_stderr_tail", []))[-n:]

    # Test helpers
    def queue_notification(self, method: str, **params):
        # Keep legacy fixture shorthand aligned with the IDs returned by the
        # fake thread/start and turn/start responses.
        if params.get("threadId") in {"t", "th"}:
            params["threadId"] = "thread-fake-001"
        if params.get("turnId") == "tu1":
            params["turnId"] = "turn-fake-001"
        turn = params.get("turn")
        if isinstance(turn, dict) and turn.get("id") == "tu1":
            turn = dict(turn)
            turn["id"] = "turn-fake-001"
            params["turn"] = turn
        self._notifications.append({"method": method, "params": params})

    def queue_server_request(self, method: str, request_id: Any = "srv-1", **params):
        self._server_requests.append({"id": request_id, "method": method, "params": params})

    def set_stderr_tail(self, lines):
        """Test helper: seed stderr_tail() output for OAuth-refresh classifier tests."""
        self._stderr_tail = list(lines)


def make_session(client: FakeClient, **kwargs) -> CodexAppServerSession:
    return CodexAppServerSession(
        cwd="/tmp",
        client_factory=lambda **kw: client,
        **kwargs,
    )


# ---- choice mapping ----

class TestApprovalChoiceMapping:
    @pytest.mark.parametrize("choice,expected", [
        ("once", "accept"),
        ("session", "acceptForSession"),
        ("always", "acceptForSession"),
        ("deny", "decline"),
        ("timeout", "cancel"),
        ("unavailable", "cancel"),
        ("anything-else", "cancel"),
    ])
    def test_mapping(self, choice, expected):
        assert _approval_choice_to_codex_decision(choice) == expected


class TestTurnInputCoercion:
    _PNG_DATA_URL = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/"
        "5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    )

    def test_inline_image_becomes_private_local_image(self):
        items, temp_dir = _prepare_turn_input_items([
            {"type": "text", "text": "caption"},
            {"type": "image_url", "image_url": {"url": self._PNG_DATA_URL}},
        ])
        assert temp_dir is not None
        path = Path(items[1]["path"])
        try:
            assert items[0] == {"type": "text", "text": "caption"}
            assert items[1]["type"] == "localImage"
            assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
            assert path.stat().st_mode & 0o777 == 0o600
        finally:
            temp_dir.cleanup()
        assert not path.exists()

    def test_remote_image_keeps_native_url(self):
        items, temp_dir = _prepare_turn_input_items([
            {"type": "input_image", "image_url": "https://example.com/shot.png"},
        ])
        assert items == [
            {"type": "image", "url": "https://example.com/shot.png"}
        ]
        assert temp_dir is None

    def test_declared_media_type_must_match_bytes(self):
        with pytest.raises(ValueError, match="do not match"):
            _prepare_turn_input_items([
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUg=="
                    },
                }
            ])


# ---- lifecycle ----

class TestLifecycle:
    def test_ensure_started_is_idempotent(self):
        client = FakeClient()
        s = make_session(client)
        tid_a = s.ensure_started()
        tid_b = s.ensure_started()
        assert tid_a == tid_b == "thread-fake-001"
        # thread/start should be called exactly once
        method_calls = [m for (m, _) in client.requests if m == "thread/start"]
        assert len(method_calls) == 1

    def test_thread_start_passes_cwd_only(self):
        """thread/start carries cwd. We intentionally do NOT pass `permissions`
        on this codex version (experimentalApi-gated + requires matching
        config.toml [permissions] table). Letting codex use its default
        (read-only unless user configures otherwise) is the documented path."""
        client = FakeClient()
        s = make_session(client, permission_profile="workspace-write")
        s.ensure_started()
        method, params = next(r for r in client.requests if r[0] == "thread/start")
        assert params["cwd"] == "/tmp"
        assert "permissions" not in params  # see session.ensure_started() comment

    def test_existing_native_thread_is_resumed(self):
        client = FakeClient()
        s = make_session(client, resume_thread_id="thread-existing-123")

        assert s.ensure_started() == "thread-existing-123"
        assert s._resumed_existing_thread is True
        assert [method for method, _ in client.requests].count("thread/resume") == 1
        assert not any(method == "thread/start" for method, _ in client.requests)
        _, params = next(r for r in client.requests if r[0] == "thread/resume")
        assert params == {
            "cwd": "/tmp",
            "threadId": "thread-existing-123",
        }

    def test_failed_resume_starts_fresh_for_durable_history_fallback(self):
        client = FakeClient()

        def handle(method, params):
            if method == "thread/resume":
                raise CodexAppServerError(code=-32602, message="thread missing")
            if method == "thread/start":
                return {"thread": {"id": "thread-recovered-456"}}
            return {}

        client._request_handler = handle
        s = make_session(client, resume_thread_id="thread-missing-123")

        assert s.ensure_started() == "thread-recovered-456"
        assert s._resumed_existing_thread is False
        assert [method for method, _ in client.requests] == [
            "thread/resume",
            "thread/start",
        ]

    def test_close_idempotent(self):
        client = FakeClient()
        s = make_session(client)
        s.ensure_started()
        s.close()
        s.close()
        assert client._closed is True


# ---- turn loop ----

class TestRunTurn:
    def test_simple_text_turn_returns_final_message(self):
        client = FakeClient()
        client.queue_notification("turn/started", threadId="t", turn={"id": "tu1"})
        client.queue_notification(
            "item/completed",
            item={"type": "agentMessage", "id": "m1", "text": "hello world"},
            threadId="t", turnId="tu1",
        )
        client.queue_notification(
            "turn/completed",
            threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )
        s = make_session(client)
        r = s.run_turn("hi", turn_timeout=2.0)
        assert r.final_text == "hello world"
        assert r.interrupted is False
        assert r.error is None
        assert any(m["role"] == "assistant" and m.get("content") == "hello world"
                   for m in r.projected_messages)
        # turn_id propagated for downstream session-DB linkage
        assert r.turn_id == "turn-fake-001"

    def test_inline_image_reaches_turn_start_and_is_cleaned_up(self):
        client = FakeClient()
        client.queue_notification(
            "item/completed",
            item={"type": "agentMessage", "id": "m1", "text": "I see it"},
            threadId="t", turnId="tu1",
        )
        client.queue_notification(
            "turn/completed",
            threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )
        session = make_session(client)
        result = session.run_turn(
            [
                {"type": "text", "text": "Describe this image"},
                {
                    "type": "image_url",
                    "image_url": {"url": TestTurnInputCoercion._PNG_DATA_URL},
                },
            ],
            turn_timeout=2.0,
        )

        turn_params = next(
            params for method, params in client.requests if method == "turn/start"
        )
        local_image = turn_params["input"][1]
        assert turn_params["input"][0] == {
            "type": "text",
            "text": "Describe this image",
        }
        assert local_image["type"] == "localImage"
        assert result.final_text == "I see it"
        assert not Path(local_image["path"]).exists()



    def test_foreign_completion_in_server_request_drain_is_ignored(self):
        """Approval draining must not project a child result into the parent."""
        client = FakeClient()
        client.queue_server_request(
            "item/commandExecution/requestApproval",
            request_id="approval-1",
            command="pwd",
            cwd="/tmp",
        )
        client.queue_notification(
            "item/completed",
            threadId="thread-child-001",
            turnId="turn-child-001",
            item={
                "type": "agentMessage",
                "id": "child-message",
                "text": "child drain summary",
            },
        )
        client.queue_notification(
            "turn/completed",
            threadId="thread-child-001",
            turn={
                "id": "turn-child-001",
                "status": "completed",
                "error": None,
            },
        )

        original_respond = client.respond

        def respond_and_release_parent(request_id, response):
            original_respond(request_id, response)
            client.queue_notification(
                "item/completed",
                threadId="thread-fake-001",
                turnId="turn-fake-001",
                item={
                    "type": "agentMessage",
                    "id": "parent-message",
                    "text": "parent after approval",
                },
            )
            client.queue_notification(
                "turn/completed",
                threadId="thread-fake-001",
                turn={
                    "id": "turn-fake-001",
                    "status": "completed",
                    "error": None,
                },
            )

        client.respond = respond_and_release_parent
        session = make_session(
            client,
            request_routing=_ServerRequestRouting(auto_approve_exec=True),
        )

        result = session.run_turn("delegate then continue", turn_timeout=2.0)

        assert client.responses == [("approval-1", {"decision": "accept"})]
        assert result.final_text == "parent after approval"
        assert result.projected_messages == [
            {"role": "assistant", "content": "parent after approval"}
        ]



    def test_tool_iteration_counter_ticks(self):
        client = FakeClient()
        # Two completed exec items + one final agent message
        for i, item_id in enumerate(("ex1", "ex2"), start=1):
            client.queue_notification(
                "item/completed",
                item={
                    "type": "commandExecution", "id": item_id,
                    "command": f"cmd{i}", "cwd": "/tmp",
                    "status": "completed", "aggregatedOutput": "ok",
                    "exitCode": 0, "commandActions": [],
                },
                threadId="t", turnId="tu1",
            )
        client.queue_notification(
            "item/completed",
            item={"type": "agentMessage", "id": "m1", "text": "done"},
            threadId="t", turnId="tu1",
        )
        client.queue_notification(
            "turn/completed", threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )
        s = make_session(client)
        r = s.run_turn("do stuff", turn_timeout=2.0)
        assert r.tool_iterations == 2
        # Each tool item produces (assistant, tool) — 2*2 + final assistant = 5 msgs
        assert len(r.projected_messages) == 5


    def test_turn_start_failure_attaches_redacted_stderr_tail(self):
        """When codex stderr has content (non-OAuth), the tail gets attached
        to the user-facing error so config/provider problems are debuggable
        instead of just 'Internal error'. Credential-shaped values in stderr
        are redacted via agent.redact(force=True); web-URL query params pass
        through (see fix(redact): pass web URLs through unchanged)."""
        client = FakeClient()
        client.set_stderr_tail([
            "ERROR: provider auth failed",
            "Authorization: Bearer sk-live-deadbeefdeadbeef",
            "url=https://api.example.com/v1?token=querysecret12345",
        ])
        from agent.transports.codex_app_server import CodexAppServerError

        def boom(method, params):
            if method == "turn/start":
                raise CodexAppServerError(code=-32603, message="Internal error")
            return {"thread": {"id": "t"}, "activePermissionProfile": {"id": "x"}}

        client._request_handler = boom
        s = make_session(client)
        r = s.run_turn("hi", turn_timeout=2.0)
        assert r.error is not None
        assert "turn/start failed" in r.error
        assert "Internal error" in r.error
        # Stderr tail attached
        assert "codex stderr" in r.error
        assert "provider auth failed" in r.error
        # Credential-shaped values still redacted (sk- prefix + Bearer header)
        assert "sk-live-deadbeefdeadbeef" not in r.error
        # Non-OAuth → should NOT retire (subprocess JSON-RPC is still healthy).
        assert r.should_retire is False

    def test_turn_start_timeout_attaches_redacted_stderr_tail(self):
        """A non-OAuth TimeoutError on turn/start surfaces with codex stderr
        context attached and marks the session for retirement."""
        client = FakeClient()
        client.set_stderr_tail([
            "WARN: provider request stalled",
            "Authorization: Bearer sk-stalled-secret-abc123",
        ])

        def stall(method, params):
            if method == "turn/start":
                raise TimeoutError("codex method 'turn/start' timed out after 10s")
            return {"thread": {"id": "t"}, "activePermissionProfile": {"id": "x"}}

        client._request_handler = stall
        s = make_session(client)
        r = s.run_turn("hi", turn_timeout=2.0)
        assert r.error is not None
        assert "turn/start timed out" in r.error
        assert "provider request stalled" in r.error
        assert "sk-stalled-secret-abc123" not in r.error
        assert r.should_retire is True




    def test_steer_appends_input_to_active_turn(self):
        client = FakeClient()
        s = make_session(client)
        s.ensure_started()
        with s._active_turn_lock:
            s._active_turn_id = "turn-live-123"

        assert s.request_steer("Use Postgres instead") is True
        method, params = client.requests[-1]
        assert method == "turn/steer"
        assert params == {
            "threadId": "thread-fake-001",
            "input": [{"type": "text", "text": "Use Postgres instead"}],
            "expectedTurnId": "turn-live-123",
        }







class TestCompactThread:
    def test_compact_thread_sends_rpc_and_waits_for_completion(self):
        client = FakeClient()
        client.queue_notification(
            "turn/started",
            threadId="thread-fake-001",
            turn={"id": "compact-turn-1"},
        )
        client.queue_notification(
            "item/completed",
            threadId="thread-fake-001",
            turnId="compact-turn-1",
            item={"type": "contextCompaction", "id": "compact-item-1"},
        )
        client.queue_notification(
            "item/completed",
            threadId="thread-fake-001",
            turnId="compact-turn-1",
            item={"type": "agentMessage", "id": "m1", "text": "compacted"},
        )
        client.queue_notification(
            "thread/tokenUsage/updated",
            threadId="thread-fake-001",
            turnId="compact-turn-1",
            tokenUsage={
                "last": {"inputTokens": 10, "outputTokens": 2, "totalTokens": 12},
                "total": {"inputTokens": 100, "outputTokens": 20, "totalTokens": 120},
                "modelContextWindow": 200000,
            },
        )
        client.queue_notification(
            "turn/completed",
            threadId="thread-fake-001",
            turn={"id": "compact-turn-1", "status": "completed", "error": None},
        )

        r = make_session(client).compact_thread(turn_timeout=2.0)

        assert ("thread/compact/start", {"threadId": "thread-fake-001"}) in client.requests
        assert r.error is None
        assert r.thread_id == "thread-fake-001"
        assert r.turn_id == "compact-turn-1"
        assert r.compacted is True
        assert r.final_text == "compacted"
        assert r.token_usage_last["totalTokens"] == 12
        assert r.model_context_window == 200000

    def test_compact_thread_ignores_foreign_child_completion(self):
        client = FakeClient()
        client.queue_notification(
            "turn/started",
            threadId="thread-child-001",
            turn={"id": "child-compact-turn"},
        )
        client.queue_notification(
            "item/completed",
            threadId="thread-child-001",
            turnId="child-compact-turn",
            item={
                "type": "agentMessage",
                "id": "child-compact-message",
                "text": "child compact summary",
            },
        )
        client.queue_notification(
            "turn/completed",
            threadId="thread-child-001",
            turn={
                "id": "child-compact-turn",
                "status": "completed",
                "error": None,
            },
        )
        client.queue_notification(
            "turn/started",
            threadId="thread-fake-001",
            turn={"id": "compact-turn-1"},
        )
        client.queue_notification(
            "item/completed",
            threadId="thread-fake-001",
            turnId="compact-turn-1",
            item={
                "type": "agentMessage",
                "id": "parent-compact-message",
                "text": "parent compacted",
            },
        )
        client.queue_notification(
            "turn/completed",
            threadId="thread-fake-001",
            turn={
                "id": "compact-turn-1",
                "status": "completed",
                "error": None,
            },
        )

        result = make_session(client).compact_thread(turn_timeout=2.0)

        assert result.error is None
        assert result.turn_id == "compact-turn-1"
        assert result.final_text == "parent compacted"
        assert result.projected_messages == [
            {"role": "assistant", "content": "parent compacted"}
        ]





# ---- approval bridge ----

class TestServerRequestRouting:

    @pytest.mark.parametrize(
        "choice,expected",
        [
            ("once", "accept"),
            ("deny", "decline"),
            ("timeout", "cancel"),
            ("unavailable", "cancel"),
        ],
    )
    def test_exec_approval_outcome_reaches_codex(self, choice, expected):
        client = FakeClient()
        client.queue_server_request(
            "item/commandExecution/requestApproval",
            request_id="exec-approval",
            command="git push origin main",
            cwd="/workspace",
        )
        client.queue_notification(
            "turn/completed",
            threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )

        def callback(command, description, *, allow_permanent=True):
            return choice

        make_session(client, approval_callback=callback).run_turn(
            "continue", turn_timeout=1.0
        )

        assert ("exec-approval", {"decision": expected}) in client.responses

    def test_missing_approval_callback_cancels_instead_of_declining(self):
        client = FakeClient()
        client.queue_server_request(
            "item/commandExecution/requestApproval",
            request_id="exec-unavailable",
            command="git push origin main",
            cwd="/workspace",
        )
        client.queue_notification(
            "turn/completed",
            threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )

        make_session(client).run_turn("continue", turn_timeout=1.0)

        assert (
            "exec-unavailable",
            {"decision": "cancel"},
        ) in client.responses

    @pytest.mark.parametrize("server_name", ["hermes-tools", "reccli"])
    def test_local_runtime_mcp_elicitation_is_accepted(self, server_name):
        client = FakeClient()
        client.queue_server_request(
            "mcpServer/elicitation/request",
            request_id="mcp-approval",
            serverName=server_name,
            mode="form",
            message="Allow this local MCP tool call?",
            requestedSchema={"type": "object"},
        )
        client.queue_notification(
            "turn/completed", threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )

        make_session(client).run_turn("hi", turn_timeout=1.0)

        assert (
            "mcp-approval",
            {"action": "accept", "content": None, "_meta": None},
        ) in client.responses

    def test_untrusted_mcp_elicitation_is_declined(self):
        client = FakeClient()
        client.queue_server_request(
            "mcpServer/elicitation/request",
            request_id="mcp-approval",
            serverName="external-server",
            mode="form",
            message="Allow this external MCP tool call?",
            requestedSchema={"type": "object"},
        )
        client.queue_notification(
            "turn/completed", threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )

        make_session(client).run_turn("hi", turn_timeout=1.0)

        assert (
            "mcp-approval",
            {"action": "decline", "content": None, "_meta": None},
        ) in client.responses



    def test_unknown_server_request_replied_with_error(self):
        client = FakeClient()
        client.queue_server_request("totally/unknown", request_id="req-3")
        client.queue_notification(
            "turn/completed", threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )
        s = make_session(client)
        s.run_turn("hi", turn_timeout=1.0)
        assert any(
            rid == "req-3" and code == -32601
            for (rid, code, _msg) in client.error_responses
        )

    def test_on_event_fires_during_approval_drain(self):
        """When a server-initiated approval request arrives, the session
        drains up to 8 pending notifications first so per-turn state
        (e.g. _pending_file_changes for fileChange approvals) is current.
        Those drained notifications must also reach the on_event display
        hook — otherwise tool bubbles around approvals silently disappear.

        Regression for the issue where item/started events that landed
        in the queue alongside (or just before) an approval request got
        projected into messages but never displayed.
        """
        client = FakeClient()
        # An item/started notification is queued first, then a server
        # request — the session sees both during a single drain loop.
        client.queue_notification(
            "item/started",
            item={
                "type": "commandExecution",
                "id": "exec-1",
                "command": "echo drained",
                "cwd": "/tmp",
            },
        )
        client.queue_server_request(
            "item/commandExecution/requestApproval", request_id="req-d",
            command="echo drained",
            cwd="/tmp",
        )
        client.queue_notification(
            "turn/completed", threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )

        events: list[dict] = []

        def cb(command, description, *, allow_permanent=True):
            return "once"

        s = make_session(
            client,
            approval_callback=cb,
            on_event=events.append,
        )
        s.run_turn("hi", turn_timeout=1.0)

        # The on_event hook must have seen the item/started even though
        # it was drained as part of the approval roundtrip — not just
        # events that arrive on the main notification path.
        item_started_events = [
            e for e in events
            if e.get("method") == "item/started"
        ]
        assert item_started_events, (
            "item/started drained alongside the approval was not "
            "forwarded to on_event — display will miss tool bubbles "
            "around approvals"
        )



    def test_routing_auto_approve_bypass(self):
        client = FakeClient()
        client.queue_server_request("item/commandExecution/requestApproval", request_id="r1",
                                    command="ls", cwd="/")
        client.queue_notification(
            "turn/completed", threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )
        # No callback, but routing says auto-approve. Should approve.
        s = make_session(client, request_routing=_ServerRequestRouting(
            auto_approve_exec=True))
        s.run_turn("hi", turn_timeout=1.0)
        assert ("r1", {"decision": "accept"}) in client.responses

    @pytest.mark.parametrize(
        "command",
        [
            "rm -rf ./build",
            "git reset --hard",
            "git push --force origin main",
            "sudo -S id",
            "",
        ],
    )
    def test_bounded_no_prompt_declines_guarded_exec(self, command):
        session = make_session(
            FakeClient(),
            request_routing=_ServerRequestRouting(
                auto_approve_exec=True,
                guard_no_prompt_exec=True,
            ),
        )

        assert session._decide_exec_approval({"command": command}) == "decline"

    @pytest.mark.parametrize(
        "command",
        [
            "git status --short",
            "git push origin main",
            "npm test",
            "python3 -m pytest -q",
        ],
    )
    def test_bounded_no_prompt_accepts_routine_exec(self, command):
        session = make_session(
            FakeClient(),
            request_routing=_ServerRequestRouting(
                auto_approve_exec=True,
                guard_no_prompt_exec=True,
            ),
        )

        assert session._decide_exec_approval({"command": command}) == "accept"

    @pytest.mark.parametrize(
        ("kind", "expected"),
        [
            ("add", "accept"),
            ("update", "accept"),
            ("delete", "decline"),
            ("rename", "decline"),
        ],
    )
    def test_bounded_no_prompt_file_change_policy(self, kind, expected):
        session = make_session(
            FakeClient(),
            request_routing=_ServerRequestRouting(
                auto_approve_apply_patch=True,
                guard_no_prompt_file_changes=True,
            ),
        )
        session._track_pending_file_change(
            {
                "method": "item/started",
                "params": {
                    "item": {
                        "type": "fileChange",
                        "id": "fc-guarded",
                        "changes": [
                            {"kind": {"type": kind}, "path": "/tmp/file.txt"}
                        ],
                    }
                },
            }
        )

        assert session._decide_apply_patch_approval(
            {"itemId": "fc-guarded"}
        ) == expected

    def test_bounded_no_prompt_declines_uninspectable_file_change(self):
        session = make_session(
            FakeClient(),
            request_routing=_ServerRequestRouting(
                auto_approve_apply_patch=True,
                guard_no_prompt_file_changes=True,
            ),
        )

        assert session._decide_apply_patch_approval(
            {"itemId": "missing"}
        ) == "decline"



# ---- enriched approval prompts ----

class TestApprovalPromptEnrichment:
    """Quirk #4: apply_patch prompt should show what's changing.
    Quirk #10: exec prompt should never show empty cwd."""

    def test_exec_falls_back_to_session_cwd(self):
        """When codex omits cwd from the approval params, the prompt shows
        the session cwd, not an empty string."""
        client = FakeClient()
        client.queue_server_request(
            "item/commandExecution/requestApproval", request_id="r1",
            command="ls",  # no cwd
        )
        client.queue_notification(
            "turn/completed", threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )
        captured = {}
        def cb(command, description, *, allow_permanent=True):
            captured["description"] = description
            return "once"
        s = make_session(client, approval_callback=cb)
        s.run_turn("hi", turn_timeout=1.0)
        # Session cwd is /tmp by default in make_session()
        assert "/tmp" in captured["description"]
        assert "Codex requests exec in <unknown>" not in captured["description"]

    def test_apply_patch_prompt_summarizes_pending_changes(self):
        """When the projector has cached the fileChange item from item/started,
        the approval prompt surfaces the change summary."""
        client = FakeClient()
        # item/started fires first (carries the changes), then approval request
        client.queue_notification(
            "item/started",
            item={"type": "fileChange", "id": "fc-1",
                  "changes": [
                      {"kind": {"type": "add"}, "path": "/tmp/new.py"},
                      {"kind": {"type": "update"}, "path": "/tmp/old.py"},
                  ]},
            threadId="t", turnId="tu1",
        )
        client.queue_server_request(
            "item/fileChange/requestApproval", request_id="req-2",
            itemId="fc-1", turnId="tu1", threadId="t",
            startedAtMs=1234567890,
            reason="add and update files",
        )
        client.queue_notification(
            "turn/completed", threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )
        captured = {}
        def cb(command, description, *, allow_permanent=True):
            captured["command"] = command
            captured["description"] = description
            return "once"
        s = make_session(client, approval_callback=cb)
        s.run_turn("hi", turn_timeout=1.0)
        # Both add and update kinds should be in the summary
        assert "1 add" in captured["command"] or "1 add" in captured["description"]
        assert "1 update" in captured["command"] or "1 update" in captured["description"]
        # And at least one of the paths
        joined = captured["command"] + " " + captured["description"]
        assert "/tmp/new.py" in joined or "/tmp/old.py" in joined

    def test_apply_patch_prompt_works_without_cached_summary(self):
        """When approval arrives before item/started (or without changes
        info), prompt falls back to whatever codex provided."""
        client = FakeClient()
        client.queue_server_request(
            "item/fileChange/requestApproval", request_id="req-2",
            itemId="fc-orphan", turnId="tu1", threadId="t",
            startedAtMs=1234567890,
            reason="apply some changes",
        )
        client.queue_notification(
            "turn/completed", threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )
        captured = {}
        def cb(command, description, *, allow_permanent=True):
            captured["command"] = command
            return "once"
        s = make_session(client, approval_callback=cb)
        s.run_turn("hi", turn_timeout=1.0)
        # Falls back to the reason
        assert "apply some changes" in captured["command"]


# ---- openclaw beta.8 parity: retire/wedge/oauth/abort marker ----

class TestSessionRetirement:
    """Mirrors openclaw beta.8's resilience fixes:
      - retire timed-out app-server clients (should_retire on deadline)
      - bounded turn inactivity and absolute deadlines
      - <turn_aborted> raw marker as terminal (don't wait for turn/completed
        that never comes)
      - OAuth refresh failure classification (suggest `codex login` instead
        of raw RPC error strings)
      - dead subprocess detection between iterations
    """



    def test_final_agent_message_without_turn_completed_is_recovered(self):
        """A completed assistant item is still a usable terminal response when
        codex omits turn/completed and then goes quiet.
        """
        client = FakeClient()
        client.queue_notification(
            "item/completed",
            item={
                "type": "agentMessage",
                "id": "m1",
                "phase": "final_answer",
                "text": "done",
            },
            threadId="t",
            turnId="tu1",
        )
        s = make_session(client)
        r = s.run_turn(
            "hi",
            turn_timeout=0.05,
            notification_poll_timeout=0.01,
        )
        assert r.final_text == "done"
        assert r.interrupted is False
        assert r.error is None
        assert r.should_retire is False
        assert any(
            msg["role"] == "assistant" and msg.get("content") == "done"
            for msg in r.projected_messages
        )
        assert not any(method == "turn/interrupt" for method, _ in client.requests)

    def test_commentary_without_turn_completed_times_out_instead_of_completing(self):
        client = FakeClient()
        client.queue_notification(
            "item/completed",
            item={
                "type": "agentMessage",
                "id": "m-commentary",
                "phase": "commentary",
                "text": "I am still validating the catalog.",
            },
            threadId="t",
            turnId="tu1",
        )
        s = make_session(client)
        r = s.run_turn(
            "validate",
            turn_timeout=0.05,
            notification_poll_timeout=0.01,
        )
        assert r.final_text == ""
        assert r.final_answer_seen is False
        assert r.interrupted is True
        assert r.should_retire is True
        assert r.error and "timed out" in r.error
        assert any(
            msg.get("phase") == "commentary"
            and msg.get("content") == "I am still validating the catalog."
            for msg in r.projected_messages
        )
        assert any(method == "turn/interrupt" for method, _ in client.requests)

    def test_active_turn_progress_refreshes_inactivity_deadline(self):
        """Total turn time may exceed the timeout while current-turn events
        keep arriving inside the inactivity window.
        """
        client = FakeClient()
        client.queue_notification(
            "item/completed",
            item={
                "type": "agentMessage",
                "id": "m-commentary-1",
                "phase": "commentary",
                "text": "Still working.",
            },
            threadId="t",
            turnId="tu1",
        )
        client.queue_notification(
            "item/completed",
            item={
                "type": "agentMessage",
                "id": "m-commentary-2",
                "phase": "commentary",
                "text": "Validation is progressing.",
            },
            threadId="t",
            turnId="tu1",
        )
        client.queue_notification(
            "item/completed",
            item={
                "type": "agentMessage",
                "id": "m-final",
                "phase": "final_answer",
                "text": "Validation completed.",
            },
            threadId="t",
            turnId="tu1",
        )
        client.queue_notification(
            "turn/completed",
            threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )
        original_take_notification = client.take_notification

        def delayed_notification(timeout):
            time.sleep(0.03)
            return original_take_notification(timeout)

        client.take_notification = delayed_notification
        result = make_session(client).run_turn(
            "validate",
            turn_timeout=0.05,
            notification_poll_timeout=0.01,
        )

        assert result.final_text == "Validation completed."
        assert result.final_answer_seen is True
        assert result.interrupted is False
        assert result.error is None

    def test_phase_less_message_cannot_replace_explicit_final_at_deadline(self):
        client = FakeClient()
        client.queue_notification(
            "item/completed",
            item={
                "type": "agentMessage",
                "id": "m-final",
                "phase": "final_answer",
                "text": "authoritative answer",
            },
            threadId="t",
            turnId="tu1",
        )
        client.queue_notification(
            "item/completed",
            item={
                "type": "agentMessage",
                "id": "m-legacy",
                "text": "late unclassified text",
            },
            threadId="t",
            turnId="tu1",
        )
        s = make_session(client)
        r = s.run_turn(
            "validate",
            turn_timeout=0.05,
            notification_poll_timeout=0.01,
        )
        assert r.final_text == "authoritative answer"
        assert r.final_answer_seen is True
        assert r.interrupted is False


    def test_tool_completion_does_not_shorten_inactivity_timeout(self):
        """A quiet reasoning phase after a tool uses the normal turn timeout.

        The former 90-second post-tool watchdog interrupted xhigh reasoning
        before Codex could emit its next item. Advance the monotonic clock by
        100 seconds between the tool and final answer to pin the regression.
        """
        client = FakeClient()
        client.queue_notification(
            "item/completed",
            item={
                "type": "commandExecution", "id": "ex1",
                "command": "echo hi", "cwd": "/tmp",
                "status": "completed", "aggregatedOutput": "hi",
                "exitCode": 0, "commandActions": [],
            },
            threadId="t", turnId="tu1",
        )
        client.queue_notification(
            "item/completed",
            item={
                "type": "agentMessage",
                "id": "m1",
                "text": "reasoning finished",
                "phase": "final_answer",
            },
            threadId="t", turnId="tu1",
        )
        client.queue_notification(
            "turn/completed", threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )
        s = make_session(client)
        monotonic_values = iter([
            1000.0, 1000.0, 1000.0,
            1100.0, 1100.0,
            1100.0, 1100.0,
        ])
        with patch.object(
            session_mod.time,
            "monotonic",
            side_effect=lambda: next(monotonic_values),
        ):
            r = s.run_turn(
                "tool then silence",
                turn_timeout=600.0,
                notification_poll_timeout=0.0,
            )
        assert r.final_text == "reasoning finished"
        assert r.interrupted is False
        assert r.should_retire is False
        assert r.error is None

    def test_absolute_turn_timeout_interrupts_runaway_turn(self):
        client = FakeClient()
        s = make_session(client)
        monotonic_values = iter([1000.0, 1000.0, 1000.2])
        with patch.object(
            session_mod.time,
            "monotonic",
            side_effect=lambda: next(monotonic_values),
        ):
            r = s.run_turn(
                "never completes",
                turn_timeout=600.0,
                notification_poll_timeout=0.0,
                absolute_turn_timeout=0.15,
            )
        assert r.interrupted is True
        assert r.should_retire is True
        assert r.error and "absolute timeout of 0.15s" in r.error







    def test_dead_subprocess_detected_between_iterations(self):
        """If codex dies (segfault, OOM, killed by its auth refresh
        thread), the inter-iteration is_alive check breaks the loop
        instead of waiting on a queue that will never fill."""
        client = FakeClient()
        s = make_session(client)
        s.ensure_started()
        # Simulate subprocess death by setting _closed (FakeClient's
        # is_alive returns False when closed).
        client._closed = True
        client.set_stderr_tail([
            "thread 'tokio-runtime-worker' panicked at 'oauth: invalid_grant'",
        ])
        r = s.run_turn("x", turn_timeout=2.0,
                       notification_poll_timeout=0.01)
        assert r.should_retire is True
        # Stderr-derived auth hint takes precedence over generic message
        assert r.error and "codex login" in r.error


# ---- thread/start cross-fill ----

class TestThreadStartCrossFill:
    """Mirrors openclaw beta.8's tolerance for thread.id/sessionId aliasing."""

    def test_thread_id_under_thread_key(self):
        client = FakeClient()
        s = make_session(client)
        tid = s.ensure_started()
        assert tid == "thread-fake-001"



    def test_missing_thread_id_raises(self):
        from agent.transports.codex_app_server import CodexAppServerError

        client = FakeClient()
        client._request_handler = lambda method, params: (
            {"thread": {}, "activePermissionProfile": {"id": "x"}}
            if method == "thread/start" else
            {"turn": {"id": "tu1"}}
        )
        s = make_session(client)
        with pytest.raises(CodexAppServerError, match="no thread id"):
            s.ensure_started()


class TestHasTurnAbortedMarker:
    """Unit coverage for the marker matcher itself."""

    def test_empty_string(self):
        from agent.transports.codex_app_server_session import (
            _has_turn_aborted_marker,
        )
        assert _has_turn_aborted_marker("") is False
        assert _has_turn_aborted_marker(None) is False  # type: ignore[arg-type]

    def test_plain_text_no_marker(self):
        from agent.transports.codex_app_server_session import (
            _has_turn_aborted_marker,
        )
        assert _has_turn_aborted_marker("normal response with no markers") is False

    def test_open_marker(self):
        from agent.transports.codex_app_server_session import (
            _has_turn_aborted_marker,
        )
        assert _has_turn_aborted_marker("blah <turn_aborted> blah") is True



class TestClassifyOAuthFailure:
    """Unit coverage for the OAuth classifier; conservative on purpose."""



    def test_401_classified(self):
        from agent.transports.codex_app_server_session import (
            _classify_oauth_failure,
        )
        hint = _classify_oauth_failure("HTTP 401 Unauthorized")
        assert hint is not None


    def test_empty_inputs(self):
        from agent.transports.codex_app_server_session import (
            _classify_oauth_failure,
        )
        assert _classify_oauth_failure() is None
        assert _classify_oauth_failure("") is None
        assert _classify_oauth_failure("", None) is None  # type: ignore[arg-type]
