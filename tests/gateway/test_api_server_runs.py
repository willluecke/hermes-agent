"""Tests for /v1/runs endpoints: start, status, events, steer, and stop.

Covers:
- POST /v1/runs — start a run (202)
- GET /v1/runs/{run_id} — poll run status
- GET /v1/runs/{run_id}/events — SSE event stream
- POST /v1/runs/{run_id}/steer — inject guidance into a running agent
- POST /v1/runs/{run_id}/stop — interrupt a running agent
- Auth, error handling, and cleanup
"""

import asyncio
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    ResponseStore,
    _ReplayableRunEventStream,
    _api_goal_command_args,
    _approval_event_choices,
    _resolve_run_workspace,
    cors_middleware,
    security_headers_middleware,
)
from tools import approval as approval_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("smart_denied", "allow_permanent", "expected"),
    [
        (False, True, ["once", "session", "always", "deny"]),
        (False, False, ["once", "session", "deny"]),
        (True, True, ["once", "deny"]),
        (True, False, ["once", "deny"]),
    ],
)
def test_approval_event_choices_follow_backend_capabilities(
    smart_denied, allow_permanent, expected
):
    assert _approval_event_choices(
        smart_denied=smart_denied,
        allow_permanent=allow_permanent,
    ) == expected


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    """Create an adapter with optional API key."""
    extra = {}
    if api_key:
        extra["key"] = api_key
    config = PlatformConfig(enabled=True, extra=extra)
    adapter = APIServerAdapter(config)
    return adapter


def _create_runs_app(adapter: APIServerAdapter) -> web.Application:
    """Create an aiohttp app with /v1/runs routes registered."""
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)
    app.router.add_post("/v1/runs/{run_id}/approval", adapter._handle_run_approval)
    app.router.add_post("/v1/runs/{run_id}/steer", adapter._handle_steer_run)
    app.router.add_post("/v1/runs/{run_id}/stop", adapter._handle_stop_run)
    return app


def _make_slow_agent(**kwargs):
    """Create a mock agent that blocks in run_conversation until interrupted.

    Returns (mock_agent, agent_ready_event, interrupt_event) where
    agent_ready_event is set once run_conversation starts, and
    interrupt_event is set when interrupt() is called.
    """
    ready = threading.Event()
    interrupted = threading.Event()

    mock_agent = MagicMock()

    def _do_interrupt(message=None):
        interrupted.set()

    mock_agent.interrupt = MagicMock(side_effect=_do_interrupt)

    def _slow_run(user_message=None, conversation_history=None, task_id=None):
        ready.set()
        # Block until interrupt() is called
        interrupted.wait(timeout=10)
        return {"final_response": "interrupted"}

    mock_agent.run_conversation.side_effect = _slow_run
    mock_agent.session_prompt_tokens = 0
    mock_agent.session_completion_tokens = 0
    mock_agent.session_total_tokens = 0

    return mock_agent, ready, interrupted


def test_durable_run_claim_is_atomic_across_store_connections(tmp_path):
    """Two gateway processes sharing one profile must mint one run id."""
    db_path = str(tmp_path / "response_store.db")
    first_store = ResponseStore(db_path=db_path)
    second_store = ResponseStore(db_path=db_path)
    barrier = threading.Barrier(2)

    def _claim(store, run_id):
        barrier.wait(timeout=2)
        return store.claim_run_idempotency(
            "api_server.runs:default",
            "same-turn",
            "same-fingerprint",
            run_id,
        )

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = [
                future.result(timeout=5)
                for future in (
                    pool.submit(_claim, first_store, "run_first"),
                    pool.submit(_claim, second_store, "run_second"),
                )
            ]
    finally:
        first_store.close()
        second_store.close()

    assert sorted(state for state, _run_id in results) == ["claimed", "replay"]
    assert len({_run_id for _state, _run_id in results}) == 1


@pytest.fixture
def adapter():
    return _make_adapter()


@pytest.fixture
def auth_adapter():
    return _make_adapter(api_key="sk-secret")


# ---------------------------------------------------------------------------
# POST /v1/runs — start a run
# ---------------------------------------------------------------------------


class TestStartRun:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("/goal ship it", "ship it"),
            ("  /goal status  ", "status"),
            ("/goal", ""),
            ("/goals ship it", None),
            ("explain /goal ship it", None),
            ([{"type": "text", "text": "/goal ship it"}], None),
        ],
    )
    def test_goal_command_recognition_is_exact(self, value, expected):
        assert _api_goal_command_args(value) == expected

    def test_workspace_resolver_accepts_only_server_configured_keys(self, tmp_path):
        project_root = tmp_path / "reg-watch"
        project_root.mkdir()
        config = {
            "codex_runtime": {
                "workspaces": {
                    "default_project": "reg-watch",
                    "projects": {"reg-watch": str(project_root)},
                }
            }
        }

        assert _resolve_run_workspace(config, None) == (
            "reg-watch",
            str(project_root.resolve()),
            None,
            0,
        )
        project, cwd, error, status = _resolve_run_workspace(config, "/etc")
        assert (project, cwd, status) == ("", "", 400)
        assert "configured project key" in error

    def test_workspace_resolver_reloads_server_owned_project_registry(self, tmp_path):
        first = tmp_path / "first"
        second = tmp_path / "second"
        first.mkdir()
        second.mkdir()
        registry = tmp_path / "projects.json"
        registry.write_text(json.dumps({"first": str(first)}), encoding="utf-8")
        config = {
            "codex_runtime": {
                "workspaces": {
                    "default_project": "first",
                    "projects_file": str(registry),
                }
            }
        }

        assert _resolve_run_workspace(config, "first")[:2] == (
            "first",
            str(first.resolve()),
        )
        registry.write_text(json.dumps({"second": str(second)}), encoding="utf-8")
        assert _resolve_run_workspace(config, "second")[:2] == (
            "second",
            str(second.resolve()),
        )
        assert _resolve_run_workspace(config, "first")[3] == 400

    def test_workspace_resolver_fails_closed_on_broken_project_registry(self, tmp_path):
        registry = tmp_path / "projects.json"
        registry.write_text("not json", encoding="utf-8")
        config = {
            "codex_runtime": {
                "workspaces": {"projects_file": str(registry)}
            }
        }

        project, cwd, error, status = _resolve_run_workspace(config, "anything")
        assert (project, cwd, status) == ("", "", 503)
        assert "registry" in error

    @pytest.mark.asyncio
    async def test_start_binds_server_resolved_project_cwd(self, adapter, tmp_path):
        project_root = tmp_path / "reg-watch"
        project_root.mkdir()
        config = {
            "codex_runtime": {
                "workspaces": {
                    "default_project": "reg-watch",
                    "projects": {"reg-watch": str(project_root)},
                }
            }
        }
        captured = {}
        mock_agent = MagicMock()

        def _capture_run(user_message=None, conversation_history=None, task_id=None):
            from agent.runtime_cwd import resolve_agent_cwd

            captured["cwd"] = str(resolve_agent_cwd())
            captured["agent_cwd"] = mock_agent.session_cwd
            return {"final_response": "done"}

        mock_agent.run_conversation.side_effect = _capture_run
        mock_agent.session_prompt_tokens = 0
        mock_agent.session_completion_tokens = 0
        mock_agent.session_total_tokens = 0
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch(
                "gateway.run._load_gateway_config", return_value=config
            ), patch.object(
                adapter, "_create_agent", return_value=mock_agent
            ):
                response = await cli.post(
                    "/v1/runs",
                    json={"input": "hello", "project": "reg-watch"},
                )
                assert response.status == 202
                run_id = (await response.json())["run_id"]
                for _ in range(80):
                    status = await (await cli.get(f"/v1/runs/{run_id}")).json()
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.025)

        assert captured == {
            "cwd": str(project_root.resolve()),
            "agent_cwd": str(project_root.resolve()),
        }
        assert status["project"] == "reg-watch"

    @pytest.mark.asyncio
    async def test_start_rejects_unknown_project_before_agent_creation(
        self, adapter, tmp_path
    ):
        config = {
            "codex_runtime": {
                "workspaces": {
                    "default_project": "reg-watch",
                    "projects": {"reg-watch": str(tmp_path)},
                }
            }
        }
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch(
                "gateway.run._load_gateway_config", return_value=config
            ), patch.object(adapter, "_create_agent") as create_agent:
                response = await cli.post(
                    "/v1/runs",
                    json={"input": "hello", "project": "../../etc"},
                )

        assert response.status == 400
        create_agent.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_returns_202(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 10
                mock_agent.session_completion_tokens = 5
                mock_agent.session_total_tokens = 15
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                assert data["status"] == "started"
                assert data["run_id"].startswith("run_")

                status_resp = await cli.get(f"/v1/runs/{data['run_id']}")
                assert status_resp.status == 200
                status = await status_resp.json()
                assert status["run_id"] == data["run_id"]
                assert status["status"] in {"queued", "running", "completed"}
                assert status["object"] == "hermes.run"

    @pytest.mark.asyncio
    async def test_idempotency_replay_returns_original_active_run_before_limit(
        self, adapter, tmp_path
    ):
        """A retry must converge even while its original owns the last slot."""
        adapter._response_store.close()
        adapter._response_store = ResponseStore(
            db_path=str(tmp_path / "response_store.db")
        )
        adapter._max_concurrent_runs = 1
        mock_agent, ready, interrupted = _make_slow_agent()
        app = _create_runs_app(adapter)
        headers = {"Idempotency-Key": "turn-active-replay"}

        try:
            async with TestClient(TestServer(app)) as cli:
                with patch.object(
                    adapter, "_create_agent", return_value=mock_agent
                ) as create_agent:
                    first = await cli.post(
                        "/v1/runs", json={"input": "ship it"}, headers=headers
                    )
                    assert first.status == 202
                    first_data = await first.json()
                    assert await asyncio.to_thread(ready.wait, 2)

                    replay = await cli.post(
                        "/v1/runs", json={"input": "ship it"}, headers=headers
                    )
                    replay_data = await replay.json()

                    assert replay.status == 202
                    assert replay_data == {
                        "run_id": first_data["run_id"],
                        "status": "started",
                        "idempotent_replay": True,
                    }
                    assert replay.headers["Idempotency-Replayed"] == "true"
                    create_agent.assert_called_once()
                    interrupted.set()
                    for _ in range(80):
                        if not adapter._active_run_tasks:
                            break
                        await asyncio.sleep(0.025)
        finally:
            interrupted.set()
            adapter._response_store.close()

    @pytest.mark.asyncio
    async def test_idempotency_key_reuse_with_different_request_conflicts(
        self, adapter, tmp_path
    ):
        adapter._response_store.close()
        adapter._response_store = ResponseStore(
            db_path=str(tmp_path / "response_store.db")
        )
        mock_agent, ready, interrupted = _make_slow_agent()
        app = _create_runs_app(adapter)
        headers = {"Idempotency-Key": "turn-conflict"}

        try:
            async with TestClient(TestServer(app)) as cli:
                with patch.object(
                    adapter, "_create_agent", return_value=mock_agent
                ) as create_agent:
                    first = await cli.post(
                        "/v1/runs", json={"input": "first"}, headers=headers
                    )
                    assert first.status == 202
                    first_run_id = (await first.json())["run_id"]
                    assert await asyncio.to_thread(ready.wait, 2)

                    conflict = await cli.post(
                        "/v1/runs", json={"input": "different"}, headers=headers
                    )
                    conflict_data = await conflict.json()

                    assert conflict.status == 409
                    assert conflict_data["error"]["code"] == "idempotency_conflict"
                    create_agent.assert_called_once()
                    interrupted.set()
                    for _ in range(80):
                        status = await (
                            await cli.get(f"/v1/runs/{first_run_id}")
                        ).json()
                        if status["status"] == "completed":
                            break
                        await asyncio.sleep(0.025)
        finally:
            interrupted.set()
            adapter._response_store.close()

    @pytest.mark.asyncio
    async def test_idempotency_replay_survives_gateway_adapter_restart(self, tmp_path):
        store_path = tmp_path / "response_store.db"
        request_body = {
            "input": "persist this turn",
            "session_id": "durable-idempotency-session",
        }
        headers = {"Idempotency-Key": "turn-after-restart"}

        first_adapter = _make_adapter()
        first_adapter._response_store.close()
        first_adapter._response_store = ResponseStore(db_path=str(store_path))
        first_app = _create_runs_app(first_adapter)
        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = {"final_response": "done"}
        mock_agent.session_prompt_tokens = 0
        mock_agent.session_completion_tokens = 0
        mock_agent.session_total_tokens = 0

        async with TestClient(TestServer(first_app)) as cli:
            with patch.object(
                first_adapter, "_create_agent", return_value=mock_agent
            ):
                first = await cli.post(
                    "/v1/runs", json=request_body, headers=headers
                )
                first_run_id = (await first.json())["run_id"]
                for _ in range(80):
                    status = await (
                        await cli.get(f"/v1/runs/{first_run_id}")
                    ).json()
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.025)
                assert status["status"] == "completed"
        first_adapter._response_store.close()

        restarted_adapter = _make_adapter()
        restarted_adapter._response_store.close()
        restarted_adapter._response_store = ResponseStore(db_path=str(store_path))
        restarted_app = _create_runs_app(restarted_adapter)
        try:
            async with TestClient(TestServer(restarted_app)) as cli:
                with patch.object(restarted_adapter, "_create_agent") as create_agent:
                    replay = await cli.post(
                        "/v1/runs", json=request_body, headers=headers
                    )
                    replay_data = await replay.json()

            assert replay.status == 202
            assert replay_data["run_id"] == first_run_id
            assert replay_data["idempotent_replay"] is True
            create_agent.assert_not_called()
        finally:
            restarted_adapter._response_store.close()

    @pytest.mark.asyncio
    async def test_existing_session_uses_canonical_history_over_client_projection(
        self, adapter
    ):
        """A reduced browser transcript must not erase resumable tool context."""
        app = _create_runs_app(adapter)
        captured = {}
        canonical_history = [
            {"role": "user", "content": "build the loop"},
            {
                "role": "assistant",
                "content": "Building the scratch subject now.",
                "finish_reason": "tool_calls",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "todo", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "todo",
                "content": '{"status":"in_progress"}',
            },
            {
                "role": "assistant",
                "content": "Iteration-limit handoff: resume the loop implementation.",
            },
        ]
        lossy_client_history = [
            {"role": "user", "content": "build the loop"},
            {
                "role": "assistant",
                "content": "Run ended before an authoritative final answer.",
            },
        ]
        mock_agent = MagicMock()

        def _capture_run(user_message=None, conversation_history=None, task_id=None):
            captured["history"] = conversation_history
            return {"final_response": "resumed"}

        mock_agent.run_conversation.side_effect = _capture_run
        mock_agent.session_prompt_tokens = 0
        mock_agent.session_completion_tokens = 0
        mock_agent.session_total_tokens = 0

        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                adapter,
                "_conversation_history_for_session",
                return_value=canonical_history,
            ) as load_history, patch.object(
                adapter, "_create_agent", return_value=mock_agent
            ):
                response = await cli.post(
                    "/v1/runs",
                    json={
                        "input": "continue",
                        "session_id": "iteration-limited-session",
                        "conversation_history": lossy_client_history,
                    },
                )
                assert response.status == 202
                run_id = (await response.json())["run_id"]
                for _ in range(80):
                    status = await (
                        await cli.get(f"/v1/runs/{run_id}")
                    ).json()
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.025)

        load_history.assert_awaited_once_with("iteration-limited-session")
        assert captured["history"] == canonical_history

    @pytest.mark.asyncio
    async def test_new_session_keeps_client_bootstrap_history(self, adapter):
        app = _create_runs_app(adapter)
        captured = {}
        bootstrap_history = [
            {"role": "user", "content": "imported prompt"},
            {"role": "assistant", "content": "imported answer"},
        ]
        mock_agent = MagicMock()

        def _capture_run(user_message=None, conversation_history=None, task_id=None):
            captured["history"] = conversation_history
            return {"final_response": "continued"}

        mock_agent.run_conversation.side_effect = _capture_run
        mock_agent.session_prompt_tokens = 0
        mock_agent.session_completion_tokens = 0
        mock_agent.session_total_tokens = 0

        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                adapter,
                "_conversation_history_for_session",
                return_value=[],
            ), patch.object(adapter, "_create_agent", return_value=mock_agent):
                response = await cli.post(
                    "/v1/runs",
                    json={
                        "input": "continue",
                        "session_id": "not-yet-persisted",
                        "conversation_history": bootstrap_history,
                    },
                )
                run_id = (await response.json())["run_id"]
                for _ in range(80):
                    status = await (
                        await cli.get(f"/v1/runs/{run_id}")
                    ).json()
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.025)

        assert captured["history"] == bootstrap_history

    @pytest.mark.asyncio
    async def test_iteration_limit_failure_preserves_generated_summary(self, adapter):
        app = _create_runs_app(adapter)
        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = {
            "final_response": "Resume from task two; task one is complete.",
            "completed": False,
            "failed": False,
            "partial": False,
            "turn_exit_reason": "max_iterations_reached(50/50)",
        }
        mock_agent.session_prompt_tokens = 10
        mock_agent.session_completion_tokens = 5
        mock_agent.session_total_tokens = 15

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", return_value=mock_agent):
                response = await cli.post("/v1/runs", json={"input": "build it"})
                run_id = (await response.json())["run_id"]
                for _ in range(80):
                    status = await (
                        await cli.get(f"/v1/runs/{run_id}")
                    ).json()
                    if status["status"] == "failed":
                        break
                    await asyncio.sleep(0.025)
                events = await (
                    await cli.get(f"/v1/runs/{run_id}/events")
                ).text()

        assert status["output"] == "Resume from task two; task one is complete."
        assert status["output_kind"] == "summary"
        assert '"event": "run.failed"' in events
        assert '"output_kind": "summary"' in events
        assert '"output": "Resume from task two; task one is complete."' in events

    @pytest.mark.asyncio
    async def test_empty_final_response_is_failed_not_completed(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {
                    "final_response": "",
                    "completed": True,
                    "partial": False,
                }
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await resp.json())["run_id"]
                for _ in range(80):
                    status = await (await cli.get(f"/v1/runs/{run_id}")).json()
                    if status["status"] in {"completed", "failed"}:
                        break
                    await asyncio.sleep(0.025)

                assert status["status"] == "failed"
                assert "without final assistant text" in status["error"]

                events = await (await cli.get(f"/v1/runs/{run_id}/events")).text()
                assert '"event": "run.failed"' in events
                assert '"event": "run.completed"' not in events

    @pytest.mark.asyncio
    @pytest.mark.parametrize("raises", [False, True])
    async def test_run_keeps_codex_resident_and_retires_only_after_crash(
        self, adapter, raises
    ):
        app = _create_runs_app(adapter)
        codex_session = MagicMock()
        mock_agent = MagicMock()
        mock_agent.session_id = "codex-writer-release"
        mock_agent._codex_session = codex_session
        mock_agent.session_prompt_tokens = 0
        mock_agent.session_completion_tokens = 0
        mock_agent.session_total_tokens = 0
        if raises:
            mock_agent.run_conversation.side_effect = RuntimeError("turn failed")
            terminal_status = "failed"
        else:
            mock_agent.run_conversation.return_value = {"final_response": "done"}
            terminal_status = "completed"

        def _release(agent):
            native = agent._codex_session
            agent._codex_session = None
            native.close()

        runner = MagicMock()
        runner._agent_cache = OrderedDict()
        runner._agent_cache_lock = threading.Lock()
        runner._enforce_agent_cache_cap.side_effect = lambda: None
        runner._init_cached_agent_for_turn.side_effect = lambda *_args: None
        runner._release_evicted_agent_soft.side_effect = _release
        adapter.gateway_runner = runner

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", return_value=mock_agent):
                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello", "session_id": "codex-writer-release"},
                )
                assert resp.status == 202
                run_id = (await resp.json())["run_id"]

                for _ in range(80):
                    status = await (await cli.get(f"/v1/runs/{run_id}")).json()
                    if status["status"] in {"completed", "failed"}:
                        break
                    await asyncio.sleep(0.025)

                if not raises:
                    second = await cli.post(
                        "/v1/runs",
                        json={
                            "input": "continue",
                            "session_id": "codex-writer-release",
                        },
                    )
                    second_run_id = (await second.json())["run_id"]
                    for _ in range(80):
                        second_status = await (
                            await cli.get(f"/v1/runs/{second_run_id}")
                        ).json()
                        if second_status["status"] in {"completed", "failed"}:
                            break
                        await asyncio.sleep(0.025)

        assert status["status"] == terminal_status
        if raises:
            codex_session.close.assert_called_once_with()
            assert mock_agent._codex_session is None
        else:
            assert second_status["status"] == "completed"
            assert mock_agent.run_conversation.call_count == 2
            codex_session.close.assert_not_called()
            assert mock_agent._codex_session is codex_session

    @pytest.mark.asyncio
    async def test_start_preserves_latest_multimodal_user_message(self, adapter):
        app = _create_runs_app(adapter)
        captured = {}

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()

                def _capture_run(
                    user_message=None,
                    conversation_history=None,
                    task_id=None,
                ):
                    captured["user_message"] = user_message
                    return {"final_response": "done"}

                mock_agent.run_conversation.side_effect = _capture_run
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent
                image_content = [
                    {"type": "text", "text": "Describe this"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                    },
                ]

                resp = await cli.post(
                    "/v1/runs",
                    json={
                        "input": [{"role": "user", "content": image_content}],
                        "session_id": "multimodal-run",
                    },
                )
                assert resp.status == 202
                run_id = (await resp.json())["run_id"]
                for _ in range(40):
                    status = await (await cli.get(f"/v1/runs/{run_id}")).json()
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.05)

        assert captured["user_message"] == image_content

    @pytest.mark.asyncio
    async def test_start_binds_chat_id_for_delegation_wake_target(self, adapter):
        """/v1/runs must bind the raw session id as the api_server chat_id
        (like every other agent-entry route does via _run_agent): the async
        delegation dispatch reads HERMES_SESSION_CHAT_ID to pick its wake
        self-post target, and an empty binding forces background delegations
        on this route back to synchronous execution."""
        app = _create_runs_app(adapter)
        captured = {}

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()

                def _capture_run(user_message=None, conversation_history=None, task_id=None):
                    from tools.async_delegation import _current_origin_session_id

                    captured["origin_session_id"] = _current_origin_session_id()
                    return {"final_response": "done"}

                mock_agent.run_conversation.side_effect = _capture_run
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello", "session_id": "runs-raw-sid"},
                )
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                for _ in range(40):
                    status_resp = await cli.get(f"/v1/runs/{run_id}")
                    status = await status_resp.json()
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.05)

        assert captured.get("origin_session_id") == "runs-raw-sid", (
            "runs route must bind chat_id so delegation dispatch sees a wake target"
        )


    @pytest.mark.asyncio
    async def test_start_rejects_conflicting_route_and_request_provider(self):
        adapter = APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "model_routes": {
                        "alias": {
                            "model": "route/model",
                            "provider": "openrouter",
                        }
                    }
                },
            )
        )
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                resp = await cli.post(
                    "/v1/runs",
                    json={
                        "input": "hello",
                        "model": "alias",
                        "provider": "minimax",
                    },
                )
                data = await resp.json()

        assert resp.status == 400
        assert "provider" in data["error"]["message"].lower()
        assert adapter._run_streams == {}
        assert adapter._run_statuses == {}
        mock_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_passes_request_model_provider_options_to_create_agent(self, adapter):
        app = _create_runs_app(adapter)
        model_options = {"reasoning_effort": "medium", "service_tier": "priority"}
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post(
                    "/v1/runs",
                    json={
                        "input": "hello",
                        "model": "MiniMax-M3",
                        "provider": "minimax",
                        "model_options": model_options,
                    },
                )
                assert resp.status == 202
                for _ in range(20):
                    if mock_create.call_args is not None:
                        break
                    await asyncio.sleep(0.05)

        kwargs = mock_create.call_args.kwargs
        assert kwargs["requested_model"] == "MiniMax-M3"
        assert kwargs["requested_provider"] == "minimax"
        assert kwargs["model_options"] == model_options
        assert kwargs["single_model"] is False

    @pytest.mark.asyncio
    async def test_single_model_start_locks_explicit_provider_and_model(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create, patch.object(
                adapter, "_resolve_route"
            ) as route_resolver:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post(
                    "/v1/runs",
                    json={
                        "input": "hello",
                        "model": "new/openrouter-model",
                        "provider": "openrouter",
                        "execution_mode": "single_model",
                    },
                )
                assert resp.status == 202
                run_id = (await resp.json())["run_id"]
                for _ in range(20):
                    if mock_create.call_args is not None:
                        break
                    await asyncio.sleep(0.05)
                status = await (await cli.get(f"/v1/runs/{run_id}")).json()

        kwargs = mock_create.call_args.kwargs
        assert kwargs["requested_model"] == "new/openrouter-model"
        assert kwargs["requested_provider"] == "openrouter"
        assert kwargs["single_model"] is True
        assert status["execution_mode"] == "single_model"
        route_resolver.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "payload",
        [
            {"execution_mode": "single_model", "provider": "openrouter"},
            {"execution_mode": "single_model", "model": "new/model"},
            {
                "execution_mode": "automatic",
                "provider": "openrouter",
                "model": "new/model",
            },
        ],
    )
    async def test_single_model_start_fails_closed_on_incomplete_contract(
        self, adapter, payload
    ):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello", **payload},
                )

        assert resp.status == 400
        mock_create.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("runtime", "expected_provider", "expected_model", "single_model"),
        [
            ({}, None, None, False),
            (
                {
                    "execution_mode": "single_model",
                    "provider": "anthropic",
                    "model": "claude-opus-test",
                },
                "anthropic",
                "claude-opus-test",
                True,
            ),
            (
                {
                    "execution_mode": "single_model",
                    "provider": "openai-codex",
                    "model": "gpt-codex-test",
                },
                "openai-codex",
                "gpt-codex-test",
                True,
            ),
        ],
    )
    async def test_goal_loop_runs_for_orchestrated_and_single_model_runtimes(
        self,
        adapter,
        runtime,
        expected_provider,
        expected_model,
        single_model,
    ):
        app = _create_runs_app(adapter)
        mock_agent = MagicMock()
        mock_agent.run_conversation.side_effect = [
            {"final_response": "first pass"},
            {"final_response": "authoritative finish"},
        ]
        mock_agent.session_prompt_tokens = 20
        mock_agent.session_completion_tokens = 10
        mock_agent.session_total_tokens = 30
        decisions = [
            {
                "should_continue": True,
                "continuation_prompt": "continue toward the goal",
                "message": "Continuing toward goal",
            },
            {
                "should_continue": False,
                "continuation_prompt": None,
                "message": "Goal achieved",
            },
        ]

        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                adapter, "_create_agent", return_value=mock_agent
            ) as create_agent, patch(
                "gateway.platforms.api_server._prepare_api_goal_command",
                return_value={
                    "notice": "Goal set",
                    "run_prompt": "ship it",
                    "continuation": False,
                },
            ), patch(
                "gateway.platforms.api_server._evaluate_api_goal_turn",
                side_effect=decisions,
            ) as evaluate:
                response = await cli.post(
                    "/v1/runs",
                    json={
                        "input": "/goal ship it",
                        "session_id": "goal-runtime-session",
                        **runtime,
                    },
                )
                assert response.status == 202
                run_id = (await response.json())["run_id"]
                events_response = await cli.get(f"/v1/runs/{run_id}/events")
                events = await events_response.text()
                status = await (await cli.get(f"/v1/runs/{run_id}")).json()

        assert status["status"] == "completed"
        assert status["output"] == "authoritative finish"
        assert '"text": "first pass"' in events
        assert '"text": "Continuing toward goal"' in events
        assert '"output": "authoritative finish"' in events
        assert mock_agent.run_conversation.call_count == 2
        first, second = mock_agent.run_conversation.call_args_list
        assert first.kwargs["user_message"] == "ship it"
        assert first.kwargs["persist_user_message"] == "/goal ship it"
        assert second.kwargs["user_message"] == "continue toward the goal"
        assert second.kwargs["persist_user_display_kind"] == "auto_continue"
        assert second.kwargs["conversation_history"] is None
        assert evaluate.call_count == 2
        created = create_agent.call_args.kwargs
        assert created["requested_provider"] == expected_provider
        assert created["requested_model"] == expected_model
        assert created["single_model"] is single_model

    @pytest.mark.asyncio
    async def test_goal_status_completes_without_running_a_model_turn(self, adapter):
        app = _create_runs_app(adapter)
        mock_agent = MagicMock()
        mock_agent.session_prompt_tokens = 0
        mock_agent.session_completion_tokens = 0
        mock_agent.session_total_tokens = 0

        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                adapter, "_create_agent", return_value=mock_agent
            ) as create_agent, patch(
                "gateway.platforms.api_server._prepare_api_goal_command",
                return_value={"response": "Goal active: ship it"},
            ):
                response = await cli.post(
                    "/v1/runs",
                    json={
                        "input": "/goal status",
                        "session_id": "goal-control-session",
                    },
                )
                run_id = (await response.json())["run_id"]
                events_response = await cli.get(f"/v1/runs/{run_id}/events")
                events = await events_response.text()

        mock_agent.run_conversation.assert_not_called()
        create_agent.assert_not_called()
        assert '"output": "Goal active: ship it"' in events


# ---------------------------------------------------------------------------
# GET /v1/runs/{run_id} — poll run status
# ---------------------------------------------------------------------------


class TestRunStatus:

    @pytest.mark.asyncio
    async def test_status_reflects_explicit_session_id(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello", "session_id": "space-session"},
                )
                data = await resp.json()
                run_id = data["run_id"]

                for _ in range(20):
                    status_resp = await cli.get(f"/v1/runs/{run_id}")
                    status = await status_resp.json()
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.05)

                mock_agent.run_conversation.assert_called_once()
                assert mock_agent.run_conversation.call_args.kwargs["task_id"] == "space-session"
                assert status["session_id"] == "space-session"


# ---------------------------------------------------------------------------
# GET /v1/runs/{run_id}/events — SSE event stream
# ---------------------------------------------------------------------------


class TestRunEvents:
    @pytest.mark.asyncio
    async def test_events_stream_returns_completed(self, adapter):
        """Events stream should receive run.completed when agent finishes."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "Hello!"}
                mock_agent.session_prompt_tokens = 10
                mock_agent.session_completion_tokens = 5
                mock_agent.session_total_tokens = 15
                mock_create.return_value = mock_agent

                # Start run
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                # Subscribe to events
                events_resp = await cli.get(f"/v1/runs/{run_id}/events")
                assert events_resp.status == 200
                body = await events_resp.text()

                # Should contain run.completed
                assert "run.completed" in body
                assert "Hello!" in body

    @pytest.mark.asyncio
    async def test_completed_run_inlines_safe_media_image(self, adapter, tmp_path):
        image = tmp_path / "preview.png"
        image.write_bytes(
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\rIHDR"
            b"\x00\x00\x00\x01\x00\x00\x00\x01"
        )
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {
                    "final_response": f"Rendered preview:\n\nMEDIA:{image}"
                }
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                response = await cli.post("/v1/runs", json={"input": "render"})
                run_id = (await response.json())["run_id"]
                events_response = await cli.get(f"/v1/runs/{run_id}/events")
                body = await events_response.text()
                status_response = await cli.get(f"/v1/runs/{run_id}")
                status = await status_response.json()

        assert "data:image/png;base64," in body
        assert "MEDIA:" not in body
        assert "data:image/png;base64," in status["output"]
        assert "MEDIA:" not in status["output"]

    @pytest.mark.asyncio
    async def test_completed_run_promotes_requested_codex_image_view(
        self, adapter, tmp_path
    ):
        image = tmp_path / "catalog.png"
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()

                def run_conversation(**_kwargs):
                    image.write_bytes(
                        b"\x89PNG\r\n\x1a\n"
                        b"\x00\x00\x00\rIHDR"
                        b"\x00\x00\x00\x01\x00\x00\x00\x01"
                    )
                    image_view = {
                        "type": "imageView",
                        "id": "view_catalog",
                        "path": str(image),
                    }
                    return {
                        "final_response": "This is the catalog-only view.",
                        "messages": [
                            {
                                "role": "assistant",
                                "content": (
                                    "[codex imageView] "
                                    + json.dumps(image_view)
                                ),
                            }
                        ],
                    }

                mock_agent.run_conversation.side_effect = run_conversation
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                response = await cli.post(
                    "/v1/runs",
                    json={"input": "Can you show me the catalog screenshot?"},
                )
                run_id = (await response.json())["run_id"]
                events_response = await cli.get(f"/v1/runs/{run_id}/events")
                body = await events_response.text()
                status = await (await cli.get(f"/v1/runs/{run_id}")).json()

        assert "data:image/png;base64," in body
        assert str(image) not in body
        assert "data:image/png;base64," in status["output"]
        assert str(image) not in status["output"]

    @pytest.mark.asyncio
    async def test_interim_assistant_text_has_its_own_run_event(self, adapter):
        """Tool-call commentary must not masquerade as final answer text."""
        app = _create_runs_app(adapter)

        def create_agent(**kwargs):
            mock_agent = MagicMock()

            def run_conversation(**_run_kwargs):
                kwargs["interim_assistant_callback"](
                    "I will inspect the files now.",
                    already_streamed=True,
                )
                return {"final_response": "Inspection complete."}

            mock_agent.run_conversation.side_effect = run_conversation
            mock_agent.session_prompt_tokens = 10
            mock_agent.session_completion_tokens = 5
            mock_agent.session_total_tokens = 15
            return mock_agent

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", side_effect=create_agent):
                resp = await cli.post("/v1/runs", json={"input": "inspect"})
                run_id = (await resp.json())["run_id"]
                events_resp = await cli.get(f"/v1/runs/{run_id}/events")
                body = await events_resp.text()

        assert '"event": "message.interim"' in body
        assert f'"message_id": "{run_id}:commentary:1"' in body
        assert '"text": "I will inspect the files now."' in body
        assert '"already_streamed": true' in body
        assert '"event": "run.completed"' in body
        assert "Inspection complete." in body

    @pytest.mark.asyncio
    async def test_lifecycle_status_is_persisted_in_run_chronology(self, adapter):
        """Provider retry countdowns must survive detach and replay."""
        app = _create_runs_app(adapter)

        def create_agent(**kwargs):
            mock_agent = MagicMock()

            def run_conversation(**_run_kwargs):
                kwargs["status_callback"](
                    "lifecycle",
                    "Provider busy (HTTP 429). Retry 1/7 in 5.0s; "
                    "the same run and model are preserved.",
                )
                return {"final_response": "Recovered."}

            mock_agent.run_conversation.side_effect = run_conversation
            mock_agent.session_prompt_tokens = 0
            mock_agent.session_completion_tokens = 0
            mock_agent.session_total_tokens = 0
            return mock_agent

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", side_effect=create_agent):
                resp = await cli.post("/v1/runs", json={"input": "continue"})
                run_id = (await resp.json())["run_id"]
                events_resp = await cli.get(f"/v1/runs/{run_id}/events")
                body = await events_resp.text()

        assert '"event": "message.interim"' in body
        assert f'"message_id": "{run_id}:commentary:1"' in body
        assert "Provider busy (HTTP 429). Retry 1/7" in body
        assert '"already_streamed": false' in body
        assert '"event": "run.completed"' in body

    @pytest.mark.asyncio
    async def test_partial_result_is_failed_not_completed(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {
                    "final_response": "This is only progress.",
                    "completed": False,
                    "partial": True,
                    "error": "turn timed out after 600s",
                    "error_code": "claude_first_event_timeout",
                }
                mock_agent.session_prompt_tokens = 10
                mock_agent.session_completion_tokens = 5
                mock_agent.session_total_tokens = 15
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "inspect"})
                run_id = (await resp.json())["run_id"]
                events_resp = await cli.get(f"/v1/runs/{run_id}/events")
                body = await events_resp.text()
                status = await (await cli.get(f"/v1/runs/{run_id}")).json()

        assert '"event": "run.failed"' in body
        assert "turn timed out after 600s" in body
        assert '"error_code": "claude_first_event_timeout"' in body
        assert status["error_code"] == "claude_first_event_timeout"
        assert '"event": "run.completed"' not in body

    @pytest.mark.asyncio
    async def test_events_replay_after_first_stream_closes(self, adapter):
        """A completed run remains attachable until its transport TTL expires."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {
                    "final_response": "Replay me"
                }
                mock_agent.session_prompt_tokens = 1
                mock_agent.session_completion_tokens = 2
                mock_agent.session_total_tokens = 3
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await resp.json())["run_id"]

                first = await cli.get(f"/v1/runs/{run_id}/events")
                first_body = await first.text()
                second = await cli.get(f"/v1/runs/{run_id}/events")
                second_body = await second.text()

        assert first.status == 200
        assert second.status == 200
        assert "run.completed" in first_body
        assert "Replay me" in first_body
        assert second_body == first_body

    @pytest.mark.asyncio
    async def test_disconnect_keeps_history_for_running_run(self, adapter):
        """Dropping one SSE reader must not remove the run transport."""
        run_id = "run_reconnecttest"
        stream = _ReplayableRunEventStream(adapter._RUN_EVENT_HISTORY_LIMIT)
        stream.put_nowait({
            "event": "message.delta",
            "run_id": run_id,
            "timestamp": time.time(),
            "delta": "before-",
        })
        adapter._run_streams[run_id] = stream
        adapter._run_streams_created[run_id] = time.time()
        adapter._set_run_status(run_id, "running")

        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            first = await cli.get(f"/v1/runs/{run_id}/events")
            line = await first.content.readline()
            assert b"before-" in line
            first.close()
            await asyncio.sleep(0)

            assert run_id in adapter._run_streams
            stream.put_nowait({
                "event": "run.completed",
                "run_id": run_id,
                "timestamp": time.time(),
                "output": "before-after",
            })
            stream.put_nowait(None)

            second = await cli.get(f"/v1/runs/{run_id}/events")
            body = await second.text()

        assert second.status == 200
        assert "before-" in body
        assert "before-after" in body

    def test_event_history_is_bounded(self, adapter):
        stream = _ReplayableRunEventStream(history_limit=2)
        for index in range(3):
            stream.put_nowait({"event": "message.delta", "delta": str(index)})

        events, cursor, closed = stream.read_from(0)

        assert [event["delta"] for event in events] == ["1", "2"]
        assert cursor == 3
        assert closed is False


    @pytest.mark.asyncio
    async def test_approval_resolve_all_is_scoped_to_target_run(self, auth_adapter):
        """Same client session_id must not let one run approve another run's queue."""
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_create_agent") as mock_create:
                victim_agent, victim_ready, victim_interrupted = _make_slow_agent()
                attacker_agent, attacker_ready, attacker_interrupted = _make_slow_agent()
                mock_create.side_effect = [victim_agent, attacker_agent]

                victim_resp = await cli.post(
                    "/v1/runs",
                    json={"input": "victim", "session_id": "shared-project"},
                    headers={"Authorization": "Bearer sk-secret"},
                )
                attacker_resp = await cli.post(
                    "/v1/runs",
                    json={"input": "attacker", "session_id": "shared-project"},
                    headers={"Authorization": "Bearer sk-secret"},
                )
                assert victim_resp.status == 202
                assert attacker_resp.status == 202
                victim_run = (await victim_resp.json())["run_id"]
                attacker_run = (await attacker_resp.json())["run_id"]

                victim_ready.wait(timeout=3.0)
                attacker_ready.wait(timeout=3.0)
                assert auth_adapter._run_approval_sessions[victim_run] == victim_run
                assert auth_adapter._run_approval_sessions[attacker_run] == attacker_run
                assert auth_adapter._run_approval_sessions[victim_run] != auth_adapter._run_approval_sessions[attacker_run]

                victim_entry = approval_mod._ApprovalEntry({
                    "command": "bash -c victim-danger",
                    "description": "victim approval",
                    "pattern_keys": ["shell-c"],
                })
                attacker_entry = approval_mod._ApprovalEntry({
                    "command": "bash -c attacker-danger",
                    "description": "attacker approval",
                    "pattern_keys": ["shell-c"],
                })
                with approval_mod._lock:
                    approval_mod._gateway_queues[victim_run] = [victim_entry]
                    approval_mod._gateway_queues[attacker_run] = [attacker_entry]

                approval_resp = await cli.post(
                    f"/v1/runs/{attacker_run}/approval",
                    json={"choice": "always", "resolve_all": True},
                    headers={"Authorization": "Bearer sk-secret"},
                )
                approval_data = await approval_resp.json()

                assert approval_resp.status == 200
                assert approval_data["resolved"] == 1
                assert attacker_entry.result == "always"
                assert attacker_entry.event.is_set()
                assert victim_entry.result is None
                assert not victim_entry.event.is_set()
                with approval_mod._lock:
                    assert approval_mod._gateway_queues[victim_run] == [victim_entry]
                    assert victim_run in approval_mod._gateway_queues
                    assert attacker_run not in approval_mod._gateway_queues

                # Clean up the synthetic pending victim approval and unblock the
                # slow test agents so their background run tasks can finish.
                with approval_mod._lock:
                    approval_mod._gateway_queues.pop(victim_run, None)
                victim_interrupted.set()
                attacker_interrupted.set()

    @pytest.mark.asyncio
    async def test_approval_response_targets_exact_request_id(self, auth_adapter):
        app = _create_runs_app(auth_adapter)
        run_id = "run_targetedapproval"
        auth_adapter._set_run_status(run_id, "waiting_for_approval")
        auth_adapter._run_approval_sessions[run_id] = run_id
        auth_adapter._run_streams[run_id] = _ReplayableRunEventStream(1000)

        first = approval_mod._ApprovalEntry({
            "request_id": "approval-first",
            "command": "first",
        })
        second = approval_mod._ApprovalEntry({
            "request_id": "approval-second",
            "command": "second",
        })
        with approval_mod._lock:
            approval_mod._gateway_queues[run_id] = [first, second]

        try:
            async with TestClient(TestServer(app)) as cli:
                response = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={"choice": "once", "request_id": "approval-second"},
                    headers={"Authorization": "Bearer sk-secret"},
                )
                payload = await response.json()

            assert response.status == 200
            assert payload["request_id"] == "approval-second"
            assert second.result == "once"
            assert second.event.is_set()
            assert first.result is None
            assert not first.event.is_set()
            with approval_mod._lock:
                assert approval_mod._gateway_queues[run_id] == [first]

            events, _, _ = auth_adapter._run_streams[run_id].read_from(0)
            assert events[-1]["event"] == "approval.responded"
            assert events[-1]["request_id"] == "approval-second"
        finally:
            with approval_mod._lock:
                approval_mod._gateway_queues.pop(run_id, None)
            auth_adapter._run_approval_sessions.pop(run_id, None)
            auth_adapter._run_streams.pop(run_id, None)


# ---------------------------------------------------------------------------
# POST /v1/runs/{run_id}/steer — steer a running agent
# ---------------------------------------------------------------------------


class TestSteerRun:
    @pytest.mark.asyncio
    async def test_steer_running_agent(self, adapter):
        app = _create_runs_app(adapter)
        agent = MagicMock()
        agent.steer.return_value = True
        queue = asyncio.Queue()
        adapter._active_run_agents["run_123"] = agent
        adapter._run_streams["run_123"] = queue
        adapter._set_run_status("run_123", "running")

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_123/steer", json={"input": "tighten the ending"})
            payload = await resp.json()

        assert resp.status == 200
        assert payload == {
            "object": "hermes.run.steer",
            "run_id": "run_123",
            "accepted": True,
        }
        agent.steer.assert_called_once_with("tighten the ending")
        assert adapter._run_statuses["run_123"]["last_event"] == "run.steered"
        event = queue.get_nowait()
        assert event["event"] == "run.steered"
        assert event["run_id"] == "run_123"
        assert event["accepted"] is True

    @pytest.mark.asyncio
    async def test_steer_nonexistent_run_returns_404(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_missing/steer", json={"input": "hello"})
            payload = await resp.json()

        assert resp.status == 404
        assert payload["error"]["code"] == "run_not_found"

    @pytest.mark.asyncio
    async def test_steer_inactive_run_returns_409(self, adapter):
        app = _create_runs_app(adapter)
        adapter._set_run_status("run_done", "completed")

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_done/steer", json={"input": "hello"})
            payload = await resp.json()

        assert resp.status == 409
        assert payload["error"]["code"] == "run_not_accepting_steer"

    @pytest.mark.asyncio
    async def test_steer_missing_input_returns_400(self, adapter):
        app = _create_runs_app(adapter)
        agent = MagicMock()
        agent.steer.return_value = True
        adapter._active_run_agents["run_123"] = agent
        adapter._set_run_status("run_123", "running")

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_123/steer", json={"input": ""})
            payload = await resp.json()

        assert resp.status == 400
        assert payload["error"]["code"] == "invalid_steer_input"
        agent.steer.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_then_steer_rejects_retained_agent_ref(self, adapter):
        """Steer must reject a stopping run even if the executor thread is still live."""
        app = _create_runs_app(adapter)
        run_can_finish = threading.Event()
        run_started = threading.Event()

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_agent.steer = MagicMock(return_value=True)

                def _interrupt(_message=None):
                    return None

                def _run_conversation(*_args, **_kwargs):
                    run_started.set()
                    run_can_finish.wait(timeout=5)
                    return {"final_response": "late result"}

                mock_agent.interrupt = MagicMock(side_effect=_interrupt)
                mock_agent.run_conversation.side_effect = _run_conversation
                mock_create.return_value = mock_agent

                start_resp = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await start_resp.json())["run_id"]
                assert run_started.wait(timeout=3.0)

                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                assert run_id in adapter._active_run_agents

                steer_resp = await cli.post(
                    f"/v1/runs/{run_id}/steer",
                    json={"input": "tighten the ending"},
                )
                steer_data = await steer_resp.json()

                assert steer_resp.status == 409
                assert steer_data["error"]["code"] == "run_not_accepting_steer"
                mock_agent.steer.assert_not_called()

                run_can_finish.set()
                for _ in range(40):
                    if run_id not in adapter._active_run_tasks:
                        break
                    await asyncio.sleep(0.05)

    @pytest.mark.asyncio
    async def test_pending_steer_preserved_on_run_completed(self, adapter):
        """A steer drained by the turn finalizer (accepted after the final
        response) must surface as pending_steer on the terminal run status
        instead of being silently dropped."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_agent.run_conversation.return_value = {
                    "final_response": "done",
                    "pending_steer": "tighten the ending",
                }
                mock_create.return_value = mock_agent

                start_resp = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await start_resp.json())["run_id"]

                for _ in range(40):
                    status = adapter._run_statuses.get(run_id, {})
                    if status.get("status") == "completed":
                        break
                    await asyncio.sleep(0.05)

        assert adapter._run_statuses[run_id]["status"] == "completed"
        assert adapter._run_statuses[run_id]["pending_steer"] == "tighten the ending"

    @pytest.mark.asyncio
    async def test_steer_requires_auth(self, auth_adapter):
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_any/steer", json={"input": "hello"})

        assert resp.status == 401


# ---------------------------------------------------------------------------
# Tool event identity and output on /v1/runs
# ---------------------------------------------------------------------------


class TestRunToolEventIdentity:

    @staticmethod
    def _agent_factory(script):
        """create_agent side_effect that runs `script(tool_progress_callback)`
        inside run_conversation and then returns a final response."""

        def create_agent(**kwargs):
            mock_agent = MagicMock()

            def run_conversation(**_run_kwargs):
                script(kwargs["tool_progress_callback"])
                return {"final_response": "done"}

            mock_agent.run_conversation.side_effect = run_conversation
            mock_agent.session_prompt_tokens = 0
            mock_agent.session_completion_tokens = 0
            mock_agent.session_total_tokens = 0
            return mock_agent

        return create_agent

    @pytest.mark.asyncio
    async def test_runtime_first_event_timeout_is_durable_and_structured(
        self, adapter
    ):
        app = _create_runs_app(adapter)

        def script(cb):
            cb(
                "runtime.first_event_timeout",
                "claude-code",
                "Claude did not acknowledge; resetting the runtime.",
                None,
                code="claude_first_event_timeout",
                attempt=1,
                retrying=True,
                timeout_seconds=30.0,
            )

        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                adapter,
                "_create_agent",
                side_effect=self._agent_factory(script),
            ):
                resp = await cli.post("/v1/runs", json={"input": "go"})
                run_id = (await resp.json())["run_id"]
                body = await (
                    await cli.get(f"/v1/runs/{run_id}/events")
                ).text()

        events = [
            json.loads(line[len("data: "):])
            for line in body.splitlines()
            if line.startswith("data: ")
        ]
        watchdog = next(
            event
            for event in events
            if event.get("event") == "runtime.first_event_timeout"
        )
        assert watchdog == {
            "event": "runtime.first_event_timeout",
            "run_id": run_id,
            "timestamp": watchdog["timestamp"],
            "runtime": "claude-code",
            "code": "claude_first_event_timeout",
            "attempt": 1,
            "retrying": True,
            "timeout_seconds": 30.0,
            "message": "Claude did not acknowledge; resetting the runtime.",
        }

    @pytest.mark.asyncio
    async def test_codex_first_event_timeout_keeps_thread_and_turn_ids(
        self, adapter
    ):
        app = _create_runs_app(adapter)

        def script(cb):
            cb(
                "runtime.first_event_timeout",
                "codex-app-server",
                "Codex accepted the turn but emitted no activity.",
                None,
                code="codex_first_event_timeout",
                attempt=1,
                retrying=False,
                timeout_seconds=60.0,
                thread_id="thread-123",
                turn_id="turn-456",
            )

        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                adapter,
                "_create_agent",
                side_effect=self._agent_factory(script),
            ):
                resp = await cli.post("/v1/runs", json={"input": "go"})
                run_id = (await resp.json())["run_id"]
                body = await (
                    await cli.get(f"/v1/runs/{run_id}/events")
                ).text()

        events = [
            json.loads(line[len("data: "):])
            for line in body.splitlines()
            if line.startswith("data: ")
        ]
        watchdog = next(
            event
            for event in events
            if event.get("event") == "runtime.first_event_timeout"
        )
        assert watchdog["runtime"] == "codex-app-server"
        assert watchdog["code"] == "codex_first_event_timeout"
        assert watchdog["retrying"] is False
        assert watchdog["thread_id"] == "thread-123"
        assert watchdog["turn_id"] == "turn-456"

    @pytest.mark.asyncio
    async def test_tool_events_carry_stable_ids_and_output(self, adapter):
        """Without caller ids, emission mints {run_id}:tool:{n} and pairs
        completions FIFO by tool name, so replays carry one fixed id."""
        app = _create_runs_app(adapter)

        def script(cb):
            cb("tool.started", "terminal", "ls -la", {"command": "ls -la"})
            cb("tool.started", "terminal", "pwd", {"command": "pwd"})
            cb("tool.completed", "terminal", None, None,
               duration=0.25, is_error=False, result="file-a\nfile-b")
            cb("tool.completed", "terminal", None, None,
               duration=0.05, is_error=True, result="[exit 1]\nboom")

        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                adapter, "_create_agent",
                side_effect=self._agent_factory(script),
            ):
                resp = await cli.post("/v1/runs", json={"input": "go"})
                run_id = (await resp.json())["run_id"]
                body = await (await cli.get(f"/v1/runs/{run_id}/events")).text()

        import json as _json
        events = [
            _json.loads(line[len("data: "):])
            for line in body.splitlines()
            if line.startswith("data: ")
        ]
        tool_events = [e for e in events if e.get("event", "").startswith("tool.")]
        assert [e["event"] for e in tool_events] == [
            "tool.started", "tool.started", "tool.completed", "tool.completed",
        ]
        assert tool_events[0]["tool_call_id"] == f"{run_id}:tool:1"
        assert tool_events[1]["tool_call_id"] == f"{run_id}:tool:2"
        # FIFO pairing: first completion belongs to the first start.
        assert tool_events[2]["tool_call_id"] == f"{run_id}:tool:1"
        assert tool_events[3]["tool_call_id"] == f"{run_id}:tool:2"
        assert tool_events[2]["output"] == "file-a\nfile-b"
        assert tool_events[2]["duration"] == 0.25
        assert tool_events[2]["error"] is False
        assert tool_events[3]["output"] == "[exit 1]\nboom"
        assert tool_events[3]["error"] is True

    @pytest.mark.asyncio
    async def test_tool_events_use_caller_supplied_tool_call_id(self, adapter):
        """Executor/codex callers pass their canonical id; it must survive
        verbatim on both started and completed events."""
        app = _create_runs_app(adapter)

        def script(cb):
            cb("tool.started", "exec_command", "npm test", {"command": "npm test"},
               tool_call_id="call_abc123")
            cb("tool.completed", "exec_command", None, None,
               duration=1.5, is_error=False, result="ok",
               tool_call_id="call_abc123")

        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                adapter, "_create_agent",
                side_effect=self._agent_factory(script),
            ):
                resp = await cli.post("/v1/runs", json={"input": "go"})
                run_id = (await resp.json())["run_id"]
                body = await (await cli.get(f"/v1/runs/{run_id}/events")).text()

        assert body.count('"tool_call_id": "call_abc123"') == 2

    @pytest.mark.asyncio
    async def test_live_tool_output_is_ordered_redacted_capped_and_replayable(
        self, adapter
    ):
        """Output deltas are first-class replay events. Redaction must see
        across producer chunk boundaries, and the display cap is disclosed."""
        app = _create_runs_app(adapter)
        secret = "AKIAIOSFODNN7EXAMPLE"
        big = "x" * (adapter._RUN_TOOL_OUTPUT_EVENT_CHARS + 5000)

        def script(cb):
            cb(
                "tool.started",
                "exec_command",
                "run checks",
                {"command": "run checks"},
                tool_call_id="call_stream1",
            )
            cb(
                "tool.output.delta",
                "exec_command",
                chunk="key=AKIAIOS",
                channel="combined",
                tool_call_id="call_stream1",
            )
            cb(
                "tool.output.delta",
                "exec_command",
                chunk="FODNN7EXAMPLE\nfirst line\n",
                channel="combined",
                tool_call_id="call_stream1",
            )
            cb(
                "tool.output.delta",
                "exec_command",
                chunk=big,
                channel="combined",
                tool_call_id="call_stream1",
            )
            cb(
                "tool.completed",
                "exec_command",
                duration=1.5,
                is_error=False,
                result=f"key={secret}\nfirst line\n{big}",
                tool_call_id="call_stream1",
            )

        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                adapter, "_create_agent", side_effect=self._agent_factory(script)
            ):
                resp = await cli.post("/v1/runs", json={"input": "go"})
                run_id = (await resp.json())["run_id"]
                first = await (
                    await cli.get(f"/v1/runs/{run_id}/events")
                ).text()
                second = await (
                    await cli.get(f"/v1/runs/{run_id}/events")
                ).text()

        import json as _json

        events = [
            _json.loads(line[len("data: "):])
            for line in first.splitlines()
            if line.startswith("data: ")
        ]
        tool_events = [
            event for event in events if event.get("event", "").startswith("tool.")
        ]
        assert [event["event"] for event in tool_events] == [
            "tool.started",
            "tool.output.delta",
            "tool.output.delta",
            "tool.completed",
        ]
        assert [event["tool_call_id"] for event in tool_events] == [
            "call_stream1"
        ] * 4
        output_events = tool_events[1:3]
        assert [event["sequence"] for event in output_events] == [1, 2]
        assert [event["event_id"] for event in output_events] == [
            "call_stream1:output:1",
            "call_stream1:output:2",
        ]
        assert secret not in first
        assert "first line" in first
        assert "live tool output truncated at" in first
        assert second == first, "reattach must replay byte-identical output events"

    @pytest.mark.asyncio
    async def test_tool_completed_output_is_redacted_and_capped(self, adapter):
        """Tool results carry terminal output: secrets must be redacted and
        oversized text truncated with disclosure, never silently."""
        app = _create_runs_app(adapter)
        secret = "AKIAIOSFODNN7EXAMPLE"
        big = "x" * (adapter._RUN_TOOL_OUTPUT_EVENT_CHARS + 500)

        def script(cb):
            cb("tool.started", "terminal", "env", {"command": "env"})
            cb("tool.completed", "terminal", None, None,
               duration=0.1, is_error=False,
               result=f"key={secret}\n{big}")

        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                adapter, "_create_agent",
                side_effect=self._agent_factory(script),
            ):
                resp = await cli.post("/v1/runs", json={"input": "go"})
                run_id = (await resp.json())["run_id"]
                body = await (await cli.get(f"/v1/runs/{run_id}/events")).text()

        assert secret not in body
        assert "truncated" in body

    @pytest.mark.asyncio
    async def test_replayed_tool_events_keep_identical_ids(self, adapter):
        """Two attaches must serve byte-identical tool ids — the property
        that makes client-side replay dedupe possible at all."""
        app = _create_runs_app(adapter)

        def script(cb):
            cb("tool.started", "terminal", "ls", {"command": "ls"})
            cb("tool.completed", "terminal", None, None,
               duration=0.1, is_error=False, result="out")

        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                adapter, "_create_agent",
                side_effect=self._agent_factory(script),
            ):
                resp = await cli.post("/v1/runs", json={"input": "go"})
                run_id = (await resp.json())["run_id"]
                first = await (await cli.get(f"/v1/runs/{run_id}/events")).text()
                second = await (await cli.get(f"/v1/runs/{run_id}/events")).text()

        import re as _re
        ids_first = _re.findall(r'"tool_call_id": "([^"]+)"', first)
        ids_second = _re.findall(r'"tool_call_id": "([^"]+)"', second)
        assert ids_first == ids_second == [f"{run_id}:tool:1", f"{run_id}:tool:1"]


# ---------------------------------------------------------------------------
# Run lifecycle TTL sweeping
# ---------------------------------------------------------------------------


class TestRunLifecycleSweep:

    @pytest.mark.asyncio
    async def test_live_run_transport_survives_sweep(self, adapter):
        """A live run's replay ring is never swept — every event emitted
        after a transport sweep would be silently unrecoverable for clients
        that reattach later, even though the run is still executing. Memory
        stays bounded by the per-run event ring, not by dropping transports.
        Control state (approvals, stop, concurrency) is unaffected."""
        app = _create_runs_app(adapter)
        adapter._max_concurrent_runs = 1

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, _ = _make_slow_agent()
                mock_create.return_value = mock_agent

                start_resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert start_resp.status == 202
                run_id = (await start_resp.json())["run_id"]
                assert agent_ready.wait(timeout=3.0)

                task = adapter._active_run_tasks[run_id]
                assert isinstance(task, asyncio.Task)
                assert not task.done()

                pending = approval_mod._ApprovalEntry({
                    "command": "bash -c long-running",
                    "description": "approval after stream TTL",
                    "pattern_keys": ["shell-c"],
                })
                with approval_mod._lock:
                    approval_mod._gateway_queues[run_id] = [pending]

                adapter._run_streams_created[run_id] -= adapter._RUN_STREAM_TTL + 1
                # Exercise one real sweeper iteration without waiting 60 seconds.
                with patch(
                    "gateway.platforms.api_server.asyncio.sleep",
                    side_effect=[None, asyncio.CancelledError()],
                ):
                    with pytest.raises(asyncio.CancelledError):
                        await adapter._sweep_orphaned_runs()

                assert adapter._active_run_tasks[run_id] is task
                assert adapter._active_run_agents[run_id] is mock_agent
                assert run_id in adapter._run_streams
                assert run_id in adapter._run_streams_created
                assert adapter._run_approval_sessions[run_id] == run_id

                limited = adapter._concurrency_limited_response()
                assert limited is not None
                assert limited.status == 429

                approval_resp = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={"choice": "once"},
                )
                assert approval_resp.status == 200
                assert pending.event.is_set()
                assert pending.result == "once"

                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                mock_agent.interrupt.assert_called_once_with("Stop requested via API")


    @pytest.mark.asyncio
    async def test_terminal_run_transport_expires_on_post_terminal_grace(self, adapter):
        """After the run ends, replay stays available for _RUN_STREAM_TTL
        seconds measured from the terminal transition (not run creation),
        then is swept; the pollable status survives on its own longer TTL."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await resp.json())["run_id"]
                # Drain to completion so the task settles.
                body = await (await cli.get(f"/v1/runs/{run_id}/events")).text()
                assert "run.completed" in body

        terminal_at = adapter._run_statuses[run_id]["updated_at"]

        # Old transport age alone must not sweep a freshly-terminal run:
        # age the creation stamp past the TTL but keep terminal recent.
        adapter._run_streams_created[run_id] -= adapter._RUN_STREAM_TTL * 10
        adapter._sweep_orphaned_runs_once(terminal_at + adapter._RUN_STREAM_TTL - 5)
        assert run_id in adapter._run_streams

        # Past the post-terminal grace the transport goes; status remains.
        adapter._sweep_orphaned_runs_once(terminal_at + adapter._RUN_STREAM_TTL + 5)
        assert run_id not in adapter._run_streams
        assert adapter._run_statuses[run_id]["status"] == "completed"

        # Status expires on its own, longer TTL.
        adapter._sweep_orphaned_runs_once(terminal_at + adapter._RUN_STATUS_TTL + 5)
        assert run_id not in adapter._run_statuses


# ---------------------------------------------------------------------------
# POST /v1/runs/{run_id}/stop — interrupt a running agent
# ---------------------------------------------------------------------------


class TestStopRun:

    @pytest.mark.asyncio
    async def test_stop_keeps_uncooperative_executor_tracked_until_exit(self, adapter):
        """Cancelling an asyncio wrapper must not hide its live executor thread."""
        app = _create_runs_app(adapter)
        run_can_finish = threading.Event()
        run_finished = threading.Event()

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                started = threading.Event()

                def _run_conversation(*_args, **_kwargs):
                    started.set()
                    run_can_finish.wait(timeout=5)
                    run_finished.set()
                    return {"final_response": "late result"}

                mock_agent.run_conversation.side_effect = _run_conversation
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await resp.json())["run_id"]
                assert started.wait(timeout=3)

                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                await asyncio.sleep(0.1)

                assert not run_finished.is_set()
                assert run_id in adapter._active_run_agents
                assert run_id in adapter._active_run_tasks
                assert adapter._run_statuses[run_id]["status"] == "stopping"

                run_can_finish.set()
                for _ in range(40):
                    if run_id not in adapter._active_run_tasks:
                        break
                    await asyncio.sleep(0.05)

                assert run_id not in adapter._active_run_agents
                assert run_id not in adapter._active_run_tasks
                assert adapter._run_statuses[run_id]["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_stop_running_agent(self, adapter):
        """Stop should interrupt the agent and cancel the task."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, _ = _make_slow_agent()
                mock_create.return_value = mock_agent

                # Start run
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                # Wait for agent to start running in the thread
                agent_ready.wait(timeout=3.0)
                await asyncio.sleep(0.1)

                # Verify agent ref is stored
                assert run_id in adapter._active_run_agents

                # Stop the run
                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                stop_data = await stop_resp.json()
                assert stop_data["run_id"] == run_id
                assert stop_data["status"] == "stopping"

                # Agent interrupt should have been called
                mock_agent.interrupt.assert_called_once_with("Stop requested via API")

                status_resp = await cli.get(f"/v1/runs/{run_id}")
                assert status_resp.status == 200
                status_data = await status_resp.json()
                assert status_data["status"] in {"stopping", "cancelled"}

                # Refs should be cleaned up
                await asyncio.sleep(0.2)
                assert run_id not in adapter._active_run_agents
                assert run_id not in adapter._active_run_tasks


    @pytest.mark.asyncio
    async def test_stop_sends_sentinel_to_events_stream(self, adapter):
        """After stop, the events stream should close."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, _ = _make_slow_agent()
                mock_create.return_value = mock_agent

                # Start run
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                agent_ready.wait(timeout=3.0)
                await asyncio.sleep(0.1)

                # Subscribe to events in background
                events_task = asyncio.ensure_future(
                    cli.get(f"/v1/runs/{run_id}/events")
                )

                await asyncio.sleep(0.1)

                # Stop the run
                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200

                # Events stream should close
                events_resp = await asyncio.wait_for(events_task, timeout=5.0)
                assert events_resp.status == 200
                body = await events_resp.text()
                # Stream should have received run.failed and closed
                assert "run.failed" in body or "stream closed" in body


class TestRunsProviderAuthFailure:
    @pytest.mark.asyncio
    async def test_status_reports_provider_auth_failure_distinctly(self, adapter):
        """/v1/runs builds its own agent via _create_agent() and does not
        route through _run_agent(), so the controlled "Provider
        authentication failed" message added there does not cover this
        endpoint. _handle_runs()'s own _ProviderAuthResolutionError branch
        must give the same distinguished message instead of the generic
        except-Exception "run failed" text."""
        from gateway.platforms.api_server import _ProviderAuthResolutionError

        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_create.side_effect = _ProviderAuthResolutionError(
                    "No credentials found for provider 'nous'"
                )

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                for _ in range(40):
                    status_resp = await cli.get(f"/v1/runs/{run_id}")
                    status = await status_resp.json()
                    if status["status"] == "failed":
                        break
                    await asyncio.sleep(0.05)

                assert status["status"] == "failed"
                assert status["error"] == "⚠️ Provider authentication failed: No credentials found for provider 'nous'"
                assert status["last_event"] == "run.failed"
