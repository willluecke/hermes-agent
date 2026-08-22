"""Governed Sol-to-Opus implementation orchestration tests."""

import json

import tools.opus_worker_tool as worker


def _authority():
    return {
        "provider": "openai-codex",
        "model": "gpt-5.6-sol",
        "effort": "xhigh",
        "runtime": "codex_app_server",
    }


def _completed_job(job_id: str, thread_id: str) -> dict:
    return {
        "id": job_id,
        "threadId": thread_id,
        "provider": "claude",
        "model": "claude-opus-5",
        "effort": "high",
        "mode": "implement",
        "project": "hermes-chat",
        "status": "completed",
        "workerId": "command-center",
        "summary": "Implemented the accepted specification.",
        "result": "Outcome: complete\nChecks: 12 passed",
        "branch": f"agent/{job_id}",
        "worktree": f"/tmp/{job_id}/worktree",
        "commit": "a" * 40,
        "baseCommit": "b" * 40,
    }


def test_run_queues_exact_opus_job_and_returns_review_artifact(monkeypatch):
    monkeypatch.setattr(worker, "_authority", _authority)
    captured = {}

    def request(method, path, payload=None, **kwargs):
        if method == "GET" and path.startswith("/management/jobs/"):
            raise worker._SyncError("not found", status=404)
        if method == "GET" and path == "/management/workers":
            return {
                "workers": [
                    {
                        "updatedAt": worker.time.time() * 1_000,
                        "providers": {
                            "claude": {"available": True, "authenticated": True}
                        },
                    }
                ]
            }
        if method == "POST" and path == "/management/jobs":
            captured.update(payload)
            return {"job": _completed_job(payload["id"], payload["threadId"])}
        raise AssertionError((method, path, payload))

    monkeypatch.setattr(worker, "_sync_request", request)
    result = json.loads(
        worker.opus_code_worker_tool(
            "run",
            session_id="hermes-chat-session_123",
            project="hermes-chat",
            title="Implement orchestration",
            specification="Add the bounded worker handoff.",
            acceptance_checks=["pytest -q tests/tools/test_opus_worker_tool.py"],
            constraints=["Do not deploy."],
            wait_seconds=0,
        )
    )

    assert result["ok"] is True
    assert result["status"] == "completed"
    assert result["review_required"] is True
    assert result["commit"] == "a" * 40
    assert captured["provider"] == "claude"
    assert captured["model"] == "claude-opus-5"
    assert captured["effort"] == "high"
    assert captured["mode"] == "implement"
    assert captured["threadId"].startswith("hermes_hermes-chat-session_123_")
    assert "DECISION_NEEDED" in captured["brief"]
    assert "pytest -q" in captured["brief"]


def test_exact_retry_reuses_deterministic_existing_job(monkeypatch):
    monkeypatch.setattr(worker, "_authority", _authority)
    thread_id = worker._thread_id("stable-session")
    job_id = worker._job_id(
        thread_id=thread_id,
        project="hermes-chat",
        title="Stable job",
        specification="Make one change.",
        acceptance_checks=["test it"],
        constraints=[],
        attempt=1,
    )
    existing = _completed_job(job_id, thread_id)
    posts = []

    def request(method, path, payload=None, **kwargs):
        if method == "GET" and path == f"/management/jobs/{job_id}":
            return {"job": existing}
        if method == "POST":
            posts.append(payload)
        raise AssertionError((method, path, payload))

    monkeypatch.setattr(worker, "_sync_request", request)
    result = json.loads(
        worker.opus_code_worker_tool(
            "run",
            session_id="stable-session",
            project="hermes-chat",
            title="Stable job",
            specification="Make one change.",
            acceptance_checks=["test it"],
            wait_seconds=0,
        )
    )

    assert result["job_id"] == job_id
    assert result["status"] == "completed"
    assert posts == []


def test_running_status_returns_recovery_instruction(monkeypatch):
    monkeypatch.setattr(worker, "_authority", _authority)
    thread_id = worker._thread_id("same-session")
    running = {
        "id": "job_running123",
        "threadId": thread_id,
        "provider": "claude",
        "model": "claude-opus-5",
        "effort": "high",
        "mode": "implement",
        "project": "reccli",
        "status": "running",
        "activity": "Claude: Bash",
        "progress": "Running tests",
    }
    monkeypatch.setattr(
        worker,
        "_sync_request",
        lambda method, path, payload=None, **kwargs: {"job": running},
    )

    result = json.loads(
        worker.opus_code_worker_tool(
            "status",
            session_id="same-session",
            job_id="job_running123",
            wait_seconds=0,
        )
    )

    assert result["ok"] is True
    assert result["status"] == "running"
    assert result["activity"] == "Claude: Bash"
    assert "action='status'" in result["next_action"]


def test_status_rejects_job_from_another_conversation(monkeypatch):
    monkeypatch.setattr(worker, "_authority", _authority)
    foreign = _completed_job("job_foreign", worker._thread_id("other-session"))
    monkeypatch.setattr(
        worker,
        "_sync_request",
        lambda method, path, payload=None, **kwargs: {"job": foreign},
    )

    result = json.loads(
        worker.opus_code_worker_tool(
            "status",
            session_id="current-session",
            job_id="job_foreign",
        )
    )

    assert result["ok"] is False
    assert "does not belong" in result["error"]


def test_cancel_updates_owned_nonterminal_job(monkeypatch):
    monkeypatch.setattr(worker, "_authority", _authority)
    thread_id = worker._thread_id("cancel-session")
    running = {
        "id": "job_cancel123",
        "threadId": thread_id,
        "provider": "claude",
        "mode": "implement",
        "project": "hermes-chat",
        "status": "running",
    }
    calls = []

    def request(method, path, payload=None, **kwargs):
        calls.append((method, path, payload))
        if method == "GET":
            return {"job": running}
        return {"job": {**running, "status": "cancelled"}}

    monkeypatch.setattr(worker, "_sync_request", request)
    result = json.loads(
        worker.opus_code_worker_tool(
            "cancel",
            session_id="cancel-session",
            job_id="job_cancel123",
        )
    )

    assert result["status"] == "cancelled"
    assert calls[-1] == (
        "PATCH",
        "/management/jobs/job_cancel123",
        {"status": "cancelled"},
    )


def test_tool_is_core_registered_and_exposed():
    from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS
    from tools.registry import registry
    import toolsets

    assert registry.get_entry("opus_code_worker") is not None
    assert "opus_code_worker" in toolsets._HERMES_CORE_TOOLS
    assert toolsets.TOOLSETS["opus_worker"]["tools"] == ["opus_code_worker"]
    assert "opus_code_worker" in EXPOSED_TOOLS
