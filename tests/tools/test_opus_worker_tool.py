"""Governed Sol-to-Opus implementation orchestration tests."""

import pytest
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


# ---- Parent-runtime authority propagated through the managed MCP child ----


def _clear_parent_runtime(monkeypatch):
    for name in (
        "HERMES_PARENT_PROVIDER",
        "HERMES_PARENT_MODEL",
        "HERMES_PARENT_EFFORT",
        "HERMES_PARENT_OPUS_WORKER",
    ):
        monkeypatch.delenv(name, raising=False)


def _set_parent_runtime(monkeypatch, provider, model, effort, opus="1"):
    monkeypatch.setenv("HERMES_PARENT_PROVIDER", provider)
    monkeypatch.setenv("HERMES_PARENT_MODEL", model)
    monkeypatch.setenv("HERMES_PARENT_EFFORT", effort)
    monkeypatch.setenv("HERMES_PARENT_OPUS_WORKER", opus)


def test_eligible_gpt6_astra_parent_is_an_accepted_authority(monkeypatch):
    """The bounded single-model exception reaches the MCP child through the
    whitelisted env, and the child accepts it on its own runtime facts."""
    _set_parent_runtime(monkeypatch, "openai-codex", "gpt-6-astra", "high")

    assert worker._authority() == {
        "provider": "openai-codex",
        "model": "gpt-6-astra",
        "effort": "high",
        "runtime": "codex_app_server",
    }


def test_exact_sol_orchestrator_runtime_is_still_accepted(monkeypatch):
    _set_parent_runtime(monkeypatch, "openai-codex", "gpt-5.6-sol", "xhigh")

    assert worker._authority()["model"] == "gpt-5.6-sol"


def test_disallowed_direct_parent_is_rejected(monkeypatch, tmp_path):
    """A runtime that is neither the exact orchestrator nor the eligible
    GPT-6 Astra parent must not reach Opus, even though the host config still
    describes a valid Sol authority and the sync key is present — so the tool
    is not merely unusable, it is never registered."""
    monkeypatch.setattr(
        "tools.decision_log_tool._authority", _authority, raising=False
    )
    key_file = tmp_path / "hermes-api-key"
    key_file.write_text("test-key", encoding="utf-8")
    monkeypatch.setattr(worker, "_key_path", lambda: key_file)

    _set_parent_runtime(monkeypatch, "openai-codex", "gpt-6-astra", "high")
    assert worker._configured() is True

    for provider, model, effort in (
        ("openrouter", "gpt-6-astra", "high"),
        ("openai-codex", "gpt-5.5", "high"),
        ("openai-codex", "gpt-5.6-sol", "high"),
        ("anthropic", "claude-opus-5", "high"),
    ):
        _set_parent_runtime(monkeypatch, provider, model, effort)
        try:
            worker._authority()
        except ValueError as exc:
            assert "Opus delegation requires" in str(exc)
        else:  # pragma: no cover - contract violation
            raise AssertionError(f"{provider}/{model} was accepted")
        assert worker._configured() is False


def test_parent_that_denied_opus_worker_is_rejected(monkeypatch):
    _set_parent_runtime(monkeypatch, "openai-codex", "gpt-6-astra", "high", opus="0")

    try:
        worker._authority()
    except ValueError as exc:
        assert "disabled opus_worker" in str(exc)
    else:  # pragma: no cover - contract violation
        raise AssertionError("a denied parent was accepted")


def test_absent_runtime_metadata_falls_back_to_config_authority(monkeypatch):
    """Native/non-gateway paths that carry no runtime metadata keep the
    existing config-derived orchestrator authority."""
    _clear_parent_runtime(monkeypatch)
    monkeypatch.setattr(
        "tools.decision_log_tool._authority", _authority, raising=False
    )

    assert worker._authority()["model"] == "gpt-5.6-sol"


def test_run_names_the_actual_gpt6_authority_in_job_metadata(monkeypatch):
    """Queued job metadata and the worker brief must name the authority that
    actually accepted the specification, not a hard-coded Hermes/Sol."""
    _set_parent_runtime(monkeypatch, "openai-codex", "gpt-6-astra", "high")
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
            session_id="astra-session",
            project="hermes-agent",
            title="Implement the bounded exception",
            specification="Make the delegation contract explicit.",
            acceptance_checks=["pytest -q tests/tools/test_opus_worker_tool.py"],
            wait_seconds=0,
        )
    )

    assert result["ok"] is True
    assert captured["contextDetail"].startswith("Authority: gpt-6-astra at high;")
    assert captured["contextTitle"] == "Hermes/gpt-6-astra at high implementation handoff"
    assert captured["brief"].startswith(
        "Hermes/gpt-6-astra at high has accepted the following bounded"
    )
    assert "Hermes/Sol" not in captured["brief"]
    # The worker itself stays pinned to exact Opus 5 regardless of authority.
    assert captured["model"] == "claude-opus-5"
    assert captured["effort"] == "high"


def test_tool_is_core_registered_and_exposed():
    from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS
    from tools.registry import registry
    import toolsets

    assert registry.get_entry("opus_code_worker") is not None
    assert "opus_code_worker" in toolsets._HERMES_CORE_TOOLS
    assert toolsets.TOOLSETS["opus_worker"]["tools"] == ["opus_code_worker"]
    assert "opus_code_worker" in EXPOSED_TOOLS


# Parent continuity: a browser-authorized worker must stay reachable from the
# conversation that dispatched it, without that parent identity ever being
# mistaken for the worker's own execution identity.
def test_browser_parent_identity_is_not_a_worker_execution_identity():
    assert worker._parent_context("hermes-chat-parent_123") == {
        "parentConversationId": "parent_123", "parentSessionId": "hermes-chat-parent_123"
    }
    assert worker._parent_context("cli-session") == {}
    assert worker._parent_context("hermes-chat-../../outside") == {}


def test_owned_isolated_child_and_legacy_jobs_are_both_recoverable(monkeypatch):
    parent_thread = worker._thread_id("hermes-chat-parent_123")
    job = {"id": "job_one", "threadId": "worker_one", "contextId": parent_thread,
           "contextType": "hermes-orchestration", "parentConversationId": "parent_123",
           "provider": "claude", "mode": "implement"}
    monkeypatch.setattr(worker, "_get_job", lambda _: job)
    assert worker._owned_job("job_one", parent_thread) == job
    with pytest.raises(ValueError, match="does not belong"):
        worker._owned_job("job_one", "another_parent")
    assert worker._belongs_to_thread({"threadId": parent_thread}, parent_thread)


def test_completion_receipt_is_saved_without_marking_human_review(monkeypatch):
    calls = []
    monkeypatch.setattr(worker, "_sync_request", lambda *args, **kwargs: calls.append(args) or {"ok": True})
    result = worker._wait_for_job({"id": "job_one", "status": "completed",
        "parentSessionId": "hermes-chat-parent", "parentConversationId": "parent",
        "result": "Tests pass", "threadId": "worker_one"}, 0)
    assert result["parent_receipt_saved"] is True
    assert result["review_required"] is True
    assert result["parent_conversation_id"] == "parent"
    assert calls == [("POST", "/management/orchestration/jobs/job_one/observed", {"parentSessionId": "hermes-chat-parent"})]


def test_dispatch_attaches_parent_metadata_and_accepts_host_isolated_thread(monkeypatch):
    posted = []
    monkeypatch.setattr(worker, "_authority", lambda: {"model": "gpt-6-astra", "effort": "high"})
    monkeypatch.setattr(worker, "_get_job_or_none", lambda _: None)
    monkeypatch.setattr(worker, "_worker_available", lambda: True)
    def request(method, path, payload):
        posted.append(payload)
        return {"job": {**payload, "threadId": "worker_isolated", "status": "queued"}}
    monkeypatch.setattr(worker, "_sync_request", request)
    result = worker._run_job(project="hermes-chat", title="Navigation", specification="Implement saved choices",
        acceptance_checks=["Mobile checks pass"], session_id="hermes-chat-parent", wait_seconds=0)
    assert posted[0]["parentConversationId"] == "parent"
    assert posted[0]["parentSessionId"] == "hermes-chat-parent"
    assert result["thread_id"] == "worker_isolated"
