"""Background work that outlives a turn: kept alive by the cache sweeps and
listed for the Hermes Chat indicator (GET /api/sessions/{id}/background)."""

import json
import threading
import time
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway import background_jobs
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


class _Session:
    def __init__(self, age=None, tasks=None):
        self.age = age
        self.tasks = tasks or []

    def background_work_age(self):
        return self.age

    def background_tasks(self):
        return list(self.tasks)


def _runner():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._agent_cache = OrderedDict()
    runner._agent_cache_lock = threading.Lock()
    runner._running_agents = {}
    runner._cleanup_agent_resources = MagicMock()
    return runner


def _agent(idle_seconds=0.0, session=None):
    agent = MagicMock()
    agent._last_activity_ts = time.time() - idle_seconds
    agent._claude_code_session = session
    return agent


def test_idle_sweep_keeps_an_agent_whose_cli_runs_background_work(monkeypatch):
    from gateway import run as gw_run

    monkeypatch.setattr(gw_run, "_AGENT_CACHE_IDLE_TTL_SECS", 0.01)
    runner = _runner()
    working = _agent(idle_seconds=10.0, session=_Session(age=3_700.0))
    idle = _agent(idle_seconds=10.0, session=_Session(age=None))
    runner._agent_cache["working"] = (working, "sig")
    runner._agent_cache["idle"] = (idle, "sig")

    evicted = runner._sweep_idle_cached_agents()

    assert evicted == 1
    assert "working" in runner._agent_cache
    assert "idle" not in runner._agent_cache


def test_idle_sweep_evicts_background_work_past_the_age_bound(monkeypatch):
    from gateway import run as gw_run

    monkeypatch.setattr(gw_run, "_AGENT_CACHE_IDLE_TTL_SECS", 0.01)
    runner = _runner()
    stuck = _agent(
        idle_seconds=10.0,
        session=_Session(age=gw_run._BACKGROUND_WORK_MAX_AGE_SECS + 1),
    )
    runner._agent_cache["stuck"] = (stuck, "sig")

    assert runner._sweep_idle_cached_agents() == 1
    assert "stuck" not in runner._agent_cache


def test_cap_skips_an_lru_agent_whose_cli_runs_background_work(monkeypatch):
    from gateway import run as gw_run

    monkeypatch.setattr(gw_run, "_AGENT_CACHE_MAX_SIZE", 1)
    runner = _runner()
    working = _agent(session=_Session(age=60.0))
    newer = _agent()
    runner._agent_cache["working"] = (working, "sig")
    runner._agent_cache["newer"] = (newer, "sig")

    with runner._agent_cache_lock:
        runner._enforce_agent_cache_cap()

    assert "working" in runner._agent_cache
    assert "newer" in runner._agent_cache


def _write_job(directory, unit, session_id, **extra):
    entry = {
        "id": unit,
        "unit": unit,
        "session_id": session_id,
        "title": extra.pop("title", "Build"),
        "kind": "job",
        "started_at": extra.pop("started_at", 1000.0),
        "status_file": extra.pop("status_file", None),
        "command": ["true"],
        **extra,
    }
    path = directory / f"{unit}.json"
    path.write_text(json.dumps(entry))
    return path


def test_registered_jobs_lists_running_jobs_of_the_session_with_status(tmp_path):
    status = tmp_path / "status.json"
    status.write_text(json.dumps({
        "stage": "implementing",
        "steps": {"implement-app": {"state": "running", "log": "x"}},
        "updated_at": 1500.0,
    }))
    _write_job(tmp_path, "hermes-bg-a", "hermes-chat-c1", status_file=str(status))
    _write_job(tmp_path, "hermes-bg-b", "hermes-chat-other")

    jobs = background_jobs.registered_jobs(
        "hermes-chat-c1", directory=tmp_path, now=2000.0,
        unit_state=lambda unit: "active",
    )

    assert jobs == [{
        "id": "hermes-bg-a",
        "kind": "job",
        "title": "Build",
        "running": True,
        "started_at": 1000.0,
        "ended_at": None,
        "end_state": None,
        "status": {
            "stage": "implementing",
            "steps": {"implement-app": "running"},
            "updated_at": 1500.0,
        },
    }]


def test_registered_jobs_stamps_the_end_once_then_hides_it(tmp_path):
    path = _write_job(tmp_path, "hermes-bg-a", "hermes-chat-c1")
    probes = []

    def state(unit):
        probes.append(unit)
        return "inactive"

    first = background_jobs.registered_jobs(
        "hermes-chat-c1", directory=tmp_path, now=2000.0, unit_state=state,
    )
    assert [job["running"] for job in first] == [False]
    assert first[0]["ended_at"] == 2000.0
    assert json.loads(path.read_text())["ended_at"] == 2000.0

    later = background_jobs.registered_jobs(
        "hermes-chat-c1", directory=tmp_path,
        now=2000.0 + background_jobs.FINISHED_JOB_VISIBLE_SECONDS + 1,
        unit_state=state,
    )
    assert later == []
    assert probes == ["hermes-bg-a"]  # an ended job is never probed again

    background_jobs.registered_jobs(
        "hermes-chat-c1", directory=tmp_path,
        now=2000.0 + background_jobs.FINISHED_JOB_RETENTION_SECONDS + 1,
        unit_state=state,
    )
    assert not path.exists()


def test_registered_jobs_reports_nothing_when_systemd_cannot_answer(tmp_path):
    _write_job(tmp_path, "hermes-bg-a", "hermes-chat-c1")

    jobs = background_jobs.registered_jobs(
        "hermes-chat-c1", directory=tmp_path, now=2000.0,
        unit_state=lambda unit: None,
    )

    assert jobs == []


@pytest.mark.asyncio
async def test_session_background_route_merges_cli_tasks_and_jobs(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    registry = tmp_path / "background-jobs"
    registry.mkdir()
    _write_job(registry, "hermes-bg-a", "hermes-chat-c1", title="Plan question UI build")
    monkeypatch.setattr(background_jobs, "systemd_unit_state", lambda unit: "active")

    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "secret"}))
    task = {"task_id": "w1", "task_type": "local_workflow",
            "description": "review-changes", "ambient": False, "started_at": 1.0}
    working = _agent(session=_Session(age=5.0, tasks=[task]))
    other = _agent(session=_Session(age=5.0, tasks=[{**task, "task_id": "w2"}]))
    adapter.gateway_runner = SimpleNamespace(
        _agent_cache=OrderedDict({
            "api_server:default:hermes-chat-c1": (working, "sig", 3, "hermes-chat-c1"),
            "api_server:default:hermes-chat-c2": (other, "sig", 3, "hermes-chat-c2"),
        }),
        _agent_cache_lock=threading.Lock(),
    )
    app = web.Application()
    app.router.add_get(
        "/api/sessions/{session_id}/background", adapter._handle_session_background
    )

    async with TestClient(TestServer(app)) as client:
        denied = await client.get("/api/sessions/hermes-chat-c1/background")
        assert denied.status == 401
        response = await client.get(
            "/api/sessions/hermes-chat-c1/background",
            headers={"Authorization": "Bearer secret"},
        )
        assert response.status == 200
        payload = await response.json()

    assert payload["object"] == "hermes.session.background"
    assert [t["task_id"] for t in payload["tasks"]] == ["w1"]
    assert payload["tasks"][0]["kind"] == "claude_task"
    assert [j["title"] for j in payload["jobs"]] == ["Plan question UI build"]
    assert payload["jobs"][0]["running"] is True
