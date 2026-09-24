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

    def youngest_background_work_age(self):
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
        "exit_status": None,
        "ok": None,
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
    # nothing recorded and systemd gives no verdict: the outcome is unknown
    assert (first[0]["exit_status"], first[0]["ok"]) == (None, None)
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


def test_registered_jobs_prefers_the_recorded_outcome_to_probing(tmp_path):
    _write_job(tmp_path, "hermes-bg-ok", "hermes-chat-c1", ended_at=1900.0, exit_status=0)
    _write_job(tmp_path, "hermes-bg-bad", "hermes-chat-c1", ended_at=1950.0, exit_status=3)
    probes = []

    jobs = background_jobs.registered_jobs(
        "hermes-chat-c1", directory=tmp_path, now=2000.0,
        unit_state=lambda unit: probes.append(unit) or "active",
    )

    assert probes == []
    assert [(j["id"], j["running"], j["ended_at"], j["exit_status"], j["ok"]) for j in jobs] == [
        ("hermes-bg-bad", False, 1950.0, 3, False),
        ("hermes-bg-ok", False, 1900.0, 0, True),
    ]


def test_a_failed_unit_with_nothing_recorded_is_not_ok(tmp_path):
    path = _write_job(tmp_path, "hermes-bg-a", "hermes-chat-c1")

    [job] = background_jobs.registered_jobs(
        "hermes-chat-c1", directory=tmp_path, now=2000.0,
        unit_state=lambda unit: "failed",
    )

    assert (job["running"], job["end_state"], job["exit_status"], job["ok"]) == (
        False, "failed", None, False,
    )
    assert json.loads(path.read_text())["end_state"] == "failed"


@pytest.mark.parametrize("state", ["inactive", "not-found"])
def test_a_unit_that_reads_gone_right_after_launch_is_not_stamped(tmp_path, state):
    path = _write_job(tmp_path, "hermes-bg-a", "hermes-chat-c1", started_at=1990.0)

    [job] = background_jobs.registered_jobs(
        "hermes-chat-c1", directory=tmp_path, now=2000.0,
        unit_state=lambda unit: state,
    )

    assert job["running"] is True
    assert "ended_at" not in json.loads(path.read_text())

    [later] = background_jobs.registered_jobs(
        "hermes-chat-c1", directory=tmp_path,
        now=1990.0 + background_jobs.UNIT_START_GRACE_SECONDS + 1,
        unit_state=lambda unit: state,
    )
    assert later["running"] is False
    assert json.loads(path.read_text())["end_state"] == state


def test_a_failed_unit_is_ended_even_inside_the_launch_grace(tmp_path):
    _write_job(tmp_path, "hermes-bg-a", "hermes-chat-c1", started_at=1990.0)

    [job] = background_jobs.registered_jobs(
        "hermes-chat-c1", directory=tmp_path, now=2000.0,
        unit_state=lambda unit: "failed",
    )

    assert (job["running"], job["ok"]) == (False, False)


def test_an_outcome_recorded_while_probing_is_kept_not_stamped_over(tmp_path):
    path = _write_job(tmp_path, "hermes-bg-a", "hermes-chat-c1")

    def runner_finishes_then_unit_is_collected(unit):
        entry = json.loads(path.read_text())
        path.write_text(json.dumps({**entry, "ended_at": 1999.0, "exit_status": 0}))
        return "not-found"

    [job] = background_jobs.registered_jobs(
        "hermes-chat-c1", directory=tmp_path, now=2000.0,
        unit_state=runner_finishes_then_unit_is_collected,
    )

    assert (job["ended_at"], job["exit_status"], job["ok"]) == (1999.0, 0, True)
    stored = json.loads(path.read_text())
    assert (stored["ended_at"], stored["exit_status"]) == (1999.0, 0)
    assert "end_state" not in stored


def test_the_stamp_replaces_the_registry_file_whole(tmp_path):
    registry = tmp_path / "background-jobs"
    registry.mkdir()
    path = _write_job(registry, "hermes-bg-a", "hermes-chat-c1")

    background_jobs.registered_jobs(
        "hermes-chat-c1", directory=registry, now=2000.0,
        unit_state=lambda unit: "not-found",
    )

    assert json.loads(path.read_text())["ended_at"] == 2000.0
    assert sorted(p.name for p in registry.iterdir()) == ["hermes-bg-a.json"]


@pytest.mark.parametrize("stdout, expected", [
    ("ActiveState=active\nLoadState=loaded\n", "active"),
    ("LoadState=loaded\nActiveState=failed\n", "failed"),
    ("LoadState=not-found\nActiveState=inactive\n", "not-found"),
    ("", None),
])
def test_systemd_unit_state_reads_load_and_active_state(monkeypatch, stdout, expected):
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(stdout=stdout, returncode=0)

    monkeypatch.setattr(background_jobs.subprocess, "run", run)

    assert background_jobs.systemd_unit_state("hermes-bg-a") == expected
    assert "--property=LoadState,ActiveState" in calls[0]


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


@pytest.mark.asyncio
async def test_session_background_route_lists_a_late_turn_and_job_outcomes(tmp_path, monkeypatch):
    from agent.transports.claude_code_session import ClaudeCodeSession

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    registry = tmp_path / "background-jobs"
    registry.mkdir()
    now = time.time()
    _write_job(registry, "hermes-bg-a", "hermes-chat-c1", title="Render",
               started_at=now - 60, ended_at=now - 5, exit_status=3)
    monkeypatch.setattr(background_jobs, "systemd_unit_state", lambda unit: "active")

    session = ClaudeCodeSession(cwd="/tmp", model="claude-fable-5")
    process = SimpleNamespace(pid=40101, poll=lambda: None)
    session._process = process
    session._background_tasks.follow(process)
    session._observe_stdout_line(process, json.dumps({"type": "assistant", "message": {}}) + "\n")

    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "secret"}))
    adapter.gateway_runner = SimpleNamespace(
        _agent_cache=OrderedDict({
            "api_server:default:hermes-chat-c1": (_agent(session=session), "sig", 3, "hermes-chat-c1"),
        }),
        _agent_cache_lock=threading.Lock(),
    )
    app = web.Application()
    app.router.add_get(
        "/api/sessions/{session_id}/background", adapter._handle_session_background
    )

    async with TestClient(TestServer(app)) as client:
        response = await client.get(
            "/api/sessions/hermes-chat-c1/background",
            headers={"Authorization": "Bearer secret"},
        )
        payload = await response.json()

    [late] = payload["tasks"]
    assert late == {
        "task_id": "late-turn", "task_type": "late_turn",
        "description": "Writing a follow-up reply", "ambient": False,
        "started_at": late["started_at"], "kind": "claude_task", "running": True,
    }
    assert now - 5 <= late["started_at"] <= time.time()
    [job] = payload["jobs"]
    assert (job["running"], job["exit_status"], job["ok"]) == (False, 3, False)


def test_pressure_sweep_keeps_an_agent_whose_cli_runs_background_work(monkeypatch):
    import gateway.agent_cache_pressure as acp
    from gateway.agent_cache_pressure import AgentCacheBounds

    monkeypatch.setattr(acp, "read_anon_rss_mb", lambda: 4000)
    runner = _runner()
    runner._agent_cache_bounds_cache = AgentCacheBounds(
        memory_high_mb=1000, max_evictions_per_pass=8, protect_recent=0
    )
    released = []
    runner._commit_then_release_soft = lambda agent, key: released.append(key)
    for key, session in (("working", _Session(age=60.0)), ("idle", _Session(age=None))):
        agent = _agent(session=session)
        agent._session_messages = [{"role": "user", "content": "x"}]
        agent._last_flushed_db_idx = 1  # persisted, so only the work guard can keep it
        runner._agent_cache[key] = (agent, "sig")

    assert runner._sweep_agent_cache_under_pressure() == 1

    assert "working" in runner._agent_cache
    assert "idle" not in runner._agent_cache


# --- A late answer does not evict the agent that wrote it -------------------


class _ResidentAgent:
    """Just enough of an AIAgent for the API server's resident-agent cache."""

    def __init__(self, session_id, session_db):
        self.session_id = session_id
        self._session_db = session_db
        self._claude_code_session = MagicMock()

    def release_clients(self):
        # AIAgent.release_clients closes the native session: it kills the CLI.
        session, self._claude_code_session = self._claude_code_session, None
        session.close()


@pytest.fixture
def resident_cache(tmp_path, monkeypatch):
    from agent import claude_runtime
    from hermes_state import SessionDB

    monkeypatch.setattr(claude_runtime, "_sync_store_ping", lambda session_id: None)
    db = SessionDB(tmp_path / "state.db")
    session_id = db.create_session("hermes-chat-c_late", "api_server")
    db.append_message(session_id, "user", "Start the build")
    db.append_message(session_id, "assistant", "Started; I will report back.")

    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "secret"}))
    adapter._session_db = db
    adapter.gateway_runner = _runner()
    created = []

    def create_agent(session_id=None, **kwargs):
        created.append(_ResidentAgent(session_id, db))
        return created[-1]

    adapter._create_agent = create_agent

    def acquire():
        return adapter._create_or_reuse_runtime_agent(
            cache_key=f"api_server:default:{session_id}",
            signature="sig",
            session_id=session_id,
        )

    try:
        yield SimpleNamespace(db=db, session_id=session_id, acquire=acquire,
                              cache=adapter.gateway_runner._agent_cache, created=created)
    finally:
        db.close()


def test_a_late_answer_keeps_the_resident_agent_and_its_cli(resident_cache):
    from agent.claude_runtime import save_claude_late_answer

    agent, reused = resident_cache.acquire()
    assert reused is False
    cli = agent._claude_code_session

    assert save_claude_late_answer(agent, {"text": "The build passed.", "complete": True})
    again, reused = resident_cache.acquire()

    assert reused is True and again is agent
    cli.close.assert_not_called()
    assert agent._claude_code_session is cli
    # the checkpoint moved past the late answer
    [entry] = resident_cache.cache.values()
    assert entry[2] == resident_cache.db.get_session(resident_cache.session_id)["message_count"]


def test_a_long_run_of_late_answers_keeps_the_resident_agent(resident_cache):
    from agent.claude_runtime import save_claude_late_answer

    agent, _ = resident_cache.acquire()
    cli = agent._claude_code_session
    # a Monitor on a CI log wakes the CLI many times while nobody chats
    for n in range(40):
        assert save_claude_late_answer(agent, {"text": f"CI event {n}", "complete": True})

    again, reused = resident_cache.acquire()

    assert reused is True and again is agent
    cli.close.assert_not_called()


def test_a_late_answer_landing_mid_check_cannot_hide_real_drift(resident_cache, monkeypatch):
    from agent.claude_runtime import save_claude_late_answer

    agent, _ = resident_cache.acquire()
    cli = agent._claude_code_session
    db = resident_cache.db
    db.append_message(resident_cache.session_id, "user", "Ship it")  # real drift
    tail_read = db.newest_message_display_kinds
    landed = []

    def tail_read_after_a_late_answer(session_id, limit):
        # the late-answer thread writes between the count and the tail read
        if not landed:
            landed.append(save_claude_late_answer(agent, {"text": "Done.", "complete": True}))
        return tail_read(session_id, limit)

    monkeypatch.setattr(db, "newest_message_display_kinds", tail_read_after_a_late_answer)

    replacement, reused = resident_cache.acquire()

    assert len(landed) == 1 and landed[0]  # the late answer was stored
    assert reused is False and replacement is not agent
    cli.close.assert_called_once()


def test_a_genuine_transcript_change_still_replaces_the_resident_agent(resident_cache):
    from agent.claude_runtime import save_claude_late_answer

    agent, _ = resident_cache.acquire()
    cli = agent._claude_code_session
    save_claude_late_answer(agent, {"text": "The build passed.", "complete": True})
    # another writer (a second tab, a handoff) adds to the transcript
    resident_cache.db.append_message(resident_cache.session_id, "user", "Ship it")

    replacement, reused = resident_cache.acquire()

    assert reused is False and replacement is not agent
    cli.close.assert_called_once()


def test_concurrent_registry_writes_never_tear_the_file(tmp_path):
    import concurrent.futures

    path = tmp_path / "hermes-bg-race.json"
    entry = {"unit": "hermes-bg-race", "session_id": "hermes-chat-c1",
             "padding": "x" * 200_000}  # slow enough writes to overlap
    background_jobs._write_entry(path, entry)
    torn = []
    stop = threading.Event()

    def read_continuously():
        while not stop.is_set():
            try:
                json.loads(path.read_text())
            except ValueError:
                torn.append(1)
            except OSError:
                pass

    def write_many(n):
        for index in range(40):
            background_jobs._write_entry(path, {**entry, "writer": n, "index": index})

    reader = threading.Thread(target=read_continuously)
    reader.start()
    try:
        with concurrent.futures.ThreadPoolExecutor(6) as pool:
            for future in [pool.submit(write_many, n) for n in range(6)]:
                future.result()
    finally:
        stop.set()
        reader.join()

    assert torn == []
    assert json.loads(path.read_text())["unit"] == "hermes-bg-race"
    assert not list(tmp_path.glob(".*.tmp"))  # no temporary files left behind


def test_concurrent_polls_stamp_an_ended_job_once(tmp_path, monkeypatch):
    import concurrent.futures

    path = _write_job(tmp_path, "hermes-bg-once", "hermes-chat-c1")
    writes = []
    real_write = background_jobs._write_entry

    def counting_write(target, entry):
        writes.append(target.name)
        real_write(target, entry)

    monkeypatch.setattr(background_jobs, "_write_entry", counting_write)
    real_load = background_jobs._load_entry

    def slow_load(target):
        loaded = real_load(target)
        time.sleep(0.05)  # hold every poll between its read and its stamp
        return loaded

    monkeypatch.setattr(background_jobs, "_load_entry", slow_load)
    gate = threading.Barrier(4)

    def poll():
        gate.wait()
        return background_jobs.registered_jobs(
            "hermes-chat-c1", directory=tmp_path, now=2000.0,
            unit_state=lambda unit: "inactive",
        )

    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        results = [future.result() for future in [pool.submit(poll) for _ in range(4)]]

    assert writes == ["hermes-bg-once.json"]  # the lock lets one poll stamp it
    assert json.loads(path.read_text())["ended_at"] == 2000.0
    assert all([job["running"] for job in jobs] == [False] for jobs in results)
