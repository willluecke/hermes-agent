"""A worker PID recorded in another PID namespace is not this host's process.

A kanban worker dispatched from inside a sandboxed turn records its
namespace PID (``5``). On the host that number is a kernel thread, so the
liveness probe said "alive" forever and the dispatcher extended a claim for
a worker that had died with its sandbox (reg-watch, 2026-10-18). The kernel
now records the spawner's PID namespace with the PID and trusts the probe
only when the namespaces match.
"""

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _running_with_expired_claim(conn, title, *, pid, pid_ns):
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title=title, assignee="a")
    kb.claim_task(conn, tid, claimer=f"{host}:worker")
    kb._set_worker_pid(conn, tid, pid)
    conn.execute(
        "UPDATE tasks SET worker_pid_ns = ?, claim_expires = ? WHERE id = ?",
        (pid_ns, int(time.time()) - 60, tid),
    )
    return tid


def _events(conn, tid, kind):
    return [e for e in kb.list_events(conn, tid) if e.kind == kind]


def _mine():
    mine = kb._pid_namespace_id()
    if mine is None:
        pytest.skip("PID namespaces are not observable on this platform")
    return mine


def test_probe_trust_follows_the_recorded_namespace():
    mine = _mine()
    assert kb._worker_pid_probe_trusted(None) is True  # legacy row
    assert kb._worker_pid_probe_trusted(mine) is True
    assert kb._worker_pid_probe_trusted(mine + 1) is False


def test_set_worker_pid_records_the_spawner_namespace(kanban_home):
    mine = _mine()
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="spawn", assignee="a")
        kb._set_worker_pid(conn, tid, 4242)
        row = conn.execute(
            "SELECT worker_pid, worker_pid_ns FROM tasks WHERE id = ?", (tid,)
        ).fetchone()
        assert (row["worker_pid"], row["worker_pid_ns"]) == (4242, mine)
        spawned = _events(conn, tid, "spawned")[-1]
        assert spawned.payload == {"pid": 4242, "pid_ns": mine}


def test_foreign_namespace_pid_is_neither_extended_nor_signalled(
    kanban_home, monkeypatch,
):
    """Host PID 5 is alive (a kernel thread) but it is not our worker."""
    mine = _mine()
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: True)
    signalled = []
    with kb.connect() as conn:
        tid = _running_with_expired_claim(conn, "phantom", pid=5, pid_ns=mine + 1)
        reclaimed = kb.release_stale_claims(
            conn, signal_fn=lambda pid, sig: signalled.append(pid)
        )
        task = kb.get_task(conn, tid)
        assert reclaimed == 1
        assert task.status != "running" and task.claim_lock is None
        assert _events(conn, tid, "claim_extended") == []
        payload = _events(conn, tid, "reclaimed")[-1].payload
        assert payload["pid_probe_trusted"] is False
        assert payload["termination_attempted"] is False
    assert signalled == []


def test_same_namespace_live_pid_still_extends_the_claim(kanban_home, monkeypatch):
    mine = _mine()
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: True)
    with kb.connect() as conn:
        tid = _running_with_expired_claim(conn, "slow but alive", pid=4242, pid_ns=mine)
        assert kb.release_stale_claims(conn, signal_fn=lambda *_: None) == 0
        assert kb.get_task(conn, tid).status == "running"
        assert _events(conn, tid, "claim_extended")[-1].payload["reason"] == "pid_alive"


def test_crash_detector_skips_a_foreign_namespace_pid(kanban_home, monkeypatch):
    """Dead on the host means nothing for a PID from another namespace; the
    claim TTL path owns that row. A same-namespace dead PID is a crash."""
    mine = _mine()
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    with kb.connect() as conn:
        foreign = _running_with_expired_claim(conn, "foreign", pid=5, pid_ns=mine + 1)
        local = _running_with_expired_claim(conn, "local", pid=99999, pid_ns=mine)
        crashed = kb.detect_crashed_workers(conn)
        assert crashed == [local]
        assert kb.get_task(conn, foreign).status == "running"
        assert kb.get_task(conn, local).status != "running"
