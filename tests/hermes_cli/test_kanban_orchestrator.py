"""Per-project orchestrator: pure planner + apply + CLI.

The orchestrator derives a project dossier from a board, decides within an
explicit envelope, and changes only task routing — reopening a blocked lane
that left a forward dossier (gap A) or retiring one that exhausted its attempt
budget. See project-orchestrator-design.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_orchestrator as orch


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _running_task(conn, title="t"):
    tid = kb.create_task(conn, title=title, assignee="worker")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    assert kb.claim_task(conn, tid, claimer="worker") is not None
    return tid


# ---------------------------------------------------------------------------
# Pure planner
# ---------------------------------------------------------------------------


def test_plan_respects_concurrency_cap() -> None:
    snap = orch.BoardSnapshot(tasks=(
        orch.TaskView("r1", "a", "running"),
        orch.TaskView("r2", "b", "running"),
        orch.TaskView("b1", "c", "blocked", dossier={"next": ["x"]}),
        orch.TaskView("b2", "d", "blocked", dossier={"next": ["y"]}),
    ))
    plan = orch.plan_cycle(
        snap, orch.Envelope(max_concurrent_workers=3, max_attempts=3)
    )
    assert len(plan.open_ids) == 1          # headroom = 3 - 2 in flight
    assert len(plan.held_ids) == 1
    assert set(plan.open_ids) | set(plan.held_ids) == {"b1", "b2"}


def test_plan_retires_over_budget() -> None:
    snap = orch.BoardSnapshot(tasks=(
        orch.TaskView("b1", "a", "blocked", block_kind="capability",
                      block_recurrences=3),
        orch.TaskView("b2", "b", "blocked", block_kind="capability",
                      block_recurrences=1, dossier={"next": ["x"]}),
    ))
    plan = orch.plan_cycle(
        snap, orch.Envelope(max_concurrent_workers=5, max_attempts=3)
    )
    assert plan.retire_ids == ("b1",)
    assert plan.open_ids == ("b2",)


def test_plan_prefers_dossier_when_headroom_scarce() -> None:
    snap = orch.BoardSnapshot(tasks=(
        orch.TaskView("nod", "no dossier", "blocked"),
        orch.TaskView("dos", "has dossier", "blocked", dossier={"next": ["x"]}),
    ))
    plan = orch.plan_cycle(
        snap, orch.Envelope(max_concurrent_workers=1, max_attempts=3)
    )
    assert plan.open_ids == ("dos",)
    assert plan.held_ids == ("nod",)


def test_plan_ignores_dependency_blocks() -> None:
    snap = orch.BoardSnapshot(tasks=(
        orch.TaskView("d1", "dep", "blocked", block_kind="dependency",
                      dossier={"next": ["x"]}),
    ))
    plan = orch.plan_cycle(snap, orch.Envelope())
    assert plan.open_ids == () and plan.retire_ids == () and plan.held_ids == ()


def test_plan_opens_nothing_when_full() -> None:
    snap = orch.BoardSnapshot(tasks=(
        orch.TaskView("r1", "a", "running"),
        orch.TaskView("r2", "b", "ready"),
        orch.TaskView("b1", "c", "blocked", dossier={"next": ["x"]}),
    ))
    plan = orch.plan_cycle(
        snap, orch.Envelope(max_concurrent_workers=2, max_attempts=3)
    )
    assert plan.open_ids == ()
    assert plan.held_ids == ("b1",)


# ---------------------------------------------------------------------------
# Snapshot / dossier over a real board
# ---------------------------------------------------------------------------


def test_summarize_board_collects_dossiers(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        done = kb.create_task(conn, title="done", assignee="w")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='done' WHERE id=?", (done,))
        running = _running_task(conn, title="busy")
        blocked = _running_task(conn, title="stuck")
        kb.block_task(conn, blocked, reason="x", kind="capability",
                      handoff={"tried": ["a"], "next": ["b"]})
        summ = orch.summarize_board(conn)
    assert summ["counts"].get("done") == 1
    assert running in summ["frontier"]
    blocked_by_id = {b["id"]: b for b in summ["blocked"]}
    assert blocked in blocked_by_id
    assert blocked_by_id[blocked]["dossier"] == {"tried": ["a"], "next": ["b"]}


# ---------------------------------------------------------------------------
# Apply cycle
# ---------------------------------------------------------------------------


def test_run_cycle_reopens_and_retires(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        t_open = _running_task(conn, title="open me")
        kb.block_task(conn, t_open, reason="x", kind="capability",
                      handoff={"next": ["frontier"]})
        t_ret = _running_task(conn, title="retire me")
        kb.block_task(conn, t_ret, reason="x", kind="capability")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET block_recurrences=3 WHERE id=?", (t_ret,)
            )
        result = orch.run_cycle(
            conn, envelope=orch.Envelope(max_concurrent_workers=5, max_attempts=3)
        )
        assert t_open in result["actions"]["opened"]
        assert t_ret in result["actions"]["retired"]
        assert kb.get_task(conn, t_open).status in ("ready", "todo", "review")
        assert kb.get_task(conn, t_ret).status == "triage"


def test_run_cycle_cap_holds_extra(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        _running_task(conn, title="busy")  # 1 in flight
        b1 = _running_task(conn, title="b1")
        kb.block_task(conn, b1, reason="x", kind="capability",
                      handoff={"next": ["a"]})
        b2 = _running_task(conn, title="b2")
        kb.block_task(conn, b2, reason="x", kind="capability",
                      handoff={"next": ["b"]})
        result = orch.run_cycle(
            conn, envelope=orch.Envelope(max_concurrent_workers=2, max_attempts=3)
        )
    assert len(result["actions"]["opened"]) == 1  # headroom = 2 - 1 in flight
    assert len(result["actions"]["held"]) == 1


def test_run_cycle_leaves_running_tasks_untouched(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        busy = _running_task(conn, title="busy")
        orch.run_cycle(conn, envelope=orch.Envelope())
        assert kb.get_task(conn, busy).status == "running"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_orchestrate_once(kanban_home: Path, capsys: pytest.CaptureFixture) -> None:
    import argparse

    from hermes_cli import kanban as kc

    with kb.connect_closing() as conn:
        tid = _running_task(conn, title="x")
        kb.block_task(conn, tid, reason="x", kind="capability",
                      handoff={"next": ["a"]})
    ns = argparse.Namespace(
        board=None, max_concurrent=5, max_attempts=3, once=True
    )
    assert kc._cmd_orchestrate(ns) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "dossier" in payload and "actions" in payload
    assert tid in payload["actions"]["opened"]
    assert payload["envelope"]["max_concurrent_workers"] == 5


# ---------------------------------------------------------------------------
# Talos dispatch adapter
# ---------------------------------------------------------------------------


def test_plan_to_dispatch_maps_only_open_lanes() -> None:
    snap = orch.BoardSnapshot(tasks=(
        orch.TaskView("b1", "wire exporter", "blocked",
                      dossier={"next": ["read main.py"]}),
        orch.TaskView("b2", "retire me", "blocked"),
        orch.TaskView("b3", "held", "blocked", dossier={"next": ["later"]}),
    ))
    plan = orch.Plan(open_ids=("b1",), retire_ids=("b2",), held_ids=("b3",))
    reqs = orch.plan_to_dispatch(
        snap, plan, project="proj", direction_id="dir_1", version=2,
        authorization="grant_x",
    )
    assert [r.task_id for r in reqs] == ["b1"]
    p = reqs[0].payload()
    assert p["lane"] == "conversation"
    assert set(p) >= {"direction_id", "version", "request_key", "lane", "reason",
                      "scope", "authorization", "work_class", "target", "goal"}
    assert p["direction_id"] == "dir_1" and p["version"] == 2
    assert p["authorization"] == "grant_x"
    assert p["goal"]["model"] == p["target"]["model"]
    assert p["goal"]["max_turns"] in (1, 6, 12)
    assert "read main.py" in p["goal"]["text"]


def test_dispatch_goal_text_from_dossier() -> None:
    text = orch.dossier_goal_text("port script", {
        "failed_because": "no python2", "tried": ["pyenv"], "next": ["use 2to3"],
    })
    assert "no python2" in text and "pyenv" in text and "use 2to3" in text


def test_dispatch_policy_rejects_core_non_frontier() -> None:
    bad = orch.DispatchPolicy(model="claude-code::claude-opus-5-5", work_class="core")
    with pytest.raises(ValueError):
        orch.plan_to_dispatch(
            orch.BoardSnapshot(), orch.Plan(), project="p",
            direction_id=None, version=1, authorization=None, policy=bad,
        )


def test_dispatch_policy_rejects_bad_caps() -> None:
    with pytest.raises(ValueError):
        orch.DispatchPolicy(max_turns=5).validate()
    with pytest.raises(ValueError):
        orch.DispatchPolicy(effort="low").validate()


def test_request_key_is_idempotent() -> None:
    snap = orch.BoardSnapshot(tasks=(
        orch.TaskView("b1", "t", "blocked", dossier={"next": ["x"]}),
    ))
    plan = orch.Plan(open_ids=("b1",))
    k1 = orch.plan_to_dispatch(snap, plan, project="p", direction_id=None,
                               version=1, authorization=None)[0].request_key
    k2 = orch.plan_to_dispatch(snap, plan, project="p", direction_id=None,
                               version=1, authorization=None)[0].request_key
    assert k1 == k2
    snap2 = orch.BoardSnapshot(tasks=(
        orch.TaskView("b1", "t", "blocked", dossier={"next": ["different"]}),
    ))
    k3 = orch.plan_to_dispatch(snap2, plan, project="p", direction_id=None,
                               version=1, authorization=None)[0].request_key
    assert k3 != k1


def test_dispatch_does_not_fabricate_authorization() -> None:
    snap = orch.BoardSnapshot(tasks=(
        orch.TaskView("b1", "t", "blocked", dossier={"next": ["x"]}),
    ))
    reqs = orch.plan_to_dispatch(snap, orch.Plan(open_ids=("b1",)), project="p",
                                 direction_id=None, version=1, authorization=None)
    assert reqs[0].payload()["authorization"] is None


def test_cli_emit_dispatch_preview(kanban_home: Path, capsys: pytest.CaptureFixture) -> None:
    import argparse

    from hermes_cli import kanban as kc

    with kb.connect_closing() as conn:
        tid = _running_task(conn, title="wire exporter")
        kb.block_task(conn, tid, reason="x", kind="capability",
                      handoff={"next": ["read worker/main.py"]})
    ns = argparse.Namespace(
        board=None, max_concurrent=5, max_attempts=3, once=True,
        emit_dispatch=True, direction="dir_1", version=1, authorization="grant_x",
        model=None, effort=None, max_turns=None, work_class=None,
    )
    assert kc._cmd_orchestrate(ns) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["sent"] is False
    assert len(payload["dispatch"]) == 1
    assert payload["dispatch"][0]["lane"] == "conversation"
    assert "read worker/main.py" in payload["dispatch"][0]["goal"]["text"]
    with kb.connect_closing() as conn:
        assert kb.get_task(conn, tid).status == "blocked"  # preview changed no routing
