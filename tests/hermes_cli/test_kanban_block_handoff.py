"""Gap A: a blocking worker leaves a forward dossier for its successor.

``block_task`` accepts an optional structured ``handoff`` and persists it on
the closing run's ``metadata`` — the same channel ``complete_task`` already
uses for structured handoff facts. The point is convergence: the next worker
(or a sibling lane, or the lead) reads what was tried, why it failed, and what
to try next, and resumes at the frontier instead of re-deriving the dead end.

See docs: orchestration-integration-into-hermes.md (gap A).
"""

from __future__ import annotations

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


def _running_task(conn, title="t"):
    tid = kb.create_task(conn, title=title, assignee="worker")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    assert kb.claim_task(conn, tid, claimer="worker") is not None
    return tid


DOSSIER = {
    "tried": ["reverse-engineer the flag", "patch the data loader"],
    "failed_because": "the scorer rejects an out-of-band flag",
    "next": ["read the grader source", "try the template path"],
    "artifacts": ["ds://notes/abc"],
}


def test_block_handoff_persists_on_closing_run(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        assert kb.block_task(
            conn, tid, reason="stuck", kind="capability", handoff=DOSSIER
        )
        closed = [r for r in kb.list_runs(conn, tid) if r.outcome == "blocked"]
        assert closed, "expected a closed blocked run"
        assert closed[-1].metadata == DOSSIER
        blocked_ev = [e for e in kb.list_events(conn, tid) if e.kind == "blocked"][-1]
        assert (blocked_ev.payload or {}).get("handoff_keys") == sorted(DOSSIER)


def test_block_without_handoff_leaves_metadata_none(kanban_home: Path) -> None:
    # Backward compatibility: the existing single-reason block is unchanged.
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        assert kb.block_task(conn, tid, reason="stuck", kind="capability")
        closed = [r for r in kb.list_runs(conn, tid) if r.outcome == "blocked"]
        assert closed and closed[-1].metadata is None
        blocked_ev = [e for e in kb.list_events(conn, tid) if e.kind == "blocked"][-1]
        assert (blocked_ev.payload or {}).get("handoff_keys") is None


def test_block_handoff_on_never_claimed_task_is_synthesized(kanban_home: Path) -> None:
    # A ready task blocked with only a dossier (no reason, never claimed) still
    # preserves the dossier via a synthesized closing run.
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="t", assignee="worker")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
        assert kb.block_task(conn, tid, kind="capability", handoff=DOSSIER)
        closed = [r for r in kb.list_runs(conn, tid) if r.outcome == "blocked"]
        assert closed and closed[-1].metadata == DOSSIER


def test_dependency_block_carries_handoff(kanban_home: Path) -> None:
    # The dependency-routing branch also preserves the dossier.
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        assert kb.block_task(conn, tid, reason="waiting", kind="dependency", handoff=DOSSIER)
        closed = [r for r in kb.list_runs(conn, tid) if r.outcome == "blocked"]
        assert closed and closed[-1].metadata == DOSSIER


def test_block_handoff_must_be_a_dict(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        with pytest.raises(ValueError):
            kb.block_task(conn, tid, reason="x", handoff="not a dict")
