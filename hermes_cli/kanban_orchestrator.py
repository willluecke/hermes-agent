"""Per-project orchestrator — one coordinator per board.

A project is a Kanban board. This module is the standing "lead" layer from the
design sketch: it derives a project dossier from the board, decides within an
explicit envelope, and changes only task *routing*. It reopens a blocked lane
that left a forward dossier (the gap-A handoff) so a successor resumes at the
frontier, or retires a lane that has exhausted its attempt budget to triage
(safe exit). It never executes task work — the dispatcher still spawns ready
tasks. Decision (:func:`plan_cycle`) is pure and separate from apply
(:func:`run_cycle`), so the policy is unit-testable without a database.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from hermes_cli import kanban_db as kb

# Tasks that occupy a worker slot right now; these count against the envelope.
IN_FLIGHT_STATUSES = ("running", "ready")
# 'dependency' blocks auto-resume via parent gating — not the orchestrator's job.
AUTO_RESUMING_BLOCK_KIND = "dependency"


@dataclass(frozen=True)
class Envelope:
    """The per-project resource envelope. Unbounded authority, bounded here."""

    max_concurrent_workers: int = 4
    # block_recurrences at or over this retires the lane to triage (safe exit).
    max_attempts: int = 3


@dataclass(frozen=True)
class TaskView:
    """The slice of a task the orchestrator reasons about."""

    id: str
    title: str
    status: str
    block_kind: Optional[str] = None
    block_recurrences: int = 0
    dossier: Optional[dict] = None  # gap-A forward handoff from the latest run


@dataclass(frozen=True)
class BoardSnapshot:
    tasks: tuple[TaskView, ...] = ()

    @property
    def in_flight(self) -> int:
        return sum(1 for t in self.tasks if t.status in IN_FLIGHT_STATUSES)


@dataclass(frozen=True)
class Plan:
    open_ids: tuple[str, ...] = ()     # reopen (within the concurrency headroom)
    retire_ids: tuple[str, ...] = ()   # over attempt budget -> triage
    held_ids: tuple[str, ...] = ()     # reopenable but over the cap this cycle

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "open": list(self.open_ids),
            "retire": list(self.retire_ids),
            "held": list(self.held_ids),
        }


def plan_cycle(snapshot: BoardSnapshot, envelope: Envelope) -> Plan:
    """Pure decision step: which blocked lanes to reopen, retire, or hold.

    Never opens more than the envelope allows. A truly-blocked lane that has
    hit ``max_attempts`` is retired (safe exit) instead of reopened. Scarce
    headroom is spent first on lanes that left a forward dossier, because those
    resume at a known frontier rather than from the start line.
    """
    retire: list[str] = []
    candidates: list[TaskView] = []
    for t in snapshot.tasks:
        if t.status != "blocked":
            continue
        if t.block_kind == AUTO_RESUMING_BLOCK_KIND:
            continue
        if t.block_recurrences >= envelope.max_attempts:
            retire.append(t.id)
        else:
            candidates.append(t)
    candidates.sort(key=lambda t: (t.dossier is None, t.block_recurrences))
    headroom = max(0, envelope.max_concurrent_workers - snapshot.in_flight)
    open_ids = tuple(t.id for t in candidates[:headroom])
    held_ids = tuple(t.id for t in candidates[headroom:])
    return Plan(open_ids=open_ids, retire_ids=tuple(retire), held_ids=held_ids)


def snapshot_board(conn, board: Optional[str] = None) -> BoardSnapshot:
    """Build a snapshot from the (already board-scoped) connection."""
    views: list[TaskView] = []
    for t in kb.list_tasks(conn):
        dossier = None
        if t.status == "blocked":
            run = kb.latest_run(conn, t.id)
            if run is not None and isinstance(run.metadata, dict):
                dossier = run.metadata
        views.append(
            TaskView(
                id=t.id,
                title=t.title,
                status=t.status,
                block_kind=t.block_kind,
                block_recurrences=t.block_recurrences,
                dossier=dossier,
            )
        )
    return BoardSnapshot(tasks=tuple(views))


def summarize_board(conn, board: Optional[str] = None) -> dict[str, Any]:
    """The project dossier: counts by status, the frontier, block dossiers."""
    snap = snapshot_board(conn, board)
    counts: dict[str, int] = {}
    frontier: list[str] = []
    blocked: list[dict[str, Any]] = []
    for t in snap.tasks:
        counts[t.status] = counts.get(t.status, 0) + 1
        if t.status in IN_FLIGHT_STATUSES:
            frontier.append(t.id)
        if t.status == "blocked":
            blocked.append(
                {
                    "id": t.id,
                    "title": t.title,
                    "kind": t.block_kind,
                    "attempts": t.block_recurrences,
                    "dossier": t.dossier,
                }
            )
    return {
        "board": board or kb.get_current_board(),
        "counts": counts,
        "in_flight": snap.in_flight,
        "frontier": frontier,
        "blocked": blocked,
    }


def run_cycle(
    conn, board: Optional[str] = None, envelope: Optional[Envelope] = None
) -> dict[str, Any]:
    """Apply one orchestration cycle. Changes only task routing.

    Reopens the planned lanes (``unblock_task``) and retires the over-budget
    ones (``retire_to_triage``), then returns the post-cycle project dossier
    alongside the actions taken and the envelope in force. It issues no work
    itself; the dispatcher claims and spawns whatever this reopens.
    """
    envelope = envelope or Envelope()
    snap = snapshot_board(conn, board)
    plan = plan_cycle(snap, envelope)
    opened: list[str] = []
    for tid in plan.open_ids:
        if kb.unblock_task(conn, tid):
            opened.append(tid)
    retired: list[str] = []
    for tid in plan.retire_ids:
        if kb.retire_to_triage(
            conn, tid, note="orchestrator: attempt budget exhausted"
        ):
            retired.append(tid)
    return {
        "dossier": summarize_board(conn, board),
        "actions": {
            "opened": opened,
            "retired": retired,
            "held": list(plan.held_ids),
        },
        "envelope": {
            "max_concurrent_workers": envelope.max_concurrent_workers,
            "max_attempts": envelope.max_attempts,
        },
    }
