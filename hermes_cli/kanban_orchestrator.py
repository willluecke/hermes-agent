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

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Optional

from hermes_cli import kanban_db as kb

# Tasks that occupy a worker slot right now; these count against the envelope.
IN_FLIGHT_STATUSES = ("running", "ready")
# 'dependency' blocks auto-resume via parent gating — not the orchestrator's job.
AUTO_RESUMING_BLOCK_KIND = "dependency"
# Blocks that wait on a human decision. The orchestrator neither reopens nor
# retires these: a discovery candidate parked ``needs_input`` stays parked until
# someone runs ``hermes kanban accept`` (or ``reject``).
HUMAN_GATED_BLOCK_KINDS = frozenset({"needs_input"})
# Statuses that mean the board already has work for the dispatcher or a
# specifier/decomposer; discovery is not recommended while any exist.
OPEN_STATUSES = frozenset({"triage", "todo", "scheduled", "ready", "running", "review"})
DISCOVERY_CREATOR = "discovery-sweep"


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
    created_by: Optional[str] = None

    @property
    def human_gated(self) -> bool:
        return self.status == "blocked" and self.block_kind in HUMAN_GATED_BLOCK_KINDS

    @property
    def is_candidate(self) -> bool:
        return self.human_gated and self.created_by == DISCOVERY_CREATOR


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
        if t.block_kind in HUMAN_GATED_BLOCK_KINDS:
            continue  # waits on a person, not on headroom
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
                created_by=t.created_by,
            )
        )
    return BoardSnapshot(tasks=tuple(views))


def discovery_recommended(snapshot: BoardSnapshot) -> tuple[bool, str]:
    """Pure: should the orchestrator read the project for new work?

    Yes only when the board has nothing to route: no open work, no blocked
    lane the orchestrator could reopen, and no discovery candidates already
    waiting on a decision (re-sweeping while a list sits unanswered only
    piles up more of the same).
    """
    open_work = [t for t in snapshot.tasks if t.status in OPEN_STATUSES]
    if open_work:
        return False, f"board has {len(open_work)} open task(s)"
    reopenable = [
        t for t in snapshot.tasks
        if t.status == "blocked"
        and t.block_kind != AUTO_RESUMING_BLOCK_KIND
        and t.block_kind not in HUMAN_GATED_BLOCK_KINDS
    ]
    if reopenable:
        return False, f"{len(reopenable)} blocked lane(s) can be reopened"
    pending = [t for t in snapshot.tasks if t.is_candidate]
    if pending:
        return False, f"{len(pending)} discovery candidate(s) await acceptance"
    return True, "no open work on the board"


def summarize_board(conn, board: Optional[str] = None) -> dict[str, Any]:
    """The project dossier: counts by status, the frontier, block dossiers."""
    snap = snapshot_board(conn, board)
    counts: dict[str, int] = {}
    frontier: list[str] = []
    blocked: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for t in snap.tasks:
        counts[t.status] = counts.get(t.status, 0) + 1
        if t.status in IN_FLIGHT_STATUSES:
            frontier.append(t.id)
        if t.is_candidate:
            candidates.append({"id": t.id, "title": t.title})
        elif t.status == "blocked":
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
        "candidates": candidates,
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
    recommended, why = discovery_recommended(snapshot_board(conn, board))
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
        # The read-first step: when nothing is left to route, the CLI runs a
        # read-only discovery sweep (``hermes_cli.kanban_discovery``) unless
        # told not to, and parks the candidates for a human to accept.
        "discovery": {"recommended": recommended, "reason": why},
    }


# ---------------------------------------------------------------------------
# Talos bridge: turn the decision (a Plan + dossiers) into Talos dispatch
# payloads. Talos owns the model policy, envelope, execution, and governance;
# this is only the seam that carries the kanban decision signal into a
# conversation-lane dispatch. It is pure — no network, no DB writes, and it
# never fabricates authorization. The live POST to the Talos operator ledger
# is deliberately out of scope here and gated to a human.
# ---------------------------------------------------------------------------

# Mirrors talos/steward.py FRONTIERS (Talos is authoritative). Used only to
# enforce "core work requires a frontier pin" before a payload is emitted.
TALOS_FRONTIER_MODELS = frozenset({
    "openai-codex::gpt-6-astra",
    "claude-code::claude-fable-5-1",
})
TALOS_MANAGED_TURN_CAPS = frozenset({1, 6, 12})
TALOS_EFFORTS = frozenset({"high", "xhigh", "max"})
TALOS_WORK_CLASSES = frozenset({"core", "peripheral"})


@dataclass(frozen=True)
class DispatchPolicy:
    """Per-dispatch runtime pin, mirroring Talos's model policy.

    ``model`` is a namespaced ``harness::model`` string. The default is a
    frontier pin so core resume work is valid. Talos remains the authority on
    the full model catalog; this only pre-validates the constraints Talos will
    enforce, so a bad preview fails here instead of at the live call.
    """

    model: str = "claude-code::claude-fable-5-1"
    effort: str = "high"
    max_turns: int = 6
    work_class: str = "core"

    def validate(self) -> None:
        if self.work_class not in TALOS_WORK_CLASSES:
            raise ValueError(
                f"work_class must be one of {sorted(TALOS_WORK_CLASSES)}"
            )
        if self.effort not in TALOS_EFFORTS:
            raise ValueError(f"effort must be one of {sorted(TALOS_EFFORTS)}")
        if self.max_turns not in TALOS_MANAGED_TURN_CAPS:
            raise ValueError(
                f"max_turns must be one of {sorted(TALOS_MANAGED_TURN_CAPS)} "
                "(Talos managed caps)"
            )
        if not self.model or "::" not in self.model:
            raise ValueError("model must be a namespaced 'harness::model' pin")
        if self.work_class == "core" and self.model not in TALOS_FRONTIER_MODELS:
            raise ValueError(
                "core work requires a frontier-pinned model "
                f"(one of {sorted(TALOS_FRONTIER_MODELS)})"
            )


@dataclass(frozen=True)
class DispatchRequest:
    """A single Talos conversation-lane dispatch, plus kanban provenance."""

    direction_id: Optional[str]
    version: int
    request_key: str
    lane: str
    reason: str
    scope: Any
    authorization: Any
    work_class: str
    target: dict
    goal: dict
    task_id: str  # the kanban lane this resumes; provenance, not sent to Talos

    def payload(self) -> dict[str, Any]:
        """The exact body Talos's /api/delegations/dispatch expects."""
        return {
            "direction_id": self.direction_id,
            "version": self.version,
            "request_key": self.request_key,
            "lane": self.lane,
            "reason": self.reason,
            "scope": self.scope,
            "authorization": self.authorization,
            "work_class": self.work_class,
            "target": self.target,
            "goal": self.goal,
        }


def dossier_goal_text(title: str, dossier: Optional[dict]) -> str:
    """Assemble a resume brief from a gap-A block dossier.

    The point of the dossier is that the successor starts at the frontier: it
    is told what already failed (don't repeat it) and what to try next, rather
    than re-deriving the dead end.
    """
    def _join(value: Any) -> str:
        if isinstance(value, (list, tuple)):
            return "; ".join(str(v) for v in value)
        return str(value)

    lines = [
        f"Resume this task: {title}.",
        "A previous worker blocked this lane and left a dossier. Continue from "
        "the frontier; do not repeat what already failed.",
    ]
    d = dossier or {}
    if d.get("failed_because"):
        lines.append(f"What blocked it: {_join(d['failed_because'])}")
    if d.get("tried"):
        lines.append(f"Already tried (do not repeat): {_join(d['tried'])}")
    if d.get("next"):
        lines.append(f"Try next: {_join(d['next'])}")
    if d.get("artifacts"):
        lines.append(f"Artifacts: {_join(d['artifacts'])}")
    if not dossier:
        lines.append("No dossier was left; re-establish context before acting.")
    return "\n".join(lines)


def _request_key(project: str, task_id: str, dossier: Optional[dict]) -> str:
    """Deterministic idempotency key for a resume dispatch."""
    canonical = json.dumps(
        {"project": project, "task": task_id, "dossier": dossier},
        sort_keys=True, ensure_ascii=False, default=str,
    )
    return "orq_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


def plan_to_dispatch(
    snapshot: BoardSnapshot,
    plan: Plan,
    *,
    project: str,
    direction_id: Optional[str],
    version: int,
    authorization: Any,
    policy: Optional[DispatchPolicy] = None,
    scope: Any = None,
    reason: Optional[str] = None,
) -> list[DispatchRequest]:
    """Map a plan's opened lanes to Talos conversation-lane dispatch requests.

    One request per opened lane; retired and held lanes produce nothing. The
    model/effort come from ``policy`` (Talos's pins); the goal text is built
    from each lane's gap-A dossier. ``authorization`` is passed through and
    never fabricated. Pure: no network, no DB writes.
    """
    policy = policy or DispatchPolicy()
    policy.validate()
    by_id = {t.id: t for t in snapshot.tasks}
    requests: list[DispatchRequest] = []
    for tid in plan.open_ids:
        tv = by_id.get(tid)
        if tv is None:
            continue
        goal = {
            "text": dossier_goal_text(tv.title, tv.dossier),
            "max_turns": policy.max_turns,
            "model": policy.model,
            "effort": policy.effort,
        }
        target = {
            "project": project,
            "model": policy.model,
            "effort": policy.effort,
        }
        requests.append(
            DispatchRequest(
                direction_id=direction_id,
                version=version,
                request_key=_request_key(project, tid, tv.dossier),
                lane="conversation",
                reason=reason or "orchestrator: resume blocked lane at its frontier",
                scope=scope,
                authorization=authorization,
                work_class=policy.work_class,
                target=target,
                goal=goal,
                task_id=tid,
            )
        )
    return requests


def preview_dispatches(
    conn,
    board: Optional[str] = None,
    *,
    envelope: Optional[Envelope] = None,
    policy: Optional[DispatchPolicy] = None,
    direction_id: Optional[str] = None,
    version: int = 1,
    authorization: Any = None,
    scope: Any = None,
) -> dict[str, Any]:
    """Dry-run: the dispatch payloads this cycle's plan would emit to Talos.

    Applies no routing and performs no live write. ``sent`` is always False —
    the live POST to the Talos operator ledger is gated and out of scope here.
    """
    envelope = envelope or Envelope()
    snap = snapshot_board(conn, board)
    plan = plan_cycle(snap, envelope)
    project = board or kb.get_current_board()
    requests = plan_to_dispatch(
        snap, plan, project=project, direction_id=direction_id,
        version=version, authorization=authorization, policy=policy, scope=scope,
    )
    return {
        "project": project,
        "plan": plan.as_dict(),
        "dispatch": [r.payload() for r in requests],
        "dossier": summarize_board(conn, board),
        "sent": False,
    }
