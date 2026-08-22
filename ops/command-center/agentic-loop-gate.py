#!/usr/bin/env python3
"""Deterministic wake gate for the governed Hermes agentic loop.

The script is intended to run as a Hermes cron pre-check. Its final stdout
line is JSON containing ``wakeAgent``. Hermes exits before constructing an
agent when that value is false, so ordinary polling consumes no model quota.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sqlite3
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator
from zoneinfo import ZoneInfo


STATE_VERSION = 1
OUTPUT_VERSION = "hermes.agentic-loop-gate.v1"
TERMINAL_JOB_STATUSES = {"completed", "failed"}
AUTOMATION_OWNERS = {"agent", "command center", "hermes"}
MAX_EVENTS_PER_BATCH = 24
MAX_REMEMBERED_INBOX_IDS = 500


@dataclass(frozen=True)
class GateConfig:
    database: Path
    state: Path
    inbox: Path
    timezone: str = "America/Los_Angeles"
    daily_wake_budget: int = 3
    retry_seconds: int = 6 * 60 * 60
    queued_stale_seconds: int = 30 * 60
    running_stale_seconds: int = 2 * 60 * 60


def _default_config() -> GateConfig:
    home = Path.home()
    root = home / ".hermes" / "agentic-loop"
    return GateConfig(
        database=home / ".hermes-chat-sync" / "sync.db",
        state=root / "gate-state.json",
        inbox=root / "inbox.jsonl",
        timezone="America/Los_Angeles",
        daily_wake_budget=3,
        retry_seconds=6 * 60 * 60,
    )


def _clean_text(value: Any, limit: int = 240) -> str:
    text = " ".join(str(value or "").split())
    return "".join(ch for ch in text if ch.isprintable())[:limit]


def _fingerprint(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _event(kind: str, event_key: str, **fields: Any) -> dict[str, Any]:
    return {
        "eventKey": f"{kind}:{event_key}",
        "kind": kind,
        **{name: value for name, value in fields.items() if value not in (None, "")},
    }


def _initial_state() -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "jobFingerprints": {},
        "workFingerprints": {},
        "metricFingerprints": {},
        "alertFingerprints": {},
        "inboxSeen": [],
        "inboxOffset": 0,
        "pending": None,
        "backlog": [],
        "budget": {"date": "", "count": 0},
        "lastAck": None,
    }


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _initial_state()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read gate state: {exc}") from exc
    if not isinstance(data, dict) or data.get("version") != STATE_VERSION:
        raise RuntimeError("unsupported agentic-loop gate state")
    merged = _initial_state()
    merged.update(data)
    return merged


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    fd, temporary = tempfile.mkstemp(prefix=".gate-state-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(state, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


@contextmanager
def _state_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    with lock_path.open("a+", encoding="utf-8") as handle:
        os.chmod(lock_path, 0o600)
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield


def _parse_data(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(str(raw or "{}"))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _connect_readonly(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise RuntimeError(f"management database not found: {path}")
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5.0)
    connection.row_factory = sqlite3.Row
    return connection


def _collect_job_events(
    connection: sqlite3.Connection,
    state: dict[str, Any],
    now_ms: int,
    config: GateConfig,
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, str]]:
    rows = connection.execute(
        "SELECT id, status, provider, mode, project, created_at, updated_at, "
        "worker_id, data "
        "FROM agent_jobs ORDER BY updated_at ASC"
    ).fetchall()
    previous_jobs = state.get("jobFingerprints", {})
    job_fingerprints: dict[str, str] = {}
    events: list[dict[str, Any]] = []
    alerts: dict[str, str] = {}

    for row in rows:
        updated_at = int(row["updated_at"] or 0)
        data = _parse_data(row["data"])
        status = str(row["status"] or data.get("status") or "")
        context_type = str(data.get("contextType") or "")

        if status in TERMINAL_JOB_STATUSES and context_type == "hermes-orchestration":
            summary = _clean_text(data.get("summary") or data.get("error"), 320)
            fingerprint = _fingerprint({
                "status": status,
                "updatedAt": updated_at,
                "summary": summary,
                "commit": data.get("commit"),
            })
            job_id = str(row["id"])
            job_fingerprints[job_id] = fingerprint
            if previous_jobs.get(job_id) != fingerprint:
                events.append(
                    _event(
                        "orchestration_result",
                        f"{job_id}:{fingerprint}",
                        jobId=job_id,
                        status=status,
                        project=_clean_text(row["project"], 80),
                        mode=_clean_text(row["mode"], 40),
                        title=_clean_text(data.get("title"), 160),
                        commit=_clean_text(data.get("commit"), 64),
                        decisionNeeded="DECISION_NEEDED" in summary.upper(),
                        summary=summary,
                    )
                )

        age_seconds = max(0, (now_ms - updated_at) // 1000)
        stale_after = None
        if status == "queued":
            stale_after = config.queued_stale_seconds
        elif status == "running":
            stale_after = config.running_stale_seconds
        if stale_after is not None and age_seconds >= stale_after:
            alert_key = f"stalled_job:{row['id']}:{status}"
            alert_value = _fingerprint([status, updated_at, row["worker_id"]])
            alerts[alert_key] = alert_value
            if state.get("alertFingerprints", {}).get(alert_key) != alert_value:
                events.append(
                    _event(
                        "stalled_job",
                        f"{row['id']}:{status}:{updated_at}",
                        jobId=str(row["id"]),
                        status=status,
                        project=_clean_text(row["project"], 80),
                        workerId=_clean_text(row["worker_id"], 100),
                        ageSeconds=age_seconds,
                    )
                )

    return events, job_fingerprints, alerts


def _collect_entity_events(
    connection: sqlite3.Connection,
    state: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, str]]:
    rows = connection.execute(
        "SELECT kind, id, data FROM management_entities "
        "WHERE kind IN ('work-item', 'metric') ORDER BY kind, id"
    ).fetchall()
    previous_work = state.get("workFingerprints", {})
    previous_metrics = state.get("metricFingerprints", {})
    work_fingerprints: dict[str, str] = {}
    metric_fingerprints: dict[str, str] = {}
    events: list[dict[str, Any]] = []

    for row in rows:
        data = _parse_data(row["data"])
        kind = str(row["kind"])
        entity_id = str(row["id"])
        if kind == "work-item":
            semantic = {
                key: data.get(key)
                for key in (
                    "title",
                    "company",
                    "workstream",
                    "owner",
                    "due",
                    "status",
                    "priority",
                    "nextAction",
                    "consequence",
                )
            }
            fingerprint = _fingerprint(semantic)
            work_fingerprints[entity_id] = fingerprint
            changed = previous_work.get(entity_id) != fingerprint
            owner = str(data.get("owner") or "").strip().lower()
            status = str(data.get("status") or "")
            priority = str(data.get("priority") or "")
            agent_owned = owner in AUTOMATION_OWNERS
            actionable = status == "Review" or (
                agent_owned
                and status in {"Ready", "Doing", "Blocked"}
                and priority in {"Critical", "High"}
            )
            if changed and actionable:
                events.append(
                    _event(
                        "work_item",
                        f"{entity_id}:{fingerprint}",
                        itemId=entity_id,
                        title=_clean_text(data.get("title"), 180),
                        company=_clean_text(data.get("company"), 60),
                        workstream=_clean_text(data.get("workstream"), 60),
                        owner=_clean_text(data.get("owner"), 80),
                        status=status,
                        priority=priority,
                        nextAction=_clean_text(data.get("nextAction"), 240),
                    )
                )
        else:
            semantic = {
                key: data.get(key)
                for key in ("key", "label", "company", "value", "detail", "tone")
            }
            fingerprint = _fingerprint(semantic)
            metric_fingerprints[entity_id] = fingerprint
            changed = previous_metrics.get(entity_id) != fingerprint
            if changed and str(data.get("tone") or "") == "warning":
                events.append(
                    _event(
                        "warning_metric",
                        f"{entity_id}:{fingerprint}",
                        metricId=entity_id,
                        key=_clean_text(data.get("key"), 80),
                        label=_clean_text(data.get("label"), 120),
                        company=_clean_text(data.get("company"), 60),
                        value=_clean_text(data.get("value"), 100),
                        detail=_clean_text(data.get("detail"), 240),
                    )
                )

    return events, work_fingerprints, metric_fingerprints


def _read_inbox(
    path: Path, seen_ids: list[str], offset: int
) -> tuple[list[dict[str, Any]], list[str], int]:
    if not path.exists():
        return [], seen_ids, offset
    size = path.stat().st_size
    if offset > size:
        # The managed inbox is append-only. If an operator truncates it, treat
        # existing replacement content as a new baseline instead of replaying
        # records whose disposition is already durable in gate state.
        return [], seen_ids, size
    events: list[dict[str, Any]] = []
    next_seen = list(seen_ids)
    seen_lookup = set(seen_ids)
    next_offset = offset
    with path.open("rb") as handle:
        handle.seek(offset)
        for raw_line in handle:
            if not raw_line.endswith(b"\n"):
                break
            next_offset += len(raw_line)
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                item = {
                    "id": f"invalid-record-{_fingerprint(line)}",
                    "task": "The agentic-loop inbox contains an invalid JSON record.",
                    "priority": "High",
                }
            if not isinstance(item, dict):
                continue
            item_id = _clean_text(item.get("id"), 100)
            if not item_id or item_id in seen_lookup:
                continue
            next_seen.append(item_id)
            seen_lookup.add(item_id)
            events.append(
                _event(
                    "inbox",
                    item_id,
                    inboxId=item_id,
                    project=_clean_text(item.get("project"), 100),
                    priority=_clean_text(item.get("priority") or "Normal", 20),
                    task=_clean_text(item.get("task"), 320),
                )
            )
    return events, next_seen[-MAX_REMEMBERED_INBOX_IDS:], next_offset


def _merge_events(
    existing: list[dict[str, Any]], incoming: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    merged = {str(item.get("eventKey")): item for item in existing}
    for item in incoming:
        merged[str(item.get("eventKey"))] = item
    return list(merged.values())


def _new_pending(events: list[dict[str, Any]], now_ms: int) -> dict[str, Any]:
    return {
        "batchId": _batch_id(events),
        "events": events,
        "createdAt": now_ms,
        "lastWakeAt": 0,
        "attempts": 0,
    }


def _batch_id(events: list[dict[str, Any]]) -> str:
    keys = sorted(str(item.get("eventKey")) for item in events)
    return "loop_" + _fingerprint(keys)[:16]


def _local_date(now_ms: int, timezone: str) -> str:
    return datetime.fromtimestamp(now_ms / 1000, ZoneInfo(timezone)).date().isoformat()


def evaluate_gate(
    config: GateConfig,
    *,
    now_ms: int | None = None,
    prime: bool = False,
) -> dict[str, Any]:
    now_ms = int(now_ms if now_ms is not None else time.time() * 1000)
    with _state_lock(config.state):
        state = _load_state(config.state)
        with _connect_readonly(config.database) as connection:
            job_events, job_fingerprints, job_alerts = _collect_job_events(
                connection, state, now_ms, config
            )
            entity_events, work_fingerprints, metric_fingerprints = (
                _collect_entity_events(connection, state)
            )
        inbox_events, inbox_seen, inbox_offset = _read_inbox(
            config.inbox,
            list(state.get("inboxSeen", [])),
            int(state.get("inboxOffset") or 0),
        )

        state["jobFingerprints"] = job_fingerprints
        state["workFingerprints"] = work_fingerprints
        state["metricFingerprints"] = metric_fingerprints
        state["alertFingerprints"] = job_alerts
        state["inboxSeen"] = inbox_seen
        state["inboxOffset"] = inbox_offset

        local_date = _local_date(now_ms, config.timezone)
        budget = state.get("budget") or {}
        if budget.get("date") != local_date:
            budget = {"date": local_date, "count": 0}
        state["budget"] = budget

        if prime:
            state["pending"] = None
            state["backlog"] = []
            _write_state(config.state, state)
            return {
                "version": OUTPUT_VERSION,
                "wakeAgent": False,
                "reason": "baseline_primed",
                "dailyWakeBudget": config.daily_wake_budget,
            }

        incoming = job_events + entity_events + inbox_events
        pending = state.get("pending")
        backlog = list(state.get("backlog") or [])
        if pending and incoming:
            previous_events = list(pending.get("events") or [])
            merged = _merge_events(previous_events + backlog, incoming)
            current_events = merged[:MAX_EVENTS_PER_BATCH]
            backlog = merged[MAX_EVENTS_PER_BATCH:]
            if current_events != previous_events:
                pending = _new_pending(current_events, now_ms)
        elif not pending and (incoming or backlog):
            merged = _merge_events(backlog, incoming)
            pending = _new_pending(merged[:MAX_EVENTS_PER_BATCH], now_ms)
            backlog = merged[MAX_EVENTS_PER_BATCH:]
        state["pending"] = pending
        state["backlog"] = backlog

        if not pending:
            _write_state(config.state, state)
            return {
                "version": OUTPUT_VERSION,
                "wakeAgent": False,
                "reason": "no_actionable_change",
                "budgetRemaining": max(
                    0, config.daily_wake_budget - int(budget.get("count") or 0)
                ),
            }

        used = int(budget.get("count") or 0)
        if used >= config.daily_wake_budget:
            _write_state(config.state, state)
            return {
                "version": OUTPUT_VERSION,
                "wakeAgent": False,
                "reason": "daily_wake_budget_exhausted",
                "pendingBatchId": pending["batchId"],
                "pendingEventCount": len(pending["events"]),
                "backlogEventCount": len(backlog),
                "budgetRemaining": 0,
            }

        last_wake = int(pending.get("lastWakeAt") or 0)
        if last_wake and now_ms - last_wake < config.retry_seconds * 1000:
            _write_state(config.state, state)
            return {
                "version": OUTPUT_VERSION,
                "wakeAgent": False,
                "reason": "awaiting_batch_acknowledgement",
                "pendingBatchId": pending["batchId"],
                "retryAfterSeconds": max(
                    0, config.retry_seconds - ((now_ms - last_wake) // 1000)
                ),
                "budgetRemaining": config.daily_wake_budget - used,
            }

        pending["lastWakeAt"] = now_ms
        pending["attempts"] = int(pending.get("attempts") or 0) + 1
        budget["count"] = used + 1
        state["pending"] = pending
        state["budget"] = budget
        _write_state(config.state, state)
        return {
            "version": OUTPUT_VERSION,
            "wakeAgent": True,
            "reason": "actionable_state_change",
            "batchId": pending["batchId"],
            "attempt": pending["attempts"],
            "events": pending["events"],
            "backlogEventCount": len(backlog),
            "ackCommand": (
                "/home/will/.hermes/scripts/agentic-loop-gate.py ack "
                f"--batch-id {pending['batchId']}"
            ),
            "budgetRemaining": config.daily_wake_budget - int(budget["count"]),
        }


def acknowledge(
    config: GateConfig, batch_id: str, *, now_ms: int | None = None
) -> dict[str, Any]:
    now_ms = int(now_ms if now_ms is not None else time.time() * 1000)
    with _state_lock(config.state):
        state = _load_state(config.state)
        pending = state.get("pending")
        if not pending or pending.get("batchId") != batch_id:
            return {
                "acknowledged": False,
                "reason": "batch_not_pending",
                "pendingBatchId": pending.get("batchId") if pending else None,
            }
        state["lastAck"] = {
            "batchId": batch_id,
            "acknowledgedAt": now_ms,
            "eventCount": len(pending.get("events") or []),
        }
        backlog = list(state.get("backlog") or [])
        if backlog:
            next_events = backlog[:MAX_EVENTS_PER_BATCH]
            state["pending"] = _new_pending(next_events, now_ms)
            state["backlog"] = backlog[MAX_EVENTS_PER_BATCH:]
        else:
            state["pending"] = None
            state["backlog"] = []
        _write_state(config.state, state)
        return {
            "acknowledged": True,
            "batchId": batch_id,
            "nextBatchId": (
                state["pending"].get("batchId") if state.get("pending") else None
            ),
        }


def enqueue(
    config: GateConfig,
    *,
    item_id: str,
    task: str,
    project: str = "",
    priority: str = "Normal",
) -> dict[str, Any]:
    item_id = _clean_text(item_id, 100)
    task = _clean_text(task, 1000)
    if not item_id or not task:
        raise ValueError("enqueue requires non-empty --id and --task")
    record = {
        "id": item_id,
        "task": task,
        "project": _clean_text(project, 100),
        "priority": _clean_text(priority, 20),
    }
    with _state_lock(config.state):
        config.inbox.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with config.inbox.open("a", encoding="utf-8") as handle:
            os.chmod(config.inbox, 0o600)
            handle.write(
                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
    return {"enqueued": True, "id": item_id}


def gate_status(config: GateConfig) -> dict[str, Any]:
    with _state_lock(config.state):
        state = _load_state(config.state)
    pending = state.get("pending")
    return {
        "version": OUTPUT_VERSION,
        "pendingBatchId": pending.get("batchId") if pending else None,
        "pendingEventCount": len(pending.get("events") or []) if pending else 0,
        "pendingAttempts": int(pending.get("attempts") or 0) if pending else 0,
        "backlogEventCount": len(state.get("backlog") or []),
        "budget": state.get("budget"),
        "lastAck": state.get("lastAck"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        nargs="?",
        choices=("check", "prime", "status", "enqueue", "ack"),
        default="check",
    )
    parser.add_argument("--batch-id")
    parser.add_argument("--id")
    parser.add_argument("--task")
    parser.add_argument("--project", default="")
    parser.add_argument("--priority", default="Normal")
    parser.add_argument("--db", type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--inbox", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    defaults = _default_config()
    config = GateConfig(
        database=args.db or defaults.database,
        state=args.state or defaults.state,
        inbox=args.inbox or defaults.inbox,
        timezone=defaults.timezone,
        daily_wake_budget=defaults.daily_wake_budget,
        retry_seconds=defaults.retry_seconds,
        queued_stale_seconds=defaults.queued_stale_seconds,
        running_stale_seconds=defaults.running_stale_seconds,
    )
    try:
        if args.action == "ack":
            if not args.batch_id:
                raise ValueError("ack requires --batch-id")
            result = acknowledge(config, args.batch_id)
        elif args.action == "enqueue":
            result = enqueue(
                config,
                item_id=args.id or "",
                task=args.task or "",
                project=args.project,
                priority=args.priority,
            )
        elif args.action == "status":
            result = gate_status(config)
        else:
            result = evaluate_gate(config, prime=args.action == "prime")
    except Exception as exc:
        result = {
            "version": OUTPUT_VERSION,
            "wakeAgent": False,
            "reason": "gate_error",
            "error": _clean_text(exc, 300),
        }
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        # The scheduled check must fail closed. A non-zero pre-check causes
        # Hermes to pass the script error to an agent, turning an outage into
        # repeated model usage. Interactive and install-time actions still
        # return failure so their callers cannot mistake an error for success.
        return 0 if args.action == "check" else 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
