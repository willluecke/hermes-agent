from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "ops" / "command-center" / "agentic-loop-gate.py"
SPEC = importlib.util.spec_from_file_location("agentic_loop_gate", MODULE_PATH)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)


@pytest.fixture
def configured_gate(tmp_path):
    database = tmp_path / "sync.db"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE agent_jobs (
              id TEXT PRIMARY KEY,
              status TEXT NOT NULL,
              provider TEXT NOT NULL,
              mode TEXT NOT NULL,
              project TEXT NOT NULL,
              priority INTEGER NOT NULL,
              created_at INTEGER NOT NULL,
              updated_at INTEGER NOT NULL,
              claimed_at INTEGER,
              worker_id TEXT,
              data TEXT NOT NULL
            );
            CREATE TABLE management_entities (
              kind TEXT NOT NULL,
              id TEXT NOT NULL,
              updated_at INTEGER NOT NULL,
              data TEXT NOT NULL,
              PRIMARY KEY (kind, id)
            );
            """
        )
    config = gate.GateConfig(
        database=database,
        state=tmp_path / "agentic-loop" / "gate-state.json",
        inbox=tmp_path / "agentic-loop" / "inbox.jsonl",
        timezone="UTC",
        daily_wake_budget=3,
        retry_seconds=3600,
        queued_stale_seconds=1800,
        running_stale_seconds=7200,
    )
    return config


def _insert_job(
    config,
    *,
    job_id: str,
    status: str,
    updated_at: int,
    context_type: str | None = "hermes-orchestration",
    project: str = "hermes-chat",
):
    data = {
        "id": job_id,
        "status": status,
        "title": f"Job {job_id}",
        "contextType": context_type,
        "summary": "Bounded result",
        "commit": "abc123" if status == "completed" else None,
    }
    with sqlite3.connect(config.database) as connection:
        connection.execute(
            "INSERT INTO agent_jobs "
            "(id,status,provider,mode,project,priority,created_at,updated_at,data) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (
                job_id,
                status,
                "claude",
                "implement",
                project,
                50,
                updated_at - 1,
                updated_at,
                json.dumps(data),
            ),
        )


def _upsert_entity(config, kind: str, entity_id: str, updated_at: int, data: dict):
    with sqlite3.connect(config.database) as connection:
        connection.execute(
            "INSERT INTO management_entities (kind,id,updated_at,data) VALUES (?,?,?,?) "
            "ON CONFLICT(kind,id) DO UPDATE SET updated_at=excluded.updated_at,data=excluded.data",
            (kind, entity_id, updated_at, json.dumps(data)),
        )


def test_prime_baselines_existing_state_without_waking(configured_gate):
    _insert_job(
        configured_gate,
        job_id="historical",
        status="completed",
        updated_at=1_000,
    )

    primed = gate.evaluate_gate(configured_gate, now_ms=2_000, prime=True)
    unchanged = gate.evaluate_gate(configured_gate, now_ms=3_000)

    assert primed["wakeAgent"] is False
    assert primed["reason"] == "baseline_primed"
    assert unchanged == {
        "version": gate.OUTPUT_VERSION,
        "wakeAgent": False,
        "reason": "no_actionable_change",
        "budgetRemaining": 3,
    }


def test_orchestration_result_wakes_once_until_exact_ack(configured_gate):
    gate.evaluate_gate(configured_gate, now_ms=1_000, prime=True)
    _insert_job(
        configured_gate,
        job_id="worker-result",
        status="completed",
        updated_at=2_000,
    )

    first = gate.evaluate_gate(configured_gate, now_ms=3_000)
    duplicate = gate.evaluate_gate(configured_gate, now_ms=4_000)
    wrong_ack = gate.acknowledge(configured_gate, "loop_wrong", now_ms=5_000)
    acknowledged = gate.acknowledge(configured_gate, first["batchId"], now_ms=6_000)
    settled = gate.evaluate_gate(configured_gate, now_ms=7_000)

    assert first["wakeAgent"] is True
    assert first["events"][0]["kind"] == "orchestration_result"
    assert duplicate["reason"] == "awaiting_batch_acknowledgement"
    assert wrong_ack["acknowledged"] is False
    assert acknowledged["acknowledged"] is True
    assert settled["reason"] == "no_actionable_change"


def test_direct_native_job_does_not_enter_governed_loop(configured_gate):
    gate.evaluate_gate(configured_gate, now_ms=1_000, prime=True)
    _insert_job(
        configured_gate,
        job_id="direct-chat",
        status="completed",
        updated_at=2_000,
        context_type=None,
    )

    result = gate.evaluate_gate(configured_gate, now_ms=3_000)

    assert result["wakeAgent"] is False
    assert result["reason"] == "no_actionable_change"


def test_same_timestamp_orchestration_results_are_not_missed(configured_gate):
    gate.evaluate_gate(configured_gate, now_ms=1_000, prime=True)
    _insert_job(configured_gate, job_id="first", status="completed", updated_at=2_000)
    first = gate.evaluate_gate(configured_gate, now_ms=3_000)
    gate.acknowledge(configured_gate, first["batchId"], now_ms=4_000)

    _insert_job(configured_gate, job_id="second", status="completed", updated_at=2_000)
    second = gate.evaluate_gate(configured_gate, now_ms=5_000)

    assert second["wakeAgent"] is True
    assert second["events"][0]["jobId"] == "second"


def test_human_ready_work_is_ignored_but_review_transition_wakes(configured_gate):
    gate.evaluate_gate(configured_gate, now_ms=1_000, prime=True)
    item = {
        "title": "Review customer request",
        "company": "3DCarParts",
        "workstream": "Customers",
        "owner": "Will",
        "due": "Today",
        "status": "Ready",
        "priority": "Critical",
        "nextAction": "Read the request",
        "consequence": 95,
    }
    _upsert_entity(configured_gate, "work-item", "work-1", 2_000, item)
    ignored = gate.evaluate_gate(configured_gate, now_ms=3_000)

    item["status"] = "Review"
    _upsert_entity(configured_gate, "work-item", "work-1", 4_000, item)
    review = gate.evaluate_gate(configured_gate, now_ms=5_000)

    assert ignored["wakeAgent"] is False
    assert review["wakeAgent"] is True
    assert review["events"][0]["kind"] == "work_item"


def test_warning_metric_transition_wakes(configured_gate):
    gate.evaluate_gate(configured_gate, now_ms=1_000, prime=True)
    metric = {
        "key": "source-health",
        "label": "Source health",
        "company": "RegWatch",
        "value": "Healthy",
        "detail": "All sources current",
        "tone": "positive",
    }
    _upsert_entity(configured_gate, "metric", "metric-1", 2_000, metric)
    healthy = gate.evaluate_gate(configured_gate, now_ms=3_000)

    metric.update(value="Degraded", detail="One source stale", tone="warning")
    _upsert_entity(configured_gate, "metric", "metric-1", 4_000, metric)
    warning = gate.evaluate_gate(configured_gate, now_ms=5_000)

    assert healthy["wakeAgent"] is False
    assert warning["wakeAgent"] is True
    assert warning["events"][0]["kind"] == "warning_metric"


def test_stale_job_alert_wakes_only_once(configured_gate):
    _insert_job(configured_gate, job_id="stalled", status="queued", updated_at=1_000)
    gate.evaluate_gate(configured_gate, now_ms=2_000, prime=True)

    stale = gate.evaluate_gate(configured_gate, now_ms=1_802_000)
    gate.acknowledge(configured_gate, stale["batchId"], now_ms=1_803_000)
    duplicate = gate.evaluate_gate(configured_gate, now_ms=1_804_000)

    assert stale["wakeAgent"] is True
    assert stale["events"][0]["kind"] == "stalled_job"
    assert duplicate["wakeAgent"] is False


def test_new_events_supersede_unacknowledged_batch(configured_gate):
    gate.evaluate_gate(configured_gate, now_ms=1_000, prime=True)
    _insert_job(
        configured_gate,
        job_id="job-one",
        status="completed",
        updated_at=2_000,
    )
    first = gate.evaluate_gate(configured_gate, now_ms=3_000)

    _insert_job(
        configured_gate,
        job_id="job-two",
        status="failed",
        updated_at=4_000,
    )
    second = gate.evaluate_gate(configured_gate, now_ms=5_000)

    assert second["wakeAgent"] is True
    assert second["batchId"] != first["batchId"]
    assert len(second["events"]) == 2
    assert gate.acknowledge(configured_gate, first["batchId"])["acknowledged"] is False


def test_daily_budget_holds_pending_work_until_next_day(configured_gate):
    config = gate.GateConfig(**{
        **configured_gate.__dict__,
        "daily_wake_budget": 1,
    })
    day = 1_800_000_000_000
    gate.evaluate_gate(config, now_ms=day, prime=True)
    _insert_job(config, job_id="first", status="completed", updated_at=day + 1)
    first = gate.evaluate_gate(config, now_ms=day + 2)
    gate.acknowledge(config, first["batchId"], now_ms=day + 3)

    _insert_job(config, job_id="second", status="failed", updated_at=day + 4)
    held = gate.evaluate_gate(config, now_ms=day + 5)
    next_day = gate.evaluate_gate(config, now_ms=day + 86_400_000)

    assert held["wakeAgent"] is False
    assert held["reason"] == "daily_wake_budget_exhausted"
    assert next_day["wakeAgent"] is True
    assert next_day["events"][0]["jobId"] == "second"


def test_explicit_inbox_event_is_private_and_deduplicated(configured_gate):
    gate.evaluate_gate(configured_gate, now_ms=1_000, prime=True)
    enqueued = gate.enqueue(
        configured_gate,
        item_id="explicit-1",
        project="reg-watch",
        priority="High",
        task="Inspect one failing source.",
    )
    result = gate.evaluate_gate(configured_gate, now_ms=2_000)

    assert enqueued == {"enqueued": True, "id": "explicit-1"}
    assert result["wakeAgent"] is True
    assert result["events"][0]["kind"] == "inbox"
    assert configured_gate.inbox.stat().st_mode & 0o777 == 0o600


def test_event_burst_is_backlogged_without_replay(configured_gate):
    gate.evaluate_gate(configured_gate, now_ms=1_000, prime=True)
    for index in range(25):
        gate.enqueue(
            configured_gate,
            item_id=f"burst-{index:02d}",
            task=f"Inspect bounded event {index}.",
        )

    first = gate.evaluate_gate(configured_gate, now_ms=2_000)
    duplicate = gate.evaluate_gate(configured_gate, now_ms=3_000)
    acknowledged = gate.acknowledge(configured_gate, first["batchId"], now_ms=4_000)
    second = gate.evaluate_gate(configured_gate, now_ms=5_000)

    assert len(first["events"]) == gate.MAX_EVENTS_PER_BATCH
    assert first["backlogEventCount"] == 1
    assert duplicate["reason"] == "awaiting_batch_acknowledgement"
    assert acknowledged["nextBatchId"] == second["batchId"]
    assert len(second["events"]) == 1
    assert second["events"][0]["inboxId"] == "burst-24"


def test_scheduled_check_fails_closed_without_model_wake(tmp_path, capsys):
    exit_code = gate.main([
        "check",
        "--db",
        str(tmp_path / "missing.db"),
        "--state",
        str(tmp_path / "state.json"),
        "--inbox",
        str(tmp_path / "inbox.jsonl"),
    ])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["wakeAgent"] is False
    assert output["reason"] == "gate_error"
