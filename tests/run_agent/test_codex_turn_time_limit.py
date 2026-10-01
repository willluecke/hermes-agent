"""What a Codex turn shows when it compacts and when a limit ends it.

2026-10-01, talos: Codex compacted at 48 and 103 minutes and kept working;
the then fixed 2-hour cap cut the turn. The compaction row and the
"Compacting context ... so I can continue" status were both emitted after the
turn had returned, so they came last and the run looked killed by compaction.
The failed run's error only reached the app as a banner a reload clears.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import hermes_cli.config
import run_agent
from agent.conversation_compression import COMPACTION_STATUS
from agent.transports.codex_app_server_session import (
    TURN_INACTIVITY_ERROR_CODE,
    TURN_TIME_LIMIT_ERROR_CODE,
    CodexAppServerSession,
    TurnResult,
)


def _compaction(method: str, item_id: str) -> dict:
    return {
        "method": method,
        "params": {
            "threadId": "thread-limit-1",
            "turnId": "turn-limit-1",
            "item": {"type": "contextCompaction", "id": item_id},
        },
    }


def _make_agent(timeline: list):
    def progress(event, *args, **kwargs):
        if event == "session.continuity":
            timeline.append(("row", kwargs.get("mode"), args[1]))

    def status(kind, text):
        timeline.append(("status", kind, text))

    return run_agent.AIAgent(
        api_key="stub",
        base_url="https://stub.invalid",
        provider="openai",
        api_mode="codex_app_server",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        tool_progress_callback=progress,
        status_callback=status,
    )


@pytest.fixture
def fixed_config(monkeypatch):
    holder = {"agent": {}}
    monkeypatch.setattr(hermes_cli.config, "load_config", lambda: holder)
    monkeypatch.setattr(
        CodexAppServerSession, "ensure_started", lambda self: "thread-limit-1"
    )
    return holder


def _run(agent):
    with patch.object(agent, "_spawn_background_review", return_value=None):
        return agent.run_conversation("keep building talos")


def test_the_configured_cap_reaches_every_codex_turn(monkeypatch, fixed_config):
    fixed_config["agent"]["turn_time_limit_hours"] = 3
    seen = []

    def fake_run_turn(self, user_input, **kwargs):
        seen.append(kwargs.get("absolute_turn_timeout"))
        return TurnResult(final_text="done", thread_id="thread-limit-1",
                          turn_id="turn-limit-1")

    monkeypatch.setattr(CodexAppServerSession, "run_turn", fake_run_turn)
    result = _run(_make_agent([]))

    assert result["completed"] is True
    assert seen == [3 * 60 * 60]


def test_talos_sequence_shows_compactions_live_and_the_limit_last(monkeypatch, fixed_config):
    """Compactions appear while the turn runs; the stop row says what ended it."""
    timeline: list = []

    def fake_run_turn(self, user_input, **kwargs):
        for method, item_id in (
            ("item/started", "compact-1"), ("item/completed", "compact-1"),
            ("item/started", "compact-2"), ("item/completed", "compact-2"),
        ):
            self._on_event(_compaction(method, item_id))
            timeline.append(("work", method, item_id))
        return TurnResult(
            final_text="", thread_id="thread-limit-1", turn_id="turn-limit-1",
            compacted=True, interrupted=True, should_retire=True,
            error_code=TURN_TIME_LIMIT_ERROR_CODE,
            error="turn reached its time limit (absolute timeout of 43200s)",
        )

    monkeypatch.setattr(CodexAppServerSession, "run_turn", fake_run_turn)
    agent = _make_agent(timeline)
    result = _run(agent)

    statuses = [entry for entry in timeline if entry[0] == "status" and entry[2] == COMPACTION_STATUS]
    rows = [entry for entry in timeline if entry[0] == "row"]
    # One status as each compaction starts and one row as it finishes, both
    # before the work that followed them, none repeated after the turn.
    assert len(statuses) == 2
    assert [row[1] for row in rows] == ["compacted", "compacted", "stopped"]
    first_work_after = timeline.index(("work", "item/completed", "compact-1"))
    assert timeline.index(statuses[0]) < first_work_after
    assert timeline.index(rows[0]) < first_work_after
    stop_at = timeline.index(rows[-1])
    assert all(entry[0] != "status" for entry in timeline[stop_at:])
    assert rows[-1][2] == (
        "Codex stopped this turn at the 12-hour turn limit while it was still "
        "working. Send a message to have it continue."
    )
    assert result["completed"] is False
    assert result["error_code"] == TURN_TIME_LIMIT_ERROR_CODE


def test_a_compaction_the_bridge_missed_still_gets_one_row(monkeypatch, fixed_config):
    timeline: list = []

    def fake_run_turn(self, user_input, **kwargs):
        return TurnResult(final_text="done", thread_id="thread-limit-1",
                          turn_id="turn-limit-1", compacted=True)

    monkeypatch.setattr(CodexAppServerSession, "run_turn", fake_run_turn)
    result = _run(_make_agent(timeline))

    assert result["completed"] is True
    assert [entry[1] for entry in timeline if entry[0] == "row"] == ["compacted"]
    # The status line is never printed after the turn has returned.
    assert not [entry for entry in timeline if entry[0] == "status" and entry[2] == COMPACTION_STATUS]


def test_an_inactivity_stop_names_the_silence(monkeypatch, fixed_config):
    timeline: list = []

    def fake_run_turn(self, user_input, **kwargs):
        return TurnResult(
            final_text="", thread_id="thread-limit-1", turn_id="turn-limit-1",
            interrupted=True, should_retire=True,
            error_code=TURN_INACTIVITY_ERROR_CODE,
            error="turn timed out after 600.0s without activity",
        )

    monkeypatch.setattr(CodexAppServerSession, "run_turn", fake_run_turn)
    _run(_make_agent(timeline))

    rows = [entry for entry in timeline if entry[0] == "row"]
    assert [row[1] for row in rows] == ["stopped"]
    assert "after 10 minutes with no activity" in rows[0][2]


def test_other_failures_leave_no_stop_row(monkeypatch, fixed_config):
    timeline: list = []

    def fake_run_turn(self, user_input, **kwargs):
        return TurnResult(final_text="", thread_id="thread-limit-1",
                          turn_id="turn-limit-1", error="turn/start failed")

    monkeypatch.setattr(CodexAppServerSession, "run_turn", fake_run_turn)
    _run(_make_agent(timeline))

    assert [entry for entry in timeline if entry[0] == "row"] == []
