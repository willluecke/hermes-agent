"""Continuity rows, the rebuild budget and the transcript retention floor."""

from __future__ import annotations

import json

from agent import claude_runtime, codex_runtime
from agent.continuity import (
    TRANSCRIPT_RETENTION_DAYS,
    claude_transcript_retention_days,
    ensure_claude_transcript_retention,
)


class _Agent:
    def __init__(self):
        self.events = []
        self.tool_progress_callback = (
            lambda event, name, preview, args, **kw: self.events.append((event, name, preview, kw))
        )


def test_rebuild_keeps_the_whole_transcript_that_fits_and_names_the_rest(monkeypatch):
    messages = []
    for n in range(40):
        messages.append({"role": "user", "content": f"question {n} " + "x" * 80})
        messages.append({"role": "assistant", "content": f"answer {n} " + "y" * 80})

    full = claude_runtime.claude_history_handoff(messages, "next")
    assert "question 0 " in full and "answer 39 " in full  # no 24-message cap
    assert "omitted" not in full

    monkeypatch.setattr(claude_runtime, "CLAUDE_HANDOFF_MAX_CHARS", 1_000)
    cut = claude_runtime.claude_history_handoff(messages, "next")
    carried, omitted = claude_runtime.claude_handoff_coverage(messages)
    assert omitted and carried + omitted == 80
    assert f"[{omitted} older messages omitted" in cut
    assert "answer 39 " in cut and "question 0 " not in cut
    assert cut.endswith("Current request:\nnext")


def test_rebuild_row_names_the_reason_and_what_was_carried():
    agent = _Agent()
    claude_runtime._announce_claude_continuity(
        agent,
        prior_messages=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        rebuild_reason="no Claude session was recorded for this conversation on this runtime",
        resumed_from_disk=False,
        delta=[],
        session_id="",
    )
    [(event, runtime, text, kw)] = agent.events
    assert (event, runtime, kw["mode"]) == ("session.continuity", "claude-code", "rebuilt")
    assert text == (
        "Session rebuilt from the stored transcript: no Claude session was recorded "
        "for this conversation on this runtime. The new Claude session was given 2 messages."
    )


def test_a_first_turn_or_plain_continuation_shows_no_row():
    agent = _Agent()
    claude_runtime._announce_claude_continuity(
        agent, prior_messages=[], rebuild_reason="no Claude session was recorded",
        resumed_from_disk=False, delta=[], session_id="",
    )
    claude_runtime._announce_claude_continuity(
        agent, prior_messages=[{"role": "user", "content": "hi"}], rebuild_reason="",
        resumed_from_disk=False, delta=[], session_id="abc",
    )
    assert agent.events == []


def test_compaction_is_a_row():
    agent = _Agent()
    bridge = claude_runtime.make_claude_code_event_bridge(agent)
    bridge({
        "type": "system", "subtype": "compact_boundary",
        "compact_metadata": {"trigger": "auto", "pre_tokens": 967_606, "post_tokens": 8_891},
    })
    [(event, runtime, text, kw)] = agent.events
    assert kw["mode"] == "compacted" and kw["pre_tokens"] == 967_606
    assert text.startswith("Claude Code compacted this session's context (968k → 9k tokens), automatically.")


def test_codex_rows_for_rebuild_resume_and_catch_up():
    agent = _Agent()
    entries = [("user", "a"), ("assistant", "b")]
    codex_runtime._announce_codex_continuity(
        agent, rebuilt=True, resumed=False, entries=entries,
        reason="transcript-diverged", thread_id="0123456789",
    )
    codex_runtime._announce_codex_continuity(
        agent, rebuilt=False, resumed=True, entries=entries[:1], reason="resume", thread_id="0123456789",
    )
    codex_runtime._announce_codex_continuity(
        agent, rebuilt=False, resumed=False, entries=entries, reason="resume", thread_id="0123456789",
    )
    codex_runtime._announce_codex_continuity(
        agent, rebuilt=True, resumed=False, entries=[], reason="no-bound-thread", thread_id="x",
    )
    texts = [(kw["mode"], text) for _, _, text, kw in agent.events]
    assert texts == [
        ("rebuilt", "Session rebuilt from the stored transcript: the transcript was edited or rolled back since the thread's last turn. New Codex thread 01234567 was given 2 messages."),
        ("resumed", "Session resumed from disk: Codex thread 01234567, plus the 1 message added since its last turn."),
        ("caught_up", "Passed 2 messages added since this thread's last turn to Codex thread 01234567."),
    ]


def test_retention_floor_is_raised_and_keeps_other_settings(tmp_path):
    (tmp_path / "settings.json").write_text(json.dumps({"theme": "dark"}))
    assert claude_transcript_retention_days(tmp_path) == 30
    assert ensure_claude_transcript_retention(tmp_path) is True
    assert json.loads((tmp_path / "settings.json").read_text()) == {
        "theme": "dark", "cleanupPeriodDays": TRANSCRIPT_RETENTION_DAYS,
    }
    assert ensure_claude_transcript_retention(tmp_path) is False


def test_retention_leaves_an_unreadable_settings_file_alone(tmp_path):
    (tmp_path / "settings.json").write_text("{not json")
    assert claude_transcript_retention_days(tmp_path) is None
    assert ensure_claude_transcript_retention(tmp_path) is False
    assert (tmp_path / "settings.json").read_text() == "{not json"


def test_readiness_reports_expiring_transcripts(tmp_path, monkeypatch):
    from gateway.readiness import _probe_transcript_retention

    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    assert _probe_transcript_retention()["status"] == "ok"  # no sessions yet
    (tmp_path / "projects").mkdir()
    assert _probe_transcript_retention() == {
        "status": "degraded", "detail": "transcripts expire", "days": 30,
        "required_days": TRANSCRIPT_RETENTION_DAYS,
    }
    (tmp_path / "settings.json").write_text(json.dumps({"cleanupPeriodDays": 3650}))
    assert _probe_transcript_retention() == {"status": "ok", "days": 3650}
