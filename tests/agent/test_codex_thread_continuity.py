"""A native Codex thread may only continue a transcript it can be proven to hold.

Codex threads stay resumable indefinitely, so the bound thread id says nothing
about whether that thread is current. When a conversation runs turns on another
runtime — a deliberate model switch, or the automatic one after a credit limit —
the Hermes record advances while the native thread stands still. Resuming it then
silently rolls the conversation back to a stale tail, with no signal anywhere in
the UI.

These tests cover the record that makes that detectable (a fingerprint of the
dialogue prefix the thread consumed), the delta recovery that uses it, and the
two ways the record could be fooled: storage-layer normalization on reload, and
a handoff that omits messages from the middle of the transcript.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from agent.codex_runtime import (
    _codex_catch_up_handoff,
    _codex_dialogue_entries,
    _codex_history_fingerprint,
    _codex_history_handoff,
    _codex_resume_plan,
    _normalized_codex_cwd,
    _render_history_blocks,
    _with_turn_note,
    _without_input_echo,
)


CWD = "/home/will/coding-projects/hermes-chat"


def _state(entries, thread_id="thread-1", cwd=CWD, **overrides):
    state = {
        "version": 1,
        "thread_id": thread_id,
        "cwd": _normalized_codex_cwd(cwd),
        "seen_count": len(entries),
        "fingerprint": _codex_history_fingerprint(entries),
        "updated_at": 0.0,
    }
    state.update(overrides)
    return state


def _entries(*pairs):
    return [(role, text) for role, text in pairs]


class TestResumePlan:
    def test_matching_prefix_resumes_with_no_delta(self):
        entries = _entries(("user", "a"), ("assistant", "b"))
        thread, pending, reason = _codex_resume_plan(
            thread_id="thread-1",
            state=_state(entries),
            prior_entries=entries,
            cwd=CWD,
        )
        assert (thread, pending, reason) == ("thread-1", [], "resume")

    def test_appended_messages_become_the_catch_up_delta(self):
        seen = _entries(("user", "a"), ("assistant", "b"))
        now = seen + _entries(("user", "c"), ("assistant", "d"))
        thread, pending, reason = _codex_resume_plan(
            thread_id="thread-1",
            state=_state(seen),
            prior_entries=now,
            cwd=CWD,
        )
        assert thread == "thread-1"
        assert reason == "resume"
        assert pending == _entries(("user", "c"), ("assistant", "d"))

    def test_thread_without_a_record_is_never_trusted(self):
        """Every thread bound before this record existed arrives this way."""
        entries = _entries(("user", "a"))
        thread, pending, reason = _codex_resume_plan(
            thread_id="thread-legacy",
            state={},
            prior_entries=entries,
            cwd=CWD,
        )
        assert thread == ""
        assert reason == "no-recorded-history"
        assert pending == entries

    def test_rewritten_prefix_is_not_an_append(self):
        seen = _entries(("user", "a"), ("assistant", "b"))
        rewound = _entries(("user", "a"), ("assistant", "DIFFERENT"))
        thread, _pending, reason = _codex_resume_plan(
            thread_id="thread-1",
            state=_state(seen),
            prior_entries=rewound,
            cwd=CWD,
        )
        assert thread == ""
        assert reason == "transcript-diverged"

    def test_transcript_shorter_than_the_thread_is_divergence(self):
        seen = _entries(("user", "a"), ("assistant", "b"), ("user", "c"))
        thread, _pending, reason = _codex_resume_plan(
            thread_id="thread-1",
            state=_state(seen),
            prior_entries=seen[:1],
            cwd=CWD,
        )
        assert thread == ""
        assert reason == "transcript-shortened"

    def test_project_switch_does_not_resume(self):
        entries = _entries(("user", "a"))
        thread, _pending, reason = _codex_resume_plan(
            thread_id="thread-1",
            state=_state(entries, cwd="/home/will/coding-projects/ToneTrace"),
            prior_entries=entries,
            cwd=CWD,
        )
        assert thread == ""
        assert reason == "cwd-changed"

    def test_record_belonging_to_another_thread_is_rejected(self):
        entries = _entries(("user", "a"))
        thread, _pending, reason = _codex_resume_plan(
            thread_id="thread-2",
            state=_state(entries, thread_id="thread-1"),
            prior_entries=entries,
            cwd=CWD,
        )
        assert thread == ""
        assert reason == "thread-rebound"

    def test_unusable_seen_count_is_rejected(self):
        entries = _entries(("user", "a"))
        for bad in (None, -1, "2", True):
            thread, _pending, reason = _codex_resume_plan(
                thread_id="thread-1",
                state=_state(entries, seen_count=bad),
                prior_entries=entries,
                cwd=CWD,
            )
            assert thread == "", bad
            assert reason == "unusable-seen-count", bad


class TestHandoffWindow:
    def test_budget_truncation_keeps_an_unbroken_recent_window(self):
        """Skipping one oversized message and continuing with older, smaller
        ones punches a hole in the middle of the transcript and discloses it
        only as a count — which reads as 'older context omitted' when it really
        means 'the reply you are about to see is missing'."""
        entries = _entries(
            ("user", "OLDEST"),
            ("assistant", "X" * 300),
            ("user", "NEWER"),
            ("assistant", "NEWEST"),
        )
        rendered, omitted, truncated = _render_history_blocks(entries, 200)
        body = "\n\n".join(rendered)

        assert "NEWEST" in body and "NEWER" in body
        # The oversized message did not fit, so everything older stops here.
        assert "OLDEST" not in body
        assert omitted == 2
        assert truncated is False

    def test_a_single_oversized_message_is_cut_not_dropped(self):
        rendered, omitted, truncated = _render_history_blocks(
            _entries(("assistant", "HEAD" + "X" * 400 + "TAIL")), 100
        )
        body = "\n\n".join(rendered)
        assert "TAIL" in body
        assert "HEAD" not in body
        assert truncated is True
        assert omitted == 0

    def test_omissions_are_disclosed_in_the_handoff(self):
        from agent.codex_runtime import _CODEX_HISTORY_HANDOFF_MAX_CHARS

        entries = _entries(
            ("user", "OLDEST"),
            ("assistant", "X" * _CODEX_HISTORY_HANDOFF_MAX_CHARS),
        )
        text = _codex_history_handoff(entries, "do the thing")
        assert "1 older messages omitted" in text
        assert "cut at its start" in text
        assert "OLDEST" not in text

    def test_catch_up_carries_only_the_missed_messages(self):
        text = _codex_catch_up_handoff(
            _entries(("user", "I have been sending 3 a day")), "did you see this?"
        )
        assert "<hermes_missed_messages>" in text
        assert "I have been sending 3 a day" in text
        assert "<current_user_request>\ndid you see this?" in text

    def test_empty_delta_leaves_the_user_message_alone(self):
        assert _codex_catch_up_handoff([], "hello") == "hello"
        assert _codex_history_handoff([], "hello") == "hello"

    def test_multimodal_user_message_keeps_its_parts(self):
        parts = [{"type": "input_text", "text": "look"}, {"type": "image_url"}]
        wrapped = _codex_catch_up_handoff(_entries(("user", "missed")), parts)
        assert isinstance(wrapped, list)
        assert wrapped[1:-1] == parts
        assert "missed" in wrapped[0]["text"]


class TestFingerprintSurvivesStorage:
    def test_fingerprint_is_stable_across_a_real_persist_and_reload(self):
        """The guard is only safe if the fingerprint computed from the live
        message list still matches the one computed from the transcript the
        next turn reloads. If it drifts, every turn looks like divergence and
        the thread is discarded on each one.

        Recalled memory is injected into the live user message and stripped by
        the storage layer on load, so this is a real drift source, not a
        hypothetical.
        """
        from hermes_state import SessionDB
        from run_agent import AIAgent

        tmp = tempfile.mkdtemp(prefix="codex_continuity_")
        db = SessionDB(Path(tmp) / "state.db")
        sid = "sess-codex-continuity"
        db.create_session(session_id=sid, source="api_server", model="codex")
        agent = AIAgent(
            api_key="test-key",
            base_url="https://stub.invalid",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_db=db,
            session_id=sid,
        )
        agent._session_db_created = True

        live = [
            {
                "role": "user",
                "content": (
                    "<memory-context>recalled: the mailbox is Pending"
                    "</memory-context>\nDid you see this or no?"
                ),
            },
            {"role": "assistant", "content": "Warmup started Sep 3."},
            {"role": "user", "content": "How long do I warm it up for?"},
        ]
        assert agent._flush_messages_to_session_db(live) is not False

        reloaded = db.get_messages_as_conversation(sid)
        assert [m["role"] for m in reloaded] == ["user", "assistant", "user"]
        # Guard the guard: if storage ever stops stripping this, the test above
        # would pass without exercising any normalization at all.
        assert "<memory-context>" in live[0]["content"]
        assert "<memory-context>" not in reloaded[0]["content"]

        assert _codex_history_fingerprint(
            _codex_dialogue_entries(live)
        ) == _codex_history_fingerprint(_codex_dialogue_entries(reloaded))

    def test_image_turn_fingerprint_survives_persist_and_reload(self):
        """The store writes each image as a ``[screenshot]`` placeholder. A
        fingerprint of the live list that skipped images reported divergence
        on the turn after every image turn (2026-09-25, logoception)."""
        from hermes_state import SessionDB
        from run_agent import AIAgent

        tmp = tempfile.mkdtemp(prefix="codex_continuity_img_")
        db = SessionDB(Path(tmp) / "state.db")
        sid = "sess-codex-image"
        db.create_session(session_id=sid, source="api_server", model="codex")
        agent = AIAgent(
            api_key="test-key",
            base_url="https://stub.invalid",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_db=db,
            session_id=sid,
        )
        agent._session_db_created = True
        live = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "make the glass logos"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBBB"}},
                ],
            },
            {"role": "assistant", "content": "Which vector files?"},
        ]
        assert agent._flush_messages_to_session_db(live) is not False
        reloaded = db.get_messages_as_conversation(sid)
        assert "[screenshot]" in reloaded[0]["content"]
        assert _codex_history_fingerprint(
            _codex_dialogue_entries(live)
        ) == _codex_history_fingerprint(_codex_dialogue_entries(reloaded))

    def test_tool_rows_do_not_change_the_dialogue_fingerprint(self):
        """Tool rows are projected differently by each runtime, so including
        them would report divergence for identical dialogue."""
        dialogue = [
            {"role": "user", "content": "run it"},
            {"role": "assistant", "content": "done"},
        ]
        with_tools = [
            dialogue[0],
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "exec_1",
                        "type": "function",
                        "function": {"name": "exec_command", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "exec_1", "content": "ok"},
            dialogue[1],
        ]
        assert _codex_history_fingerprint(
            _codex_dialogue_entries(dialogue)
        ) == _codex_history_fingerprint(_codex_dialogue_entries(with_tools))


class TestTurnInput:
    def test_a_note_keeps_image_parts_as_parts(self):
        """Formatting a list into a string sent images to Codex as base64
        text: 566k characters of it on one logoception turn."""
        parts = [
            {"type": "text", "text": "make the glass logos"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]
        noted = _with_turn_note(parts, "Preflight from Jev ...")
        assert noted[:2] == parts
        assert noted[2] == {"type": "text", "text": "Preflight from Jev ..."}
        assert all("base64" not in str(p.get("text", "")) for p in noted)

    def test_a_note_on_text_input_stays_text(self):
        assert _with_turn_note("hi", "note") == "hi\n\nnote"


class TestInputEcho:
    def test_the_echo_of_the_sent_input_is_dropped(self):
        sent = [
            {"type": "text", "text": "make the glass logos"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            {"type": "text", "text": "Preflight from Jev ..."},
        ]
        projected = [
            {"role": "user", "content": "make the glass logos\nPreflight from Jev ..."},
            {"role": "assistant", "content": "On it."},
        ]
        assert _without_input_echo(projected, sent) == [projected[1]]

    def test_a_steer_is_kept(self):
        """A mid-turn turn/steer arrives as a userMessage too, and the echo is
        its only record."""
        projected = [
            {"role": "user", "content": "read agents.md"},
            {"role": "assistant", "content": "Reading."},
            {"role": "user", "content": "actually skip the README"},
            {"role": "assistant", "content": "Skipped."},
        ]
        kept = _without_input_echo(projected, "read agents.md")
        assert [m["content"] for m in kept] == ["Reading.", "actually skip the README", "Skipped."]

    def test_nothing_matching_leaves_rows_alone(self):
        projected = [{"role": "user", "content": "something else"}]
        assert _without_input_echo(projected, "read agents.md") == projected
