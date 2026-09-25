from __future__ import annotations

import base64
from pathlib import Path
from uuid import uuid4
from unittest.mock import patch

import pytest

import run_agent
from agent.transports.claude_code_session import (
    CLAUDE_AUTH_ERROR_CODE,
    ClaudeCodeError,
    ClaudeCodeSession,
    ClaudeCodeTurnResult,
)
from hermes_state import SessionDB


@pytest.fixture(autouse=True)
def _claude_config_dir(tmp_path, monkeypatch):
    """Claude Code's config dir: the retention floor and --resume transcripts."""
    root = tmp_path / "claude-config"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(root))
    return root


def _write_transcript(session_id):
    """What Claude Code does once a session exists: its transcript on disk."""
    import os

    path = Path(os.environ["CLAUDE_CONFIG_DIR"]) / "projects" / "-work" / f"{session_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n", encoding="utf-8")


def _make_agent(*, session_id=None, session_db=None, reasoning_config=None):
    return run_agent.AIAgent(
        api_key="claude-cli-subscription-auth",
        base_url="claude-code://local",
        provider="claude-code",
        api_mode="claude_code",
        model="claude-fable-5",
        session_id=session_id,
        session_db=session_db,
        reasoning_config=reasoning_config,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        skip_background_review=True,
    )


def test_claude_code_api_mode_dispatches_through_hermes(monkeypatch):
    monkeypatch.setattr(
        ClaudeCodeSession,
        "run_turn",
        lambda self, prompt: ClaudeCodeTurnResult(
            final_text="Fable completed inside Hermes.",
            session_id="00000000-0000-4000-8000-000000000000",
            usage={"input_tokens": 10, "output_tokens": 5},
            tool_iterations=1,
        ),
    )
    agent = _make_agent()

    with patch.object(agent, "_spawn_background_review", return_value=None):
        result = agent.run_conversation("continue")

    assert agent.api_mode == "claude_code"
    assert result["final_response"] == "Fable completed inside Hermes."
    assert result["completed"] is True
    assert result["api_calls"] == 1
    assert result["agent_persisted"] is True


def test_claude_code_read_only_posture_reaches_native_session(monkeypatch):
    observed = {}

    def _run_turn(session, prompt):
        observed["read_only"] = session.read_only
        return ClaudeCodeTurnResult(final_text="inspection only")

    monkeypatch.setattr(ClaudeCodeSession, "run_turn", _run_turn)
    agent = _make_agent()
    agent.read_only = True

    with patch.object(agent, "_spawn_background_review", return_value=None):
        agent.run_conversation("inspect")

    assert observed["read_only"] is True


def test_claude_auth_preflight_error_is_structured_and_terminal(monkeypatch):
    def _run_turn(_session, _prompt):
        raise ClaudeCodeError(
            "Claude Max authentication is unavailable. Reauthenticate.",
            error_code=CLAUDE_AUTH_ERROR_CODE,
        )

    monkeypatch.setattr(ClaudeCodeSession, "run_turn", _run_turn)
    agent = _make_agent()

    with patch.object(agent, "_spawn_background_review", return_value=None):
        result = agent.run_conversation("continue")

    assert result["completed"] is False
    assert result["partial"] is True
    assert result["error_code"] == CLAUDE_AUTH_ERROR_CODE
    assert "Claude Max authentication is unavailable" in result["error"]


def test_claude_watchdog_uses_configured_deadlines_and_reports_progress(monkeypatch):
    observed = {}
    progress = []
    statuses = []

    def _resolve_timeout(key, *, default, env_var=None):
        del default, env_var
        return {
            "claude_code.resident_first_event": 11.0,
            "claude_code.startup_first_event": 22.0,
        }[key]

    def _run_turn(session, prompt):
        observed["prompt"] = prompt
        observed["resident_timeout"] = session.resident_first_event_timeout
        observed["startup_timeout"] = session.startup_first_event_timeout
        session.on_watchdog_timeout(
            {
                "code": "claude_first_event_timeout",
                "attempt": 1,
                "retrying": True,
                "timeout_seconds": 11.0,
                "session_id": session.session_id,
            }
        )
        return ClaudeCodeTurnResult(final_text="recovered", watchdog_retries=1)

    monkeypatch.setattr("agent.deadline.resolve_timeout", _resolve_timeout)
    monkeypatch.setattr(ClaudeCodeSession, "run_turn", _run_turn)
    agent = _make_agent()
    agent.tool_progress_callback = lambda *args, **kwargs: progress.append(
        (args, kwargs)
    )
    agent.status_callback = lambda *args: statuses.append(args)

    with patch.object(agent, "_spawn_background_review", return_value=None):
        result = agent.run_conversation("repair")

    assert result["completed"] is True
    assert result["watchdog_retries"] == 1
    assert observed == {
        "prompt": "repair",
        "resident_timeout": 11.0,
        "startup_timeout": 22.0,
    }
    assert progress[0][0][:3] == (
        "runtime.first_event_timeout",
        "claude-code",
        "Claude did not acknowledge the turn within 11 seconds—resetting the runtime "
        "and retrying once.",
    )
    assert progress[0][1]["code"] == "claude_first_event_timeout"
    assert statuses == [
        (
            "lifecycle",
            "Claude did not acknowledge the turn within 11 seconds—resetting the "
            "runtime and retrying once.",
        )
    ]


def test_continuing_claude_turn_materializes_multimodal_input(monkeypatch):
    prompts = []

    def _run_turn(session, prompt):
        prompts.append(prompt)
        session.on_session_id(session.session_id)
        _write_transcript(session.session_id)
        return ClaudeCodeTurnResult(
            final_text=f"answer {len(prompts)}",
            session_id=session.session_id,
            session_confirmed=True,
        )

    monkeypatch.setattr(ClaudeCodeSession, "run_turn", _run_turn)
    db = SessionDB()
    outer_session_id = f"claude-image-continuation-{uuid4()}"
    agent = _make_agent(session_id=outer_session_id, session_db=db)

    agent.run_conversation("Start the repair.")
    history = db.get_messages_as_conversation(outer_session_id)
    image_data = base64.b64encode(b"small test image").decode("ascii")
    content = [
        {"type": "text", "text": "Inspect this screenshot."},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_data}"},
        },
    ]
    result = agent.run_conversation(content, conversation_history=history)

    assert result["completed"] is True
    assert isinstance(prompts[1], str)
    assert prompts[1].startswith("Inspect this screenshot.\n\n")
    assert "Attached images are available at:" in prompts[1]
    assert "hermes-image-1.png" in prompts[1]
    assert "data:image" not in prompts[1]
    db.close()


def test_reattached_history_images_are_reference_not_new_uploads(monkeypatch):
    prompts = []

    def _run_turn(session, prompt):
        prompts.append(prompt)
        session.on_session_id(session.session_id)
        _write_transcript(session.session_id)
        return ClaudeCodeTurnResult(
            final_text=f"answer {len(prompts)}",
            session_id=session.session_id,
            session_confirmed=True,
        )

    monkeypatch.setattr(ClaudeCodeSession, "run_turn", _run_turn)
    db = SessionDB()
    outer_session_id = f"claude-image-reference-{uuid4()}"
    agent = _make_agent(session_id=outer_session_id, session_db=db)
    image = "data:image/png;base64," + base64.b64encode(b"small test image").decode("ascii")

    agent.run_conversation(
        [
            {"type": "text", "text": "Move the artwork behind the headline."},
            {"type": "image_url", "image_url": {"url": image}},
        ]
    )
    history = db.get_messages_as_conversation(outer_session_id)
    note = (
        "(Re-attached for reference — image the user shared earlier in this "
        "conversation, not a new upload: IMG_8807.png.)"
    )
    result = agent.run_conversation(
        [
            {"type": "text", "text": "Also remove the play/pause button"},
            {"type": "text", "text": note},
            {"type": "image_url", "image_url": {"url": image}},
        ],
        conversation_history=history,
    )

    assert result["completed"] is True
    assert "Attached images are available at:" in prompts[0]
    followup = prompts[1]
    assert followup.startswith("Also remove the play/pause button\n\n")
    assert "Re-attached for reference" not in followup
    assert "Attached images are available at:" not in followup
    assert "re-supplied for reference only" in followup
    assert "hermes-image-2.png" in followup
    db.close()


def test_materialize_images_splits_new_uploads_from_reattached_history(tmp_path):
    from agent.claude_runtime import _materialize_images

    image = "data:image/png;base64," + base64.b64encode(b"small test image").decode("ascii")
    attached, referenced = _materialize_images(
        [
            {"type": "text", "text": "Compare these."},
            {"type": "image_url", "image_url": {"url": image}},
            {
                "type": "text",
                "text": "(Re-attached for reference — image the user shared earlier in this conversation: old.png.)",
            },
            {"type": "image_url", "image_url": {"url": image}},
        ],
        str(tmp_path),
    )

    assert [Path(path).name for path in attached] == ["hermes-image-1.png"]
    assert [Path(path).name for path in referenced] == ["hermes-image-3.png"]


def test_history_handoff_drops_reattach_notes_and_their_placeholders():
    from agent.claude_runtime import claude_history_handoff

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "It should sit behind the headline."},
                {"type": "text", "text": "[screenshot]"},
                {
                    "type": "text",
                    "text": "(Re-attached for reference — images the user shared earlier in this conversation: a.png, b.gif.)",
                },
                {"type": "text", "text": "[screenshot]"},
                {"type": "text", "text": "[screenshot]"},
            ],
        },
        {"role": "assistant", "content": "Done."},
        {
            "role": "user",
            "content": "Continue\n(Re-attached for reference — images the user shared earlier in this conversation: a.png, b.gif.)\n[screenshot]\n[screenshot]",
        },
        {"role": "assistant", "content": "Continuing."},
    ]

    handoff = claude_history_handoff(messages, "Also remove the play/pause button")

    assert (
        "Will: It should sit behind the headline.\n[screenshot]\n\nAssistant: Done.\n\nWill: Continue\n\nAssistant: Continuing."
        in handoff
    )
    assert "Re-attached" not in handoff
    assert handoff.endswith("Current request:\nAlso remove the play/pause button")


def test_new_hermes_parent_resumes_same_claude_session(monkeypatch):
    calls = []

    def _run_turn(session, prompt):
        calls.append(
            {
                "session_id": session.session_id,
                "resume": session.resume,
                "effort": session.effort,
                "prompt": prompt,
            }
        )
        session.on_session_id(session.session_id)
        _write_transcript(session.session_id)
        return ClaudeCodeTurnResult(
            final_text=f"answer {len(calls)}",
            session_id=session.session_id,
            session_confirmed=True,
        )

    monkeypatch.setattr(ClaudeCodeSession, "run_turn", _run_turn)
    db = SessionDB()
    outer_session_id = f"claude-continuation-{uuid4()}"

    first = _make_agent(
        session_id=outer_session_id,
        session_db=db,
        reasoning_config={"enabled": True, "effort": "medium"},
    )
    first.run_conversation("Start the repository repair.")
    history = db.get_messages_as_conversation(outer_session_id)

    second = _make_agent(
        session_id=outer_session_id,
        session_db=db,
        reasoning_config={"enabled": True, "effort": "medium"},
    )
    result = second.run_conversation("Continue", conversation_history=history)

    assert result["completed"] is True
    assert calls[0]["resume"] is False
    assert calls[1]["resume"] is True
    assert calls[1]["session_id"] == calls[0]["session_id"]
    assert calls[1]["prompt"] == "Continue"
    assert calls[1]["effort"] == "medium"
    db.close()


def test_failed_turn_still_persists_claude_session_for_resume(monkeypatch):
    calls = []

    def _run_turn(session, prompt):
        calls.append((session.session_id, session.resume, prompt))
        session.on_session_id(session.session_id)
        _write_transcript(session.session_id)
        if len(calls) == 1:
            return ClaudeCodeTurnResult(
                session_id=session.session_id,
                session_confirmed=True,
                error="You've hit your session limit",
            )
        return ClaudeCodeTurnResult(
            final_text="Recovered in the original parent.",
            session_id=session.session_id,
            session_confirmed=True,
        )

    monkeypatch.setattr(ClaudeCodeSession, "run_turn", _run_turn)
    db = SessionDB()
    outer_session_id = f"claude-failed-continuation-{uuid4()}"

    first = _make_agent(session_id=outer_session_id, session_db=db)
    failed = first.run_conversation("Do the remaining work.")
    state = db.get_session_model_config_value(
        outer_session_id, "claude_code_session"
    )
    history = db.get_messages_as_conversation(outer_session_id)

    second = _make_agent(session_id=outer_session_id, session_db=db)
    recovered = second.run_conversation("Continue", conversation_history=history)

    assert failed["partial"] is True
    assert state["session_id"] == calls[0][0]
    assert recovered["completed"] is True
    assert calls[1][0] == calls[0][0]
    assert calls[1][1] is True
    assert calls[1][2] == "Continue"
    db.close()


def test_transcript_discontinuity_starts_fresh_claude_parent(monkeypatch):
    calls = []

    def _run_turn(session, prompt):
        calls.append((session.session_id, session.resume, prompt))
        session.on_session_id(session.session_id)
        _write_transcript(session.session_id)
        return ClaudeCodeTurnResult(
            final_text=f"answer {len(calls)}",
            session_id=session.session_id,
            session_confirmed=True,
        )

    monkeypatch.setattr(ClaudeCodeSession, "run_turn", _run_turn)
    db = SessionDB()
    outer_session_id = f"claude-runtime-switch-{uuid4()}"

    first = _make_agent(session_id=outer_session_id, session_db=db)
    first.run_conversation("Use Claude first.")
    history = db.get_messages_as_conversation(outer_session_id)
    # An edited earlier row: what the session saw is no longer the prefix.
    history[0] = {**history[0], "content": "Use Claude first, edited."}
    history.extend(
        [
            {"role": "user", "content": "Use a different runtime."},
            {"role": "assistant", "content": "Other runtime answer."},
        ]
    )

    events = []
    second = _make_agent(session_id=outer_session_id, session_db=db)
    second.tool_progress_callback = lambda event, name, preview, args, **kw: events.append((event, preview, kw))
    second.run_conversation("Return to Claude.", conversation_history=history)

    assert calls[1][1] is False
    assert calls[1][0] != calls[0][0]
    assert "could not be resumed (the transcript was edited or rolled back" in calls[1][2]
    assert "Transcript:\nWill: Use Claude first, edited." in calls[1][2]
    assert "Other runtime answer." in calls[1][2]
    rows = [(preview, kw) for event, preview, kw in events if event == "session.continuity"]
    assert len(rows) == 1 and rows[0][1]["mode"] == "rebuilt"
    assert rows[0][0].startswith("Session rebuilt from the stored transcript: the transcript was edited")
    assert rows[0][1]["carried"] == 4 and rows[0][1]["omitted"] == 0
    db.close()


def test_appended_rows_resume_the_same_session_with_a_delta(monkeypatch):
    calls = []

    def _run_turn(session, prompt):
        calls.append((session.session_id, session.resume, prompt))
        session.on_session_id(session.session_id)
        _write_transcript(session.session_id)
        return ClaudeCodeTurnResult(
            final_text=f"answer {len(calls)}",
            session_id=session.session_id,
            session_confirmed=True,
        )

    monkeypatch.setattr(ClaudeCodeSession, "run_turn", _run_turn)
    db = SessionDB()
    outer_session_id = f"claude-appended-{uuid4()}"
    first = _make_agent(session_id=outer_session_id, session_db=db)
    first.run_conversation("Use Claude first.")
    history = db.get_messages_as_conversation(outer_session_id)
    history.extend(
        [
            {"role": "user", "content": "Use a different runtime."},
            {"role": "assistant", "content": "Other runtime answer."},
        ]
    )

    events = []
    second = _make_agent(session_id=outer_session_id, session_db=db)
    second.tool_progress_callback = lambda event, name, preview, args, **kw: events.append((event, preview, kw))
    second.run_conversation("Return to Claude.", conversation_history=history)

    assert calls[1][1] is True and calls[1][0] == calls[0][0]
    assert "Other runtime answer." in calls[1][2]
    assert "Use Claude first." not in calls[1][2]
    rows = [(preview, kw) for event, preview, kw in events if event == "session.continuity"]
    assert [kw["mode"] for _, kw in rows] == ["resumed"]
    assert rows[0][0].startswith(f"Session resumed from disk: Claude session {calls[0][0][:8]} (the CLI process was not running), plus the 2 messages")
    db.close()


def test_missing_transcript_file_rebuilds_instead_of_resuming(monkeypatch, _claude_config_dir):
    calls = []

    def _run_turn(session, prompt):
        calls.append((session.session_id, session.resume, prompt))
        session.on_session_id(session.session_id)
        return ClaudeCodeTurnResult(
            final_text=f"answer {len(calls)}",
            session_id=session.session_id,
            session_confirmed=True,
        )

    monkeypatch.setattr(ClaudeCodeSession, "run_turn", _run_turn)
    monkeypatch.setattr("agent.claude_runtime._RETENTION_CHECKED", False)
    db = SessionDB()
    outer_session_id = f"claude-swept-{uuid4()}"
    first = _make_agent(session_id=outer_session_id, session_db=db)
    first.run_conversation("Remember the heron.")
    history = db.get_messages_as_conversation(outer_session_id)

    events = []
    second = _make_agent(session_id=outer_session_id, session_db=db)
    second.tool_progress_callback = lambda event, name, preview, args, **kw: events.append((event, preview, kw))
    second.run_conversation("Which bird?", conversation_history=history)

    # No transcript was ever written, as after Claude Code's cleanup sweep:
    # a --resume would fail the turn, so the session is rebuilt instead.
    assert calls[1][1] is False and calls[1][0] != calls[0][0]
    assert "Will: Remember the heron." in calls[1][2]
    rows = [(preview, kw) for event, preview, kw in events if event == "session.continuity"]
    assert rows[0][1]["mode"] == "rebuilt"
    assert "its transcript file is gone" in rows[0][0]
    # The retention floor was raised before the first CLI started.
    import json as _json

    settings = _json.loads((_claude_config_dir / "settings.json").read_text())
    assert settings["cleanupPeriodDays"] == 3650
    db.close()
