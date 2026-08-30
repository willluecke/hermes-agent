from __future__ import annotations

from uuid import uuid4
from unittest.mock import patch

import run_agent
from agent.transports.claude_code_session import (
    CLAUDE_AUTH_ERROR_CODE,
    ClaudeCodeError,
    ClaudeCodeSession,
    ClaudeCodeTurnResult,
)
from hermes_state import SessionDB


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
    history.extend(
        [
            {"role": "user", "content": "Use a different runtime."},
            {"role": "assistant", "content": "Other runtime answer."},
        ]
    )

    second = _make_agent(session_id=outer_session_id, session_db=db)
    second.run_conversation("Return to Claude.", conversation_history=history)

    assert calls[1][1] is False
    assert calls[1][0] != calls[0][0]
    assert "Recent transcript:" in calls[1][2]
    assert "Other runtime answer." in calls[1][2]
    db.close()
