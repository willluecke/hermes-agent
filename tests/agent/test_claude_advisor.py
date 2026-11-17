"""Claude Code's advisor tool on the Hermes claude-code lane.

Hermes Chat lets a person pick an advisor for one Claude conversation; the
gateway hands it over as ``model_options.advisor`` and the session launches
with ``claude --advisor <model>``. It must never appear anywhere it was not
explicitly passed (workers, delegations, consults), never for the steward or
a read-only session, and changing it must restart the conversation's Claude
process with ``--resume`` so the conversation keeps its context.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest

import run_agent
from agent.transports import claude_code_session as ccs
from agent.transports.claude_code_session import (
    ClaudeCodeSession,
    ClaudeCodeTurnResult,
    claude_code_args,
    clean_claude_advisor,
    launch_advisor,
)
from gateway.platforms import api_server
from gateway.platforms.api_server import APIServerAdapter
from hermes_state import SessionDB

SESSION = "00000000-0000-4000-8000-000000000000"


def _flag(args, name):
    return args[args.index(name) + 1] if name in args else None


def _settings_documents(args):
    return [args[i + 1] for i, arg in enumerate(args) if arg == "--settings"]


# ── Launch arguments ────────────────────────────────────────────────────────


def test_no_advisor_unless_one_is_passed():
    args = claude_code_args(model="claude-opus-5-5", session_id=SESSION)
    assert "--advisor" not in args


def test_a_passed_advisor_reaches_the_launch():
    args = claude_code_args(model="claude-opus-5-5", session_id=SESSION, advisor="claude-fable-5-2")
    assert _flag(args, "--advisor") == "claude-fable-5-2"
    assert _flag(args, "--model") == "claude-opus-5-5"


def test_a_resumed_launch_keeps_the_advisor_and_the_session():
    args = claude_code_args(
        model="claude-opus-5-5", session_id=SESSION, resume=True, effort="high", advisor="claude-opus-5-5"
    )
    assert _flag(args, "--advisor") == "claude-opus-5-5"
    assert _flag(args, "--resume") == SESSION
    assert "--session-id" not in args


@pytest.mark.parametrize("posture", [{"steward": True}, {"read_only": True}, {"no_tools": True}])
def test_the_steward_and_read_only_sessions_never_launch_one(posture):
    args = claude_code_args(model="claude-opus-5-5", session_id=SESSION, advisor="claude-fable-5-2", **posture)
    assert "--advisor" not in args


@pytest.mark.parametrize(
    "value, expected",
    [
        ("opus", "opus"),
        ("fable", "fable"),
        ("sonnet", "sonnet"),
        (" Claude-Fable-5-2 ", "claude-fable-5-2"),
        ("claude-opus-5-5", "claude-opus-5-5"),
        ("", None),
        (None, None),
        (5, None),
        ("haiku 4.5", None),
        ("--model", None),
        ("opus;rm -rf /", None),
        ("claude-" + "x" * 80, None),
    ],
)
def test_only_an_alias_or_a_claude_model_id_is_accepted(value, expected):
    assert clean_claude_advisor(value) == expected
    args = claude_code_args(model="claude-opus-5-5", session_id=SESSION, advisor=value)
    assert _flag(args, "--advisor") == expected


def test_no_settings_document_ever_carries_advisor_model():
    # The flag is per session; a global advisorModel would follow every
    # Claude Code session on the machine, including workers.
    for posture in ({}, {"steward": True}, {"read_only": True}):
        args = claude_code_args(
            model="claude-opus-5-5", session_id=SESSION, advisor="claude-fable-5-2",
            settings_json=json.dumps({"hooks": {}}), **posture,
        )
        for document in _settings_documents(args):
            assert "advisorModel" not in json.loads(document)


def test_no_hermes_source_writes_an_advisor_model_setting():
    root = Path(__file__).resolve().parents[2]
    literal = re.compile(r"""["']advisorModel["']""")
    offenders = [
        str(path.relative_to(root))
        for folder in ("agent", "gateway", "hermes_cli", "tools")
        for path in (root / folder).rglob("*.py")
        if literal.search(path.read_text(encoding="utf-8", errors="ignore"))
    ]
    assert offenders == []


# ── The resident session ────────────────────────────────────────────────────


def test_the_session_launches_with_its_advisor(monkeypatch, tmp_path):
    launched = []

    class _Stop(Exception):
        pass

    def _popen(argv, **_kwargs):
        launched.append(argv)
        raise _Stop

    monkeypatch.setattr(ccs, "find_claude_binary", lambda: "/usr/bin/claude")
    monkeypatch.setattr(ccs, "claude_subscription_auth_available", lambda **_kw: True)
    monkeypatch.setattr(ccs, "claude_subscription_env", lambda: {})
    monkeypatch.setattr(ccs.subprocess, "Popen", _popen)
    for advisor in ("claude-fable-5-2", None):
        session = ClaudeCodeSession(cwd=str(tmp_path), model="claude-opus-5-5", advisor=advisor)
        session.late_answer_log = None
        with pytest.raises(_Stop):
            session._start_process()
    assert _flag(launched[0], "--advisor") == "claude-fable-5-2"
    assert "--advisor" not in launched[1]


def test_a_changed_advisor_makes_the_resident_process_unusable(tmp_path):
    session = ClaudeCodeSession(cwd=str(tmp_path), model="claude-opus-5-5", advisor="claude-opus-5-5")
    same = dict(cwd=str(tmp_path), model="claude-opus-5-5", effort=None, system_prompt="")
    assert session.compatible_with(**same, advisor="claude-opus-5-5")
    assert not session.compatible_with(**same, advisor="claude-fable-5-2")
    assert not session.compatible_with(**same, advisor=None)
    plain = ClaudeCodeSession(cwd=str(tmp_path), model="claude-opus-5-5")
    assert plain.advisor is None
    assert plain.compatible_with(**same)
    assert not plain.compatible_with(**same, advisor="claude-opus-5-5")


@pytest.mark.parametrize("posture", [{"steward": True}, {"read_only": True}, {"no_tools": True}])
def test_restricted_sessions_drop_a_passed_advisor(tmp_path, posture):
    session = ClaudeCodeSession(cwd=str(tmp_path), model="claude-opus-5-5", advisor="claude-fable-5-2", **posture)
    assert session.advisor is None
    assert launch_advisor("claude-fable-5-2", **posture) is None


# ── The runtime: per conversation, and a change resumes the same session ────


@pytest.fixture
def claude_config_dir(tmp_path, monkeypatch):
    root = tmp_path / "claude-config"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(root))
    return root


def _write_transcript(session_id):
    path = Path(os.environ["CLAUDE_CONFIG_DIR"]) / "projects" / "-work" / f"{session_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n", encoding="utf-8")


def _make_agent(*, session_id=None, session_db=None, model="claude-opus-5-5"):
    return run_agent.AIAgent(
        api_key="claude-cli-subscription-auth",
        base_url="claude-code://local",
        provider="claude-code",
        api_mode="claude_code",
        model=model,
        session_id=session_id,
        session_db=session_db,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        skip_background_review=True,
    )


def _record_turns(monkeypatch):
    calls = []

    def _run_turn(session, prompt):
        calls.append({
            "session": session, "session_id": session.session_id, "resume": session.resume,
            "advisor": session.advisor, "prompt": prompt,
        })
        session.on_session_id(session.session_id)
        _write_transcript(session.session_id)
        return ClaudeCodeTurnResult(
            final_text=f"answer {len(calls)}", session_id=session.session_id, session_confirmed=True,
        )

    monkeypatch.setattr(ClaudeCodeSession, "run_turn", _run_turn)
    return calls


def _run(agent, text, **kwargs):
    with patch.object(agent, "_spawn_background_review", return_value=None):
        return agent.run_conversation(text, **kwargs)


def test_an_agent_built_without_one_runs_no_advisor(monkeypatch, claude_config_dir):
    # Kanban workers, Talos delegations and every other caller that does not
    # pass an advisor build the agent this way.
    calls = _record_turns(monkeypatch)
    _run(_make_agent(), "Work the card.")
    assert calls[0]["advisor"] is None


def test_the_conversations_advisor_reaches_its_claude_session(monkeypatch, claude_config_dir):
    calls = _record_turns(monkeypatch)
    agent = _make_agent()
    agent.claude_advisor = "claude-fable-5-2"
    _run(agent, "Plan the refactor.")
    assert calls[0]["advisor"] == "claude-fable-5-2"


def test_a_read_only_agent_runs_no_advisor(monkeypatch, claude_config_dir):
    calls = _record_turns(monkeypatch)
    agent = _make_agent()
    agent.read_only = True
    agent.claude_advisor = "claude-fable-5-2"
    _run(agent, "Inspect only.")
    assert calls[0]["advisor"] is None


def test_the_steward_runs_no_advisor(monkeypatch, claude_config_dir, tmp_path):
    from agent import steward_policy

    docs = tmp_path / ".hermes" / "talos-steward" / "docs"
    docs.mkdir(parents=True)
    (docs / "steward-instructions.md").write_text("You are Jim.")
    monkeypatch.setattr(steward_policy.Path, "home", classmethod(lambda cls: tmp_path))
    calls = _record_turns(monkeypatch)
    agent = _make_agent(session_id="hermes-chat-talos")
    agent.session_project = "talos-steward"
    agent.claude_advisor = "claude-fable-5-2"
    _run(agent, "Status?")
    assert calls[0]["session"].steward is True
    assert calls[0]["advisor"] is None


def test_changing_the_advisor_restarts_claude_with_resume_on_the_same_session(monkeypatch, claude_config_dir):
    calls = _record_turns(monkeypatch)
    db = SessionDB()
    outer = f"claude-advisor-{uuid4()}"
    agent = _make_agent(session_id=outer, session_db=db)
    _run(agent, "Start the repair.")

    agent.claude_advisor = "claude-opus-5-5"
    events = []
    agent.tool_progress_callback = lambda event, name, preview, args, **kw: events.append((event, preview, kw))
    _run(agent, "Continue with an advisor.", conversation_history=db.get_messages_as_conversation(outer))

    first, second = calls
    assert first["advisor"] is None and second["advisor"] == "claude-opus-5-5"
    assert second["session"] is not first["session"], "a new process carries the new flag"
    assert second["resume"] is True and second["session_id"] == first["session_id"], "--resume keeps the context"
    assert second["prompt"] == "Continue with an advisor."
    rows = [preview for event, preview, kw in events if event == "session.continuity"]
    assert rows and "(the model, effort or advisor changed)" in rows[0]

    # Unchanged on the next turn: the same resident process takes it.
    _run(agent, "And again.", conversation_history=db.get_messages_as_conversation(outer))
    assert calls[2]["session"] is second["session"]
    db.close()


def test_a_new_agent_for_a_changed_advisor_resumes_the_same_claude_session(monkeypatch, claude_config_dir):
    # What the gateway does: a changed advisor is a changed settings signature,
    # so the conversation's next turn gets a new agent carrying the reason.
    calls = _record_turns(monkeypatch)
    db = SessionDB()
    outer = f"claude-advisor-agent-{uuid4()}"
    _run(_make_agent(session_id=outer, session_db=db), "First message.")
    replacement = _make_agent(session_id=outer, session_db=db)
    replacement.claude_advisor = "claude-fable-5-2"
    replacement._resume_reason = "after the advisor changed"
    events = []
    replacement.tool_progress_callback = lambda event, name, preview, args, **kw: events.append((event, preview, kw))
    _run(replacement, "Second message.", conversation_history=db.get_messages_as_conversation(outer))
    assert calls[1]["resume"] is True and calls[1]["session_id"] == calls[0]["session_id"]
    assert calls[1]["advisor"] == "claude-fable-5-2"
    rows = [preview for event, preview, kw in events if event == "session.continuity"]
    assert rows == [f"Session resumed from disk: Claude session {calls[0]['session_id'][:8]} (after the advisor changed)."]
    db.close()


# ── The gateway request ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "options, expected",
    [
        ({"advisor": "claude-fable-5-2", "reasoning_effort": "high"}, "claude-fable-5-2"),
        ({"advisor": "opus"}, "opus"),
        ({"reasoning_effort": "high"}, None),
        ({"advisor": "not a model"}, None),
        ({"advisor": None}, None),
        (None, None),
        ("advisor", None),
    ],
)
def test_the_gateway_reads_the_advisor_from_model_options(options, expected):
    assert api_server._request_claude_advisor(options) == expected


def _signature(model_options):
    return APIServerAdapter._runtime_request_signature(
        profile=None, ephemeral_system_prompt="instructions", requested_model="claude-opus-5-5",
        requested_provider="claude-code", model_options=model_options, route=None,
        session_model=None, confirmed_runtime_lock=True, single_model=True, cwd="/work", project="p",
    )


def test_an_advisor_change_is_a_settings_change_named_as_the_advisor():
    off = _signature({"reasoning_effort": "high"})
    on = _signature({"reasoning_effort": "high", "advisor": "claude-fable-5-2"})
    other = _signature({"reasoning_effort": "high", "advisor": "claude-opus-5-5"})
    assert len({off, on, other}) == 3
    assert api_server._changed_signature_inputs(off, on) == ["advisor"]
    assert api_server._changed_signature_inputs(on, other) == ["advisor"]
    assert api_server._settings_change_reason(["advisor"]) == "after the advisor changed"
    assert _signature({"reasoning_effort": "high", "advisor": "claude-fable-5-2"}) == on, "stable"


def test_an_effort_change_is_still_named_as_the_effort():
    high = _signature({"reasoning_effort": "high", "advisor": "claude-fable-5-2"})
    top = _signature({"reasoning_effort": "max", "advisor": "claude-fable-5-2"})
    assert api_server._changed_signature_inputs(high, top) == ["model_options"]
