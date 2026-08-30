from __future__ import annotations

import io
import json
import queue
from types import SimpleNamespace
from unittest.mock import patch

from agent.claude_runtime import make_claude_code_event_bridge
from agent.transports.claude_code_session import (
    CLAUDE_AUTH_ERROR_CODE,
    ClaudeCodeError,
    ClaudeCodeSession,
    claude_code_args,
)


def _install_fake_process(session, process, events):
    output = queue.Queue()
    for event in events:
        output.put(json.dumps(event) + "\n")
    session._process = process
    session._output_queue = output
    return process


def _fake_process(pid):
    return SimpleNamespace(
        pid=pid,
        stdin=io.StringIO(),
        poll=lambda: None,
    )


def test_invocation_uses_subscription_model_and_hermes_mcp(tmp_path):
    args = claude_code_args(
        model="claude-fable-5",
        session_id="00000000-0000-4000-8000-000000000000",
        system_prompt="Hermes runtime contract",
        additional_dirs=[str(tmp_path)],
    )

    assert args[args.index("--model") + 1] == "claude-fable-5"
    assert args[args.index("--input-format") + 1] == "stream-json"
    assert "--replay-user-messages" in args
    assert args[args.index("--permission-mode") + 1] == "bypassPermissions"
    assert args[args.index("--session-id") + 1] == "00000000-0000-4000-8000-000000000000"
    assert "--resume" not in args
    assert "--effort" not in args
    assert args[args.index("--append-system-prompt") + 1] == "Hermes runtime contract"
    assert args[args.index("--add-dir") + 1] == str(tmp_path)
    config = json.loads(args[args.index("--mcp-config") + 1])
    assert "hermes-tools" in config["mcpServers"]


def test_resume_invocation_is_exclusive_and_uses_selected_effort():
    args = claude_code_args(
        model="claude-fable-5",
        session_id="00000000-0000-4000-8000-000000000000",
        resume=True,
        effort="medium",
    )

    assert args[args.index("--resume") + 1] == "00000000-0000-4000-8000-000000000000"
    assert "--session-id" not in args
    assert args[args.index("--effort") + 1] == "medium"


def test_read_only_invocation_uses_plan_mode_and_no_mcp_servers():
    args = claude_code_args(
        model="claude-fable-5",
        session_id="00000000-0000-4000-8000-000000000000",
        read_only=True,
    )

    assert args[args.index("--permission-mode") + 1] == "plan"
    assert "--strict-mcp-config" in args
    assert "--safe-mode" in args
    assert json.loads(args[args.index("--mcp-config") + 1]) == {"mcpServers": {}}


def test_error_result_confirms_session_before_returning():
    session_id = "00000000-0000-4000-8000-000000000000"
    stdout = io.StringIO(
        "\n".join(
            [
                json.dumps(
                    {"type": "system", "subtype": "init", "session_id": session_id}
                ),
                json.dumps(
                    {
                        "type": "result",
                        "session_id": session_id,
                        "is_error": True,
                        "result": "You've hit your session limit",
                    }
                ),
            ]
        )
        + "\n"
    )
    process = SimpleNamespace(
        pid=12345,
        stdin=io.StringIO(),
        stdout=stdout,
        stderr=io.StringIO(),
        wait=lambda timeout=None: 0,
        poll=lambda: 0,
    )
    confirmed = []
    session = ClaudeCodeSession(
        cwd="/tmp",
        model="claude-fable-5",
        session_id=session_id,
        on_session_id=confirmed.append,
    )

    with patch(
        "agent.transports.claude_code_session.find_claude_binary",
        return_value="/usr/bin/claude",
    ), patch(
        "agent.transports.claude_code_session.claude_subscription_auth_available",
        return_value=True,
    ), patch(
        "agent.transports.claude_code_session.subprocess.Popen",
        return_value=process,
    ):
        result = session.run_turn("Continue")

    assert confirmed == [session_id]
    assert result.session_confirmed is True
    assert result.session_id == session_id
    assert result.error == "You've hit your session limit"


def test_two_turns_share_one_streaming_process():
    session_id = "00000000-0000-4000-8000-000000000000"
    stdout = io.StringIO(
        "\n".join(
            [
                json.dumps(
                    {"type": "system", "subtype": "init", "session_id": session_id}
                ),
                json.dumps(
                    {"type": "result", "session_id": session_id, "result": "first"}
                ),
                json.dumps(
                    {"type": "result", "session_id": session_id, "result": "second"}
                ),
            ]
        )
        + "\n"
    )
    stdin = io.StringIO()
    process = SimpleNamespace(
        pid=12346,
        stdin=stdin,
        stdout=stdout,
        stderr=io.StringIO(),
        wait=lambda timeout=None: 0,
        poll=lambda: None,
        send_signal=lambda _sig: None,
    )
    session = ClaudeCodeSession(cwd="/tmp", model="claude-fable-5")

    with patch(
        "agent.transports.claude_code_session.find_claude_binary",
        return_value="/usr/bin/claude",
    ), patch(
        "agent.transports.claude_code_session.claude_subscription_auth_available",
        return_value=True,
    ), patch(
        "agent.transports.claude_code_session.subprocess.Popen",
        return_value=process,
    ) as popen, patch(
        "agent.transports.claude_code_session.os.killpg"
    ):
        first = session.run_turn("one")
        second = session.run_turn("two")
        assert session.pid == process.pid
        session.close()

    assert first.final_text == "first"
    assert second.final_text == "second"
    popen.assert_called_once()
    records = [json.loads(line) for line in stdin.getvalue().splitlines()]
    assert [record["message"]["content"] for record in records] == ["one", "two"]


def test_new_process_rejects_missing_subscription_auth_before_spawn():
    session = ClaudeCodeSession(cwd="/tmp", model="claude-fable-5")

    with patch(
        "agent.transports.claude_code_session.find_claude_binary",
        return_value="/usr/bin/claude",
    ), patch(
        "agent.transports.claude_code_session.claude_subscription_auth_available",
        return_value=False,
    ), patch(
        "agent.transports.claude_code_session.subprocess.Popen",
    ) as popen:
        try:
            session.run_turn("continue")
        except ClaudeCodeError as exc:
            assert exc.error_code == CLAUDE_AUTH_ERROR_CODE
            assert "claude auth login --claudeai" in str(exc)
        else:
            raise AssertionError("missing subscription auth must fail closed")

    popen.assert_not_called()


def test_explicit_claude_oauth_override_satisfies_subscription_preflight(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "subscription-oauth-token")

    with patch(
        "agent.anthropic_adapter.read_claude_code_credentials"
    ) as read_credentials:
        from agent.transports.claude_code_session import (
            claude_subscription_auth_available,
        )

        assert claude_subscription_auth_available() is True

    read_credentials.assert_not_called()


def test_auth_error_is_terminal_and_not_forwarded_as_commentary():
    session_id = "00000000-0000-4000-8000-000000000000"
    forwarded = []
    session = ClaudeCodeSession(
        cwd="/tmp",
        model="claude-fable-5",
        session_id=session_id,
        resume=True,
        on_event=forwarded.append,
    )
    process = _install_fake_process(
        session,
        _fake_process(30001),
        [
            {
                "type": "assistant",
                "session_id": session_id,
                "error": "authentication_failed",
                "isApiErrorMessage": True,
                "message": {
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Failed to authenticate: OAuth session expired "
                                "and could not be refreshed"
                            ),
                        }
                    ]
                },
            }
        ],
    )

    with patch.object(session, "close") as close:
        result = session.run_turn("continue")

    assert process.stdin.getvalue()
    assert result.error_code == CLAUDE_AUTH_ERROR_CODE
    assert "OAuth session expired" in result.error
    assert "claude auth login --claudeai" in result.error
    assert result.should_retire is True
    assert result.prompt_acknowledged is True
    assert result.session_confirmed is False
    assert result.final_text == ""
    assert forwarded == []
    close.assert_called_once_with()


def test_first_event_watchdog_restarts_unacknowledged_resident_once():
    session_id = "00000000-0000-4000-8000-000000000000"
    watchdog_events = []
    confirmed = []
    session = ClaudeCodeSession(
        cwd="/tmp",
        model="claude-fable-5",
        session_id=session_id,
        resume=True,
        on_session_id=confirmed.append,
        on_watchdog_timeout=watchdog_events.append,
        resident_first_event_timeout=0.03,
        startup_first_event_timeout=0.2,
        inactivity_timeout=1.0,
    )
    stalled = _install_fake_process(
        session,
        _fake_process(31001),
        [
            {
                "type": "system",
                "subtype": "init",
                "session_id": session_id,
            }
        ],
    )
    healthy = _fake_process(31002)

    def _retire(process, *, grace_seconds=5.0):
        assert grace_seconds > 0
        if session._process is process:
            session._process = None
        return True

    def _restart():
        return _install_fake_process(
            session,
            healthy,
            [
                {
                    "type": "user",
                    "session_id": session_id,
                    "message": {"role": "user", "content": "repair"},
                },
                {
                    "type": "result",
                    "session_id": session_id,
                    "result": "recovered",
                },
            ],
        )

    with patch.object(session, "_retire_process", side_effect=_retire), patch.object(
        session, "_start_process", side_effect=_restart
    ) as start_process:
        result = session.run_turn("repair")

    assert result.final_text == "recovered"
    assert result.prompt_acknowledged is True
    assert result.watchdog_retries == 1
    assert result.error_code is None
    assert confirmed == [session_id]
    assert watchdog_events == [
        {
            "code": "claude_first_event_timeout",
            "attempt": 1,
            "retrying": True,
            "timeout_seconds": 0.03,
            "session_id": session_id,
        }
    ]
    start_process.assert_called_once_with()
    assert json.loads(stalled.stdin.getvalue())["message"]["content"] == "repair"
    assert json.loads(healthy.stdin.getvalue())["message"]["content"] == "repair"


def test_first_event_watchdog_fails_after_one_fresh_process_retry():
    session_id = "00000000-0000-4000-8000-000000000000"
    watchdog_events = []
    confirmed = []
    session = ClaudeCodeSession(
        cwd="/tmp",
        model="claude-fable-5",
        session_id=session_id,
        resume=True,
        on_session_id=confirmed.append,
        on_watchdog_timeout=watchdog_events.append,
        resident_first_event_timeout=0.03,
        startup_first_event_timeout=0.03,
        inactivity_timeout=1.0,
    )
    stalled = _install_fake_process(session, _fake_process(32001), [])
    retry = _fake_process(32002)

    def _retire(process, *, grace_seconds=5.0):
        assert grace_seconds > 0
        if session._process is process:
            session._process = None
        return True

    def _restart():
        return _install_fake_process(
            session,
            retry,
            [
                {
                    "type": "system",
                    "subtype": "init",
                    "session_id": session_id,
                }
            ],
        )

    with patch.object(session, "_retire_process", side_effect=_retire), patch.object(
        session, "_start_process", side_effect=_restart
    ):
        result = session.run_turn("repair")

    assert result.final_text == ""
    assert result.error_code == "claude_first_event_timeout"
    assert result.should_retire is True
    assert result.prompt_acknowledged is False
    assert result.session_confirmed is False
    assert result.watchdog_retries == 1
    assert confirmed == []
    assert [event["retrying"] for event in watchdog_events] == [True, False]
    assert [event["attempt"] for event in watchdog_events] == [1, 2]
    assert json.loads(stalled.stdin.getvalue())["message"]["content"] == "repair"
    assert json.loads(retry.stdin.getvalue())["message"]["content"] == "repair"


def test_event_bridge_keeps_commentary_tool_and_final_channels_separate():
    activity = []
    commentary = []
    agent = SimpleNamespace(
        show_commentary=True,
        tool_progress_callback=lambda *args, **kwargs: activity.append((args, kwargs)),
        tool_start_callback=lambda *args: activity.append((("start", *args), {})),
        tool_complete_callback=lambda *args: activity.append((("complete", *args), {})),
        _emit_interim_assistant_message=lambda message: commentary.append(message["content"]),
        _fire_reasoning_delta=lambda _text: None,
    )
    bridge = make_claude_code_event_bridge(agent)

    bridge({
        "type": "assistant",
        "message": {"content": [
            {"type": "text", "text": "I will inspect the repository."},
            {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "git status"}},
        ]},
    })
    bridge({
        "type": "user",
        "message": {"content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "clean", "is_error": False}
        ]},
    })

    assert commentary == ["I will inspect the repository."]
    assert any(args[0] == "tool.started" and args[1] == "exec_command" for args, _ in activity)
    assert any(args[0] == "tool.completed" and args[1] == "exec_command" for args, _ in activity)
