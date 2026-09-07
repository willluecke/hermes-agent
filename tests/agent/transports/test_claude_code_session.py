from __future__ import annotations

import io
import json
import queue
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest

from agent.claude_runtime import make_claude_code_event_bridge
from agent.transports.claude_code_session import (
    CLAUDE_AUTH_ERROR_CODE,
    _ClaudeTranscriptPromptProbe,
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


def test_user_record_rejects_non_text_prompt_before_dispatch():
    with pytest.raises(ClaudeCodeError, match="prompts must be plain text"):
        ClaudeCodeSession._user_record(
            cast(str, [{"type": "text", "text": "malformed boundary"}])
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
                        "type": "user",
                        "session_id": session_id,
                        "message": {"role": "user", "content": "Continue"},
                    }
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
                    {
                        "type": "user",
                        "session_id": session_id,
                        "message": {"role": "user", "content": "one"},
                    }
                ),
                json.dumps(
                    {"type": "result", "session_id": session_id, "result": "first"}
                ),
                json.dumps(
                    {
                        "type": "user",
                        "session_id": session_id,
                        "message": {"role": "user", "content": "two"},
                    }
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


def test_rotated_oauth_generation_recycles_resident_process():
    session_id = "00000000-0000-4000-8000-000000000010"
    session = ClaudeCodeSession(
        cwd="/tmp",
        model="claude-fable-5",
        session_id=session_id,
        resume=True,
    )
    stale_process = _fake_process(12347)
    session._process = stale_process
    session._auth_generation = "stale-generation"
    fresh_process = _fake_process(12348)

    def retire(process):
        assert process is stale_process
        session._process = None
        return True

    def start():
        _install_fake_process(
            session,
            fresh_process,
            [
                {
                    "type": "user",
                    "session_id": session_id,
                    "message": {"role": "user", "content": "continue"},
                },
                {
                    "type": "result",
                    "session_id": session_id,
                    "result": "continued safely",
                },
            ],
        )
        session._auth_generation = "current-generation"
        return fresh_process

    with patch(
        "agent.transports.claude_code_session.claude_subscription_auth_available",
        return_value=True,
    ) as auth_available, patch(
        "agent.transports.claude_code_session._current_claude_auth_generation",
        return_value="current-generation",
    ), patch.object(session, "_retire_process", side_effect=retire) as retired, patch.object(
        session, "_start_process", side_effect=start
    ) as started:
        result = session.run_turn("continue")

    auth_available.assert_called_once_with(
        min_validity_seconds=session.absolute_timeout + 10 * 60
    )
    retired.assert_called_once_with(stale_process)
    started.assert_called_once_with()
    assert stale_process.stdin.getvalue() == ""
    assert fresh_process.stdin.getvalue()
    assert result.final_text == "continued safely"


def test_scheduled_wakeup_keeps_same_turn_open_until_authoritative_result():
    session_id = "00000000-0000-4000-8000-000000000000"
    forwarded = []
    session = ClaudeCodeSession(
        cwd="/tmp",
        model="claude-fable-5",
        session_id=session_id,
        resume=True,
        on_event=forwarded.append,
        inactivity_timeout=0.0,
    )
    process = _install_fake_process(
        session,
        _fake_process(20001),
        [
            {
                "type": "user",
                "session_id": session_id,
                "message": {"role": "user", "content": "Continue"},
            },
            {
                "type": "assistant",
                "session_id": session_id,
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_wakeup",
                            "name": "ScheduleWakeup",
                            "input": {"delaySeconds": 1200},
                        }
                    ]
                },
            },
            {
                "type": "user",
                "session_id": session_id,
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_wakeup",
                            "content": (
                                "Next wakeup scheduled. Nothing more to do this turn — "
                                "the harness re-invokes you when a task-notification arrives."
                            ),
                        }
                    ]
                },
            },
            {
                "type": "result",
                "session_id": session_id,
                "result": "",
                "usage": {"input_tokens": 10, "output_tokens": 20},
            },
            {
                "type": "system",
                "subtype": "task_notification",
                "session_id": session_id,
            },
            {
                "type": "result",
                "session_id": session_id,
                "result": "",
                "origin": {"kind": "task-notification"},
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
            {
                "type": "assistant",
                "session_id": session_id,
                "message": {"content": [{"type": "text", "text": "Resuming."}]},
            },
            {
                "type": "result",
                "session_id": session_id,
                "result": "Implemented the missing pieces.",
                "usage": {"input_tokens": 30, "output_tokens": 40},
            },
        ],
    )
    queued_output = session._output_queue

    class _DeferredGapQueue:
        calls = 0

        def get(self, timeout):
            del timeout
            if self.calls == 4:
                self.calls += 1
                raise queue.Empty
            self.calls += 1
            return queued_output.get_nowait()

    session._output_queue = cast(queue.Queue[str | None], _DeferredGapQueue())

    result = session.run_turn("Continue")

    assert process.stdin.getvalue()
    assert result.final_text == "Implemented the missing pieces."
    assert result.error is None
    assert result.should_retire is False
    assert result.usage["input_tokens"] == 40
    assert result.usage["output_tokens"] == 60
    assert [event["type"] for event in forwarded] == [
        "user",
        "assistant",
        "user",
        "result",
        "system",
        "result",
        "assistant",
        "result",
    ]


def test_async_agent_completion_stays_in_originating_turn():
    session_id = "00000000-0000-4000-8000-000000000000"
    tool_use_id = "toolu_background_agent"
    forwarded = []
    session = ClaudeCodeSession(
        cwd="/tmp",
        model="claude-fable-5",
        session_id=session_id,
        resume=True,
        on_event=forwarded.append,
        inactivity_timeout=0.0,
    )
    process = _install_fake_process(
        session,
        _fake_process(20003),
        [
            {
                "type": "user",
                "session_id": session_id,
                "message": {"role": "user", "content": "Run the checks"},
            },
            {
                "type": "assistant",
                "session_id": session_id,
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": tool_use_id,
                            "name": "Agent",
                            "input": {"description": "Check the implementation"},
                        }
                    ]
                },
            },
            {
                "type": "user",
                "session_id": session_id,
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "Async agent launched successfully. "
                                        "The agent is working in the background."
                                    ),
                                }
                            ],
                        }
                    ]
                },
            },
            {
                "type": "assistant",
                "session_id": session_id,
                "message": {
                    "content": [
                        {
                            "type": "text",
                            "text": "The check is still running.",
                        }
                    ]
                },
            },
            {
                "type": "result",
                "session_id": session_id,
                "result": "The check is still running.",
                "usage": {"input_tokens": 10, "output_tokens": 20},
            },
            {
                "type": "user",
                "session_id": session_id,
                "message": {
                    "role": "user",
                    "content": (
                        "<task-notification>\n"
                        "<task-id>agent-1</task-id>\n"
                        f"<tool-use-id>{tool_use_id}</tool-use-id>\n"
                        "<status>completed</status>\n"
                        "<result>Everything passed.</result>\n"
                        "</task-notification>"
                    ),
                },
            },
            {
                "type": "result",
                "session_id": session_id,
                "result": "",
                "origin": {"kind": "task-notification"},
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
            {
                "type": "assistant",
                "session_id": session_id,
                "message": {
                    "content": [
                        {
                            "type": "text",
                            "text": "All checks passed and the fix is complete.",
                        }
                    ]
                },
            },
            {
                "type": "result",
                "session_id": session_id,
                "result": "All checks passed and the fix is complete.",
                "usage": {"input_tokens": 30, "output_tokens": 40},
            },
        ],
    )

    result = session.run_turn("Run the checks")

    assert process.stdin.getvalue()
    assert result.final_text == "All checks passed and the fix is complete."
    assert result.error is None
    assert result.should_retire is False
    assert result.usage["input_tokens"] == 40
    assert result.usage["output_tokens"] == 60
    assert [event["type"] for event in forwarded] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "result",
        "user",
        "result",
        "assistant",
        "result",
    ]


def test_queued_command_completion_absent_from_stdout_releases_final(
    tmp_path, monkeypatch
):
    session_id = "00000000-0000-4000-8000-000000000006"
    tool_use_id = "toolu_queued_background_agent"
    prompt = "Run the checks"
    transcript = tmp_path / "projects" / "-tmp" / f"{session_id}.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.touch()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    session = ClaudeCodeSession(
        cwd="/tmp",
        model="claude-fable-5",
        session_id=session_id,
        resume=True,
    )
    process = _fake_process(20006)
    session._process = process
    events = [
        {
            "type": "user",
            "session_id": session_id,
            "message": {"role": "user", "content": prompt},
        },
        {
            "type": "assistant",
            "session_id": session_id,
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": tool_use_id,
                        "name": "Agent",
                        "input": {"description": "Check the implementation"},
                    }
                ]
            },
        },
        {
            "type": "user",
            "session_id": session_id,
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": "Async agent launched successfully.",
                    }
                ]
            },
        },
        {
            "type": "result",
            "session_id": session_id,
            "result": "All checks passed and the fix is complete.",
        },
    ]

    class _CompletionInTranscriptQueue:
        def get(self, timeout):
            del timeout
            event = events.pop(0)
            if event["type"] == "result":
                notification = {
                    "type": "attachment",
                    "attachment": {
                        "type": "queued_command",
                        "prompt": (
                            "<task-notification>\n"
                            f"<tool-use-id>{tool_use_id}</tool-use-id>\n"
                            "<status>completed</status>\n"
                            "</task-notification>"
                        ),
                    },
                }
                with transcript.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(notification) + "\n")
            return json.dumps(event) + "\n"

    session._output_queue = cast(
        queue.Queue[str | None], _CompletionInTranscriptQueue()
    )

    result = session.run_turn(prompt)

    assert process.stdin.getvalue()
    assert result.final_text == "All checks passed and the fix is complete."
    assert result.error is None
    assert result.should_retire is False


def test_stale_autonomous_output_cannot_claim_the_next_turn():
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
        _fake_process(20004),
        [
            {
                "type": "assistant",
                "session_id": session_id,
                "message": {
                    "content": [
                        {"type": "text", "text": "Two of three are complete."}
                    ]
                },
            },
            {
                "type": "result",
                "session_id": session_id,
                "result": "Two of three are complete.",
            },
            {
                "type": "user",
                "session_id": session_id,
                "message": {"role": "user", "content": "Did they get back?"},
            },
            {
                "type": "assistant",
                "session_id": session_id,
                "message": {
                    "content": [
                        {"type": "text", "text": "Yes, all three came back."}
                    ]
                },
            },
            {
                "type": "result",
                "session_id": session_id,
                "result": "Yes, all three came back.",
            },
        ],
    )

    result = session.run_turn("Did they get back?")

    assert process.stdin.getvalue()
    assert result.prompt_acknowledged is True
    assert result.final_text == "Yes, all three came back."
    assert [event["type"] for event in forwarded] == [
        "user",
        "assistant",
        "result",
    ]


def test_transcript_probe_only_accepts_prompt_appended_after_dispatch(
    tmp_path, monkeypatch
):
    session_id = "00000000-0000-4000-8000-000000000001"
    prompt = "Inspect the two images"
    transcript = tmp_path / "projects" / "-tmp" / f"{session_id}.jsonl"
    transcript.parent.mkdir(parents=True)
    record = {
        "type": "user",
        "session_id": session_id,
        "message": {"role": "user", "content": prompt},
    }
    transcript.write_text(json.dumps(record) + "\n", encoding="utf-8")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))

    probe = _ClaudeTranscriptPromptProbe.begin(session_id, prompt)

    assert probe.acknowledged() is False
    with transcript.open("a", encoding="utf-8") as output:
        output.write(json.dumps({**record, "message": {"content": "other"}}) + "\n")
    assert probe.acknowledged() is False
    with transcript.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record) + "\n")
    assert probe.acknowledged() is True


def test_durable_prompt_append_stops_watchdog_but_not_stream_quarantine(
    tmp_path, monkeypatch
):
    session_id = "00000000-0000-4000-8000-000000000002"
    prompt = "[2 images] Inspect this request\n" + ("A" * 350_000)
    transcript = tmp_path / "projects" / "-tmp" / f"{session_id}.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.touch()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    forwarded = []
    watchdog_events = []
    session = ClaudeCodeSession(
        cwd="/tmp",
        model="claude-fable-5",
        session_id=session_id,
        resume=True,
        on_event=forwarded.append,
        on_watchdog_timeout=watchdog_events.append,
        resident_first_event_timeout=0.0,
        inactivity_timeout=1.0,
    )
    process = _fake_process(20005)
    session._process = process
    events = [
        {
            "type": "result",
            "session_id": session_id,
            "result": "Stale autonomous result",
        },
        {
            "type": "user",
            "session_id": session_id,
            "message": {"role": "user", "content": prompt},
        },
        {
            "type": "result",
            "session_id": session_id,
            "result": "Current turn result",
        },
    ]

    class _TranscriptBeforeOutputQueue:
        first = True

        def get(self, timeout):
            del timeout
            if self.first:
                self.first = False
                with transcript.open("a", encoding="utf-8") as output:
                    output.write(
                        json.dumps(
                            {
                                "type": "user",
                                "session_id": session_id,
                                "message": {"role": "user", "content": prompt},
                            }
                        )
                        + "\n"
                    )
                raise queue.Empty
            return json.dumps(events.pop(0)) + "\n"

    session._output_queue = cast(queue.Queue[str | None], _TranscriptBeforeOutputQueue())

    with patch.object(session, "_start_process") as restart:
        result = session.run_turn(prompt)

    restart.assert_not_called()
    assert watchdog_events == []
    assert result.prompt_acknowledged is True
    assert result.final_text == "Current turn result"
    assert [event["type"] for event in forwarded] == ["user", "result"]


def test_empty_result_without_scheduled_wakeup_fails_closed():
    session_id = "00000000-0000-4000-8000-000000000000"
    session = ClaudeCodeSession(
        cwd="/tmp",
        model="claude-fable-5",
        session_id=session_id,
        resume=True,
    )
    _install_fake_process(
        session,
        _fake_process(20002),
        [
            {
                "type": "user",
                "session_id": session_id,
                "message": {"role": "user", "content": "Continue"},
            },
            {"type": "result", "session_id": session_id, "result": ""},
        ],
    )

    with patch.object(session, "close") as close:
        result = session.run_turn("Continue")

    assert result.final_text == ""
    assert result.error == "Claude Code completed without an authoritative final answer"
    assert result.should_retire is True
    close.assert_called_once_with()


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


def test_native_claude_max_status_satisfies_preflight_without_credentials_file(
    monkeypatch,
):
    from agent.claude_auth_lease import ClaudeAuthLease

    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "metered-api-key")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "metered-auth-token")
    monkeypatch.setenv("ANTHROPIC_TOKEN", "metered-oauth-token")
    native_status = SimpleNamespace(
        returncode=0,
        stdout=json.dumps(
            {
                "loggedIn": True,
                "authMethod": "claude.ai",
                "subscriptionType": "max",
            }
        ),
    )

    with patch(
        "agent.claude_auth_lease.acquire_claude_auth_lease",
        return_value=ClaudeAuthLease(available=False),
    ), patch(
        "agent.transports.claude_code_session.find_claude_binary",
        return_value="/usr/bin/claude",
    ), patch(
        "agent.transports.claude_code_session.subprocess.run",
        return_value=native_status,
    ) as run:
        from agent.transports.claude_code_session import (
            claude_subscription_auth_available,
        )

        assert claude_subscription_auth_available() is True

    assert run.call_args.args[0] == [
        "/usr/bin/claude",
        "auth",
        "status",
        "--json",
    ]
    probe_env = run.call_args.kwargs["env"]
    assert "ANTHROPIC_API_KEY" not in probe_env
    assert "ANTHROPIC_AUTH_TOKEN" not in probe_env
    assert "ANTHROPIC_TOKEN" not in probe_env


def test_native_api_key_status_does_not_satisfy_subscription_preflight(monkeypatch):
    from agent.claude_auth_lease import ClaudeAuthLease

    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    native_status = SimpleNamespace(
        returncode=0,
        stdout=json.dumps(
            {
                "loggedIn": True,
                "authMethod": "api_key",
                "subscriptionType": None,
            }
        ),
    )

    with patch(
        "agent.claude_auth_lease.acquire_claude_auth_lease",
        return_value=ClaudeAuthLease(available=False),
    ), patch(
        "agent.transports.claude_code_session.find_claude_binary",
        return_value="/usr/bin/claude",
    ), patch(
        "agent.transports.claude_code_session.subprocess.run",
        return_value=native_status,
    ):
        from agent.transports.claude_code_session import (
            claude_subscription_auth_available,
        )

        assert claude_subscription_auth_available() is False


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


def test_claude_code_args_include_debug_file_only_when_requested():
    from agent.transports.claude_code_session import claude_code_args

    args = claude_code_args(model="claude-opus-5", session_id="sid", debug_file="/tmp/claude-debug.log")
    assert args[args.index("--debug-file") + 1] == "/tmp/claude-debug.log"
    assert "--debug-file" not in claude_code_args(model="claude-opus-5", session_id="sid")


def test_next_debug_file_is_bounded_and_can_be_disabled(tmp_path, monkeypatch):
    import os
    import time
    from agent.transports.claude_code_session import ClaudeCodeSession

    monkeypatch.setenv("HERMES_CLAUDE_CODE_DEBUG_DIR", str(tmp_path))
    monkeypatch.delenv("HERMES_CLAUDE_CODE_DEBUG", raising=False)
    stale = tmp_path / "old-session.log"
    stale.write_text("old")
    os.utime(stale, (time.time() - 30 * 24 * 3600, time.time() - 30 * 24 * 3600))
    fresh = tmp_path / "recent-session.log"
    fresh.write_text("recent")
    session = ClaudeCodeSession.__new__(ClaudeCodeSession)
    session.session_id = "abc"
    session._debug_file = None
    path = session._next_debug_file()
    assert path is not None and path.startswith(str(tmp_path)) and "abc-" in path
    assert not stale.exists() and fresh.exists()
    monkeypatch.setenv("HERMES_CLAUDE_CODE_DEBUG", "0")
    assert session._next_debug_file() is None
