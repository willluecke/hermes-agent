from __future__ import annotations

import io
import json
from types import SimpleNamespace
from unittest.mock import patch

from agent.claude_runtime import make_claude_code_event_bridge
from agent.transports.claude_code_session import ClaudeCodeSession, claude_code_args


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
