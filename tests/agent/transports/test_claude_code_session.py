from __future__ import annotations

import json
from types import SimpleNamespace

from agent.claude_runtime import make_claude_code_event_bridge
from agent.transports.claude_code_session import claude_code_args


def test_invocation_uses_subscription_model_and_hermes_mcp(tmp_path):
    args = claude_code_args(
        model="claude-fable-5",
        session_id="00000000-0000-4000-8000-000000000000",
        system_prompt="Hermes runtime contract",
        additional_dirs=[str(tmp_path)],
    )

    assert args[args.index("--model") + 1] == "claude-fable-5"
    assert args[args.index("--permission-mode") + 1] == "bypassPermissions"
    assert args[args.index("--append-system-prompt") + 1] == "Hermes runtime contract"
    assert args[args.index("--add-dir") + 1] == str(tmp_path)
    config = json.loads(args[args.index("--mcp-config") + 1])
    assert "hermes-tools" in config["mcpServers"]


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
