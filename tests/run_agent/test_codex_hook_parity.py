"""The codex app-server runtime fires the same observer hooks and verify gate as the default loop."""

from __future__ import annotations

from unittest.mock import patch

import pytest

import run_agent
from agent.transports.codex_app_server_session import CodexAppServerSession, TurnResult


def _agent():
    return run_agent.AIAgent(
        api_key="stub",
        base_url="https://stub.invalid",
        provider="openai",
        api_mode="codex_app_server",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )


@pytest.fixture
def hooks(monkeypatch):
    """Route lifecycle hooks to an in-test recorder and make pre_verify decisive."""
    seen = {"post_tool_call": [], "post_llm_call": [], "pre_verify": []}

    def has_hook(name):
        return name in seen

    def invoke_hook(name, **kwargs):
        seen.setdefault(name, []).append(kwargs)
        if name == "pre_llm_call":
            return [{"context": "Preflight from Jev: budget k=1."}]
        return []

    def continue_message(**kwargs):
        seen["pre_verify"].append(kwargs)
        return "Fix criterion 2, then finish." if kwargs["attempt"] == 0 else None

    import hermes_cli.lifecycle as lifecycle
    import hermes_cli.plugins as plugins
    import agent.verify_hooks as verify_hooks

    monkeypatch.setattr(lifecycle, "has_hook", has_hook)
    monkeypatch.setattr(lifecycle, "invoke_hook", invoke_hook)
    monkeypatch.setattr(plugins, "get_pre_verify_continue_message", continue_message)
    monkeypatch.setattr(verify_hooks, "max_verify_nudges", lambda config=None: 2)
    monkeypatch.setattr(CodexAppServerSession, "ensure_started", lambda self: "thread-stub-1")
    return seen


def _turn(final, changed_path=None, command=None, mcp_todo=None):
    projected = []
    if changed_path:
        projected += [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "patch_1", "type": "function", "function": {"name": "apply_patch", "arguments": f'{{"changes": [{{"kind": "update", "path": "{changed_path}"}}]}}'}}]},
            {"role": "tool", "tool_call_id": "patch_1", "content": "apply_patch status=completed, 1 change(s)"},
        ]
    if command:
        projected += [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "exec_1", "type": "function", "function": {"name": "exec_command", "arguments": f'{{"command": "{command}", "cwd": "/tmp"}}'}}]},
            {"role": "tool", "tool_call_id": "exec_1", "content": "[exit 1]\n1 failed"},
        ]
    if mcp_todo:
        projected += [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "mcp_1", "type": "function", "function": {"name": "mcp.hermes-tools.todo", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "mcp_1", "content": mcp_todo},
        ]
    projected.append({"role": "assistant", "content": final})
    return TurnResult(final_text=final, projected_messages=projected, tool_iterations=len(projected) // 2, turn_id="turn-stub-1", thread_id="thread-stub-1")


def test_codex_turn_emits_tool_hooks_runs_the_verify_gate_and_continues_once(hooks, monkeypatch, tmp_path):
    changed = tmp_path / "app.py"
    changed.write_text("print('x')\n")
    inputs = []

    def fake_run_turn(self, user_input, **kwargs):
        inputs.append(user_input)
        if len(inputs) == 1:
            return _turn("Done, tests pass.", changed_path=str(changed), command="pytest -q", mcp_todo='{"todos": [{"id": "1", "content": "A", "status": "completed"}]}')
        return _turn("Fixed criterion 2 and re-ran the tests.", command="pytest -q")

    monkeypatch.setattr(CodexAppServerSession, "run_turn", fake_run_turn)
    agent = _agent()
    with patch.object(agent, "_spawn_background_review", return_value=None):
        result = agent.run_conversation("add the feature")

    assert len(inputs) == 2, "the verify gate sent exactly one follow-up turn"
    assert inputs[0].startswith("add the feature"), "the user's message leads the turn input"
    assert inputs[0].endswith("\n\nPreflight from Jev: budget k=1."), "the pre_llm_call note reaches the codex turn input"
    assert inputs[1] == "Fix criterion 2, then finish.", "a verify nudge carries no preflight note"
    assert result["final_response"] == "Fixed criterion 2 and re-ran the tests."
    assert result["completed"] is True
    assert result["api_calls"] == 2

    names = [call["tool_name"] for call in hooks["post_tool_call"]]
    assert names == ["patch", "terminal", "todo", "terminal"], "projected codex calls arrive under Hermes tool names"
    assert hooks["post_tool_call"][1]["args"]["command"] == "pytest -q"
    assert hooks["post_tool_call"][1]["result"].startswith("[exit 1]")
    assert '"todos"' in hooks["post_tool_call"][2]["result"]

    assert len(hooks["pre_verify"]) == 2
    assert hooks["pre_verify"][0]["changed_paths"] == [str(changed.resolve())]
    assert hooks["pre_verify"][0]["final_response"] == "Done, tests pass."
    assert hooks["pre_verify"][1]["attempt"] == 1

    assert len(hooks["post_llm_call"]) == 1
    assert hooks["post_llm_call"][0]["assistant_response"] == "Fixed criterion 2 and re-ran the tests."
    synthetic = [m for m in result["messages"] if isinstance(m, dict) and m.get("_pre_verify_synthetic")]
    assert len(synthetic) == 1 and synthetic[0]["content"] == "Fix criterion 2, then finish."


def test_codex_turn_without_file_changes_finishes_without_a_verify_gate(hooks, monkeypatch):
    inputs = []

    def fake_run_turn(self, user_input, **kwargs):
        inputs.append(user_input)
        return _turn("Just an answer.", command="ls")

    monkeypatch.setattr(CodexAppServerSession, "run_turn", fake_run_turn)
    agent = _agent()
    with patch.object(agent, "_spawn_background_review", return_value=None):
        result = agent.run_conversation("what is here?")
    assert len(inputs) == 1
    assert hooks["pre_verify"] == []
    assert [call["tool_name"] for call in hooks["post_tool_call"]] == ["terminal"]
    assert len(hooks["post_llm_call"]) == 1
    assert result["api_calls"] == 1
