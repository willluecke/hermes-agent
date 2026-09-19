"""The Claude Code runtime fires the same observer hooks and verify gate as the default loop."""

from __future__ import annotations

import json

import pytest

import run_agent
from agent.transports.claude_code_session import ClaudeCodeSession, ClaudeCodeTurnResult


def _agent():
    return run_agent.AIAgent(
        api_key="claude-cli-subscription-auth",
        base_url="claude-code://local",
        provider="claude-code",
        api_mode="claude_code",
        model="claude-fable-5",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        skip_background_review=True,
    )


@pytest.fixture
def hooks(monkeypatch):
    """Route lifecycle hooks to an in-test recorder and make pre_verify decisive."""
    seen = {"post_tool_call": [], "post_llm_call": [], "pre_verify": [], "pre_llm_call": []}

    def has_hook(name):
        return name in seen

    def invoke_hook(name, **kwargs):
        seen.setdefault(name, []).append(kwargs)
        if name == "pre_llm_call":
            return [{"context": "Preflight from Jev: k=3 candidates."}]
        return []

    def continue_message(**kwargs):
        seen["pre_verify"].append(kwargs)
        return "Fix criterion 2, then finish." if kwargs["attempt"] == 0 else None

    import agent.verify_hooks as verify_hooks
    import hermes_cli.lifecycle as lifecycle
    import hermes_cli.plugins as plugins

    monkeypatch.setattr(lifecycle, "has_hook", has_hook)
    monkeypatch.setattr(lifecycle, "invoke_hook", invoke_hook)
    monkeypatch.setattr(plugins, "get_pre_verify_continue_message", continue_message)
    monkeypatch.setattr(verify_hooks, "max_verify_nudges", lambda config=None: 2)
    return seen


def _tool_use(session, call_id, name, args):
    session.on_event({"type": "assistant", "message": {"content": [{"type": "tool_use", "id": call_id, "name": name, "input": args}]}})


def _tool_result(session, call_id, text):
    session.on_event({"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": call_id, "content": text}]}})


def test_claude_turn_replays_tool_hooks_runs_the_verify_gate_and_continues_once(hooks, monkeypatch, tmp_path):
    changed = tmp_path / "app.py"
    changed.write_text("print('x')\n")
    prompts = []

    def fake_run_turn(self, prompt):
        prompts.append(prompt)
        if len(prompts) == 1:
            _tool_use(self, "c1", "TodoWrite", {"todos": [{"content": "A", "status": "completed", "activeForm": "Doing A"}, {"content": "B", "status": "in_progress", "activeForm": "Doing B"}]})
            _tool_result(self, "c1", "Todos have been modified successfully.")
            _tool_use(self, "c2", "Edit", {"file_path": str(changed), "old_string": "x", "new_string": "y"})
            _tool_result(self, "c2", "The file has been updated.")
            _tool_use(self, "c3", "Bash", {"command": "pytest -q", "description": "Run tests"})
            _tool_result(self, "c3", "1 failed, 3 passed")
            _tool_use(self, "c4", "mcp__hermes-tools__typesafe_decide", {"state": "s", "questions": {}})
            _tool_result(self, "c4", '{"answers": {}}')
            return ClaudeCodeTurnResult(final_text="Done, tests pass.", session_id="sess-1", usage={"input_tokens": 10, "output_tokens": 5}, tool_iterations=4, session_confirmed=True)
        _tool_use(self, "c5", "Bash", {"command": "pytest -q"})
        _tool_result(self, "c5", "4 passed")
        return ClaudeCodeTurnResult(final_text="Fixed criterion 2 and re-ran the tests.", session_id="sess-1", usage={"input_tokens": 7, "output_tokens": 3}, tool_iterations=1, session_confirmed=True)

    monkeypatch.setattr(ClaudeCodeSession, "run_turn", fake_run_turn)
    agent = _agent()
    result = agent.run_conversation("add the feature")

    assert len(prompts) == 2, "the verify gate sent exactly one follow-up turn"
    assert "add the feature" in prompts[0]
    assert "Preflight from Jev: k=3 candidates." in prompts[0], "the pre_llm_call note reaches the Claude prompt"
    assert prompts[1] == "Fix criterion 2, then finish."
    assert result["final_response"] == "Fixed criterion 2 and re-ran the tests."
    assert result["completed"] is True
    assert result["api_calls"] == 2
    assert result["input_tokens"] == 17 and result["output_tokens"] == 8

    names = [call["tool_name"] for call in hooks["post_tool_call"]]
    assert names == ["todo", "patch", "terminal", "typesafe_decide", "terminal"], "Claude calls arrive under Hermes tool names"
    todo = json.loads(hooks["post_tool_call"][0]["result"])
    assert todo == {"todos": [{"id": "1", "content": "A", "status": "completed"}, {"id": "2", "content": "B", "status": "in_progress"}]}
    assert hooks["post_tool_call"][2]["args"] == {"command": "pytest -q"}
    assert hooks["post_tool_call"][2]["result"] == "1 failed, 3 passed"
    assert hooks["post_tool_call"][4]["result"] == "4 passed"

    assert len(hooks["pre_verify"]) == 2
    assert hooks["pre_verify"][0]["changed_paths"] == [str(changed.resolve())]
    assert hooks["pre_verify"][0]["final_response"] == "Done, tests pass."
    assert hooks["pre_verify"][0]["coding"] is True
    assert hooks["pre_verify"][1]["attempt"] == 1

    assert len(hooks["post_llm_call"]) == 1
    assert hooks["post_llm_call"][0]["assistant_response"] == "Fixed criterion 2 and re-ran the tests."
    synthetic = [m for m in result["messages"] if isinstance(m, dict) and m.get("_pre_verify_synthetic")]
    assert len(synthetic) == 1 and synthetic[0]["content"] == "Fix criterion 2, then finish."
    assistant_texts = [m["content"] for m in result["messages"] if isinstance(m, dict) and m.get("role") == "assistant"]
    assert assistant_texts == ["Done, tests pass.", "Fixed criterion 2 and re-ran the tests."], "the attempted answer stays in the transcript"


def test_claude_turn_without_file_changes_finishes_without_a_verify_gate(hooks, monkeypatch):
    prompts = []

    def fake_run_turn(self, prompt):
        prompts.append(prompt)
        _tool_use(self, "c1", "Bash", {"command": "ls"})
        _tool_result(self, "c1", "app.py")
        _tool_use(self, "c2", "Read", {"file_path": "/tmp/app.py"})
        _tool_result(self, "c2", "print('x')")
        return ClaudeCodeTurnResult(final_text="Just an answer.", session_id="sess-1", tool_iterations=2, session_confirmed=True)

    monkeypatch.setattr(ClaudeCodeSession, "run_turn", fake_run_turn)
    agent = _agent()
    result = agent.run_conversation("what is here?")
    assert len(prompts) == 1
    assert hooks["pre_verify"] == []
    assert [call["tool_name"] for call in hooks["post_tool_call"]] == ["terminal", "read_file"]
    assert len(hooks["post_llm_call"]) == 1
    assert result["api_calls"] == 1


def test_claude_turn_binds_the_turn_emitter_so_a_hook_can_reach_the_run_stream(hooks, monkeypatch):
    from hermes_cli import turn_events

    monkeypatch.setattr(ClaudeCodeSession, "run_turn", lambda self, prompt: ClaudeCodeTurnResult(final_text="ok", session_id="sess-1", session_confirmed=True))
    agent = _agent()
    events = []
    agent.tool_progress_callback = lambda event_type, tool_name=None, preview=None, args=None, **kwargs: events.append((event_type, kwargs))
    agent.run_conversation("hi")
    assert turn_events.turn_emitter(agent.session_id) is agent.tool_progress_callback
    assert turn_events.emit_turn_event(agent.session_id, "judge.verdict", text="x", stage="budget") is True
    assert events[-1] == ("judge.verdict", {"stage": "budget"})
