from __future__ import annotations

from unittest.mock import patch

import run_agent
from agent.transports.claude_code_session import ClaudeCodeSession, ClaudeCodeTurnResult


def _make_agent():
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
