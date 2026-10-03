"""A stop that lands after a turn has ended must not carry into the next message.

On 2026-10-02 two stop requests reached the gateway 234 ms apart. The first
ended the Codex turn cleanly; the second arrived after the turn had unwound
and set the agent's interrupt flag and the Codex session's interrupt event
with nothing running to consume them. The cached agent then carried that stop
into the user's next two messages: one aborted its turn-lease wait ("Stopped
waiting for another Hermes process"), the next ended inside the Codex session
before Codex saw it ("agent run ended before a final answer").
"""

from __future__ import annotations

import threading
from unittest.mock import patch

import pytest

import run_agent
from agent.transports.claude_code_session import ClaudeCodeSession
from agent.transports.codex_app_server_session import CodexAppServerSession, TurnResult


def _bare_agent():
    agent = run_agent.AIAgent.__new__(run_agent.AIAgent)
    agent._interrupt_requested = False
    agent._interrupt_message = None
    agent._hard_interrupt_requested = threading.Event()
    agent._execution_thread_id = None
    agent._interrupt_thread_signal_pending = False
    agent._pending_redirect = None
    agent._pending_redirect_lock = threading.Lock()
    agent._active_children = []
    agent._active_children_lock = threading.Lock()
    agent.quiet_mode = True
    return agent


class _CodexSessionStub:
    def __init__(self, active: bool = False):
        self._active_turn_lock = threading.Lock()
        self._active_turn_id = "turn-1" if active else None
        self._interrupt_event = threading.Event()

    clear_pending_interrupt = CodexAppServerSession.clear_pending_interrupt


class TestDiscardStaleInterrupt:
    def test_clears_the_agent_flag_and_the_idle_codex_session_event(self):
        agent = _bare_agent()
        agent._codex_session = _CodexSessionStub()
        agent._interrupt_requested = True
        agent._interrupt_message = "Stop requested via API"
        agent._hard_interrupt_requested.set()
        agent._codex_session._interrupt_event.set()

        assert agent.discard_stale_interrupt() is True
        assert agent._interrupt_requested is False
        assert agent._interrupt_message is None
        assert not agent._hard_interrupt_requested.is_set()
        assert not agent._codex_session._interrupt_event.is_set()

    def test_nothing_pending_reports_false(self):
        agent = _bare_agent()
        agent._codex_session = _CodexSessionStub()
        assert agent.discard_stale_interrupt() is False

    def test_leaves_a_running_codex_turn_alone(self):
        """While a turn is active the session's event is a live stop, not a stale one."""
        session = _CodexSessionStub(active=True)
        session._interrupt_event.set()
        assert session.clear_pending_interrupt() is False
        assert session._interrupt_event.is_set()

    def test_claude_session_clears_only_between_turns(self):
        session = ClaudeCodeSession.__new__(ClaudeCodeSession)
        session._turn_lock = threading.Lock()
        session._interrupt = threading.Event()
        session._interrupt_request_id = "hermes_interrupt_abc"
        session._interrupt.set()
        assert session.clear_pending_interrupt() is True
        assert not session._interrupt.is_set()
        assert session._interrupt_request_id is None

        session._interrupt.set()
        with session._turn_lock:  # a turn is running
            assert session.clear_pending_interrupt() is False
        assert session._interrupt.is_set()


def _codex_agent():
    return run_agent.AIAgent(
        api_key="stub",
        base_url="https://stub.invalid",
        provider="openai",
        api_mode="codex_app_server",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )


def test_a_stop_landing_after_the_codex_turn_ended_is_discarded(monkeypatch):
    """The second stop of 2026-10-02: it arrives during post-turn processing
    (hook replay, persistence). The turn is reported complete and nothing is
    left on the agent or its session for the next message."""
    import agent.codex_runtime as codex_runtime

    monkeypatch.setattr(CodexAppServerSession, "ensure_started", lambda self: "thread-stub-1")

    def fake_run_turn(self, user_input, **kwargs):
        return TurnResult(
            final_text="Done.",
            projected_messages=[{"role": "assistant", "content": "Done."}],
            tool_iterations=0,
            turn_id="turn-stub-1",
            thread_id="thread-stub-1",
        )

    monkeypatch.setattr(CodexAppServerSession, "run_turn", fake_run_turn)
    real_parity = codex_runtime._codex_hook_parity

    def late_stop_then_parity(agent, *args, **kwargs):
        agent.hard_interrupt("Stop requested via API")  # lands after the turn unwound
        return real_parity(agent, *args, **kwargs)

    monkeypatch.setattr(codex_runtime, "_codex_hook_parity", late_stop_then_parity)

    agent = _codex_agent()
    with patch.object(agent, "_spawn_background_review", return_value=None):
        result = agent.run_conversation("add the feature")

    assert result["final_response"] == "Done."
    assert result["completed"] is True
    assert result["interrupted"] is False
    assert agent._interrupt_requested is False, "the late stop is not left for the next run"
    assert agent._interrupt_message is None
    assert not agent._codex_session._interrupt_event.is_set(), "nor on the Codex session"
