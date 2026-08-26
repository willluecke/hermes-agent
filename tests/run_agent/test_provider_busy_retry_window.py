"""Bounded transient-429 recovery for a single Hermes model call."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.error_classifier import FailoverReason, classify_api_error
from run_agent import AIAgent


class Busy429(Exception):
    status_code = 429

    def __init__(self, message: str = "too many requests"):
        super().__init__(message)
        self.response = SimpleNamespace(headers={})
        self.body = {"error": {"message": message}}


def _tool_defs():
    return [
        {
            "type": "function",
            "function": {
                "name": "terminal",
                "description": "run a command",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def _response(content: str):
    message = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _agent(*, fallback=False, max_retries=3):
    fallback_model = (
        [{"provider": "openai", "model": "fallback/model"}]
        if fallback
        else None
    )
    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs()),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI", return_value=MagicMock()),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            provider="openrouter",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
    agent.client = MagicMock()
    agent._api_max_retries = max_retries
    return agent


def _run(agent, api_side_effect, retry_numbers):
    from agent import conversation_loop

    notices = []
    agent.status_callback = lambda _kind, text: notices.append(text)
    with (
        patch.object(agent, "_interruptible_api_call", side_effect=api_side_effect),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(
            conversation_loop,
            "provider_busy_retry_delay",
            side_effect=lambda attempt: retry_numbers.append(attempt) or 0.0,
        ),
        patch.object(conversation_loop.time, "sleep", return_value=None),
    ):
        result = agent.run_conversation("continue")
    return result, notices


def test_transient_429_can_recover_on_seventh_retry():
    agent = _agent()
    attempts = MagicMock(side_effect=[Busy429()] * 7 + [_response("recovered")])
    retry_numbers = []

    result, notices = _run(agent, attempts, retry_numbers)

    assert result["completed"] is True
    assert result["final_response"] == "recovered"
    assert attempts.call_count == 8
    assert retry_numbers == [1, 2, 3, 4, 5, 6, 7]
    assert len([notice for notice in notices if "Provider busy" in notice]) == 7
    assert all("same run and model" in notice for notice in notices)


def test_transient_429_ceiling_overrides_larger_generic_retry_setting():
    agent = _agent(max_retries=20)
    attempts = MagicMock(side_effect=Busy429())
    retry_numbers = []

    result, notices = _run(agent, attempts, retry_numbers)

    assert result["completed"] is False
    assert result["failure_reason"] == FailoverReason.rate_limit.value
    assert attempts.call_count == 8
    assert retry_numbers == [1, 2, 3, 4, 5, 6, 7]
    assert any("remained busy after 7 retries" in notice for notice in notices)


def test_orchestrated_fallback_waits_until_busy_window_exhausts():
    agent = _agent(fallback=True)
    attempts = MagicMock(
        side_effect=[Busy429()] * 8 + [_response("fallback recovered")]
    )
    retry_numbers = []

    def activate_fallback(*, reason=None):
        assert reason == FailoverReason.rate_limit
        agent._fallback_index = len(agent._fallback_chain)
        agent._fallback_activated = True
        agent.provider = "openai"
        agent.model = "fallback/model"
        return True

    with patch.object(
        agent, "_try_activate_fallback", side_effect=activate_fallback
    ) as fallback:
        result, _notices = _run(agent, attempts, retry_numbers)

    assert result["completed"] is True
    assert result["final_response"] == "fallback recovered"
    assert attempts.call_count == 9
    fallback.assert_called_once_with(reason=FailoverReason.rate_limit)


def test_explicit_credit_exhaustion_429_is_not_retried():
    error = Busy429("insufficient credits; top up your credits")
    classified = classify_api_error(
        error,
        provider="openrouter",
        model="test/model",
    )

    assert classified.reason == FailoverReason.billing
    assert classified.retryable is False

    agent = _agent()
    attempts = MagicMock(side_effect=error)
    result, _notices = _run(agent, attempts, [])

    assert result["completed"] is False
    assert result["failure_reason"] == FailoverReason.billing.value
    assert attempts.call_count == 1
