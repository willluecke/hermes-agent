"""Hermes Chat's Ultracode on the subscription CLI lanes (2026-09-23)."""

from __future__ import annotations

from agent.claude_runtime import _claude_code_effort
from agent.reasoning_effort import requested_effort
from agent.transports.claude_code_session import claude_code_args
from gateway.platforms.api_server import _request_reasoning_config


def test_ultracode_is_accepted_as_max_effort_plus_a_flag():
    expected = {"enabled": True, "effort": "max", "ultracode": True}
    assert _request_reasoning_config({"reasoning_effort": "ultracode"}) == expected
    assert _request_reasoning_config({"reasoning": {"effort": "Ultracode"}}) == expected
    # providers other than the CLI lanes only ever read the effort: max
    assert requested_effort(expected) == "max"


def test_other_efforts_are_unchanged():
    assert _request_reasoning_config({"reasoning_effort": "xhigh"}) == {"enabled": True, "effort": "xhigh"}
    assert _request_reasoning_config({"reasoning_effort": "bogus"}) is None


def test_claude_lane_runs_claude_codes_own_ultracode():
    assert _claude_code_effort({"enabled": True, "effort": "max", "ultracode": True}) == "ultracode"
    assert _claude_code_effort({"enabled": True, "effort": "max"}) == "max"
    args = claude_code_args(model="claude-opus-5-5", session_id="00000000-0000-4000-8000-000000000000",
                            effort="ultracode")
    assert args[args.index("--effort") + 1] == "ultracode"
