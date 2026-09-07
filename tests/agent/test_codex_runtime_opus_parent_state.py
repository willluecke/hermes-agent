"""The codex runtime must resolve the parent's Opus-worker state exactly.

Codex spawns the hermes-tools MCP server as its own process, which builds its
own tool list and cannot see this agent's ``disabled_toolsets``. The parent's
effective state is therefore resolved here and propagated with the session
(agent/opus_delegation.py); a wrong answer either leaks Opus into a denied
single-model turn or silently breaks the governed orchestration loop.
"""

from types import SimpleNamespace

from agent.codex_runtime import _agent_opus_worker_enabled


def _agent(enabled=None, disabled=None):
    return SimpleNamespace(enabled_toolsets=enabled, disabled_toolsets=disabled)


def test_orchestrated_api_server_agent_keeps_the_handoff():
    assert _agent_opus_worker_enabled(
        _agent(enabled=["hermes-api-server", "opus_worker", "terminal"])
    ) is True


def test_composite_toolset_alone_is_enough():
    """hermes-api-server resolves to opus_code_worker through the core set."""
    assert _agent_opus_worker_enabled(_agent(enabled=["hermes-api-server"])) is True


def test_single_model_denial_is_honored():
    assert _agent_opus_worker_enabled(
        _agent(
            enabled=["hermes-api-server", "opus_worker"],
            disabled=["delegation", "opus_worker"],
        )
    ) is False


def test_toolset_allowlist_without_the_worker_is_denied():
    assert _agent_opus_worker_enabled(_agent(enabled=["web"])) is False


def test_no_allowlist_means_every_registered_toolset():
    assert _agent_opus_worker_enabled(_agent()) is True


def test_unresolvable_state_fails_closed():
    class Exploding:
        @property
        def disabled_toolsets(self):
            raise RuntimeError("toolset resolution is unavailable")

    assert _agent_opus_worker_enabled(Exploding()) is False
