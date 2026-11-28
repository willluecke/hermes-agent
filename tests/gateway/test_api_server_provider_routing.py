"""Hermes Chat runs (/v1/runs -> APIServerAdapter._create_agent) apply the
OpenRouter ``provider_routing`` in config.yaml, like the messaging gateway.

Until 2026-11-27 the API server dropped the section, so no Hermes Chat run could
be routed (Will wanted Gemini kept on one Google backend so its prompt cache is
not split between AI Studio and Vertex).
"""
from unittest.mock import MagicMock, patch

from gateway.config import PlatformConfig

_RUNTIME = {
    "api_key": "k",
    "base_url": "https://openrouter.ai/api/v1",
    "provider": "openrouter",
    "api_mode": "openai",
    "command": None,
    "args": [],
}


def _agent_kwargs(routing):
    from gateway.platforms.api_server import APIServerAdapter

    adapter = APIServerAdapter(PlatformConfig())
    with patch(
        "gateway.platforms.api_server._resolve_request_runtime_agent_kwargs",
        return_value=_RUNTIME,
    ), patch("gateway.run._load_gateway_config", return_value={}), patch(
        "gateway.run.GatewayRunner._load_provider_routing", return_value=routing
    ), patch("run_agent.AIAgent") as agent_cls:
        agent_cls.return_value = MagicMock()
        adapter._create_agent(
            requested_model="google/gemini-4-neon",
            requested_provider="openrouter",
            single_model=True,
            session_id="hermes-chat-c1",
        )
    return agent_cls.call_args.kwargs


@patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", True)
def test_a_chat_run_carries_the_configured_provider_routing():
    kwargs = _agent_kwargs(
        {"ignore": ["google-ai-studio"], "sort": "price", "data_collection": "deny"}
    )
    assert kwargs["providers_ignored"] == ["google-ai-studio"]
    assert kwargs["provider_sort"] == "price"
    assert kwargs["provider_data_collection"] == "deny"
    # Nothing the config did not ask for: an order would turn OpenRouter's
    # sticky (cache-keeping) routing off.
    assert kwargs["providers_order"] is None
    assert kwargs["providers_allowed"] is None
    assert kwargs["provider_require_parameters"] is False


@patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", True)
def test_no_provider_routing_section_sends_no_preferences():
    kwargs = _agent_kwargs({})
    for key in (
        "providers_allowed",
        "providers_ignored",
        "providers_order",
        "provider_sort",
        "provider_data_collection",
    ):
        assert kwargs[key] is None, key
    assert kwargs["provider_require_parameters"] is False
