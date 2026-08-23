"""Tests for hermes-api-server toolset and API server tool availability."""
from unittest.mock import patch, MagicMock


from toolsets import resolve_toolset, get_toolset, validate_toolset


class TestHermesApiServerToolset:
    """Tests for the hermes-api-server toolset definition."""


    def test_toolset_includes_web_tools(self):
        tools = resolve_toolset("hermes-api-server")
        assert "web_search" in tools
        assert "web_extract" in tools

    def test_toolset_includes_core_tools(self):
        tools = resolve_toolset("hermes-api-server")
        expected = [
            "terminal", "process",
            "read_file", "write_file", "patch", "search_files",
            "vision_analyze", "image_generate",
            "execute_code", "delegate_task",
            "todo", "memory", "opus_code_worker", "session_search", "cronjob",
        ]
        for tool in expected:
            assert tool in tools, f"Missing expected tool: {tool}"

    def test_toolset_includes_browser_tools(self):
        tools = resolve_toolset("hermes-api-server")
        for tool in ["browser_navigate", "browser_snapshot", "browser_click",
                      "browser_type", "browser_scroll", "browser_back",
                      "browser_press"]:
            assert tool in tools, f"Missing browser tool: {tool}"


class TestApiServerPlatformConfig:

    def test_default_api_server_includes_terminal_toolset(self):
        """Regression #49622: desktop-only read_terminal is registered into the
        'terminal' toolset (ships in-repo), so resolve_toolset('terminal') grows
        to include it after discovery. read_terminal is NOT in the
        hermes-api-server composite, so the old all-tools subset test dropped
        'terminal' entirely. Its static membership (terminal, process) IS in the
        composite, so it must stay enabled."""
        from tools.registry import discover_builtin_tools
        from hermes_cli.tools_config import _get_platform_tools
        discover_builtin_tools()
        toolsets = _get_platform_tools({}, "api_server")
        assert "terminal" in toolsets
        assert "opus_worker" in toolsets


class TestApiServerAdapterToolset:
    @patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", True)
    def test_create_agent_reads_config_toolsets(self):
        """API server resolves toolsets from config like all other platforms."""
        from gateway.platforms.api_server import APIServerAdapter
        from gateway.config import PlatformConfig

        adapter = APIServerAdapter(PlatformConfig())

        with patch("gateway.run._resolve_runtime_agent_kwargs") as mock_kwargs, \
             patch("gateway.run._resolve_gateway_model") as mock_model, \
             patch("gateway.run._load_gateway_config") as mock_config, \
             patch("run_agent.AIAgent") as mock_agent_cls:

            mock_kwargs.return_value = {"api_key": "test-key", "base_url": None,
                                        "provider": None, "api_mode": None,
                                        "command": None, "args": []}
            mock_model.return_value = "test/model"
            # No platform_toolsets override — should fall back to hermes-api-server default
            mock_config.return_value = {}
            mock_agent_cls.return_value = MagicMock()

            adapter._create_agent()

            mock_agent_cls.assert_called_once()
            call_kwargs = mock_agent_cls.call_args
            toolsets = call_kwargs.kwargs.get("enabled_toolsets")
            assert isinstance(toolsets, list)
            assert len(toolsets) > 0
            assert call_kwargs.kwargs.get("platform") == "api_server"
            assert call_kwargs.kwargs.get("disabled_toolsets") is None
            assert call_kwargs.kwargs.get("skip_context_files") is False
            assert call_kwargs.kwargs.get("skip_background_review") is False

    @patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", True)
    def test_single_model_agent_isolated_from_orchestrator_runtime(self):
        """A direct picker choice must not bootstrap through the configured
        Sol runtime or retain any model-spawning surface."""
        from gateway.platforms.api_server import APIServerAdapter
        from gateway.config import PlatformConfig

        adapter = APIServerAdapter(PlatformConfig())
        selected_runtime = {
            "api_key": "selected-key",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "openrouter",
            "api_mode": "openai",
            "command": None,
            "args": [],
        }

        with patch(
            "gateway.platforms.api_server._resolve_request_runtime_agent_kwargs",
            return_value=selected_runtime,
        ) as selected_resolver, patch(
            "gateway.run._resolve_runtime_agent_kwargs"
        ) as global_resolver, patch(
            "gateway.run._resolve_gateway_model"
        ) as global_model, patch(
            "gateway.run._load_gateway_config", return_value={}
        ), patch(
            "gateway.run.GatewayRunner._load_fallback_model"
        ) as fallback_loader, patch(
            "run_agent.AIAgent"
        ) as agent_cls:
            agent_cls.return_value = MagicMock()

            adapter._create_agent(
                requested_model="new/openrouter-model",
                requested_provider="openrouter",
                single_model=True,
                session_id="direct-session",
            )

        assert selected_resolver.call_count >= 1
        assert selected_resolver.call_args_list[0].args == ("openrouter",)
        assert selected_resolver.call_args_list[0].kwargs == {
            "target_model": "new/openrouter-model"
        }
        global_resolver.assert_not_called()
        global_model.assert_not_called()
        fallback_loader.assert_not_called()

        kwargs = agent_cls.call_args.kwargs
        assert kwargs["model"] == "new/openrouter-model"
        assert kwargs["provider"] == "openrouter"
        assert kwargs["fallback_model"] is None
        assert kwargs["disabled_toolsets"] == ["delegation", "opus_worker"]
        assert kwargs["skip_context_files"] is True
        assert kwargs["skip_background_review"] is True
        runtime = agent_cls.return_value._hermes_api_runtime
        assert runtime["execution_mode"] == "single_model"
        assert runtime["route_source"] == "direct_model"

    def test_direct_mode_denylist_removes_every_model_spawning_tool(self):
        from model_tools import get_tool_definitions
        from tools.registry import discover_builtin_tools

        discover_builtin_tools()
        tools = get_tool_definitions(
            enabled_toolsets=["hermes-api-server"],
            disabled_toolsets=["delegation", "opus_worker"],
            quiet_mode=True,
        )
        names = {tool["function"]["name"] for tool in tools}
        assert "delegate_task" not in names
        assert "opus_code_worker" not in names
        assert "terminal" in names
