"""The gateway registers every built-in tool before its first request.

On 2026-09-30 the first agent a fresh gateway created imported ``run_agent``
lazily, which registered about a hundred tools and moved the tool registry
generation that every runtime signature includes. The chat that caused it
dropped its live CLI on its next message (06:08, and the 11:31 smoke).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

_PROBE = r"""
import json
import gateway.run
import gateway.platforms.api_server
from tools.registry import registry

before = registry._generation
warmed = gateway.run._warm_agent_runtime()
# Everything the first request does: _create_agent's imports and the agent's tool list.
import run_agent, agent.opus_delegation, hermes_cli.tools_config, agent.claude_runtime  # noqa
import model_tools
model_tools.get_tool_definitions(quiet_mode=True)
print(json.dumps({"before": before, "warmed": warmed, "after_first_agent": registry._generation}))
"""


def test_a_fresh_gateway_settles_the_registry_before_the_first_agent():
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE], cwd=REPO, capture_output=True, text=True, timeout=180,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["warmed"] > result["before"], "the warm-up is what registers the built-in tools"
    assert result["after_first_agent"] == result["warmed"], "the first agent's imports and tool list register nothing more"


def test_a_real_tool_change_still_changes_the_runtime_signature():
    from gateway.run import GatewayRunner
    from tools.registry import registry

    before = GatewayRunner._extract_cache_busting_config({})["tools.registry_generation"]
    registry.register(
        name="warmup_probe_tool", toolset="warmup-probe", handler=lambda args, **kwargs: "ok",
        schema={"name": "warmup_probe_tool", "description": "probe", "parameters": {"type": "object", "properties": {}}},
    )
    try:
        after = GatewayRunner._extract_cache_busting_config({})["tools.registry_generation"]
        assert after != before, "a tool registered at runtime still rebuilds cached agents"
    finally:
        registry.deregister("warmup_probe_tool")
