import json

import toolsets
from tools import model_consult_tool as consult
from tools.registry import registry


def test_lists_authenticated_provider_models(monkeypatch):
    monkeypatch.setattr(
        consult,
        "_inventory_rows",
        lambda: [
            {
                "slug": "openrouter",
                "authenticated": True,
                "models": ["z-ai/glm-5.3", "z-ai/glm-5.3-flash"],
            }
        ],
    )

    result = json.loads(consult.model_consult_tool("list", provider="openrouter"))

    assert result == {
        "ok": True,
        "provider": "openrouter",
        "models": ["z-ai/glm-5.3", "z-ai/glm-5.3-flash"],
    }


def test_consult_routes_exact_advertised_model(monkeypatch):
    monkeypatch.setattr(
        consult,
        "_provider_row",
        lambda provider: {
            "slug": provider,
            "authenticated": True,
            "models": ["z-ai/glm-5.3"],
        },
    )
    observed = {}

    def fake_call_provider(**kwargs):
        observed.update(kwargs)
        return "Use four short steps.", {
            "provider": "openrouter",
            "model": "z-ai/glm-5.3",
        }

    monkeypatch.setattr(consult, "_call_provider", fake_call_provider)

    result = json.loads(
        consult.model_consult_tool(
            "consult",
            provider="openrouter",
            model="z-ai/glm-5.3",
            prompt="Recommend the wording.",
            context="A parts marketplace.",
        )
    )

    assert result["ok"] is True
    assert result["provider"] == "openrouter"
    assert result["model"] == "z-ai/glm-5.3"
    assert result["response"] == "Use four short steps."
    assert observed["prompt"] == "Recommend the wording."


def test_consult_rejects_unadvertised_model_without_calling_provider(monkeypatch):
    monkeypatch.setattr(
        consult,
        "_provider_row",
        lambda provider: {
            "slug": provider,
            "authenticated": True,
            "models": ["z-ai/glm-5.3"],
        },
    )
    monkeypatch.setattr(
        consult,
        "_call_provider",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not call")),
    )

    result = json.loads(
        consult.model_consult_tool(
            "consult",
            provider="openrouter",
            model="glm-5.3",
            prompt="Recommend wording.",
        )
    )

    assert result["ok"] is False
    assert "action='list'" in result["error"]


def test_model_consult_is_registered_and_exposed():
    from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS

    assert registry.get_entry("model_consult") is not None
    assert "model_consult" in toolsets._HERMES_CORE_TOOLS
    assert toolsets.TOOLSETS["model_consult"]["tools"] == ["model_consult"]
    assert "model_consult" in EXPOSED_TOOLS
