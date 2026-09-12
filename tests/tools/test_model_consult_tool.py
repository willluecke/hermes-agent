import json
from pathlib import Path
from types import SimpleNamespace

import pytest

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
        "authenticated": True,
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


@pytest.fixture
def native_route(monkeypatch):
    import agent.transports.claude_code_session as native
    monkeypatch.setattr(native, "find_claude_binary", lambda: "/fake/claude")
    monkeypatch.setattr(native, "claude_subscription_auth_available", lambda **kw: True)
    monkeypatch.setattr(consult, "_api_inventory_rows", lambda: [{
        "slug": "openrouter", "authenticated": True,
        "models": ["anthropic/claude-fable-5", "z-ai/glm-5.3"],
    }])
    return native


def test_inventory_discovers_native_subscription_before_api_routes(native_route):
    result = json.loads(consult.model_consult_tool("list"))
    native, api = result["providers"]
    assert native["provider"] == "claude-code"
    assert native["authenticated"] is True
    assert native["billing"] == "subscription"
    assert native["transport"] == "native_cli"
    assert native["preferred"] is True
    assert native["models"]
    assert api["provider"] == "openrouter"


def test_unavailable_native_route_is_visible_without_claiming_login(native_route, monkeypatch):
    monkeypatch.setattr(native_route, "claude_subscription_auth_available", lambda: False)
    result = json.loads(consult.model_consult_tool("list", provider="claude-code"))
    assert result["ok"] is True
    assert result["authenticated"] is False
    assert "unavailable" in result["note"]


def test_native_inventory_survives_api_catalog_failure(native_route, monkeypatch):
    def broken_api():
        raise RuntimeError("API catalog is offline")
    monkeypatch.setattr(consult, "_api_inventory_rows", broken_api)
    result = json.loads(consult.model_consult_tool("list"))
    assert result["providers"][0]["authenticated"] is True
    assert "temporarily unavailable" in result["providers"][0]["note"]


@pytest.mark.parametrize("provider, model, expected", [
    ("", "fable", "fable"),
    ("auto", "claude-fable-5", "claude-fable-5"),
    ("claude-code", "claude-future-model-99", "claude-future-model-99"),
    ("openrouter", "anthropic/claude-fable-5", "claude-fable-5"),
    ("anthropic", "claude-fable-5", "claude-fable-5"),
])
def test_claude_is_subscription_first_even_when_api_was_selected(native_route, monkeypatch, provider, model, expected):
    calls = []
    def native_call(**kwargs):
        calls.append(kwargs)
        return "Native advice", {"provider": "claude-code", "model": kwargs["model"], "billing": "subscription"}
    def forbidden(**kwargs):
        pytest.fail("must not use an API call or catalog for a native consultation")
    monkeypatch.setattr(consult, "_call_native_claude", native_call)
    monkeypatch.setattr(consult, "_call_provider", forbidden)
    monkeypatch.setattr(consult, "_api_inventory_rows", forbidden)
    result = json.loads(consult.model_consult_tool("consult", provider=provider, model=model, prompt="Advise"))
    assert result["ok"] is True
    assert result["provider"] == "claude-code"
    assert result["model"] == expected
    assert calls[0]["model"] == expected


def test_paid_claude_route_requires_explicit_boolean_override(native_route, monkeypatch):
    calls = []
    def api_call(**kwargs):
        calls.append(kwargs)
        return "API advice", {"provider": kwargs["provider"], "model": kwargs["model"]}
    monkeypatch.setattr(consult, "_call_provider", api_call)
    result = json.loads(consult.model_consult_tool("consult", provider="openrouter",
        model="anthropic/claude-fable-5", prompt="Advise", allow_paid_claude=True))
    assert result["ok"] is True
    assert result["provider"] == "openrouter"
    assert calls[0]["model"] == "anthropic/claude-fable-5"
    invalid = json.loads(consult.model_consult_tool("consult", provider="openrouter",
        model="anthropic/claude-fable-5", prompt="Advise", allow_paid_claude="true"))
    assert invalid["ok"] is False
    assert len(calls) == 1


def test_expired_subscription_never_falls_through_to_paid_route(native_route, monkeypatch):
    monkeypatch.setattr(native_route, "claude_subscription_auth_available", lambda: False)
    monkeypatch.setattr(consult, "_call_provider", lambda **kw: pytest.fail("paid fallback"))
    monkeypatch.setattr(consult, "_call_native_claude", lambda **kw: pytest.fail("missing auth"))
    result = json.loads(consult.model_consult_tool("consult", provider="openrouter",
        model="anthropic/claude-fable-5", prompt="Advise"))
    assert result["ok"] is False
    assert "no paid API fallback" in result["error"]


def test_real_native_adapter_receives_no_tools_in_isolated_context(native_route, monkeypatch):
    import io
    created = []
    prompt = "Context:\nA parts marketplace.\n\nQuestion:\nAdvise"
    monkeypatch.setattr(native_route.ClaudeCodeSession, "_next_debug_file", lambda _: None)
    monkeypatch.setattr(native_route, "_current_claude_auth_generation", lambda: "")
    def popen(args, **kwargs):
        created.append((args, kwargs))
        process = SimpleNamespace(pid=12345, stdin=io.StringIO(), stderr=io.StringIO(),
            stdout=io.StringIO("\n".join(json.dumps(event) for event in [
                {"type": "user", "message": {"role": "user", "content": prompt}},
                {"type": "result", "result": "Native advice"},
            ]) + "\n"), poll=lambda: 0, wait=lambda **kw: 0)
        return process
    monkeypatch.setattr(native_route.subprocess, "Popen", popen)
    monkeypatch.setattr(consult, "_call_provider", lambda **kw: pytest.fail("API fallback"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-reach-child")
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/tmp/isolated-claude-test-config")
    result = json.loads(consult.model_consult_tool("consult", provider="claude-code",
        model="claude-fable-5", prompt="Advise", context="A parts marketplace."))
    assert result["ok"] is True
    assert result["response"] == "Native advice"
    args, options = created[0]
    assert args[args.index("--model") + 1] == "claude-fable-5"
    assert args[args.index("--tools") + 1] == ""
    assert args[args.index("--setting-sources") + 1] == ""
    assert "--safe-mode" in args and "--strict-mcp-config" in args
    assert "--fallback-model" not in args and "--resume" not in args
    assert "ANTHROPIC_API_KEY" not in options["env"]
    assert "CLAUDE_CODE_USE_BEDROCK" not in options["env"]
    assert options["env"]["CLAUDE_CONFIG_DIR"] == "/tmp/isolated-claude-test-config"
    assert not Path(options["cwd"]).exists()


@pytest.mark.parametrize("outcome", ["error", "interrupted", "exception", "empty"])
def test_native_failures_are_bounded_closed_and_do_not_expose_logs(native_route, monkeypatch, outcome):
    sessions = []
    class Session:
        def __init__(self, **kwargs):
            self.options = kwargs
            self.closed = False
            sessions.append(self)
        def run_turn(self, prompt):
            if outcome == "exception":
                raise RuntimeError("secret-in-stderr")
            return SimpleNamespace(final_text="", error="secret-in-stderr" if outcome == "error" else None,
                                   interrupted=outcome == "interrupted")
        def close(self):
            self.closed = True
    monkeypatch.setattr(native_route, "ClaudeCodeSession", Session)
    monkeypatch.setattr(consult, "_call_provider", lambda **kw: pytest.fail("paid fallback"))
    result = json.loads(consult.model_consult_tool("consult", model="fable", prompt="Advise", timeout_seconds=20))
    assert result["ok"] is False
    assert "secret-in-stderr" not in result["error"]
    assert sessions[0].closed
    assert sessions[0].options["absolute_timeout"] == 20
    assert sessions[0].options["no_tools"] is True
    assert not Path(sessions[0].options["cwd"]).exists()


def test_api_advisor_context_uses_one_user_message(monkeypatch):
    import agent.auxiliary_client as auxiliary
    calls = []
    monkeypatch.setattr(auxiliary, "call_llm", lambda **kw: calls.append(kw) or "answer")
    monkeypatch.setattr(auxiliary, "extract_content_or_reasoning", lambda response: response)
    consult._call_provider(provider="openrouter", model="z-ai/glm-5.3", prompt="Advise", context="Context", timeout_seconds=20)
    assert [message["role"] for message in calls[0]["messages"]] == ["system", "user"]
    assert "Context" in calls[0]["messages"][1]["content"]
    assert "Advise" in calls[0]["messages"][1]["content"]
