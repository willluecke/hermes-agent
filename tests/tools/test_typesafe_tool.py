"""Tests for the TypeSafe Jev decision tool."""

import json

import httpx
import pytest

import toolsets
from tools import typesafe_tool
from tools.typesafe_tool import (
    TYPESAFE_SCHEMA,
    check_typesafe_requirements,
    typesafe_decide,
)


class _Response:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body
        self.text = json.dumps(body)

    def json(self):
        return self._body


QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this",
        "criteria": {"billing": "Payments", "technical": "Bugs", "sales": "Pricing"},
    },
    "frustration": {
        "type": "score",
        "instructions": "How frustrated the customer appears",
        "criteria": ["Calm", "Frustrated but civil", "Very angry"],
    },
    "is_urgent": {"type": "noul", "instructions": "The message conveys urgency"},
}
ANSWERS = {
    "department": {"type": "choice", "choice": "technical", "probabilities": {"technical": 0.84}, "confidence": 0.6},
    "frustration": {"type": "score", "score": 1.035, "legend": {"0": "Calm"}, "confidence": 0.84},
    "is_urgent": {"type": "noul", "noul": 0.999},
}


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setattr(
        typesafe_tool, "_env_value", lambda name: "ts-test-key" if name == "TYPESAFE_API_KEY" else ""
    )


@pytest.fixture
def no_api_key(monkeypatch):
    monkeypatch.setattr(typesafe_tool, "_env_value", lambda name: "")


def _capture(monkeypatch, status=200, body=None, raise_exc=None):
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        if raise_exc:
            raise raise_exc
        return _Response(status, body if body is not None else {"model": "jev-latest", "answers": ANSWERS, "usage": {"input_tokens": 312, "output_tokens": 48}})

    monkeypatch.setattr(typesafe_tool.httpx, "post", fake_post)
    return calls


def test_registered_in_the_core_toolset_and_schema_is_typed():
    assert TYPESAFE_SCHEMA["name"] == "typesafe_decide"
    assert TYPESAFE_SCHEMA["parameters"]["required"] == ["state", "questions"]
    assert "typesafe_decide" in toolsets._HERMES_CORE_TOOLS
    assert toolsets.TOOLSETS["typesafe"]["tools"] == ["typesafe_decide", "acceptance_criteria", "report_results"]


def test_unavailable_without_a_key(no_api_key, monkeypatch):
    calls = _capture(monkeypatch)
    assert check_typesafe_requirements() is False
    result = json.loads(typesafe_decide({"state": "hello", "questions": QUESTIONS}))
    assert "TYPESAFE_API_KEY" in result["error"]
    assert calls == []


def test_sends_the_documented_request_and_returns_typed_answers(api_key, monkeypatch):
    calls = _capture(monkeypatch)
    assert check_typesafe_requirements() is True
    result = json.loads(typesafe_decide({"state": "Stripe keeps failing, please help ASAP", "questions": QUESTIONS}))

    assert len(calls) == 1
    call = calls[0]
    assert call["url"] == "https://api.typesafe.ai/v1/systemone"
    assert call["headers"]["Authorization"] == "Bearer ts-test-key"
    assert call["json"] == {"state": "Stripe keeps failing, please help ASAP", "model": "jev-latest", "questions": QUESTIONS}
    assert call["timeout"] == typesafe_tool.TIMEOUT_SECONDS

    assert result["model"] == "jev-latest"
    assert result["answers"]["department"]["choice"] == "technical"
    assert result["answers"]["is_urgent"]["noul"] == 0.999
    assert result["usage"]["input_tokens"] == 312


def test_structured_state_and_explicit_model_pass_through(api_key, monkeypatch):
    calls = _capture(monkeypatch)
    typesafe_decide({"state": {"x": 1, "y": [2, 3]}, "questions": QUESTIONS, "model": "jev-2026-09"})
    assert calls[0]["json"]["state"] == {"x": 1, "y": [2, 3]}
    assert calls[0]["json"]["model"] == "jev-2026-09"


@pytest.mark.parametrize(
    "args, fragment",
    [
        ({"state": "", "questions": QUESTIONS}, "state must not be empty"),
        ({"state": 42, "questions": QUESTIONS}, "state must be text"),
        ({"state": "x", "questions": {}}, "non-empty object"),
        ({"state": "x", "questions": {"q": {"type": "essay", "instructions": "?"}}}, "type must be one of"),
        ({"state": "x", "questions": {"q": {"type": "noul"}}}, "instructions are required"),
        ({"state": "x", "questions": {"q": {"type": "choice", "instructions": "?", "criteria": {"only": "one"}}}}, "at least two option keys"),
        ({"state": "x", "questions": {"q": {"type": "score", "instructions": "?", "criteria": "low-high"}}}, "ordered list"),
    ],
)
def test_bad_input_is_refused_before_any_request(api_key, monkeypatch, args, fragment):
    calls = _capture(monkeypatch)
    result = json.loads(typesafe_decide(args))
    assert fragment in result["error"]
    assert calls == []


@pytest.mark.parametrize(
    "status, fragment, retryable",
    [(401, "invalid API key", False), (422, "failed validation", False), (429, "rate limit", True), (529, "overloaded", True)],
)
def test_http_errors_are_named_and_marked_retryable(api_key, monkeypatch, status, fragment, retryable):
    _capture(monkeypatch, status=status, body={"error": "nope"})
    result = json.loads(typesafe_decide({"state": "x", "questions": QUESTIONS}))
    assert fragment in result["error"]
    assert result["retryable"] is retryable


def test_network_failure_is_reported_not_raised(api_key, monkeypatch):
    _capture(monkeypatch, raise_exc=httpx.ConnectError("boom"))
    result = json.loads(typesafe_decide({"state": "x", "questions": QUESTIONS}))
    assert "ConnectError" in result["error"]
    assert result["retryable"] is True


# ---------------------------------------------------------------------------
# Providers. OpenRouter proxies the same protocol at a different URL with its
# own key; TYPESAFE_PROVIDER orders the providers and a failure other than a
# rejected request body moves to the next one. Verified live on 2026-09-20:
# OpenRouter answered the plain jev-latest id identically to TypeSafe direct.
# ---------------------------------------------------------------------------

from tools.typesafe_tool import OPENROUTER_DECISIONS_URL, ask_jev, provider_chain

OR_BODY = {
    "model": "typesafe/jev-1.13-20260917",
    "answers": ANSWERS,
    "usage": {"input_tokens": 370, "output_tokens": 57, "cost": 0.00001554},
}


def _env(monkeypatch, **values):
    monkeypatch.setattr(typesafe_tool, "_env_value", lambda name: values.get(name, ""))


def _capture_sequence(monkeypatch, outcomes):
    """Each outcome is ``(status, body)`` or an exception, consumed in order."""
    calls = []
    queue = list(outcomes)

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        outcome = queue.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        status, body = outcome
        return _Response(status, body)

    monkeypatch.setattr(typesafe_tool.httpx, "post", fake_post)
    return calls


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("", ("typesafe",)),
        ("openrouter", ("openrouter",)),
        ("typesafe,openrouter", ("typesafe", "openrouter")),
        (" OpenRouter , TypeSafe ", ("openrouter", "typesafe")),
        ("bogus,openrouter", ("openrouter",)),
        ("typesafe,typesafe,openrouter", ("typesafe", "openrouter")),
        ("bogus", ("typesafe",)),
    ],
)
def test_provider_chain_comes_from_typesafe_provider(monkeypatch, raw, expected):
    _env(monkeypatch, TYPESAFE_PROVIDER=raw)
    assert provider_chain() == expected


def test_openrouter_sends_the_same_request_to_the_decisions_endpoint_with_its_own_key(monkeypatch):
    _env(monkeypatch, TYPESAFE_PROVIDER="openrouter", OPENROUTER_API_KEY="or-test-key", TYPESAFE_API_KEY="ts-test-key")
    calls = _capture_sequence(monkeypatch, [(200, dict(OR_BODY))])
    assert check_typesafe_requirements() is True
    result = json.loads(typesafe_decide({"state": "Stripe keeps failing", "questions": QUESTIONS}))

    assert [c["url"] for c in calls] == [OPENROUTER_DECISIONS_URL]
    assert calls[0]["url"] == "https://openrouter.ai/api/alpha/decisions"
    assert calls[0]["headers"]["Authorization"] == "Bearer or-test-key"
    assert calls[0]["json"] == {"state": "Stripe keeps failing", "model": "jev-latest", "questions": QUESTIONS}
    assert result["provider"] == "openrouter"
    assert result["model"] == "typesafe/jev-1.13-20260917"
    assert result["answers"]["is_urgent"]["noul"] == 0.999
    assert result["usage"]["cost"] == 0.00001554


def test_the_default_chain_reports_typesafe_direct_as_the_provider(api_key, monkeypatch):
    _capture(monkeypatch)
    result = json.loads(typesafe_decide({"state": "x", "questions": QUESTIONS}))
    assert result["provider"] == "typesafe"


@pytest.mark.parametrize(
    "first",
    [
        (503, {"error": "model unavailable"}),
        (529, {"error": "overloaded"}),
        (429, {"error": "slow down"}),
        (401, {"error": "bad key"}),
        (402, {"error": "no credits"}),
        httpx.ConnectError("boom"),
        httpx.ReadTimeout("slow"),
    ],
)
def test_a_failure_on_the_first_provider_moves_to_the_next(monkeypatch, first):
    _env(monkeypatch, TYPESAFE_PROVIDER="typesafe,openrouter", TYPESAFE_API_KEY="ts-test-key", OPENROUTER_API_KEY="or-test-key")
    calls = _capture_sequence(monkeypatch, [first, (200, dict(OR_BODY))])
    result = json.loads(typesafe_decide({"state": "x", "questions": QUESTIONS}))

    assert [c["url"] for c in calls] == ["https://api.typesafe.ai/v1/systemone", OPENROUTER_DECISIONS_URL]
    assert calls[0]["headers"]["Authorization"] == "Bearer ts-test-key"
    assert calls[1]["headers"]["Authorization"] == "Bearer or-test-key"
    assert "error" not in result
    assert result["provider"] == "openrouter"


@pytest.mark.parametrize("status", [400, 422])
def test_a_rejected_request_body_does_not_move_on(monkeypatch, status):
    _env(monkeypatch, TYPESAFE_PROVIDER="typesafe,openrouter", TYPESAFE_API_KEY="ts-test-key", OPENROUTER_API_KEY="or-test-key")
    calls = _capture_sequence(monkeypatch, [(status, {"error": "bad body"}), (200, dict(OR_BODY))])
    result = json.loads(typesafe_decide({"state": "x", "questions": QUESTIONS}))
    assert len(calls) == 1, "the next provider would reject the same body"
    assert result["retryable"] is False
    assert "TypeSafe returned HTTP %d" % status in result["error"]


def test_every_provider_failing_names_each_failure(monkeypatch):
    _env(monkeypatch, TYPESAFE_PROVIDER="typesafe,openrouter", TYPESAFE_API_KEY="ts-test-key", OPENROUTER_API_KEY="or-test-key")
    calls = _capture_sequence(monkeypatch, [(503, {"error": "down"}), httpx.ConnectError("boom")])
    result = json.loads(typesafe_decide({"state": "x", "questions": QUESTIONS}))
    assert len(calls) == 2
    assert "TypeSafe returned HTTP 503: service unavailable" in result["error"]
    assert "OpenRouter request failed: ConnectError" in result["error"]
    assert result["retryable"] is True


def test_a_provider_without_a_key_is_skipped_not_tried(monkeypatch):
    _env(monkeypatch, TYPESAFE_PROVIDER="typesafe,openrouter", OPENROUTER_API_KEY="or-test-key")
    calls = _capture_sequence(monkeypatch, [(200, dict(OR_BODY))])
    assert check_typesafe_requirements() is True
    result = json.loads(typesafe_decide({"state": "x", "questions": QUESTIONS}))
    assert [c["url"] for c in calls] == [OPENROUTER_DECISIONS_URL]
    assert result["provider"] == "openrouter"


def test_unavailable_when_no_provider_in_the_chain_has_a_key(monkeypatch):
    _env(monkeypatch, TYPESAFE_PROVIDER="typesafe,openrouter")
    calls = _capture_sequence(monkeypatch, [])
    assert check_typesafe_requirements() is False
    result = json.loads(typesafe_decide({"state": "x", "questions": QUESTIONS}))
    assert "TYPESAFE_API_KEY or OPENROUTER_API_KEY is not configured" in result["error"]
    assert calls == []


def test_ask_jev_raises_once_with_the_whole_story(monkeypatch):
    _env(monkeypatch, TYPESAFE_PROVIDER="typesafe,openrouter", TYPESAFE_API_KEY="ts", OPENROUTER_API_KEY="or")
    _capture_sequence(monkeypatch, [(529, {"e": 1}), (402, {"e": 2})])
    with pytest.raises(typesafe_tool.JevError) as excinfo:
        ask_jev("x", QUESTIONS)
    err = excinfo.value
    assert err.status == 402
    assert err.provider == "openrouter"
    assert err.retryable is True, "a 529 on the way is retryable even though the last answer was 402"


def test_an_explicit_api_key_goes_once_to_the_first_provider(monkeypatch):
    _env(monkeypatch, TYPESAFE_PROVIDER="openrouter,typesafe")
    calls = _capture_sequence(monkeypatch, [(503, {"e": 1})])
    with pytest.raises(typesafe_tool.JevError):
        ask_jev("x", QUESTIONS, api_key="explicit")
    assert len(calls) == 1
    assert calls[0]["url"] == OPENROUTER_DECISIONS_URL
    assert calls[0]["headers"]["Authorization"] == "Bearer explicit"


def test_callers_can_pin_providers_regardless_of_env(monkeypatch):
    _env(monkeypatch, TYPESAFE_PROVIDER="typesafe", TYPESAFE_API_KEY="ts", OPENROUTER_API_KEY="or")
    calls = _capture_sequence(monkeypatch, [(200, dict(OR_BODY))])
    body = ask_jev("x", QUESTIONS, providers=["openrouter"])
    assert calls[0]["url"] == OPENROUTER_DECISIONS_URL
    assert body["provider"] == "openrouter"


# ---------------------------------------------------------------------------
# The chain is read from ~/.hermes/.env before the process environment, so a
# deliberate edit lands on the next call instead of after a gateway restart.
# ---------------------------------------------------------------------------


def test_a_dotenv_edit_beats_the_value_the_process_inherited(monkeypatch, tmp_path):
    from hermes_cli import config

    (tmp_path / ".env").write_text("TYPESAFE_PROVIDER=openrouter,typesafe\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TYPESAFE_PROVIDER", "typesafe,openrouter")
    config.invalidate_env_cache()
    assert typesafe_tool._env_value("TYPESAFE_PROVIDER") == "openrouter,typesafe"
    assert provider_chain() == ("openrouter", "typesafe")


def test_the_process_environment_still_answers_when_dotenv_is_silent(monkeypatch, tmp_path):
    from hermes_cli import config

    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TYPESAFE_PROVIDER", "openrouter")
    config.invalidate_env_cache()
    assert provider_chain() == ("openrouter",)
