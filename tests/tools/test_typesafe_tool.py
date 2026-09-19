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
