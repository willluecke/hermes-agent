#!/usr/bin/env python3
"""TypeSafe AI System One decisions (the Jev model).

Jev is not a chat model. It takes a piece of state (text, a log line, a
ticket, JSON) plus typed questions and returns typed answers with calibrated
probabilities in well under a second. This tool lets Hermes ask those
questions mid-conversation instead of reasoning a classification, routing or
scoring decision out in prose.

Contract: POST https://api.typesafe.ai/v1/systemone with a Bearer key.
"""

import json
import os
from typing import Any, Dict

import httpx

from tools.registry import registry

TYPESAFE_URL = os.environ.get(
    "TYPESAFE_API_URL", "https://api.typesafe.ai/v1/systemone"
)
DEFAULT_MODEL = "jev-latest"
TIMEOUT_SECONDS = 30.0
QUESTION_TYPES = ("noul", "choice", "score")
MAX_QUESTIONS = 32
MAX_STATE_CHARS = 200_000

_STATUS_MEANINGS = {
    401: "missing or invalid API key",
    422: "request body failed validation",
    429: "rate limit exceeded",
    529: "service overloaded",
}


def _env_value(name: str) -> str:
    """Resolve ``name`` through Hermes' config/.env layer, then process env."""
    try:
        from hermes_cli.config import get_env_value

        value = get_env_value(name)
    except Exception:
        value = None
    if not value:
        value = os.environ.get(name)
    return str(value or "").strip()


def check_typesafe_requirements() -> bool:
    return bool(_env_value("TYPESAFE_API_KEY"))


def _validate_questions(questions: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a non-empty object keyed by question id")
    if len(questions) > MAX_QUESTIONS:
        raise ValueError(f"at most {MAX_QUESTIONS} questions per call")
    for question_id, question in questions.items():
        if not isinstance(question, dict):
            raise ValueError(f"question {question_id!r} must be an object")
        question_type = question.get("type")
        if question_type not in QUESTION_TYPES:
            raise ValueError(
                f"question {question_id!r}: type must be one of noul, choice, score"
            )
        if not str(question.get("instructions") or "").strip():
            raise ValueError(f"question {question_id!r}: instructions are required")
        criteria = question.get("criteria")
        if question_type == "choice" and (
            not isinstance(criteria, dict) or len(criteria) < 2
        ):
            raise ValueError(
                f"question {question_id!r}: choice needs criteria as an object of "
                "at least two option keys with descriptions"
            )
        if question_type == "score" and (
            not isinstance(criteria, list) or len(criteria) < 2
        ):
            raise ValueError(
                f"question {question_id!r}: score needs criteria as an ordered list "
                "of at least two level descriptions"
            )
    return questions


class JevError(RuntimeError):
    """A TypeSafe call that produced no usable answer."""

    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False, detail: str = ""):
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.detail = detail


def ask_jev(
    state: Any,
    questions: Dict[str, Dict[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
    timeout: float = TIMEOUT_SECONDS,
    api_key: str | None = None,
) -> Dict[str, Any]:
    """POST one System One request and return the decoded response body.

    Shared by the ``typesafe_decide`` tool and the system-one-preflight
    plugin. Raises :class:`JevError` for every failure so callers decide
    how much a missing answer matters to them.
    """
    key = api_key or _env_value("TYPESAFE_API_KEY")
    if not key:
        raise JevError("TYPESAFE_API_KEY is not configured")
    payload = {"state": state, "model": model, "questions": questions}
    try:
        response = httpx.post(
            TYPESAFE_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )
    except httpx.HTTPError as exc:
        raise JevError(
            f"TypeSafe request failed: {exc.__class__.__name__}", retryable=True
        ) from exc
    if response.status_code != 200:
        try:
            detail = str(response.text or "")[:500]
        except Exception:
            detail = ""
        meaning = _STATUS_MEANINGS.get(response.status_code, "unexpected status")
        raise JevError(
            f"TypeSafe returned HTTP {response.status_code}: {meaning}",
            status=response.status_code,
            retryable=response.status_code in (429, 529),
            detail=detail,
        )
    try:
        body = response.json()
    except ValueError as exc:
        raise JevError("TypeSafe returned a non-JSON body") from exc
    if not isinstance(body, dict):
        raise JevError("TypeSafe returned an unexpected body")
    return body


def typesafe_decide(args: Dict[str, Any]) -> str:
    """Ask Jev typed questions about ``state`` and return its typed answers."""
    if not _env_value("TYPESAFE_API_KEY"):
        return json.dumps({"error": "TYPESAFE_API_KEY is not configured"})

    state = args.get("state")
    if isinstance(state, str):
        if not state.strip():
            return json.dumps({"error": "state must not be empty"})
        if len(state) > MAX_STATE_CHARS:
            return json.dumps(
                {"error": f"state exceeds {MAX_STATE_CHARS} characters; pass less"}
            )
    elif not isinstance(state, (dict, list)):
        return json.dumps({"error": "state must be text or a JSON object or array"})

    try:
        questions = _validate_questions(args.get("questions"))
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

    model = str(args.get("model") or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    try:
        body = ask_jev(state, questions, model=model)
    except JevError as exc:
        failure: Dict[str, Any] = {"error": str(exc), "retryable": exc.retryable}
        if exc.detail:
            failure["detail"] = exc.detail
        return json.dumps(failure)
    return json.dumps(
        {
            "model": body.get("model", model),
            "answers": body.get("answers", {}),
            "usage": body.get("usage", {}),
        },
        ensure_ascii=False,
    )


TYPESAFE_SCHEMA = {
    "name": "typesafe_decide",
    "description": (
        "Ask TypeSafe's Jev decision model typed questions about a piece of state "
        "(text, a log, a ticket, JSON). It never writes prose: every question comes "
        "back as a typed answer with calibrated probabilities in under a second, "
        "far cheaper than reasoning the decision out in text. Use it to classify, "
        "route, score, or check a yes/no fact. Question types: noul (returns the "
        "probability the answer is yes), choice (picks one option from criteria "
        "{key: description}, with per-option probabilities and confidence), score "
        "(places the state on ordered criteria levels, with confidence)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "state": {
                "type": "string",
                "description": (
                    "The material to judge: raw text, or a JSON-encoded object or array."
                ),
            },
            "questions": {
                "type": "object",
                "description": (
                    "Map of question id to a question: {type: 'noul' | 'choice' | "
                    "'score', instructions: string, criteria}. choice criteria is an "
                    "object of option key to description; score criteria is an ordered "
                    "list of level descriptions from lowest to highest; noul needs no "
                    "criteria. Ask several questions in one call when they share the state."
                ),
                "additionalProperties": {"type": "object"},
            },
            "model": {
                "type": "string",
                "description": "TypeSafe model id. Defaults to jev-latest.",
            },
        },
        "required": ["state", "questions"],
    },
}


registry.register(
    name="typesafe_decide",
    toolset="typesafe",
    schema=TYPESAFE_SCHEMA,
    handler=lambda args, **kwargs: typesafe_decide(args),
    check_fn=check_typesafe_requirements,
    requires_env=["TYPESAFE_API_KEY"],
    emoji="🧭",
)
