#!/usr/bin/env python3
"""TypeSafe AI System One decisions (the Jev model).

Jev is not a chat model. It takes a piece of state (text, a log line, a
ticket, JSON) plus typed questions and returns typed answers with calibrated
probabilities in well under a second. This tool lets Hermes ask those
questions mid-conversation instead of reasoning a classification, routing or
scoring decision out in prose.

Two providers speak the same wire format, ``{state, model, questions}`` in and
``{model, answers, usage}`` out:

    typesafe    POST https://api.typesafe.ai/v1/systemone       Bearer TYPESAFE_API_KEY
    openrouter  POST https://openrouter.ai/api/alpha/decisions  Bearer OPENROUTER_API_KEY

``TYPESAFE_PROVIDER`` orders them, comma separated; the default is
``typesafe``. With ``typesafe,openrouter`` a call that fails on TypeSafe for
any reason other than a rejected request body is retried on OpenRouter, so a
TypeSafe outage (HTTP 503 "model unavailable", seen on 2026-09-17) no longer
loses the decision. OpenRouter accepts the plain ``jev-latest`` id and answers
identically; its ``model`` field names the dated build it served, e.g.
``typesafe/jev-1.13-20260917``. Its endpoint is alpha, which is why it is the
fallback and not the default.

Jev cannot be a chat model anywhere in Hermes: OpenRouter's chat endpoint
rejects it with HTTP 400 "is a decisions model".
"""

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx

from tools.registry import registry

TYPESAFE_URL = os.environ.get(
    "TYPESAFE_API_URL", "https://api.typesafe.ai/v1/systemone"
)
OPENROUTER_DECISIONS_URL = os.environ.get(
    "OPENROUTER_DECISIONS_URL", "https://openrouter.ai/api/alpha/decisions"
)
DEFAULT_MODEL = "jev-latest"
TIMEOUT_SECONDS = 30.0
QUESTION_TYPES = ("noul", "choice", "score")
MAX_QUESTIONS = 32
MAX_STATE_CHARS = 200_000

PROVIDERS: Dict[str, Dict[str, str]] = {
    "typesafe": {
        "url": TYPESAFE_URL,
        "key_env": "TYPESAFE_API_KEY",
        "label": "TypeSafe",
    },
    "openrouter": {
        "url": OPENROUTER_DECISIONS_URL,
        "key_env": "OPENROUTER_API_KEY",
        "label": "OpenRouter",
    },
}
DEFAULT_PROVIDER_CHAIN: Tuple[str, ...] = ("typesafe",)

_STATUS_MEANINGS = {
    400: "request rejected",
    401: "missing or invalid API key",
    402: "insufficient credits",
    404: "unknown endpoint or model",
    422: "request body failed validation",
    429: "rate limit exceeded",
    502: "bad gateway",
    503: "service unavailable",
    529: "service overloaded",
}
_RETRYABLE_STATUSES = {429, 502, 503, 529}
# A rejected request body is this side's fault and the next provider would
# reject it the same way. Every other failure is worth one try elsewhere.
_NO_FALLTHROUGH_STATUSES = {400, 422}


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


def provider_chain() -> Tuple[str, ...]:
    """The ordered providers to try, from ``TYPESAFE_PROVIDER``.

    Unknown names are dropped, duplicates collapse to their first position,
    and an empty result falls back to :data:`DEFAULT_PROVIDER_CHAIN`.
    """
    seen: List[str] = []
    for raw in _env_value("TYPESAFE_PROVIDER").split(","):
        name = raw.strip().lower()
        if name in PROVIDERS and name not in seen:
            seen.append(name)
    return tuple(seen) or DEFAULT_PROVIDER_CHAIN


def _keyed_providers(names: Iterable[str]) -> List[Tuple[str, str]]:
    """``(name, key)`` for each provider in ``names`` whose key is configured."""
    keyed = []
    for name in names:
        key = _env_value(PROVIDERS[name]["key_env"])
        if key:
            keyed.append((name, key))
    return keyed


def check_typesafe_requirements() -> bool:
    return bool(_keyed_providers(provider_chain()))


def _missing_keys_message() -> str:
    names = provider_chain()
    envs = " or ".join(PROVIDERS[name]["key_env"] for name in names)
    return f"{envs} is not configured (provider chain: {', '.join(names)})"


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
    """A System One call that produced no usable answer."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        retryable: bool = False,
        detail: str = "",
        provider: str = "",
    ):
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.detail = detail
        self.provider = provider


def _post(
    provider: str, key: str, payload: Dict[str, Any], timeout: float
) -> Dict[str, Any]:
    """One request to one provider; raises :class:`JevError` on any failure."""
    spec = PROVIDERS[provider]
    try:
        response = httpx.post(
            spec["url"],
            json=payload,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )
    except httpx.HTTPError as exc:
        raise JevError(
            f"{spec['label']} request failed: {exc.__class__.__name__}",
            retryable=True,
            provider=provider,
        ) from exc
    if response.status_code != 200:
        try:
            detail = str(response.text or "")[:500]
        except Exception:
            detail = ""
        meaning = _STATUS_MEANINGS.get(response.status_code, "unexpected status")
        raise JevError(
            f"{spec['label']} returned HTTP {response.status_code}: {meaning}",
            status=response.status_code,
            retryable=response.status_code in _RETRYABLE_STATUSES,
            detail=detail,
            provider=provider,
        )
    try:
        body = response.json()
    except ValueError as exc:
        raise JevError(
            f"{spec['label']} returned a non-JSON body", provider=provider
        ) from exc
    if not isinstance(body, dict):
        raise JevError(
            f"{spec['label']} returned an unexpected body", provider=provider
        )
    body["provider"] = provider
    return body


def ask_jev(
    state: Any,
    questions: Dict[str, Dict[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
    timeout: float = TIMEOUT_SECONDS,
    api_key: str | None = None,
    providers: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """POST one System One request and return the decoded response body.

    Shared by the ``typesafe_decide`` tool and the system-one-preflight
    plugin. Providers are tried in chain order; a failure other than a
    rejected request body moves to the next one. The returned body carries
    ``provider`` naming the one that answered. Raises :class:`JevError`
    once every provider has failed, so callers decide how much a missing
    answer matters to them.

    An explicit ``api_key`` belongs to no provider in particular, so it is
    sent once, to the first provider in the chain.
    """
    payload = {"state": state, "model": model, "questions": questions}
    chain = tuple(name for name in (providers or provider_chain()) if name in PROVIDERS)
    if not chain:
        chain = provider_chain()
    if api_key:
        return _post(chain[0], api_key, payload, timeout)

    attempts = _keyed_providers(chain)
    if not attempts:
        raise JevError(_missing_keys_message())

    failures: List[JevError] = []
    for index, (name, key) in enumerate(attempts):
        try:
            return _post(name, key, payload, timeout)
        except JevError as exc:
            failures.append(exc)
            if index == len(attempts) - 1 or exc.status in _NO_FALLTHROUGH_STATUSES:
                break
    if len(failures) == 1:
        raise failures[0]
    last = failures[-1]
    raise JevError(
        "; ".join(str(failure) for failure in failures),
        status=last.status,
        retryable=any(failure.retryable for failure in failures),
        detail=last.detail,
        provider=last.provider,
    )


def typesafe_decide(args: Dict[str, Any]) -> str:
    """Ask Jev typed questions about ``state`` and return its typed answers."""
    if not check_typesafe_requirements():
        return json.dumps({"error": _missing_keys_message()})

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
            "provider": body.get("provider", ""),
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
        "(places the state on ordered criteria levels, with confidence). Reached "
        "through TypeSafe directly or through OpenRouter, as configured; the "
        "result names which one answered."
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
                "description": (
                    "Jev model id. Defaults to jev-latest, which both providers accept."
                ),
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
