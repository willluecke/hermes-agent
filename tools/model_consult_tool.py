#!/usr/bin/env python3
"""Read-only cross-provider consultations through Hermes' secured router."""

from __future__ import annotations

import json
from typing import Any

from tools.registry import registry


_MAX_PROMPT_CHARS = 50_000
_MAX_CONTEXT_CHARS = 50_000
_MAX_RESPONSE_CHARS = 40_000
_DEFAULT_TIMEOUT_SECONDS = 120


def _clean_text(value: Any, name: str, limit: int, *, required: bool = False) -> str:
    text = str(value or "").strip()
    if required and not text:
        raise ValueError(f"{name} is required")
    if len(text) > limit:
        raise ValueError(f"{name} exceeds the {limit:,}-character limit")
    return text


def _inventory_rows() -> list[dict[str, Any]]:
    from hermes_cli.inventory import build_models_payload, load_picker_context

    payload = build_models_payload(
        load_picker_context(),
        explicit_only=True,
        picker_hints=True,
        probe_custom_providers=False,
        for_picker=True,
        max_models=500,
    )
    rows = payload.get("providers") if isinstance(payload, dict) else []
    return [row for row in rows if isinstance(row, dict)]


def _provider_row(provider: str) -> dict[str, Any]:
    normalized = provider.strip().lower()
    row = next(
        (
            candidate
            for candidate in _inventory_rows()
            if str(candidate.get("slug") or "").strip().lower() == normalized
        ),
        None,
    )
    if row is None or row.get("authenticated") is False:
        raise ValueError(f"provider '{provider}' is not authenticated in Hermes")
    return row


def _list_models(provider: str = "") -> dict[str, Any]:
    provider = str(provider or "").strip().lower()
    rows = _inventory_rows()
    if provider:
        row = next(
            (
                candidate
                for candidate in rows
                if str(candidate.get("slug") or "").strip().lower() == provider
            ),
            None,
        )
        if row is None or row.get("authenticated") is False:
            raise ValueError(f"provider '{provider}' is not authenticated in Hermes")
        return {
            "ok": True,
            "provider": provider,
            "models": [str(model) for model in row.get("models") or []],
        }
    return {
        "ok": True,
        "providers": [
            {
                "provider": str(row.get("slug") or ""),
                "models": [str(model) for model in row.get("models") or []],
            }
            for row in rows
            if row.get("authenticated") is not False and row.get("slug")
        ],
    }


def _call_provider(
    *,
    provider: str,
    model: str,
    prompt: str,
    context: str,
    timeout_seconds: int,
) -> tuple[str, dict[str, str]]:
    from agent.auxiliary_client import call_llm, extract_content_or_reasoning

    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a read-only advisor consulted by another engineering agent. "
                "Answer the requested question directly and concisely. Do not claim to "
                "have edited files, run tools, or performed external actions."
            ),
        }
    ]
    if context:
        messages.append({"role": "user", "content": f"Context:\n{context}"})
    messages.append({"role": "user", "content": prompt})
    route: dict[str, str] = {}
    response = call_llm(
        task="model_consult",
        provider=provider,
        model=model,
        messages=messages,
        timeout=float(timeout_seconds),
        route_info=route,
    )
    return extract_content_or_reasoning(response).strip(), route


def _consult(
    *,
    provider: str,
    model: str,
    prompt: str,
    context: str = "",
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    provider = _clean_text(provider, "provider", 64, required=True).lower()
    model = _clean_text(model, "model", 256, required=True)
    prompt = _clean_text(prompt, "prompt", _MAX_PROMPT_CHARS, required=True)
    context = _clean_text(context, "context", _MAX_CONTEXT_CHARS)
    timeout_seconds = max(10, min(int(timeout_seconds or _DEFAULT_TIMEOUT_SECONDS), 300))

    row = _provider_row(provider)
    models = [str(candidate) for candidate in row.get("models") or []]
    if model not in models:
        raise ValueError(
            f"model '{model}' is not advertised by authenticated provider "
            f"'{provider}'; call model_consult with action='list' first"
        )

    answer, route = _call_provider(
        provider=provider,
        model=model,
        prompt=prompt,
        context=context,
        timeout_seconds=timeout_seconds,
    )
    if not answer:
        raise RuntimeError("the consulted model returned no text")
    truncated = len(answer) > _MAX_RESPONSE_CHARS
    if truncated:
        answer = answer[:_MAX_RESPONSE_CHARS] + "\n\n[consultation output truncated]"
    return {
        "ok": True,
        "provider": route.get("provider") or provider,
        "model": route.get("model") or model,
        "response": answer,
        "truncated": truncated,
    }


def model_consult_tool(action: str, **kwargs: Any) -> str:
    """List routed models or request one bounded, read-only consultation."""
    try:
        normalized = str(action or "").strip().lower()
        if normalized == "list":
            result = _list_models(str(kwargs.get("provider") or ""))
        elif normalized == "consult":
            result = _consult(
                provider=str(kwargs.get("provider") or ""),
                model=str(kwargs.get("model") or ""),
                prompt=str(kwargs.get("prompt") or ""),
                context=str(kwargs.get("context") or ""),
                timeout_seconds=int(
                    kwargs.get("timeout_seconds") or _DEFAULT_TIMEOUT_SECONDS
                ),
            )
        else:
            raise ValueError("action must be 'list' or 'consult'")
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)


MODEL_CONSULT_SCHEMA = {
    "name": "model_consult",
    "description": (
        "List authenticated Hermes provider models or consult one model through "
        "Hermes' secured provider router. Use this whenever the user asks you to "
        "talk to, ask, consult, or get an opinion from another model such as GLM. "
        "Do not invoke an unrelated local CLI with --model. This tool is read-only, "
        "has no tools in the child call, and never exposes provider credentials."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["list", "consult"]},
            "provider": {
                "type": "string",
                "description": "Hermes provider slug, for example openrouter.",
            },
            "model": {
                "type": "string",
                "description": "Exact provider model id returned by action='list'.",
            },
            "prompt": {"type": "string"},
            "context": {
                "type": "string",
                "description": "Only the bounded context the advisor needs.",
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 10,
                "maximum": 300,
            },
        },
        "required": ["action"],
    },
}


registry.register(
    name="model_consult",
    toolset="model_consult",
    schema=MODEL_CONSULT_SCHEMA,
    handler=lambda args, **kwargs: model_consult_tool(**args),
    check_fn=lambda: True,
    emoji="",
    max_result_size_chars=50_000,
)
