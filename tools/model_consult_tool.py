#!/usr/bin/env python3
"""Read-only advice through an explicitly selected authenticated model."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory
from typing import Any

from tools.registry import registry


_MAX_PROMPT_CHARS = 50_000
_MAX_CONTEXT_CHARS = 50_000
_MAX_RESPONSE_CHARS = 40_000
_DEFAULT_TIMEOUT_SECONDS = 120
_CLAUDE_ALIASES = ("fable", "opus", "sonnet", "haiku")
_ADVISOR_SYSTEM = (
    "You are a read-only advisor consulted by another assistant. "
    "Answer the requested question directly and concisely using only the supplied "
    "context. You have no tools. Do not claim to have edited files, run tools, "
    "or performed external actions."
)


def _native_claude_model(model: str) -> str:
    candidate = model.lower()
    for prefix in ("anthropic/", "claude-code/"):
        if candidate.startswith(prefix):
            candidate = candidate[len(prefix):]
            break
    if candidate in _CLAUDE_ALIASES or re.fullmatch(r"claude-[a-z0-9]+(?:-[a-z0-9]+)*", candidate):
        return candidate
    return ""


def _native_claude_row() -> dict[str, Any]:
    from agent.transports.claude_code_session import (
        claude_subscription_auth_available, find_claude_binary,
    )
    from hermes_cli.models import _PROVIDER_MODELS

    # Catalog entries are discovery hints, not a Max entitlement check. Native
    # Claude can accept a new exact model before any API catalog includes it.
    models = list(dict.fromkeys([
        *_PROVIDER_MODELS.get("anthropic", []), *_CLAUDE_ALIASES,
    ]))
    available = False
    try:
        find_claude_binary()
        available = claude_subscription_auth_available()
    except Exception:
        pass
    return {
        "slug": "claude-code", "authenticated": available, "models": models,
        "transport": "native_cli", "billing": "subscription", "preferred": True,
        "note": (
            "Native Claude subscription route; not dependent on the API model list. "
            "Exact Claude model IDs and native aliases are accepted; the CLI checks "
            "model access when called. No paid API fallback."
            + ("" if available else " Claude subscription login or CLI is unavailable; restore it before retrying.")
        ),
    }


def _native_codex_row() -> dict[str, Any]:
    """The picker uses Codex app-server models and the ChatGPT login."""
    from hermes_cli.codex_models import get_codex_app_server_model_ids

    binary = shutil.which("codex")
    authenticated = False
    if binary:
        try:
            status = subprocess.run(
                [binary, "login", "status"], capture_output=True, text=True,
                timeout=10, check=False,
            )
            # `codex login status` prints "Logged in using ChatGPT" on stderr
            # (0.160 and 0.170 both, checked 2026-11-18); reading stdout alone
            # reported every Codex consult as "login unavailable".
            output = f"{getattr(status, 'stdout', '') or ''}\n{getattr(status, 'stderr', '') or ''}".lower()
            authenticated = status.returncode == 0 and "chatgpt" in output
        except (OSError, subprocess.TimeoutExpired):
            pass
    try:
        models = get_codex_app_server_model_ids(force_refresh=False) if binary else []
    except Exception:
        models = []
    return {
        "slug": "openai-codex", "authenticated": authenticated,
        "models": list(dict.fromkeys(models)),
        "transport": "native_cli", "billing": "subscription",
        "note": "Uses the Codex ChatGPT login in a fresh read-only session; no API-key fallback.",
    }


def _clean_text(value: Any, name: str, limit: int, *, required: bool = False) -> str:
    text = str(value or "").strip()
    if required and not text:
        raise ValueError(f"{name} is required")
    if len(text) > limit:
        raise ValueError(f"{name} exceeds the {limit:,}-character limit")
    return text


def _api_inventory_rows() -> list[dict[str, Any]]:
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


def _inventory_rows() -> list[dict[str, Any]]:
    native = _native_claude_row()
    codex = _native_codex_row()
    try:
        api_rows = _api_inventory_rows()
    except Exception:
        # API discovery must not conceal a working subscription route.
        native["note"] += " API provider discovery is temporarily unavailable."
        api_rows = []
    return [native, codex, *(row for row in api_rows if row.get("slug") not in {"claude-code", "openai-codex"})]


def _provider_row(provider: str) -> dict[str, Any]:
    normalized = provider.strip().lower()
    if normalized == "claude-code":
        row = _native_claude_row()
        if not row["authenticated"]:
            raise ValueError(
                "Claude subscription authentication or CLI is unavailable. Restore the "
                "native Claude login and retry; no paid API fallback was attempted."
            )
        return row
    if normalized == "openai-codex":
        row = _native_codex_row()
        if not row["authenticated"]:
            raise ValueError("Codex ChatGPT login is unavailable. No paid API fallback was attempted.")
        return row
    row = next(
        (
            candidate
            for candidate in _api_inventory_rows()
            if str(candidate.get("slug") or "").strip().lower() == normalized
        ),
        None,
    )
    if row is None or row.get("authenticated") is False:
        raise ValueError(f"provider '{provider}' is not authenticated in Hermes")
    return row


def _list_models(provider: str = "") -> dict[str, Any]:
    provider = str(provider or "").strip().lower()
    rows = ([_native_claude_row()] if provider == "claude-code" else
            [_native_codex_row()] if provider == "openai-codex" else _inventory_rows())

    def describe(row: dict[str, Any]) -> dict[str, Any]:
        result = {
            "provider": str(row.get("slug") or ""),
            "models": [str(model) for model in row.get("models") or []],
        }
        for field in ("authenticated", "transport", "billing", "preferred", "note"):
            if field in row:
                result[field] = row[field]
        return result
    if provider:
        row = next(
            (
                candidate
                for candidate in rows
                if str(candidate.get("slug") or "").strip().lower() == provider
            ),
            None,
        )
        if row is None or (row.get("authenticated") is False and provider not in {"claude-code", "openai-codex"}):
            raise ValueError(f"provider '{provider}' is not authenticated in Hermes")
        return {
            "ok": True,
            **describe(row),
        }
    return {
        "ok": True,
        "providers": [
            describe(row)
            for row in rows
            if row.get("slug") and (row.get("authenticated") is not False or row.get("slug") in {"claude-code", "openai-codex"})
        ],
        "routing_policy": "Claude and Codex use their native subscription routes by default, never automatic paid API fallback. An API-only list does not describe native subscription availability.",
    }


def _advisor_prompt(prompt: str, context: str) -> str:
    return f"Context:\n{context}\n\nQuestion:\n{prompt}" if context else prompt


def _call_native_claude(
    *, model: str, prompt: str, context: str, timeout_seconds: int,
    effort: str = "",
) -> tuple[str, dict[str, Any]]:
    from agent.transports.claude_code_session import ClaudeCodeSession

    # A fresh, no-tools consultation is neither the parent's execution thread
    # nor a writable worker. Never bring project instructions or MCPs along.
    with TemporaryDirectory(prefix="hermes-consult-") as cwd:
        session = ClaudeCodeSession(
            cwd=cwd, model=model, system_prompt=_ADVISOR_SYSTEM, no_tools=True,
            effort=effort or None,
            inactivity_timeout=timeout_seconds, absolute_timeout=timeout_seconds,
        )
        try:
            result = session.run_turn(_advisor_prompt(prompt, context))
        except Exception:
            raise RuntimeError("Native Claude consultation could not start. No paid API fallback was attempted; check the native Claude login and logs.") from None
        finally:
            session.close()
        if result.interrupted:
            raise RuntimeError("Native Claude consultation was interrupted; no paid API fallback was attempted.")
        if result.error:
            # Never return stderr/debug material through the consultation
            # tool: it can contain host details or provider credentials.
            raise RuntimeError("Native Claude consultation failed (login, model access, quota, or timeout). No paid API fallback was attempted; check the native Claude logs.")
        return result.final_text.strip(), {
            "provider": "claude-code", "model": model,
            "transport": "native_cli", "billing": "subscription",
            "usage": result.usage,
        }


def _call_native_codex(
    *, model: str, prompt: str, context: str, timeout_seconds: int,
    effort: str = "",
) -> tuple[str, dict[str, Any]]:
    from agent.transports.codex_app_server_session import CodexAppServerSession

    # A fresh temporary workspace keeps this consult separate from the parent
    # conversation. Read-only mode disables external tools and write access.
    with TemporaryDirectory(prefix="hermes-consult-") as cwd:
        effective_effort = effort or "medium"
        session = CodexAppServerSession(
            cwd=cwd, model=model, effort=effective_effort,
            read_only=True, require_exact=True, project_key="general",
            first_event_timeout=min(timeout_seconds, 60),
        )
        try:
            result = session.run_turn(
                f"{_ADVISOR_SYSTEM}\n\n{_advisor_prompt(prompt, context)}",
                turn_timeout=timeout_seconds,
                absolute_turn_timeout=timeout_seconds,
                post_compaction_grace=0,
            )
        except Exception:
            raise RuntimeError("Native Codex consultation could not start. No paid API fallback was attempted; check the Codex login and logs.") from None
        finally:
            session.close()
        if result.interrupted or result.error:
            raise RuntimeError("Native Codex consultation failed (login, model access, quota, or timeout). No paid API fallback was attempted; check the Codex logs.")
        if result.tool_iterations:
            raise RuntimeError("Native Codex consultation attempted to use tools; its answer was discarded.")
        return result.final_text.strip(), {
            "provider": "openai-codex", "model": model,
            "transport": "native_cli", "billing": "subscription",
            "effort": effective_effort,
            "usage": result.token_usage_last or result.token_usage_total or {},
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
            "content": _ADVISOR_SYSTEM,
        }
    ]
    messages.append({"role": "user", "content": _advisor_prompt(prompt, context)})
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
    allow_paid_claude: bool = False,
    effort: str = "",
) -> dict[str, Any]:
    provider = _clean_text(provider, "provider", 64).lower()
    model = _clean_text(model, "model", 256, required=True)
    prompt = _clean_text(prompt, "prompt", _MAX_PROMPT_CHARS, required=True)
    context = _clean_text(context, "context", _MAX_CONTEXT_CHARS)
    effort = _clean_text(effort, "effort", 16).lower()
    if effort and effort not in {"low", "medium", "high", "xhigh", "max", "ultra"}:
        raise ValueError("effort must be low, medium, high, xhigh, max, or ultra")
    timeout_seconds = max(10, min(int(timeout_seconds or _DEFAULT_TIMEOUT_SECONDS), 300))

    if not isinstance(allow_paid_claude, bool):
        raise ValueError("allow_paid_claude must be a boolean")
    native_model = _native_claude_model(model)
    requested_provider = provider
    if native_model and (not allow_paid_claude or provider in {"", "auto", "claude-code", "claude"}):
        provider, model = "claude-code", native_model
    elif provider in {"codex", "openai-codex"} or (
        provider in {"", "auto"} and re.fullmatch(r"gpt-[a-z0-9]+(?:[.-][a-z0-9]+)*", model.lower())
    ):
        provider = "openai-codex"
    elif provider in {"", "auto"}:
        raise ValueError("Select an authenticated provider from action='list' for this model")
    elif not native_model and (provider in {"claude-code", "claude"} or ("claude" in model.lower() and not allow_paid_claude)):
        raise ValueError("Use an exact native Claude model ID or alias; paid Claude API routing requires explicit user approval and allow_paid_claude=true")

    row = _provider_row(provider)
    if provider == "claude-code" and effort in {"xhigh", "ultra"}:
        raise ValueError("Claude consultation effort must be low, medium, high, or max")
    if effort and provider not in {"claude-code", "openai-codex"}:
        raise ValueError("effort is supported only for native Claude or Codex consultation")
    models = [str(candidate) for candidate in row.get("models") or []]
    if provider != "claude-code" and model not in models:
        raise ValueError(
            f"model '{model}' is not advertised by authenticated provider "
            f"'{provider}'; call model_consult with action='list' first"
        )

    call = (_call_native_claude if provider == "claude-code" else
            _call_native_codex if provider == "openai-codex" else _call_provider)
    answer, route = call(
        **({"provider": provider} if provider not in {"claude-code", "openai-codex"} else {}),
        model=model,
        prompt=prompt,
        context=context,
        timeout_seconds=timeout_seconds,
        **({"effort": effort} if provider in {"claude-code", "openai-codex"} else {}),
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
        "requested_provider": requested_provider or "auto",
        **{key: route[key] for key in ("transport", "billing") if key in route},
        **({"effort": route.get("effort") or effort} if route.get("effort") or effort else {}),
        **({"usage": route["usage"]} if "usage" in route else {}),
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
                allow_paid_claude=kwargs.get("allow_paid_claude", False),
                effort=str(kwargs.get("effort") or ""),
            )
        else:
            raise ValueError("action must be 'list' or 'consult'")
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)


MODEL_CONSULT_SCHEMA = {
    "name": "model_consult",
    "description": (
        "List native subscription and authenticated API routes, or consult one model. "
        "Claude uses native Claude Code and Codex uses the ChatGPT subscription by default, "
        "even if an API catalog omits it. Never infer native availability from an API-only "
        "list, switch to OpenRouter, or substitute another model after a native failure. "
        "Paid Claude API use requires explicit user approval and allow_paid_claude=true. "
        "Use this whenever the user asks you to "
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
                "description": "Use claude-code for Claude or openai-codex for ChatGPT subscription advice (also their defaults). Direct API models require an exact listed provider, e.g. openrouter.",
            },
            "model": {
                "type": "string",
                "description": "Exact model ID listed for the chosen provider; native Claude also accepts exact claude-* IDs and native aliases fable, opus, sonnet, haiku. The CLI verifies access.",
            },
            "allow_paid_claude": {
                "type": "boolean", "default": False,
                "description": "Only set true if the user explicitly approved metered Claude API usage on the specified provider. Never set merely because native login, quota, model access, or a tool list failed.",
            },
            "prompt": {"type": "string"},
            "effort": {
                "type": "string", "enum": ["low", "medium", "high", "xhigh", "max", "ultra"],
                "description": "Native Claude or Codex consultation effort. Codex defaults to medium when omitted; Claude uses the CLI default.",
            },
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
