"""Governed Opus delegation contract shared by parent runtimes and children.

`execution_mode=single_model` is an exact, no-fallback parent-runtime
contract: the selected provider/model is the only LLM that can answer the
turn, so every model-spawning tool stays denied. One bounded exception
exists — a Hermes API single-model parent whose *resolved* provider is
``openai-codex`` and whose model is GPT-6 Astra keeps the governed
``opus_code_worker`` handoff (generic ``delegate_task`` remains disabled).
Every other single-model selection keeps both denials.

The Codex app-server runtime hands the turn to a subprocess that spawns the
hermes-tools stdio MCP server in a restricted environment. That child builds
its own tool list and cannot see the parent's ``disabled_toolsets``, so the
parent's decision has to travel with it. This module owns:

  * the eligibility predicate both sides evaluate,
  * the non-secret env keys that carry the parent runtime to a managed child
    (whitelisted in the managed Codex MCP entry's ``env_vars``), and
  * the authority validation the child applies before exposing Opus.

Everything here is non-secret: provider slug, model slug, reasoning effort,
and a boolean. No credentials are ever propagated.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

#: Non-secret per-session parent runtime facts handed to a managed child.
PARENT_PROVIDER_ENV = "HERMES_PARENT_PROVIDER"
PARENT_MODEL_ENV = "HERMES_PARENT_MODEL"
PARENT_EFFORT_ENV = "HERMES_PARENT_EFFORT"
PARENT_OPUS_WORKER_ENV = "HERMES_PARENT_OPUS_WORKER"

#: Whitelist order used by the managed Codex MCP entry's ``env_vars``.
PARENT_RUNTIME_ENV_VARS: tuple[str, ...] = (
    PARENT_PROVIDER_ENV,
    PARENT_MODEL_ENV,
    PARENT_EFFORT_ENV,
    PARENT_OPUS_WORKER_ENV,
)

#: The exact persistent orchestrator that may delegate implementation work.
ORCHESTRATOR_MODEL = "gpt-5.6-sol"
ORCHESTRATOR_EFFORT = "xhigh"
ORCHESTRATOR_PROVIDERS = frozenset({"openai", "openai-codex"})

#: The bounded single-model exception: exact provider + model, any effort.
SINGLE_MODEL_OPUS_PROVIDER = "openai-codex"
SINGLE_MODEL_OPUS_MODELS = frozenset({"gpt-6-astra"})


def _normalize(value: Any) -> str:
    return str(value or "").strip().lower()


def single_model_allows_opus_worker(provider: Any, model: Any) -> bool:
    """Whether a single-model parent keeps the governed Opus worker.

    This is the only exception to the single-model denial list, and it is
    keyed off the *resolved* provider so an alias or a route cannot smuggle
    a different runtime into the exception.
    """
    return (
        _normalize(provider) == SINGLE_MODEL_OPUS_PROVIDER
        and _normalize(model) in SINGLE_MODEL_OPUS_MODELS
    )


def single_model_disabled_toolsets(provider: Any, model: Any) -> list[str]:
    """Toolsets denied to a single-model parent runtime.

    ``delegation`` is denied unconditionally: single-model mode never gains a
    generic model-spawning tool. ``opus_worker`` is denied unless the bounded
    GPT-6 Astra exception applies.
    """
    if single_model_allows_opus_worker(provider, model):
        return ["delegation"]
    return ["delegation", "opus_worker"]


def parent_runtime_env(
    *,
    provider: Any,
    model: Any,
    effort: Any = "",
    opus_worker_enabled: bool = False,
) -> dict[str, str]:
    """Build the non-secret parent-runtime environment for a managed child.

    Returns an empty mapping when the parent runtime is not fully known; the
    child then falls back to its config-derived authority instead of being
    handed a half-populated contract it cannot validate.
    """
    resolved_provider = _normalize(provider)
    resolved_model = _normalize(model)
    if not resolved_provider or not resolved_model:
        return {}
    return {
        PARENT_PROVIDER_ENV: resolved_provider,
        PARENT_MODEL_ENV: resolved_model,
        PARENT_EFFORT_ENV: _normalize(effort),
        PARENT_OPUS_WORKER_ENV: "1" if opus_worker_enabled else "0",
    }


def parent_runtime_from_env(
    env: Optional[Mapping[str, str]] = None,
) -> Optional[dict[str, Any]]:
    """Read the propagated parent runtime, or None when it is absent.

    ``None`` means "this process was not handed runtime metadata" — native
    and non-gateway paths that predate the propagation keep their existing
    config-derived authority. It never means "unrestricted".
    """
    if env is None:
        import os

        env = os.environ
    provider = _normalize(env.get(PARENT_PROVIDER_ENV))
    model = _normalize(env.get(PARENT_MODEL_ENV))
    if not provider and not model:
        return None
    return {
        "provider": provider,
        "model": model,
        "effort": _normalize(env.get(PARENT_EFFORT_ENV)),
        "opus_worker_enabled": _normalize(env.get(PARENT_OPUS_WORKER_ENV))
        in {"1", "true", "yes", "on"},
    }


def validate_parent_authority(runtime: Mapping[str, Any]) -> dict[str, str]:
    """Return the accepted delegation authority, or raise ``ValueError``.

    Accepts the exact ``gpt-5.6-sol`` xhigh orchestrator and an eligible
    single-model GPT-6 Astra parent. Anything else is a disallowed direct
    parent and must not reach the Opus worker.
    """
    provider = _normalize(runtime.get("provider"))
    model = _normalize(runtime.get("model"))
    effort = _normalize(runtime.get("effort"))
    if not runtime.get("opus_worker_enabled"):
        raise ValueError(
            "the parent runtime disabled opus_worker; Opus delegation is "
            "unavailable for this turn"
        )
    eligible_single_model = single_model_allows_opus_worker(provider, model)
    exact_orchestrator = (
        provider in ORCHESTRATOR_PROVIDERS
        and model == ORCHESTRATOR_MODEL
        and effort == ORCHESTRATOR_EFFORT
    )
    if not (eligible_single_model or exact_orchestrator):
        raise ValueError(
            "Opus delegation requires exact gpt-5.6-sol at xhigh or an eligible "
            "openai-codex gpt-6-astra parent; got "
            f"{provider or 'unknown'}/{model or 'unknown'} at {effort or 'unknown'}"
        )
    return {
        "provider": provider,
        "model": model,
        "effort": effort,
        "runtime": "codex_app_server",
    }


def authority_label(authority: Mapping[str, Any]) -> str:
    """Human-readable name of the authority that accepted a specification."""
    model = str(authority.get("model") or "").strip() or "unknown-model"
    effort = str(authority.get("effort") or "").strip()
    return f"Hermes/{model}" + (f" at {effort}" if effort else "")


__all__ = [
    "PARENT_PROVIDER_ENV",
    "PARENT_MODEL_ENV",
    "PARENT_EFFORT_ENV",
    "PARENT_OPUS_WORKER_ENV",
    "PARENT_RUNTIME_ENV_VARS",
    "ORCHESTRATOR_MODEL",
    "ORCHESTRATOR_EFFORT",
    "ORCHESTRATOR_PROVIDERS",
    "SINGLE_MODEL_OPUS_PROVIDER",
    "SINGLE_MODEL_OPUS_MODELS",
    "single_model_allows_opus_worker",
    "single_model_disabled_toolsets",
    "parent_runtime_env",
    "parent_runtime_from_env",
    "validate_parent_authority",
    "authority_label",
]
