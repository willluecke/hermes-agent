"""The single-model runtime contract (Hermes Chat's direct model picker).

``execution_mode=single_model`` is an exact, no-fallback parent-runtime
contract: the selected provider/model is the only LLM that can answer the
turn, so the generic model-spawning ``delegation`` toolset is always denied.

Until 2026-11-17 this module was ``agent/opus_delegation.py`` and carried one
exception: a GPT-6 Astra parent could hand implementation to a governed Opus
worker (``opus_code_worker``). Will retired delegation to Opus that day; Opus
now takes part as a per-conversation Claude advisor or through read-only
``model_consult``. What remains is the denial list and the read-only pin.
"""

from __future__ import annotations

from typing import Any


def _normalize(value: Any) -> str:
    return str(value or "").strip().lower()


def single_model_disabled_toolsets(provider: Any = None, model: Any = None) -> list[str]:
    """Toolsets denied to a single-model parent runtime: never a model spawner."""
    return ["delegation"]


# Providers whose single-model runs are pinned to the built-in ``read_only``
# toolset (an allowlist: read_file, search_files, web_search, web_extract,
# vision_analyze, skills_list, skill_view, session_search, current_time,
# kanban_show, kanban_list, kanban_attachments). Untrusted third-party
# endpoints whose safety alignment the operator does not rely on belong
# here; the pin strips terminal execution, file mutation, patch, browser,
# message posting, cron, memory/decision writes and MCP tools from those
# runs without touching any other provider.
SINGLE_MODEL_READ_ONLY_PROVIDERS: frozenset[str] = frozenset(
    {"custom:abliteration"}
)


def single_model_read_only_toolset(provider: Any) -> list[str] | None:
    """The allowlist toolset for a single-model parent runtime, or None.

    Returns ``["read_only"]`` when the *requested* provider identity (the
    ``custom:<name>`` slug the picker sends; the resolved runtime collapses
    named customs to plain ``custom``) is pinned to the read-only allowlist,
    else ``None`` so the caller keeps its normal enabled/disabled toolset
    resolution. Callers pass the request's provider string, so a route or
    fallback cannot smuggle a pinned runtime out of the sandbox.
    """
    if _normalize(provider) in SINGLE_MODEL_READ_ONLY_PROVIDERS:
        return ["read_only"]
    return None
