"""Canonical reasoning-effort vocabulary and wire clamping.

Hermes' internal effort ladder (``hermes_constants.VALID_REASONING_EFFORTS``
plus the ``none`` disable level) is wider than what any single provider wire
accepts. Historically every transport and provider profile hand-rolled its own
translation map, and the class of bugs that produced was constant: a new
internal level (``ultra``) leaking to a wire that rejects it with HTTP 400
(#89503, #70058), or an unknown level being dropped to a weak default so the
strongest ask resolved *weaker* than an explicit ``high`` — a ladder
inversion (#74295, #87279).

This module is the single source of truth both kinds of code use instead:

- :data:`EFFORT_LADDER` — canonical low→high ordering.
- :func:`clamp_effort` — the one clamping policy: keep a supported level
  verbatim, otherwise take the **nearest weaker** supported level (never
  silently escalate cost above what was asked), and only when nothing weaker
  exists take the weakest supported level (a provider whose minimum thinking
  level is ``high`` serves ``high`` for a ``low`` ask — GLM-5.2's shape).
- Named wire-vocabulary constants for the common OpenAI-compatible surfaces,
  so call sites declare *data* ("this route accepts these levels") rather
  than logic.

Rules for call sites:

1. **Wire shape stays local.** Whether a route wants ``extra_body.reasoning``,
   a top-level ``reasoning_effort`` string, or a ``thinking`` toggle is the
   caller's business. Only the *vocabulary math* lives here.
2. **Unset stays unset.** ``clamp_effort`` translates an explicit request; it
   does not invent one. When the user expressed no effort, prefer omitting
   the field so the server default applies.
3. **Never patch a predicate.** When a provider rejects a level, fix its
   declared supported set (data), never add another vendor-name special case
   at the call site.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

#: K3 slug detector — matches ``k3`` as a delimited token (``k3``,
#: ``k3-256k``, ``kimi-k3``, ``kimi-k3-cot``) without matching K2-era names
#: (``kimi-k2.6``). From #76427 by @ruizanthony.
_KIMI_K3_SLUG_RE = re.compile(r"(?:^|[^a-z0-9])k3(?:[^a-z0-9]|$)")

# Canonical low→high ordering used for nearest-level clamping. Superset of
# hermes_constants.VALID_REASONING_EFFORTS ("none" included so an explicit
# disable can be clamped too when a provider publishes it as a level).
EFFORT_LADDER: tuple[str, ...] = (
    "none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra",
)

# ``ultra`` is primarily Hermes/Codex product vocabulary. OpenAI-compatible
# provider wires stop at ``max`` and clamp it down, while the native Codex app
# server advertises it for selected models such as GPT-5.6 Sol and Terra.

#: The widest OpenAI-compatible wire vocabulary (OpenRouter, Nous Portal):
#: exactly max|xhigh|high|medium|low|minimal|none.
OPENAI_COMPAT_WIRE_EFFORTS: tuple[str, ...] = (
    "none", "minimal", "low", "medium", "high", "xhigh", "max",
)

#: OpenAI/Codex Responses backend — per-model vocabulary, live-verified
#: (Aug 2026): ``minimal`` is rejected by both generations (clamps to low);
#: ``max`` is gpt-5.6-only — gpt-5.5 rejects it with "Supported values are:
#: 'none', 'low', 'medium', 'high', 'xhigh'" (#68365's premise, confirmed).
CODEX_GPT56_EFFORTS: tuple[str, ...] = (
    "none", "low", "medium", "high", "xhigh", "max",
)
CODEX_LEGACY_EFFORTS: tuple[str, ...] = (
    "none", "low", "medium", "high", "xhigh",
)

#: Native Codex app-server vocabulary reported by ``model/list``. This differs
#: from the OpenAI Responses API: app-server does not expose ``none`` and the
#: current GPT-5.6 family may expose ``ultra``.
CODEX_APP_SERVER_GPT56_SOL_TERRA_EFFORTS: tuple[str, ...] = (
    "low", "medium", "high", "xhigh", "max", "ultra",
)
CODEX_APP_SERVER_GPT56_LUNA_EFFORTS: tuple[str, ...] = (
    "low", "medium", "high", "xhigh", "max",
)
CODEX_APP_SERVER_LEGACY_EFFORTS: tuple[str, ...] = (
    "low", "medium", "high", "xhigh",
)

#: Claude Code subscription runtime. ``auto`` is represented by omitting the
#: effort field entirely; these are the explicit values accepted by the native
#: CLI. Keep this provider vocabulary separate from the OpenAI-compatible
#: ladder so browser pickers never offer ``minimal``/``none`` to Claude.
CLAUDE_CODE_EFFORTS: tuple[str, ...] = (
    "low", "medium", "high", "xhigh", "max",
)


def codex_supported_efforts(model: Optional[str]) -> tuple[str, ...]:
    """Supported effort set for an OpenAI/Codex Responses model."""
    if "gpt-5.6" in (model or "").lower():
        return CODEX_GPT56_EFFORTS
    return CODEX_LEGACY_EFFORTS


def codex_app_server_supported_efforts(model: Optional[str]) -> tuple[str, ...]:
    """Supported effort set reported by the native Codex app server."""
    model_name = (model or "").lower()
    if "gpt-5.6-sol" in model_name or "gpt-5.6-terra" in model_name:
        return CODEX_APP_SERVER_GPT56_SOL_TERRA_EFFORTS
    if "gpt-5.6" in model_name:
        return CODEX_APP_SERVER_GPT56_LUNA_EFFORTS
    return CODEX_APP_SERVER_LEGACY_EFFORTS


#: Backward-compat alias (pre-#68365-verification name).
CODEX_RESPONSES_EFFORTS: tuple[str, ...] = CODEX_GPT56_EFFORTS

#: xAI Responses — Grok 4.6+ accepts xhigh; older Grok tops out at high.
XAI_GROK46_EFFORTS: tuple[str, ...] = ("low", "medium", "high", "xhigh")
XAI_LEGACY_EFFORTS: tuple[str, ...] = ("low", "medium", "high")

#: Actual Computer relays (SGLang/vLLM): none/low/medium/high/max.
ACTUAL_RELAY_EFFORTS: tuple[str, ...] = ("none", "low", "medium", "high", "max")

#: Moonshot/Kimi K3: low/high/max (server default high).
KIMI_K3_EFFORTS: tuple[str, ...] = ("low", "high", "max")
#: Moonshot/Kimi K2-era models: low/medium/high.
KIMI_K2_EFFORTS: tuple[str, ...] = ("low", "medium", "high")

#: OpenCode "Ox Alpha" stealth model (x-preview-f-free): thinking is always
#: on and the wire accepts exactly low/high/max — medium/none/xhigh 400 with
#: "This model always engages in thinking and cannot be disabled; please use
#: low, high, or max" (verified live 2026-08-21). xhigh rounds up to max.
OX_ALPHA_EFFORTS: tuple[str, ...] = ("low", "high", "max")
OX_ALPHA_OVERRIDES: dict[str, str] = {"xhigh": "max"}

#: Tencent TokenHub: low/medium/high.
TOKENHUB_EFFORTS: tuple[str, ...] = ("low", "medium", "high")

#: Kimi K3's vendor-documented translation quirks (platform.kimi.ai
#: thinking-model guide): ``high`` is K3's positional middle AND server
#: default, so ``medium`` rounds to it rather than down to ``low``; ``xhigh``
#: rounds up to ``max`` (K3's top tier), matching the kimi-coding plugin.
KIMI_K3_OVERRIDES: dict[str, str] = {"medium": "high", "xhigh": "max"}

#: GLM-5.2 native reasoning_effort knob: exactly two enabled levels,
#: ``high`` (its minimum thinking level) and ``max`` (per Z.AI/BigModel
#: docs). ``xhigh`` requests the top tier, not the floor.
GLM52_EFFORTS: tuple[str, ...] = ("high", "max")
GLM52_OVERRIDES: dict[str, str] = {"xhigh": "max"}

#: GLM-5.3 widens the knob to a graded low/medium/high/max scale — verified
#: live on api.z.ai/api/coding/paas/v4 (issue #91789, 2026-08-21): every
#: level accepted with monotonic reasoning-token scaling (low=4, medium=11,
#: high=98, max=125 on the probe prompt). ``xhigh`` requests the top tier.
GLM53_EFFORTS: tuple[str, ...] = ("low", "medium", "high", "max")
GLM53_OVERRIDES: dict[str, str] = {"xhigh": "max"}

#: DeepSeek V4 OpenAI-compat endpoint: low/medium/high/max; ``xhigh``
#: requests the top tier (matches the shipped profile mapping).
DEEPSEEK_V4_EFFORTS: tuple[str, ...] = ("low", "medium", "high", "max")
DEEPSEEK_V4_OVERRIDES: dict[str, str] = {"xhigh": "max"}

#: Ollama Cloud /v1/chat/completions: accepts {none, low, medium, high, max};
#: rejects ``minimal`` with HTTP 400. ``xhigh`` requests the top tier.
OLLAMA_CLOUD_EFFORTS: tuple[str, ...] = ("none", "low", "medium", "high", "max")
OLLAMA_CLOUD_OVERRIDES: dict[str, str] = {"xhigh": "max"}

#: Meta Model API (Muse): minimal..xhigh; rejects ``none``.
META_AI_EFFORTS: tuple[str, ...] = ("minimal", "low", "medium", "high", "xhigh")

#: Upstage Solar Pro/Open: low/medium/high.
SOLAR_EFFORTS: tuple[str, ...] = ("low", "medium", "high")


def kimi_supported_efforts(model: Optional[str]) -> tuple[str, ...]:
    """Supported effort set for a Moonshot/Kimi model slug.

    K3 is served as the bare slug ``k3``, plan variants like ``k3-256k``,
    and the ``kimi-k3*`` aliases; its documented set is low/high/max.
    Everything earlier speaks low/medium/high. Boundary-matched so K2-era
    names (``kimi-k2.6``) never match (detection regex from #76427 by
    @ruizanthony).
    """
    m = (model or "").strip().lower().split("/")[-1]
    if _KIMI_K3_SLUG_RE.search(m):
        return KIMI_K3_EFFORTS
    return KIMI_K2_EFFORTS


def supported_efforts_for_runtime(
    provider: Optional[str],
    model: Optional[str],
    advertised: Optional[Sequence[str]] = None,
) -> tuple[str, ...]:
    """Return the explicit effort vocabulary a provider/model picker may show.

    Known native/provider contracts win, then a serving aggregator's advertised
    list, and finally the broad OpenAI-compatible vocabulary. The result carries
    wire values rather than Hermes' wider internal ladder, so clients never
    offer a label that only happens to clamp to a different provider value.
    """
    provider_name = str(provider or "").strip().lower().replace("_", "-")
    model_name = str(model or "").strip().lower()
    flat_model = model_name.rsplit("/", 1)[-1]
    advertised_set = {
        str(candidate).strip().lower()
        for candidate in (advertised or ())
    }
    advertised_levels = tuple(
        level
        for level in EFFORT_LADDER
        if level in advertised_set and level != "ultra"
    )

    # Model-specific contracts that apply through direct and aggregator paths.
    if "ox-alpha" in model_name or flat_model == "x-preview-f-free":
        return OX_ALPHA_EFFORTS
    if "glm-5.3" in model_name:
        return GLM53_EFFORTS
    if "glm-5.2" in model_name:
        return GLM52_EFFORTS
    if "deepseek-v4" in model_name:
        return DEEPSEEK_V4_EFFORTS

    if provider_name == "openai-codex":
        return codex_app_server_supported_efforts(model_name)
    if provider_name in {"openai-api", "openai"}:
        return codex_supported_efforts(model_name)
    if provider_name in {"claude-code", "anthropic"}:
        return CLAUDE_CODE_EFFORTS
    if provider_name in {"kimi", "moonshot", "kimi-coding"}:
        return kimi_supported_efforts(model_name)
    if provider_name in {"tencent", "tokenhub", "tencent-tokenhub"}:
        return TOKENHUB_EFFORTS
    if provider_name in {"actual", "actual-computer"}:
        return ACTUAL_RELAY_EFFORTS
    if provider_name in {"ollama-cloud", "ollama"}:
        return OLLAMA_CLOUD_EFFORTS
    if provider_name in {"meta-ai", "meta"}:
        return META_AI_EFFORTS
    if provider_name == "upstage":
        return SOLAR_EFFORTS
    if provider_name in {"xai", "x-ai"}:
        return (
            XAI_GROK46_EFFORTS
            if "grok-4.6" in model_name
            else XAI_LEGACY_EFFORTS
        )

    if advertised_levels:
        return advertised_levels
    return OPENAI_COMPAT_WIRE_EFFORTS


def clamp_effort(
    effort: Optional[str],
    supported: Optional[Sequence[str]],
    overrides: Optional[dict[str, str]] = None,
) -> Optional[str]:
    """Clamp a requested reasoning effort onto a wire's supported levels.

    ``overrides`` is an optional declared mapping consulted first, for routes
    whose vendor documents a translation that differs from nearest-weaker
    (Kimi K3 documents ``medium → high``: high is its positional middle and
    server default). Overrides are data, not logic — a call site never adds
    vendor ``if``\\ s around this function.

    Otherwise: returns the requested effort unchanged when it is supported,
    when the supported set is unknown (``None``/empty), or when the effort
    isn't a recognized ladder level (custom providers may use bespoke names —
    pass through rather than guess). Otherwise returns the **nearest weaker**
    supported level, so a clamp never silently escalates cost; when nothing
    weaker exists, the weakest supported level is returned (the caller asked
    for *some* thinking and the provider's floor is the closest honest match).

    The policy is monotonic: a stronger request never resolves to a weaker
    wire level than a weaker request would.
    """
    requested = str(effort or "").strip().lower()
    if not requested or not supported:
        return effort
    supported_norm = [
        str(level).strip().lower()
        for level in supported
        if str(level).strip().lower() in EFFORT_LADDER
    ]
    if not supported_norm or requested in supported_norm:
        return effort
    if overrides:
        mapped = overrides.get(requested)
        if mapped in supported_norm:
            return mapped
    if requested not in EFFORT_LADDER:
        return effort
    # "none" disables reasoning — it is never a *degradation target* for an
    # enabled ask (clamping "minimal" to "none" would silently switch
    # thinking off). It still passes through verbatim when requested.
    candidates = [level for level in supported_norm if level != "none"]
    if not candidates:
        return effort
    requested_idx = EFFORT_LADDER.index(requested)
    below = [
        level for level in candidates
        if EFFORT_LADDER.index(level) < requested_idx
    ]
    if below:
        return max(below, key=EFFORT_LADDER.index)
    return min(candidates, key=EFFORT_LADDER.index)


def requested_effort(reasoning_config: Optional[dict]) -> Optional[str]:
    """Extract the user's explicit effort from a reasoning config, or None.

    Returns ``None`` when the config is absent, malformed, carries no effort,
    or reasoning is explicitly disabled — callers should then omit the wire
    field entirely so the server default applies (rule 2 above).
    """
    if not isinstance(reasoning_config, dict):
        return None
    if reasoning_config.get("enabled") is False:
        return None
    effort = str(reasoning_config.get("effort") or "").strip().lower()
    return effort or None
