"""The wall-clock cap on one native CLI turn (Codex app-server, Claude Code).

A turn that is still working runs until it finishes or reaches this cap; a
silent one is stopped much sooner by each transport's inactivity watchdog.
Both transports used a fixed 2-hour cap until 2026-10-01, when it cut off a
Codex turn on the talos repo mid-work. That turn had compacted twice and kept
going both times, but the compaction notices were printed after the cut, so
the run looked like it had died from compacting.

The cap is ``agent.turn_time_limit_hours`` in config.yaml, read on every turn,
so an edit applies to the next turn without a gateway restart.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

logger = logging.getLogger(__name__)

CONFIG_KEY = "turn_time_limit_hours"
DEFAULT_TURN_TIME_LIMIT_HOURS = 12.0
DEFAULT_TURN_TIME_LIMIT_SECONDS = DEFAULT_TURN_TIME_LIMIT_HOURS * 3600.0

# The continuity mode of the row that says a limit ended a turn.
STOPPED_MODE = "stopped"


def turn_time_limit_seconds(config: Optional[dict[str, Any]] = None) -> float:
    """``agent.turn_time_limit_hours`` in seconds; ``0`` means no cap.

    A missing, unreadable or malformed value falls back to the default rather
    than to no cap: a typo must not lift the bound on every turn.
    """
    if config is None:
        try:
            from hermes_cli.config import load_config

            config = load_config()
        except Exception:
            logger.debug("turn time limit: config unreadable", exc_info=True)
            config = {}
    agent_cfg = config.get("agent") if isinstance(config, dict) else None
    raw = agent_cfg.get(CONFIG_KEY) if isinstance(agent_cfg, dict) else None
    if raw is None:
        return DEFAULT_TURN_TIME_LIMIT_SECONDS
    try:
        hours = float(raw)
    except (TypeError, ValueError):
        hours = math.nan
    if math.isnan(hours) or isinstance(raw, bool):
        logger.warning(
            "agent.%s=%r is not a number of hours; using %g",
            CONFIG_KEY,
            raw,
            DEFAULT_TURN_TIME_LIMIT_HOURS,
        )
        return DEFAULT_TURN_TIME_LIMIT_SECONDS
    if hours <= 0 or math.isinf(hours):
        return 0.0
    return hours * 3600.0


def _amount(seconds: float) -> tuple[float, str]:
    if seconds >= 3600:
        return seconds / 3600, "hour"
    if seconds >= 60:
        return seconds / 60, "minute"
    return seconds, "second"


def describe_limit(seconds: float) -> str:
    """The limit as a compound adjective: ``12-hour``, ``90-minute``."""
    value, unit = _amount(seconds)
    return f"{value:g}-{unit}"


def describe_duration(seconds: float) -> str:
    """The duration as a noun phrase: ``10 minutes``, ``1 hour``."""
    value, unit = _amount(seconds)
    return f"{value:g} {unit}{'' if value == 1 else 's'}"


def time_limit_stop_text(runtime_label: str, seconds: float) -> str:
    """The row a turn ended by the cap leaves in the chat."""
    return (
        f"{runtime_label} stopped this turn at the {describe_limit(seconds)} "
        "turn limit while it was still working. Send a message to have it "
        "continue."
    )


def inactivity_stop_text(runtime_label: str, seconds: float) -> str:
    """The row a turn ended by the inactivity watchdog leaves in the chat."""
    return (
        f"{runtime_label} stopped this turn after {describe_duration(seconds)} "
        "with no activity. Send a message to have it continue."
    )
