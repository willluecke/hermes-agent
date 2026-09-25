"""Continuity of a native CLI session (Claude Code, Codex) across Hermes turns.

A conversation's record of truth is the stored transcript; the native session
file is a lossless, resumable copy of it; the resident CLI process is a cache
in front of that. A turn continues the native session when it can prove the
session saw the transcript so far, and otherwise rebuilds a fresh session from
the stored transcript -- the whole of it, as far as a generous budget allows,
never a short summary.

Every step away from a plain continuation is put on the run stream as a
``session.continuity`` event, which Hermes Chat shows as a row in the turn and
the archive keeps. Continuity loss was silent twice before (2026-09-24/25).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)

CONTINUITY_EVENT = "session.continuity"


def emit_continuity(
    agent: Any, runtime: str, mode: str, text: str, **payload: Any
) -> bool:
    """Put a continuity row on this turn's run stream. Never raises.

    ``mode`` is one of ``resumed`` (native session reloaded from disk),
    ``caught_up`` (rows the session missed were passed to it), ``rebuilt``
    (a fresh session seeded from the stored transcript), ``compacted`` (the
    CLI summarized its own context) or ``switch_deferred`` (a model or effort
    change waited for background work).
    """
    progress = getattr(agent, "tool_progress_callback", None)
    if progress is None or not text:
        return False
    try:
        progress(CONTINUITY_EVENT, runtime, text, None, mode=mode, **payload)
        return True
    except Exception:
        logger.debug("continuity event for %s failed", runtime, exc_info=True)
        return False


def render_history_blocks(
    entries: List[tuple],
    budget: int,
    render: Optional[Callable[[str, str], str]] = None,
) -> tuple:
    """Render the newest ``(role, text)`` entries that fit ``budget`` characters.

    Returns ``(blocks, omitted, truncated)``: the blocks oldest first, how many
    of the oldest entries did not fit, and whether the oldest included block
    was cut at its start.
    """
    render = render or (lambda role, text: f"{role.upper()}:\n{text}")
    rendered: list[str] = []
    used = 0
    omitted = 0
    truncated = False
    for index in range(len(entries) - 1, -1, -1):
        role, text = entries[index]
        block = render(role, text)
        block_truncated = False
        if len(block) > budget:
            block = block[-budget:]
            block_truncated = True
        if used + len(block) > budget:
            # Stop here rather than skipping this message and continuing with
            # older, smaller ones. Skipping punches a hole in the middle of the
            # transcript and discloses it only as a count, which reads as
            # "older context omitted" when it is really "a reply you are about
            # to see is missing". An unbroken recent window is the honest cut.
            omitted = index + 1
            break
        rendered.append(block)
        used += len(block)
        truncated = truncated or block_truncated
    rendered.reverse()
    return rendered, omitted, truncated


def handoff_disclosure(omitted: int, truncated: bool) -> str:
    notes: list[str] = []
    if omitted:
        notes.append(
            f"[{omitted} older messages omitted — handoff size limit reached.]"
        )
    if truncated:
        notes.append(
            "[The oldest included message was cut at its start to fit.]"
        )
    return ("\n" + "\n".join(notes)) if notes else ""


# Claude Code deletes session transcripts older than ``cleanupPeriodDays``
# (default 30) at startup, silently, and a deleted transcript can never be
# resumed. Hermes conversations live for years, so the runtime keeps this at
# ten years in the config dir its CLI uses and the gateway's readiness check
# reports drift.
TRANSCRIPT_RETENTION_DAYS = 3650
CLAUDE_DEFAULT_CLEANUP_DAYS = 30


def claude_config_dir() -> Path:
    configured = os.environ.get("CLAUDE_CONFIG_DIR", "").strip()
    return Path(configured).expanduser() if configured else Path.home() / ".claude"


def claude_transcript_retention_days(root: Optional[Path] = None) -> Optional[int]:
    """Days Claude Code keeps transcripts under ``root``; None if unreadable."""
    path = (root or claude_config_dir()) / "settings.json"
    try:
        settings = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError):
        return None
    if not isinstance(settings, dict):
        return None
    value = settings.get("cleanupPeriodDays", CLAUDE_DEFAULT_CLEANUP_DAYS)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def ensure_claude_transcript_retention(root: Optional[Path] = None) -> bool:
    """Raise ``cleanupPeriodDays`` to the retention floor. Returns whether it wrote.

    Leaves an unreadable settings file alone: rewriting what it cannot parse
    would lose the user's other settings. The readiness check reports it.
    """
    root = root or claude_config_dir()
    days = claude_transcript_retention_days(root)
    if days is None or days >= TRANSCRIPT_RETENTION_DAYS:
        return False
    path = root / "settings.json"
    try:
        settings = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        settings["cleanupPeriodDays"] = TRANSCRIPT_RETENTION_DAYS
        root.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.hermes-{os.getpid()}")
        tmp.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    except (OSError, ValueError):
        logger.warning("Could not raise Claude Code transcript retention in %s", path, exc_info=True)
        return False
    logger.warning(
        "Claude Code would delete transcripts after %d days in %s; raised cleanupPeriodDays to %d",
        days, root, TRANSCRIPT_RETENTION_DAYS,
    )
    return True


def plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


__all__ = [
    "CONTINUITY_EVENT",
    "TRANSCRIPT_RETENTION_DAYS",
    "claude_config_dir",
    "claude_transcript_retention_days",
    "emit_continuity",
    "ensure_claude_transcript_retention",
    "handoff_disclosure",
    "plural",
    "render_history_blocks",
]
