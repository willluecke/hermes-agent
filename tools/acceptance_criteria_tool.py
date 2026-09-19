#!/usr/bin/env python3
"""Pre-register acceptance criteria for the verify judge.

The Jev verify gate judges a finished change against criteria the model
wrote *before* editing, one noul per criterion. On the default Hermes loop
those come from the ``todo`` tool. The Codex and Claude runtimes execute
tools themselves and reach Hermes only through the stateless MCP bridge,
which cannot carry ``todo`` (it needs the live agent loop). This tool is the
stateless equivalent: it takes the criteria, returns them in the todo
tool's JSON shape, and the runtime's hook parity replays that result to the
``system-one-preflight`` plugin, which stores it exactly as it stores a todo
list. Nothing is written anywhere; the record is the tool result itself.

Items may carry a status (``pending``, ``in_progress``, ``completed``) so the
model can record progress by calling again with the same statements; the
plugin's drift check steers toward the earliest criterion still open.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from tools.registry import registry

MAX_ITEMS = 12
MAX_CHARS = 400
STATUSES = ("pending", "in_progress", "completed")


def acceptance_criteria(args: Dict[str, Any]) -> str:
    raw = args.get("criteria") if isinstance(args, dict) else None
    if not isinstance(raw, list) or not raw:
        return json.dumps({"error": "criteria must be a non-empty list of single checkable statements"})
    items: List[Dict[str, str]] = []
    for value in raw:
        status = "in_progress"
        if isinstance(value, dict):
            text = str(value.get("content") or "").strip()
            wanted = str(value.get("status") or "").strip().lower()
            if wanted in STATUSES:
                status = wanted
        else:
            text = str(value or "").strip()
        if not text:
            continue
        items.append({"id": str(len(items) + 1), "content": text[:MAX_CHARS], "status": status})
        if len(items) >= MAX_ITEMS:
            break
    if not items:
        return json.dumps({"error": "criteria must contain at least one non-empty statement"})
    return json.dumps(
        {
            "todos": items,
            "note": (
                f"{len(items)} acceptance criteria registered. Each will be judged "
                "against the diff and check output before this turn finishes."
            ),
        },
        ensure_ascii=False,
    )


ACCEPTANCE_CRITERIA_SCHEMA = {
    "name": "acceptance_criteria",
    "description": (
        "Register the acceptance criteria for a change before making it. Write "
        "each criterion as one single, checkable statement about the finished "
        "work (for example 'The --json flag prints valid JSON that jq can parse'). "
        "A typed judge checks every criterion against the diff and the check "
        "output before the turn finishes and sends the work back when one is "
        "unmet. Call it once, before the first edit; call again to replace the list. "
        "To record progress, call again with objects {content, status} where status is "
        "pending, in_progress or completed; a drift check steers toward the earliest "
        "criterion still open."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "criteria": {
                "type": "array",
                "items": {
                    "anyOf": [
                        {"type": "string"},
                        {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string"},
                                "status": {"type": "string", "enum": list(STATUSES)},
                            },
                            "required": ["content"],
                        },
                    ]
                },
                "description": "Single checkable statements, one per criterion, at most 12; or {content, status} objects to record progress.",
            }
        },
        "required": ["criteria"],
    },
}


registry.register(
    name="acceptance_criteria",
    toolset="typesafe",
    schema=ACCEPTANCE_CRITERIA_SCHEMA,
    handler=lambda args, **kwargs: acceptance_criteria(args),
    emoji="📋",
)
