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

Criteria are a running list across requests: what is still open carries
until the judge rates it met. Leaving the list any other way is an explicit
act with a stated reason the user sees as its own row: ``retire`` names
items, ``clear`` drops every carried item, and a ``cancelled`` status needs a
``reason`` too. The tool refuses a retirement without a reason.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from tools.registry import registry

MAX_ITEMS = 12
MAX_CHARS = 400
STATUSES = ("pending", "in_progress", "completed", "cancelled")
MAX_REASON_CHARS = 300


def _reason(value: Any) -> str:
    return str(value or "").strip()[:MAX_REASON_CHARS] if isinstance(value, (str, int, float)) else ""


def acceptance_criteria(args: Dict[str, Any]) -> str:
    args = args if isinstance(args, dict) else {}
    raw = args.get("criteria")
    raw_retire = args.get("retire")
    raw_clear = args.get("clear")
    retire: List[Dict[str, str]] = []
    if raw_retire is not None:
        if not isinstance(raw_retire, list):
            return json.dumps({"error": "retire must be a list of {content or id, reason} objects"})
        for value in raw_retire:
            if not isinstance(value, dict):
                return json.dumps({"error": "retire items must be {content or id, reason} objects"})
            target = str(value.get("content") or value.get("id") or "").strip()
            reason = _reason(value.get("reason"))
            if not target:
                return json.dumps({"error": "each retire item needs the criterion's content or id"})
            if not reason:
                return json.dumps({"error": f"retiring \"{target[:80]}\" needs a reason the user will see"})
            retire.append({"target": target[:MAX_CHARS], "reason": reason})
    clear: Dict[str, str] = {}
    if raw_clear is not None and raw_clear is not False:
        reason = _reason(raw_clear.get("reason")) if isinstance(raw_clear, dict) else _reason(raw_clear if isinstance(raw_clear, str) else "")
        if not reason:
            return json.dumps({"error": "clear needs a reason the user will see: {\"reason\": ...}"})
        clear = {"reason": reason}
    if raw is None and (retire or clear):
        raw = []
    if not isinstance(raw, list) or (not raw and not retire and not clear):
        return json.dumps({"error": "criteria must be a non-empty list of single checkable statements"})
    items: List[Dict[str, str]] = []
    for value in raw:
        status = "in_progress"
        reason = ""
        if isinstance(value, dict):
            text = str(value.get("content") or "").strip()
            wanted = str(value.get("status") or "").strip().lower()
            if wanted in STATUSES:
                status = wanted
            reason = _reason(value.get("reason"))
            if status == "cancelled" and not reason:
                return json.dumps({"error": f"cancelling \"{text[:80]}\" needs a reason the user will see"})
        else:
            text = str(value or "").strip()
        if not text:
            continue
        item = {"id": str(len(items) + 1), "content": text[:MAX_CHARS], "status": status}
        if reason:
            item["reason"] = reason
        items.append(item)
        if len(items) >= MAX_ITEMS:
            break
    if not items and not retire and not clear:
        return json.dumps({"error": "criteria must contain at least one non-empty statement"})
    notes = []
    if items:
        notes.append(f"{len(items)} acceptance criteria registered. Each will be judged against the diff and check output before this turn finishes.")
    if retire:
        notes.append(f"{len(retire)} criteria retired with a stated reason; the user sees each.")
    if clear:
        notes.append("Carried criteria cleared with a stated reason; the user sees it.")
    result: Dict[str, Any] = {"todos": items, "note": " ".join(notes)}
    if retire:
        result["retire"] = retire
    if clear:
        result["clear"] = clear
    return json.dumps(result, ensure_ascii=False)


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
        "pending, in_progress, completed or cancelled (cancelled needs a reason); a "
        "drift check steers toward the earliest criterion still open. Criteria carry "
        "across requests until the judge rates them met. To drop carried criteria the "
        "user no longer wants, pass retire: [{content, reason}] for specific ones, or "
        "clear: {reason} for all of them (refused while this turn has criteria of its "
        "own); every retirement is shown to the user with its reason, so never retire "
        "silently or without one."
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
                                "reason": {"type": "string", "description": "Required when status is cancelled; shown to the user."},
                            },
                            "required": ["content"],
                        },
                    ]
                },
                "description": "Single checkable statements, one per criterion, at most 12; or {content, status} objects to record progress. May be empty when only retiring or clearing.",
            },
            "retire": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "The carried criterion's text, as registered."},
                        "id": {"type": "string", "description": "Or its carried id (p1, p2, ...)."},
                        "reason": {"type": "string", "description": "Why it no longer applies; shown to the user."},
                    },
                    "required": ["reason"],
                },
                "description": "Carried criteria to drop, each with a reason the user will see.",
            },
            "clear": {
                "type": "object",
                "properties": {"reason": {"type": "string", "description": "Why every carried criterion no longer applies; shown to the user."}},
                "required": ["reason"],
                "description": "Drop every carried criterion at once: the user changed direction, accepted the previous turn, or a fresh list supersedes them. Refused while this turn has criteria of its own.",
            },
        },
        "required": [],
    },
}


registry.register(
    name="acceptance_criteria",
    toolset="typesafe",
    schema=ACCEPTANCE_CRITERIA_SCHEMA,
    handler=lambda args, **kwargs: acceptance_criteria(args),
    emoji="📋",
)
