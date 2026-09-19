#!/usr/bin/env python3
"""Register the result claims of a change so code can check them against the evidence ledger.

The verify judge keeps a ledger of every command the turn ran, one row per
call, built by code from the tool result (see
``plugins/system-one-preflight/evidence.py``). Free text like "98 passed"
cannot be matched against that ledger honestly: the number may describe
one suite, an earlier attempt, or an expectation. This tool is the
structured form. Each item names the claim as it will appear in the
answer, the ledger rows it rests on, and a predicate code compares
exactly:

* ``passed``: every cited row reports a pass;
* ``count``: the cited rows' counts equal ``expected`` (for example
  ``{"passed": 98, "failed": 0}``);
* ``exit_zero``: every cited row exited 0;
* ``contains``: ``expected.text`` appears in the cited rows' retained output;
* ``ran``: the cited rows exist.

A cited row that ran before a later edit is stale, and the gate re-runs
plain check commands itself. Like ``acceptance_criteria`` this tool is
stateless: the record is the tool result, which the runtime's hooks hand
to the plugin on every lane.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from tools.registry import registry

MAX_ITEMS = 16
MAX_CLAIM_CHARS = 300
MAX_EVIDENCE = 8
PREDICATES = ("passed", "count", "exit_zero", "contains", "ran")


def report_results(args: Dict[str, Any]) -> str:
    raw = args.get("results") if isinstance(args, dict) else None
    if not isinstance(raw, list):
        return json.dumps({"error": "results must be a list of {claim, evidence, predicate} objects (an empty list means no result is claimed)"})
    items: List[Dict[str, Any]] = []
    problems: List[str] = []
    for index, value in enumerate(raw, 1):
        if not isinstance(value, dict):
            problems.append(f"item {index}: not an object")
            continue
        claim = str(value.get("claim") or "").strip()
        if not claim:
            problems.append(f"item {index}: claim is empty")
            continue
        evidence_raw = value.get("evidence")
        if isinstance(evidence_raw, str):
            evidence_raw = [evidence_raw]
        evidence = [str(item).strip() for item in (evidence_raw or []) if str(item).strip()] if isinstance(evidence_raw, list) else []
        predicate = str(value.get("predicate") or "passed").strip().lower()
        if predicate not in PREDICATES:
            problems.append(f"item {index}: predicate must be one of {', '.join(PREDICATES)}")
            continue
        expected = value.get("expected") if isinstance(value.get("expected"), dict) else {}
        if predicate == "count" and not any(isinstance(v, int) and not isinstance(v, bool) for v in expected.values()):
            problems.append(f"item {index}: count needs expected counts, for example {{\"passed\": 12, \"failed\": 0}}")
            continue
        if predicate == "contains" and not str(expected.get("text") or "").strip():
            problems.append(f"item {index}: contains needs expected.text")
            continue
        items.append(
            {
                "id": f"r{len(items) + 1}",
                "criterion": str(value.get("criterion") or "").strip(),
                "claim": claim[:MAX_CLAIM_CHARS],
                "evidence": evidence[:MAX_EVIDENCE],
                "predicate": predicate,
                "expected": expected,
            }
        )
        if len(items) >= MAX_ITEMS:
            break
    if problems and not items and raw:
        return json.dumps({"error": "; ".join(problems)})
    note = (
        f"{len(items)} result claim{'s' if len(items) != 1 else ''} registered. Code checks each against the "
        "evidence ledger before this turn finishes; a claim whose rows are stale is re-run by the gate."
    )
    if not items:
        note = "No result claims registered: the answer claims no check result, so it must not state one."
    payload: Dict[str, Any] = {"manifest": items, "note": note}
    if problems:
        payload["skipped"] = problems
    return json.dumps(payload, ensure_ascii=False)


REPORT_RESULTS_SCHEMA = {
    "name": "report_results",
    "description": (
        "Register the result claims your answer will make, before finishing a change. "
        "Each item names one claim as it will appear in the answer, the evidence ledger "
        "rows it rests on (row ids like c7, shown to you in the drift and verify notes; "
        "every terminal command this turn is a row, in order), and a predicate code checks "
        "exactly: passed (every cited row reports a pass), count (the rows' counts equal "
        "expected, e.g. {passed: 98, failed: 0}), exit_zero, contains (expected.text is in "
        "the retained output), or ran. Cite the row of the command that produced the "
        "result; a row that ran before your last edit is stale and the gate re-runs it. "
        "Call once before your final message; call again to replace the list. An empty "
        "list means the answer claims no check result."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "description": "One object per result claim, at most 16.",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string", "description": "The claim as it will appear in the answer."},
                        "criterion": {"type": "string", "description": "The acceptance criterion id this result supports, if any."},
                        "evidence": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Ledger row ids the claim rests on, for example [\"c7\"].",
                        },
                        "predicate": {"type": "string", "enum": list(PREDICATES)},
                        "expected": {
                            "type": "object",
                            "description": "For count: {passed, failed, errors}; for contains: {text}.",
                        },
                    },
                    "required": ["claim", "evidence"],
                },
            }
        },
        "required": ["results"],
    },
}


registry.register(
    name="report_results",
    toolset="typesafe",
    schema=REPORT_RESULTS_SCHEMA,
    handler=lambda args, **kwargs: report_results(args),
    emoji="🧾",
)
