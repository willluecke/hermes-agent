#!/usr/bin/env python3
"""Register the result claims of a change so code can check them against the evidence ledger.

The verify judge keeps a ledger of every command the turn ran, one row per
call, built by code from the tool result (see
``plugins/system-one-preflight/evidence.py``). Free text like "98 passed"
cannot be matched against that ledger honestly: the number may describe
one suite, an earlier attempt, or an expectation. This tool is the
structured form. Each item names the claim as it will appear in the
answer, the commands it rests on as the model ran them (code matches each
to its newest run; the model is never shown row ids, so it never cites
one), and a predicate code compares exactly:

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
import re
from typing import Any, Dict, List

from tools.registry import registry

MAX_ITEMS = 16
MAX_CLAIM_CHARS = 300
MAX_EVIDENCE = 8
MAX_CITED_CHARS = 2_000
PREDICATES = ("passed", "count", "exit_zero", "contains", "ran")
# ``evidence`` is the documented key; the others are what models write for it.
EVIDENCE_KEYS = ("evidence", "commands", "command", "rows", "citations")
# A ledger row id (c7, k58, pc3). The model is never shown one, so a cited id
# is a guess; the tool says so at once, on every lane, and the gate refuses it.
ROW_ID_RE = re.compile(r"p?[ck]\d+", re.IGNORECASE)


def report_results(args: Dict[str, Any]) -> str:
    raw = args.get("results") if isinstance(args, dict) else None
    if not isinstance(raw, list):
        return json.dumps({"error": "results must be a list of {claim, evidence, predicate} objects (an empty list means no result is claimed)"})
    items: List[Dict[str, Any]] = []
    problems: List[str] = []
    warnings: List[str] = []
    for index, value in enumerate(raw, 1):
        if not isinstance(value, dict):
            problems.append(f"item {index}: not an object")
            continue
        claim = str(value.get("claim") or "").strip()
        if not claim:
            problems.append(f"item {index}: claim is empty")
            continue
        # The description says "the commands it rests on", so models reach for
        # ``commands`` (803 of 1,289 items in the week to 2026-10-09; 33 used
        # ``rows``). A silently dropped key voided the manifest and the gate
        # flagged true claims. Every spelling lands in ``evidence``.
        evidence_raw = None
        for key in EVIDENCE_KEYS:
            if value.get(key) not in (None, "", []):
                evidence_raw = value.get(key)
                break
        if isinstance(evidence_raw, str):
            evidence_raw = [evidence_raw]
        evidence = [str(item).strip()[:MAX_CITED_CHARS] for item in (evidence_raw or []) if str(item).strip()] if isinstance(evidence_raw, list) else []
        guessed = [value for value in evidence if ROW_ID_RE.fullmatch(value)]
        if guessed:
            warnings.append(
                f"item {index} cites {', '.join(guessed)}: row ids are never shown to you and are not accepted; "
                "cite the command as you ran it"
            )
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
        if predicate != "contains" and not evidence:
            # Without a command the gate can only mark the claim insufficient
            # and flag the sentence; refusing here lets the model fix it now.
            problems.append(
                f"item {index} ({claim[:60]!r}): name the command it rests on under evidence, "
                "as you ran it, for example [\"npx tsc --noEmit -p .\"]"
            )
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
        f"{len(items)} result claim{'s' if len(items) != 1 else ''} registered. Before this turn finishes, code "
        "matches each cited command to its newest run and checks the claim against it; a check that ran before a "
        "later edit is re-run by the gate where it can be."
    )
    if not items:
        note = "No result claims registered: the answer claims no check result, so it must not state one."
    payload: Dict[str, Any] = {"manifest": items, "note": note}
    if problems:
        payload["skipped"] = problems
    if warnings:
        payload["warnings"] = warnings
    return json.dumps(payload, ensure_ascii=False)


REPORT_RESULTS_SCHEMA = {
    "name": "report_results",
    "description": (
        "Register the result claims your answer will make, before finishing a change. "
        "Each item names one claim as it will appear in the answer, the commands it rests "
        "on under the key evidence (for example {\"claim\": \"Hermes Chat lint passes\", "
        "\"evidence\": [\"npx next lint\"], \"predicate\": \"exit_zero\"}), and a predicate code "
        "checks exactly: passed (every cited command reports a "
        "pass), count (their counts equal expected, e.g. {passed: 98, failed: 0}), "
        "exit_zero, contains (expected.text is in the output), or ran. Cite each command "
        "as you ran it in the terminal this turn; a distinctive part of a long command is "
        "enough, and the newest run of a matching command is the one that counts, so run "
        "a check again after your last edit and cite it the same way. Never cite a row id "
        "such as c7: you are not shown them. For contains, quote expected.text exactly as "
        "it was printed; the gate also finds the newest output that printed it. Every "
        "passed, count or exit_zero claim is re-run by the gate itself before it counts, so "
        "cite plain check commands (pytest, npm test, node --test, tsc, eslint, ruff and "
        "the like; no wrapper scripts, no '|| echo', output filters are stripped); a claim "
        "the gate cannot re-run is reported as unverified. Call once before your final "
        "message; call again to replace the list. An empty list means the answer claims no "
        "check result."
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
                            "description": "Required for passed, count, exit_zero and ran: the commands the claim rests on, as you ran them, for example [\"npx vitest run test/mcp.test.ts\"]. A distinctive part of a long command is enough. Not row ids. (Also read from commands, command or rows.)",
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
