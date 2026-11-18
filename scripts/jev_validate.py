#!/usr/bin/env python3
"""Decide whether a Jev question has earned the right to act (Fable, 2026-11-17).

Every new Jev question runs in shadow first: asked and logged by the
system-one-preflight plugin, never acted on. This script reads those logged
scores and, when labels exist, says whether the question should ship, stay
in shadow, or be dropped.

Rule:
  * ship   - at least 100 labelled cases with 30 in each class, precision at
             the operating threshold of 0.8 or more, and better than the
             agent's own answer on the same cases;
  * drop   - no better than the base rate, or more than half of all logged
             scores within 0.1 of the threshold (the threshold, not Jev, is
             then making the call);
  * shadow - anything else: keep logging and labelling.

The near-threshold check needs no labels, so it can retire a question early.

Labels are JSON lines: {"key": "<turn_id or tool_call_id>", "label": true|false}
with an optional "self": true|false for the agent's own answer. Labels should
come from Will, blind to Jev's score, or from a real outcome such as a check
that passed or failed.

Usage:
  python3 scripts/jev_validate.py --question preflight:p_design_freedom --threshold 0.5
  python3 scripts/jev_validate.py --question tool_guard:within_scope --threshold 0.4 --labels labels.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

DEFAULT_LOG = Path.home() / ".hermes" / "logs" / "system-one-preflight.jsonl"
NEAR_BAND = 0.1
MIN_LABELLED = 100
MIN_PER_CLASS = 30
SHIP_PRECISION = 0.8
MAX_NEAR_SHARE = 0.5


def load_scores(lines: Iterable[str], question: str) -> List[Dict[str, Any]]:
    """Scores for ``question`` as ``event:field``.

    ``preflight:<field>`` reads a top-level field of a preflight record (for
    example ``p_ambiguous`` or ``p_design_freedom``), keyed by turn id.
    ``tool_guard:<answer>`` reads one guard answer, keyed by tool call id.
    """
    event, _, field = question.partition(":")
    if not event or not field:
        raise ValueError("question must look like event:field, for example preflight:p_ambiguous")
    scores: List[Dict[str, Any]] = []
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict) or record.get("event") != event:
            continue
        if event == "tool_guard":
            value = (record.get("answers") or {}).get(field)
            key = record.get("tool_call_id")
        else:
            value = record.get(field)
            key = record.get("turn_id")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            scores.append({"key": str(key or ""), "score": float(value)})
    return scores


def near_threshold_share(scores: List[Dict[str, Any]], threshold: float, band: float = NEAR_BAND) -> Optional[float]:
    """Share of scores within ``band`` of ``threshold``; None without scores."""
    if not scores:
        return None
    near = sum(1 for item in scores if abs(item["score"] - threshold) <= band)
    return near / len(scores)


def load_labels(lines: Iterable[str]) -> Dict[str, Dict[str, Optional[bool]]]:
    labels: Dict[str, Dict[str, Optional[bool]]] = {}
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict) or not isinstance(record.get("label"), bool):
            continue
        key = str(record.get("key") or "").strip()
        if not key:
            continue
        own = record.get("self")
        labels[key] = {"label": record["label"], "self": own if isinstance(own, bool) else None}
    return labels


def _precision_recall(pairs: List[tuple]) -> Dict[str, Optional[float]]:
    """``pairs`` of (predicted, actual) booleans."""
    tp = sum(1 for predicted, actual in pairs if predicted and actual)
    fp = sum(1 for predicted, actual in pairs if predicted and not actual)
    fn = sum(1 for predicted, actual in pairs if not predicted and actual)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {"precision": precision, "recall": recall}


def evaluate(
    scores: List[Dict[str, Any]],
    labels: Dict[str, Dict[str, Optional[bool]]],
    threshold: float,
) -> Dict[str, Any]:
    """The report and its verdict for one question at one threshold."""
    near = near_threshold_share(scores, threshold)
    by_key = {item["key"]: item["score"] for item in scores if item["key"]}
    matched = [(by_key[key], value) for key, value in labels.items() if key in by_key]
    positives = sum(1 for _, value in matched if value["label"])
    negatives = len(matched) - positives
    report: Dict[str, Any] = {
        "scores": len(scores),
        "threshold": threshold,
        "near_threshold_share": near,
        "labelled": len(matched),
        "positives": positives,
        "negatives": negatives,
        "base_rate": positives / len(matched) if matched else None,
        "precision": None,
        "recall": None,
        "self_precision": None,
    }
    if matched:
        report.update(_precision_recall([(score >= threshold, value["label"]) for score, value in matched]))
        own = [(value["self"], value["label"]) for _, value in matched if value["self"] is not None]
        if len(own) >= MIN_PER_CLASS:
            report["self_precision"] = _precision_recall(own)["precision"]
    report["verdict"], report["reason"] = verdict(report)
    return report


def verdict(report: Dict[str, Any]) -> tuple:
    near = report.get("near_threshold_share")
    if near is not None and near > MAX_NEAR_SHARE:
        return "drop", f"{near:.0%} of scores sit within {NEAR_BAND} of the threshold, so the threshold decides, not Jev"
    if report["labelled"] < MIN_LABELLED or min(report["positives"], report["negatives"]) < MIN_PER_CLASS:
        return "shadow", f"needs {MIN_LABELLED} labelled cases with {MIN_PER_CLASS} in each class"
    precision, base = report["precision"], report["base_rate"]
    if precision is None or (base is not None and precision <= base):
        return "drop", "no better than the base rate"
    if precision < SHIP_PRECISION:
        return "shadow", f"precision {precision:.2f} is below {SHIP_PRECISION}"
    own = report.get("self_precision")
    if own is None:
        return "shadow", "no agent self-judgment to compare against yet"
    if precision <= own:
        return "shadow", f"precision {precision:.2f} is not better than the agent's own {own:.2f}"
    return "ship", f"precision {precision:.2f} at the threshold, above the base rate and the agent's own {own:.2f}"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--question", required=True, help="event:field, e.g. preflight:p_design_freedom or tool_guard:within_scope")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--labels", type=Path, help="JSON lines of {key, label, self?}")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)
    with args.log.open(encoding="utf-8", errors="ignore") as handle:
        scores = load_scores(handle, args.question)
    labels: Dict[str, Dict[str, Optional[bool]]] = {}
    if args.labels:
        with args.labels.open(encoding="utf-8") as handle:
            labels = load_labels(handle)
    report = {"question": args.question, **evaluate(scores, labels, args.threshold)}
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    near = report["near_threshold_share"]
    print(f"{args.question} at threshold {args.threshold}: {report['scores']} logged scores")
    print(f"  within {NEAR_BAND} of the threshold: {'n/a' if near is None else f'{near:.0%}'}")
    print(f"  labelled: {report['labelled']} ({report['positives']} yes, {report['negatives']} no)")
    for key in ("base_rate", "precision", "recall", "self_precision"):
        value = report[key]
        print(f"  {key.replace('_', ' ')}: {'n/a' if value is None else f'{value:.2f}'}")
    print(f"verdict: {report['verdict']} ({report['reason']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
