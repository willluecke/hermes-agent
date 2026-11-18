"""scripts/jev_validate.py: when a shadow Jev question may act (Fable, 2026-11-17)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "jev_validate.py"
_spec = importlib.util.spec_from_file_location("jev_validate_under_test", _SCRIPT)
jv = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(jv)


def _lines(records):
    return [json.dumps(record) for record in records]


def test_scores_come_from_preflight_fields_and_guard_answers():
    log = _lines([
        {"event": "preflight", "turn_id": "t1", "p_design_freedom": 0.9, "p_ambiguous": 0.4},
        {"event": "preflight", "turn_id": "t2", "p_design_freedom": None},
        {"event": "tool_guard", "tool_call_id": "c1", "answers": {"within_scope": 0.45}},
        {"event": "verify", "turn_id": "t1"},
    ]) + ["not json"]
    assert jv.load_scores(log, "preflight:p_design_freedom") == [{"key": "t1", "score": 0.9}]
    assert jv.load_scores(log, "tool_guard:within_scope") == [{"key": "c1", "score": 0.45}]


def test_a_question_whose_scores_crowd_the_threshold_is_dropped_without_labels():
    scores = [{"key": f"t{i}", "score": 0.42} for i in range(6)] + [{"key": "t9", "score": 0.9}]
    report = jv.evaluate(scores, {}, 0.4)
    assert report["near_threshold_share"] > 0.5
    assert report["verdict"] == "drop" and "threshold decides" in report["reason"]


def test_too_few_labels_keep_a_question_in_shadow():
    scores = [{"key": f"t{i}", "score": 0.9 if i % 2 else 0.1} for i in range(40)]
    labels = {f"t{i}": {"label": bool(i % 2), "self": None} for i in range(40)}
    report = jv.evaluate(scores, labels, 0.5)
    assert report["verdict"] == "shadow" and "100 labelled cases" in report["reason"]


def _labelled(n_pos, n_neg, jev_right, self_right):
    """Positives score high unless Jev is wrong on them; the agent's own answer is right at its own rate."""
    scores, labels = [], {}
    for i in range(n_pos + n_neg):
        actual = i < n_pos
        right = (i % 100) < jev_right * 100
        score = (0.9 if actual else 0.1) if right else (0.1 if actual else 0.9)
        own_right = (i % 100) < self_right * 100
        scores.append({"key": f"t{i}", "score": score})
        labels[f"t{i}"] = {"label": actual, "self": actual if own_right else not actual}
    return scores, labels


def test_ship_needs_precision_above_point_eight_and_above_the_agents_own_answer():
    scores, labels = _labelled(60, 60, jev_right=0.95, self_right=0.7)
    report = jv.evaluate(scores, labels, 0.5)
    assert report["labelled"] == 120 and report["positives"] == 60
    assert report["precision"] > 0.8 and report["precision"] > report["self_precision"]
    assert report["verdict"] == "ship"
    # Not better than the agent itself: it stays in shadow.
    scores, labels = _labelled(60, 60, jev_right=0.9, self_right=0.97)
    assert jv.evaluate(scores, labels, 0.5)["verdict"] == "shadow"


def test_a_question_no_better_than_the_base_rate_is_dropped():
    scores, labels = _labelled(60, 60, jev_right=0.4, self_right=0.7)
    report = jv.evaluate(scores, labels, 0.5)
    assert report["precision"] <= report["base_rate"] and report["verdict"] == "drop"


def test_labels_file_reads_key_label_and_the_optional_self_answer():
    labels = jv.load_labels(_lines([
        {"key": "t1", "label": True, "self": False},
        {"key": "t2", "label": False},
        {"key": "t3", "label": "yes"},
        {"label": True},
    ]))
    assert labels == {"t1": {"label": True, "self": False}, "t2": {"label": False, "self": None}}


def test_the_cli_prints_a_verdict(tmp_path, capsys):
    log = tmp_path / "log.jsonl"
    log.write_text("\n".join(_lines([{"event": "preflight", "turn_id": f"t{i}", "p_design_freedom": 0.95} for i in range(5)])))
    assert jv.main(["--question", "preflight:p_design_freedom", "--threshold", "0.5", "--log", str(log)]) == 0
    out = capsys.readouterr().out
    assert "5 logged scores" in out and "verdict: shadow" in out
