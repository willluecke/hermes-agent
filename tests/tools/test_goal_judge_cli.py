"""A declared completion ends a standing goal; the met score alone never re-runs it."""
import io
import json

import pytest

from tools import goal_judge_cli


def run(monkeypatch, capsys, met, claims, blocked):
    monkeypatch.setattr("tools.typesafe_tool.ask_jev", lambda state, questions: {
        "provider": "test", "answers": {"met": {"noul": met}, "claims_done": {"noul": claims}, "blocked": {"noul": blocked}}})
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"goal": "Find 40 prospects", "reply": "Research complete; 31 of 40."})))
    assert goal_judge_cli.main() == 0
    return json.loads(capsys.readouterr().out)


@pytest.mark.parametrize("met,claims,blocked,verdict", [
    (0.41, 0.98, 0.23, "done"),      # the 2026-10-19 research reply: honest shortfall, explicit completion
    (0.98, 0.06, 0.02, "done"),      # fully met without saying so
    (0.30, 0.50, 0.10, "continue"),  # neither met nor declared
    (0.41, 0.98, 0.95, "blocked"),   # declared impossible or waiting on a person
])
def test_declared_completion_ends_the_goal(monkeypatch, capsys, met, claims, blocked, verdict):
    out = run(monkeypatch, capsys, met, claims, blocked)
    assert out["verdict"] == verdict
    assert out["done"] is (verdict == "done")
    assert out["p"] == met
    if verdict == "done" and met < goal_judge_cli.DONE_THRESHOLD:
        assert "declares completion" in out["reason"]


def test_incomplete_judge_answer_is_an_error(monkeypatch):
    monkeypatch.setattr("tools.typesafe_tool.ask_jev", lambda state, questions: {"answers": {"met": {"noul": 0.5}}})
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"goal": "g", "reply": "r"})))
    with pytest.raises(ValueError):
        goal_judge_cli.main()
