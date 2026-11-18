"""The feature pipeline (Will, 2026-11-18): plan, Astra consensus, regions, integration.

The pure parts (plan validation, regions, the feature-system measure, the
stop rule) are tested directly. The kanban parts run on a real temporary
board with a fake Astra, through the same transitions the dispatcher and
the review lane use.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from hermes_cli import feature_pipeline as fp
from hermes_cli import kanban_db as kb


def _item(item_id, files, goal=None, check="pytest -q", depends_on=None):
    return {"id": item_id, "goal": goal or f"goal {item_id}", "files": files, "check": check,
            "depends_on": depends_on or []}


PLAN = {
    "summary": "Notes: page, API and store.",
    "items": [
        _item("i1", ["src/app/notes/page.tsx"]),
        _item("i2", ["src/app/notes/"]),
        _item("i3", ["server/api/notes.py"]),
    ],
}


def _verdict(verdict, *issues):
    findings = [{"item": "plan", "issue": issue, "fix": "change it"} for issue in issues]
    return "I read the code.\n" + json.dumps({"verdict": verdict, "findings": findings})


# --- the plan ---------------------------------------------------------------

def test_validate_plan_lists_every_problem_at_once():
    with pytest.raises(fp.PipelineError) as err:
        fp.validate_plan({"summary": "", "items": [
            {"id": "a", "goal": "", "files": [], "check": ""},
            {"id": "a", "goal": "g", "files": ["/etc/passwd", "src/../../x"], "check": "c", "depends_on": ["zz", "a"]},
        ]})
    text = str(err.value)
    for expected in ("summary is required", "item a needs a goal", "needs the files", "needs a check",
                     "used twice", "must be inside the project's repo", "unknown item zz", "depends on itself"):
        assert expected in text


def test_validate_plan_canonicalizes_paths_rejects_cycles_and_oversized_plans():
    plan = fp.validate_plan(json.dumps({"summary": "s", "items": [_item("x", ["./src//a.ts", "src/./b/", "src/a.ts"])]}))
    assert plan["items"][0]["files"] == ["src/a.ts", "src/b"]
    with pytest.raises(fp.PipelineError, match="cycle"):
        fp.validate_plan({"summary": "s", "items": [_item("a", ["a.py"], depends_on=["b"]),
                                                     _item("b", ["b.py"], depends_on=["a"])]})
    with pytest.raises(fp.PipelineError, match="Astra must see all of it"):
        fp.validate_plan({"summary": "s", "items": [_item(f"i{n}", [f"f{n}.py"], goal="x" * 2000) for n in range(30)]})


def test_regions_join_items_that_share_files_or_depend_on_each_other():
    items = fp.validate_plan(PLAN)["items"]
    assert fp.group_regions(items) == [["i1", "i2"], ["i3"]]
    chained = fp.validate_plan({"summary": "s", "items": [
        _item("a", ["x.py"], depends_on=["c"]), _item("b", ["y.py"]), _item("c", ["z.py"])]})["items"]
    assert fp.group_regions(chained) == [["c", "a"], ["b"]], "a dependency comes first inside its region"
    assert fp.group_regions([_item("a", ["src/a.ts"]), _item("b", ["src/ab.ts"])]) == [["a"], ["b"]], \
        "a shared name prefix is not an overlap"
    whole = fp.validate_plan({"summary": "s", "items": [_item("a", ["."]), _item("b", ["src/a.py"])]})["items"]
    assert fp.group_regions(whole) == [["a", "b"]], "the repo root overlaps everything"


def test_the_definition_is_measured_on_the_plan():
    measured = fp.feature_system(fp.validate_plan(PLAN))
    assert measured["feature_system"] and measured["subsystems"] == ["server/api", "src/app"]
    one = fp.feature_system(fp.validate_plan({"summary": "s", "items": [
        _item("a", ["src/lib/time.ts"]), _item("b", ["tests/time.test.ts"])]}))
    assert one == {"feature_system": False, "subsystems": ["src/lib"], "items": 2, "separate_file_items": 2}


# --- Astra's verdict and the stop rule ----------------------------------------

def test_parse_verdict_takes_the_last_json_verdict_after_prose():
    text = 'Looked at {"not": "this"} first.\n' + _verdict("changes", "item i3 has no test")
    assert fp.parse_verdict(text)["findings"][0]["issue"] == "item i3 has no test"
    assert fp.parse_verdict('{"verdict": "approve", "findings": [{"issue": "x"}]}') == {"verdict": "approve", "findings": []}
    with pytest.raises(fp.PipelineError):
        fp.parse_verdict("no json here")
    with pytest.raises(fp.PipelineError):
        fp.parse_verdict('{"verdict": "changes", "findings": []}')


def _round(n, digest, *issues):
    return {"type": "plan_round", "round": n, "digest": digest, "verdict": "changes",
            "findings": [{"item": "plan", "issue": i} for i in issues]}


def test_an_unchanged_plan_that_astra_still_rejects_goes_to_will():
    changes = fp.parse_verdict(_verdict("changes", "the store has no migration"))
    assert fp.decide_round([_round(1, "d1", "x")], "d1", changes)["action"] == "escalate"
    assert fp.decide_round([_round(1, "d1", "x")], "d2", changes)["action"] == "revise"
    assert fp.decide_round([_round(1, "d1", "x")], "d1", changes, will_answered=True)["action"] == "revise"


def test_a_finding_raised_three_rounds_running_goes_to_will():
    issue = "the store has no migration for the notes table"
    history = [_round(1, "d1", issue), _round(2, "d2", issue + " yet")]
    stuck = fp.decide_round(history, "d3", fp.parse_verdict(_verdict("changes", issue)))
    assert stuck["action"] == "escalate" and "three rounds" in stuck["reason"]
    twice = fp.decide_round(history[:1], "d2", fp.parse_verdict(_verdict("changes", issue)))
    assert twice["action"] == "revise", "one repeat gets another revision"
    other = fp.decide_round(history, "d3", fp.parse_verdict(_verdict("changes", "the page has no empty state")))
    assert other["action"] == "revise"


# --- the pipeline on a board ---------------------------------------------------

@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    hermes = tmp_path / ".hermes"
    hermes.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for name in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID",
                 "HERMES_KANBAN_WORKSPACES_ROOT", "HERMES_GATEWAY_SESSION_ID"):
        monkeypatch.delenv(name, raising=False)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    repo = tmp_path / "proj"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q",
                    "--allow-empty", "-m", "base"], check=True)
    (hermes / "subscription-projects.json").write_text(json.dumps({"proj": str(repo)}))
    notes = []
    monkeypatch.setattr("hermes_cli.kanban_discovery.post_report_to_session",
                        lambda session, text, board, **kw: notes.append((session, text)) or len(notes))
    return {"repo": repo, "notes": notes}


class FakeAstra:
    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def __call__(self, prompt, *, cwd):
        self.calls.append({"prompt": prompt, "cwd": cwd})
        return {"response": self.answers.pop(0), "usage": {"output_tokens": 7}}


def _claim(root):
    with kb.connect_closing(board="proj") as conn:
        kb.recompute_ready(conn)
        claimed = kb.claim_task(conn, root, claimer="opus")
        assert claimed is not None, kb.get_task(conn, root).status
        return claimed.current_run_id


def _through_review(card):
    """Opus builds and requests review; Astra's review run approves."""
    with kb.connect_closing(board="proj") as conn:
        run = kb.claim_task(conn, card, claimer="opus")
        assert kb.request_review(conn, card, summary="built", expected_run_id=run.current_run_id)
        review = kb.claim_review_task(conn, card, claimer="astra")
        assert kb.review_spawn_task(review).model_override == "gpt-6.2-astra"
        assert kb.complete_task(conn, card, result="approved", expected_run_id=review.current_run_id)


def _status(root):
    return fp.status(root_id=root, board="proj")


def _launched_root(plan=PLAN):
    root = fp.start(project="proj", request="Build the notes feature: page, API and store.", conversation_id="c1")["root"]
    run = _claim(root)
    fp.plan_review(root_id=root, plan=plan, board="proj", run_id=run, reviewer=FakeAstra(_verdict("approve")))
    cards = [r["card"] for r in fp.launch(root_id=root, board="proj", run_id=run)["regions"]]
    return root, cards


def test_a_build_runs_from_start_to_finish(home):
    started = fp.start(project="proj", request="Build the notes feature: page, API and store.", conversation_id="c1")
    root = started["root"]
    assert started["board"] == "proj"
    assert kb.read_board_metadata("proj")["default_workdir"] == str(home["repo"]), "the board is made for the project"
    assert _status(root)["target"]["branch"] == "main", "the branch to land on is recorded at the start"
    with kb.connect_closing(board="proj") as conn:
        task = kb.get_task(conn, root)
        assert (task.model_override, task.provider_override, task.reasoning_effort) == ("claude-opus-5-5", "claude-code", "high")
        assert task.skills == ["feature-pipeline"] and task.workspace_kind == "worktree" and task.status == "ready"
    run = _claim(root)
    assert [row["task_id"] for row in kb.list_orchestrated_tasks_for_session("hermes-chat-c1")] == [root]

    astra = FakeAstra(_verdict("changes", "item i3 has no check for an empty body"), _verdict("approve"))
    first = fp.plan_review(root_id=root, plan={**PLAN, "items": PLAN["items"][:2]}, board="proj", run_id=run, reviewer=astra)
    assert first["action"] == "revise" and first["findings"][0]["issue"].startswith("item i3")
    second = fp.plan_review(root_id=root, plan=PLAN, board="proj", run_id=run, reviewer=astra)
    assert second["action"] == "approve" and second["feature_system"]["feature_system"]
    assert astra.calls[0]["cwd"] == str(home["repo"]), "Astra reviews inside the project checkout"
    assert "Earlier rounds" in astra.calls[1]["prompt"] and "empty body" in astra.calls[1]["prompt"]
    assert _status(root)["phase"] == "approved"
    assert home["notes"][-1] == ("hermes-chat-c1", "Feature pipeline: Astra approved the plan in round 2: "
                                 "3 items in 2 region(s). Notes: page, API and store.")

    launched = fp.launch(root_id=root, board="proj", run_id=run)
    cards = [region["card"] for region in launched["regions"]]
    assert [region["items"] for region in launched["regions"]] == [["i1", "i2"], ["i3"]]
    with kb.connect_closing(board="proj") as conn:
        assert kb.get_task(conn, root).status == "todo", "the root waits on its regions"
        for card in cards:
            region = kb.get_task(conn, card)
            assert region.status == "ready" and region.skills == ["feature-pipeline-region"]
            assert (region.model_override, region.review_model_override) == ("claude-opus-5-5", "gpt-6.2-astra")
            assert region.reasoning_effort == "high" and "Your items, in order:" in region.body
    assert _status(root)["phase"] == "building"
    followed = {row["task_id"] for row in kb.list_orchestrated_tasks_for_session("hermes-chat-c1")}
    assert followed == set(cards), "the chat follows the regions too (the digest keeps their routine events quiet)"
    with pytest.raises(fp.PipelineError):
        fp.plan_review(root_id=root, plan=PLAN, board="proj", reviewer=astra)

    for card in cards:
        _through_review(card)
    run = _claim(root)
    state = _status(root)
    assert state["phase"] == "integrating"
    with kb.connect_closing(board="proj") as conn:
        integration = kb.get_task(conn, root).branch_name
        assert [card["branch"] for card in state["cards"]] == [kb.get_task(conn, c).branch_name for c in cards]
    assert integration.startswith("fp/") and integration.endswith("/integration")
    assert all(card["branch"].startswith(integration.rsplit("/", 1)[0] + "/region-") for card in state["cards"])

    fixed = fp.fix(root_id=root, findings=["the API returns 500 on an empty body"], board="proj", run_id=run)
    assert fixed["action"] == "fix"
    with kb.connect_closing(board="proj") as conn:
        fix_card = kb.get_task(conn, fixed["card"])
        assert integration in fix_card.body and fix_card.review_model_override == "gpt-6.2-astra"
        assert kb.get_task(conn, root).status == "todo"
    assert _status(root)["phase"] == "fixing"
    _through_review(fixed["card"])
    run = _claim(root)

    fp.finish(root_id=root, summary="Notes built and merged into main; pytest passes.", board="proj", run_id=run)
    with kb.connect_closing(board="proj") as conn:
        assert kb.get_task(conn, root).status == "done"
    assert _status(root)["phase"] == "done"
    assert home["notes"][-1][1] == "Feature pipeline finished. Notes built and merged into main; pytest passes."
    assert len(home["notes"]) == 3 and "2 region(s) started" in home["notes"][1][1]


def test_a_deadlock_goes_to_will_and_his_answer_reaches_astra(home):
    root = fp.start(project="proj", request="Audit the notes system.", conversation_id="c1", kind="audit")["root"]
    run = _claim(root)
    astra = FakeAstra(_verdict("changes", "the plan skips the sync path"),
                      _verdict("changes", "the plan skips the sync path"),
                      _verdict("approve"))
    fp.plan_review(root_id=root, plan=PLAN, board="proj", run_id=run, reviewer=astra)
    stuck = fp.plan_review(root_id=root, plan=PLAN, context="Sync is out of scope.", board="proj", run_id=run,
                           reviewer=astra)
    assert stuck["action"] == "escalate" and "feature_pipeline ask" in stuck["next"]
    assert "reviewing an audit plan" in astra.calls[0]["prompt"]
    assert _status(root)["phase"] == "escalated"
    with pytest.raises(fp.PipelineError, match="call ask"):
        fp.plan_review(root_id=root, plan=PLAN, board="proj", run_id=run, reviewer=astra)
    with pytest.raises(fp.PipelineError, match="not waiting on Will"):
        fp.answer(root_id=root, answer="early", board="proj")

    fp.ask(root_id=root, question="Astra wants the sync path audited; I say it is out of scope. Which?",
           board="proj", run_id=run)
    with kb.connect_closing(board="proj") as conn:
        assert kb.get_task(conn, root).status == "blocked"
    assert _status(root)["phase"] == "asked"
    assert home["notes"][-1] == ("hermes-chat-c1", "Feature pipeline question for Will: Astra wants the sync path "
                                 "audited; I say it is out of scope. Which?"), "the whole question reaches Will"
    with pytest.raises(fp.PipelineError, match="waiting on Will"):
        fp.plan_review(root_id=root, plan=PLAN, board="proj", reviewer=astra)
    assert fp.answer(root_id=root, answer="Skip sync; it is audited next week.", board="proj")["resumed"] is True

    run = _claim(root)
    state = _status(root)
    assert state["phase"] == "answered" and state["decisions"][-1]["answer"] == "Skip sync; it is audited next week."
    settled = fp.plan_review(root_id=root, plan=PLAN, board="proj", run_id=run, reviewer=astra)
    assert settled["action"] == "approve"
    assert ("- Asked: Astra wants the sync path audited; I say it is out of scope. Which?\n"
            "  Will: Skip sync; it is audited next week.") in astra.calls[-1]["prompt"]
    assert _status(root)["phase"] == "approved"
    cards = [r["card"] for r in fp.launch(root_id=root, board="proj", run_id=run)["regions"]]
    with kb.connect_closing(board="proj") as conn:
        body = kb.get_task(conn, cards[0]).body
    assert "Will's decisions since the request (they win where they differ from it):" in body
    assert "  Will: Skip sync; it is audited next week." in body, "the region worker and its reviewer see his answer"


def test_a_second_question_waits_for_will_instead_of_going_to_triage(home):
    root = fp.start(project="proj", request="Build notes.", conversation_id="c1")["root"]
    for number in (1, 2, 3):
        run = _claim(root)
        fp.ask(root_id=root, question=f"question {number}", board="proj", run_id=run)
        with kb.connect_closing(board="proj") as conn:
            assert kb.get_task(conn, root).status == "blocked", f"question {number}"
        fp.answer(root_id=root, answer=f"answer {number}", board="proj")
    assert [d["answer"] for d in _status(root)["decisions"]] == ["answer 1", "answer 2", "answer 3"]


def test_a_question_whose_card_was_never_parked_is_asked_again(home, monkeypatch):
    root = fp.start(project="proj", request="Build notes.", conversation_id="c1")["root"]
    run = _claim(root)
    with monkeypatch.context() as patched:
        patched.setattr("hermes_cli.kanban_db.block_task", lambda *a, **kw: False)
        with pytest.raises(fp.PipelineError, match="could not be parked"):
            fp.ask(root_id=root, question="Ship without undo?", board="proj", run_id=run)
    assert _status(root)["phase"] == "escalated", "an unparked question has not reached Will"
    with pytest.raises(fp.PipelineError, match="not waiting on Will"):
        fp.answer(root_id=root, answer="yes", board="proj")
    fp.ask(root_id=root, question="Ship without undo?", board="proj", run_id=run)
    assert _status(root)["phase"] == "asked"


def test_an_answer_amended_during_a_review_voids_that_review(home):
    root = fp.start(project="proj", request="Build notes.", conversation_id="c1")["root"]
    run = _claim(root)
    fp.ask(root_id=root, question="Keep the legacy export?", board="proj", run_id=run)
    fp.answer(root_id=root, answer="Yes", board="proj")
    run = _claim(root)

    def astra_while_will_changes_his_mind(prompt, *, cwd):
        fp.answer(root_id=root, answer="No, drop it", board="proj")
        return {"response": _verdict("approve")}

    with pytest.raises(fp.PipelineError, match="changed while Astra was reviewing"):
        fp.plan_review(root_id=root, plan=PLAN, board="proj", run_id=run, reviewer=astra_while_will_changes_his_mind)
    assert _status(root)["phase"] == "answered"
    assert _status(root)["decisions"] == [{"question": "Keep the legacy export?", "answer": "No, drop it"}], \
        "the amended answer replaces the first one"
    retry = FakeAstra(_verdict("approve"))
    fp.plan_review(root_id=root, plan=PLAN, board="proj", run_id=run, reviewer=retry)
    assert "Will: No, drop it" in retry.calls[0]["prompt"] and "Will: Yes" not in retry.calls[0]["prompt"]


def test_a_question_on_a_reclaimed_card_is_asked_again(home):
    root = fp.start(project="proj", request="Build notes.", conversation_id="c1")["root"]
    _claim(root)
    with kb.connect_closing(board="proj") as conn:
        fp.record(conn, root, {"type": "ask", "question": "Undo?"})  # the run died before parking the card
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'ready', current_run_id = NULL, claim_lock = NULL WHERE id = ?",
                         (root,))
    assert _status(root)["phase"] == "escalated"
    with pytest.raises(fp.PipelineError, match="not waiting on Will"):
        fp.answer(root_id=root, answer="yes", board="proj")


def test_start_needs_a_git_checkout_with_a_commit(home, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    subprocess.run(["git", "init", "-q", str(empty)], check=True)
    registry = tmp_path / ".hermes" / "subscription-projects.json"
    registry.write_text(json.dumps({"proj": str(home["repo"]), "empty": str(empty)}))
    with pytest.raises(fp.PipelineError, match="no git checkout with a commit"):
        fp.start(project="empty", request="Build notes.", conversation_id="c1")


def test_decisions_keep_each_question_and_only_the_latest_answer_to_it():
    events = [
        {"type": "ask", "question": "Undo?"},
        {"type": "will_answer", "answer": "No"},
        {"type": "will_answer", "card": "t_r1", "question": "CSV or JSON?", "answer": "JSON"},
        {"type": "will_answer", "answer": "Yes, undo"},
        {"type": "will_answer", "card": "t_r1", "question": "CSV or JSON?", "answer": "CSV"},
        {"type": "will_answer", "card": "t_r1", "question": "Which date format?", "answer": "ISO"},
        {"type": "will_answer", "card": "t_r2", "answer": "tabs"},
        {"type": "will_answer", "card": "t_r2", "answer": "UTF-8"},
        {"type": "will_answer", "card": "t_r3", "question": "log\n" * 1500 + "Retry?", "answer": "yes"},
        {"type": "will_answer", "card": "t_r3", "question": "log\n" * 1500 + "Skip?", "answer": "no"},
    ]
    assert [d["answer"] for d in fp.decisions(events)] == ["Yes, undo", "CSV", "ISO", "tabs", "UTF-8", "yes", "no"], \
        "replies without a recorded question, and long questions that differ only at the end, stay apart"
    assert fp.decisions(events)[:3] == [
        {"question": "Undo?", "answer": "Yes, undo"},
        {"question": "On card t_r1: CSV or JSON?", "answer": "CSV"},
        {"question": "On card t_r1: Which date format?", "answer": "ISO"},
    ]


def test_a_build_needs_a_branch_to_land_on(home):
    repo = str(home["repo"])
    subprocess.run(["git", "-C", repo, "checkout", "-q", "--detach"], check=True)
    with pytest.raises(fp.PipelineError, match="detached HEAD"):
        fp.start(project="proj", request="Build notes.", conversation_id="c1")
    assert fp.start(project="proj", request="Audit notes.", conversation_id="c1", kind="audit")["root"]


def test_start_needs_the_review_lane(home, monkeypatch):
    monkeypatch.setattr("hermes_cli.kanban_db.review_dispatch_enabled", lambda: False)
    with pytest.raises(fp.PipelineError, match="review lane is off"):
        fp.start(project="proj", request="Build notes.", conversation_id="c1")


def test_every_card_starts_from_the_commit_the_plan_was_reviewed_against(home):
    repo = str(home["repo"])
    git = lambda *a: subprocess.run(["git", "-C", repo, *a], check=True, capture_output=True, text=True).stdout.strip()
    branch = lambda card: kb.get_task(conn, card).branch_name
    base = git("rev-parse", "HEAD")
    root = fp.start(project="proj", request="Build notes.", conversation_id="c1")["root"]
    with kb.connect_closing(board="proj") as conn:
        assert git("rev-parse", branch(root)) == base, "the branch exists before the card does"
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "someone else's work")
    run = _claim(root)
    fp.plan_review(root_id=root, plan=PLAN, board="proj", run_id=run, reviewer=FakeAstra(_verdict("approve")))
    cards = [r["card"] for r in fp.launch(root_id=root, board="proj", run_id=run)["regions"]]
    with kb.connect_closing(board="proj") as conn:
        assert {git("rev-parse", branch(card)) for card in cards} == {base}, "regions ignore the newer HEAD"
        integration = branch(root)
    for card in cards:
        _through_review(card)
    git("branch", "-f", integration, "main")  # the root's integration moved its branch on
    run = _claim(root)
    fix_card = fp.fix(root_id=root, findings=["x"], board="proj", run_id=run)["card"]
    with kb.connect_closing(board="proj") as conn:
        assert git("rev-parse", branch(fix_card)) == git("rev-parse", integration), "a fix starts from the integration"


def test_pin_branch_refuses_what_it_cannot_verify(home):
    with pytest.raises(fp.PipelineError, match="not available"):
        fp.pin_branch("/dev/null", "fp/x/integration", "nonempty")
    with pytest.raises(fp.PipelineError, match="not available"):
        fp.pin_branch(str(home["repo"]), "fp/x/integration", "0" * 40)


def test_a_launch_cut_off_before_linking_resumes_on_the_next_dispatch(home, monkeypatch):
    root = fp.start(project="proj", request="Build notes.", conversation_id="c1")["root"]
    run = _claim(root)
    fp.plan_review(root_id=root, plan=PLAN, board="proj", run_id=run, reviewer=FakeAstra(_verdict("approve")))
    with monkeypatch.context() as patched:
        patched.setattr("hermes_cli.kanban_db.link_tasks", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("died")))
        with pytest.raises(RuntimeError):
            fp.launch(root_id=root, board="proj", run_id=run)
    with kb.connect_closing(board="proj") as conn:
        with kb.write_txn(conn):  # the dispatcher reclaims the dead worker's card
            conn.execute("UPDATE tasks SET status = 'ready', current_run_id = NULL, claim_lock = NULL WHERE id = ?",
                         (root,))
    run = _claim(root)  # nothing gates the root on a half-made card
    assert _status(root)["phase"] == "approved"
    cards = [r["card"] for r in fp.launch(root_id=root, board="proj", run_id=run)["regions"]]
    with kb.connect_closing(board="proj") as conn:
        assert sorted(kb.parent_ids(conn, root)) == sorted(cards)
        assert len([t for t in kb.list_tasks(conn) if t.title.startswith("Region")]) == len(cards) == 2
        assert kb.get_task(conn, root).status == "todo"


def test_a_stale_run_cannot_act_for_the_card(home):
    root = fp.start(project="proj", request="Build notes.", conversation_id="c1")["root"]
    run = _claim(root)
    with pytest.raises(fp.PipelineError, match="no longer"):
        fp.plan_review(root_id=root, plan=PLAN, board="proj", run_id=run + 100, reviewer=FakeAstra(_verdict("approve")))


def test_a_launch_that_stopped_part_way_resumes_without_duplicating_regions(home, monkeypatch):
    root = fp.start(project="proj", request="Build notes.", conversation_id="c1")["root"]
    run = _claim(root)
    fp.plan_review(root_id=root, plan=PLAN, board="proj", run_id=run, reviewer=FakeAstra(_verdict("approve")))
    real_record = fp.record
    calls = {"n": 0}

    def crash_on_launch_record(conn, root_id, event):
        if event.get("type") == "launch" and not calls["n"]:
            calls["n"] += 1
            raise RuntimeError("process died")
        return real_record(conn, root_id, event)

    monkeypatch.setattr(fp, "record", crash_on_launch_record)
    with pytest.raises(RuntimeError):
        fp.launch(root_id=root, board="proj", run_id=run)
    cards = [r["card"] for r in fp.launch(root_id=root, board="proj", run_id=run)["regions"]]
    with kb.connect_closing(board="proj") as conn:
        regions = [t for t in kb.list_tasks(conn) if t.title.startswith("Region")]
        assert sorted(t.id for t in regions) == sorted(cards), "the retry found the cards it had made"
        assert kb.get_task(conn, root).status == "todo"
    with pytest.raises(fp.PipelineError, match="already launched"):
        fp.launch(root_id=root, board="proj")
    for card in cards:
        _through_review(card)
    run = _claim(root)
    with pytest.raises(fp.PipelineError, match="already launched"):
        fp.launch(root_id=root, board="proj", run_id=run)
    with kb.connect_closing(board="proj") as conn:
        assert kb.get_task(conn, root).status == "running", "a late launch call leaves the integrating root alone"


def test_a_finding_that_survives_two_fix_loops_goes_to_will(home):
    root, cards = _launched_root()
    for card in cards:
        _through_review(card)
    finding = "the notes API returns 500 on an empty body"
    for _ in range(2):
        run = _claim(root)
        card = fp.fix(root_id=root, findings=[finding], board="proj", run_id=run)["card"]
        _through_review(card)
    run = _claim(root)
    third = fp.fix(root_id=root, findings=[finding], board="proj", run_id=run)
    assert third["action"] == "escalate" and "card" not in third
    assert _status(root)["phase"] == "escalated"
    with pytest.raises(fp.PipelineError, match="call ask"):
        fp.finish(root_id=root, summary="ship it", board="proj", run_id=run)
    fp.ask(root_id=root, question="The empty-body 500 survived two fixes. 400 or 422?", board="proj", run_id=run)
    fp.answer(root_id=root, answer="Return 400 instead.", board="proj")
    run = _claim(root)
    answered = fp.fix(root_id=root, findings=[finding], board="proj", run_id=run)
    assert answered["action"] == "fix"
    with kb.connect_closing(board="proj") as conn:
        body = kb.get_task(conn, answered["card"]).body
    assert "Will was asked: The empty-body 500 survived two fixes. 400 or 422?" in body
    assert "Will decided: Return 400 instead." in body


def test_will_can_stop_a_pipeline_before_anything_is_built(home):
    root = fp.start(project="proj", request="Build notes.", conversation_id="c1")["root"]
    run = _claim(root)
    with pytest.raises(fp.PipelineError):
        fp.finish(root_id=root, summary="nothing", board="proj", run_id=run)
    fp.ask(root_id=root, question="The request conflicts with the sync redesign. Stop?", board="proj", run_id=run)
    fp.answer(root_id=root, answer="Yes, stop.", board="proj")
    run = _claim(root)
    fp.finish(root_id=root, summary="Stopped at Will's word before building.", board="proj", run_id=run)
    assert _status(root)["phase"] == "done"


def test_wills_reply_to_a_stuck_region_reaches_that_card(home):
    root, cards = _launched_root()
    with kb.connect_closing(board="proj") as conn:
        run = kb.claim_task(conn, cards[0], claimer="opus")
        assert kb.block_task(conn, cards[0], reason="CSV or JSON export?", kind="needs_input",
                             expected_run_id=run.current_run_id)
    with pytest.raises(fp.PipelineError, match="not one of this pipeline"):
        fp.answer(root_id=root, answer="CSV", card="t_elsewhere", board="proj")
    with pytest.raises(fp.PipelineError, match="ask Will for one complete answer") as too_long:
        fp.answer(root_id=root, answer="x" * 20001, card=cards[0], board="proj")
    assert "parts" not in str(too_long.value), "a split answer would replace its first part"
    assert fp.answer(root_id=root, answer="CSV", card=cards[0], board="proj")["resumed"] is True
    with kb.connect_closing(board="proj") as conn:
        assert kb.get_task(conn, cards[0]).status == "ready"
        assert any(c.body == "Will (through the chat): CSV" and c.author == fp.STATE_AUTHOR
                   for c in kb.list_comments(conn, cards[0]))
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'triage' WHERE id = ?", (cards[1],))
    with pytest.raises(fp.PipelineError, match="went to triage"):
        fp.answer(root_id=root, answer="retry", card=cards[1], board="proj")
    assert _status(root)["decisions"] == [{"question": f"On card {cards[0]}: CSV or JSON export?", "answer": "CSV"}], \
        "his reply to a region reaches later fix cards and Astra through the decisions"
    with kb.connect_closing(board="proj") as conn:  # the same card blocks again, this time with no reason
        run = kb.claim_task(conn, cards[0], claimer="opus")
        assert kb.block_task(conn, cards[0], expected_run_id=run.current_run_id)  # a plain `kanban block`
    fp.answer(root_id=root, answer="Keep going", card=cards[0], board="proj")
    assert [d["answer"] for d in _status(root)["decisions"]] == ["CSV", "Keep going"], "both replies survive"
    with kb.connect_closing(board="proj") as conn:  # and once more while it is ready, which opens no run
        assert kb.block_task(conn, cards[0], kind="capability")
    fp.answer(root_id=root, answer="Still going", card=cards[0], board="proj")
    assert [d["answer"] for d in _status(root)["decisions"]] == ["CSV", "Keep going", "Still going"]


def test_a_question_must_fit_the_chat_notice_whole(home):
    root = fp.start(project="proj", request="Build notes.", conversation_id="c1")["root"]
    run = _claim(root)
    with pytest.raises(fp.PipelineError, match="too long for the chat notice"):
        fp.ask(root_id=root, question="x" * 600, board="proj", run_id=run)
    fp.ask(root_id=root, question="Export format?\nA: CSV\nB: JSON", board="proj", run_id=run)
    with kb.connect_closing(board="proj") as conn:
        reasons = [r.summary for r in kb.list_runs(conn, root) if r.summary]
    assert reasons[-1] == "Question for Will: Export format? / A: CSV / B: JSON", "one line, options included"


def test_start_refuses_a_checkout_the_dispatcher_would_reuse(home, tmp_path):
    repo = str(home["repo"])
    linked = tmp_path / "proj-linked"
    subprocess.run(["git", "-C", repo, "worktree", "add", "-q", "-b", "side", str(linked)], check=True)
    registry = tmp_path / ".hermes" / "subscription-projects.json"
    registry.write_text(json.dumps({"proj": repo, "linked": str(linked), "sub": str(home["repo"] / "src")}))
    (home["repo"] / "src").mkdir()
    with pytest.raises(fp.PipelineError, match="main checkout"):
        fp.start(project="linked", request="Build notes.", conversation_id="c1")
    with pytest.raises(fp.PipelineError, match="main checkout"):
        fp.start(project="sub", request="Build notes.", conversation_id="c1")
    kb.create_board("dot", default_workdir=".")
    with pytest.raises(fp.PipelineError, match="absolute checkout path"):
        fp.start(project="dot", request="Build notes.", conversation_id="c1")


def test_every_card_is_routed_to_the_checkout_its_branch_was_pinned_in(home):
    root, cards = _launched_root()
    with kb.connect_closing(board="proj") as conn:
        assert {kb.get_task(conn, card).workspace_path for card in [root, *cards]} == {str(home["repo"])}


def test_a_pr_link_in_wills_request_does_not_hold_the_root(home):
    """Astra, 2026-11-18: the dispatcher's respawn guard read a PR URL quoted in
    the start record as a worker's open PR and held the root for a day."""
    root = fp.start(project="proj", request="Audit the checkout change in https://github.com/example/shop/pull/42",
                    conversation_id="c1")["root"]
    with kb.connect_closing(board="proj") as conn:
        assert kb.check_respawn_guard(conn, root) is None
        kb.add_comment(conn, root, "default", "Opened https://github.com/example/shop/pull/43")
        assert kb.check_respawn_guard(conn, root) == "active_pr", "a worker's own PR link still holds it"
    assert fp.STATE_AUTHOR in kb._RESPAWN_GUARD_IGNORED_AUTHORS


def test_resolve_board_fills_a_missing_working_directory_and_refuses_unknown_projects(home):
    kb.create_board("proj")
    assert not kb.read_board_metadata("proj").get("default_workdir")
    assert fp.resolve_board("proj") == "proj"
    assert kb.read_board_metadata("proj")["default_workdir"] == str(home["repo"])
    with pytest.raises(fp.PipelineError, match="not in"):
        fp.resolve_board("nowhere")


# --- the tool --------------------------------------------------------------

def test_the_tool_keeps_start_and_answer_with_the_chat_and_the_rest_with_the_root(home, monkeypatch):
    from tools.feature_pipeline_tool import feature_pipeline_tool as tool

    assert "feature system" in json.loads(tool("definition"))["definition"]
    assert "conversation_id is required" in json.loads(tool("start", project="proj", request="x"))["error"]
    monkeypatch.setenv("HERMES_GATEWAY_SESSION_ID", "hermes-chat-c9")
    started = json.loads(tool("start", project="proj", request="Build the notes feature."))
    root = started["root"]
    assert kb.list_orchestrated_tasks_for_session("hermes-chat-c9")[0]["task_id"] == root, \
        "the lane's session id stands in for a missing conversation_id"
    assert json.loads(tool("status", root_id=root))["phase"] == "planning", "status finds the board itself"
    refused = json.loads(tool("ask", root_id=root, question="?"))
    assert refused["ok"] is False and "action answer" in refused["error"]
    assert "not waiting on Will" in json.loads(tool("answer", root_id=root, answer="yes"))["error"]

    run = _claim(root)
    monkeypatch.setenv("HERMES_KANBAN_TASK", root)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(kb.kanban_db_path(board="proj")))
    assert "cannot start" in json.loads(tool("start", project="proj", request="Another one."))["error"]
    assert "cannot answer" in json.loads(tool("answer", answer="yes"))["error"]
    assert json.loads(tool("status"))["phase"] == "planning", "a worker's own card is the default root"
    assert "launch" in json.loads(tool("launch"))["error"], "the root's own worker reaches launch's own checks"
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run + 1))
    assert "no longer" in json.loads(tool("ask", question="?"))["error"], "the run id comes from the worker env"
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID")
    assert "HERMES_KANBAN_RUN_ID is missing" in json.loads(tool("ask", question="?"))["error"]
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run))
    from agent.delegation_context import non_dispatcher_owned_context

    with non_dispatcher_owned_context():
        assert "can only read" in json.loads(tool("ask", question="?"))["error"], "a cron job inside the worker"
        assert "can only read" in json.loads(tool("answer", root_id=root, answer="yes"))["error"]
        assert json.loads(tool("status", root_id=root))["phase"] == "planning"
