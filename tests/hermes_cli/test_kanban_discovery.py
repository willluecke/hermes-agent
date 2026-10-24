"""Discovery sweep: the orchestrator reads first, parks candidates, a human picks.

Everything here runs without a model: the sweep runner is injected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_discovery as disc
from hermes_cli import kanban_orchestrator as orch


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _cand(title, conf="medium", evidence=(), angle="half-built", **kw):
    return disc.Candidate(
        title=title, confidence=conf, evidence=list(evidence), angles=[angle], **kw
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_parse_bare_array_and_fenced_array():
    raw = '[{"title": "A", "confidence": "high", "evidence": ["x.py:1"]}]'
    assert [c.title for c in disc.parse_candidates(raw, "k")] == ["A"]
    fenced = "Here you go:\n```json\n" + raw + "\n```"
    out = disc.parse_candidates(fenced, "k")
    assert out[0].title == "A" and out[0].confidence == "high"
    assert out[0].evidence == ["x.py:1"] and out[0].angles == ["k"]


def test_parse_object_with_candidates_key_and_garbage():
    raw = '{"candidates": [{"title": "B", "confidence": "bogus", "evidence": "y.ts:4"}]}'
    out = disc.parse_candidates(raw, "k")
    assert out[0].title == "B"
    assert out[0].confidence == "medium"  # unknown confidence normalised
    assert out[0].evidence == ["y.ts:4"]  # string evidence wrapped
    assert disc.parse_candidates("no json here", "k") == []
    assert disc.parse_candidates("", "k") == []
    assert disc.parse_candidates('[{"confidence": "high"}]', "k") == []  # no title


# ---------------------------------------------------------------------------
# Merge / rank / exclude
# ---------------------------------------------------------------------------

def test_merge_folds_duplicates_across_angles():
    a = [_cand("Queued turn can retry a 425 start forever", "medium", ["q.mjs:371"], "half-built")]
    b = [_cand("Queued turn retries a 425 start forever and can't be removed", "high", ["q.mjs:371", "page.tsx:5125"], "reliability")]
    c = [_cand("Composer rejects m4a files", "low", ["Composer.tsx:403"], "promises")]
    merged = disc.merge_candidates([a, b, c])
    assert len(merged) == 2
    dup = next(m for m in merged if "425" in m.title)
    assert dup.confidence == "high"
    assert sorted(dup.angles) == ["half-built", "reliability"]
    assert dup.evidence == ["q.mjs:371", "page.tsx:5125"]


def test_merge_matches_on_shared_evidence_even_with_different_titles():
    a = [_cand("Login route 500s on bad env", "medium", ["login/route.ts:47"])]
    b = [_cand("Optional preview-auth setting breaks password login", "medium", ["login/route.ts:47"], "promises")]
    merged = disc.merge_candidates([a, b])
    assert len(merged) == 1 and sorted(merged[0].angles) == ["half-built", "promises"]


def test_rank_prefers_confidence_then_angles_then_evidence():
    ranked = disc.rank_candidates([
        _cand("low one", "low"),
        _cand("high one", "high"),
        disc.Candidate(title="medium two angles", confidence="medium", angles=["a", "b"]),
        _cand("medium one angle", "medium"),
    ])
    assert [c.title for c in ranked] == [
        "high one", "medium two angles", "medium one angle", "low one",
    ]


def test_exclude_known_drops_similar_titles():
    cands = [_cand("Push notifications have no off switch"), _cand("Docs drift across README")]
    kept = disc.exclude_known(cands, ["push notifications: no off switch"])
    assert [c.title for c in kept] == ["Docs drift across README"]


# ---------------------------------------------------------------------------
# Board I/O
# ---------------------------------------------------------------------------

def test_post_candidates_parks_as_needs_input_and_is_idempotent(kanban_home: Path):
    cands = [_cand("Sync has no timeout", "high", ["sync.ts:160"]), _cand("Offline page lies", "medium")]
    with kb.connect_closing() as conn:
        posted, skipped = disc.post_candidates(conn, cands, source="test")
        assert len(posted) == 2 and skipped == []
        t = kb.get_task(conn, posted[0])
        assert t.status == "blocked"
        assert t.block_kind == "needs_input"
        assert t.created_by == disc.CANDIDATE_CREATOR
        assert "confidence: high" in t.body and "sync.ts:160" in t.body
        assert "rank: 1/2" in t.body
        assert [c.id for c in cands] == posted
        # the whole list is visible as candidates, in rank order
        assert [r.id for r in disc.list_candidates(conn)] == posted
        # nothing for the dispatcher or decomposer
        assert kb.list_tasks(conn, status="ready") == []
        assert kb.list_tasks(conn, status="triage") == []
        # second sweep with the same findings posts nothing new
        again, skipped2 = disc.post_candidates(conn, [_cand("Sync has no timeout")], source="test")
        assert again == [] and skipped2[0]["id"] == posted[0]


def test_rejected_candidate_is_not_reposted(kanban_home: Path):
    with kb.connect_closing() as conn:
        (tid,), _ = disc.post_candidates(conn, [_cand("Composer rejects m4a")], source="t")
        assert kb.archive_task(conn, tid)
        assert disc.list_candidates(conn) == []
        posted, skipped = disc.post_candidates(conn, [_cand("Composer rejects m4a")], source="t")
        assert posted == [] and skipped[0]["reason"] == "already on board"
        assert disc.known_titles(conn) == ["Composer rejects m4a"]


def test_accept_candidate_moves_to_triage_and_only_candidates(kanban_home: Path):
    with kb.connect_closing() as conn:
        (tid,), _ = disc.post_candidates(conn, [_cand("Notes lose unsaved edits")], source="t")
        assert kb.accept_candidate(conn, tid, note="worth it")
        t = kb.get_task(conn, tid)
        assert t.status == "triage" and t.block_kind is None
        assert not kb.accept_candidate(conn, tid)  # already accepted
        # an ordinary blocked task (no needs_input kind) is not accept-able
        other = kb.create_task(conn, title="plain", initial_status="blocked")
        assert not kb.accept_candidate(conn, other)
        events = [(r["kind"], json.loads(r["payload"] or "null")) for r in conn.execute(
            "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id", (tid,)
        )]
        kinds = [k for k, _ in events]
        assert "blocked" in kinds and "candidate_accepted" in kinds
        parked = next(pl for k, pl in events if k == "blocked")
        assert parked["kind"] == "needs_input" and parked["candidate"] is True


def test_parked_candidates_survive_recompute_ready(kanban_home: Path):
    """recompute_ready auto-promotes parentless blocked tasks unless the block
    is sticky; a parked candidate must never be promoted behind Will's back."""
    with kb.connect_closing() as conn:
        (cid,), _ = disc.post_candidates(conn, [_cand("Parked")], source="t")
        plain = kb.create_task(conn, title="untyped block", initial_status="blocked")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, cid).status == "blocked"
        assert kb.get_task(conn, plain).status == "ready"  # the pre-existing auto-recover path
        # archiving a sibling runs recompute_ready too
        assert kb.archive_task(conn, plain)
        assert kb.get_task(conn, cid).status == "blocked"
        assert kb.list_tasks(conn, status="ready") == []


def test_park_candidate_requires_blocked_status(kanban_home: Path):
    with kb.connect_closing() as conn:
        ready = kb.create_task(conn, title="ready task")
        assert not kb.park_candidate(conn, ready, reason="x")
        with pytest.raises(ValueError):
            kb.park_candidate(conn, ready, reason="x", kind="not-a-kind")


# ---------------------------------------------------------------------------
# discover(): end to end with an injected runner
# ---------------------------------------------------------------------------

def _runner_for(outputs: dict[str, str], seen: list | None = None):
    def run(prompt: str, project_root: str, timeout: int) -> str:
        if seen is not None:
            seen.append((prompt, project_root, timeout))
        for key, out in outputs.items():
            if f"Angle: {key}" in prompt:
                if isinstance(out, Exception):
                    raise out
                return out
        return "[]"
    return run


def test_discover_sweeps_every_angle_and_parks_ranked_candidates(kanban_home: Path, tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    seen: list = []
    outputs = {
        "Half-built features": json.dumps([
            {"title": "Dashboard approve only flips a status", "confidence": "medium",
             "evidence": ["Dashboard.tsx:982"], "impact": "approving runs nothing"},
        ]),
        "Promises vs reality": json.dumps([
            {"title": "Offline page promises cached conversations", "confidence": "high",
             "evidence": ["offline/page.tsx:7"]},
            {"title": "Dashboard Approve/Reject only flips a status", "confidence": "high",
             "evidence": ["Dashboard.tsx:982"]},
        ]),
        "Reliability and test gaps": RuntimeError("model hiccup"),
    }
    with kb.connect_closing() as conn:
        kb.create_task(conn, title="Already queued fix")  # known title, excluded from prompts
        report = disc.discover(
            conn, board="proj", project_root=str(repo), runner=_runner_for(outputs, seen),
            timeout=5, parallel=False, log=lambda m: None,
        )
    assert len(seen) == 3
    assert all(root == str(repo) and t == 5 for _, root, t in seen)
    assert all("Already queued fix" in prompt for prompt, _, _ in seen)
    assert all("read_only" not in prompt for prompt, _, _ in seen)  # toolset is the runner's job
    assert report["errors"] == {"reliability": "model hiccup"}
    assert report["parsed"] == {"half-built": 1, "promises": 2, "reliability": 0}
    titles = [c["title"] for c in report["candidates"]]
    # the duplicate merged, kept the stronger confidence, and ranks first on two angles
    assert titles[0].startswith("Dashboard")
    assert report["candidates"][0]["confidence"] == "high"
    assert sorted(report["candidates"][0]["angles"]) == ["half-built", "promises"]
    assert len(report["posted"]) == 2
    with kb.connect_closing() as conn:
        rows = disc.list_candidates(conn)
        assert [r.id for r in rows] == report["posted"]
        assert all(r.status == "blocked" and r.block_kind == "needs_input" for r in rows)


def test_discover_rejects_missing_root(kanban_home: Path, tmp_path: Path):
    with kb.connect_closing() as conn:
        with pytest.raises(ValueError):
            disc.discover(conn, board="p", project_root=str(tmp_path / "nope"), runner=_runner_for({}))


def test_hermes_chat_runner_uses_read_only_toolset(monkeypatch, tmp_path: Path):
    calls = []

    class _Proc:
        returncode = 0
        stdout = "[]"
        stderr = ""

    def fake_run(argv, **kw):
        calls.append((argv, kw))
        return _Proc()

    monkeypatch.setattr(disc.subprocess, "run", fake_run)
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker")
    out = disc.hermes_chat_runner("PROMPT", str(tmp_path), 7)
    assert out == "[]"
    argv, kw = calls[0]
    assert argv[-5:] == ["chat", "-Q", "-t", "read_only", "-q"] or argv[-6:-1] == ["chat", "-Q", "-t", "read_only", "-q"]
    assert "-q" in argv and argv[argv.index("-q") + 1] == "PROMPT"
    assert kw["cwd"] == str(tmp_path) and kw["timeout"] == 7
    assert "HERMES_KANBAN_TASK" not in kw["env"]  # never looks like a worker


def test_format_candidates_lists_accept_command(kanban_home: Path):
    with kb.connect_closing() as conn:
        disc.post_candidates(conn, [_cand("Sync has no timeout", "high")], source="t")
        text = disc.format_candidates(disc.list_candidates(conn), board="proj")
    assert "awaiting acceptance on proj" in text
    assert "[high]  Sync has no timeout" in text
    assert "hermes kanban accept" in text
    assert disc.format_candidates([]) == "No discovery candidates awaiting acceptance."


# ---------------------------------------------------------------------------
# Orchestrator integration
# ---------------------------------------------------------------------------

def test_plan_never_reopens_or_retires_human_gated_blocks():
    snap = orch.BoardSnapshot(tasks=(
        orch.TaskView("c1", "candidate", "blocked", block_kind="needs_input",
                      block_recurrences=99, created_by="discovery-sweep"),
        orch.TaskView("b1", "real block", "blocked", block_kind="capability"),
    ))
    plan = orch.plan_cycle(snap, orch.Envelope(max_concurrent_workers=4, max_attempts=3))
    assert plan.open_ids == ("b1",)
    assert plan.retire_ids == () and plan.held_ids == ()


def test_discovery_recommended_only_when_nothing_to_route():
    empty = orch.BoardSnapshot()
    assert orch.discovery_recommended(empty) == (True, "no open work on the board")
    done_only = orch.BoardSnapshot(tasks=(orch.TaskView("d", "x", "done"),))
    assert orch.discovery_recommended(done_only)[0] is True
    with_todo = orch.BoardSnapshot(tasks=(orch.TaskView("t", "x", "todo"),))
    assert orch.discovery_recommended(with_todo)[0] is False
    reopenable = orch.BoardSnapshot(tasks=(orch.TaskView("b", "x", "blocked", block_kind="capability"),))
    assert orch.discovery_recommended(reopenable)[0] is False
    pending = orch.BoardSnapshot(tasks=(
        orch.TaskView("c", "x", "blocked", block_kind="needs_input", created_by="discovery-sweep"),
    ))
    ok, why = orch.discovery_recommended(pending)
    assert ok is False and "await acceptance" in why


def test_summary_separates_candidates_from_blocked(kanban_home: Path):
    with kb.connect_closing() as conn:
        (cid,), _ = disc.post_candidates(conn, [_cand("A candidate")], source="t")
        plain = kb.create_task(conn, title="plain block", initial_status="blocked")
        summary = orch.summarize_board(conn)
    assert summary["candidates"] == [{"id": cid, "title": "A candidate"}]
    assert [b["id"] for b in summary["blocked"]] == [plain]


def test_run_cycle_reports_discovery_recommendation(kanban_home: Path):
    with kb.connect_closing() as conn:
        res = orch.run_cycle(conn)
    assert res["discovery"] == {"recommended": True, "reason": "no open work on the board"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _orch_ns(**over):
    base = dict(
        board=None, board_positional=None, max_concurrent=4, max_attempts=3, once=True,
        emit_dispatch=False, discover=False, no_discover=False, project_root=None,
        max_candidates=None, sweep_timeout=None,
    )
    base.update(over)
    return argparse.Namespace(**base)


# The runner is patched by dotted path, not via the ``disc`` alias above:
# another test file in this directory drops every ``hermes_cli.*`` module from
# ``sys.modules`` (test_kanban_cli_dispatch_passthrough), after which the CLI
# imports a fresh kanban_discovery module and a patch on the stale object
# would let the real ``hermes chat`` runner through.


def test_cli_orchestrate_runs_discovery_on_an_empty_board(kanban_home: Path, tmp_path: Path, capsys, monkeypatch):
    from hermes_cli import kanban as kc

    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(
        "hermes_cli.kanban_discovery.hermes_chat_runner",
        _runner_for({"Reliability and test gaps": json.dumps([
            {"title": "Sync has no timeout", "confidence": "high", "evidence": ["sync.ts:160"]},
        ])}),
    )
    assert kc._cmd_orchestrate(_orch_ns(project_root=str(repo))) == 0
    out, err = capsys.readouterr()
    payload = json.loads(out)
    assert payload["discovery"]["ran"] is True
    assert payload["discovery"]["recommended"] is True
    assert [c["title"] for c in payload["discovery"]["candidates"]] == ["Sync has no timeout"], payload["discovery"]
    assert len(payload["discovery"]["posted"]) == 1
    assert "awaiting acceptance" in err and "hermes kanban accept" in err
    with kb.connect_closing() as conn:
        assert len(disc.list_candidates(conn)) == 1
    # a second cycle does not re-sweep: candidates are waiting on a decision
    assert kc._cmd_orchestrate(_orch_ns(project_root=str(repo))) == 0
    payload2 = json.loads(capsys.readouterr().out)
    assert payload2["discovery"]["ran"] is False
    assert "await acceptance" in payload2["discovery"]["reason"]


def test_cli_orchestrate_no_discover_and_missing_root(kanban_home: Path, capsys, monkeypatch):
    from hermes_cli import kanban as kc

    monkeypatch.setattr("hermes_cli.kanban_discovery.hermes_chat_runner", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not run")))
    assert kc._cmd_orchestrate(_orch_ns(no_discover=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["discovery"]["ran"] is False and payload["discovery"]["skipped"] == "--no-discover"
    # recommended, but the board has no default workdir and none was passed
    assert kc._cmd_orchestrate(_orch_ns()) == 0
    out, err = capsys.readouterr()
    payload = json.loads(out)
    assert payload["discovery"]["skipped"] == "no project root"
    assert "set-default-workdir" in err


def test_cli_orchestrate_discover_flag_forces_a_sweep(kanban_home: Path, tmp_path: Path, capsys, monkeypatch):
    from hermes_cli import kanban as kc

    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr("hermes_cli.kanban_discovery.hermes_chat_runner", _runner_for({}))
    with kb.connect_closing() as conn:
        kb.create_task(conn, title="open work")  # ready: discovery not recommended
    assert kc._cmd_orchestrate(_orch_ns(project_root=str(repo))) == 0
    assert json.loads(capsys.readouterr().out)["discovery"]["ran"] is False
    assert kc._cmd_orchestrate(_orch_ns(project_root=str(repo), discover=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["discovery"]["ran"] is True and payload["discovery"]["posted"] == []


def test_cli_accept_reject_and_candidates(kanban_home: Path, capsys):
    from hermes_cli import kanban as kc

    with kb.connect_closing() as conn:
        posted, _ = disc.post_candidates(
            conn, [_cand("Keep me", "high"), _cand("Drop me"), _cand("Not yours")], source="t",
        )
        plain = kb.create_task(conn, title="plain block", initial_status="blocked")
    keep, drop, other = posted

    assert kc._cmd_candidates(argparse.Namespace()) == 0
    listed = capsys.readouterr().out
    assert "Keep me" in listed and "Drop me" in listed and "plain block" not in listed

    assert kc._cmd_accept(argparse.Namespace(task_id=keep, reason=["worth", "it"], ids=None)) == 0
    assert "Accepted" in capsys.readouterr().out
    assert kc._cmd_reject(argparse.Namespace(task_id=drop, reason=[], ids=None)) == 2  # reason required
    capsys.readouterr()
    assert kc._cmd_reject(argparse.Namespace(task_id=drop, reason=["not", "now"], ids=None)) == 0
    capsys.readouterr()
    # a plain blocked task is neither accept- nor reject-able through this path
    assert kc._cmd_accept(argparse.Namespace(task_id=plain, reason=[], ids=None)) == 1
    assert kc._cmd_reject(argparse.Namespace(task_id=plain, reason=["x"], ids=None)) == 1
    capsys.readouterr()
    with kb.connect_closing() as conn:
        assert kb.get_task(conn, keep).status == "triage"
        assert kb.get_task(conn, drop).status == "archived"
        assert kb.get_task(conn, plain).status not in ("triage", "archived")
        assert [r.id for r in disc.list_candidates(conn)] == [other]
        comments = [c.body for c in kb.list_comments(conn, drop)] if hasattr(kb, "list_comments") else []
        if comments:
            assert any("REJECT: not now" in c for c in comments)


def test_parser_wires_discovery_flags_and_commands():
    from hermes_cli import kanban as kc
    from hermes_cli._parser import build_top_level_parser

    parser, sub, _chat = build_top_level_parser()
    kc.build_parser(sub)
    args = parser.parse_args([
        "kanban", "orchestrate", "proj", "--discover", "--project-root", "/tmp/x",
        "--max-candidates", "5", "--sweep-timeout", "30",
    ])
    assert args.discover is True and args.no_discover is False
    assert args.project_root == "/tmp/x" and args.max_candidates == 5 and args.sweep_timeout == 30
    # reason words come before --ids, as with `block`: a greedy --ids would
    # otherwise swallow them
    args = parser.parse_args(["kanban", "accept", "t_1", "good", "one", "--ids", "t_2"])
    assert args.task_id == "t_1" and args.ids == ["t_2"] and args.reason == ["good", "one"]
    args = parser.parse_args(["kanban", "reject", "t_1", "not", "now"])
    assert args.reason == ["not", "now"]
    assert parser.parse_args(["kanban", "candidates"]).kanban_action == "candidates"


# ---------------------------------------------------------------------------
# Focused sweeps (hermes kanban orchestrate --focus)
# ---------------------------------------------------------------------------

def test_prompt_aims_at_the_goal_only_when_one_is_given():
    angle = disc.ANGLES[0]
    plain = disc.build_sweep_prompt(angle, "/repo")
    focused = disc.build_sweep_prompt(angle, "/repo", goal="get paid file sales live")
    assert "get paid file sales live" not in plain
    assert "report candidate problems worth fixing" in plain
    assert "stands between the current code and this goal: get paid file sales live" in focused
    assert "Ignore unrelated issues" in focused
    # the angle, rules and output contract are unchanged
    for text in (plain, focused):
        assert f"Angle: {angle.title}." in text
        assert "Return ONLY a JSON array" in text


def test_discover_sends_the_goal_to_every_angle_and_records_it(kanban_home: Path, tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    seen: list = []
    outputs = {"Half-built features": json.dumps([
        {"title": "Buy button posts to a route that 404s", "confidence": "high", "evidence": ["buy.tsx:12"]},
    ])}
    with kb.connect_closing() as conn:
        report = disc.discover(
            conn, board="proj", project_root=str(repo), runner=_runner_for(outputs, seen),
            parallel=False, goal="  get paid file sales live  ",
        )
        (row,) = disc.list_candidates(conn)
    assert report["goal"] == "get paid file sales live"
    assert len(seen) == len(disc.ANGLES)
    assert all("this goal: get paid file sales live" in prompt for prompt, _, _ in seen)
    assert "for: get paid file sales live" in row.body


def test_cli_focus_forces_a_sweep_and_passes_the_goal(kanban_home: Path, tmp_path: Path, capsys, monkeypatch):
    from hermes_cli import kanban as kc

    repo = tmp_path / "repo"
    repo.mkdir()
    seen: list = []
    monkeypatch.setattr("hermes_cli.kanban_discovery.hermes_chat_runner", _runner_for({}, seen))
    with kb.connect_closing() as conn:
        kb.create_task(conn, title="open work")  # a sweep would not be recommended
    assert kc._cmd_orchestrate(_orch_ns(project_root=str(repo), focus="get paid file sales live")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["discovery"]["ran"] is True
    assert payload["discovery"]["goal"] == "get paid file sales live"
    assert seen and all("get paid file sales live" in prompt for prompt, _, _ in seen)


def test_parser_wires_focus():
    from hermes_cli import kanban as kc
    from hermes_cli._parser import build_top_level_parser

    parser, sub, _chat = build_top_level_parser()
    kc.build_parser(sub)
    args = parser.parse_args(["kanban", "orchestrate", "proj", "--focus", "get paid file sales live"])
    assert args.focus == "get paid file sales live"
    assert parser.parse_args(["kanban", "orchestrate", "proj"]).focus is None
