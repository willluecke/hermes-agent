"""Worktree lifecycle: placement, landing, release and the disposition record.

Will, 2026-11-24: the RegWatch board left ~90 task worktrees behind, 46 of
them nested inside a stray ``.worktrees/kanban-main`` checkout the board had
been pointed at, and nobody could tell which held work. These tests pin the
fix:

* task trees always go under the main checkout, never inside a linked one;
* "landed" includes work merged locally and work cherry-picked into main,
  and another task's branch never counts as the place work landed;
* a clean kanban-owned tree with unlanded commits loses only its checkout:
  the branch keeps every commit (``keep`` keeps the checkout too);
* a repo with no remote no longer deletes a task branch's unique commits;
* every settlement is a ``worktree_disposition`` event, ``kanban gc``
  removes only what has landed, and ``hermes kanban worktrees`` reports it
  all read-only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        [
            "git", "-C", str(cwd),
            "-c", "user.name=t", "-c", "user.email=t@example.com",
            "-c", "commit.gpgsign=false",
            *args,
        ],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=60,
    )
    assert result.returncode == 0, f"git {' '.join(args)} failed: {result.stderr}"
    return result.stdout.strip()


def _commit(cwd: Path, name: str, text: str) -> str:
    (cwd / name).write_text(text, encoding="utf-8")
    _git(cwd, "add", name)
    _git(cwd, "commit", "-m", f"add {name}")
    return _git(cwd, "rev-parse", "HEAD")


def _branch_sha(repo: Path, branch: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"],
        capture_output=True, text=True,
    )
    return out.stdout.strip()


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A project with a remote, history fully pushed, main checked out."""
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", "-b", "main", str(origin)], check=True, capture_output=True)
    project = tmp_path / "project"
    subprocess.run(["git", "clone", str(origin), str(project)], check=True, capture_output=True)
    _git(project, "checkout", "-B", "main")
    _commit(project, "README.md", "hello\n")
    _git(project, "push", "origin", "main")
    _git(project, "remote", "set-head", "origin", "main")
    return project


@pytest.fixture
def local_repo(tmp_path: Path) -> Path:
    """A project with no remote at all."""
    project = tmp_path / "local"
    subprocess.run(["git", "init", "-b", "main", str(project)], check=True, capture_output=True)
    _commit(project, "README.md", "hello\n")
    return project


def _task_tree(repo: Path, task_id: str, branch: str | None = None) -> Path:
    target = repo / ".worktrees" / task_id
    kb._ensure_git_worktree(repo, target, branch or f"wt/{task_id}")
    return target


def _worktree_task(conn, repo: Path, title: str = "wt-task") -> tuple[str, Path]:
    tid = kb.create_task(conn, title=title, assignee="worker")
    wt = _task_tree(repo, tid)
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET workspace_kind='worktree', workspace_path=?, "
            "branch_name=?, status='ready' WHERE id=?",
            (str(wt), f"wt/{tid}", tid),
        )
    assert kb.claim_task(conn, tid, claimer="worker") is not None
    return tid, wt


def _dispositions(conn, task_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? ORDER BY id",
        (task_id, kb.WORKTREE_DISPOSITION_EVENT),
    ).fetchall()
    return [json.loads(r["payload"]) for r in rows]


# ---------------------------------------------------------------------------
# Landing and release
# ---------------------------------------------------------------------------


def test_unlanded_commits_release_checkout_and_keep_branch(repo: Path) -> None:
    wt = _task_tree(repo, "t_aaaa0001")
    sha = _commit(wt, "work.txt", "only on the task branch\n")
    outcome = kb._cleanup_worktree_workspace("t_aaaa0001", str(wt))
    assert not wt.exists()
    assert _branch_sha(repo, "wt/t_aaaa0001") == sha
    assert outcome["state"] == "unlanded"
    assert outcome["unlanded_commits"] == 1
    assert outcome["action"] == "released"
    assert outcome["branch_action"] == "kept"


def test_pushed_task_branch_is_preserved_not_landed(repo: Path) -> None:
    """``origin/wt/t_x`` keeps the work safe off-box but is not integration."""
    wt = _task_tree(repo, "t_aaaa0009")
    sha = _commit(wt, "pr.txt", "pushed for a PR\n")
    _git(wt, "push", "origin", "wt/t_aaaa0009")
    state = kb.task_worktree_state(wt, branch="wt/t_aaaa0009")
    assert state["state"] == "unlanded"
    assert state["pushed"] is True
    outcome = kb._cleanup_worktree_workspace("t_aaaa0009", str(wt))
    assert outcome["action"] == "released"
    assert _branch_sha(repo, "wt/t_aaaa0009") == sha


def test_keep_decision_keeps_unlanded_checkout(repo: Path) -> None:
    wt = _task_tree(repo, "t_aaaa0002")
    _commit(wt, "work.txt", "keep me checked out\n")
    outcome = kb._cleanup_worktree_workspace(
        "t_aaaa0002", str(wt), disposition={"decision": "keep", "reason": "preview runs here"},
    )
    assert wt.is_dir()
    assert outcome["action"] == "preserved"
    assert outcome["decision"] == "keep"


def test_cherry_picked_work_counts_as_landed(repo: Path) -> None:
    wt = _task_tree(repo, "t_aaaa0003")
    sha = _commit(wt, "fix.txt", "a fix\n")
    _commit(repo, "other.txt", "main moved on\n")
    _git(repo, "cherry-pick", sha)  # main gets a copy with a new sha
    assert _git(repo, "rev-parse", "HEAD") != sha
    _git(repo, "push", "origin", "main")
    outcome = kb._cleanup_worktree_workspace("t_aaaa0003", str(wt))
    assert not wt.exists()
    assert _branch_sha(repo, "wt/t_aaaa0003") == ""
    assert outcome["state"] == "landed"
    assert outcome["landed_via"] == "patch-equivalent"
    assert outcome["branch_action"] == "deleted"


def test_local_merge_into_main_counts_as_landed(repo: Path) -> None:
    wt = _task_tree(repo, "t_aaaa0004")
    _commit(wt, "feature.txt", "merged locally, not pushed\n")
    _git(repo, "merge", "--no-ff", "-m", "merge task", "wt/t_aaaa0004")
    outcome = kb._cleanup_worktree_workspace("t_aaaa0004", str(wt))
    assert not wt.exists()
    assert _branch_sha(repo, "wt/t_aaaa0004") == ""
    assert outcome["landed_via"] == "reachable"


def test_another_task_branch_never_vouches_for_landing(repo: Path) -> None:
    wt = _task_tree(repo, "t_aaaa0005")
    sha = _commit(wt, "shared.txt", "same commit on two task branches\n")
    _git(repo, "branch", "wt/t_bbbb0005", sha)
    state = kb.task_worktree_state(wt, branch="wt/t_aaaa0005")
    assert state["state"] == "unlanded"
    assert state["unlanded_commits"] == 1


def test_side_branch_copy_preserves_but_does_not_land(repo: Path) -> None:
    """An archive or side branch keeps the commits safe; it is not main."""
    wt = _task_tree(repo, "t_aaaa000a")
    sha = _commit(wt, "side.txt", "copied aside\n")
    _git(repo, "branch", "archive/wt/t_aaaa000a", sha)
    state = kb.task_worktree_state(wt, branch="wt/t_aaaa000a")
    assert state["state"] == "unlanded"
    assert state["contained_in"] == ["archive/wt/t_aaaa000a"]
    _git(repo, "merge", "--no-ff", "-m", "land", "archive/wt/t_aaaa000a")
    landed = kb.task_worktree_state(wt, branch="wt/t_aaaa000a")
    assert landed["state"] == "landed"
    assert landed["landed_in"] == ["origin/HEAD", "main"] or "main" in landed["landed_in"]


def test_own_branch_checked_out_in_main_checkout_does_not_land_itself(local_repo: Path) -> None:
    _git(local_repo, "checkout", "-b", "feature/mine")
    sha = _commit(local_repo, "mine.txt", "on the task's own branch\n")
    _git(local_repo, "branch", "-f", "main", "HEAD~1")
    state = kb.task_worktree_state(None, branch="feature/mine", repo=local_repo)
    assert state["head"] == sha
    assert state["state"] == "unlanded"


def test_no_remote_repo_keeps_unique_task_branch(local_repo: Path) -> None:
    """Before: with no remote refs every commit counted as pushed, so
    completion force-deleted the wt/ branch and its only copy of the work."""
    wt = _task_tree(local_repo, "t_aaaa0006")
    sha = _commit(wt, "work.txt", "the only copy\n")
    kb._cleanup_worktree_workspace("t_aaaa0006", str(wt))
    assert _branch_sha(local_repo, "wt/t_aaaa0006") == sha


def test_dirty_tree_preserved_whatever_the_decision(repo: Path) -> None:
    wt = _task_tree(repo, "t_aaaa0007")
    (wt / "scratch.log").write_text("untracked\n", encoding="utf-8")
    outcome = kb._cleanup_worktree_workspace(
        "t_aaaa0007", str(wt), disposition={"decision": "discard", "reason": "junk"},
    )
    assert wt.is_dir()
    assert (wt / "scratch.log").exists()
    assert outcome["state"] == "dirty"
    assert outcome["dirty_files"] == 1
    assert outcome["action"] == "preserved"


def test_user_named_worktree_with_unlanded_work_is_preserved(repo: Path, tmp_path: Path) -> None:
    named = tmp_path / "project-feature"
    _git(repo, "worktree", "add", "-b", "feature/x", str(named), "HEAD")
    _commit(named, "x.txt", "someone's lane\n")
    outcome = kb._cleanup_worktree_workspace("t_aaaa0008", str(named), "feature/x")
    assert named.is_dir()
    assert outcome["action"] == "preserved"


# ---------------------------------------------------------------------------
# Completion records the disposition
# ---------------------------------------------------------------------------


def test_complete_task_records_disposition_event(kanban_home: Path, repo: Path) -> None:
    with kb.connect_closing() as conn:
        tid, wt = _worktree_task(conn, repo)
        sha = _commit(wt, "work.txt", "waiting for a merge\n")
        assert kb.complete_task(
            conn, tid, summary="done",
            metadata={"worktree": {"decision": "merge", "reason": "parent integrates", "into": "main"}},
        )
        events = _dispositions(conn, tid)
    assert not wt.exists()
    assert _branch_sha(repo, f"wt/{tid}") == sha
    assert len(events) == 1
    ev = events[0]
    assert ev["branch"] == f"wt/{tid}"
    assert ev["head"] == sha
    assert ev["state"] == "unlanded"
    assert ev["unlanded_commits"] == 1
    assert ev["dirty_files"] == 0
    assert ev["action"] == "released"
    assert ev["branch_action"] == "kept"
    assert ev["decision"] == "merge"
    assert ev["reason"] == "parent integrates"
    assert ev["into"] == "main"


def test_archive_records_disposition_for_landed_tree(kanban_home: Path, repo: Path) -> None:
    with kb.connect_closing() as conn:
        tid, wt = _worktree_task(conn, repo)
        assert kb.archive_task(conn, tid)
        events = _dispositions(conn, tid)
    assert not wt.exists()
    assert events and events[0]["state"] == "landed"
    assert events[0]["action"] == "removed"


# ---------------------------------------------------------------------------
# Placement: never nest inside a linked worktree
# ---------------------------------------------------------------------------


def _linked_anchor(repo: Path) -> tuple[Path, str]:
    """A linked worktree on its own branch, one commit ahead of main."""
    anchor = repo / ".worktrees" / "kanban-main"
    _git(repo, "worktree", "add", "-b", "base/pinned", str(anchor), "HEAD")
    return anchor, _commit(anchor, "base.txt", "the base tasks should start from\n")


def test_linked_worktree_board_anchor_places_tree_under_main_checkout(
    kanban_home: Path, repo: Path
) -> None:
    anchor, anchor_head = _linked_anchor(repo)
    kb.write_board_metadata(kb.get_current_board(), default_workdir=str(anchor))
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="t", workspace_kind="worktree")
        task = kb.get_task(conn, tid)
    workspace, branch = kb._resolve_worktree_workspace(task)
    assert workspace == repo.resolve() / ".worktrees" / tid
    assert not (anchor / ".worktrees").exists()
    assert branch == f"wt/{tid}"
    assert _git(workspace, "rev-parse", "HEAD") == anchor_head


def test_pathless_task_on_linked_anchor_is_not_nested(kanban_home: Path, repo: Path) -> None:
    """Decompose children carry no path and resolve from the board anchor:
    this is the route that put 46 RegWatch trees inside kanban-main."""
    anchor, anchor_head = _linked_anchor(repo)
    kb.write_board_metadata(kb.get_current_board(), default_workdir=str(anchor))
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="child", workspace_kind="worktree")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET workspace_path = NULL WHERE id = ?", (tid,))
        task = kb.get_task(conn, tid)
    workspace, _branch = kb._resolve_worktree_workspace(task)
    assert workspace == repo.resolve() / ".worktrees" / tid
    assert not (anchor / ".worktrees").exists()
    assert _git(workspace, "rev-parse", "HEAD") == anchor_head


def test_derived_path_inside_linked_worktree_is_moved_to_main_checkout(
    kanban_home: Path, repo: Path
) -> None:
    anchor, anchor_head = _linked_anchor(repo)
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="t", workspace_kind="worktree")
        # e.g. a project whose primary path is the linked checkout
        kb.set_workspace_path(conn, tid, str(anchor / ".worktrees" / tid))
        task = kb.get_task(conn, tid)
    workspace, _branch = kb._resolve_worktree_workspace(task)
    assert workspace == repo.resolve() / ".worktrees" / tid
    assert not (anchor / ".worktrees" / tid).exists()
    assert _git(workspace, "rev-parse", "HEAD") == anchor_head


def test_main_checkout_anchor_unchanged(kanban_home: Path, repo: Path) -> None:
    kb.write_board_metadata(kb.get_current_board(), default_workdir=str(repo))
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="t", workspace_kind="worktree")
        task = kb.get_task(conn, tid)
    workspace, _branch = kb._resolve_worktree_workspace(task)
    assert workspace == repo.resolve() / ".worktrees" / tid
    assert _git(workspace, "rev-parse", "HEAD") == _git(repo, "rev-parse", "HEAD")


# ---------------------------------------------------------------------------
# gc and the audit
# ---------------------------------------------------------------------------


def _run_kanban(action: str, **kwargs) -> None:
    from hermes_cli import kanban as kcli

    handler = {"gc": kcli._cmd_gc, "worktrees": kcli._cmd_worktrees}[action]
    defaults = {"gc": {"event_retention_days": 30, "log_retention_days": 30}, "worktrees": {}}
    assert handler(argparse.Namespace(**{**defaults[action], **kwargs})) == 0


def test_gc_deletes_branch_only_after_it_lands(kanban_home: Path, repo: Path) -> None:
    with kb.connect_closing() as conn:
        tid, wt = _worktree_task(conn, repo)
        _commit(wt, "work.txt", "lands later\n")
        assert kb.complete_task(conn, tid, summary="done")
    assert not wt.exists()  # released at completion
    _run_kanban("gc")
    assert _branch_sha(repo, f"wt/{tid}")  # unlanded: gc leaves it alone
    _git(repo, "merge", "--no-ff", "-m", "land it", f"wt/{tid}")
    _run_kanban("gc")
    assert _branch_sha(repo, f"wt/{tid}") == ""
    with kb.connect_closing() as conn:
        last = _dispositions(conn, tid)[-1]
    assert last["branch_action"] == "deleted"
    assert last["source"] == "gc"


def test_gc_never_releases_unlanded_checkouts(kanban_home: Path, repo: Path) -> None:
    """A pre-existing done task's checkout (the RegWatch backlog) is only
    reported, never removed, while its work is unlanded."""
    with kb.connect_closing() as conn:
        tid, wt = _worktree_task(conn, repo)
        _commit(wt, "work.txt", "legacy unlanded work\n")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='done' WHERE id=?", (tid,))
    _run_kanban("gc")
    assert wt.is_dir()


def test_audit_reports_states_decisions_and_orphans(
    kanban_home: Path, repo: Path, capsys: pytest.CaptureFixture
) -> None:
    kb.write_board_metadata(kb.get_current_board(), default_workdir=str(repo))
    with kb.connect_closing() as conn:
        released, released_wt = _worktree_task(conn, repo, title="released")
        _commit(released_wt, "a.txt", "a\n")
        assert kb.complete_task(
            conn, released, summary="done",
            metadata={"worktree": {"decision": "merge", "reason": "needs review"}},
        )
        dirty, dirty_wt = _worktree_task(conn, repo, title="dirty")
        (dirty_wt / "wip.txt").write_text("wip\n", encoding="utf-8")
        landed, _landed_wt = _worktree_task(conn, repo, title="landed")
        assert kb.complete_task(conn, landed, summary="nothing to land")
    orphan = repo / ".worktrees" / "t_0ff1ce00"
    _git(repo, "worktree", "add", "-b", "wt/t_0ff1ce00", str(orphan), "HEAD")

    with kb.connect_closing() as conn:
        report = kb.audit_task_worktrees(conn)
    rows = {r["task_id"]: r for r in report["tasks"]}
    assert landed not in rows  # checkout and branch both gone
    assert rows[released]["state"] == "unlanded"
    assert rows[released]["checkout"] is False
    assert rows[released]["decision"] == "merge"
    assert rows[released]["reason"] == "needs review"
    assert rows[dirty]["state"] == "dirty"
    assert rows[dirty]["checkout"] is True
    assert rows[dirty]["nested"] is False
    assert [o["path"] for o in report["orphans"]] == [str(orphan)]

    with kb.connect_closing() as conn:
        everything = kb.audit_task_worktrees(conn, include_settled=True)
    assert landed in {r["task_id"] for r in everything["tasks"]}

    _run_kanban("worktrees", json=True)
    printed = json.loads(capsys.readouterr().out)
    assert {r["task_id"] for r in printed["tasks"]} == {released, dirty}
    _run_kanban("worktrees")
    text = capsys.readouterr().out
    assert "unlanded:1" in text and "dirty:1" in text and "t_0ff1ce00" in text
    assert "reason: needs review" in text
