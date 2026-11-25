"""kanban_complete asks a builder what should happen to an unlanded branch.

Without a recorded decision a finished task's branch is just a name; that is
how the RegWatch board reached ~90 worktrees nobody could safely remove
(2026-11-24). The gate fires only when the tree holds work that has not
landed, and never for the review lane, whose reviewer does not commit.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        [
            "git", "-C", str(cwd),
            "-c", "user.name=t", "-c", "user.email=t@example.com",
            "-c", "commit.gpgsign=false",
            *args,
        ],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.fixture
def worktree_worker(monkeypatch, tmp_path):
    """A running worker whose task has a real kanban-owned git worktree."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True)
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "init")

    from hermes_cli import kanban_db as kb

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="wt", assignee="test-worker")
        wt = repo / ".worktrees" / tid
        kb._ensure_git_worktree(repo, wt, f"wt/{tid}")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET workspace_kind='worktree', workspace_path=?, branch_name=? WHERE id=?",
                (str(wt), f"wt/{tid}", tid),
            )
        assert kb.claim_task(conn, tid) is not None
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    return {"tid": tid, "repo": repo, "wt": wt}


def _commit_work(wt: Path) -> str:
    (wt / "work.txt").write_text("unlanded\n", encoding="utf-8")
    _git(wt, "add", "work.txt")
    _git(wt, "commit", "-m", "work")
    return _git(wt, "rev-parse", "HEAD")


def _status(tid: str) -> str:
    from hermes_cli import kanban_db as kb

    with kb.connect_closing() as conn:
        return kb.get_task(conn, tid).status


def test_unlanded_commits_without_disposition_are_refused(worktree_worker):
    from tools import kanban_tools as kt

    _commit_work(worktree_worker["wt"])
    out = json.loads(kt._handle_complete({"summary": "done"}))
    assert "error" in out
    msg = out["error"]
    assert "1 commit(s)" in msg
    assert 'worktree={"decision": "merge"' in msg
    assert "still in flight" in msg
    assert _status(worktree_worker["tid"]) == "running"
    assert worktree_worker["wt"].is_dir()


def test_retry_with_disposition_completes_and_records_it(worktree_worker):
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    tid, wt, repo = worktree_worker["tid"], worktree_worker["wt"], worktree_worker["repo"]
    sha = _commit_work(wt)
    out = json.loads(kt._handle_complete({
        "summary": "done",
        "worktree": {"decision": "merge", "reason": "parent card integrates", "into": "main"},
    }))
    assert out.get("ok") is True, out
    assert _status(tid) == "done"
    with kb.connect_closing() as conn:
        run = kb.latest_run(conn, tid)
        events = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind=?",
            (tid, kb.WORKTREE_DISPOSITION_EVENT),
        ).fetchall()
    assert run.metadata["worktree"] == {
        "decision": "merge", "reason": "parent card integrates", "into": "main",
    }
    payload = json.loads(events[-1]["payload"])
    assert payload["decision"] == "merge"
    assert payload["action"] == "released"
    assert not wt.exists()
    assert _git(repo, "rev-parse", f"wt/{tid}") == sha


def test_invalid_disposition_is_refused(worktree_worker):
    from tools import kanban_tools as kt

    _commit_work(worktree_worker["wt"])
    out = json.loads(kt._handle_complete({
        "summary": "done", "worktree": {"decision": "ship-it", "reason": "x"},
    }))
    assert "error" in out and "decision one of" in out["error"]
    out = json.loads(kt._handle_complete({
        "summary": "done", "worktree": {"decision": "keep", "reason": "  "},
    }))
    assert "error" in out
    assert _status(worktree_worker["tid"]) == "running"


def test_uncommitted_files_are_named(worktree_worker):
    from tools import kanban_tools as kt

    (worktree_worker["wt"] / "notes.md").write_text("forgot to commit\n", encoding="utf-8")
    out = json.loads(kt._handle_complete({"summary": "done"}))
    assert "error" in out
    assert "1 uncommitted file(s) (notes.md)" in out["error"]


def test_landed_tree_needs_no_disposition(worktree_worker):
    from tools import kanban_tools as kt

    out = json.loads(kt._handle_complete({"summary": "research only, no commits"}))
    assert out.get("ok") is True, out
    assert not worktree_worker["wt"].exists()


def test_review_lane_completion_is_not_gated(worktree_worker):
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    tid = worktree_worker["tid"]
    _commit_work(worktree_worker["wt"])
    with kb.connect_closing() as conn:
        assert kb.request_review(conn, tid, summary="built it", force=True)
        assert kb.claim_review_task(conn, tid) is not None
    out = json.loads(kt._handle_complete({"summary": "approved"}))
    assert out.get("ok") is True, out
    assert _status(tid) == "done"


def test_schema_documents_worktree_field():
    from tools import kanban_tools as kt

    prop = kt.KANBAN_COMPLETE_SCHEMA["parameters"]["properties"]["worktree"]
    assert prop["properties"]["decision"]["enum"] == ["merge", "keep", "discard"]
    assert prop["required"] == ["decision", "reason"]
    assert "worktree" in kt.KANBAN_COMPLETE_SCHEMA["description"]
