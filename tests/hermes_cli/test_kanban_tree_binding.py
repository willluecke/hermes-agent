"""Tree binding for kanban worktree runs.

Every run that works in a git worktree records which commit it started from
(``base_sha``, stamped by the dispatcher once the worktree exists) and what it
changed (``head_sha`` + ``changed_paths``, stamped when the run closes through
``complete_task`` or ``request_review``), all on ``task_runs.metadata``, with a
``tree_bound`` event for the timeline. Git trouble records ``tree_error`` and
never blocks the transition; ``dir:`` and ``scratch`` workspaces record nothing.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(cwd), "-c", "user.name=Tree Test", "-c", "user.email=tree@example.com",
         "-c", "commit.gpgsign=false", *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
    )
    assert result.returncode == 0, f"git {' '.join(args)} failed: {result.stderr}"
    return result.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    subprocess.run(["git", "init", "-b", "main", str(project)], check=True, capture_output=True, text=True)
    (project / "README.md").write_text("base\n", encoding="utf-8")
    _git(project, "add", "README.md")
    _git(project, "commit", "-m", "init")
    return project


def _run_metadata(conn, tid: str) -> dict | None:
    row = conn.execute(
        "SELECT metadata FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1", (tid,),
    ).fetchone()
    assert row is not None
    return json.loads(row["metadata"]) if row["metadata"] else None


def _events(conn, tid: str, kind: str) -> list[dict | None]:
    rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? ORDER BY id", (tid, kind),
    ).fetchall()
    return [json.loads(r["payload"]) if r["payload"] else None for r in rows]


def _dispatch_worktree_task(conn, repo: Path, title: str = "bind me") -> tuple[str, Path, int]:
    """Create a worktree task anchored on ``repo`` and dispatch it with a stub spawn."""
    tid = kb.create_task(conn, title=title, assignee="alice", workspace_kind="worktree", workspace_path=str(repo))
    res = kb.dispatch_once(conn, spawn_fn=lambda task, workspace, board=None: 4242)
    assert [s[0] for s in res.spawned] == [tid], res
    task = kb.get_task(conn, tid)
    assert task is not None and task.status == "running" and task.current_run_id
    workspace = Path(task.workspace_path)
    assert workspace == repo / ".worktrees" / tid and workspace.is_dir()
    return tid, workspace, int(task.current_run_id)


def test_dispatch_records_base_sha_on_the_worktree_run(kanban_home, repo, all_assignees_spawnable):
    with kb.connect() as conn:
        tid, workspace, run_id = _dispatch_worktree_task(conn, repo)
        assert _run_metadata(conn, tid) == {"base_sha": _git(repo, "rev-parse", "HEAD")}
        assert _git(workspace, "rev-parse", "HEAD") == _git(repo, "rev-parse", "HEAD")
        assert _events(conn, tid, kb.TREE_BOUND_EVENT) == []       # bound only when the run closes


def test_complete_task_binds_head_sha_and_changed_paths(kanban_home, repo, all_assignees_spawnable):
    with kb.connect() as conn:
        tid, workspace, run_id = _dispatch_worktree_task(conn, repo)
        base_sha = _run_metadata(conn, tid)["base_sha"]
        (workspace / "feature.py").write_text("print('hi')\n", encoding="utf-8")
        (workspace / "README.md").write_text("base\nmore\n", encoding="utf-8")
        _git(workspace, "add", "feature.py", "README.md")
        _git(workspace, "commit", "-m", "feature")
        (workspace / "notes.txt").write_text("uncommitted\n", encoding="utf-8")
        head_sha = _git(workspace, "rev-parse", "HEAD")
        assert head_sha != base_sha

        assert kb.complete_task(conn, tid, result="done", metadata={"tests_run": ["pytest"]}, expected_run_id=run_id)

        meta = _run_metadata(conn, tid)
        assert meta == {"base_sha": base_sha, "head_sha": head_sha, "changed_paths": ["README.md", "feature.py", "notes.txt"],
                        "tests_run": ["pytest"]}                   # worker metadata merged, not replaced
        assert _events(conn, tid, kb.TREE_BOUND_EVENT) == [
            {"base_sha": base_sha, "head_sha": head_sha, "changed_paths": ["README.md", "feature.py", "notes.txt"]}]
        assert kb.get_task(conn, tid).status == "done"
        run = kb.get_run(conn, run_id)
        assert run.outcome == "completed" and run.metadata == meta


def test_request_review_binds_the_tree_too(kanban_home, repo, all_assignees_spawnable):
    with kb.connect() as conn:
        tid, workspace, run_id = _dispatch_worktree_task(conn, repo)
        base_sha = _run_metadata(conn, tid)["base_sha"]
        (workspace / "impl.py").write_text("x = 1\n", encoding="utf-8")
        _git(workspace, "add", "impl.py")
        _git(workspace, "commit", "-m", "impl")
        head_sha = _git(workspace, "rev-parse", "HEAD")

        assert kb.request_review(conn, tid, summary="please review", metadata=None, expected_run_id=run_id)

        assert _run_metadata(conn, tid) == {"base_sha": base_sha, "head_sha": head_sha, "changed_paths": ["impl.py"]}
        assert _events(conn, tid, kb.TREE_BOUND_EVENT) == [{"base_sha": base_sha, "head_sha": head_sha, "changed_paths": ["impl.py"]}]
        assert kb.get_task(conn, tid).status == "review"


def test_unchanged_worktree_binds_with_no_changed_paths(kanban_home, repo, all_assignees_spawnable):
    with kb.connect() as conn:
        tid, workspace, run_id = _dispatch_worktree_task(conn, repo)
        sha = _run_metadata(conn, tid)["base_sha"]
        assert kb.complete_task(conn, tid, summary="nothing to do", expected_run_id=run_id)
        assert _run_metadata(conn, tid) == {"base_sha": sha, "head_sha": sha, "changed_paths": []}


def test_dir_and_scratch_workspaces_record_nothing(kanban_home, tmp_path, all_assignees_spawnable):
    plain = tmp_path / "plain"
    with kb.connect() as conn:
        dir_task = kb.create_task(conn, title="dir", assignee="alice", workspace_kind="dir", workspace_path=str(plain))
        scratch_task = kb.create_task(conn, title="scratch", assignee="alice", workspace_kind="scratch")
        res = kb.dispatch_once(conn, spawn_fn=lambda task, workspace, board=None: 4242)
        assert {s[0] for s in res.spawned} == {dir_task, scratch_task}
        for tid in (dir_task, scratch_task):
            assert _run_metadata(conn, tid) is None
            run_id = kb.get_task(conn, tid).current_run_id
            assert kb.complete_task(conn, tid, result="ok", metadata={"k": "v"}, expected_run_id=run_id)
            assert _run_metadata(conn, tid) == {"k": "v"}
            assert _events(conn, tid, kb.TREE_BOUND_EVENT) == []


def test_git_failure_records_tree_error_without_blocking_completion(kanban_home, repo, all_assignees_spawnable):
    with kb.connect() as conn:
        tid, workspace, run_id = _dispatch_worktree_task(conn, repo)
        bogus = "0" * 40
        with kb.write_txn(conn):
            conn.execute("UPDATE task_runs SET metadata = ? WHERE id = ?", (json.dumps({"base_sha": bogus}), run_id))
        (workspace / "wip.txt").write_text("wip\n", encoding="utf-8")

        assert kb.complete_task(conn, tid, result="done", metadata={"tests_run": []}, expected_run_id=run_id)

        meta = _run_metadata(conn, tid)
        assert meta["base_sha"] == bogus and meta["tests_run"] == []
        assert "head_sha" not in meta and "changed_paths" not in meta
        assert meta["tree_error"].startswith("git diff --name-only")
        assert _events(conn, tid, kb.TREE_BOUND_EVENT) == []
        assert kb.get_task(conn, tid).status == "done"


def test_missing_worktree_records_tree_error(kanban_home, repo, all_assignees_spawnable):
    import shutil
    with kb.connect() as conn:
        tid, workspace, run_id = _dispatch_worktree_task(conn, repo)
        shutil.rmtree(workspace)
        assert kb.request_review(conn, tid, summary="gone", expected_run_id=run_id)
        meta = _run_metadata(conn, tid)
        assert set(meta) == {"base_sha", "tree_error"} and meta["tree_error"].startswith("workspace missing")


def test_record_run_base_sha_outside_the_dispatcher(kanban_home, repo, tmp_path):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="manual", workspace_kind="worktree", workspace_path=str(repo))
        assert kb.record_run_base_sha(conn, tid, repo, "worktree") is None         # no current run yet
        assert kb.claim_task(conn, tid) is not None
        assert kb.record_run_base_sha(conn, tid, repo, "dir") is None              # not a worktree kind
        assert _run_metadata(conn, tid) is None
        sha = kb.record_run_base_sha(conn, tid, repo, "worktree")
        assert sha == _git(repo, "rev-parse", "HEAD") and _run_metadata(conn, tid) == {"base_sha": sha}
        assert kb.record_run_base_sha(conn, tid, tmp_path / "nowhere", "worktree") is None
        assert set(_run_metadata(conn, tid)) == {"base_sha", "tree_error"}         # merged, base_sha kept
