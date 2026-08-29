from __future__ import annotations

import os
from pathlib import Path

import pytest

from tools.workspace_snapshots import (
    WorkspaceSnapshotConflict,
    WorkspaceSnapshotError,
    WorkspaceSnapshotPolicy,
    capture_workspace_snapshot,
    list_workspace_snapshots,
    materialize_workspace_snapshot,
)


@pytest.fixture
def snapshot_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    hermes_home = tmp_path / "hermes-home"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    policy = WorkspaceSnapshotPolicy(
        enabled=True,
        keep_snapshots=3,
        timeout_seconds=30,
    )
    return hermes_home, workspace, policy


def _capture(workspace: Path, policy: WorkspaceSnapshotPolicy, task: str):
    return capture_workspace_snapshot(
        workspace_root=str(workspace),
        project="reg-watch",
        session_id="session-1",
        task_id=task,
        policy=policy,
    )


def test_incremental_snapshot_preserves_deleted_and_changed_files(snapshot_env):
    _home, workspace, policy = snapshot_env
    unchanged = workspace / "unchanged.txt"
    removed = workspace / "removed.txt"
    changed = workspace / "changed.txt"
    unchanged.write_text("same\n", encoding="utf-8")
    removed.write_text("recover me\n", encoding="utf-8")
    changed.write_text("before\n", encoding="utf-8")

    first = _capture(workspace, policy, "turn-1")
    assert first is not None
    removed.unlink()
    changed.write_text("after\n", encoding="utf-8")
    second = _capture(workspace, policy, "turn-2")
    assert second is not None

    first_payload = Path(first.path) / "payload"
    second_payload = Path(second.path) / "payload"
    assert (first_payload / "removed.txt").read_text(encoding="utf-8") == "recover me\n"
    assert (first_payload / "changed.txt").read_text(encoding="utf-8") == "before\n"
    assert not (second_payload / "removed.txt").exists()
    assert (second_payload / "changed.txt").read_text(encoding="utf-8") == "after\n"
    assert os.stat(first_payload / "unchanged.txt").st_ino == os.stat(
        second_payload / "unchanged.txt"
    ).st_ino


def test_materialize_restores_to_new_destination_and_refuses_overwrite(snapshot_env):
    _home, workspace, policy = snapshot_env
    (workspace / "evidence.json").write_text('{"done": false}\n', encoding="utf-8")
    snapshot = _capture(workspace, policy, "turn-1")
    assert snapshot is not None

    destination = workspace.parent / "recovered"
    result = materialize_workspace_snapshot(
        snapshot.snapshot_id,
        project="reg-watch",
        destination=str(destination),
        timeout_seconds=30,
    )
    assert result == str(destination)
    assert (destination / "evidence.json").read_text(encoding="utf-8") == '{"done": false}\n'
    with pytest.raises(WorkspaceSnapshotConflict, match="occupied"):
        materialize_workspace_snapshot(
            snapshot.snapshot_id,
            project="reg-watch",
            destination=str(destination),
        )


def test_retention_keeps_only_the_newest_complete_snapshots(snapshot_env):
    _home, workspace, _policy = snapshot_env
    policy = WorkspaceSnapshotPolicy(enabled=True, keep_snapshots=2, timeout_seconds=30)
    for index in range(3):
        (workspace / "value.txt").write_text(str(index), encoding="utf-8")
        _capture(workspace, policy, f"turn-{index}")

    snapshots = list_workspace_snapshots("reg-watch")
    assert len(snapshots) == 2
    assert snapshots[0]["task_id"] == "turn-2"
    assert snapshots[1]["task_id"] == "turn-1"


def test_snapshot_failure_is_explicit_and_leaves_no_partial(snapshot_env, monkeypatch):
    home, workspace, policy = snapshot_env
    (workspace / "file.txt").write_text("data", encoding="utf-8")
    monkeypatch.setattr("tools.workspace_snapshots.shutil.which", lambda _name: None)

    with pytest.raises(WorkspaceSnapshotError, match="rsync is required"):
        _capture(workspace, policy, "turn-1")

    store = home / "workspace-snapshots"
    pending = (
        [path for path in store.rglob("*") if path.name.endswith(".pending")]
        if store.exists()
        else []
    )
    assert pending == []


def test_disabled_policy_has_no_filesystem_effect(snapshot_env):
    home, workspace, _policy = snapshot_env
    assert capture_workspace_snapshot(
        workspace_root=str(workspace),
        project="reg-watch",
        session_id="session-1",
        task_id="turn-1",
        policy=WorkspaceSnapshotPolicy(enabled=False),
    ) is None
    assert not (home / "workspace-snapshots").exists()


def test_vanished_source_files_are_a_warning_not_a_failure(
    snapshot_env, monkeypatch
):
    """rsync exit 24 (files vanished mid-transfer) must not abort the turn.

    Workspaces legitimately contain transient files (run media, temp dirs)
    that can disappear between rsync's directory scan and the copy. The
    snapshot is still a valid recovery point for everything that existed.
    """
    import subprocess as _subprocess

    home, workspace, policy = snapshot_env
    (workspace / "file.txt").write_text("data", encoding="utf-8")

    real_run = _subprocess.run
    from types import SimpleNamespace

    def fake_run(command, **kwargs):
        assert command[0].endswith("rsync") or "rsync" in command[0]
        return real_run(command, **kwargs)

    def fake_rsync(command, **kwargs):
        # Simulate exit 24: some files vanished before transfer.
        return SimpleNamespace(returncode=24, stderr="", stdout="")

    monkeypatch.setattr(
        "tools.workspace_snapshots.shutil.which", lambda _name: "/usr/bin/rsync"
    )
    monkeypatch.setattr(
        "tools.workspace_snapshots.subprocess.run", fake_rsync
    )

    snapshot = _capture(workspace, policy, "turn-vanished")
    assert snapshot is not None, "exit 24 must not fail the snapshot"

    # The manifest must exist and the snapshot dir must be complete (not
    # left pending) — the recovery point is usable.
    record = Path(snapshot.path)
    assert record.is_dir()
    assert (record / "manifest.json").is_file()
