from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from tools.reversible_deletion import (
    ReversibleDeletionPolicy,
    RestoreConflict,
    TrashQuotaExceeded,
    UnsafeDeleteTarget,
    capture_delete_command,
    capture_file_change_paths,
    list_trash_items,
    looks_like_file_delete,
    operation_key_for,
    parse_delete_command,
    purge_trash_item,
    restore_trash_item,
)


@pytest.fixture
def trash_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / "hermes-home"
    workspace = tmp_path / "workspace"
    temp_root = tmp_path / "temp"
    workspace.mkdir()
    temp_root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    policy = ReversibleDeletionPolicy(
        enabled=True,
        max_item_bytes=1024 * 1024,
        max_total_bytes=4 * 1024 * 1024,
        max_entries=100,
        temp_roots=(str(temp_root),),
    )
    return home, workspace, temp_root, policy


def test_static_rm_is_parsed_inside_workspace(trash_env):
    _home, workspace, _temp_root, policy = trash_env
    target = workspace / "build output"
    plan = parse_delete_command(
        f"/bin/bash -lc 'rm -rf \"{target}\"'",
        cwd=str(workspace),
        workspace_root=str(workspace),
        policy=policy,
    )
    assert plan.targets == (str(target),)


@pytest.mark.parametrize(
    "command",
    [
        "rm old.txt",
        "unlink old.txt",
        "python -c 'import shutil; shutil.rmtree(\"old\")'",
        "node -e 'fs.rm(\"old\", {recursive: true})'",
        "find . -type f -delete",
        "find build -exec rm -rf {} +",
        "find . -print0 | xargs -0 rm",
        "git clean -fdx",
        "git reset --hard HEAD",
        "rsync -a --delete source/ destination/",
        "shred -u secret.txt",
        "truncate -s 0 evidence.json",
        "dd if=/dev/zero of=evidence.bin bs=1 count=1",
        "make clean",
        "npm run clean",
        "ruby -e 'FileUtils.rm_rf(\"build\")'",
        "perl -e 'unlink \"old.txt\"'",
        "pwsh -c 'Remove-Item -Recurse build'",
    ],
)
def test_file_delete_detection_covers_plain_and_opaque_forms(command: str):
    assert looks_like_file_delete(command) is True


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf .",
        "rm -rf ../outside",
        "rm -rf '$TARGET'",
        "rm -rf *.log",
        "find . -delete",
        "rm -rf one && rm -rf two",
    ],
)
def test_dynamic_or_out_of_boundary_delete_is_not_auto_trashed(
    trash_env, command: str
):
    _home, workspace, _temp_root, policy = trash_env
    with pytest.raises(UnsafeDeleteTarget):
        parse_delete_command(
            command,
            cwd=str(workspace),
            workspace_root=str(workspace),
            policy=policy,
        )


def test_temp_descendant_is_allowed_but_temp_root_is_not(trash_env):
    _home, workspace, temp_root, policy = trash_env
    target = temp_root / "scratch"
    plan = parse_delete_command(
        f"rm -rf {target}",
        cwd=str(workspace),
        workspace_root=str(workspace),
        policy=policy,
    )
    assert plan.targets == (str(target),)
    with pytest.raises(UnsafeDeleteTarget):
        parse_delete_command(
            f"rm -rf {temp_root}",
            cwd=str(workspace),
            workspace_root=str(workspace),
            policy=policy,
        )


def test_capture_precedes_delete_and_restore_refuses_overwrite(trash_env):
    home, workspace, _temp_root, policy = trash_env
    target = workspace / "generated"
    target.mkdir()
    (target / "report.txt").write_text("authoritative\n", encoding="utf-8")
    result = capture_delete_command(
        "rm -rf generated",
        cwd=str(workspace),
        workspace_root=str(workspace),
        project="reg-watch",
        run_id="run-1",
        operation_key=operation_key_for("run-1", "command-1"),
        policy=policy,
    )
    assert result.handled is True
    assert len(result.items) == 1
    assert target.exists(), "capture must not remove the source"
    item = result.items[0]
    payload = home / "trash" / "payloads" / item.item_id / "payload"
    assert (payload / "report.txt").read_text(encoding="utf-8") == "authoritative\n"

    with pytest.raises(RestoreConflict, match="occupied"):
        restore_trash_item(
            item.item_id,
            project="reg-watch",
            workspace_root=str(workspace),
            policy=policy,
        )

    shutil.rmtree(target)
    restored = restore_trash_item(
        item.item_id,
        project="reg-watch",
        workspace_root=str(workspace),
        policy=policy,
    )
    assert restored["status"] == "restored"
    assert (target / "report.txt").read_text(encoding="utf-8") == "authoritative\n"


def test_symlink_is_captured_without_following_target(trash_env):
    home, workspace, _temp_root, policy = trash_env
    outside = workspace.parent / "outside.txt"
    outside.write_text("do not copy", encoding="utf-8")
    target = workspace / "outside-link"
    target.symlink_to(outside)
    result = capture_delete_command(
        "unlink outside-link",
        cwd=str(workspace),
        workspace_root=str(workspace),
        project="reg-watch",
        run_id="run-2",
        operation_key="op-symlink",
        policy=policy,
    )
    payload = home / "trash" / "payloads" / result.items[0].item_id / "payload"
    assert payload.is_symlink()
    assert os.readlink(payload) == str(outside)


def test_capture_is_idempotent_for_operation_and_path(trash_env):
    _home, workspace, _temp_root, policy = trash_env
    target = workspace / "cache.txt"
    target.write_text("one", encoding="utf-8")
    kwargs = dict(
        command="rm cache.txt",
        cwd=str(workspace),
        workspace_root=str(workspace),
        project="reg-watch",
        run_id="run-3",
        operation_key="same-op",
        policy=policy,
    )
    first = capture_delete_command(**kwargs)
    second = capture_delete_command(**kwargs)
    assert first.items[0].item_id == second.items[0].item_id
    assert len(list_trash_items("reg-watch")) == 1


def test_missing_force_cleanup_is_a_safe_handled_noop(trash_env):
    _home, workspace, _temp_root, policy = trash_env
    result = capture_delete_command(
        "rm -rf missing",
        cwd=str(workspace),
        workspace_root=str(workspace),
        project="reg-watch",
        run_id="run-4",
        operation_key="missing-op",
        policy=policy,
    )
    assert result.handled is True
    assert result.items == ()
    assert result.missing_targets == (str(workspace / "missing"),)


def test_file_change_delete_paths_use_same_store(trash_env):
    _home, workspace, _temp_root, policy = trash_env
    target = workspace / "old.ts"
    target.write_text("old", encoding="utf-8")
    result = capture_file_change_paths(
        ["old.ts"],
        cwd=str(workspace),
        workspace_root=str(workspace),
        project="reg-watch",
        run_id="run-5",
        operation_key="file-change-1",
        policy=policy,
    )
    assert result.handled is True
    assert result.items[0].original_path == str(target)


def test_item_quota_fails_before_authorizing_delete(trash_env):
    _home, workspace, _temp_root, policy = trash_env
    tiny_policy = ReversibleDeletionPolicy(
        enabled=True,
        max_item_bytes=3,
        max_total_bytes=10,
        max_entries=10,
        temp_roots=policy.temp_roots,
    )
    (workspace / "large.txt").write_text("large", encoding="utf-8")
    result = capture_delete_command(
        "rm large.txt",
        cwd=str(workspace),
        workspace_root=str(workspace),
        project="reg-watch",
        run_id="run-6",
        operation_key="quota-op",
        policy=tiny_policy,
    )
    assert result.handled is False
    assert "byte limit" in (result.reason or "")
    assert (workspace / "large.txt").exists()


def test_purge_removes_payload_but_keeps_tombstone(trash_env):
    home, workspace, _temp_root, policy = trash_env
    target = workspace / "obsolete.txt"
    target.write_text("obsolete", encoding="utf-8")
    result = capture_delete_command(
        "rm obsolete.txt",
        cwd=str(workspace),
        workspace_root=str(workspace),
        project="reg-watch",
        run_id="run-7",
        operation_key="purge-op",
        policy=policy,
    )
    item = result.items[0]
    payload_dir = home / "trash" / "payloads" / item.item_id
    assert payload_dir.exists()
    purged = purge_trash_item(item.item_id, project="reg-watch")
    assert purged["status"] == "purged"
    assert not payload_dir.exists()
    rows = list_trash_items("reg-watch", include_terminal=True)
    assert rows[0]["status"] == "purged"


def test_store_metadata_is_owner_only(trash_env):
    home, workspace, _temp_root, policy = trash_env
    (workspace / "private.txt").write_text("private", encoding="utf-8")
    capture_delete_command(
        "rm private.txt",
        cwd=str(workspace),
        workspace_root=str(workspace),
        project="reg-watch",
        run_id="run-8",
        operation_key="private-op",
        policy=policy,
    )
    assert (home / "trash").stat().st_mode & 0o777 == 0o700
    assert (home / "trash" / "trash.db").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("store_name", ["trash", "workspace-snapshots"])
def test_recovery_stores_are_never_delete_targets(trash_env, store_name: str):
    home, workspace, _temp_root, policy = trash_env
    protected = home / store_name / "payload"
    with pytest.raises(UnsafeDeleteTarget, match="recovery storage"):
        parse_delete_command(
            f"rm -rf {protected}",
            cwd=str(workspace),
            workspace_root=str(workspace),
            policy=policy,
        )
