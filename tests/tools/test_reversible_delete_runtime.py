from __future__ import annotations

import os
import subprocess
from pathlib import Path

from tools.reversible_delete_runtime import install_reversible_delete_runtime
from tools.reversible_deletion import ReversibleDeletionPolicy, list_trash_items


def test_runtime_shim_captures_expanded_glob_before_real_rm(
    tmp_path: Path, monkeypatch
):
    hermes_home = tmp_path / "hermes-home"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    first = workspace / "one.txt"
    second = workspace / "two.log"
    first.write_text("one", encoding="utf-8")
    second.write_text("two", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    policy = ReversibleDeletionPolicy(
        enabled=True, temp_roots=(str(tmp_path / "temp"),)
    )
    runtime = install_reversible_delete_runtime(
        workspace_root=str(workspace),
        project="reg-watch",
        run_id="runtime-run",
        policy=policy,
        inherited_path=os.environ.get("PATH", ""),
    )

    env = dict(os.environ)
    env.update(runtime.env)
    env["TARGET"] = first.name
    result = subprocess.run(
        ["/bin/sh", "-c", 'rm -f "$TARGET" *.log'],
        cwd=workspace,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert not first.exists() and not second.exists()
    assert {item["original_path"] for item in list_trash_items("reg-watch")} == {
        str(first),
        str(second),
    }


def test_runtime_shim_blocks_outside_workspace_without_deleting(
    tmp_path: Path, monkeypatch
):
    hermes_home = tmp_path / "hermes-home"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("keep", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    runtime = install_reversible_delete_runtime(
        workspace_root=str(workspace),
        project="reg-watch",
        run_id="runtime-run",
        policy=ReversibleDeletionPolicy(
            enabled=True,
            temp_roots=(str(tmp_path / "temp"),),
        ),
        inherited_path=os.environ.get("PATH", ""),
    )

    result = subprocess.run(
        [str(runtime.bin_dir / "rm"), "-f", str(outside)],
        cwd=workspace,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 125
    assert "outside the selected workspace" in result.stderr
    assert outside.exists()
