"""Install owner-only command shims for reversible Codex deletion."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import hermes_constants
from hermes_constants import get_hermes_home
from tools import reversible_deletion
from tools.reversible_deletion import ReversibleDeletionPolicy


@dataclass(frozen=True)
class ReversibleDeleteRuntime:
    bin_dir: Path
    runner: Path
    env: dict[str, str]
    codex_config_args: tuple[str, ...]


def _sha256(path: str | os.PathLike[str]) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_write(path: Path, content: str, *, mode: int) -> None:
    pending = path.with_name(f".{path.name}.pending-{os.getpid()}")
    pending.write_text(content, encoding="utf-8")
    os.chmod(pending, mode)
    os.replace(pending, path)


def install_reversible_delete_runtime(
    *,
    workspace_root: str,
    project: str,
    run_id: str,
    policy: ReversibleDeletionPolicy,
    inherited_path: str,
) -> ReversibleDeleteRuntime:
    """Install a run-scoped protected PATH prefix for removal primitives."""
    if not policy.enabled:
        raise ValueError("reversible deletion is disabled")
    workspace = os.path.realpath(os.path.abspath(workspace_root))
    project_key = str(project or "").strip()
    if not project_key:
        raise ValueError("a selected project is required for reversible deletion")

    runtime_key = hashlib.sha256(
        json.dumps(
            [workspace, project_key, str(run_id or "")], separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()[:24]
    runtime_root = get_hermes_home() / "reversible-delete-runtime" / runtime_key
    bin_dir = runtime_root / "bin"
    lib_dir = runtime_root / "lib"
    tools_dir = lib_dir / "tools"
    runtime_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    bin_dir.mkdir(mode=0o700, exist_ok=True)
    lib_dir.mkdir(mode=0o700, exist_ok=True)
    tools_dir.mkdir(mode=0o700, exist_ok=True)
    os.chmod(runtime_root, 0o700)
    os.chmod(bin_dir, 0o700)
    os.chmod(lib_dir, 0o700)
    os.chmod(tools_dir, 0o700)

    source_runner = Path(__file__).with_name("reversible_delete_runner.py")
    runner = runtime_root / "runner.py"
    _atomic_write(runner, source_runner.read_text(encoding="utf-8"), mode=0o700)
    _atomic_write(tools_dir / "__init__.py", "", mode=0o600)
    _atomic_write(
        tools_dir / "reversible_deletion.py",
        Path(reversible_deletion.__file__).read_text(encoding="utf-8"),
        mode=0o600,
    )
    _atomic_write(
        lib_dir / "hermes_constants.py",
        Path(hermes_constants.__file__).read_text(encoding="utf-8"),
        mode=0o600,
    )

    policy_json = json.dumps(asdict(policy), separators=(",", ":"), sort_keys=True)
    path = (
        f"{bin_dir}{os.pathsep}{inherited_path}"
        if inherited_path
        else str(bin_dir)
    )
    shell_env = runtime_root / "shell-env.sh"
    _atomic_write(
        shell_env,
        f"export PATH={shlex.quote(path)}\n",
        mode=0o600,
    )
    common = [
        shlex.quote(sys.executable),
        "-I",
        shlex.quote(str(runner)),
    ]
    for executable in ("rm", "unlink", "rmdir"):
        real_executable = shutil.which(executable, path=os.defpath)
        if not real_executable:
            raise RuntimeError(f"system {executable} executable was not found")
        args = [
            *common,
            "--executable",
            shlex.quote(executable),
            "--real-executable",
            shlex.quote(real_executable),
            "--workspace",
            shlex.quote(workspace),
            "--project",
            shlex.quote(project_key),
            "--run-id",
            shlex.quote(str(run_id or "unscoped")),
            "--policy-json",
            shlex.quote(policy_json),
            "--module-sha256",
            shlex.quote(_sha256(reversible_deletion.__file__)),
            "--constants-sha256",
            shlex.quote(_sha256(hermes_constants.__file__)),
            "--",
            '"$@"',
        ]
        script = (
            "#!/bin/sh\n"
            f"export HERMES_HOME={shlex.quote(str(get_hermes_home()))}\n"
            "exec " + " ".join(args) + "\n"
        )
        _atomic_write(bin_dir / executable, script, mode=0o700)

    codex_config_args = (
        "-c",
        f"shell_environment_policy.set.PATH={json.dumps(path)}",
        "-c",
        (
            "shell_environment_policy.set.BASH_ENV="
            f"{json.dumps(str(shell_env))}"
        ),
    )
    return ReversibleDeleteRuntime(
        bin_dir=bin_dir,
        runner=runner,
        env={
            "PATH": path,
            "BASH_ENV": str(shell_env),
        },
        codex_config_args=codex_config_args,
    )
