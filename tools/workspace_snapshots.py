"""Incremental, server-owned recovery points for native coding-agent turns.

Command inspection can protect exact delete targets, but it cannot see an
``unlink(2)`` hidden inside an arbitrary executable.  This module closes that
gap for files that exist when a turn starts: before Codex runs, Hermes creates
an rsync snapshot of the selected workspace under ``$HERMES_HOME``.  Later
snapshots hard-link unchanged files to the previous immutable recovery point,
so retention costs track changed bytes rather than complete project size.

Snapshots are not model tools.  Recovery materializes into a new destination
and refuses to overwrite an existing path.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

from hermes_constants import get_hermes_home


DEFAULT_KEEP_SNAPSHOTS = 32
DEFAULT_TIMEOUT_SECONDS = 600


class WorkspaceSnapshotError(RuntimeError):
    """Raised when Hermes cannot create or safely materialize a snapshot."""


class WorkspaceSnapshotNotFound(WorkspaceSnapshotError):
    """Raised for an unknown project/snapshot pair."""


class WorkspaceSnapshotConflict(WorkspaceSnapshotError):
    """Raised when materialization would overwrite an existing destination."""


@dataclass(frozen=True)
class WorkspaceSnapshotPolicy:
    enabled: bool = False
    keep_snapshots: int = DEFAULT_KEEP_SNAPSHOTS
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS

    @classmethod
    def from_config(cls, raw: Any) -> "WorkspaceSnapshotPolicy":
        if not isinstance(raw, Mapping):
            return cls()

        def _positive_int(name: str, default: int) -> int:
            value = raw.get(name, default)
            if isinstance(value, bool):
                return default
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                return default
            return parsed if parsed > 0 else default

        return cls(
            enabled=raw.get("enabled") is True,
            keep_snapshots=_positive_int(
                "keep_snapshots", DEFAULT_KEEP_SNAPSHOTS
            ),
            timeout_seconds=_positive_int(
                "timeout_seconds", DEFAULT_TIMEOUT_SECONDS
            ),
        )


@dataclass(frozen=True)
class WorkspaceSnapshot:
    snapshot_id: str
    project: str
    workspace_root: str
    session_id: str
    task_id: str
    created_at: float
    previous_snapshot_id: Optional[str]
    path: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def workspace_snapshot_store_root() -> Path:
    return get_hermes_home() / "workspace-snapshots"


def _safe_project_component(project: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", project.strip()).strip("-._")
    readable = normalized[:48] or "workspace"
    digest = hashlib.sha256(project.encode("utf-8")).hexdigest()[:12]
    return f"{readable}-{digest}"


def _project_store(project: str) -> Path:
    return workspace_snapshot_store_root() / _safe_project_component(project)


def _manifest_path(snapshot_dir: Path) -> Path:
    return snapshot_dir / "manifest.json"


def _load_manifest(snapshot_dir: Path) -> WorkspaceSnapshot:
    try:
        raw = json.loads(_manifest_path(snapshot_dir).read_text(encoding="utf-8"))
        return WorkspaceSnapshot(
            snapshot_id=str(raw["snapshot_id"]),
            project=str(raw["project"]),
            workspace_root=str(raw["workspace_root"]),
            session_id=str(raw.get("session_id") or ""),
            task_id=str(raw.get("task_id") or ""),
            created_at=float(raw["created_at"]),
            previous_snapshot_id=(
                str(raw["previous_snapshot_id"])
                if raw.get("previous_snapshot_id")
                else None
            ),
            path=str(snapshot_dir),
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise WorkspaceSnapshotError(
            f"workspace snapshot manifest is invalid: {snapshot_dir.name}"
        ) from exc


def _complete_snapshot_dirs(project: str) -> list[Path]:
    root = _project_store(project)
    if not root.exists():
        return []
    return sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir()
            and not path.name.startswith(".")
            and _manifest_path(path).is_file()
        ),
        key=lambda path: path.name,
    )


@contextmanager
def _project_lock(project: str) -> Iterator[None]:
    root = _project_store(project)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(workspace_snapshot_store_root(), 0o700)
    os.chmod(root, 0o700)
    lock_path = root / ".snapshot.lock"
    with lock_path.open("a+b") as lock_file:
        os.chmod(lock_path, 0o600)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _write_manifest(path: Path, snapshot: WorkspaceSnapshot) -> None:
    payload = snapshot.as_dict()
    payload.pop("path", None)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def _rsync_binary() -> str:
    binary = shutil.which("rsync")
    if not binary:
        raise WorkspaceSnapshotError("rsync is required for workspace snapshots")
    return binary


def _run_rsync(command: list[str], timeout_seconds: int) -> None:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise WorkspaceSnapshotError("workspace snapshot timed out") from exc
    except OSError as exc:
        raise WorkspaceSnapshotError("workspace snapshot could not start") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "rsync failed").strip()
        raise WorkspaceSnapshotError(
            f"workspace snapshot failed (rsync exit {result.returncode}): "
            f"{detail[-1000:]}"
        )


def _prune_snapshots(project: str, keep: int) -> None:
    snapshots = _complete_snapshot_dirs(project)
    for snapshot_dir in snapshots[: max(0, len(snapshots) - keep)]:
        shutil.rmtree(snapshot_dir)


def capture_workspace_snapshot(
    *,
    workspace_root: str,
    project: str,
    session_id: str,
    task_id: str,
    policy: WorkspaceSnapshotPolicy,
) -> Optional[WorkspaceSnapshot]:
    """Create one complete recovery point before a coding-agent turn.

    A configured snapshot failure is fail-closed: callers must not start the
    turn after this function raises.
    """
    if not policy.enabled:
        return None
    workspace = Path(workspace_root).expanduser().resolve()
    if not workspace.is_dir():
        raise WorkspaceSnapshotError("selected workspace is not a directory")
    store = workspace_snapshot_store_root().resolve()
    try:
        store.relative_to(workspace)
    except ValueError:
        pass
    else:
        raise WorkspaceSnapshotError(
            "workspace snapshot store cannot be inside the selected workspace"
        )

    project_key = project.strip() or workspace.name or "workspace"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    snapshot_id = f"{timestamp}-{uuid.uuid4().hex[:10]}"
    root = _project_store(project_key)

    with _project_lock(project_key):
        previous_dir: Optional[Path] = None
        previous_record: Optional[WorkspaceSnapshot] = None
        complete = _complete_snapshot_dirs(project_key)
        if complete:
            candidate = complete[-1]
            candidate_record = _load_manifest(candidate)
            if candidate_record.workspace_root == str(workspace):
                previous_dir = candidate
                previous_record = candidate_record

        rsync = _rsync_binary()
        pending = root / f".{snapshot_id}.pending"
        final = root / snapshot_id
        payload = pending / "payload"
        pending.mkdir(mode=0o700)
        payload.mkdir(mode=0o700)
        command = [
            rsync,
            "--archive",
            "--hard-links",
            "--one-file-system",
            "--delete",
        ]
        if previous_dir is not None:
            command.append(f"--link-dest={previous_dir / 'payload'}")
        command.extend(["--", f"{workspace}/", f"{payload}/"])
        try:
            _run_rsync(command, policy.timeout_seconds)
            created_at = time.time()
            record = WorkspaceSnapshot(
                snapshot_id=snapshot_id,
                project=project_key,
                workspace_root=str(workspace),
                session_id=str(session_id or ""),
                task_id=str(task_id or ""),
                created_at=created_at,
                previous_snapshot_id=(
                    previous_record.snapshot_id if previous_record else None
                ),
                path=str(final),
            )
            _write_manifest(_manifest_path(pending), record)
            os.replace(pending, final)
            _prune_snapshots(project_key, policy.keep_snapshots)
            return record
        except Exception:
            shutil.rmtree(pending, ignore_errors=True)
            raise


def list_workspace_snapshots(project: str) -> list[dict[str, Any]]:
    records = [
        _load_manifest(snapshot_dir).as_dict()
        for snapshot_dir in reversed(_complete_snapshot_dirs(project))
    ]
    return records


def materialize_workspace_snapshot(
    snapshot_id: str,
    *,
    project: str,
    destination: str,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """Restore a snapshot into a new path without touching the live workspace."""
    destination_path = Path(destination).expanduser().absolute()
    if os.path.lexists(destination_path):
        raise WorkspaceSnapshotConflict("snapshot destination is occupied")
    snapshot_dir = _project_store(project) / snapshot_id
    if not snapshot_dir.is_dir() or not _manifest_path(snapshot_dir).is_file():
        raise WorkspaceSnapshotNotFound("workspace snapshot was not found")
    record = _load_manifest(snapshot_dir)
    if record.project != project:
        raise WorkspaceSnapshotNotFound("workspace snapshot was not found")

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.mkdir(mode=0o700)
    try:
        _run_rsync(
            [
                _rsync_binary(),
                "--archive",
                "--hard-links",
                "--one-file-system",
                "--delete",
                "--",
                f"{snapshot_dir / 'payload'}/",
                f"{destination_path}/",
            ],
            timeout_seconds,
        )
    except Exception:
        shutil.rmtree(destination_path, ignore_errors=True)
        raise
    return str(destination_path)


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Inspect or safely materialize Hermes workspace snapshots"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    list_parser = subparsers.add_parser("list", help="list project snapshots")
    list_parser.add_argument("--project", required=True)
    restore_parser = subparsers.add_parser(
        "materialize", help="copy one snapshot into a new directory"
    )
    restore_parser.add_argument("--project", required=True)
    restore_parser.add_argument("--snapshot", required=True)
    restore_parser.add_argument("--destination", required=True)
    args = parser.parse_args()

    if args.command == "list":
        print(json.dumps(list_workspace_snapshots(args.project), indent=2))
        return 0
    restored = materialize_workspace_snapshot(
        args.snapshot,
        project=args.project,
        destination=args.destination,
    )
    print(restored)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
