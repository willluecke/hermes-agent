"""Server-enforced reversible deletion for unattended Codex runs.

The Codex app-server approval bridge calls this module before accepting an
inspectable delete.  Targets are copied into an owner-only store outside the
workspace, then recorded in SQLite.  The original command remains responsible
for removing the source; a failed capture never authorizes the delete.

This is intentionally not a model tool.  Agents cannot restore or purge the
store directly.  Those operations are exposed only through authenticated
gateway routes using server-resolved project keys and item IDs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from hermes_constants import get_hermes_home


DEFAULT_MAX_ITEM_BYTES = 5 * 1024**3
DEFAULT_MAX_TOTAL_BYTES = 100 * 1024**3
DEFAULT_MAX_ENTRIES = 100_000
DEFAULT_TEMP_ROOTS = ("/tmp", "/var/tmp")

_STORE_LOCK = threading.RLock()
_SHELL_METACHARACTERS = frozenset("$`*?[]{}<>|&;\n\r")


class ReversibleDeletionError(RuntimeError):
    """Base error for capture, restore, and purge failures."""


class UnsafeDeleteTarget(ReversibleDeletionError):
    """Raised when a target falls outside the configured reversible boundary."""


class TrashQuotaExceeded(ReversibleDeletionError):
    """Raised when capture would exceed an item or store limit."""


class TrashItemNotFound(ReversibleDeletionError):
    """Raised for an unknown item or a project/item mismatch."""


class RestoreConflict(ReversibleDeletionError):
    """Raised when restore would overwrite an existing filesystem entry."""


@dataclass(frozen=True)
class ReversibleDeletionPolicy:
    enabled: bool = False
    max_item_bytes: int = DEFAULT_MAX_ITEM_BYTES
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES
    max_entries: int = DEFAULT_MAX_ENTRIES
    temp_roots: tuple[str, ...] = DEFAULT_TEMP_ROOTS

    @classmethod
    def from_config(cls, raw: Any) -> "ReversibleDeletionPolicy":
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

        configured_roots = raw.get("temp_roots", DEFAULT_TEMP_ROOTS)
        roots: list[str] = []
        if isinstance(configured_roots, (list, tuple)):
            for value in configured_roots:
                if isinstance(value, str) and value.strip():
                    root = os.path.realpath(os.path.abspath(value.strip()))
                    if root not in roots:
                        roots.append(root)
        return cls(
            enabled=raw.get("enabled") is True,
            max_item_bytes=_positive_int("max_item_bytes", DEFAULT_MAX_ITEM_BYTES),
            max_total_bytes=_positive_int("max_total_bytes", DEFAULT_MAX_TOTAL_BYTES),
            max_entries=_positive_int("max_entries", DEFAULT_MAX_ENTRIES),
            temp_roots=tuple(roots) or DEFAULT_TEMP_ROOTS,
        )


@dataclass(frozen=True)
class DeletePlan:
    command: str
    targets: tuple[str, ...]


@dataclass(frozen=True)
class TrashItem:
    item_id: str
    project: str
    run_id: str
    operation_key: str
    original_path: str
    item_type: str
    size_bytes: int
    entry_count: int
    captured_at: float
    status: str
    restored_at: Optional[float]
    purged_at: Optional[float]
    source_present: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CaptureResult:
    handled: bool
    items: tuple[TrashItem, ...] = ()
    missing_targets: tuple[str, ...] = ()
    reason: Optional[str] = None


def _store_root() -> Path:
    return get_hermes_home() / "trash"


def _protected_store_roots() -> tuple[Path, ...]:
    return (
        _store_root(),
        get_hermes_home() / "workspace-snapshots",
        get_hermes_home() / "reversible-delete-runtime",
    )


def _db_path() -> Path:
    return _store_root() / "trash.db"


def _payloads_root() -> Path:
    return _store_root() / "payloads"


def _ensure_store_dirs() -> None:
    root = _store_root()
    payloads = _payloads_root()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    payloads.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    os.chmod(payloads, 0o700)


def _connect() -> sqlite3.Connection:
    _ensure_store_dirs()
    conn = sqlite3.connect(_db_path(), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        current_row = conn.execute("PRAGMA journal_mode").fetchone()
        current_mode = str(current_row[0] if current_row else "").lower()
        sqlite_version = tuple(sqlite3.sqlite_version_info[:3])
        wal_reset_vulnerable = (
            (3, 7, 0) <= sqlite_version < (3, 51, 3)
            and not (3, 50, 7) <= sqlite_version < (3, 51, 0)
            and not (3, 44, 6) <= sqlite_version < (3, 45, 0)
        )
        requested_mode = (
            "DELETE" if wal_reset_vulnerable and current_mode != "wal" else "WAL"
        )
        try:
            effective_row = conn.execute(
                f"PRAGMA journal_mode={requested_mode}"
            ).fetchone()
            effective_mode = str(effective_row[0] if effective_row else "").lower()
            if requested_mode == "WAL" and effective_mode != "wal":
                conn.execute("PRAGMA journal_mode=DELETE")
        except sqlite3.OperationalError:
            conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS trash_items (
                item_id TEXT PRIMARY KEY,
                project TEXT NOT NULL,
                run_id TEXT NOT NULL,
                operation_key TEXT NOT NULL,
                original_path TEXT NOT NULL,
                payload_relpath TEXT NOT NULL,
                item_type TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                entry_count INTEGER NOT NULL,
                captured_at REAL NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('active', 'restored', 'purged')),
                restored_at REAL,
                purged_at REAL,
                UNIQUE(operation_key, original_path)
            )"""
        )
        conn.execute(
            """CREATE INDEX IF NOT EXISTS idx_trash_items_project_status
               ON trash_items(project, status, captured_at DESC)"""
        )
        conn.commit()
        os.chmod(_db_path(), 0o600)
    except Exception:
        conn.close()
        raise
    return conn


def _lexists(path: str | os.PathLike[str]) -> bool:
    return os.path.lexists(os.fspath(path))


def _is_descendant(path: str, root: str) -> bool:
    try:
        return os.path.commonpath((path, root)) == root and path != root
    except ValueError:
        return False


def _canonical_delete_target(raw_path: str, cwd: str) -> str:
    if not raw_path or raw_path.startswith("~"):
        raise UnsafeDeleteTarget("delete target is empty or uses home expansion")
    if any(ch in raw_path for ch in _SHELL_METACHARACTERS):
        raise UnsafeDeleteTarget("delete target is dynamic or contains shell syntax")
    absolute = os.path.abspath(
        raw_path if os.path.isabs(raw_path) else os.path.join(cwd, raw_path)
    )
    parent = os.path.realpath(os.path.dirname(absolute))
    return os.path.join(parent, os.path.basename(absolute))


def validate_delete_target(
    raw_path: str,
    *,
    cwd: str,
    workspace_root: str,
    policy: ReversibleDeletionPolicy,
) -> str:
    """Resolve one target without following the target symlink itself."""
    target = _canonical_delete_target(raw_path, cwd)
    workspace = os.path.realpath(os.path.abspath(workspace_root))
    for protected_root_path in _protected_store_roots():
        protected_root = os.path.realpath(protected_root_path)
        if target == protected_root or _is_descendant(target, protected_root):
            raise UnsafeDeleteTarget("Hermes recovery storage is protected")
    if _is_descendant(target, workspace):
        return target
    for configured_root in policy.temp_roots:
        temp_root = os.path.realpath(os.path.abspath(configured_root))
        if _is_descendant(target, temp_root):
            return target
    raise UnsafeDeleteTarget("delete target is outside the selected workspace")


def validate_expanded_delete_target(
    raw_path: str,
    *,
    cwd: str,
    workspace_root: str,
    policy: ReversibleDeletionPolicy,
) -> str:
    """Validate one argv operand after the shell has already expanded it.

    Unlike :func:`validate_delete_target`, metacharacters are ordinary filename
    bytes here. The protected command shim receives an argv vector directly,
    so there is no second shell interpretation to defend against.
    """
    if not raw_path or raw_path.startswith("~"):
        raise UnsafeDeleteTarget("delete target is empty or uses home expansion")
    absolute = os.path.abspath(
        raw_path if os.path.isabs(raw_path) else os.path.join(cwd, raw_path)
    )
    parent = os.path.realpath(os.path.dirname(absolute))
    target = os.path.join(parent, os.path.basename(absolute))
    workspace = os.path.realpath(os.path.abspath(workspace_root))
    for protected_root_path in _protected_store_roots():
        protected_root = os.path.realpath(protected_root_path)
        if target == protected_root or _is_descendant(target, protected_root):
            raise UnsafeDeleteTarget("Hermes recovery storage is protected")
    if _is_descendant(target, workspace):
        return target
    for configured_root in policy.temp_roots:
        temp_root = os.path.realpath(os.path.abspath(configured_root))
        if _is_descendant(target, temp_root):
            return target
    raise UnsafeDeleteTarget("delete target is outside the selected workspace")


def _unwrap_static_command(command: str) -> list[str]:
    try:
        argv = shlex.split(command, posix=True)
    except ValueError as exc:
        raise UnsafeDeleteTarget("delete command could not be parsed") from exc
    if not argv:
        raise UnsafeDeleteTarget("delete command is empty")
    executable = os.path.basename(argv[0])
    if executable in {"bash", "sh", "zsh", "dash"}:
        if len(argv) != 3 or argv[1] not in {"-c", "-lc"}:
            raise UnsafeDeleteTarget("shell wrapper is not a single static command")
        return _unwrap_static_command(argv[2])
    return argv


def protected_removal_command_name(command: str) -> Optional[str]:
    """Return the removal primitive for one shim-covered shell command.

    Variables and globs are permitted because the runtime shim validates their
    expanded argv. Chained commands, redirects, command substitution, explicit
    binary paths, and wrappers other than the normal login shell are excluded;
    those retain the existing approval/checkpoint path.
    """
    candidate = str(command or "").strip()
    if not candidate or "$(" in candidate or "`" in candidate:
        return None
    try:
        argv = shlex.split(candidate, posix=True)
        if argv and os.path.basename(argv[0]) == "bash":
            if len(argv) != 3 or argv[1] not in {"-c", "-lc"}:
                return None
            candidate = argv[2]
        lexer = shlex.shlex(
            candidate,
            posix=True,
            punctuation_chars=";&|<>",
        )
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:
        return None
    if not tokens or any(
        token and set(token).issubset(set(";&|<>")) for token in tokens
    ):
        return None
    executable = tokens[0]
    if os.path.dirname(executable):
        return None
    return executable if executable in {"rm", "unlink", "rmdir"} else None


def _rm_operands(argv: Sequence[str]) -> list[str]:
    operands: list[str] = []
    options_done = False
    for token in argv[1:]:
        if not options_done and token == "--":
            options_done = True
            continue
        if not options_done and token.startswith("-") and token != "-":
            if token.startswith("--"):
                if token not in {"--force", "--recursive", "--dir", "--verbose"}:
                    raise UnsafeDeleteTarget("rm option is not supported for auto-trash")
            elif not set(token[1:]).issubset(set("frdv")):
                raise UnsafeDeleteTarget("rm option is not supported for auto-trash")
            continue
        operands.append(token)
    return operands


def _expanded_rm_operands(argv: Sequence[str]) -> list[str]:
    """Return operands from a shell-expanded GNU/BSD ``rm`` argv."""
    operands: list[str] = []
    options_done = False
    supported_long = {
        "--dir",
        "--force",
        "--interactive",
        "--no-preserve-root",
        "--one-file-system",
        "--preserve-root",
        "--recursive",
        "--verbose",
    }
    for token in argv[1:]:
        if not options_done and token == "--":
            options_done = True
            continue
        if not options_done and token.startswith("-") and token != "-":
            if token in {"--help", "--version"}:
                continue
            if token.startswith("--interactive=") or token.startswith(
                "--preserve-root="
            ):
                continue
            if token.startswith("--"):
                if token not in supported_long:
                    raise UnsafeDeleteTarget(
                        "rm option is not supported by the protected shim"
                    )
            elif not set(token[1:]).issubset(set("dfiIRrv")):
                raise UnsafeDeleteTarget(
                    "rm option is not supported by the protected shim"
                )
            continue
        operands.append(token)
    return operands


def parse_expanded_delete_argv(
    executable: str,
    args: Sequence[str],
    *,
    cwd: str,
    workspace_root: str,
    policy: ReversibleDeletionPolicy,
) -> DeletePlan:
    """Build a delete plan from argv received by the protected command shim."""
    name = os.path.basename(executable)
    argv = [name, *[str(arg) for arg in args]]
    if name == "rm":
        operands = _expanded_rm_operands(argv)
    elif name == "unlink":
        operands = [token for token in argv[1:] if token != "--"]
        if operands in (["--help"], ["--version"]):
            operands = []
        elif len(operands) != 1 or operands[0].startswith("-"):
            raise UnsafeDeleteTarget(
                "unlink must name exactly one target through the protected shim"
            )
    elif name == "rmdir":
        operands = []
        options_done = False
        for token in argv[1:]:
            if not options_done and token == "--":
                options_done = True
                continue
            if not options_done and token in {"--help", "--version"}:
                continue
            if not options_done and token.startswith("-"):
                if token not in {
                    "--ignore-fail-on-non-empty",
                    "--verbose",
                    "-v",
                }:
                    raise UnsafeDeleteTarget(
                        "rmdir option is not supported by the protected shim"
                    )
                continue
            operands.append(token)
    else:
        raise UnsafeDeleteTarget("command is not a protected removal primitive")

    targets: list[str] = []
    for operand in operands:
        target = validate_expanded_delete_target(
            operand,
            cwd=cwd,
            workspace_root=workspace_root,
            policy=policy,
        )
        if target not in targets:
            targets.append(target)
    return DeletePlan(command=shlex.join(argv), targets=tuple(targets))


def parse_delete_command(
    command: str,
    *,
    cwd: str,
    workspace_root: str,
    policy: ReversibleDeletionPolicy,
) -> DeletePlan:
    """Parse an exact static rm/unlink/rmdir command into validated targets."""
    argv = _unwrap_static_command(command)
    executable = os.path.basename(argv[0])
    if executable == "rm":
        operands = _rm_operands(argv)
    elif executable == "unlink":
        if len(argv) != 2 or argv[1].startswith("-"):
            raise UnsafeDeleteTarget("unlink must name exactly one static target")
        operands = [argv[1]]
    elif executable == "rmdir":
        operands = []
        for token in argv[1:]:
            if token.startswith("-"):
                if token not in {"--ignore-fail-on-non-empty", "--verbose", "-v"}:
                    raise UnsafeDeleteTarget("rmdir option is not supported for auto-trash")
                continue
            operands.append(token)
    else:
        raise UnsafeDeleteTarget("command is not a supported static delete")
    if not operands:
        raise UnsafeDeleteTarget("delete command has no targets")

    targets: list[str] = []
    for operand in operands:
        target = validate_delete_target(
            operand,
            cwd=cwd,
            workspace_root=workspace_root,
            policy=policy,
        )
        if target not in targets:
            targets.append(target)
    return DeletePlan(command=command, targets=tuple(targets))


def command_targets_trash_store(command: str, *, cwd: str) -> bool:
    """Detect direct attempts to mutate a protected recovery namespace.

    The historical function name is retained for callers; it now protects both
    per-target Trash payloads and whole-workspace snapshots.
    """
    protected_roots = tuple(
        os.path.realpath(path) for path in _protected_store_roots()
    )
    lowered = command.lower()
    textual_markers = tuple(str(path).lower() for path in _protected_store_roots()) + (
        "$hermes_home/trash",
        "${hermes_home}/trash",
        "~/.hermes/trash",
        "$hermes_home/workspace-snapshots",
        "${hermes_home}/workspace-snapshots",
        "~/.hermes/workspace-snapshots",
    )
    if any(marker in lowered for marker in textual_markers):
        return True
    try:
        argv = _unwrap_static_command(command)
    except ReversibleDeletionError:
        return False
    executable = os.path.basename(argv[0])
    if executable not in {"rm", "unlink", "rmdir"}:
        return False
    try:
        operands = (
            _rm_operands(argv)
            if executable == "rm"
            else [token for token in argv[1:] if not token.startswith("-")]
        )
        for operand in operands:
            target = _canonical_delete_target(operand, cwd)
            for protected_root in protected_roots:
                if target == protected_root or _is_descendant(target, protected_root):
                    return True
    except ReversibleDeletionError:
        return False
    return False


def looks_like_file_delete(command: str) -> bool:
    """Return whether a command can remove or irreversibly replace file data.

    Exact rm/unlink/rmdir targets are captured by the Trash layer. Other forms
    are deliberately classified as reviewable; the pre-turn workspace snapshot
    remains the recovery layer when their targets are dynamic or opaque.
    """
    try:
        argv = _unwrap_static_command(command)
    except ReversibleDeletionError:
        argv = []
    if argv:
        executable = os.path.basename(argv[0]).lower()
        lowered_argv = [token.lower() for token in argv[1:]]
        if executable in {"rm", "unlink", "rmdir", "srm", "wipe", "truncate"}:
            return True
        if executable == "shred" and any(
            token == "-u" or "u" in token[1:] or token.startswith("--remove")
            for token in lowered_argv
            if token.startswith("-")
        ):
            return True
        if executable == "busybox" and lowered_argv[:1] in (
            ["rm"],
            ["unlink"],
            ["rmdir"],
        ):
            return True
        if executable == "find" and (
            "-delete" in lowered_argv
            or any(
                token in {"rm", "unlink", "rmdir", "shred"}
                for token in lowered_argv
            )
        ):
            return True
        if executable == "xargs" and any(
            os.path.basename(token) in {"rm", "unlink", "rmdir", "shred"}
            for token in lowered_argv
        ):
            return True
        if executable == "git" and any(
            token in {"clean", "reset", "restore", "checkout"}
            for token in lowered_argv[:2]
        ):
            return True
        if executable == "rsync" and any(
            token == "--delete" or token.startswith("--delete-")
            for token in lowered_argv
        ):
            return True
        if executable == "dd" and any(token.startswith("of=") for token in argv[1:]):
            return True
        if executable == "tee" and "-a" not in lowered_argv and "--append" not in lowered_argv:
            return True
        if executable in {"make", "gmake"} and "clean" in lowered_argv:
            return True
        if executable in {"npm", "pnpm", "yarn", "bun"} and any(
            token in {"clean", "run"} for token in lowered_argv[:2]
        ) and "clean" in lowered_argv:
            return True
    lowered = command.lower()
    opaque_patterns = (
        r"\bshutil\.rmtree\s*\(",
        r"\bos\.(?:remove|unlink|rmdir|removedirs)\s*\(",
        r"\bpathlib\b.*\.(?:unlink|rmdir)\s*\(",
        r"\.unlink\s*\(",
        r"\bfs(?:\.promises)?\.(?:rm|rmdir|unlink|removedir)\s*\(",
        r"\bdeno\.remove\s*\(",
        r"\bfileutils\.(?:rm|rm_f|rm_r|rm_rf|remove|remove_dir|remove_entry)\s*\(",
        r"\bfile\.(?:delete|unlink)\s*\(",
        r"\bfiles\.delete(?:ifexists)?\s*\(",
        r"\bos\.(?:remove|removeall)\s*\(",
        r"\bstd::fs::(?:remove_file|remove_dir|remove_dir_all)\s*\(",
        r"\bremove-item\b",
        r"\bunlink\b",
    )
    return any(re.search(pattern, lowered) for pattern in opaque_patterns)


def _entry_type(path: str) -> str:
    if os.path.islink(path):
        return "symlink"
    if os.path.isdir(path):
        return "directory"
    if os.path.isfile(path):
        return "file"
    return "other"


def _measure(path: str, policy: ReversibleDeletionPolicy) -> tuple[int, int]:
    if os.path.islink(path):
        return (len(os.readlink(path).encode("utf-8", errors="surrogateescape")), 1)
    if not os.path.isdir(path):
        size = os.lstat(path).st_size
        if size > policy.max_item_bytes:
            raise TrashQuotaExceeded("trash item exceeds the configured byte limit")
        return (size, 1)

    total = 0
    entries = 1
    for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
        for name in tuple(dirnames) + tuple(filenames):
            entry = os.path.join(dirpath, name)
            entries += 1
            if entries > policy.max_entries:
                raise TrashQuotaExceeded("trash item exceeds the filesystem entry limit")
            try:
                if os.path.islink(entry):
                    total += len(
                        os.readlink(entry).encode("utf-8", errors="surrogateescape")
                    )
                elif os.path.isfile(entry):
                    total += os.lstat(entry).st_size
            except FileNotFoundError as exc:
                raise ReversibleDeletionError(
                    "delete target changed while Hermes was capturing it"
                ) from exc
            if total > policy.max_item_bytes:
                raise TrashQuotaExceeded("trash item exceeds the configured byte limit")
    return (total, entries)


def _copy_entry(source: str, destination: Path) -> None:
    if os.path.islink(source):
        destination.symlink_to(os.readlink(source), target_is_directory=False)
    elif os.path.isdir(source):
        shutil.copytree(source, destination, symlinks=True, copy_function=shutil.copy2)
    elif os.path.isfile(source):
        shutil.copy2(source, destination, follow_symlinks=False)
    else:
        raise ReversibleDeletionError("special filesystem entries cannot be auto-trashed")


def _row_to_item(row: sqlite3.Row) -> TrashItem:
    return TrashItem(
        item_id=str(row["item_id"]),
        project=str(row["project"]),
        run_id=str(row["run_id"]),
        operation_key=str(row["operation_key"]),
        original_path=str(row["original_path"]),
        item_type=str(row["item_type"]),
        size_bytes=int(row["size_bytes"]),
        entry_count=int(row["entry_count"]),
        captured_at=float(row["captured_at"]),
        status=str(row["status"]),
        restored_at=(
            float(row["restored_at"]) if row["restored_at"] is not None else None
        ),
        purged_at=float(row["purged_at"]) if row["purged_at"] is not None else None,
        source_present=_lexists(str(row["original_path"])),
    )


def _existing_item(
    conn: sqlite3.Connection, operation_key: str, original_path: str
) -> Optional[TrashItem]:
    row = conn.execute(
        "SELECT * FROM trash_items WHERE operation_key=? AND original_path=?",
        (operation_key, original_path),
    ).fetchone()
    return _row_to_item(row) if row is not None else None


def _active_size(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        """SELECT COALESCE(SUM(size_bytes), 0) AS total
           FROM trash_items WHERE status!='purged'"""
    ).fetchone()
    return int(row["total"] if row is not None else 0)


def _capture_target(
    target: str,
    *,
    project: str,
    run_id: str,
    operation_key: str,
    policy: ReversibleDeletionPolicy,
) -> TrashItem:
    with _STORE_LOCK:
        conn = _connect()
        try:
            existing = _existing_item(conn, operation_key, target)
            if existing is not None:
                return existing
            size_bytes, entry_count = _measure(target, policy)
            if _active_size(conn) + size_bytes > policy.max_total_bytes:
                raise TrashQuotaExceeded("trash store exceeds the configured byte limit")

            item_id = uuid.uuid4().hex
            pending_dir = _payloads_root() / f".pending-{item_id}"
            item_dir = _payloads_root() / item_id
            payload = pending_dir / "payload"
            pending_dir.mkdir(mode=0o700)
            try:
                _copy_entry(target, payload)
                copied_size, copied_entries = _measure(str(payload), policy)
                if (copied_size, copied_entries) != (size_bytes, entry_count):
                    raise ReversibleDeletionError(
                        "delete target changed while Hermes was capturing it"
                    )
                os.replace(pending_dir, item_dir)
                captured_at = time.time()
                payload_relpath = str(Path("payloads") / item_id / "payload")
                try:
                    conn.execute(
                        """INSERT INTO trash_items (
                            item_id, project, run_id, operation_key,
                            original_path, payload_relpath, item_type,
                            size_bytes, entry_count, captured_at, status
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')""",
                        (
                            item_id,
                            project,
                            run_id,
                            operation_key,
                            target,
                            payload_relpath,
                            _entry_type(target),
                            size_bytes,
                            entry_count,
                            captured_at,
                        ),
                    )
                    conn.commit()
                except sqlite3.IntegrityError:
                    shutil.rmtree(item_dir, ignore_errors=True)
                    existing = _existing_item(conn, operation_key, target)
                    if existing is None:
                        raise
                    return existing
            except Exception:
                shutil.rmtree(pending_dir, ignore_errors=True)
                if item_dir.exists():
                    shutil.rmtree(item_dir, ignore_errors=True)
                raise
            row = conn.execute(
                "SELECT * FROM trash_items WHERE item_id=?", (item_id,)
            ).fetchone()
            assert row is not None
            return _row_to_item(row)
        finally:
            conn.close()


def operation_key_for(*parts: str) -> str:
    encoded = json.dumps([str(part) for part in parts], separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def capture_delete_command(
    command: str,
    *,
    cwd: str,
    workspace_root: str,
    project: str,
    run_id: str,
    operation_key: str,
    policy: ReversibleDeletionPolicy,
) -> CaptureResult:
    """Protect every exact target, or report why normal approval is required."""
    if not policy.enabled:
        return CaptureResult(
            handled=False, reason="reversible deletion is disabled"
        )
    try:
        plan = parse_delete_command(
            command,
            cwd=cwd,
            workspace_root=workspace_root,
            policy=policy,
        )
    except ReversibleDeletionError as exc:
        return CaptureResult(handled=False, reason=str(exc))

    items: list[TrashItem] = []
    missing: list[str] = []
    try:
        for target in plan.targets:
            if not _lexists(target):
                missing.append(target)
                continue
            items.append(
                _capture_target(
                    target,
                    project=project,
                    run_id=run_id,
                    operation_key=operation_key,
                    policy=policy,
                )
            )
    except ReversibleDeletionError as exc:
        return CaptureResult(handled=False, reason=str(exc))
    return CaptureResult(
        handled=True,
        items=tuple(items),
        missing_targets=tuple(missing),
    )


def capture_delete_argv(
    executable: str,
    args: Sequence[str],
    *,
    cwd: str,
    workspace_root: str,
    project: str,
    run_id: str,
    operation_key: str,
    policy: ReversibleDeletionPolicy,
) -> CaptureResult:
    """Capture shell-expanded removal operands before invoking the real binary."""
    if not policy.enabled:
        return CaptureResult(handled=False, reason="reversible deletion is disabled")
    try:
        plan = parse_expanded_delete_argv(
            executable,
            args,
            cwd=cwd,
            workspace_root=workspace_root,
            policy=policy,
        )
    except ReversibleDeletionError as exc:
        return CaptureResult(handled=False, reason=str(exc))

    items: list[TrashItem] = []
    missing: list[str] = []
    try:
        for target in plan.targets:
            if not _lexists(target):
                missing.append(target)
                continue
            items.append(
                _capture_target(
                    target,
                    project=project,
                    run_id=run_id,
                    operation_key=operation_key,
                    policy=policy,
                )
            )
    except ReversibleDeletionError as exc:
        return CaptureResult(handled=False, reason=str(exc))
    return CaptureResult(True, tuple(items), tuple(missing))


def capture_file_change_paths(
    paths: Iterable[str],
    *,
    cwd: str,
    workspace_root: str,
    project: str,
    run_id: str,
    operation_key: str,
    policy: ReversibleDeletionPolicy,
) -> CaptureResult:
    if not policy.enabled:
        return CaptureResult(handled=False, reason="reversible deletion is disabled")
    targets: list[str] = []
    try:
        for raw_path in paths:
            target = validate_delete_target(
                raw_path,
                cwd=cwd,
                workspace_root=workspace_root,
                policy=policy,
            )
            if target not in targets:
                targets.append(target)
    except ReversibleDeletionError as exc:
        return CaptureResult(handled=False, reason=str(exc))
    if not targets:
        return CaptureResult(handled=False, reason="file change has no delete targets")

    items: list[TrashItem] = []
    missing: list[str] = []
    try:
        for target in targets:
            if not _lexists(target):
                missing.append(target)
                continue
            items.append(
                _capture_target(
                    target,
                    project=project,
                    run_id=run_id,
                    operation_key=operation_key,
                    policy=policy,
                )
            )
    except ReversibleDeletionError as exc:
        return CaptureResult(handled=False, reason=str(exc))
    return CaptureResult(True, tuple(items), tuple(missing))


def list_trash_items(project: str, *, include_terminal: bool = False) -> list[dict[str, Any]]:
    with _STORE_LOCK:
        conn = _connect()
        try:
            if include_terminal:
                rows = conn.execute(
                    "SELECT * FROM trash_items WHERE project=? ORDER BY captured_at DESC",
                    (project,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT * FROM trash_items
                       WHERE project=? AND status!='purged'
                       ORDER BY captured_at DESC""",
                    (project,),
                ).fetchall()
            return [_row_to_item(row).as_dict() for row in rows]
        finally:
            conn.close()


def _load_item(conn: sqlite3.Connection, item_id: str, project: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM trash_items WHERE item_id=? AND project=?",
        (item_id, project),
    ).fetchone()
    if row is None:
        raise TrashItemNotFound("trash item was not found for this project")
    return row


def restore_trash_item(
    item_id: str,
    *,
    project: str,
    workspace_root: str,
    policy: ReversibleDeletionPolicy,
) -> dict[str, Any]:
    with _STORE_LOCK:
        conn = _connect()
        try:
            row = _load_item(conn, item_id, project)
            if row["status"] == "purged":
                raise TrashItemNotFound("trash item was permanently deleted")
            destination = validate_delete_target(
                str(row["original_path"]),
                cwd=workspace_root,
                workspace_root=workspace_root,
                policy=policy,
            )
            if _lexists(destination):
                raise RestoreConflict("original destination is occupied")
            payload = _store_root() / str(row["payload_relpath"])
            if not _lexists(payload):
                raise TrashItemNotFound("trash payload is missing")
            destination_parent = Path(destination).parent
            destination_parent.mkdir(parents=True, exist_ok=True)
            revalidated = validate_delete_target(
                destination,
                cwd=workspace_root,
                workspace_root=workspace_root,
                policy=policy,
            )
            if revalidated != destination:
                raise RestoreConflict("restore parent changed during validation")
            temporary = destination_parent / f".hermes-restore-{item_id}"
            if _lexists(temporary):
                raise RestoreConflict("restore staging path is occupied")
            try:
                _copy_entry(str(payload), temporary)
                os.replace(temporary, destination)
            except Exception:
                if _lexists(temporary):
                    if temporary.is_dir() and not temporary.is_symlink():
                        shutil.rmtree(temporary, ignore_errors=True)
                    else:
                        temporary.unlink(missing_ok=True)
                raise
            restored_at = time.time()
            conn.execute(
                "UPDATE trash_items SET status='restored', restored_at=? WHERE item_id=?",
                (restored_at, item_id),
            )
            conn.commit()
            updated = _load_item(conn, item_id, project)
            return _row_to_item(updated).as_dict()
        finally:
            conn.close()


def purge_trash_item(item_id: str, *, project: str) -> dict[str, Any]:
    with _STORE_LOCK:
        conn = _connect()
        try:
            row = _load_item(conn, item_id, project)
            if row["status"] == "purged":
                return _row_to_item(row).as_dict()
            payload = _store_root() / str(row["payload_relpath"])
            item_dir = payload.parent
            if item_dir.exists():
                shutil.rmtree(item_dir)
            purged_at = time.time()
            conn.execute(
                "UPDATE trash_items SET status='purged', purged_at=? WHERE item_id=?",
                (purged_at, item_id),
            )
            conn.commit()
            updated = _load_item(conn, item_id, project)
            return _row_to_item(updated).as_dict()
        finally:
            conn.close()
