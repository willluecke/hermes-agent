"""Detached background jobs registered to a Hermes Chat conversation.

Long work (a multi-agent build, a render) that must outlive the turn and the
gateway itself runs as a transient systemd user unit started by
``hermes-bg`` (command-center ``bin/hermes-bg``, linked from
``~/.local/bin``). Once the unit has started, the launcher writes one
registry file per job to ``~/.hermes/background-jobs/<unit>.json`` naming the
conversation's session, a title, and optionally a status file the job keeps
current. Inside the unit the launcher runs the command and records its
``ended_at`` and ``exit_status`` in that file before the unit ends. This
module reads that registry for the app's background-work indicator.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

# A finished job stays listed this long so the indicator can say it ended.
FINISHED_JOB_VISIBLE_SECONDS = 15 * 60
# Registry files of jobs that ended longer ago than this are removed.
FINISHED_JOB_RETENTION_SECONDS = 7 * 24 * 3600
# A unit that reads inactive or not-found this soon after its job started is
# not taken as ended: the probe may have raced the launch, and an end stamped
# then would stand for good.
UNIT_START_GRACE_SECONDS = 30.0

_RUNNING_UNIT_STATES = frozenset({"active", "activating", "deactivating", "reloading"})


def registry_dir() -> Path:
    home = Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser()
    return home / "background-jobs"


def systemd_unit_state(unit: str) -> Optional[str]:
    """The unit's ActiveState, "not-found" once systemd has collected it, or
    None when systemd cannot be asked."""
    try:
        completed = subprocess.run(
            [
                "systemctl", "--user", "show",
                "--property=LoadState,ActiveState", unit,
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    properties = dict(
        line.split("=", 1) for line in completed.stdout.splitlines() if "=" in line
    )
    if properties.get("LoadState") == "not-found":
        return "not-found"
    return properties.get("ActiveState") or None


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _exit_status(entry: dict[str, Any]) -> Optional[int]:
    value = entry.get("exit_status")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _load_entry(path: Path) -> Optional[dict[str, Any]]:
    try:
        entry = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return entry if isinstance(entry, dict) else None


# The gateway answers each /background poll on a worker thread, so two polls
# (phone and laptop) can reach the same end stamp at once.
_STAMP_LOCK = threading.Lock()


def _write_entry(path: Path, entry: dict[str, Any]) -> None:
    # Replace, never rewrite in place: a reader must not see half a file. Each
    # writer gets its own temporary file; a shared name let two writers
    # truncate each other's bytes before the rename.
    try:
        handle, temporary = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
    except OSError:
        return
    try:
        with os.fdopen(handle, "w") as stream:
            stream.write(json.dumps(entry, indent=1))
        os.replace(temporary, path)
    except OSError:
        try:
            os.unlink(temporary)
        except OSError:
            pass


def _read_status_file(path: str) -> Optional[dict[str, Any]]:
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    steps = payload.get("steps")
    summary: dict[str, str] = {}
    if isinstance(steps, dict):
        for name, step in steps.items():
            if isinstance(step, dict) and isinstance(step.get("state"), str):
                summary[str(name)] = step["state"]
    stage = payload.get("stage")
    return {
        "stage": stage if isinstance(stage, str) else None,
        "steps": summary,
        "updated_at": payload.get("updated_at")
        if isinstance(payload.get("updated_at"), (int, float))
        else None,
    }


def registered_jobs(
    session_id: str,
    *,
    directory: Optional[Path] = None,
    now: Optional[float] = None,
    unit_state: Optional[Callable[[str], Optional[str]]] = None,
) -> list[dict[str, Any]]:
    """Jobs registered to ``session_id`` that run or ended moments ago."""
    unit_state = unit_state or systemd_unit_state
    session_id = str(session_id or "").strip()
    if not session_id:
        return []
    directory = directory or registry_dir()
    now = time.time() if now is None else now
    try:
        paths = sorted(directory.glob("*.json"))
    except OSError:
        return []
    jobs: list[dict[str, Any]] = []
    for path in paths:
        entry = _load_entry(path)
        if entry is None:
            continue
        unit = str(entry.get("unit") or "").strip()
        ended_at = _number(entry.get("ended_at"))
        if ended_at is not None and now - ended_at > FINISHED_JOB_RETENTION_SECONDS:
            try:
                path.unlink()
            except OSError:
                pass
            continue
        if entry.get("session_id") != session_id or not unit:
            continue
        started_at = _number(entry.get("started_at"))
        running = False
        if ended_at is None:
            # Nothing recorded yet, so ask systemd. A job the launcher ran
            # records its own end before its unit stops; only an older job,
            # or a runner that was killed outright, is left to this probe.
            state = unit_state(unit)
            if state is None:
                # systemd could not be asked; report nothing rather than guess.
                continue
            running = state in _RUNNING_UNIT_STATES or bool(
                state in ("inactive", "not-found")
                and started_at is not None
                and now - started_at < UNIT_START_GRACE_SECONDS
            )
            if not running:
                with _STAMP_LOCK:
                    recorded = _load_entry(path)
                    if (
                        recorded is not None
                        and _number(recorded.get("ended_at")) is not None
                    ):
                        # The runner, or another poll, recorded the end
                        # after the first read.
                        entry = recorded
                    else:
                        # First sighting after the unit ended (a collected
                        # unit reads "not-found"): stamp the end so it stops
                        # being probed.
                        entry["ended_at"] = now
                        entry["end_state"] = state
                        _write_entry(path, entry)
                ended_at = _number(entry.get("ended_at"))
        if not running and now - float(ended_at) > FINISHED_JOB_VISIBLE_SECONDS:
            continue
        exit_status = None if running else _exit_status(entry)
        end_state = None if running else entry.get("end_state")
        ok: Optional[bool] = None
        if exit_status is not None:
            ok = exit_status == 0
        elif end_state == "failed":
            ok = False
        status_file = entry.get("status_file")
        status = (
            _read_status_file(status_file)
            if isinstance(status_file, str) and status_file
            else None
        )
        jobs.append({
            "id": str(entry.get("id") or unit),
            "kind": "job",
            "title": str(entry.get("title") or unit),
            "running": running,
            "started_at": started_at,
            "ended_at": None if running else ended_at,
            "end_state": end_state,
            "exit_status": exit_status,
            "ok": ok,
            "status": status,
        })
    return jobs
