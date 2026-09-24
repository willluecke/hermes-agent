"""Detached background jobs registered to a Hermes Chat conversation.

Long work (a multi-agent build, a render) that must outlive the turn and the
gateway itself runs as a transient systemd user unit started by
``~/.local/bin/hermes-bg``. The launcher writes one registry file per job to
``~/.hermes/background-jobs/<unit>.json`` naming the conversation's session,
a title, and optionally a status file the job keeps current. This module
reads that registry for the app's background-work indicator.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Optional

# A finished job stays listed this long so the indicator can say it ended.
FINISHED_JOB_VISIBLE_SECONDS = 15 * 60
# Registry files of jobs that ended longer ago than this are removed.
FINISHED_JOB_RETENTION_SECONDS = 7 * 24 * 3600

_RUNNING_UNIT_STATES = frozenset({"active", "activating", "deactivating", "reloading"})


def registry_dir() -> Path:
    home = Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser()
    return home / "background-jobs"


def systemd_unit_state(unit: str) -> Optional[str]:
    """The unit's ActiveState, or None when systemd cannot be asked."""
    try:
        completed = subprocess.run(
            ["systemctl", "--user", "show", "--property=ActiveState", "--value", unit],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    state = completed.stdout.strip()
    return state or None


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
        try:
            entry = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(entry, dict):
            continue
        unit = str(entry.get("unit") or "").strip()
        ended_at = entry.get("ended_at")
        if isinstance(ended_at, (int, float)):
            if now - ended_at > FINISHED_JOB_RETENTION_SECONDS:
                try:
                    path.unlink()
                except OSError:
                    pass
                continue
        if entry.get("session_id") != session_id or not unit:
            continue
        state = None if isinstance(ended_at, (int, float)) else unit_state(unit)
        if state is None and not isinstance(ended_at, (int, float)):
            # systemd could not be asked; report nothing rather than guess.
            continue
        running = state in _RUNNING_UNIT_STATES
        if not running and not isinstance(ended_at, (int, float)):
            # First sighting after the unit ended (a collected unit reads
            # "inactive"): stamp the end so it stops being probed.
            ended_at = now
            entry["ended_at"] = now
            entry["end_state"] = state
            try:
                path.write_text(json.dumps(entry, indent=1))
            except OSError:
                pass
        if not running and now - float(ended_at) > FINISHED_JOB_VISIBLE_SECONDS:
            continue
        status_file = entry.get("status_file")
        status = (
            _read_status_file(status_file)
            if isinstance(status_file, str) and status_file
            else None
        )
        started_at = entry.get("started_at")
        jobs.append({
            "id": str(entry.get("id") or unit),
            "kind": "job",
            "title": str(entry.get("title") or unit),
            "running": running,
            "started_at": started_at if isinstance(started_at, (int, float)) else None,
            "ended_at": None if running else ended_at,
            "end_state": None if running else entry.get("end_state", state),
            "status": status,
        })
    return jobs
