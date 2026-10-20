"""Jim's own scheduled jobs: the cron entries that run in his voice.

Will, 2026-10-19: "jim should be able to change anything that you can change",
said of the morning-plan cron job. The steward has no shell or file tools, so
this tool is his write path to those jobs and nothing else: a job is his when
its record carries ``owner: "jim"``; he can read them and change their prompt,
schedule, name or enabled flag. Every write backs up jobs.json first, goes
through cron.jobs.update_job (which parses the schedule and refreshes the next
run), passes the same prompt threat scan every cron prompt passes, and is
appended to an audit log. Other jobs, other files and other fields are out of
reach, and a bound review session never sees this tool at all.
"""
from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.registry import registry

OWNER = "jim"
PROMPT_LIMIT = 24_000
EDITABLE = ("prompt", "schedule", "enabled", "name")
VIEW = ("id", "name", "owner", "schedule", "schedule_display", "enabled", "state", "model", "provider",
        "updated_at", "next_run_at", "last_run_at", "last_status", "prompt")


def _owned(job: dict) -> bool:
    return isinstance(job, dict) and job.get("owner") == OWNER


def _view(job: dict) -> dict:
    return {key: job.get(key) for key in VIEW}


def _backup(jobs_file: Path) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = jobs_file.with_name(f"{jobs_file.name}.bak-jim-{stamp}")
    if jobs_file.exists():
        shutil.copy2(jobs_file, target)
    return str(target)


def _audit(jobs_file: Path, record: dict) -> None:
    with (jobs_file.parent / "jim-job-edits.log").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def steward_jobs(action: str = "", **kwargs: Any) -> str:
    from cron import jobs as store

    try:
        if action == "jobs":
            if kwargs:
                raise ValueError("jobs takes no arguments")
            owned = [_view(job) for job in store.load_jobs() if _owned(job)]
            return json.dumps({"ok": True, "action": action, "result": {"jobs": owned}}, ensure_ascii=False)
        if action == "job_save":
            job_id = kwargs.pop("id", None)
            expected = kwargs.pop("expected_updated_at", None)
            reason = kwargs.pop("reason", None)
            unknown = sorted(set(kwargs) - set(EDITABLE))
            if unknown:
                raise ValueError("job_save changes only " + ", ".join(EDITABLE) + "; not " + ", ".join(unknown))
            if not kwargs:
                raise ValueError("job_save changes at least one of " + ", ".join(EDITABLE))
            if not isinstance(reason, str) or not reason.strip() or len(reason) > 500:
                raise ValueError("job_save needs a bounded reason: what Will asked for")
            job = next((j for j in store.load_jobs() if isinstance(j, dict) and j.get("id") == job_id), None)
            if job is None or not _owned(job):
                raise ValueError("Not one of Jim's jobs; read them with jobs first")
            if expected is not None and job.get("updated_at") != expected:
                raise ValueError("The job changed since it was read; read it again before saving")
            updates: dict[str, Any] = {}
            if "prompt" in kwargs:
                prompt = kwargs["prompt"]
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError("prompt must be non-empty text")
                if len(prompt) > PROMPT_LIMIT:
                    raise ValueError(f"prompt exceeds {PROMPT_LIMIT} characters")
                from tools.cronjob_tools import _scan_cron_prompt
                blocked = _scan_cron_prompt(prompt)
                if blocked:
                    raise ValueError(blocked)
                updates["prompt"] = prompt
            if "schedule" in kwargs:
                schedule = kwargs["schedule"]
                if not isinstance(schedule, str) or not schedule.strip():
                    raise ValueError("schedule is a cron expression such as '30 6 * * *'")
                updates["schedule"] = store.parse_schedule(schedule.strip())
            if "enabled" in kwargs:
                if not isinstance(kwargs["enabled"], bool):
                    raise ValueError("enabled is true or false")
                updates["enabled"] = kwargs["enabled"]
            if "name" in kwargs:
                name = kwargs["name"]
                if not isinstance(name, str) or not name.strip() or len(name) > 120:
                    raise ValueError("name is up to 120 characters")
                updates["name"] = name.strip()
            updates["updated_at"] = datetime.now(timezone.utc).astimezone().isoformat()
            jobs_file = Path(store._current_cron_store().jobs_file)
            backup = _backup(jobs_file)
            updated = store.update_job(job_id, updates)
            if updated is None:
                raise ValueError("The job disappeared while saving")
            _audit(jobs_file, {"at": updates["updated_at"], "job": job_id, "fields": sorted(k for k in updates if k != "updated_at"),
                               "reason": reason.strip(), "backup": backup})
            return json.dumps({"ok": True, "action": action, "result": _view(updated), "backup": backup}, ensure_ascii=False)
        raise ValueError("Unknown action; use jobs or job_save")
    except ValueError as exc:
        return json.dumps({"ok": False, "action": action, "error": str(exc)})


STEWARD_JOBS_SCHEMA = {
    "name": "steward_jobs",
    "description": (
        "Jim's own scheduled jobs: the cron entries that run in his voice (owner jim), such as the daily "
        "morning plan. jobs lists them with their prompt, schedule and last run. job_save(id, reason, "
        "prompt?, schedule?, enabled?, name?, expected_updated_at?) changes one; the prompt is what the job "
        "does each run, the schedule is a cron expression in the job's own timezone. Every save backs up "
        "jobs.json, is audited, and takes effect on the next run. No other job, file or field can be touched."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["jobs", "job_save"]},
            "id": {"type": "string", "description": "The job id from jobs."},
            "reason": {"type": "string", "maxLength": 500, "description": "For job_save: what Will asked for."},
            "prompt": {"type": "string", "maxLength": PROMPT_LIMIT, "description": "The full new prompt text."},
            "schedule": {"type": "string", "description": "Cron expression, five fields, e.g. '30 6 * * *'."},
            "enabled": {"type": "boolean"},
            "name": {"type": "string", "maxLength": 120},
            "expected_updated_at": {"type": "string", "description": "The updated_at seen in jobs; the save is refused if it changed."},
        },
        "required": ["action"],
    },
}


def _handler(args: dict, **kwargs: Any) -> str:
    args = dict(args)
    return steward_jobs(args.pop("action", ""), **args)


registry.register(
    name="steward_jobs",
    toolset="talos",
    schema=STEWARD_JOBS_SCHEMA,
    handler=_handler,
    check_fn=lambda: True,
    emoji="",
    max_result_size_chars=200_000,
)
