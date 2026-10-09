#!/usr/bin/env python3
"""Talos steward tool: read the autonomous runtime's ledger and act on it.

Talos binds loopback on Command Center and gates its operator API with a
token file only the runtime, hermes-sync and this host's Hermes agent can
read. This tool is how the Talos conversation's model sees state and makes
bounded changes. Every write goes through the ledger's own rules (contracts,
versions, grants); the tool cannot forge evidence, weaken a contract or
release anything. Prompts, packets and credentials never pass through it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib import error, request

from tools.registry import registry

DEFAULT_URL = "http://127.0.0.1:41005"
DEFAULT_TOKEN_FILE = Path.home() / "coding-projects" / "talos" / ".talos" / "runtime" / "dashboard.token"
TIMEOUT_SECONDS = 20
MAX_RESULT_BYTES = 400_000

READ_ACTIONS = {
    "state": "/api/state",
    "live": "/api/live",
    "sessions": "/api/sessions",
}
WRITE_ACTIONS = {
    "admit": ("/api/tasks/admit", ("project_id", "contract", "dedupe_key")),
    "project_save": ("/api/projects/save", ("id", "repo", "body", "mode")),
    "decision_answer": ("/api/decisions/answer", ("id", "version", "answer")),
    "commitment_action": ("/api/commitments/action", ("id", "action")),
    "control": ("/api/tasks/control", ("id", "version", "action", "reason")),
    "routine_create": ("/api/routines/create", ("project_id", "id", "body")),
    "grant_revoke": ("/api/grants/revoke", ("id",)),
    "hold": ("/api/hold", ("action",)),
}
MANDATE_FIELDS = ("name", "objective", "success_measure", "milestone", "allowed_paths", "daily_calls", "authority", "task_templates", "planning_contract", "facts")


def _base_url() -> str:
    return (os.environ.get("HERMES_TALOS_URL") or DEFAULT_URL).rstrip("/")


def _token() -> str:
    path = Path(os.environ.get("HERMES_TALOS_TOKEN_FILE") or DEFAULT_TOKEN_FILE)
    try:
        token = path.read_text().strip()
    except OSError as exc:
        raise ValueError(f"Talos is not available on this host: cannot read its operator token ({exc.__class__.__name__})") from None
    if len(token) < 32:
        raise ValueError("Talos operator token is invalid")
    return token


def _call(path: str, payload: dict | None = None) -> Any:
    data = json.dumps(payload).encode() if payload is not None else None
    req = request.Request(
        _base_url() + path,
        data=data,
        method="POST" if data is not None else "GET",
        headers={
            "Authorization": "Bearer " + _token(),
            **({"Content-Type": "application/json"} if data is not None else {}),
        },
    )
    try:
        with request.urlopen(req, timeout=TIMEOUT_SECONDS) as response:
            raw = response.read(MAX_RESULT_BYTES + 1)
    except error.HTTPError as exc:
        body = exc.read(4096).decode("utf-8", "replace")
        try:
            detail = json.loads(body).get("error") or body
        except ValueError:
            detail = body
        raise ValueError(f"Talos refused ({exc.code}): {detail}") from None
    except (error.URLError, TimeoutError, OSError) as exc:
        raise ValueError(f"Talos is unreachable: {getattr(exc, 'reason', exc)}") from None
    if len(raw) > MAX_RESULT_BYTES:
        raise ValueError("Talos response exceeded the tool's size limit")
    return json.loads(raw)


def _compact_state(state: dict) -> dict:
    """The dashboard payload minus the raw journal; the model asks for a task when it needs detail."""
    tasks = [
        {k: t.get(k) for k in ("id", "project_id", "state", "waiting", "detail", "generation", "updated", "version")}
        | {"artifact": ((t.get("body") or {}).get("contract") or {}).get("artifact"), "evaluation": bool((t.get("body") or {}).get("evaluation")), "progress": t.get("progress")}
        for t in state.get("tasks", [])
    ]
    projects = [
        {k: p.get(k) for k in ("id", "repo", "mode", "priority", "version")}
        | {k: (p.get("body") or {}).get(k) for k in ("name", "objective", "success_measure", "milestone", "allowed_paths", "daily_calls", "evaluation")}
        for p in state.get("projects", [])
    ]
    return {
        "at": state.get("at"), "timezone": state.get("timezone"), "health": state.get("health"),
        "models": state.get("models"), "projects": projects, "tasks": tasks,
        "decisions": state.get("decisions"), "commitments": state.get("commitments"),
        "routines": [{k: r.get(k) for k in ("id", "project_id", "owner", "enabled", "next_due")} | {"title": (r.get("body") or {}).get("title")} for r in state.get("routines", [])],
        "budgets": state.get("budgets"),
    }


def _project(project_id: str) -> dict:
    if not project_id:
        raise ValueError("mandate requires a project id")
    state = _call("/api/state")
    for project in state.get("projects", []):
        if project.get("id") == project_id:
            return {
                "id": project["id"], "repo": project.get("repo"), "mode": project.get("mode"),
                "priority": project.get("priority"), "version": project.get("version"),
                "mandate": {k: (project.get("body") or {}).get(k) for k in MANDATE_FIELDS},
                "held": bool((state.get("holds") or {}).get(project_id) or (state.get("holds") or {}).get("*")),
            }
    raise ValueError(f"Unknown Talos project '{project_id}'")


def _mandate_save(kwargs: dict) -> Any:
    """Merge the given mandate fields onto the current mandate and save with its version.

    The model never has to resend the whole body, and a concurrent edit is
    refused by Talos's compare-and-swap rather than silently overwritten.
    """
    project_id = str(kwargs.get("id") or "")
    current = _project(project_id)
    expected = kwargs.get("version", current["version"])
    if expected != current["version"]:
        raise ValueError(f"Mandate version changed (now {current['version']}); read it again before saving")
    state = _call("/api/state")
    body = next(p for p in state["projects"] if p["id"] == project_id).get("body") or {}
    fields = kwargs.get("mandate") if isinstance(kwargs.get("mandate"), dict) else {}
    fields = {**fields, **{k: kwargs[k] for k in MANDATE_FIELDS if k in kwargs and kwargs[k] is not None}}
    unknown = [k for k in fields if k not in MANDATE_FIELDS]
    if unknown:
        raise ValueError(f"Unknown mandate fields: {', '.join(unknown)}")
    if not fields and not any(k in kwargs for k in ("mode", "priority")):
        raise ValueError("mandate_save needs at least one mandate field, mode or priority")
    body = {**body, **fields, "authority": fields.get("authority") or f"steward:{kwargs.get('actor') or 'talos-thread'}"}
    payload = {"id": project_id, "repo": kwargs.get("repo") or current["repo"], "version": current["version"],
               "mode": kwargs.get("mode") or current["mode"], "priority": kwargs.get("priority", current["priority"]), "body": body}
    return _call("/api/projects/save", payload)


def talos_tool(action: str, **kwargs: Any) -> str:
    action = str(action or "").strip().lower()
    try:
        if action in READ_ACTIONS:
            result = _call(READ_ACTIONS[action])
            if action == "state":
                result = _compact_state(result)
            return json.dumps({"ok": True, "action": action, "result": result})
        if action == "mandate":
            project = _project(str(kwargs.get("id") or ""))
            return json.dumps({"ok": True, "action": action, "result": project})
        if action == "mandate_save":
            return json.dumps({"ok": True, "action": action, "result": _mandate_save(kwargs)})
        if action == "task":
            task_id = str(kwargs.get("id") or "").strip()
            if not task_id or "/" in task_id:
                raise ValueError("task requires a Talos task id")
            return json.dumps({"ok": True, "action": action, "result": _call("/api/tasks/" + task_id)})
        if action in WRITE_ACTIONS:
            path, required = WRITE_ACTIONS[action]
            payload = {k: v for k, v in kwargs.items() if v is not None}
            if "verb" in payload:
                # The tool's own `action` names the operation; Talos's field of
                # the same name (done/defer/delegate, pause/cancel) arrives as verb.
                payload["action"] = payload.pop("verb")
            missing = [k for k in required if k not in payload]
            if missing:
                raise ValueError(f"{action} requires {', '.join(missing)}")
            return json.dumps({"ok": True, "action": action, "result": _call(path, payload)})
        raise ValueError(
            "Unknown action; use state, live, sessions, task, mandate, mandate_save, hold, admit, "
            "project_save, decision_answer, commitment_action, control, routine_create or grant_revoke"
        )
    except ValueError as exc:
        return json.dumps({"ok": False, "action": action, "error": str(exc)})


TALOS_SCHEMA = {
    "name": "talos",
    "description": (
        "Talos, the autonomous project runtime on Command Center. mandate reads a project's current "
        "mandate (outcome, success measure, milestone, allowed paths, mode, priority, hold); mandate_save "
        "merges changed fields onto it with a version check; hold pauses (verb=pause) or resumes "
        "(verb=resume) all projects or one (scope). Reads: state (projects, tasks, "
        "decisions, commitments, routines, health), live (open work per project with stages and "
        "call summaries), sessions (observed Hermes chats), task (one task's findings, reviews and "
        "verification receipts). Writes: admit a task contract, project_save a mandate, "
        "decision_answer, commitment_action, control a task (pause/cancel/resume), routine_create, "
        "grant_revoke. Every write is validated by the ledger; the tool cannot forge evidence, "
        "weaken a contract or release anything. Use it for every fact about Talos instead of guessing."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["state", "live", "sessions", "task", "mandate", "mandate_save", "hold", "admit",
                         "project_save", "decision_answer", "commitment_action", "control", "routine_create", "grant_revoke"],
            },
            "id": {"type": "string", "description": "Task, project, decision, commitment, routine or grant id, depending on the action."},
            "version": {"type": "integer", "description": "Current record version for decision_answer and control (compare-and-swap)."},
            "project_id": {"type": "string"},
            "repo": {"type": "string", "description": "Absolute repository path for project_save."},
            "mode": {"type": "string", "enum": ["active", "maintain", "paused"]},
            "priority": {"type": "integer", "minimum": 0, "maximum": 100},
            "body": {"type": "object", "description": "Project mandate body for project_save, or routine body for routine_create."},
            "contract": {"type": "object", "description": "Task contract for admit: objective, why_now, artifact, authority, allowed_paths, obligations, checks, limits."},
            "dedupe_key": {"type": "string"},
            "dependencies": {"type": "array", "items": {"type": "string"}},
            "answer": {"type": "object", "description": "For decision_answer: {action, reason}."},
            "action_name": {"type": "string"},
            "action": {"type": "string"},
            "reason": {"type": "string"},
            "evidence": {"type": "string"},
            "defer_until": {"type": "number"},
            "owner": {"type": "string", "enum": ["agent", "human"]},
            "mandate": {"type": "object", "description": "For mandate_save: the fields to change (objective, success_measure, milestone, allowed_paths, daily_calls, name). Unchanged fields are kept."},
            "scope": {"type": "string", "description": "For hold: a project id, or omit for all projects."},
        },
        "required": ["action"],
    },
}
# `action` names the tool action; the commitment/control verb travels as `verb`.
TALOS_SCHEMA["parameters"]["properties"].pop("action_name")
TALOS_SCHEMA["parameters"]["properties"]["verb"] = {
    "type": "string",
    "description": "For commitment_action: done, defer or delegate. For control: the task control action.",
}


def _handler(args: dict, **kwargs: Any) -> str:
    args = dict(args)
    return talos_tool(args.pop("action", ""), **args)


registry.register(
    name="talos",
    toolset="talos",
    schema=TALOS_SCHEMA,
    handler=_handler,
    check_fn=lambda: True,
    emoji="",
    max_result_size_chars=200_000,
)
