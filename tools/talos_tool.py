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

import hashlib
import json
import os
from pathlib import Path
from typing import Any
from urllib import error, request, parse
import re

from tools.registry import registry

# Every-turn final review (docs/turn-final-reviews.md in the Talos repo).
# These actions exist only inside the trusted review invocation: the
# restricted MCP server calls review_tool() with the gateway-bound context,
# never through the model-facing talos_tool() path below.
REVIEW_ACTIONS = ("review", "review_content", "review_ack")
REVIEW_CONTENT_LIMIT = 64_000  # bytes of complete final content one review may read
REVIEW_TEXT_LIMIT = 1500
ASSESSMENTS = ("aligned", "drift", "needs_action", "insufficient")
NEEDED_ACTIONS = ("none", "clarification", "approval", "propose_followup")
COVERAGES = ("complete", "insufficient")
_HASH = re.compile(r"^[0-9a-f]{64}$")

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
STEWARD_WRITES = {
    "direction_save": ("directions/save", ("id", "version", "project_id", "patch", "source", "request_key")),
    "interview_save": ("interviews/save", ("project_id", "version", "focus_id", "mode", "question", "source", "request_key")),
    "dispatch": ("delegations/dispatch", ("direction_id", "version", "request_key", "lane", "reason", "scope", "authorization", "work_class")),
    "delegation_control": ("delegations/control", ("id", "version", "verb", "reason", "source", "request_key")),
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
        "budgets": state.get("budgets"), "holds": state.get("holds"),
        "direction_counts": state.get("direction_counts"), "delegation_counts": state.get("delegation_counts"),
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
    body = {**body, **fields}
    payload = {"id": project_id, "repo": kwargs.get("repo") or current["repo"], "version": current["version"],
               "mode": kwargs.get("mode") or current["mode"], "priority": kwargs.get("priority", current["priority"]), "body": body}
    return _call("/api/projects/save", payload)


def talos_tool(action: str, **kwargs: Any) -> str:
    action = str(action or "").strip().lower()
    try:
        if action in REVIEW_ACTIONS:
            raise ValueError("Final review actions run only inside a trusted internal review invocation")
        if action in ("directions", "direction", "delegation"):
            key = "project_id" if action == "directions" else "id"
            value = kwargs.get(key)
            if set(kwargs) != {key} or not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value):
                raise ValueError(f"{action} requires only a valid {key}")
            path = ("/api/directions?" + parse.urlencode({key: value}) if action == "directions"
                    else "/api/" + {"direction": "directions", "delegation": "delegations"}[action] + "/" + parse.quote(value, safe=""))
            return json.dumps({"ok": True, "action": action, "result": _call(path)})
        if action in STEWARD_WRITES:
            path, required = STEWARD_WRITES[action]
            missing = [key for key in required if key not in kwargs]
            if missing:
                raise ValueError(f"{action} requires {', '.join(missing)}")
            # Preserve explicit null focus/question and the delegation verb. The
            # frontier reducer owns strict field and transition validation.
            return json.dumps({"ok": True, "action": action, "result": _call("/api/" + path, kwargs)})
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
TALOS_SCHEMA["description"] += (
    " directions(project_id) reads a direction tree and interview; direction(id) reads one leaf and execution history. "
    "direction_save(id,version,project_id,patch,source,request_key) saves a CAS patch with source={conversation_id,message_id}. "
    "interview_save(project_id,version,focus_id,mode,question,source,request_key) persists one question. "
    "dispatch(direction_id,version,request_key,lane,reason,scope,authorization,work_class) starts authorized bounded work: "
    "scope={allowed_paths,access:read|write}, authorization={source,scope}; conversation adds target={conversation_id} or "
    "{project,title,model,effort} and goal={text,max_turns,model,effort}; ledger adds contract,dedupe_key,dependencies. "
    "Peripheral work requires spec_ref. The reconciler owns creation and kickoff; never also send. "
    "delegation(id) reads status; delegation_control(id,version,verb,reason,source,request_key,max_turns?) controls it. "
    "Resume max_turns is an explicitly authorized total, never an increment."
)
_props = TALOS_SCHEMA["parameters"]["properties"]
_props["action"] = {"type": "string", "enum": list(READ_ACTIONS) + list(WRITE_ACTIONS) + list(STEWARD_WRITES) + ["task", "mandate", "mandate_save", "directions", "direction", "delegation"]}
for _field in ("patch", "source", "authorization", "target", "goal"):
    _props[_field] = {"type": "object"}
for _field in ("request_key", "direction_id", "spec_ref", "supersedes_id"):
    _props[_field] = {"type": "string"}
_props.update({
    "lane": {"type": "string", "enum": ["conversation", "ledger"]},
    "work_class": {"type": "string", "enum": ["core", "peripheral"]},
    "focus_id": {"type": ["string", "null"]}, "question": {"type": ["object", "null"]},
    "mode": {"type": "string", "enum": ["active", "maintain", "paused"]},
    "scope": {"oneOf": [{"type": "string"}, {"type": "object", "required": ["allowed_paths", "access"], "properties": {"allowed_paths": {"type": "array", "items": {"type": "string"}}, "access": {"enum": ["read", "write"]}}, "additionalProperties": False}]},
    "max_turns": {"type": "integer", "minimum": 1, "maximum": 500},
})
# `action` names the tool action; the commitment/control verb travels as `verb`.
TALOS_SCHEMA["parameters"]["properties"].pop("action_name")
TALOS_SCHEMA["parameters"]["properties"]["verb"] = {
    "type": "string",
    "description": "For commitment_action: done, defer or delegate. For control: the task control action.",
}


REVIEW_SCHEMA = {
    "name": "talos",
    "description": (
        "Internal final-review mode: only the assigned delegated final can be read and acknowledged. "
        "review reads the assigned review record once (the final's text, availability, terminal status, "
        "completeness and digest, plus the original delegation context). review_content, only when the "
        "record's final has complete=false, reads its complete immutable content once; the trusted wrapper "
        "fetches the recorded reference and verifies its SHA-256 and byte count. review_ack acknowledges once "
        "with assessment (aligned|drift|needs_action|insufficient), a bounded reason, needed_action "
        "(none|clarification|approval|propose_followup), action_detail (omit when needed_action is none) and "
        "coverage (complete|insufficient); identity, digests and provenance come from the trusted context. "
        "coverage=complete is accepted only when the complete content was in the record or verified by "
        "review_content; a blank, unavailable, provider-error or unread final is assessed insufficient. "
        "After review_ack, end the turn."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": list(REVIEW_ACTIONS)},
            "id": {"type": "string", "description": "The assigned review id (final_<digest>)."},
            "assessment": {"type": "string", "enum": list(ASSESSMENTS)},
            "reason": {"type": "string", "description": "At most 1500 characters."},
            "needed_action": {"type": "string", "enum": list(NEEDED_ACTIONS)},
            "action_detail": {"type": "string", "description": "Omit when needed_action is none; otherwise at most 1500 characters."},
            "coverage": {"type": "string", "enum": list(COVERAGES)},
        },
        "required": ["action"],
    },
}


def _review_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > REVIEW_TEXT_LIMIT:
        raise ValueError(f"{field} must be nonempty text of at most {REVIEW_TEXT_LIMIT} characters")
    return value


def _check_record(record: Any, context: dict) -> dict:
    """The record must be the assigned final, archived, delivered and unchanged."""
    if not isinstance(record, dict) or record.get("id") != context["id"]:
        raise ValueError("Talos returned a record for another review")
    if record.get("content_hash") != context["content_hash"] or not isinstance(record.get("body"), dict):
        raise ValueError("The assigned packet digest does not match the archived Talos record; this review cannot proceed")
    final = record["body"].get("final")
    if not isinstance(final, dict) or not {"availability", "text", "complete", "sha256", "bytes", "reference"} <= set(final):
        raise ValueError("The archived final packet is incomplete")
    if not isinstance(final["sha256"], str) or not _HASH.fullmatch(final["sha256"]) or type(final["complete"]) is not bool:
        raise ValueError("The archived final packet is incomplete")
    delivery = record.get("delivery")
    if not isinstance(delivery, dict) or delivery.get("queue_id") != context["queue_id"]:
        raise ValueError("The assigned review has no matching durable delivery; it cannot be acknowledged from this invocation")
    return record


def _sync_read(path: str, size: int) -> bytes:
    """One authenticated read of a recorded complete reference from hermes-sync."""
    from tools.hermes_conversation_tool import _base_url as sync_url, _key as sync_key

    req = request.Request(sync_url() + path, method="GET",
                          headers={"Authorization": "Bearer " + sync_key(), "Accept": "text/plain"})
    try:
        with request.urlopen(req, timeout=TIMEOUT_SECONDS) as response:
            return response.read(size + 1)
    except error.HTTPError as exc:
        body = exc.read(4096).decode("utf-8", "replace")
        try:
            detail = json.loads(body).get("error") or ""
        except ValueError:
            detail = ""
        raise ValueError(f"hermes-sync refused the complete reference ({exc.code}): {str(detail)[:300]}") from None
    except (error.URLError, TimeoutError, OSError) as exc:
        raise ValueError(f"hermes-sync is unreachable: {getattr(exc, 'reason', exc)}") from None


def _review_content(review: Any, context: dict) -> dict:
    final = review.record["body"]["final"]
    reference = final.get("reference")
    if final["complete"] is not False or not isinstance(reference, dict) or set(reference) != {"path", "sha256", "bytes"}:
        raise ValueError("The record has no complete reference to read")
    path, digest, size = reference["path"], reference["sha256"], reference["bytes"]
    if (not isinstance(path, str) or not path.startswith("/conversations/") or "content=complete" not in path
            or "://" in path or any(c in path for c in " \r\n")):
        raise ValueError("The recorded reference is not an immutable final address")
    if digest != final["sha256"] or type(size) is not int or size != final["bytes"] or size <= 0:
        raise ValueError("The recorded reference does not address this final's digest and length")
    if size > REVIEW_CONTENT_LIMIT:
        raise ValueError(f"The complete final is {size} bytes; one review may read at most {REVIEW_CONTENT_LIMIT}. "
                         "Acknowledge coverage=insufficient and name the unread complete content.")
    review.attempted.add("review_content")
    raw = _sync_read(path, size)
    if len(raw) != size:
        raise ValueError(f"Complete content length mismatch: expected {size} bytes, received {len(raw)}; coverage is insufficient")
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("Complete content digest mismatch; coverage is insufficient")
    try:
        text = raw.decode("utf-8", "strict")
    except UnicodeDecodeError:
        raise ValueError("Complete content is not valid UTF-8; coverage is insufficient") from None
    review.content_verified = True
    return {"id": context["id"], "sha256": digest, "bytes": size, "text": text}


def _review_ack(review: Any, context: dict, kwargs: dict) -> dict:
    final = review.record["body"]["final"]
    unknown = set(kwargs) - {"id", "assessment", "reason", "needed_action", "action_detail", "coverage",
                             "content_hash", "queue_id", "review_run_id", "source", "final_sha256"}
    if unknown:
        raise ValueError("review_ack does not accept: " + ", ".join(sorted(unknown)))
    assessment, needed_action, coverage = kwargs.get("assessment"), kwargs.get("needed_action"), kwargs.get("coverage")
    if assessment not in ASSESSMENTS:
        raise ValueError("assessment must be one of " + ", ".join(ASSESSMENTS))
    if needed_action not in NEEDED_ACTIONS:
        raise ValueError("needed_action must be one of " + ", ".join(NEEDED_ACTIONS))
    if coverage not in COVERAGES:
        raise ValueError("coverage must be complete or insufficient")
    reason = _review_text(kwargs.get("reason"), "reason")
    action_detail = kwargs.get("action_detail")
    if needed_action == "none":
        if action_detail is not None:
            raise ValueError("action_detail must be omitted when needed_action is none")
    else:
        action_detail = _review_text(action_detail, "action_detail")
    if assessment == "aligned" and needed_action != "none":
        raise ValueError("An aligned review has no needed action")
    if assessment in {"drift", "needs_action"} and needed_action == "none":
        raise ValueError("A drift or needs_action review must propose a next action")
    if coverage == "complete" and not (final["complete"] is True or review.content_verified):
        raise ValueError("coverage=complete requires the complete final content: this record is bounded and "
                         "review_content has not verified it. Read it first, or acknowledge coverage=insufficient.")
    if (coverage == "insufficient" or final["availability"] != "present") and assessment != "insufficient":
        raise ValueError("A blank, unavailable, provider-error or insufficiently read final must be assessed insufficient")
    payload = {
        "id": context["id"], "content_hash": context["content_hash"], "queue_id": context["queue_id"],
        "source": context["source"], "review_run_id": context["review_run_id"],
        "assessment": assessment, "reason": reason, "needed_action": needed_action,
        "action_detail": action_detail, "coverage": coverage, "final_sha256": final["sha256"],
    }
    # From here the acknowledgement counts as attempted, whatever the outcome:
    # an uncertain receipt is recovered by the controller, never re-sent here.
    review.attempted.add("review_ack")
    try:
        return _call("/api/delegation-reviews/ack", payload)
    except ValueError as exc:
        raise ValueError(f"Acknowledgement outcome uncertain: {exc}. Do not retry; the controller recovers "
                         "the receipt. Report this and end the turn.") from None


def review_tool(review: Any, action: str = "", **kwargs: Any) -> str:
    """Trusted review entry point for the restricted MCP server.

    ``review`` is the turn's StewardTools instance: ``review.review`` is the
    gateway-bound context, and the fetched record plus verification state live
    on it. Provenance, identity and digests always come from that context.
    """
    action = str(action or "").strip().lower()
    try:
        context = review.review
        if action in ("review_content", "review_ack") and review.record is None:
            raise ValueError("Read the assigned record with review first")
        if action == "review":
            if set(kwargs) - {"id"}:
                raise ValueError("review accepts only the assigned id")
            review.attempted.add("review")
            record = _check_record(_call("/api/delegation-reviews/" + parse.quote(context["id"], safe="")), context)
            review.record = record
            final = record["body"]["final"]
            return json.dumps({"ok": True, "action": action, "result": record,
                               "complete_content": "in_record" if final["complete"] else "read_with_review_content",
                               "availability": final["availability"]})
        if action == "review_content":
            if set(kwargs) - {"id"}:
                raise ValueError("review_content accepts only the assigned id")
            return json.dumps({"ok": True, "action": action, "result": _review_content(review, context)}, ensure_ascii=False)
        if action == "review_ack":
            return json.dumps({"ok": True, "action": action, "result": _review_ack(review, context, kwargs)})
        raise ValueError("Internal review mode permits only review, review_content and review_ack")
    except ValueError as exc:
        return json.dumps({"ok": False, "action": action, "error": str(exc)})


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
