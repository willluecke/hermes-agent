#!/usr/bin/env python3
"""Governed Claude Code implementation jobs for the persistent Hermes authority."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

from agent.opus_delegation import (
    ORCHESTRATOR_EFFORT,
    ORCHESTRATOR_MODEL,
    authority_label,
    parent_runtime_from_env,
    validate_parent_authority,
)
from tools.interrupt import is_interrupted
from tools.registry import registry


_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
_COMPANIES = {"3DCarParts", "RegWatch", "Portfolio"}
_WORKSTREAMS = {"Build", "Sales", "Customers", "Growth", "Operations"}
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_MAX_WAIT_SECONDS = 300
_POLL_SECONDS = 2.0
_WORKER_FRESH_SECONDS = 60
_RESULT_CHARS = 80_000


class _SyncError(RuntimeError):
    def __init__(self, message: str, *, status: int = 0) -> None:
        super().__init__(message)
        self.status = status


def _sync_url() -> str:
    return os.environ.get("HERMES_SYNC_URL", "http://127.0.0.1:8643").rstrip("/")


def _key_path() -> Path:
    override = os.environ.get("HERMES_SYNC_KEY_FILE", "").strip()
    return Path(override) if override else Path.home() / ".hermes-api-key"


def _authority() -> dict[str, str]:
    """Resolve the authority that may accept an Opus implementation handoff.

    A managed child (the hermes-tools stdio MCP server codex spawns) is handed
    its parent's non-secret runtime through the env whitelist. When that
    metadata is present it is authoritative: the exact gpt-5.6-sol xhigh
    orchestrator and an eligible openai-codex/gpt-6-astra single-model parent
    are accepted, and any other direct parent is rejected — a config snapshot
    must not vouch for a runtime that is not actually running.

    Paths that carry no runtime metadata (native/non-gateway callers) keep the
    existing config-derived orchestrator authority.
    """
    runtime = parent_runtime_from_env()
    if runtime is not None:
        return validate_parent_authority(runtime)

    from tools.decision_log_tool import _authority as decision_authority

    authority = decision_authority()
    if (
        authority.get("model") != ORCHESTRATOR_MODEL
        or authority.get("effort") != ORCHESTRATOR_EFFORT
    ):
        raise ValueError(
            "Opus delegation requires exact gpt-5.6-sol at xhigh as decision authority"
        )
    return authority


def _configured() -> bool:
    try:
        return _key_path().is_file() and bool(_authority())
    except Exception:
        return False


def _bounded_error_body(handle: Any) -> str:
    try:
        body = handle.read(8_192).decode("utf-8", errors="replace")
        payload = json.loads(body)
        if isinstance(payload, dict) and payload.get("error"):
            return str(payload["error"])[:2_000]
        return body.strip()[:2_000]
    except Exception:
        return ""


def _sync_request(
    method: str,
    path: str,
    payload: Optional[dict[str, Any]] = None,
    *,
    timeout: float = 15.0,
) -> dict[str, Any]:
    if not path.startswith("/management/") or "?" in path or "#" in path:
        raise ValueError("invalid management path")
    try:
        key = _key_path().read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise _SyncError(f"cannot read Hermes sync key: {exc}") from exc
    if not key:
        raise _SyncError("Hermes sync key is empty")

    body = None
    headers = {"Authorization": f"Bearer {key}"}
    if payload is not None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        f"{_sync_url()}{path}",
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            parsed = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = _bounded_error_body(exc)
        raise _SyncError(
            detail or f"Hermes sync returned HTTP {exc.code}", status=exc.code
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise _SyncError(f"Hermes sync request failed: {exc}") from exc
    if not isinstance(parsed, dict):
        raise _SyncError("Hermes sync returned a non-object response")
    return parsed


def _clean_list(value: Any, name: str, *, required: bool = False) -> list[str]:
    if value is None:
        value = []
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array of strings")
    cleaned = [str(item).strip() for item in value if str(item).strip()]
    if required and not cleaned:
        raise ValueError(f"{name} must contain at least one item")
    return cleaned


def _thread_id(session_id: str = "") -> str:
    raw = str(session_id or os.environ.get("HERMES_GATEWAY_SESSION_ID", "")).strip()
    if not raw:
        raise ValueError("Hermes session identity is unavailable; cannot bind worker job")
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", raw).strip("_")
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
    stem = normalized[:46] or "session"
    return f"hermes_{stem}_{digest}"[:64]


def _job_id(
    *,
    thread_id: str,
    project: str,
    title: str,
    specification: str,
    acceptance_checks: list[str],
    constraints: list[str],
    attempt: int,
) -> str:
    canonical = json.dumps(
        {
            "thread_id": thread_id,
            "project": project,
            "title": title,
            "specification": specification,
            "acceptance_checks": acceptance_checks,
            "constraints": constraints,
            "attempt": attempt,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]
    return f"job_{digest}"


def _worker_available() -> bool:
    workers = _sync_request("GET", "/management/workers").get("workers", [])
    now_ms = time.time() * 1_000
    for worker in workers if isinstance(workers, list) else []:
        if not isinstance(worker, dict):
            continue
        if now_ms - float(worker.get("updatedAt") or 0) > _WORKER_FRESH_SECONDS * 1_000:
            continue
        claude = (worker.get("providers") or {}).get("claude") or {}
        if claude.get("available") is True and claude.get("authenticated") is True:
            return True
    return False


def _get_job(job_id: str) -> dict[str, Any]:
    response = _sync_request(
        "GET", f"/management/jobs/{urllib.parse.quote(job_id, safe='')}"
    )
    job = response.get("job")
    if not isinstance(job, dict):
        raise _SyncError("Hermes sync returned no worker job")
    return job


def _get_job_or_none(job_id: str) -> Optional[dict[str, Any]]:
    try:
        return _get_job(job_id)
    except _SyncError as exc:
        if exc.status == 404:
            return None
        raise


def _owned_job(job_id: str, thread_id: str) -> dict[str, Any]:
    if not _ID_RE.fullmatch(job_id):
        raise ValueError("job_id must be a valid Hermes job identifier")
    job = _get_job(job_id)
    if job.get("threadId") != thread_id:
        raise ValueError("worker job does not belong to this Hermes conversation")
    if job.get("provider") != "claude" or job.get("mode") != "implement":
        raise ValueError("worker job is not a governed Claude implementation")
    return job


def _job_result(job: dict[str, Any]) -> dict[str, Any]:
    status = str(job.get("status") or "unknown")
    result = str(job.get("result") or job.get("summary") or "")[:_RESULT_CHARS]
    error = str(job.get("error") or "")[:10_000]
    payload = {
        "ok": status not in {"failed", "cancelled"},
        "status": status,
        "job_id": job.get("id"),
        "thread_id": job.get("threadId"),
        "provider": job.get("provider"),
        "model": job.get("model") or "claude-opus-5",
        "effort": job.get("effort") or "high",
        "project": job.get("project"),
        "worker_id": job.get("workerId"),
        "activity": str(job.get("activity") or "")[:500],
        "progress": str(job.get("progress") or "")[-20_000:],
        "summary": str(job.get("summary") or "")[:10_000],
        "result": result,
        "error": error,
        "branch": job.get("branch") or "",
        "worktree": job.get("worktree") or "",
        "commit": job.get("commit"),
        "base_commit": job.get("baseCommit"),
        "decision_needed": "DECISION_NEEDED" in f"{result}\n{error}".upper(),
        "review_required": status == "completed",
    }
    if status in {"queued", "running"}:
        payload["next_action"] = (
            "Call opus_code_worker with action='status', this job_id, and a "
            "positive wait_seconds to continue waiting in the same Hermes turn."
        )
    elif status == "completed":
        payload["next_action"] = (
            "Hermes must inspect the worktree commit and test evidence against the "
            "accepted specification before recommending promotion."
        )
    elif status == "failed":
        payload["next_action"] = (
            "Hermes must diagnose the evidence or revise the specification; do not "
            "silently substitute another model."
        )
    return payload


def _wait_for_job(job: dict[str, Any], wait_seconds: int) -> dict[str, Any]:
    wait_seconds = max(0, min(int(wait_seconds or 0), _MAX_WAIT_SECONDS))
    deadline = time.monotonic() + wait_seconds
    current = job
    while str(current.get("status") or "") not in _TERMINAL_STATUSES:
        if is_interrupted():
            try:
                current = _sync_request(
                    "PATCH",
                    f"/management/jobs/{urllib.parse.quote(str(current['id']), safe='')}",
                    {"status": "cancelled"},
                ).get("job", current)
            except Exception:
                pass
            result = _job_result(current)
            result.update(
                {
                    "ok": False,
                    "status": "cancelled",
                    "error": "Hermes turn was interrupted; cancellation was requested.",
                }
            )
            return result
        if time.monotonic() >= deadline:
            break
        time.sleep(min(_POLL_SECONDS, max(0.0, deadline - time.monotonic())))
        current = _get_job(str(current["id"]))
    return _job_result(current)


def _brief(
    authority: dict[str, str],
    specification: str,
    acceptance_checks: list[str],
    constraints: list[str],
) -> str:
    lines = [
        f"{authority_label(authority)} has accepted the following bounded "
        "implementation specification.",
        "",
        "Specification:",
        specification,
        "",
        "Acceptance checks:",
        *[f"- {item}" for item in acceptance_checks],
        "",
        "Additional constraints:",
        *([f"- {item}" for item in constraints] or ["- None beyond worker policy."]),
        "",
        "Do not make product, architecture, or policy decisions. If the specification "
        "is materially ambiguous, stop and return DECISION_NEEDED with evidence and options.",
    ]
    return "\n".join(lines)


def _run_job(
    *,
    project: str,
    title: str,
    specification: str,
    acceptance_checks: Any,
    constraints: Any = None,
    company: str = "Portfolio",
    workstream: str = "Build",
    priority: int = 70,
    wait_seconds: int = 240,
    attempt: int = 1,
    session_id: str = "",
) -> dict[str, Any]:
    authority = _authority()
    project = str(project or "").strip()
    title = str(title or "").strip()
    specification = str(specification or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", project):
        raise ValueError("project must be a registered project key")
    if not title or not specification:
        raise ValueError("title and specification are required")
    checks = _clean_list(acceptance_checks, "acceptance_checks", required=True)
    limits = _clean_list(constraints, "constraints")
    if company not in _COMPANIES:
        raise ValueError(f"company must be one of {sorted(_COMPANIES)}")
    if workstream not in _WORKSTREAMS:
        raise ValueError(f"workstream must be one of {sorted(_WORKSTREAMS)}")
    attempt = max(1, min(int(attempt or 1), 99))
    priority = max(0, min(int(priority or 70), 100))
    thread = _thread_id(session_id)
    job_id = _job_id(
        thread_id=thread,
        project=project,
        title=title,
        specification=specification,
        acceptance_checks=checks,
        constraints=limits,
        attempt=attempt,
    )

    existing = _get_job_or_none(job_id)
    if existing is not None:
        if existing.get("threadId") != thread or existing.get("project") != project:
            raise ValueError("deterministic worker job identity collision")
        return _wait_for_job(existing, wait_seconds)
    if not _worker_available():
        raise ValueError("no fresh authenticated Claude worker is available")

    brief = _brief(authority, specification, checks, limits)
    job_payload = {
        "id": job_id,
        "title": title[:500],
        "brief": brief[:50_000],
        "message": brief[:50_000],
        "threadId": thread,
        "chatTitle": title[:500],
        "contextType": "hermes-orchestration",
        "contextId": thread,
        "contextTitle": f"{authority_label(authority)} implementation handoff",
        "contextDetail": (
            f"Authority: {authority['model']} at {authority['effort']}; "
            f"worker: claude-opus-5 at high; attempt: {attempt}"
        ),
        "provider": "claude",
        "model": "claude-opus-5",
        "effort": "high",
        "mode": "implement",
        "project": project,
        "company": company,
        "workstream": workstream,
        "priority": priority,
    }
    try:
        response = _sync_request("POST", "/management/jobs", job_payload)
    except _SyncError:
        # The deterministic ID makes a lost create response recoverable. If the
        # sync store committed before the connection failed, reuse that exact
        # job rather than queueing a second Opus run.
        recovered = _get_job_or_none(job_id)
        if recovered is None:
            raise
        response = {"job": recovered}
    job = response.get("job")
    if not isinstance(job, dict):
        raise _SyncError("Hermes sync did not return the queued worker job")
    return _wait_for_job(job, wait_seconds)


def opus_code_worker_tool(action: str, *, session_id: str = "", **kwargs: Any) -> str:
    """Queue, recover, or cancel a governed Opus 5 implementation job."""
    try:
        _authority()
        normalized = str(action or "").strip().lower()
        thread = _thread_id(session_id)
        if normalized == "run":
            result = _run_job(session_id=session_id, **kwargs)
        elif normalized == "status":
            job = _owned_job(str(kwargs.get("job_id") or ""), thread)
            result = _wait_for_job(job, int(kwargs.get("wait_seconds") or 0))
        elif normalized == "cancel":
            job = _owned_job(str(kwargs.get("job_id") or ""), thread)
            if job.get("status") not in _TERMINAL_STATUSES:
                response = _sync_request(
                    "PATCH",
                    f"/management/jobs/{urllib.parse.quote(str(job['id']), safe='')}",
                    {"status": "cancelled"},
                )
                job = response.get("job", job)
            result = _job_result(job)
        else:
            raise ValueError("action must be 'run', 'status', or 'cancel'")
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)


OPUS_CODE_WORKER_SCHEMA = {
    "name": "opus_code_worker",
    "description": (
        "Delegate an accepted, bounded code implementation to the exact native "
        "Claude Code Opus 5 worker, then recover its durable result for review by "
        "the accepting Hermes authority in this same conversation. Use "
        "action='run' only after Hermes has "
        "resolved product and architecture choices and stated observable acceptance "
        "checks. The worker edits an isolated Git worktree and may create a local "
        "commit, but cannot push, merge, deploy, contact anyone, or write decisions. "
        "If a run remains queued/running, call action='status' with its job_id; use "
        "action='cancel' only for an explicit stop. Increment attempt only for a "
        "deliberate rerun of the same specification."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["run", "status", "cancel"]},
            "project": {
                "type": "string",
                "description": (
                    "Registered command-center project key, such as "
                    "hermes-chat or reccli."
                ),
            },
            "title": {"type": "string"},
            "specification": {
                "type": "string",
                "description": (
                    "Accepted implementation specification; exclude unresolved "
                    "decisions."
                ),
            },
            "acceptance_checks": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Observable behavior and proportional test commands the worker "
                    "must satisfy."
                ),
            },
            "constraints": {"type": "array", "items": {"type": "string"}},
            "company": {"type": "string", "enum": sorted(_COMPANIES)},
            "workstream": {"type": "string", "enum": sorted(_WORKSTREAMS)},
            "priority": {"type": "integer", "minimum": 0, "maximum": 100},
            "wait_seconds": {
                "type": "integer",
                "minimum": 0,
                "maximum": _MAX_WAIT_SECONDS,
                "description": "Bounded wait before returning a durable status checkpoint.",
            },
            "attempt": {"type": "integer", "minimum": 1, "maximum": 99},
            "job_id": {"type": "string"},
        },
        "required": ["action"],
    },
}


registry.register(
    name="opus_code_worker",
    toolset="opus_worker",
    schema=OPUS_CODE_WORKER_SCHEMA,
    handler=lambda args, **kwargs: opus_code_worker_tool(
        session_id=str(kwargs.get("session_id") or ""), **args
    ),
    check_fn=_configured,
    emoji="",
    max_result_size_chars=100_000,
)
