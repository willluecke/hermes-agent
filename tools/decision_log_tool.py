#!/usr/bin/env python3
"""Append-only decision ledger for the persistent Codex authority."""

from __future__ import annotations

import json
import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from hermes_constants import get_hermes_home
from tools.registry import registry

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows only
    fcntl = None

try:
    import msvcrt
except ImportError:  # pragma: no cover - Unix only
    msvcrt = None


_CONFIDENCE = {"low", "medium", "high"}
_REVERSIBILITY = {"reversible", "costly", "irreversible"}
_STATUS = {"proposed", "accepted", "deferred", "superseded"}


def _ledger_path() -> Path:
    override = os.environ.get("HERMES_DECISION_LEDGER", "").strip()
    return Path(override) if override else Path(get_hermes_home()) / "decision-ledger.jsonl"


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+b") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        elif msvcrt is not None:  # pragma: no cover - Windows only
            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"0")
                lock_file.flush()
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            elif msvcrt is not None:  # pragma: no cover - Windows only
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)


def _authority() -> dict[str, str]:
    from hermes_cli.config import load_config

    config = load_config()
    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
    agent_cfg = config.get("agent", {}) if isinstance(config, dict) else {}
    provider = str(model_cfg.get("provider") or "").strip().lower()
    model = str(model_cfg.get("default") or model_cfg.get("model") or "").strip()
    effort = str(
        agent_cfg.get("reasoning_effort") or model_cfg.get("reasoning") or ""
    ).strip().lower()
    runtime = str(model_cfg.get("openai_runtime") or "").strip().lower()
    exact = model_cfg.get("openai_runtime_require_exact") is True
    if (
        provider not in {"openai", "openai-codex"}
        or runtime != "codex_app_server"
        or not exact
        or not model
        or not effort
    ):
        raise ValueError(
            "decision recording requires an exact OpenAI/Codex app-server authority"
        )
    return {
        "provider": provider,
        "model": model,
        "effort": effort,
        "runtime": runtime,
    }


def _clean_string_list(value: Any, name: str, *, required: bool = False) -> list[str]:
    if value is None and not required:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array of strings")
    cleaned = [str(item).strip() for item in value if str(item).strip()]
    if required and not cleaned:
        raise ValueError(f"{name} must contain at least one item")
    return cleaned


def _record(
    *,
    decision: str,
    rationale: str,
    evidence: list[str],
    confidence: str,
    reversibility: str,
    scope: str,
    alternatives: Optional[list[str]] = None,
    dissent: Optional[list[str]] = None,
    supersedes: str = "",
    status: str = "accepted",
) -> dict[str, Any]:
    required_strings = {
        "decision": decision,
        "rationale": rationale,
        "scope": scope,
    }
    for name, value in required_strings.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} is required")
    confidence = str(confidence or "").strip().lower()
    reversibility = str(reversibility or "").strip().lower()
    status = str(status or "accepted").strip().lower()
    if confidence not in _CONFIDENCE:
        raise ValueError(f"confidence must be one of {sorted(_CONFIDENCE)}")
    if reversibility not in _REVERSIBILITY:
        raise ValueError(f"reversibility must be one of {sorted(_REVERSIBILITY)}")
    if status not in _STATUS:
        raise ValueError(f"status must be one of {sorted(_STATUS)}")

    entry = {
        "id": f"decision_{uuid.uuid4().hex[:16]}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "scope": scope.strip(),
        "decision": decision.strip(),
        "rationale": rationale.strip(),
        "evidence": _clean_string_list(evidence, "evidence", required=True),
        "alternatives": _clean_string_list(alternatives, "alternatives"),
        "dissent": _clean_string_list(dissent, "dissent"),
        "confidence": confidence,
        "reversibility": reversibility,
        "supersedes": str(supersedes or "").strip() or None,
        "authority": _authority(),
    }
    path = _ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    with _exclusive_lock(path):
        fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            offset = 0
            while offset < len(encoded):
                offset += os.write(fd, encoded[offset:])
            os.fsync(fd)
        finally:
            os.close(fd)
    return entry


def _list_entries(limit: int = 20, scope: str = "") -> list[dict[str, Any]]:
    path = _ledger_path()
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except (TypeError, ValueError):
                continue
            if not isinstance(item, dict):
                continue
            if scope and item.get("scope") != scope:
                continue
            entries.append(item)
    return entries[-max(1, min(int(limit or 20), 200)):]


def decision_log_tool(action: str, **kwargs: Any) -> str:
    """Record or list durable decisions as structured JSON."""
    try:
        normalized = str(action or "").strip().lower()
        if normalized == "record":
            return json.dumps({"ok": True, "entry": _record(**kwargs)}, ensure_ascii=False)
        if normalized == "list":
            return json.dumps(
                {
                    "ok": True,
                    "entries": _list_entries(
                        limit=kwargs.get("limit", 20), scope=str(kwargs.get("scope") or "")
                    ),
                },
                ensure_ascii=False,
            )
        raise ValueError("action must be 'record' or 'list'")
    except Exception as exc:
        return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)


DECISION_LOG_SCHEMA = {
    "name": "decision_log",
    "description": (
        "Record or list append-only durable decisions. Record only decisions that "
        "should survive the current conversation, with evidence and uncertainty."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["record", "list"]},
            "decision": {"type": "string"},
            "rationale": {"type": "string"},
            "evidence": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "string", "enum": sorted(_CONFIDENCE)},
            "reversibility": {"type": "string", "enum": sorted(_REVERSIBILITY)},
            "scope": {"type": "string"},
            "alternatives": {"type": "array", "items": {"type": "string"}},
            "dissent": {"type": "array", "items": {"type": "string"}},
            "supersedes": {"type": "string"},
            "status": {"type": "string", "enum": sorted(_STATUS)},
            "limit": {"type": "integer", "minimum": 1, "maximum": 200},
        },
        "required": ["action"],
    },
}


registry.register(
    name="decision_log",
    toolset="decision_log",
    schema=DECISION_LOG_SCHEMA,
    handler=lambda args, **kwargs: decision_log_tool(**args),
    check_fn=lambda: True,
    emoji="",
)
