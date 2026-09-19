"""Claude Code hook bindings: the live tool-call channel for the Claude lane.

Claude Code executes its tools inside its own process, so the plugin hooks
that watch and gate tool calls on the default loop (``pre_tool_call`` and
``post_tool_call``) never fire while a Claude turn runs; they are replayed
from the transcript afterwards. That is too late for anything that must
reach the model mid-turn: the drift check's steer, or the tool guard's hold.

Claude Code's own hook system is the channel. The runtime starts every
managed ``claude`` process with ``--settings`` carrying a ``PreToolUse`` and
a ``PostToolUse`` command hook (``hermes_cli/claude_hook.py``). The hook
script posts each call to the gateway's ``/v1/hooks/claude`` endpoint with a
per-session bearer token, and the gateway dispatches the plugin hooks:

* ``PreToolUse`` runs ``pre_tool_call``; a ``block`` directive comes back as
  the hook's exit code 2 with the message on stderr, which Claude Code
  shows to the model and refuses the call.
* ``PostToolUse`` runs ``post_tool_call`` with ``steerable=True``; a hook
  result carrying ``message`` comes back the same way, so the model sees
  it right after the call it was triggered by.

The endpoint is not the API key's surface. Each managed process gets a
random token bound here to its Hermes session and agent; an unknown or
retired token is refused. Everything fails open: no endpoint means no hooks
are configured, an unreachable endpoint means the hook script exits 0, and
a dispatch error means the call proceeds.
"""

from __future__ import annotations

import secrets
import sys
import threading
import weakref
from pathlib import Path
from typing import Any, Dict, Optional

HOOK_URL_ENV = "HERMES_HOOK_URL"
HOOK_TOKEN_ENV = "HERMES_HOOK_TOKEN"
HOOK_ROUTE = "/v1/hooks/claude"
HOOK_TIMEOUT_SECONDS = 15

_lock = threading.Lock()
_endpoint: Optional[str] = None
_bindings: Dict[str, Dict[str, Any]] = {}
_LIMIT = 512


def set_hook_endpoint(url: Optional[str]) -> None:
    """Publish (or with ``None`` withdraw) the gateway URL hooks post to."""
    global _endpoint
    with _lock:
        _endpoint = str(url).strip() if url else None


def hook_endpoint() -> Optional[str]:
    with _lock:
        return _endpoint


def hook_endpoint_for(host: str, port: int) -> str:
    """The loopback URL for a listener bound to ``host``:``port``."""
    loopback = "127.0.0.1"
    cleaned = (host or "").strip().strip("[]")
    if cleaned in ("", "0.0.0.0", "::", "*"):
        cleaned = loopback
    elif ":" in cleaned:
        cleaned = f"[{cleaned}]"
    return f"http://{cleaned}:{int(port)}{HOOK_ROUTE}"


def new_hook_token() -> str:
    return secrets.token_urlsafe(24)


def bind_hook_token(token: str, *, session_id: str, agent: Any) -> None:
    """Bind ``token`` to a Hermes session and the agent object serving it.

    Rebinding an existing token updates the agent reference (a resident
    Claude process outlives the agent objects of later turns) and keeps the
    live-call count.
    """
    key = str(token or "")
    if not key:
        return
    with _lock:
        existing = _bindings.get(key) or {}
        _bindings[key] = {
            "session_id": str(session_id or ""),
            "agent": weakref.ref(agent) if agent is not None else None,
            "live_calls": int(existing.get("live_calls") or 0),
        }
        if len(_bindings) > _LIMIT:
            for stale in list(_bindings)[: len(_bindings) - _LIMIT]:
                _bindings.pop(stale, None)


def unbind_hook_token(token: str) -> None:
    with _lock:
        _bindings.pop(str(token or ""), None)


def resolve_hook_token(token: str) -> Optional[Dict[str, Any]]:
    """The binding for ``token``: session id, agent and its current turn id, or None."""
    with _lock:
        binding = _bindings.get(str(token or ""))
        if binding is None:
            return None
        ref = binding.get("agent")
        agent = ref() if ref is not None else None
        session_id = binding["session_id"]
    if agent is None:
        return None
    return {
        "token": str(token),
        "session_id": session_id or (getattr(agent, "session_id", "") or ""),
        "agent": agent,
        "turn_id": str(getattr(agent, "_current_turn_id", "") or ""),
    }


def note_live_call(token: str) -> int:
    """Count one tool call delivered live; returns the count this turn."""
    with _lock:
        binding = _bindings.get(str(token or ""))
        if binding is None:
            return 0
        binding["live_calls"] = int(binding.get("live_calls") or 0) + 1
        return binding["live_calls"]


def live_calls(token: str) -> int:
    with _lock:
        binding = _bindings.get(str(token or ""))
        return int(binding.get("live_calls") or 0) if binding else 0


def reset_live_calls(token: str) -> None:
    with _lock:
        binding = _bindings.get(str(token or ""))
        if binding is not None:
            binding["live_calls"] = 0


def hook_script_path() -> Path:
    return Path(__file__).resolve().with_name("claude_hook.py")


def hook_settings(python: Optional[str] = None, script: Optional[Path] = None) -> Dict[str, Any]:
    """The ``--settings`` document that wires both hooks to the script."""
    command = f'"{python or sys.executable}" "{script or hook_script_path()}"'
    entry = [{"hooks": [{"type": "command", "command": command, "timeout": HOOK_TIMEOUT_SECONDS}]}]
    return {"hooks": {"PreToolUse": entry, "PostToolUse": entry}}


def hook_environment(token: str, url: Optional[str] = None) -> Dict[str, str]:
    """The environment the managed process (and so its hook commands) needs."""
    endpoint = url or hook_endpoint()
    if not endpoint or not token:
        return {}
    return {HOOK_URL_ENV: endpoint, HOOK_TOKEN_ENV: str(token)}


__all__ = [
    "HOOK_ROUTE",
    "HOOK_TOKEN_ENV",
    "HOOK_URL_ENV",
    "bind_hook_token",
    "hook_endpoint",
    "hook_endpoint_for",
    "hook_environment",
    "hook_script_path",
    "hook_settings",
    "live_calls",
    "new_hook_token",
    "note_live_call",
    "reset_live_calls",
    "resolve_hook_token",
    "set_hook_endpoint",
    "unbind_hook_token",
]
