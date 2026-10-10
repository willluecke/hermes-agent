#!/usr/bin/env python3
"""Read and drive existing Hermes Chat conversations from an agent.

A Hermes conversation is a durable session: the gateway resumes it from its
transcript whenever a new turn starts, so a thread needs no live process to
be messaged. This tool reaches the hermes-sync store on Command Center with
its key and uses the same queued-turn path the app uses, so a message sent
here lands in the thread as a real user turn, the session resumes with its
full context, and the reply is archived in the conversation. Reads return
the stored messages; nothing here bypasses the conversation's own rules.
"""
from __future__ import annotations

import json
import os
import secrets
import time
from pathlib import Path
from typing import Any
from urllib import error, parse, request

from tools.registry import registry

DEFAULT_SYNC_URL = "http://127.0.0.1:8643"
DEFAULT_KEY_FILE = Path.home() / ".hermes-api-key"
TIMEOUT_SECONDS = 20
MAX_TEXT_CHARS = 12_000
MAX_REPLY_CHARS = 16_000
MAX_WAIT_SECONDS = 600
POLL_SECONDS = 3
EFFORTS = ("default", "none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra", "ultracode")


def _base_url() -> str:
    return (os.environ.get("HERMES_SYNC_URL") or DEFAULT_SYNC_URL).rstrip("/")


def _key() -> str:
    explicit = os.environ.get("HERMES_SYNC_KEY")
    if explicit:
        return explicit.strip()
    try:
        return Path(os.environ.get("HERMES_API_KEY_FILE") or DEFAULT_KEY_FILE).read_text().strip()
    except OSError as exc:
        raise ValueError(f"Hermes sync key unavailable ({exc.__class__.__name__})") from None


def _call(path: str, payload: dict | None = None, method: str | None = None) -> Any:
    data = json.dumps(payload).encode() if payload is not None else None
    req = request.Request(
        _base_url() + path,
        data=data,
        method=method or ("POST" if data is not None else "GET"),
        headers={
            "Authorization": "Bearer " + _key(),
            **({"Content-Type": "application/json"} if data is not None else {}),
        },
    )
    try:
        with request.urlopen(req, timeout=TIMEOUT_SECONDS) as response:
            return json.loads(response.read())
    except error.HTTPError as exc:
        body = exc.read(4096).decode("utf-8", "replace")
        try:
            detail = json.loads(body).get("error") or body
        except ValueError:
            detail = body
        raise ValueError(f"hermes-sync refused ({exc.code}): {detail}") from None
    except (error.URLError, TimeoutError, OSError) as exc:
        raise ValueError(f"hermes-sync is unreachable: {getattr(exc, 'reason', exc)}") from None


def _conversation(conversation_id: str) -> dict:
    if not conversation_id or "/" in conversation_id:
        raise ValueError("a conversation id is required")
    payload = _call("/conversations/" + parse.quote(conversation_id, safe=""))
    # The store wraps the record ({conversation, window, activityCounts}); the
    # window holds the most recent messages, which is where a new turn lands.
    return payload.get("conversation") if isinstance(payload.get("conversation"), dict) else payload


def _summary(conversation: dict) -> dict:
    messages = conversation.get("messages") or []
    return {
        "id": conversation.get("id"),
        "title": conversation.get("title"),
        "project": conversation.get("project"),
        "model": conversation.get("model"),
        "effort": conversation.get("effort"),
        "harness": conversation.get("harness"),
        "updatedAt": conversation.get("updatedAt"),
        "messages": len(messages),
        "automation": bool(conversation.get("automation")),
    }


def _list(project: str = "", limit: int = 20) -> dict:
    state = _call("/state")
    # The store keys conversations by id; older snapshots used a list.
    raw = state.get("conversations") or {}
    rows = list(raw.values()) if isinstance(raw, dict) else raw
    items = [
        _summary(c) for c in rows
        if isinstance(c, dict) and (not project or c.get("project") == project)
    ]
    items.sort(key=lambda item: item.get("updatedAt") or 0, reverse=True)
    return {"conversations": items[: max(1, min(int(limit or 20), 100))]}


def _trim(text: Any, limit: int) -> str:
    text = text if isinstance(text, str) else ""
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _read(conversation_id: str, tail: int = 8) -> dict:
    conversation = _conversation(conversation_id)
    messages = conversation.get("messages") or []
    tail = max(1, min(int(tail or 8), 40))
    return {
        **_summary(conversation),
        "tail": [
            {"role": m.get("role"), "content": _trim(m.get("content"), 4000), "id": m.get("id")}
            for m in messages[-tail:] if isinstance(m, dict)
        ],
    }


def _reply_after(conversation_id: str, turn_id: str) -> dict | None:
    conversation = _conversation(conversation_id)
    messages = conversation.get("messages") or []
    for index, message in enumerate(messages):
        if isinstance(message, dict) and message.get("role") == "user" and message.get("clientTurnId") == turn_id:
            following = messages[index + 1] if index + 1 < len(messages) else None
            if isinstance(following, dict) and following.get("role") == "assistant":
                content = following.get("content") or ""
                return {"content": content, "terminal": following.get("terminalState"), "runId": following.get("runId")}
            return None
    return None


def _runtime_for(conversation: dict, model: str = "", effort: str = "") -> dict:
    """The thread's saved runtime, with optional one-turn overrides.

    An override stays inside the thread's harness (the part before '::'):
    moving a Claude thread onto a Codex model changes which native session
    resumes, which is a deliberate act, not a per-turn tweak.
    """
    runtime = {k: conversation.get(k) for k in ("model", "effort", "project")}
    if not all(runtime.values()):
        raise ValueError("the conversation has no saved model, effort and project to resume with")
    if effort:
        if effort not in EFFORTS:
            raise ValueError(f"effort must be one of {', '.join(EFFORTS)}")
        runtime["effort"] = effort
    if model:
        model = model.strip()
        if not model or any(ch.isspace() for ch in model) or len(model) > 260:
            raise ValueError("model must be a single selector such as claude-code::claude-fable-5-1")
        saved = str(runtime["model"])
        if "::" in saved and "::" in model and saved.split("::", 1)[0] != model.split("::", 1)[0]:
            raise ValueError(
                f"model {model} is on a different harness than this thread's {saved}; "
                "use set_runtime to move the thread deliberately"
            )
        runtime["model"] = model
    return runtime


def _set_runtime(conversation_id: str, model: str = "", effort: str = "") -> dict:
    if not model and not effort:
        raise ValueError("set_runtime needs a model, an effort, or both")
    if effort and effort not in EFFORTS:
        raise ValueError(f"effort must be one of {', '.join(EFFORTS)}")
    _conversation(conversation_id)
    payload = {k: v for k, v in (("model", model.strip() if model else ""), ("effort", effort)) if v}
    return _call("/conversations/" + parse.quote(conversation_id, safe="") + "/runtime", payload)


def _goal(conversation_id: str, verb: str, text: str = "", max_turns: Any = None, actor: str = "") -> dict:
    verb = (verb or "status").strip().lower()
    path = "/conversations/" + parse.quote(conversation_id, safe="") + "/goal"
    if verb == "status":
        return _call(path)
    if verb == "set":
        if not (text or "").strip():
            raise ValueError("goal set needs text")
        body = {"action": "set", "text": text.strip(), "actor": actor or "Talos"}
        if max_turns is not None:
            body["max_turns"] = int(max_turns)
        return _call(path, body)
    if verb in ("pause", "resume", "clear"):
        return _call(path, {"action": verb})
    raise ValueError("goal verb must be set, status, pause, resume or clear")


def _send(conversation_id: str, text: str, wait_seconds: int = 0, actor: str = "", model: str = "", effort: str = "") -> dict:
    text = (text or "").strip()
    if not text:
        raise ValueError("send requires text")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"text exceeds {MAX_TEXT_CHARS} characters")
    conversation = _conversation(conversation_id)
    if conversation.get("automation"):
        raise ValueError("this conversation is an automation console; it queues replies on the device")
    runtime = _runtime_for(conversation, model, effort)
    turn_id = "agent_" + secrets.token_hex(8)
    prefix = f"[{actor.strip()}] " if actor and actor.strip() else ""
    queued = _call("/queued-turns", {
        "conversationId": conversation_id, "id": turn_id, "text": prefix + text,
        "runtime": runtime, "explicitlyReleased": True,
    })
    result = {"conversation_id": conversation_id, "turn_id": turn_id, "queued": queued.get("turn") or queued.get("started"),
              "runtime": runtime, "reply": None, "status": "queued"}
    deadline = time.monotonic() + max(0, min(int(wait_seconds or 0), MAX_WAIT_SECONDS))
    while time.monotonic() < deadline:
        time.sleep(POLL_SECONDS)
        reply = _reply_after(conversation_id, turn_id)
        if reply and reply["content"]:
            result["reply"] = _trim(reply["content"], MAX_REPLY_CHARS)
            result["status"] = "answered" if reply.get("terminal") in (None, "completed") else reply["terminal"]
            return result
        if reply and reply.get("terminal") in ("failed", "cancelled"):
            result["status"] = reply["terminal"]
            return result
    if wait_seconds:
        result["status"] = "running"
        result["note"] = "No final answer within the wait; read the conversation later for the reply."
    return result


def hermes_conversation_tool(action: str, **kwargs: Any) -> str:
    action = str(action or "").strip().lower()
    try:
        if action == "list":
            return json.dumps({"ok": True, "action": action, "result": _list(str(kwargs.get("project") or ""), kwargs.get("limit") or 20)})
        if action == "read":
            return json.dumps({"ok": True, "action": action, "result": _read(str(kwargs.get("id") or ""), kwargs.get("tail") or 8)})
        if action == "send":
            return json.dumps({"ok": True, "action": action, "result": _send(
                str(kwargs.get("id") or ""), str(kwargs.get("text") or ""),
                kwargs.get("wait_seconds") or 0, str(kwargs.get("actor") or ""),
                str(kwargs.get("model") or ""), str(kwargs.get("effort") or ""))})
        if action == "set_runtime":
            return json.dumps({"ok": True, "action": action, "result": _set_runtime(
                str(kwargs.get("id") or ""), str(kwargs.get("model") or ""), str(kwargs.get("effort") or ""))})
        if action == "goal":
            return json.dumps({"ok": True, "action": action, "result": _goal(
                str(kwargs.get("id") or ""), str(kwargs.get("verb") or "status"), str(kwargs.get("text") or ""),
                kwargs.get("max_turns"), str(kwargs.get("actor") or ""))})
        raise ValueError("Unknown action; use list, read, send, set_runtime or goal")
    except ValueError as exc:
        return json.dumps({"ok": False, "action": action, "error": str(exc)})


HERMES_CONVERSATION_SCHEMA = {
    "name": "hermes_conversation",
    "description": (
        "Read and drive existing Hermes Chat conversations. list: recent conversations, optionally "
        "for one project (ids, titles, models). read: a conversation's last messages. send: queue a "
        "message into an existing conversation; the gateway resumes that session with its full "
        "context, the message lands in the thread as a real turn, and the reply is archived there. "
        "A thread needs no live process to be messaged. Set wait_seconds to wait for the reply. "
        "send accepts model and effort overrides for that one turn (same harness as the thread). "
        "set_runtime changes the thread's saved model and effort persistently. goal sets a standing "
        "goal on the thread (verb=set with text and max_turns): a typed judge checks every final answer "
        "and the thread keeps taking visible continuation turns until the goal is met or the cap is "
        "hit; verbs status, pause, resume, clear manage it."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["list", "read", "send", "set_runtime", "goal"]},
            "model": {"type": "string", "description": "Model selector, e.g. claude-code::claude-fable-5-1 or openai-codex::gpt-6-astra."},
            "effort": {"type": "string", "enum": list(EFFORTS)},
            "verb": {"type": "string", "enum": ["set", "status", "pause", "resume", "clear"], "description": "For goal."},
            "max_turns": {"type": "integer", "minimum": 1, "maximum": 500, "description": "For goal set: turn cap (default 50)."},
            "id": {"type": "string", "description": "Conversation id, e.g. c_mt9bcwww_ah7ikn (from list)."},
            "project": {"type": "string", "description": "For list: only this project's conversations."},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            "tail": {"type": "integer", "minimum": 1, "maximum": 40, "description": "For read: how many trailing messages."},
            "text": {"type": "string", "description": "For send: the message to queue into the conversation."},
            "actor": {"type": "string", "description": "For send: who is speaking, prefixed in brackets (e.g. Talos)."},
            "wait_seconds": {"type": "integer", "minimum": 0, "maximum": 600, "description": "For send: wait up to this long for the reply."},
        },
        "required": ["action"],
    },
}


registry.register(
    name="hermes_conversation",
    toolset="hermes_conversation",
    schema=HERMES_CONVERSATION_SCHEMA,
    handler=lambda args, **kwargs: hermes_conversation_tool(**args),
    check_fn=lambda: True,
    emoji="",
    max_result_size_chars=100_000,
)
