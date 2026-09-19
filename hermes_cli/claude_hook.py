#!/usr/bin/env python3
"""Claude Code command hook: forward one tool call to the Hermes gateway.

Installed on every managed ``claude`` process as its ``PreToolUse`` and
``PostToolUse`` hook (see :mod:`hermes_cli.claude_hooks`). Claude Code
writes the hook payload to stdin; this script posts it to
``$HERMES_HOOK_URL`` with ``$HERMES_HOOK_TOKEN`` and turns the gateway's
answer into the hook contract:

* a ``block`` (PreToolUse) is printed to stderr and the script exits 2,
  which refuses the call and shows the message to the model;
* a ``message`` (PostToolUse) is printed to stdout as the hook's JSON
  ``{"decision": "block", "reason": ...}`` with exit 0, which Claude Code
  puts in front of the model right after the call (verified live on
  2.1.257: the model quotes it; the exit-2 form reached the transcript but
  the model did not act on it);
* anything else, including a missing endpoint, a timeout, a non-200 or a
  malformed answer, exits 0 with no output so the call proceeds.

Standard library only: it runs from whatever Python the gateway used, with
no Hermes import path, on every tool call.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

MAX_RESPONSE_CHARS = 64_000
DEFAULT_TIMEOUT_SECONDS = 8.0


def _clip(value, limit=MAX_RESPONSE_CHARS):
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "…"
    if isinstance(value, dict):
        return {str(key): _clip(item, limit // 4) for key, item in list(value.items())[:40]}
    if isinstance(value, list):
        return [_clip(item, limit // 4) for item in value[:40]]
    return value


def build_request(payload):
    """The gateway request body for one Claude Code hook payload."""
    if not isinstance(payload, dict):
        return None
    event = str(payload.get("hook_event_name") or "")
    if event not in ("PreToolUse", "PostToolUse"):
        return None
    tool_input = payload.get("tool_input")
    return {
        "event": event,
        "tool_name": str(payload.get("tool_name") or ""),
        "tool_input": _clip(tool_input) if isinstance(tool_input, dict) else {},
        "tool_response": _clip(payload.get("tool_response")),
        "tool_use_id": str(payload.get("tool_use_id") or ""),
        "claude_session_id": str(payload.get("session_id") or ""),
        # The working directory the call ran in: the evidence ledger keys its
        # workspace digest and controller re-runs on it.
        "cwd": str(payload.get("cwd") or ""),
    }


def feedback(event, answer):
    """The text the model must see for this answer, or None to let the call proceed."""
    if not isinstance(answer, dict):
        return None
    if event == "PreToolUse":
        if answer.get("action") == "block":
            message = str(answer.get("message") or "").strip()
            return message or "The Hermes gateway blocked this tool call."
        return None
    message = answer.get("message")
    return str(message).strip() if isinstance(message, str) and message.strip() else None


def post(url, token, body, timeout):
    request = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            return None
        return json.loads(response.read().decode("utf-8") or "{}")


def main(argv=None, stdin=None, stderr=None, environ=None, stdout=None):
    environ = os.environ if environ is None else environ
    stdin = sys.stdin if stdin is None else stdin
    stderr = sys.stderr if stderr is None else stderr
    stdout = sys.stdout if stdout is None else stdout
    url = str(environ.get("HERMES_HOOK_URL") or "").strip()
    token = str(environ.get("HERMES_HOOK_TOKEN") or "").strip()
    if not url or not token:
        return 0
    try:
        timeout = float(environ.get("HERMES_HOOK_TIMEOUT") or DEFAULT_TIMEOUT_SECONDS)
    except ValueError:
        timeout = DEFAULT_TIMEOUT_SECONDS
    try:
        payload = json.loads(stdin.read() or "{}")
    except (ValueError, OSError):
        return 0
    body = build_request(payload)
    if body is None:
        return 0
    try:
        answer = post(url, token, body, timeout)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError):
        return 0
    text = feedback(body["event"], answer)
    if text is None:
        return 0
    try:
        if body["event"] == "PostToolUse":
            stdout.write(json.dumps({"decision": "block", "reason": text}, ensure_ascii=False))
            stdout.flush()
            return 0
        stderr.write(text + "\n")
        stderr.flush()
    except OSError:
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
