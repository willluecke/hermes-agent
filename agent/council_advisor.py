#!/usr/bin/env python3
"""Two-model council: GPT advisor consulted once per user turn.

Fires from run_conversation() before the main (Claude Fable 5) call. The
advisor's answer is appended to the model-visible user message inside a
<council-advisor> block; the persisted transcript stays clean via the
_persist_user_message_override mechanism.

Design rails carried over from gpt_operator_tool.py:
  - secret redaction before anything leaves the Pi
  - hard input/output caps
  - audit file per call in ~/.hermes/operator-briefs/
  - never blocks the turn: any failure returns None and Hermes proceeds solo

Env knobs:
  HERMES_COUNCIL_ENABLED    default "1" ("0"/"false" disables)
  HERMES_COUNCIL_MODEL      default "gpt-5.5"
  HERMES_COUNCIL_REASONING  default "high"
  HERMES_COUNCIL_MAX_OUTPUT default 1600 tokens
  HERMES_COUNCIL_TIMEOUT    default 90 seconds
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MAX_INPUT_CHARS = 12_000
MAX_CONTEXT_CHARS = 2_000
MIN_MESSAGE_CHARS = 8
MAX_ADVISOR_CHARS = 8_000
# Synthetic/system-generated turns (cron beats, bracketed markers) skip council.
SKIP_PREFIXES = ("[", "<system", "{")

SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_\-]{12,}\b"),
    re.compile(r"\bsk-proj-[A-Za-z0-9_\-]{12,}\b"),
    re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{12,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\b[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}\b"),
    re.compile(r"(?i)\b(api[_-]?key|token|password|passwd|secret)\s*[:=]\s*['\"]?[^'\"\s,;]{6,}"),
]
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

ADVISOR_INSTRUCTIONS = (
    "You are the second voice in a two-model council behind a personal assistant. "
    "Another frontier model will read your answer alongside its own and synthesize "
    "the final reply, so be direct and substantive — no hedging preamble, no "
    "restating the question. If the message is a factual or analytical question, "
    "answer it. If it is a task or action request for an agent with tools, give "
    "brief strategic advice: the right approach, key pitfalls, what you would "
    "check first. If it is small talk, reply in one short line."
)


def _redact(text: str) -> str:
    result = ANSI_RE.sub("", text)
    for pattern in SECRET_PATTERNS:
        result = pattern.sub("[REDACTED_SECRET]", result)
    return result


def _enabled() -> bool:
    return os.environ.get("HERMES_COUNCIL_ENABLED", "1").lower() not in ("0", "false", "no")


def _api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        try:
            from hermes_cli.config import get_env_value

            key = (get_env_value("OPENAI_API_KEY") or "").strip()
        except Exception:
            key = ""
    return key


def _base_url() -> str:
    base = (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
        or "https://api.openai.com/v1"
    ).rstrip("/")
    if base == "https://api.openai.com":
        base += "/v1"
    return base


def _last_assistant_text(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages or []):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()[:MAX_CONTEXT_CHARS]
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and isinstance(block.get("text"), str):
                        return block["text"].strip()[:MAX_CONTEXT_CHARS]
    return ""


def _audit(prompt: str, answer: str, usage: dict, elapsed: float) -> None:
    try:
        out_dir = Path.home() / ".hermes" / "operator-briefs"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"council_{time.strftime('%Y%m%d_%H%M%S')}.md"
        path.write_text(
            f"# Council advisor call\n\nelapsed: {elapsed:.1f}s\nusage: {json.dumps(usage)}\n\n"
            f"## Sent\n\n{prompt}\n\n## Advisor answer\n\n{answer}\n",
            encoding="utf-8",
        )
    except Exception:  # audit is best-effort, never fatal
        pass


def get_advisor_block(user_message: str, messages: list[dict[str, Any]]) -> str | None:
    """Return the advisor's answer for this turn, or None to proceed solo."""
    if not _enabled():
        return None
    text = (user_message or "").strip()
    if len(text) < MIN_MESSAGE_CHARS or text.startswith(SKIP_PREFIXES):
        return None
    api_key = _api_key()
    if not api_key:
        return None

    prompt_parts = []
    prior = _last_assistant_text(messages)
    if prior:
        prompt_parts.append(f"(Assistant's previous reply, for context)\n{prior}\n")
    prompt_parts.append(f"User message:\n{text}")
    prompt = _redact("\n".join(prompt_parts))[:MAX_INPUT_CHARS]

    body = {
        "model": os.environ.get("HERMES_COUNCIL_MODEL", "gpt-5.5"),
        "messages": [
            {"role": "system", "content": ADVISOR_INSTRUCTIONS},
            {"role": "user", "content": prompt},
        ],
        "max_completion_tokens": int(os.environ.get("HERMES_COUNCIL_MAX_OUTPUT", "1600")),
        "reasoning_effort": os.environ.get("HERMES_COUNCIL_REASONING", "high"),
    }
    request = urllib.request.Request(
        f"{_base_url()}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )

    started = time.monotonic()
    try:
        timeout = int(os.environ.get("HERMES_COUNCIL_TIMEOUT", "90"))
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
    except Exception as exc:
        logger.warning("council advisor unavailable, proceeding solo: %s", exc)
        return None

    choices = data.get("choices") or []
    answer = ""
    if choices and isinstance(choices[0], dict):
        message = choices[0].get("message") or {}
        if isinstance(message.get("content"), str):
            answer = message["content"].strip()
    if not answer:
        return None

    answer = _redact(answer)[:MAX_ADVISOR_CHARS]
    _audit(prompt, answer, data.get("usage") or {}, time.monotonic() - started)
    return answer
