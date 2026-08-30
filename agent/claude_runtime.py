"""Hermes runtime backed by the user's signed-in Claude Code subscription."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Optional

from agent.redact import redact_sensitive_text

logger = logging.getLogger(__name__)


_TOOL_NAMES = {
    "Bash": "exec_command",
    "Edit": "apply_patch",
    "Write": "write_file",
    "Read": "read_file",
    "Glob": "search_files",
    "Grep": "search_files",
}

_CLAUDE_SESSION_STATE_KEY = "claude_code_session"
_CLAUDE_SESSION_STATE_VERSION = 1
_CLAUDE_EFFORT_MAP = {
    "none": "low",
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "max",
    "ultra": "max",
}


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    return "\n".join(
        str(block.get("text") or "").strip()
        for block in content
        if isinstance(block, dict)
        and block.get("type") in {"text", "input_text", "output_text"}
        and str(block.get("text") or "").strip()
    )


def _claude_code_effort(reasoning_config: Any) -> Optional[str]:
    """Translate Hermes's effort ladder to the installed Claude CLI."""
    if not isinstance(reasoning_config, dict):
        return None
    if reasoning_config.get("enabled") is False:
        return "low"
    effort = str(reasoning_config.get("effort") or "").strip().lower()
    return _CLAUDE_EFFORT_MAP.get(effort)


def _claude_history_fingerprint(messages: list[dict[str, Any]]) -> str:
    """Hash the outer transcript prefix represented by a Claude session."""
    digest = hashlib.sha256()
    for message in messages:
        payload = [
            str(message.get("role") or ""),
            _content_text(message.get("content")),
        ]
        digest.update(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8", errors="replace"
            )
        )
        digest.update(b"\n")
    return f"v1:{len(messages)}:{digest.hexdigest()}"


def _normalized_cwd(cwd: str) -> str:
    return str(Path(cwd).expanduser().resolve())


def _load_claude_session_state(agent: Any) -> dict[str, Any]:
    state = getattr(agent, "_claude_code_resume_state", None)
    session_db = getattr(agent, "_session_db", None)
    outer_session_id = str(getattr(agent, "session_id", "") or "")
    if session_db is not None and outer_session_id:
        try:
            stored = session_db.get_session_model_config_value(
                outer_session_id, _CLAUDE_SESSION_STATE_KEY
            )
            if isinstance(stored, dict):
                state = stored
        except Exception:
            logger.warning("Claude Code session-state read failed", exc_info=True)
    return dict(state) if isinstance(state, dict) else {}


def _persist_claude_session_state(
    agent: Any,
    *,
    claude_session_id: str,
    cwd: str,
    messages: list[dict[str, Any]],
) -> None:
    """Persist a confirmed inner Claude session against the outer Hermes session."""
    if not claude_session_id:
        return
    state = {
        "version": _CLAUDE_SESSION_STATE_VERSION,
        "session_id": claude_session_id,
        "cwd": _normalized_cwd(cwd),
        "history_fingerprint": _claude_history_fingerprint(messages),
        "updated_at": time.time(),
    }
    agent._claude_code_resume_state = state
    session_db = getattr(agent, "_session_db", None)
    outer_session_id = str(getattr(agent, "session_id", "") or "")
    if session_db is None or not outer_session_id:
        return
    try:
        session_db.patch_session_model_config(
            outer_session_id, {_CLAUDE_SESSION_STATE_KEY: state}
        )
    except Exception:
        logger.warning("Claude Code session-state persistence failed", exc_info=True)


def claude_history_handoff(messages: list[dict[str, Any]], user_message: str) -> str:
    blocks: list[str] = []
    for message in messages[-24:]:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        text = _content_text(message.get("content"))
        if not text:
            continue
        blocks.append(f"{'Will' if role == 'user' else 'Assistant'}: {text}")
    transcript = "\n\n".join(blocks)[-60_000:]
    if not transcript:
        return user_message
    return (
        "Continue this existing Hermes Chat conversation. Use the transcript "
        "as context; do not restart completed discovery.\n\n"
        f"Recent transcript:\n{transcript}\n\n"
        f"Current request:\n{user_message}"
    )


def make_claude_code_event_bridge(agent: Any) -> Callable[[dict[str, Any]], None]:
    """Project Claude Code stream-json records onto Hermes run events."""
    started: dict[str, tuple[str, dict[str, Any], float]] = {}
    pending_text: list[str] = []

    def _emit_commentary() -> None:
        if not pending_text or not getattr(agent, "show_commentary", True):
            pending_text.clear()
            return
        text = "\n\n".join(pending_text).strip()
        pending_text.clear()
        emit = getattr(agent, "_emit_interim_assistant_message", None)
        if emit and text:
            try:
                emit({"role": "assistant", "content": text})
            except Exception:
                logger.debug("Claude commentary callback failed", exc_info=True)

    def _tool_name(raw: str) -> str:
        return _TOOL_NAMES.get(raw, raw or "tool")

    def _tool_started(block: dict[str, Any]) -> None:
        _emit_commentary()
        call_id = str(block.get("id") or "")
        raw_name = str(block.get("name") or "tool")
        name = _tool_name(raw_name)
        args = block.get("input") if isinstance(block.get("input"), dict) else {}
        started[call_id] = (name, args, time.monotonic())
        progress = getattr(agent, "tool_progress_callback", None)
        if progress:
            progress(
                "tool.started",
                name,
                str(args.get("command") or args.get("description") or "")[:500],
                args,
                tool_call_id=call_id,
            )
        callback = getattr(agent, "tool_start_callback", None)
        if callback:
            callback(call_id, name, args)

    def _tool_completed(block: dict[str, Any]) -> None:
        call_id = str(block.get("tool_use_id") or "")
        prior = started.pop(call_id, None)
        name, args, started_at = prior or ("tool", {}, time.monotonic())
        result = _content_text(block.get("content"))
        if not result and isinstance(block.get("content"), str):
            result = str(block.get("content"))
        result = redact_sensitive_text(result, force=True)
        is_error = bool(block.get("is_error"))
        progress = getattr(agent, "tool_progress_callback", None)
        if progress:
            progress(
                "tool.completed",
                name,
                None,
                None,
                duration=max(0.0, time.monotonic() - started_at),
                is_error=is_error,
                result=result,
                tool_call_id=call_id,
            )
        callback = getattr(agent, "tool_complete_callback", None)
        if callback:
            callback(call_id, name, args, result)

    def on_event(event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "stream_event":
            stream_event = event.get("event") or {}
            delta = stream_event.get("delta") or {}
            if delta.get("type") == "thinking_delta":
                callback = getattr(agent, "_fire_reasoning_delta", None)
                thinking = str(delta.get("thinking") or "")
                if callback and thinking:
                    callback(thinking)
            return
        if event_type == "assistant":
            for block in ((event.get("message") or {}).get("content") or []):
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    text = str(block.get("text") or "").strip()
                    if text:
                        pending_text.append(text)
                elif block.get("type") == "tool_use":
                    _tool_started(block)
            return
        if event_type == "user":
            for block in ((event.get("message") or {}).get("content") or []):
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    _tool_completed(block)

    return on_event


def _materialize_images(original_user_message: Any, directory: str) -> list[str]:
    if not isinstance(original_user_message, list):
        return []
    paths: list[str] = []
    for index, part in enumerate(original_user_message):
        if not isinstance(part, dict) or part.get("type") != "image_url":
            continue
        image_url = part.get("image_url")
        value = image_url.get("url") if isinstance(image_url, dict) else image_url
        if not isinstance(value, str) or not value.startswith("data:image/"):
            continue
        try:
            header, encoded = value.split(",", 1)
            media_type = header[5:].split(";", 1)[0].lower()
            suffix = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}.get(media_type)
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error):
            continue
        if suffix is None or len(data) > 10 * 1024 * 1024:
            continue
        path = Path(directory, f"hermes-image-{index}{suffix}")
        path.write_bytes(data)
        paths.append(str(path))
    return paths


def run_claude_code_turn(
    agent: Any,
    *,
    user_message: str,
    original_user_message: Any,
    messages: list[dict[str, Any]],
    effective_task_id: str,
    should_review_memory: bool = False,
) -> dict[str, Any]:
    """Run one Claude subscription turn and return the standard agent result."""
    del effective_task_id, should_review_memory
    from agent.deadline import resolve_timeout
    from agent.runtime_cwd import resolve_agent_cwd
    from agent.transports.claude_code_session import (
        DEFAULT_RESIDENT_FIRST_EVENT_TIMEOUT,
        DEFAULT_STARTUP_FIRST_EVENT_TIMEOUT,
        ClaudeCodeSession,
    )

    cwd = getattr(agent, "session_cwd", None) or str(resolve_agent_cwd())
    cwd = _normalized_cwd(cwd)
    prior_messages = messages[:-1]
    prior_fingerprint = _claude_history_fingerprint(prior_messages)
    prior_state = _load_claude_session_state(agent)
    prior_session_id = str(prior_state.get("session_id") or "").strip()
    durable_resume = bool(
        prior_session_id
        and prior_state.get("version") == _CLAUDE_SESSION_STATE_VERSION
        and prior_state.get("cwd") == cwd
        and prior_state.get("history_fingerprint") == prior_fingerprint
    )
    runtime_contract = (
        "You are the selected Claude subscription runtime inside Hermes Chat. "
        "Follow the user's current request and the project's own instructions. "
        "Use Claude Code built-ins for repository work and Hermes MCP tools for "
        "Hermes capabilities. Do not invent a governance restriction: explicit "
        "user instructions about commits, branches, previews, pushes, or deploys "
        "are authoritative. Run finite installs/builds/tests in the foreground. "
        "Launch long-lived previews only in tmux or a project service and verify "
        "the reachable URL before reporting it."
    )
    read_only = bool(getattr(agent, "read_only", False))
    if read_only:
        runtime_contract += (
            " This scheduled review is read-only. Inspect and report only; do "
            "not edit files, run mutating commands, change external state, "
            "create tasks, or delegate work. Ask the user to authorize any "
            "follow-up action in an ordinary chat turn."
        )
    model = str(getattr(agent, "model", "") or "claude-fable-5")
    effort = _claude_code_effort(getattr(agent, "reasoning_config", None))
    resident_first_event_timeout = resolve_timeout(
        "claude_code.resident_first_event",
        default=DEFAULT_RESIDENT_FIRST_EVENT_TIMEOUT,
    )
    startup_first_event_timeout = resolve_timeout(
        "claude_code.startup_first_event",
        default=DEFAULT_STARTUP_FIRST_EVENT_TIMEOUT,
    )
    session = getattr(agent, "_claude_code_session", None)
    resident_continuity = bool(
        session is not None
        and session.compatible_with(
            cwd=cwd,
            model=model,
            effort=effort,
            system_prompt=runtime_contract,
            read_only=read_only,
        )
        and getattr(session, "history_fingerprint", None) == prior_fingerprint
    )
    if session is not None and not resident_continuity:
        # A project/model/transcript switch is a real continuity boundary. Do
        # not feed it into the old native process; the durable handoff path
        # below starts a correctly scoped Claude parent.
        try:
            session.close()
        except Exception:
            logger.debug("Claude Code stale-session cleanup failed", exc_info=True)
        agent._claude_code_session = None
        session = None

    with tempfile.TemporaryDirectory(prefix="hermes-claude-images-") as image_dir:
        images = _materialize_images(original_user_message, image_dir)
        prompt = (
            user_message
            if resident_continuity or durable_resume
            else claude_history_handoff(prior_messages, user_message)
        )
        if images:
            prompt += "\n\nAttached images are available at:\n" + "\n".join(
                f"- {path}" for path in images
            )

        def _remember_confirmed_session(claude_session_id: str) -> None:
            # At stream time the outer list contains the current user message
            # but not the final assistant answer yet. Persist immediately so
            # quota errors, cancellation, and process death remain resumable.
            _persist_claude_session_state(
                agent,
                claude_session_id=claude_session_id,
                cwd=cwd,
                messages=messages,
            )

        def _watchdog_timeout(payload: dict[str, Any]) -> None:
            retrying = bool(payload.get("retrying"))
            timeout = float(payload.get("timeout_seconds") or 0.0)
            message = (
                f"Claude did not acknowledge the turn within {timeout:g} seconds—"
                "resetting the runtime and retrying once."
                if retrying
                else "Claude did not acknowledge the turn after the runtime reset; "
                "ending this run."
            )
            progress = getattr(agent, "tool_progress_callback", None)
            if progress is not None:
                try:
                    progress(
                        "runtime.first_event_timeout",
                        "claude-code",
                        message,
                        None,
                        **payload,
                    )
                except Exception:
                    logger.debug(
                        "Claude Code watchdog progress callback failed",
                        exc_info=True,
                    )
            emit_status = getattr(agent, "_emit_status", None)
            if emit_status is not None:
                try:
                    emit_status(message)
                except Exception:
                    logger.debug(
                        "Claude Code watchdog status callback failed",
                        exc_info=True,
                    )

        if session is None:
            session = ClaudeCodeSession(
                cwd=cwd,
                model=model,
                session_id=prior_session_id if durable_resume else None,
                resume=durable_resume,
                effort=effort,
                system_prompt=runtime_contract,
                read_only=read_only,
                on_event=make_claude_code_event_bridge(agent),
                on_session_id=_remember_confirmed_session,
                on_watchdog_timeout=_watchdog_timeout,
                resident_first_event_timeout=resident_first_event_timeout,
                startup_first_event_timeout=startup_first_event_timeout,
            )
            session.history_fingerprint = prior_fingerprint
            agent._claude_code_session = session
        else:
            # Event callbacks are per outer turn even though the native process
            # is per conversation. Rebind them before sending the next message.
            session.on_event = make_claude_code_event_bridge(agent)
            session.on_session_id = _remember_confirmed_session
            session.on_watchdog_timeout = _watchdog_timeout
            session.resident_first_event_timeout = resident_first_event_timeout
            session.startup_first_event_timeout = startup_first_event_timeout
        try:
            turn = session.run_turn(prompt)
        except Exception as exc:
            logger.exception("Claude Code turn failed")
            turn = None
            error = str(exc)
            error_code = getattr(exc, "error_code", None)
            try:
                session.close()
            except Exception:
                pass
            agent._claude_code_session = None

    if turn is None:
        return {
            "final_response": "",
            "messages": messages,
            "api_calls": 0,
            "completed": False,
            "partial": True,
            "interrupted": False,
            "error": error,
            **({"error_code": error_code} if error_code else {}),
            "agent_persisted": True,
        }

    transcript_persisted = getattr(agent, "_session_db", None) is None
    if turn.final_text:
        callback = getattr(agent, "_fire_stream_delta", None)
        if callback:
            callback(turn.final_text)
        from agent.message_metadata import append_message

        append_message(messages, {"role": "assistant", "content": turn.final_text})
        if getattr(agent, "_session_db", None) is not None:
            try:
                agent._flush_messages_to_session_db(messages)
                transcript_persisted = True
            except Exception:
                logger.warning("Claude Code transcript persistence failed", exc_info=True)

    if turn.session_confirmed:
        # If the assistant row could not be persisted, retain the earlier
        # user-boundary fingerprint rather than claiming the outer transcript
        # contains an answer it may not be able to reload.
        synchronized_messages = (
            messages
            if not turn.final_text or transcript_persisted
            else messages[:-1]
        )
        _persist_claude_session_state(
            agent,
            claude_session_id=turn.session_id,
            cwd=cwd,
            messages=synchronized_messages,
        )
        session.history_fingerprint = _claude_history_fingerprint(
            synchronized_messages
        )

    if turn.should_retire:
        if getattr(agent, "_claude_code_session", None) is session:
            agent._claude_code_session = None
        try:
            session.close()
        except Exception:
            pass

    input_tokens = int(turn.usage.get("input_tokens") or 0)
    output_tokens = int(turn.usage.get("output_tokens") or 0)
    agent.session_prompt_tokens = getattr(agent, "session_prompt_tokens", 0) + input_tokens
    agent.session_completion_tokens = getattr(agent, "session_completion_tokens", 0) + output_tokens
    agent.session_total_tokens = getattr(agent, "session_total_tokens", 0) + input_tokens + output_tokens
    agent._iters_since_skill = getattr(agent, "_iters_since_skill", 0) + turn.tool_iterations
    user_interrupted = bool(turn.interrupted and getattr(agent, "_interrupt_requested", False))
    interrupt_message = getattr(agent, "_interrupt_message", None) if user_interrupted else None
    if user_interrupted:
        agent.clear_interrupt()
    return {
        "final_response": turn.final_text,
        "messages": messages,
        "api_calls": 1,
        "completed": not turn.interrupted and turn.error is None and bool(turn.final_text),
        "partial": turn.interrupted or turn.error is not None or not bool(turn.final_text),
        "interrupted": user_interrupted,
        **({"interrupt_message": interrupt_message} if interrupt_message else {}),
        "error": turn.error,
        **({"error_code": turn.error_code} if turn.error_code else {}),
        **(
            {"watchdog_retries": turn.watchdog_retries}
            if turn.watchdog_retries
            else {}
        ),
        "agent_persisted": True,
        "claude_session_id": turn.session_id,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
