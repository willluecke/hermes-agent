"""Hermes runtime backed by the user's signed-in Claude Code subscription."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Callable, Optional
import urllib.error
import urllib.request

from agent.continuity import (
    emit_continuity,
    ensure_claude_transcript_retention,
    handoff_disclosure,
    plural,
    render_history_blocks,
)
from agent.redact import redact_sensitive_text
from agent.transports.claude_code_session import (
    HERMES_MONITOR_NOTE,
    HERMES_RESULT_DEFERRED,
)

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
# hermes_state.LATE_ANSWER_DISPLAY_KIND, kept local so this module does not
# import the session store at load time.
_LATE_ANSWER_DISPLAY_KIND = "late_answer"
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
    """Translate Hermes's effort ladder to the installed Claude CLI.

    Ultracode maps to Claude Code's own ``--effort ultracode`` (xhigh plus
    standing Workflow orchestration since CLI 2.1.280; a model without it
    runs at its best supported effort). Effort is part of the resident
    process's identity, so toggling Ultracode restarts the CLI and the next
    turn resumes the same Claude session.
    """
    if not isinstance(reasoning_config, dict):
        return None
    if reasoning_config.get("ultracode"):
        return "ultracode"
    if reasoning_config.get("enabled") is False:
        return "low"
    effort = str(reasoning_config.get("effort") or "").strip().lower()
    return _CLAUDE_EFFORT_MAP.get(effort)


def _fingerprint_text(content: Any) -> str:
    """Render content the way the session store replays it.

    Stored history replaces each image part with a ``[screenshot]`` text
    placeholder. Hash the same shape for the in-memory message so a turn that
    carried an image does not permanently break Claude session resume.
    """
    if not isinstance(content, list):
        return _content_text(content)
    pieces: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type in {"image", "image_url", "input_image"}:
            pieces.append("[screenshot]")
        elif block_type in {"text", "input_text", "output_text"}:
            text = str(block.get("text") or "").strip()
            if text:
                pieces.append(text)
    return "\n".join(pieces)


def _is_transcript_scaffolding(message: dict[str, Any]) -> bool:
    """A row the durable transcript never keeps: the verify judge's synthetic
    nudge and its siblings (``run_agent._EPHEMERAL_SCAFFOLDING_FLAGS``)."""
    try:
        from run_agent import _is_ephemeral_scaffolding

        return bool(_is_ephemeral_scaffolding(message))
    except Exception:
        return bool(message.get("_pre_verify_synthetic") or message.get("_verification_stop_synthetic"))


def _fingerprint_rows(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The rows of the outer transcript a Claude session is measured against."""
    rows: list[dict[str, Any]] = []
    for message in messages:
        if message.get("display_kind") == _LATE_ANSWER_DISPLAY_KIND:
            # A late answer is appended to the store whenever the CLI writes
            # it, possibly while the next turn is already loading history.
            # The Claude session holds it either way, so it never counts
            # toward the prefix that decides whether that session continues.
            continue
        if _is_transcript_scaffolding(message):
            continue
        rows.append(message)
    return rows


def _claude_history_fingerprint(messages: list[dict[str, Any]]) -> str:
    """Hash the outer transcript prefix represented by a Claude session.

    Only rows the durable transcript keeps count. The live list a turn ends
    with can hold rows the store drops, and the next turn's prefix comes from
    the store: a synthetic verify nudge in the fingerprint meant every turn
    the judge sent back closed its Claude session and handed the next turn a
    60k-character transcript instead (2026-09-24, "you're asking in a fresh
    session" two minutes after the previous answer).
    """
    digest = hashlib.sha256()
    rows = _fingerprint_rows(messages)
    for message in rows:
        payload = [
            str(message.get("role") or ""),
            _fingerprint_text(message.get("content")),
        ]
        digest.update(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8", errors="replace"
            )
        )
        digest.update(b"\n")
    return f"v1:{len(rows)}:{digest.hexdigest()}"


def _claude_history_continues(
    recorded: Optional[str], messages: list[dict[str, Any]]
) -> tuple[bool, list[dict[str, Any]]]:
    """Whether the transcript a Claude session recorded is still the start of
    ``messages``, and the rows added since.

    A session keeps its context as long as nothing it saw was changed; rows
    appended after its last turn (a worker's review, an adopted late answer,
    another device's turn) are handed to it as a delta rather than costing
    the session. Only a diverged prefix (an edited or rolled-back transcript)
    reads as discontinuity.
    """
    if not recorded:
        return False, []
    try:
        _version, count_text, _digest = str(recorded).split(":", 2)
        count = int(count_text)
    except ValueError:
        return False, []
    rows = _fingerprint_rows(messages)
    if count > len(rows):
        return False, []
    if _claude_history_fingerprint(rows[:count]) != recorded:
        return False, []
    return True, rows[count:]


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


def _sync_store_ping(session_id: str) -> None:
    """Tell Hermes Chat's sync store a conversation has a row to adopt.

    Best effort: the sync store also sweeps on a timer, so a missed ping only
    delays the answer's appearance in the app.
    """
    if not session_id.startswith("hermes-chat-"):
        return
    base = os.environ.get("HERMES_SYNC_URL", "http://127.0.0.1:8643").rstrip("/")
    key_file = os.environ.get("HERMES_SYNC_KEY_FILE", "").strip()
    try:
        key = Path(key_file or Path.home() / ".hermes-api-key").read_text(
            encoding="utf-8"
        ).strip()
    except OSError:
        logger.debug("Sync store key unavailable; skipping the adoption ping")
        return
    if not key:
        return
    request = urllib.request.Request(
        f"{base}/adoptable-messages/changed",
        data=json.dumps({"sessionId": session_id}).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5.0) as response:  # noqa: S310
            response.read()
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        logger.info("Sync store adoption ping failed for %s: %s", session_id, exc)


def save_claude_late_answer(agent: Any, record: dict[str, Any]) -> Optional[int]:
    """Store output Claude wrote after its Hermes turn closed.

    The row is ordinary assistant text marked with the late-answer display
    kind, so it replays, searches and hands off like any answer, is skipped
    by the continuity fingerprint, and is adoptable by Hermes Chat.
    """
    text = str(record.get("text") or "").strip()
    if not text:
        return None
    session_db = getattr(agent, "_session_db", None)
    outer_session_id = str(getattr(agent, "session_id", "") or "")
    if (
        session_db is None
        or not outer_session_id
        or getattr(agent, "_persist_disabled", False)
    ):
        logger.warning(
            "Late Claude answer kept only in the late-answer log: no session "
            "store for session %s",
            outer_session_id or "-",
        )
        return None
    complete = bool(record.get("complete", True))
    metadata = {
        "source": "claude_code",
        "complete": complete,
        "is_error": bool(record.get("is_error")),
        "origin": str(record.get("origin") or ""),
        "claude_session_id": str(record.get("claude_session_id") or ""),
        "captured_at": record.get("captured_at"),
    }
    try:
        row_id = session_db.append_message(
            outer_session_id,
            "assistant",
            text,
            finish_reason="stop" if complete else "incomplete",
            timestamp=record.get("captured_at"),
            display_kind=_LATE_ANSWER_DISPLAY_KIND,
            display_metadata=metadata,
        )
    except Exception:
        logger.warning(
            "Could not store a late Claude answer for session %s",
            outer_session_id,
            exc_info=True,
        )
        return None
    logger.info(
        "Stored a late Claude answer as row %s in session %s",
        row_id,
        outer_session_id,
    )
    _sync_store_ping(outer_session_id)
    return row_id


CLAUDE_HANDOFF_LEAD = (
    "Continue this existing Hermes Chat conversation. Use the transcript "
    "as context; do not restart completed discovery."
)
CLAUDE_REBUILD_LEAD = (
    "You are continuing an existing Hermes Chat conversation in a new Claude "
    "session. Your earlier session for it could not be resumed ({reason}), so "
    "Hermes restored the conversation below from its stored transcript. It is "
    "the conversation so far, not a summary. Continue it: do not restart "
    "completed discovery or describe this as a fresh start."
)
CLAUDE_DELTA_LEAD = (
    "You are continuing your own Claude session in this Hermes Chat "
    "conversation. Since your last turn, the messages below were added to the "
    "conversation without you; read them as context, then answer the current "
    "request."
)
# How much of the stored transcript a rebuilt session is given. The old cap
# (24 messages, 60k characters) turned a lost session into a summary; this is
# about 200k tokens, a fifth of a Claude window, which holds the whole text
# transcript of nearly any conversation. Past it the oldest messages are
# dropped, and both the model and the user are told how many.
CLAUDE_HANDOFF_MAX_CHARS = 800_000


def _claude_handoff_entries(messages: list[dict[str, Any]]) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for message in messages:
        role = message.get("role")
        if role not in {"user", "assistant"} or _is_transcript_scaffolding(message):
            continue
        text = _content_text(message.get("content"))
        if role == "user":
            text = strip_reattached_image_notes(text)
        if text:
            entries.append((str(role), text))
    return entries


def claude_handoff_coverage(messages: list[dict[str, Any]]) -> tuple[int, int]:
    """``(carried, omitted)`` message counts a handoff of ``messages`` would have."""
    entries = _claude_handoff_entries(messages)
    _, omitted, _ = render_history_blocks(
        entries, CLAUDE_HANDOFF_MAX_CHARS, _render_claude_block
    )
    return len(entries) - omitted, omitted


def _render_claude_block(role: str, text: str) -> str:
    return f"{'Will' if role == 'user' else 'Assistant'}: {text}"


def claude_history_handoff(
    messages: list[dict[str, Any]], user_message: str, *, lead: str = CLAUDE_HANDOFF_LEAD
) -> str:
    blocks, omitted, truncated = render_history_blocks(
        _claude_handoff_entries(messages), CLAUDE_HANDOFF_MAX_CHARS, _render_claude_block
    )
    if not blocks:
        return user_message
    transcript = "\n\n".join(blocks) + handoff_disclosure(omitted, truncated)
    return (
        f"{lead}\n\n"
        f"Transcript:\n{transcript}\n\n"
        f"Current request:\n{user_message}"
    )


def claude_session_file(session_id: str) -> Optional[Path]:
    """The Claude Code transcript that ``--resume <session_id>`` would load."""
    from agent.transports.claude_code_session import _claude_transcript_root

    if not session_id or "/" in session_id or session_id.startswith("."):
        return None
    try:
        return next((_claude_transcript_root() / "projects").glob(f"*/{session_id}.jsonl"), None)
    except OSError:
        return None


def _claude_tool_preview(raw_name: str, args: dict[str, Any], cwd: Optional[str]) -> str:
    """The row label a terminal agent would print: path, pattern, URL or command."""
    from agent.tool_diff import display_path

    def text(*keys: str) -> str:
        for key in keys:
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    if raw_name in {"Edit", "MultiEdit", "Write", "Read", "NotebookEdit"}:
        return display_path(text("file_path", "notebook_path"), cwd)
    if raw_name == "Grep":
        pattern, where = text("pattern"), text("path")
        where = display_path(where, cwd) if where else ""
        return f"{pattern} in {where}" if pattern and where else pattern
    if raw_name == "Glob":
        return text("pattern")
    if raw_name == "WebFetch":
        return text("url")
    if raw_name in {"WebSearch", "ToolSearch"}:
        return text("query")
    return text("command", "description", "query", "url", "prompt")


def make_claude_code_event_bridge(
    agent: Any,
    record: Optional[Callable[[str, str, dict[str, Any], str, str], None]] = None,
    *,
    cwd: Optional[str] = None,
) -> Callable[[dict[str, Any]], None]:
    """Project Claude Code stream-json records onto Hermes run events.

    ``record(raw_name, name, args, result, call_id)`` is told about every
    completed tool call so the turn can replay them through the observer
    hooks afterwards (see :func:`_claude_hook_parity`). A successful Edit,
    MultiEdit or Write also reports its unified diff and line counts, which
    Hermes Chat renders red and green.
    """
    started: dict[str, tuple[str, dict[str, Any], float, str]] = {}
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
        started[call_id] = (name, args, time.monotonic(), raw_name)
        progress = getattr(agent, "tool_progress_callback", None)
        if progress:
            progress(
                "tool.started",
                name,
                _claude_tool_preview(raw_name, args, cwd)[:500],
                args,
                tool_call_id=call_id,
            )
        callback = getattr(agent, "tool_start_callback", None)
        if callback:
            callback(call_id, name, args)

    def _tool_completed(block: dict[str, Any], native_result: Any = None) -> None:
        call_id = str(block.get("tool_use_id") or "")
        prior = started.pop(call_id, None)
        name, args, started_at, raw_name = prior or ("tool", {}, time.monotonic(), "tool")
        result = _content_text(block.get("content"))
        if not result and isinstance(block.get("content"), str):
            result = str(block.get("content"))
        result = redact_sensitive_text(result, force=True)
        is_error = bool(block.get("is_error"))
        if record is not None:
            try:
                record(raw_name, name, args, result, call_id)
            except Exception:
                logger.debug("Claude tool record failed", exc_info=True)
        edit_diff: dict[str, Any] = {}
        if not is_error and raw_name in {"Edit", "MultiEdit", "Write"}:
            # Claude Code's own patch first; our computed diff only when the
            # CLI described none (a Write that creates a file, an older CLI).
            try:
                from agent.tool_diff import claude_native_diff, claude_tool_diff

                edit_diff = (
                    claude_native_diff(native_result, args, cwd=cwd)
                    or claude_tool_diff(raw_name, args, cwd=cwd)
                    or {}
                )
            except Exception:
                logger.debug("Claude edit diff failed", exc_info=True)
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
                **edit_diff,
            )
        callback = getattr(agent, "tool_complete_callback", None)
        if callback:
            callback(call_id, name, args, result)

    def on_event(event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == HERMES_RESULT_DEFERRED:
            # The transport is holding a result while a Monitor runs; what the
            # model said so far belongs in the transcript as commentary.
            _emit_commentary()
            return
        if event_type == HERMES_MONITOR_NOTE:
            _emit_commentary()
            note = str(event.get("text") or "").strip()
            if note:
                pending_text.append(note)
                _emit_commentary()
            return
        if event_type == "system" and event.get("subtype") == "compact_boundary":
            meta = event.get("compact_metadata") or {}
            before, after = meta.get("pre_tokens"), meta.get("post_tokens")
            sizes = (
                f" ({round(before / 1000)}k → {round(after / 1000)}k tokens)"
                if isinstance(before, int) and isinstance(after, int)
                else ""
            )
            emit_continuity(
                agent, "claude-code", "compacted",
                f"Claude Code compacted this session's context{sizes}, "
                f"{'automatically' if meta.get('trigger') == 'auto' else 'on request'}. "
                "The full session stays on disk; the transcript here is unchanged.",
                trigger=str(meta.get("trigger") or ""),
                pre_tokens=before, post_tokens=after,
            )
            return
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
            results = [
                block
                for block in ((event.get("message") or {}).get("content") or [])
                if isinstance(block, dict) and block.get("type") == "tool_result"
            ]
            # ``tool_use_result`` describes the event's single tool result
            # (Claude Code sends one result per event); never guess which of
            # several results it belongs to.
            native_result = event.get("tool_use_result") if len(results) == 1 else None
            for block in results:
                _tool_completed(block, native_result)

    return on_event


# Hermes Chat re-supplies up to two recent history images on every send
# because the run API keeps only text for older messages. The browser places
# this note between the turn's own uploads and those copies, and the API
# server recognizes the same prefix. Anything after the note was shared on an
# earlier turn; presenting it as a fresh attachment made plain-text follow-ups
# read as "the user sent another photo".
_REATTACHED_IMAGE_NOTE_PREFIX = "(re-attached for reference"
_REATTACHED_IMAGE_NOTE_RE = re.compile(
    r"\n*[ \t]*\(Re-attached for reference[^\n]*\)[ \t]*"
    r"(?:\n+[ \t]*\[screenshot\][ \t]*)*",
    re.IGNORECASE,
)


def _is_reattached_image_note(text: str) -> bool:
    return text.strip().lower().startswith(_REATTACHED_IMAGE_NOTE_PREFIX)


def strip_reattached_image_notes(text: str) -> str:
    """Drop the browser's re-attach note and the image placeholders after it.

    Stored history renders every image as ``[screenshot]``; the placeholders
    that follow the note stand for the re-supplied copies, not for anything
    the user attached on that turn.
    """
    if not text or _REATTACHED_IMAGE_NOTE_PREFIX not in text.lower():
        return text
    return _REATTACHED_IMAGE_NOTE_RE.sub("", text).strip()


def _materialize_images(
    original_user_message: Any, directory: str
) -> tuple[list[str], list[str]]:
    """Write inline images to ``directory``.

    Returns ``(attached, referenced)`` paths. Images after the re-attach note
    are earlier-turn copies the browser re-supplied for reference.
    """
    if not isinstance(original_user_message, list):
        return [], []
    attached: list[str] = []
    referenced: list[str] = []
    after_note = False
    for index, part in enumerate(original_user_message):
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type in {"text", "input_text"}:
            if _is_reattached_image_note(str(part.get("text") or "")):
                after_note = True
            continue
        if part_type not in {"image_url", "input_image"}:
            continue
        image_url = part.get("image_url")
        value = image_url.get("url") if isinstance(image_url, dict) else image_url
        if not isinstance(value, str) or not value.startswith("data:image/"):
            continue
        try:
            header, encoded = value.split(",", 1)
            media_type = header[5:].split(";", 1)[0].lower()
            suffix = {
                "image/png": ".png",
                "image/jpeg": ".jpg",
                "image/webp": ".webp",
                "image/gif": ".gif",
            }.get(media_type)
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error):
            continue
        if suffix is None or len(data) > 10 * 1024 * 1024:
            continue
        path = Path(directory, f"hermes-image-{index}{suffix}")
        path.write_bytes(data)
        (referenced if after_note else attached).append(str(path))
    return attached, referenced


_CLAUDE_HERMES_TOOL_NAMES = {
    "Bash": "terminal",
    "Edit": "patch",
    "MultiEdit": "patch",
    "NotebookEdit": "patch",
    "Write": "write_file",
    "TodoWrite": "todo",
    "Read": "read_file",
    "Grep": "search_files",
    "Glob": "search_files",
}
_CLAUDE_FILE_CHANGE_TOOLS = ("Edit", "MultiEdit", "NotebookEdit", "Write")


def _claude_hermes_call(
    raw_name: str, args: dict[str, Any], result: str
) -> tuple[str, dict[str, Any], str, Optional[str]]:
    """(hermes_name, args, result, changed_path) for one completed Claude Code call.

    The observer hooks and the verify judge know Hermes tool names and the
    todo tool's JSON result, so Claude Code's own tools are projected onto
    those shapes, the same way codex calls are.
    """
    name = _CLAUDE_HERMES_TOOL_NAMES.get(raw_name)
    if name is None:
        name = raw_name.split("__")[-1] if raw_name.startswith("mcp__") else (raw_name or "tool")
    changed = None
    if raw_name in _CLAUDE_FILE_CHANGE_TOOLS:
        value = args.get("file_path") or args.get("notebook_path")
        changed = str(value) if isinstance(value, str) and value.strip() else None
    if raw_name == "TodoWrite":
        todos = args.get("todos") if isinstance(args.get("todos"), list) else []
        result = json.dumps(
            {
                "todos": [
                    {
                        "id": str(index + 1),
                        "content": str(item.get("content") or ""),
                        "status": str(item.get("status") or "pending"),
                    }
                    for index, item in enumerate(todos)
                    if isinstance(item, dict)
                ]
            }
        )
    elif raw_name == "Bash":
        args = {"command": str(args.get("command") or "")}
    return name, args, result, changed


def _claude_file_change_path(agent: Any, value: str) -> Optional[Path]:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = Path(getattr(agent, "session_cwd", None) or Path.cwd()) / candidate
    try:
        return candidate.resolve(strict=False)
    except (OSError, RuntimeError):
        return None


def _hook_result_text(value: Any) -> str:
    """The text of a Claude Code hook's ``tool_response`` in the shape the observer hooks read."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        text = _content_text(value)
        return text if text else json.dumps(value, ensure_ascii=False, default=str)
    if isinstance(value, dict):
        if "stdout" in value or "stderr" in value:
            parts = [str(value.get("stdout") or ""), str(value.get("stderr") or "")]
            return "\n".join(part for part in parts if part).strip()
        content = value.get("content")
        if isinstance(content, (str, list)):
            return _hook_result_text(content)
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def handle_claude_hook_event(binding: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
    """Dispatch one Claude Code hook payload to the plugin tool hooks.

    Called by the gateway's ``/v1/hooks/claude`` endpoint (see
    :mod:`hermes_cli.claude_hooks`) with the token's binding and the hook
    script's request body. ``PreToolUse`` runs ``pre_tool_call``: a block
    directive returns ``{"action": "block", "message"}``, anything else
    ``{"action": "pass"}`` (an ``approve`` directive is not escalated on
    this lane; it passes). ``PostToolUse`` runs ``post_tool_call`` with
    ``steerable=True`` and returns the first hook result's ``message`` as
    ``{"message"}``, else ``{}``. Tool names, arguments and results are
    projected onto the Hermes shapes the hooks know, as the parity replay
    does. Never raises.
    """
    event = str(body.get("event") or "")
    raw_name = str(body.get("tool_name") or "")
    tool_input = body.get("tool_input") if isinstance(body.get("tool_input"), dict) else {}
    call_id = str(body.get("tool_use_id") or "")
    cwd = str(body.get("cwd") or "")
    session_id = str(binding.get("session_id") or "")
    turn_id = str(binding.get("turn_id") or "")
    try:
        if event == "PreToolUse":
            name, args, _result, _changed = _claude_hermes_call(raw_name, dict(tool_input), "")
            from hermes_cli.plugins import get_pre_tool_call_directive

            directive, message = get_pre_tool_call_directive(
                name, args, session_id=session_id, tool_call_id=call_id, turn_id=turn_id
            )
            if directive == "block" and message:
                return {"action": "block", "message": str(message)}
            return {"action": "pass"}
        if event == "PostToolUse":
            from hermes_cli import claude_hooks
            from hermes_cli.lifecycle import has_hook, invoke_hook

            claude_hooks.note_live_call(str(binding.get("token") or ""))
            result = redact_sensitive_text(_hook_result_text(body.get("tool_response")), force=True)
            name, args, result, _changed = _claude_hermes_call(raw_name, dict(tool_input), result)
            if not has_hook("post_tool_call"):
                return {}
            try:
                duration_ms = int(body.get("duration_ms") or 0)
            except (TypeError, ValueError):
                duration_ms = 0
            results = invoke_hook(
                "post_tool_call",
                tool_name=name,
                args=args,
                result=result,
                task_id="",
                session_id=session_id,
                tool_call_id=call_id,
                turn_id=turn_id,
                api_request_id="",
                duration_ms=duration_ms,
                status="ok",
                error_type=None,
                error_message=None,
                middleware_trace=[],
                steerable=True,
                cwd=cwd,
            )
            for item in results or []:
                message = item.get("message") if isinstance(item, dict) else None
                if isinstance(message, str) and message.strip():
                    return {"message": message.strip()}
            return {}
    except Exception:
        logger.debug("Claude hook dispatch failed", exc_info=True)
    return {"action": "pass"} if event == "PreToolUse" else {}


def _claude_hook_binding(agent: Any, *, read_only: bool) -> tuple[Optional[str], dict[str, str], Optional[str]]:
    """(token, process environment, settings JSON) for the live hook channel, or (None, {}, None).

    The token is created once per agent and rebound every turn so a resident
    process, whose environment was fixed at spawn, keeps resolving to the
    agent currently serving the session. Read-only sessions run in safe
    mode, which disables hooks, so they get none. Never raises.
    """
    try:
        from hermes_cli import claude_hooks

        url = claude_hooks.hook_endpoint()
        if not url or read_only:
            return None, {}, None
        token = str(getattr(agent, "_claude_hook_token", "") or "") or claude_hooks.new_hook_token()
        agent._claude_hook_token = token
        claude_hooks.bind_hook_token(token, session_id=getattr(agent, "session_id", "") or "", agent=agent)
        claude_hooks.reset_live_calls(token)
        settings = json.dumps(claude_hooks.hook_settings(), separators=(",", ":"))
        return token, claude_hooks.hook_environment(token, url), settings
    except Exception:
        logger.debug("Claude hook binding failed", exc_info=True)
        return None, {}, None


def _live_hook_calls(hook_token: Optional[str]) -> int:
    if not hook_token:
        return 0
    try:
        from hermes_cli import claude_hooks

        return claude_hooks.live_calls(hook_token)
    except Exception:
        return 0


def _claude_hook_parity(
    agent: Any,
    session: Any,
    turn: Any,
    messages: list[dict[str, Any]],
    calls: list[tuple[str, str, dict[str, Any], str, str]],
    original_user_message: Any,
    effective_task_id: str,
    hook_token: Optional[str] = None,
) -> int:
    """Give a Claude turn the observer hooks and the verify gate of the default loop.

    Claude Code executes tools inside its own process, so ``post_tool_call``,
    ``post_llm_call`` and ``pre_verify`` never fire on this path unless they
    are replayed from the calls the event bridge saw. The verify gate may
    keep the turn going: a ``continue`` directive is sent to the same Claude
    session as one more turn, the attempted answer stays in the transcript
    as an interim message, and the turn object is updated to the follow-up's
    outcome. Bounded by ``agent.max_verify_nudges``. Returns the number of
    follow-up turns run. Mirrors ``agent.codex_runtime._codex_hook_parity``.
    """
    from hermes_cli.lifecycle import has_hook, invoke_hook

    session_id = getattr(agent, "session_id", "") or ""
    turn_id = getattr(agent, "_current_turn_id", "") or ""
    platform = getattr(agent, "platform", "") or ""
    model = getattr(agent, "model", "") or ""
    changed: set[str] = set()
    replayed = 0

    def emit_tool_hooks() -> None:
        nonlocal replayed
        pending = list(calls[replayed:])
        replayed = len(calls)
        # When the process's own hooks delivered the calls live through the
        # gateway (hermes_cli.claude_hooks), the observers already saw them;
        # only the changed paths are still needed here for the verify gate.
        live = _live_hook_calls(hook_token) > 0
        for raw_name, _name, args, result, call_id in pending:
            name, hermes_args, hermes_result, changed_path = _claude_hermes_call(
                raw_name, args if isinstance(args, dict) else {}, result
            )
            if changed_path:
                path = _claude_file_change_path(agent, changed_path)
                if path is not None:
                    changed.add(str(path))
            if live or not has_hook("post_tool_call"):
                continue
            try:
                invoke_hook(
                    "post_tool_call",
                    tool_name=name,
                    args=hermes_args,
                    result=hermes_result,
                    task_id=effective_task_id or "",
                    session_id=session_id,
                    tool_call_id=call_id,
                    turn_id=turn_id,
                    api_request_id="",
                    duration_ms=0,
                    status="ok",
                    error_type=None,
                    error_message=None,
                    middleware_trace=[],
                    replay=True,
                )
            except Exception:
                logger.debug("Claude post_tool_call parity failed", exc_info=True)

    emit_tool_hooks()

    follow_ups = 0
    try:
        from agent.verify_hooks import max_verify_nudges

        limit = int(max_verify_nudges())
    except Exception:
        limit = 0
    attempt = int(getattr(agent, "_pre_verify_nudges", 0) or 0)
    while (
        changed
        and attempt < limit
        and session is not None
        and isinstance(turn.final_text, str)
        and turn.final_text.strip()
        and not turn.interrupted
        and turn.error is None
        and has_hook("pre_verify")
    ):
        from hermes_cli.plugins import get_pre_verify_continue_message

        nudge = get_pre_verify_continue_message(
            session_id=session_id,
            platform=platform,
            model=model,
            coding=True,
            attempt=attempt,
            final_response=turn.final_text,
            changed_paths=sorted(changed),
        )
        if not nudge:
            break
        attempt += 1
        agent._pre_verify_nudges = attempt
        from agent.message_metadata import append_message

        # The attempted answer stays visible as commentary and in the
        # transcript; the nudge is synthetic and stripped from the durable
        # transcript, exactly as the default loop does.
        emit = getattr(agent, "_emit_interim_assistant_message", None)
        if emit:
            try:
                emit({"role": "assistant", "content": turn.final_text})
            except Exception:
                logger.debug("Claude interim answer callback failed", exc_info=True)
        append_message(messages, {"role": "assistant", "content": turn.final_text})
        append_message(
            messages, {"role": "user", "content": nudge, "_pre_verify_synthetic": True}
        )
        try:
            follow = session.run_turn(nudge)
        except Exception:
            logger.warning("Claude pre_verify follow-up turn failed", exc_info=True)
            break
        follow_ups += 1
        emit_tool_hooks()
        turn.final_text = follow.final_text
        turn.tool_iterations = int(turn.tool_iterations or 0) + int(follow.tool_iterations or 0)
        for key, value in (follow.usage or {}).items():
            if isinstance(value, (int, float)):
                turn.usage[key] = turn.usage.get(key, 0) + value
        turn.interrupted = bool(turn.interrupted or follow.interrupted)
        turn.error = follow.error
        turn.error_code = follow.error_code
        turn.should_retire = bool(turn.should_retire or follow.should_retire)
        turn.session_confirmed = bool(turn.session_confirmed or follow.session_confirmed)
        turn.watchdog_retries = int(turn.watchdog_retries or 0) + int(follow.watchdog_retries or 0)
        if follow.session_id:
            turn.session_id = follow.session_id

    if (
        has_hook("post_llm_call")
        and isinstance(turn.final_text, str)
        and turn.final_text.strip()
        and not turn.interrupted
    ):
        try:
            invoke_hook(
                "post_llm_call",
                session_id=session_id,
                task_id=effective_task_id or "",
                turn_id=turn_id,
                user_message=original_user_message,
                assistant_response=turn.final_text,
                conversation_history=list(messages),
                model=model,
                platform=platform,
            )
        except Exception:
            logger.debug("Claude post_llm_call parity failed", exc_info=True)
    return follow_ups


_RETENTION_CHECKED = False


def _announce_claude_continuity(
    agent: Any,
    *,
    prior_messages: list[dict[str, Any]],
    rebuild_reason: str,
    resumed_from_disk: bool,
    delta: list[dict[str, Any]],
    session_id: str,
    resume_cause: str = "",
) -> None:
    """Show the user every turn that did not simply continue its session."""
    short = session_id[:8]
    if rebuild_reason:
        carried, omitted = claude_handoff_coverage(prior_messages)
        if not carried and not omitted:
            return  # A new conversation: nothing was lost.
        text = (
            f"Session rebuilt from the stored transcript: {rebuild_reason}. "
            f"The new Claude session was given {plural(carried, 'message')}"
            + (f"; the {plural(omitted, 'oldest message')} did not fit." if omitted else ".")
        )
        emit_continuity(
            agent, "claude-code", "rebuilt", text,
            reason=rebuild_reason, carried=carried, omitted=omitted,
            previous_session_id=session_id,
        )
        return
    if resumed_from_disk:
        text = (
            f"Session resumed from disk: Claude session {short}"
            + (f" ({resume_cause})" if resume_cause else "")
            + (
                f", plus the {plural(len(delta), 'message')} added since its last turn."
                if delta else "."
            )
        )
        emit_continuity(
            agent, "claude-code", "resumed", text,
            caught_up=len(delta), session_id=session_id, cause=resume_cause,
        )
        return
    if delta:
        emit_continuity(
            agent, "claude-code", "caught_up",
            f"Passed {plural(len(delta), 'message')} added since this session's last turn to Claude session {short}.",
            caught_up=len(delta), session_id=session_id,
        )


def run_claude_code_turn(
    agent: Any,
    *,
    user_message: str,
    original_user_message: Any,
    messages: list[dict[str, Any]],
    effective_task_id: str,
    should_review_memory: bool = False,
    plugin_user_context: str = "",
) -> dict[str, Any]:
    """Run one Claude subscription turn and return the standard agent result."""
    del should_review_memory
    from agent.deadline import resolve_timeout
    from agent.runtime_cwd import resolve_agent_cwd
    from agent.transports.claude_code_session import (
        DEFAULT_RESIDENT_FIRST_EVENT_TIMEOUT,
        DEFAULT_STARTUP_FIRST_EVENT_TIMEOUT,
        ClaudeCodeSession,
    )

    global _RETENTION_CHECKED
    if not _RETENTION_CHECKED:
        # Before any CLI starts: Claude Code sweeps old transcripts at
        # startup, and a swept transcript can never be resumed.
        _RETENTION_CHECKED = True
        ensure_claude_transcript_retention()
    cwd = getattr(agent, "session_cwd", None) or str(resolve_agent_cwd())
    cwd = _normalized_cwd(cwd)
    prior_messages = messages[:-1]
    prior_fingerprint = _claude_history_fingerprint(prior_messages)
    prior_state = _load_claude_session_state(agent)
    prior_session_id = str(prior_state.get("session_id") or "").strip()
    durable_continues, durable_delta = _claude_history_continues(
        prior_state.get("history_fingerprint"), prior_messages
    )
    # ``--resume`` of a transcript Claude Code no longer has fails the turn;
    # rebuilding from the stored transcript is the step down instead.
    prior_session_file = claude_session_file(prior_session_id) if prior_session_id else None
    durable_resume = bool(
        prior_session_id
        and prior_state.get("version") == _CLAUDE_SESSION_STATE_VERSION
        and prior_state.get("cwd") == cwd
        and durable_continues
        and prior_session_file is not None
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
    resident_continues, resident_delta = _claude_history_continues(
        getattr(session, "history_fingerprint", None) if session is not None else None,
        prior_messages,
    )
    resident_continuity = bool(
        session is not None
        and session.compatible_with(
            cwd=cwd,
            model=model,
            effort=effort,
            system_prompt=runtime_contract,
            read_only=read_only,
        )
        and resident_continues
    )
    resume_cause = (
        "the model or effort changed"
        if session is not None and resident_continues and not resident_continuity
        else "the CLI process was not running"
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
    continuity_delta = resident_delta if resident_continuity else durable_delta if durable_resume else []
    rebuild_reason = ""
    if not resident_continuity and not durable_resume:
        # Every one of these starts a new Claude session seeded with the
        # stored transcript. Say why, in the log and in the conversation, so
        # a lost session is never silent (2026-09-25).
        rebuild_reason = (
            "no Claude session was recorded for this conversation on this runtime"
            if not prior_session_id
            else "its session record is from an older Hermes version"
            if prior_state.get("version") != _CLAUDE_SESSION_STATE_VERSION
            else f"the working directory changed ({prior_state.get('cwd')} -> {cwd})"
            if prior_state.get("cwd") != cwd
            else "the transcript was edited or rolled back since its last turn"
            if not durable_continues
            else "its transcript file is gone from Claude Code's session store"
        )
        logger.warning(
            "Claude session %s not continued for Hermes session %s: %s; starting a new session with a transcript handoff",
            prior_session_id or "(none)",
            getattr(agent, "session_id", ""),
            rebuild_reason,
        )
    _announce_claude_continuity(
        agent,
        prior_messages=prior_messages,
        rebuild_reason=rebuild_reason,
        resumed_from_disk=bool(durable_resume and not resident_continuity),
        resume_cause=resume_cause,
        delta=continuity_delta,
        session_id=prior_session_id,
    )

    with tempfile.TemporaryDirectory(prefix="hermes-claude-images-") as image_dir:
        image_content = (
            original_user_message
            if isinstance(original_user_message, list)
            else user_message
        )
        attached_images, reference_images = _materialize_images(
            image_content, image_dir
        )
        prompt_text = strip_reattached_image_notes(
            user_message
            if isinstance(user_message, str)
            else _content_text(user_message)
        )
        if not prompt_text and attached_images:
            prompt_text = "Please inspect the attached image(s)."
        if resident_continuity or durable_resume:
            prompt = (
                claude_history_handoff(continuity_delta, prompt_text, lead=CLAUDE_DELTA_LEAD)
                if continuity_delta
                else prompt_text
            )
        else:
            prompt = claude_history_handoff(
                prior_messages,
                prompt_text,
                lead=CLAUDE_REBUILD_LEAD.format(reason=rebuild_reason),
            )
        sections = [prompt]
        if plugin_user_context:
            # The pre_llm_call hook's note (the preflight judge, gateway
            # notices) rides the user message here as it does on the default
            # loop, where it is stamped on the API copy of the message.
            sections.append(plugin_user_context)
        if attached_images:
            sections.append(
                "Attached images are available at:\n"
                + "\n".join(f"- {path}" for path in attached_images)
            )
        if reference_images:
            sections.append(
                "Images the user shared on earlier turns, re-supplied for "
                "reference only. They are not new attachments; the current "
                "request is the text above:\n"
                + "\n".join(f"- {path}" for path in reference_images)
            )
        prompt = "\n\n".join(sections)

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

        calls: list[tuple[str, str, dict[str, Any], str, str]] = []

        def _record(raw_name: str, name: str, args: dict[str, Any], result: str, call_id: str) -> None:
            calls.append((raw_name, name, args, result, call_id))

        bridge = make_claude_code_event_bridge(agent, record=_record, cwd=cwd)
        hook_token, hook_env, hook_settings = _claude_hook_binding(agent, read_only=read_only)

        def _late_answer(record: dict[str, Any]) -> None:
            save_claude_late_answer(agent, record)

        late_context = {"hermes_session_id": str(getattr(agent, "session_id", "") or "")}
        if session is None:
            session = ClaudeCodeSession(
                cwd=cwd,
                model=model,
                session_id=prior_session_id if durable_resume else None,
                resume=durable_resume,
                effort=effort,
                system_prompt=runtime_contract,
                read_only=read_only,
                on_event=bridge,
                on_session_id=_remember_confirmed_session,
                on_watchdog_timeout=_watchdog_timeout,
                on_late_result=_late_answer,
                resident_first_event_timeout=resident_first_event_timeout,
                startup_first_event_timeout=startup_first_event_timeout,
                extra_env=hook_env,
                settings_json=hook_settings,
            )
            session.late_answer_context = late_context
            session.history_fingerprint = prior_fingerprint
            agent._claude_code_session = session
        else:
            # Event callbacks are per outer turn even though the native process
            # is per conversation. Rebind them before sending the next message.
            session.on_event = bridge
            session.on_session_id = _remember_confirmed_session
            session.on_watchdog_timeout = _watchdog_timeout
            session.on_late_result = _late_answer
            session.late_answer_context = late_context
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

    # Hook parity with the default loop (observer hooks + the pre_verify
    # gate), which may extend this turn with bounded follow-up turns.
    follow_ups = 0
    try:
        follow_ups = _claude_hook_parity(
            agent, session, turn, messages, calls, original_user_message, effective_task_id,
            hook_token=hook_token,
        )
    except Exception:
        logger.debug("Claude hook parity failed", exc_info=True)

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
        "api_calls": 1 + follow_ups,
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
