"""Codex API runtime — App Server and Responses-API streaming paths.

Extracted from :class:`AIAgent` to keep the agent loop file focused.
Each function takes the parent ``AIAgent`` as its first argument
(``agent``).  AIAgent keeps thin forwarder methods for backward
compatibility.

* ``run_codex_app_server_turn`` — drives one turn through the
  ``codex_app_server`` subprocess client (used when a Codex CLI install
  is the active provider).
* ``run_codex_stream`` — streams a Codex Responses API call (the
  ``codex_responses`` api_mode).
* ``run_codex_create_stream_fallback`` — recovery path when the
  Responses ``stream=True`` initial create fails.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from difflib import SequenceMatcher
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List

from agent.stream_single_writer import claim_stream_writer, stream_writer_is_current
from agent.transports.codex_coordination import COORDINATION_ITEM_TYPES, coordination_item

logger = logging.getLogger(__name__)


_CODEX_APP_SERVER_THREAD_ID_KEY = "_codex_app_server_thread_id"
_CODEX_APP_SERVER_THREAD_STATE_KEY = "_codex_app_server_thread_state"
_CODEX_APP_SERVER_THREAD_STATE_VERSION = 1
_CODEX_HISTORY_HANDOFF_MAX_CHARS = 240_000


def _stored_codex_app_server_thread_id(agent: Any) -> str:
    """Return the native Codex thread bound to this durable Hermes session."""
    session_db = getattr(agent, "_session_db", None)
    session_id = str(getattr(agent, "session_id", "") or "")
    getter = getattr(session_db, "get_session_model_config_value", None)
    if not session_id or not callable(getter):
        return ""
    try:
        value = getter(session_id, _CODEX_APP_SERVER_THREAD_ID_KEY, "")
    except Exception:
        logger.warning(
            "could not read persisted Codex thread for Hermes session %s",
            session_id,
            exc_info=True,
        )
        return ""
    value = str(value or "").strip()
    if not value or len(value) > 200:
        return ""
    return value


def _persist_codex_thread_state(
    agent: Any, *, thread_id: str, cwd: str, entries: List[tuple]
) -> None:
    """Bind a native Codex thread to this Hermes session *and* to the exact
    transcript prefix that thread has already consumed.

    A thread id alone cannot answer the only question that matters on resume:
    is this thread current? Native threads stay resumable indefinitely, so a
    conversation whose intervening turns ran on another runtime would quietly
    continue from a stale tail with no signal to the user. Recording what the
    thread has seen makes that divergence detectable — and, when the seen
    prefix still matches, makes the gap recoverable as a bounded delta instead
    of discarding the thread.
    """
    session_db = getattr(agent, "_session_db", None)
    session_id = str(getattr(agent, "session_id", "") or "")
    patcher = getattr(session_db, "patch_session_model_config", None)
    if not thread_id:
        return
    state = {
        "version": _CODEX_APP_SERVER_THREAD_STATE_VERSION,
        "thread_id": thread_id,
        "cwd": _normalized_codex_cwd(cwd),
        "seen_count": len(entries),
        "fingerprint": _codex_history_fingerprint(entries),
        "updated_at": time.time(),
    }
    # Keep the in-process copy current even when there is no session DB, so a
    # resident agent still evaluates continuity against real state.
    agent._codex_thread_state = state
    if not session_id or not callable(patcher):
        return
    try:
        # One patch: the id and its provenance must never be written apart,
        # or a crash between them leaves a thread that looks verified against
        # the wrong transcript.
        patcher(
            session_id,
            {
                _CODEX_APP_SERVER_THREAD_ID_KEY: thread_id,
                _CODEX_APP_SERVER_THREAD_STATE_KEY: state,
            },
        )
    except Exception:
        logger.warning(
            "could not persist Codex thread %s for Hermes session %s",
            thread_id[:8],
            session_id,
            exc_info=True,
        )


def _load_codex_thread_state(agent: Any) -> Dict[str, Any]:
    """Return the recorded provenance of this session's bound Codex thread."""
    state = getattr(agent, "_codex_thread_state", None)
    session_db = getattr(agent, "_session_db", None)
    session_id = str(getattr(agent, "session_id", "") or "")
    getter = getattr(session_db, "get_session_model_config_value", None)
    if session_id and callable(getter):
        try:
            stored = getter(session_id, _CODEX_APP_SERVER_THREAD_STATE_KEY, None)
            if isinstance(stored, dict):
                state = stored
        except Exception:
            logger.warning(
                "could not read Codex thread state for Hermes session %s",
                session_id,
                exc_info=True,
            )
    return dict(state) if isinstance(state, dict) else {}


def _normalized_codex_cwd(cwd: Any) -> str:
    try:
        return str(Path(str(cwd or "")).expanduser().resolve())
    except Exception:
        return str(cwd or "")


def _history_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    return "\n".join(
        str(part.get("text") or "").strip()
        for part in content
        if isinstance(part, dict) and part.get("type") in {"text", "input_text"}
    ).strip()


def _codex_dialogue_entries(history: List[Dict[str, Any]]) -> List[tuple]:
    """Reduce a Hermes transcript to the dialogue a native thread can hold.

    Tool and system rows are excluded deliberately: they are projected
    differently depending on which runtime produced them, so including them
    would make the fingerprint below report divergence for transcripts that
    are in fact identical dialogue.
    """
    try:
        from agent.memory_manager import sanitize_context
    except Exception:  # pragma: no cover - defensive
        sanitize_context = None

    entries: List[tuple] = []
    for message in history:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").lower()
        if role not in {"user", "assistant"}:
            continue
        content = message.get("content")
        if isinstance(content, str) and sanitize_context is not None:
            # Mirror the storage layer, which sanitizes user/assistant strings
            # on load (hermes_state._rows_to_conversation). Recalled memory is
            # injected into the live user message but stripped on reload, so
            # fingerprinting the raw in-memory text would report divergence for
            # a transcript that is in fact unchanged — and cost the thread on
            # every single turn.
            content = sanitize_context(content)
        text = _history_text(content)
        if not text:
            continue
        entries.append((role, text))
    return entries


def _codex_history_fingerprint(entries: List[tuple]) -> str:
    """Hash the dialogue prefix a native Codex thread represents."""
    digest = hashlib.sha256()
    for role, text in entries:
        digest.update(
            json.dumps([role, text], ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8", errors="replace"
            )
        )
        digest.update(b"\n")
    return f"v1:{len(entries)}:{digest.hexdigest()}"


def _codex_resume_plan(
    *,
    thread_id: str,
    state: Dict[str, Any],
    prior_entries: List[tuple],
    cwd: str,
) -> tuple:
    """Decide whether a bound native thread may continue this transcript.

    Returns ``(resume_thread_id, pending_entries, reason)``. An empty
    ``resume_thread_id`` means the thread cannot be proven current, so the
    caller must start a fresh one seeded with ``pending_entries`` — the whole
    transcript. When resume is allowed, ``pending_entries`` is only the delta
    this thread has never seen, which is empty for an ordinary next turn.
    """
    if not thread_id:
        return "", prior_entries, "no-bound-thread"
    if (
        not isinstance(state, dict)
        or state.get("version") != _CODEX_APP_SERVER_THREAD_STATE_VERSION
    ):
        # Threads bound before this record existed carry no proof of what they
        # consumed. Treat unknown provenance as divergence: re-seeding costs
        # one handoff, while trusting it costs the user their context.
        return "", prior_entries, "no-recorded-history"
    if str(state.get("thread_id") or "") != thread_id:
        return "", prior_entries, "thread-rebound"
    if str(state.get("cwd") or "") != _normalized_codex_cwd(cwd):
        return "", prior_entries, "cwd-changed"
    seen = state.get("seen_count")
    if isinstance(seen, bool) or not isinstance(seen, int) or seen < 0:
        return "", prior_entries, "unusable-seen-count"
    if seen > len(prior_entries):
        # The transcript is shorter than what the thread consumed — a rewrite
        # or compaction, not an append.
        return "", prior_entries, "transcript-shortened"
    if _codex_history_fingerprint(prior_entries[:seen]) != str(
        state.get("fingerprint") or ""
    ):
        return "", prior_entries, "transcript-diverged"
    return thread_id, list(prior_entries[seen:]), "resume"


def _render_history_blocks(entries: List[tuple], budget: int) -> tuple:
    """Render the newest entries that fit, returning ``(blocks, omitted, truncated)``."""
    rendered: list[str] = []
    used = 0
    omitted = 0
    truncated = False
    for index in range(len(entries) - 1, -1, -1):
        role, text = entries[index]
        block = f"{role.upper()}:\n{text}"
        block_truncated = False
        if len(block) > budget:
            block = block[-budget:]
            block_truncated = True
        if used + len(block) > budget:
            # Stop here rather than skipping this message and continuing with
            # older, smaller ones. Skipping punches a hole in the middle of the
            # transcript and discloses it only as a count, which reads as
            # "older context omitted" when it is really "a reply you are about
            # to see is missing". An unbroken recent window is the honest cut.
            omitted = index + 1
            break
        rendered.append(block)
        used += len(block)
        truncated = truncated or block_truncated
    rendered.reverse()
    return rendered, omitted, truncated


def _handoff_disclosure(omitted: int, truncated: bool) -> str:
    notes: list[str] = []
    if omitted:
        notes.append(
            f"[{omitted} older messages omitted — handoff size limit reached.]"
        )
    if truncated:
        notes.append(
            "[The oldest included message was cut at its start to fit.]"
        )
    return ("\n" + "\n".join(notes)) if notes else ""


def _wrap_turn_input(context: str, user_message: Any, closing: str) -> Any:
    if isinstance(user_message, list):
        return [
            {"type": "text", "text": context},
            *user_message,
            {"type": "text", "text": closing},
        ]
    return context + str(user_message) + closing


def _codex_history_handoff(entries: List[tuple], user_message: Any) -> Any:
    """Seed a fresh native thread when no persisted Codex thread can resume.

    The app-server runtime accepts only the current turn at ``turn/start``.
    Preserve complete recent user/assistant messages inside a disclosed
    transcript rather than silently dropping client-managed history.
    """
    rendered, omitted, truncated = _render_history_blocks(
        entries, _CODEX_HISTORY_HANDOFF_MAX_CHARS
    )
    if not rendered:
        return user_message
    context = (
        "Hermes is restoring an existing conversation into a fresh native "
        "Codex thread. Use the following transcript as prior conversation "
        "context, then execute the current user request. Do not ask the user "
        "to reselect a project already established here.\n\n"
        "<hermes_conversation_history>\n"
        + "\n\n".join(rendered)
        + _handoff_disclosure(omitted, truncated)
        + "\n</hermes_conversation_history>\n\n"
        "<current_user_request>\n"
    )
    return _wrap_turn_input(context, user_message, "\n</current_user_request>")


def _codex_catch_up_handoff(entries: List[tuple], user_message: Any) -> Any:
    """Hand a resumed native thread only the messages it never saw.

    Switching models mid-conversation leaves the native thread intact but
    behind. Re-seeding the whole transcript would throw away the thread's own
    reasoning state; sending nothing strands it at a stale tail. The delta
    keeps both.
    """
    rendered, omitted, truncated = _render_history_blocks(
        entries, _CODEX_HISTORY_HANDOFF_MAX_CHARS
    )
    if not rendered:
        return user_message
    context = (
        "Hermes is continuing this native Codex thread. The messages below "
        "happened in this same conversation while a different runtime was "
        "answering, so this thread never received them. Treat them as prior "
        "conversation context the user considers already said — do not "
        "re-ask for anything they settle — then execute the current user "
        "request.\n\n"
        "<hermes_missed_messages>\n"
        + "\n\n".join(rendered)
        + _handoff_disclosure(omitted, truncated)
        + "\n</hermes_missed_messages>\n\n"
        "<current_user_request>\n"
    )
    return _wrap_turn_input(context, user_message, "\n</current_user_request>")


def _codex_request_failure_details(error: BaseException) -> tuple[int | None, str]:
    """Return the serialized request size and exception class chain.

    OpenAI connection exceptions retain the final ``httpx.Request``. Reading
    its already-buffered content gives us the exact byte count handed to the
    transport without logging any request content. The class-only chain keeps
    the underlying transport failure visible without exposing URLs or payloads
    from exception messages.
    """
    request_body_bytes: int | None = None
    exception_classes: list[str] = []
    current: BaseException | None = error
    seen: set[int] = set()

    while current is not None and id(current) not in seen and len(seen) < 8:
        seen.add(id(current))
        exception_classes.append(type(current).__name__)

        if request_body_bytes is None:
            try:
                request = getattr(current, "request", None)
            except Exception:
                request = None
            if request is not None:
                try:
                    content = request.content
                except Exception:
                    content = None
                if isinstance(content, str):
                    request_body_bytes = len(content.encode("utf-8"))
                elif isinstance(content, (bytes, bytearray, memoryview)):
                    request_body_bytes = len(content)

        cause = current.__cause__
        if cause is None and not current.__suppress_context__:
            cause = current.__context__
        current = cause

    return request_body_bytes, " <- ".join(exception_classes)


def _log_codex_request_failure(
    agent: Any,
    error: BaseException,
    *,
    stream_opened: bool,
) -> None:
    request_body_bytes, exception_chain = _codex_request_failure_details(error)
    logger.warning(
        "Codex Responses request failed: "
        "serialized_request_body_bytes=%s stream_opened=%s "
        "exception_chain=%s model=%s",
        request_body_bytes if request_body_bytes is not None else "unknown",
        str(stream_opened).lower(),
        exception_chain,
        getattr(agent, "model", "unknown"),
    )


def _coerce_usage_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float):
        return max(int(value), 0)
    if isinstance(value, str):
        try:
            return max(int(value), 0)
        except ValueError:
            return 0
    return 0


def _record_codex_app_server_usage(agent, turn) -> dict[str, Any]:
    """Translate Codex app-server token usage into Hermes accounting.

    Codex app-server reports usage via thread/tokenUsage/updated as:
    inputTokens, cachedInputTokens, outputTokens, reasoningOutputTokens,
    totalTokens.

    Hermes' canonical prompt bucket includes uncached input + cached input.
    The Codex app-server protocol does not currently expose cache-write tokens,
    so that bucket remains zero on this runtime.

    Even when Codex omits usage for a turn, Hermes should still count that turn
    as one API call for session/status accounting.
    """
    agent.session_api_calls += 1

    usage = getattr(turn, "token_usage_last", None)
    if not isinstance(usage, dict) or not usage:
        compressor = getattr(agent, "context_compressor", None)
        if (
            compressor is not None
            and getattr(compressor, "awaiting_real_usage_after_compression", False)
        ):
            # No usage means this turn cannot adjudicate the pending compaction.
            # Consume the marker so a later unrelated reading is not charged to
            # it and preflight deferral cannot stay latched indefinitely.
            compressor.update_from_response({})
        if agent._session_db and agent.session_id:
            try:
                if not agent._session_db_created:
                    agent._ensure_db_session()
                # Enqueued for the SessionDB background writer — keeps the
                # per-call accounting write off the turn thread (see
                # conversation_loop's queue_token_counts call).
                agent._session_db.queue_token_counts(
                    agent.session_id,
                    model=agent.model,
                    billing_provider=agent.provider,
                    billing_base_url=agent.base_url,
                    billing_mode="subscription_included",
                    api_call_count=1,
                )
            except Exception as exc:
                logger.debug(
                    "Codex app-server api-call persistence failed (session=%s): %s",
                    agent.session_id, exc,
                )
        return {}

    from agent.usage_pricing import CanonicalUsage, estimate_usage_cost

    input_tokens = _coerce_usage_int(usage.get("inputTokens"))
    cache_read_tokens = _coerce_usage_int(usage.get("cachedInputTokens"))
    output_tokens = _coerce_usage_int(usage.get("outputTokens"))
    reasoning_tokens = _coerce_usage_int(usage.get("reasoningOutputTokens"))
    reported_total = _coerce_usage_int(usage.get("totalTokens"))

    canonical_usage = CanonicalUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=0,
        reasoning_tokens=reasoning_tokens,
        raw_usage=usage,
    )
    prompt_tokens = canonical_usage.prompt_tokens
    completion_tokens = canonical_usage.output_tokens
    total_tokens = reported_total or canonical_usage.total_tokens
    usage_dict = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "input_tokens": canonical_usage.input_tokens,
        "output_tokens": canonical_usage.output_tokens,
        "cache_read_tokens": canonical_usage.cache_read_tokens,
        "cache_write_tokens": canonical_usage.cache_write_tokens,
        "reasoning_tokens": canonical_usage.reasoning_tokens,
    }

    compressor = getattr(agent, "context_compressor", None)
    if compressor is not None:
        try:
            compressor.update_from_response(usage_dict)
            context_window = getattr(turn, "model_context_window", None)
            if isinstance(context_window, int) and context_window > 0:
                compressor.context_length = context_window
        except Exception:
            logger.debug("codex app-server usage update failed", exc_info=True)

    agent.session_prompt_tokens += prompt_tokens
    agent.session_completion_tokens += completion_tokens
    agent.session_total_tokens += total_tokens
    agent.session_input_tokens += canonical_usage.input_tokens
    agent.session_output_tokens += canonical_usage.output_tokens
    agent.session_cache_read_tokens += canonical_usage.cache_read_tokens
    agent.session_cache_write_tokens += canonical_usage.cache_write_tokens
    agent.session_reasoning_tokens += canonical_usage.reasoning_tokens

    cost_result = estimate_usage_cost(
        agent.model,
        canonical_usage,
        provider=agent.provider,
        base_url=agent.base_url,
        api_key=getattr(agent, "api_key", ""),
    )
    if cost_result.amount_usd is not None:
        agent.session_estimated_cost_usd += float(cost_result.amount_usd)
    agent.session_cost_status = cost_result.status
    agent.session_cost_source = cost_result.source

    if agent._session_db and agent.session_id:
        try:
            if not agent._session_db_created:
                agent._ensure_db_session()
            # Enqueued for the SessionDB background writer (see above).
            agent._session_db.queue_token_counts(
                agent.session_id,
                input_tokens=canonical_usage.input_tokens,
                output_tokens=canonical_usage.output_tokens,
                cache_read_tokens=canonical_usage.cache_read_tokens,
                cache_write_tokens=canonical_usage.cache_write_tokens,
                reasoning_tokens=canonical_usage.reasoning_tokens,
                estimated_cost_usd=float(cost_result.amount_usd)
                if cost_result.amount_usd is not None else None,
                cost_status=cost_result.status,
                cost_source=cost_result.source,
                billing_provider=agent.provider,
                billing_base_url=agent.base_url,
                billing_mode="subscription_included"
                if cost_result.status == "included" else None,
                model=agent.model,
                api_call_count=1,
            )
        except Exception as exc:
            logger.debug(
                "Codex app-server token persistence failed (session=%s, tokens=%d): %s",
                agent.session_id, total_tokens, exc,
            )

    return {
        **usage_dict,
        "last_prompt_tokens": prompt_tokens,
        "estimated_cost_usd": float(cost_result.amount_usd)
        if cost_result.amount_usd is not None else None,
        "cost_status": cost_result.status,
        "cost_source": cost_result.source,
    }


def _record_codex_app_server_compaction(
    agent,
    turn,
    *,
    approx_tokens: int | None = None,
    force: bool = False,
) -> bool:
    """Record a Codex-native context compaction boundary in Hermes state.

    The app-server owns the compacted thread context, so Hermes should not
    rewrite local transcript rows here; state.db records the boundary via the
    session event/usage counters while preserving the visible transcript.
    """
    if not force and not getattr(turn, "compacted", False):
        return False

    thread_id = getattr(turn, "thread_id", None) or ""
    turn_id = getattr(turn, "turn_id", None) or ""
    logger.info(
        "codex app-server compaction observed: session=%s thread=%s turn=%s force=%s",
        getattr(agent, "session_id", None) or "none",
        thread_id,
        turn_id,
        force,
    )
    if not force:
        try:
            from agent.conversation_compression import COMPACTION_STATUS

            agent._emit_status(COMPACTION_STATUS)
        except Exception:
            pass

    compressor = getattr(agent, "context_compressor", None)
    if compressor is not None:
        compressor.compression_count = getattr(
            compressor, "compression_count", 0
        ) + 1
        compressor.last_compression_rough_tokens = approx_tokens or 0
        # The app server has already completed a real compaction boundary. Its
        # usage update (when supplied) is therefore the same real-vs-real
        # effectiveness verdict used by the normal compression path.
        record_boundary = getattr(
            type(compressor), "record_completed_compaction", None
        )
        if callable(record_boundary):
            # Codex owns this summary. A prior Hermes deterministic-fallback
            # flag must not leak into the native boundary's quality verdict.
            record_boundary(compressor, used_fallback=False)
        elif hasattr(compressor, "_verify_compaction_cleared_threshold"):
            compressor._verify_compaction_cleared_threshold = True
        if not getattr(turn, "token_usage_last", None):
            compressor.last_prompt_tokens = -1
            compressor.last_completion_tokens = 0
            compressor.awaiting_real_usage_after_compression = True

    agent._last_compaction_in_place = False
    try:
        if getattr(agent, "event_callback", None):
            agent.event_callback(
                "session:compress",
                {
                    "platform": getattr(agent, "platform", None) or "",
                    "session_id": getattr(agent, "session_id", None) or "",
                    "old_session_id": "",
                    "in_place": False,
                    "compression_count": getattr(
                        compressor, "compression_count", 0
                    )
                    if compressor is not None
                    else 0,
                    "runtime": "codex_app_server",
                    "thread_id": thread_id,
                    "turn_id": turn_id,
                },
            )
    except Exception:
        logger.debug("event_callback error on codex session:compress", exc_info=True)

    return True


# ---------------------------------------------------------------------------
# Codex app-server → Hermes UI bridge (#33200)
#
# The codex_app_server runtime hands the entire turn to a subprocess and
# bypasses the normal Hermes tool loop. Without this bridge gateway
# adapters (Discord, Telegram, TUI) never see live tool-progress bubbles
# or interim assistant commentary while codex is working — the user just
# stares at a quiet channel until the final answer lands. The bridge
# translates raw codex JSON-RPC notifications into the same three agent
# callbacks the standard runtime fires:
#   - tool_progress_callback("tool.started"|"tool.completed", name, ...)
#   - _fire_stream_delta(text) for streaming agentMessage chunks
#   - _emit_interim_assistant_message({...}) for completed agentMessages
# ---------------------------------------------------------------------------

# Codex item types that map to a Hermes tool_call in the projector (and
# therefore deserve a tool_progress bubble pair). The projector lives in
# agent/transports/codex_event_projector.py — keep these in sync so the
# tool name shown in the UI matches the name recorded in messages.
# webSearch is codex's built-in web search tool — it has no projector
# entry (codex handles it internally) but still deserves a bubble.
_CODEX_TOOL_ITEM_TYPES = frozenset(
    {"commandExecution", "fileChange", "mcpToolCall", "dynamicToolCall", "webSearch",
     "imageGeneration", "imageView"}
) | COORDINATION_ITEM_TYPES

# Internal MCP server that wraps Hermes' native tools for codex. When
# codex calls back through it, the inner dispatch runs in a SEPARATE
# hermes-tools-mcp-server subprocess that has no access to the parent
# agent's tool_progress_callback — so the inner call can never surface
# its own native progress event. The codex-level mcpToolCall event IS
# the display event for those calls; we strip the mcp.hermes-tools.*
# namespacing and emit the bare tool name (web_search, browser_navigate,
# vision_analyze, ...) since the user thinks of these as Hermes tools,
# not as MCP calls.
_INTERNAL_MCP_SERVER = "hermes-tools"


def _codex_item_to_tool_name(item: dict) -> str:
    """Synthetic Hermes tool name for a codex item. Mirrors
    CodexEventProjector so the progress bubble and the projected
    tool_calls entry use the same identifier."""
    item_type = item.get("type") or ""
    if item_type in COORDINATION_ITEM_TYPES:
        return coordination_item(item)[0]
    if item_type == "commandExecution":
        return "exec_command"
    if item_type == "fileChange":
        return "apply_patch"
    if item_type == "mcpToolCall":
        server = item.get("server") or "mcp"
        tool = item.get("tool") or "unknown"
        if server == _INTERNAL_MCP_SERVER:
            return tool
        return f"mcp.{server}.{tool}"
    if item_type == "dynamicToolCall":
        return item.get("tool") or "dynamic"
    if item_type == "webSearch":
        return "web_search"
    if item_type == "imageGeneration":
        return "image_generate"
    if item_type == "imageView":
        return "view_image"
    return item_type or "unknown"


def _codex_item_to_args(item: dict) -> dict:
    """Args dict surfaced to tool_progress_callback("tool.started", ...).
    Mirrors the projector's _project_command / _project_file_change /
    _project_mcp_tool_call / _project_dynamic_tool_call shapes."""
    item_type = item.get("type") or ""
    if item_type in COORDINATION_ITEM_TYPES:
        return coordination_item(item)[1]
    if item_type == "commandExecution":
        return {"command": item.get("command") or "",
                "cwd": item.get("cwd") or ""}
    if item_type == "fileChange":
        return {"changes": [
            {"kind": (c.get("kind") or {}).get("type") or "update",
             "path": c.get("path") or ""}
            for c in (item.get("changes") or []) if isinstance(c, dict)
        ]}
    if item_type in {"mcpToolCall", "dynamicToolCall"}:
        args = item.get("arguments") or {}
        return args if isinstance(args, dict) else {"arguments": args}
    if item_type == "webSearch":
        return {"query": item.get("query") or ""}
    if item_type in {"imageGeneration", "imageView"}:
        return {"path": item.get("savedPath") or item.get("path") or ""}
    return {}


def _codex_item_to_preview(item: dict) -> Any:
    """Tool details for both collapsed and expanded displays.

    The gateway owns redaction and disclosed transport limits. Cutting the
    command here silently also cuts the expanded view and durable archive.
    """
    item_type = item.get("type") or ""
    if item_type in COORDINATION_ITEM_TYPES:
        return coordination_item(item)[4]
    if item_type == "commandExecution":
        cmd = item.get("command") or ""
        return cmd or None
    if item_type == "fileChange":
        paths = [c.get("path") for c in (item.get("changes") or [])
                 if isinstance(c, dict) and c.get("path")]
        if not paths:
            return None
        preview = ", ".join(paths[:3])
        if len(paths) > 3:
            preview += f", +{len(paths) - 3} more"
        return preview
    if item_type in {"mcpToolCall", "dynamicToolCall"}:
        args = item.get("arguments") or {}
        if not isinstance(args, dict) or not args:
            return None
        try:
            return json.dumps(args, ensure_ascii=False)
        except (TypeError, ValueError):
            return None
    if item_type == "webSearch":
        query = item.get("query") or ""
        return query or None
    if item_type in {"imageGeneration", "imageView"}:
        return item.get("savedPath") or item.get("path") or "Generating image"
    return None


def _codex_item_completion_payload(item: dict) -> tuple[str, bool]:
    """Return (result_text, is_error) for a completed codex tool item.
    Mirrors the projector's tool-result content so the bubble shows the
    same outcome string that ends up in the messages list."""
    item_type = item.get("type") or ""
    if item_type in COORDINATION_ITEM_TYPES:
        _, _, result, is_error, _ = coordination_item(item)
        return result, is_error
    if item_type == "commandExecution":
        out = item.get("aggregatedOutput") or ""
        exit_code = item.get("exitCode")
        is_error = bool(exit_code is not None and exit_code != 0)
        if is_error:
            out = f"[exit {exit_code}]\n{out}"
        return out, is_error
    if item_type == "fileChange":
        status = item.get("status") or "unknown"
        n = len(item.get("changes") or [])
        return (
            f"apply_patch status={status}, {n} change(s)",
            status not in {"completed", "applied", "success"},
        )
    if item_type == "mcpToolCall":
        error = item.get("error")
        if error:
            return (
                f"[error] {json.dumps(error, ensure_ascii=False)}",
                True,
            )
        result = item.get("result")
        return (
            json.dumps(result, ensure_ascii=False)
            if result is not None else "",
            False,
        )
    if item_type == "dynamicToolCall":
        content_items = item.get("contentItems") or []
        if isinstance(content_items, list) and content_items:
            return (
                json.dumps(content_items, ensure_ascii=False),
                not bool(item.get("success", True)),
            )
        success = item.get("success", True)
        return f"success={success}", not bool(success)
    if item_type in {"imageGeneration", "imageView"}:
        failed = bool(item.get("failure")) or item.get("status") in {
            "failed", "cancelled", "error",
        }
        if failed:
            return "Image tool failed", True
        path = item.get("savedPath") or item.get("path")
        return str(path or "Image tool completed without a saved file"), False
    return "", False


_FILE_CHANGE_SNAPSHOT_MAX_BYTES = 4 * 1024 * 1024
_FILE_CHANGE_EXACT_DIFF_CELLS = 250_000


def _codex_file_change_path(agent: Any, value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = Path(
            getattr(agent, "session_cwd", None) or Path.cwd()
        ) / candidate
    try:
        return candidate.resolve(strict=False)
    except (OSError, RuntimeError):
        return None


def _codex_file_lines(path: Path) -> list[bytes] | None:
    try:
        if (
            not path.is_file()
            or path.stat().st_size > _FILE_CHANGE_SNAPSHOT_MAX_BYTES
        ):
            return [] if not path.exists() else None
        content = path.read_bytes()
    except OSError:
        return None
    if b"\0" in content:
        return None
    return content.splitlines()


def _codex_file_change_snapshot(
    agent: Any,
    item: dict,
) -> dict[str, list[bytes] | None]:
    snapshot: dict[str, list[bytes] | None] = {}
    for change in item.get("changes") or []:
        if not isinstance(change, dict):
            continue
        path = _codex_file_change_path(agent, change.get("path"))
        if path is not None:
            snapshot[str(path)] = _codex_file_lines(path)
    return snapshot


def _line_change_counts(before: list[bytes], after: list[bytes]) -> tuple[int, int]:
    if before == after:
        return 0, 0
    if len(before) * len(after) <= _FILE_CHANGE_EXACT_DIFF_CELLS:
        matcher = SequenceMatcher(a=before, b=after, autojunk=False)
        added = 0
        removed = 0
        for tag, left_start, left_end, right_start, right_end in matcher.get_opcodes():
            if tag in {"replace", "delete"}:
                removed += left_end - left_start
            if tag in {"replace", "insert"}:
                added += right_end - right_start
        return added, removed

    prefix = 0
    while (
        prefix < len(before)
        and prefix < len(after)
        and before[prefix] == after[prefix]
    ):
        prefix += 1
    suffix = 0
    while (
        suffix < len(before) - prefix
        and suffix < len(after) - prefix
        and before[-1 - suffix] == after[-1 - suffix]
    ):
        suffix += 1
    return len(after) - prefix - suffix, len(before) - prefix - suffix


def _codex_file_change_line_counts(
    agent: Any,
    item: dict,
    before: dict[str, list[bytes] | None],
) -> dict[str, Any]:
    """Exact line counts, plus the unified diff Hermes Chat renders red/green.

    Codex's own diff (``changes[].diff``) wins, with counts taken from it; the
    before/after snapshot diff is the fallback when Codex did not send one.
    """
    if item.get("type") != "fileChange":
        return {}
    if item.get("status") not in {"completed", "applied", "success"}:
        return {}
    try:
        from agent.tool_diff import codex_native_diff

        native = codex_native_diff(
            item.get("changes"), cwd=getattr(agent, "session_cwd", None)
        )
    except Exception:
        logger.debug("Codex native file-change diff failed", exc_info=True)
        native = None
    if native:
        return native
    added = 0
    removed = 0
    measured = False
    snapshots: list[tuple[str, list[bytes] | None, list[bytes] | None]] = []
    for change in item.get("changes") or []:
        if not isinstance(change, dict):
            continue
        path = _codex_file_change_path(agent, change.get("path"))
        if path is None:
            continue
        before_lines = before.get(str(path))
        after_lines = _codex_file_lines(path)
        if before_lines is None or after_lines is None:
            continue
        file_added, file_removed = _line_change_counts(before_lines, after_lines)
        added += file_added
        removed += file_removed
        measured = True
        from agent.tool_diff import display_path

        snapshots.append(
            (display_path(str(path), getattr(agent, "session_cwd", None)), before_lines, after_lines)
        )
    if not measured:
        return {}
    counts: dict[str, Any] = {"lines_added": added, "lines_removed": removed}
    try:
        from agent.tool_diff import file_change_diff

        rendered = file_change_diff(snapshots)
    except Exception:
        logger.debug("Codex file-change diff failed", exc_info=True)
        rendered = None
    if rendered:
        counts["diff"] = rendered["diff"]
    return counts


# Codex has no ultracode keyword; its multi-agent tools (spawn_agent,
# wait_agent, on by default since Codex 0.14x) are used when the model is told
# to. This rides each Ultracode turn like the preflight note: never durable.
CODEX_ULTRACODE_NOTE = (
    "[Ultracode is on for this turn: optimize for the most exhaustive, correct "
    "answer, not the fastest or cheapest. For substantive work, split independent "
    "parts across sub-agents with spawn_agent, collect them with wait_agent, and "
    "adversarially verify their findings before you report them. Work solo only "
    "on conversational turns or trivial edits. Token cost is not a constraint.]"
)


def _codex_ultracode(agent: Any) -> bool:
    config = getattr(agent, "reasoning_config", None)
    return bool(isinstance(config, dict) and config.get("ultracode"))


def _hermes_tool_name(raw: str) -> str:
    """Map a projected codex call name onto the Hermes tool it stands for."""
    if raw == "exec_command":
        return "terminal"
    if raw == "apply_patch":
        return "patch"
    if raw.startswith("mcp."):
        return raw.split(".", 2)[-1]
    return raw


def _codex_projected_tool_calls(projected_messages: list) -> list:
    """(tool_name, args, result, call_id) for each projected call/result pair."""
    results: Dict[str, Any] = {}
    for message in projected_messages or []:
        if isinstance(message, dict) and message.get("role") == "tool":
            results[str(message.get("tool_call_id") or "")] = message.get("content")
    calls = []
    for message in projected_messages or []:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function") or {}
            raw_name = str(function.get("name") or "")
            try:
                args = json.loads(function.get("arguments") or "{}")
            except Exception:
                args = {}
            if not isinstance(args, dict):
                args = {"arguments": args}
            call_id = str(call.get("id") or "")
            calls.append((_hermes_tool_name(raw_name), args, results.get(call_id, ""), call_id))
    return calls


def _codex_hook_parity(
    agent: Any,
    turn: Any,
    messages: List[Dict[str, Any]],
    original_user_message: Any,
    effective_task_id: str,
) -> int:
    """Give a codex turn the observer hooks and the verify gate of the default loop.

    codex executes tools inside its own process, so ``post_tool_call``,
    ``post_llm_call`` and ``pre_verify`` never fire on this path unless they
    are emitted from the projected rows. The verify gate may keep the turn
    going: a ``continue`` directive is sent to the same codex thread as one
    more user turn, its rows are spliced and persisted like the first, and the
    turn object is updated to the follow-up's outcome. Bounded by
    ``agent.max_verify_nudges``. Returns the number of follow-up turns run.
    """
    from hermes_cli.lifecycle import has_hook, invoke_hook

    session_id = getattr(agent, "session_id", "") or ""
    turn_id = getattr(agent, "_current_turn_id", "") or ""
    platform = getattr(agent, "platform", "") or ""
    model = getattr(agent, "model", "") or ""
    changed: set = set()

    def emit_tool_hooks(projected: list) -> None:
        for name, args, result, call_id in _codex_projected_tool_calls(projected):
            if name == "patch":
                for change in args.get("changes") or []:
                    path = _codex_file_change_path(
                        agent, change.get("path") if isinstance(change, dict) else None
                    )
                    if path is not None:
                        changed.add(str(path))
            if not has_hook("post_tool_call"):
                continue
            try:
                invoke_hook(
                    "post_tool_call",
                    tool_name=name,
                    args=args,
                    result=result,
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
                    # Replayed after the turn: no steer can reach the model.
                    replay=True,
                )
            except Exception:
                logger.debug("codex post_tool_call parity failed", exc_info=True)

    emit_tool_hooks(list(turn.projected_messages or []))

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
        and isinstance(turn.final_text, str)
        and turn.final_text.strip()
        and not turn.interrupted
        and turn.error is None
        and getattr(agent, "_codex_session", None) is not None
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

        # The nudge is synthetic and stripped from the durable transcript,
        # exactly as the default loop does; the codex thread did consume it.
        append_message(
            messages,
            {"role": "user", "content": nudge, "_pre_verify_synthetic": True},
        )
        try:
            follow = agent._codex_session.run_turn(user_input=nudge)
        except Exception:
            logger.warning("codex pre_verify follow-up turn failed", exc_info=True)
            break
        follow_ups += 1
        for message in follow.projected_messages or []:
            append_message(messages, message)
        if getattr(agent, "_session_db", None) is not None:
            try:
                agent._flush_messages_to_session_db(messages)
            except Exception:
                logger.debug("codex pre_verify follow-up flush failed", exc_info=True)
        emit_tool_hooks(list(follow.projected_messages or []))
        turn.projected_messages = list(turn.projected_messages or []) + list(
            follow.projected_messages or []
        )
        turn.tool_iterations = int(getattr(turn, "tool_iterations", 0) or 0) + int(
            getattr(follow, "tool_iterations", 0) or 0
        )
        turn.final_text = follow.final_text
        turn.interrupted = follow.interrupted
        turn.error = follow.error
        turn.error_code = getattr(follow, "error_code", None)
        if getattr(follow, "should_retire", False):
            turn.should_retire = True
        logger.debug("codex pre_verify nudge issued (attempt %d)", attempt)

    if (
        isinstance(turn.final_text, str)
        and turn.final_text.strip()
        and not turn.interrupted
        and turn.error is None
        and has_hook("post_llm_call")
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
            logger.debug("codex post_llm_call parity failed", exc_info=True)
    return follow_ups


def make_codex_app_server_event_bridge(agent) -> Callable[[dict], None]:
    """Build an ``on_event`` callback that wires codex app-server JSON-RPC
    notifications into Hermes' gateway UI callbacks.

    Returns a single-argument callable suitable for
    ``CodexAppServerSession(on_event=...)``.

    Translation map:
      * ``item/started`` for tool-shaped items → ``tool_progress_callback(
        "tool.started", name, preview, args)``
      * ``item/completed`` for tool-shaped items → ``tool_progress_callback(
        "tool.completed", name, None, None, duration=..., is_error=...,
        result=...)``
      * ``item/commandExecution/outputDelta`` → ``tool_progress_callback(
        "tool.output.delta", "exec_command", ..., chunk=...)``
      * ``item/agentMessage/delta`` for ``phase=final_answer`` →
        ``_fire_stream_delta(text)`` so chat adapters render only the
        authoritative answer in the answer slot. Commentary deltas are held
        out of that channel and surface through the completed-item path.
      * ``item/reasoning/delta`` → ``_fire_reasoning_delta(text)``
      * ``item/completed`` for ``phase=commentary`` ``agentMessage`` →
        ``_emit_interim_assistant_message({"role": "assistant",
        "content": text})``.

    All callback invocations are guarded — a buggy display callback must
    not tear down the codex turn loop. Errors are logged at DEBUG so the
    notification stream keeps flowing regardless.
    """
    # item_id -> (tool_name, args, started_wall_time). Populated on
    # item/started and consumed on item/completed so duration is correct
    # even when codex doesn't report durationMs.
    started: dict[str, tuple[str, dict, float]] = {}
    file_change_snapshots: dict[str, dict[str, list[bytes] | None]] = {}
    agent_message_phases: dict[str, str] = {}
    buffered_agent_deltas: dict[str, list[str]] = {}
    active_agent_paths: dict[str, str] = {}

    def _stable_call_id(item: dict, name: str) -> str:
        """Deterministic tool_call id mirroring CodexEventProjector, so a
        live TUI tool card correlates with the same tool call after the
        session is resumed and history is projected."""
        from agent.transports.codex_event_projector import _deterministic_call_id

        item_id = item.get("id") or ""
        item_type = item.get("type") or ""
        if item_type == "commandExecution":
            return _deterministic_call_id("exec", item_id)
        if item_type == "fileChange":
            return _deterministic_call_id("apply_patch", item_id)
        if item_type == "mcpToolCall":
            server = item.get("server") or "mcp"
            tool = item.get("tool") or "unknown"
            return _deterministic_call_id(f"mcp__{server}__{tool}", item_id)
        if item_type == "dynamicToolCall":
            tool = item.get("tool") or "unknown"
            return _deterministic_call_id(f"dyn_{tool}", item_id)
        return _deterministic_call_id(name, item_id)

    def _fire_tool_started(item: dict) -> None:
        item_id = item.get("id") or ""
        name = _codex_item_to_tool_name(item)
        args = _codex_item_to_args(item)
        if item_id:
            started[item_id] = (name, args, time.monotonic())
            if item.get("type") == "fileChange":
                file_change_snapshots[item_id] = _codex_file_change_snapshot(
                    agent, item
                )
        cb = getattr(agent, "tool_progress_callback", None)
        if cb is not None:
            try:
                preview = _codex_item_to_preview(item)
                if name == "wait_agent" and active_agent_paths:
                    preview += "\nActive agents: " + ", ".join(active_agent_paths.values())
                cb("tool.started", name, preview, args,
                   tool_call_id=_stable_call_id(item, name))
            except Exception:
                logger.debug(
                    "tool_progress_callback raised on tool.started for %s",
                    name, exc_info=True,
                )
        # Authoritative stable-ID tool card (TUI / desktop). Fires
        # alongside tool_progress so surfaces that render structured tool
        # cards (not just progress bubbles) stay correlated with the
        # projected history entry after a resume.
        start_cb = getattr(agent, "tool_start_callback", None)
        if start_cb is not None:
            try:
                start_cb(_stable_call_id(item, name), name, args)
            except Exception:
                logger.debug(
                    "tool_start_callback raised for %s", name, exc_info=True,
                )

    def _fire_tool_completed(item: dict) -> None:
        item_id = item.get("id") or ""
        name = _codex_item_to_tool_name(item)
        # Lifecycle notifications may arrive only as completed items. Retain
        # their path/kind label instead of emitting an anonymous result row.
        if item.get("type") in COORDINATION_ITEM_TYPES and item_id not in started:
            _fire_tool_started(item)
        prior = started.pop(item_id, None)
        # Prefer codex's own durationMs when present so the bubble shows
        # exact tool wall-time; fall back to our started timestamp; fall
        # back to None if we never saw an item/started (some codex
        # versions only emit completed for fast items).
        duration: Any = None
        codex_ms = item.get("durationMs")
        if isinstance(codex_ms, (int, float)) and codex_ms >= 0:
            duration = codex_ms / 1000.0
        elif prior is not None:
            duration = time.monotonic() - prior[2]
        result, is_error = _codex_item_completion_payload(item)
        line_counts = _codex_file_change_line_counts(
            agent,
            item,
            file_change_snapshots.pop(item_id, {}),
        )
        cb = getattr(agent, "tool_progress_callback", None)
        if cb is not None:
            try:
                cb("tool.completed", name, None, None,
                   duration=duration, is_error=is_error, result=result,
                   tool_call_id=_stable_call_id(item, name), **line_counts)
            except Exception:
                logger.debug(
                    "tool_progress_callback raised on tool.completed for %s",
                    name, exc_info=True,
                )
        complete_cb = getattr(agent, "tool_complete_callback", None)
        if complete_cb is not None:
            args = prior[1] if prior is not None else _codex_item_to_args(item)
            try:
                complete_cb(_stable_call_id(item, name), name, args, result)
            except Exception:
                logger.debug(
                    "tool_complete_callback raised for %s", name, exc_info=True,
                )

    def _fire_text_delta(params: dict) -> None:
        text = params.get("delta") or params.get("text") or ""
        if not isinstance(text, str) or not text:
            return
        item_id = params.get("itemId") or params.get("item_id") or ""
        phase = agent_message_phases.get(item_id)
        if phase == "commentary":
            return
        if phase != "final_answer":
            # Current app-server versions provide phase on item/started, but
            # delta notifications themselves do not. Buffer an out-of-order or
            # legacy delta until item/completed tells us whether it is progress
            # or the final answer. This favors delayed text over misclassified
            # text that is later erased.
            buffered_agent_deltas.setdefault(item_id, []).append(text)
            return
        fn = getattr(agent, "_fire_stream_delta", None)
        if fn is None:
            return
        try:
            fn(text)
        except Exception:
            logger.debug("_fire_stream_delta raised", exc_info=True)

    def _fire_reasoning_delta(params: dict) -> None:
        text = params.get("delta") or params.get("text") or ""
        if not isinstance(text, str) or not text:
            return
        fn = getattr(agent, "_fire_reasoning_delta", None)
        if fn is None:
            return
        try:
            fn(text)
        except Exception:
            logger.debug("_fire_reasoning_delta raised", exc_info=True)

    def _fire_command_output_delta(params: dict) -> None:
        delta = params.get("delta") or ""
        item_id = params.get("itemId") or params.get("item_id") or ""
        if not isinstance(delta, str) or not delta or not item_id:
            return
        name = "exec_command"
        cb = getattr(agent, "tool_progress_callback", None)
        if cb is None:
            return
        item = {"type": "commandExecution", "id": item_id}
        try:
            cb(
                "tool.output.delta",
                name,
                None,
                None,
                chunk=delta,
                channel="combined",
                tool_call_id=_stable_call_id(item, name),
            )
        except Exception:
            logger.debug(
                "tool_progress_callback raised on command output delta",
                exc_info=True,
            )

    def _fire_agent_message_completed(item: dict) -> None:
        text = item.get("text") or ""
        if not isinstance(text, str) or not text.strip():
            return
        if item.get("phase") != "commentary":
            return
        # display.show_commentary=false — mid-turn narration stays off the
        # visible interim path on this runtime too (same contract as the
        # codex_responses commentary channel).
        if not getattr(agent, "show_commentary", True):
            return
        emit = getattr(agent, "_emit_interim_assistant_message", None)
        if emit is None:
            return
        try:
            emit({"role": "assistant", "content": text})
        except Exception:
            logger.debug(
                "_emit_interim_assistant_message raised", exc_info=True,
            )

    def on_event(note: dict) -> None:
        if not isinstance(note, dict):
            return
        method = note.get("method") or ""
        params = note.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        if method == "item/agentMessage/delta":
            _fire_text_delta(params)
            return
        if method == "item/commandExecution/outputDelta":
            _fire_command_output_delta(params)
            return
        if method in {"item/reasoning/delta", "item/reasoning/summaryDelta"}:
            _fire_reasoning_delta(params)
            return
        item = params.get("item")
        if not isinstance(item, dict):
            return
        item_type = item.get("type") or ""
        if item_type == "subAgentActivity" and method in {"item/started", "item/completed"}:
            agent_id = item.get("agentThreadId") or ""
            if item.get("kind") == "started" and agent_id:
                active_agent_paths[agent_id] = item.get("agentPath") or agent_id
            elif item.get("kind") in {"completed", "interrupted"}:
                active_agent_paths.pop(agent_id, None)
        if method == "item/started":
            if item_type in _CODEX_TOOL_ITEM_TYPES:
                _fire_tool_started(item)
                return
            if item_type == "agentMessage":
                item_id = item.get("id") or ""
                phase = item.get("phase")
                if item_id and phase in {"commentary", "final_answer"}:
                    agent_message_phases[item_id] = phase
                return
        if method == "item/completed":
            if item_type in _CODEX_TOOL_ITEM_TYPES:
                _fire_tool_completed(item)
            elif item_type == "agentMessage":
                item_id = item.get("id") or ""
                phase = item.get("phase")
                if item_id and phase in {"commentary", "final_answer"}:
                    agent_message_phases[item_id] = phase
                buffered = buffered_agent_deltas.pop(item_id, [])
                if phase != "commentary":
                    # phase-less legacy messages remain compatible, but their
                    # deltas wait until completion instead of leaking a possible
                    # progress note into the final-answer slot.
                    agent_message_phases[item_id] = "final_answer"
                    for delta in buffered:
                        _fire_text_delta({
                            "delta": delta,
                            "itemId": item_id,
                        })
                _fire_agent_message_completed(item)
                agent_message_phases.pop(item_id, None)

    return on_event


def _agent_opus_worker_enabled(agent: Any) -> bool:
    """Whether this parent runtime may still reach the governed Opus worker.

    Codex spawns the hermes-tools MCP server as a separate process that builds
    its own tool list, so a toolset denial on this agent is invisible there.
    Resolve the parent's effective state here and let it travel with the
    session (see agent/opus_delegation.py). Fails closed on any resolution
    error — an unknown state must not be read as permission.
    """
    try:
        from toolsets import resolve_toolset, validate_toolset

        disabled = set(getattr(agent, "disabled_toolsets", None) or [])
        if "opus_worker" in disabled:
            return False
        enabled = getattr(agent, "enabled_toolsets", None)
        if not enabled:
            # No allowlist means every registered toolset is available.
            return True
        for name in enabled:
            if name == "opus_worker":
                return True
            if validate_toolset(name) and "opus_code_worker" in resolve_toolset(name):
                return True
        return False
    except Exception:
        logger.debug("opus_worker parent-state lookup failed", exc_info=True)
        return False


def run_codex_app_server_turn(
    agent,
    *,
    user_message: str,
    original_user_message: Any,
    messages: List[Dict[str, Any]],
    effective_task_id: str,
    should_review_memory: bool = False,
    plugin_user_context: str = "",
) -> Dict[str, Any]:
    """Codex app-server runtime path. Hands the entire turn to a `codex
    app-server` subprocess and projects its events back into Hermes'
    messages list so memory/skill review keep working.

    Called from run_conversation() when agent.api_mode == "codex_app_server".
    Returns the same dict shape as the chat_completions path.
    """
    from agent.transports.codex_app_server_session import (
        DEFAULT_FIRST_EVENT_TIMEOUT,
        CodexAppServerSession,
        _ServerRequestRouting,
    )
    from agent.deadline import resolve_timeout

    first_event_timeout = resolve_timeout(
        "codex_app_server.first_event",
        default=DEFAULT_FIRST_EVENT_TIMEOUT,
    )

    def _watchdog_timeout(payload: dict[str, Any]) -> None:
        timeout = float(payload.get("timeout_seconds") or 0.0)
        message = (
            "Codex accepted the turn but emitted no turn-scoped activity "
            f"within {timeout:g} seconds—stopping that turn and resetting "
            "the runtime. The prompt was not replayed."
        )
        progress = getattr(agent, "tool_progress_callback", None)
        if progress is not None:
            try:
                progress(
                    "runtime.first_event_timeout",
                    "codex-app-server",
                    message,
                    None,
                    **payload,
                )
            except Exception:
                logger.debug(
                    "Codex app-server watchdog progress callback failed",
                    exc_info=True,
                )
        emit_status = getattr(agent, "_emit_status", None)
        if emit_status is not None:
            try:
                emit_status(message)
            except Exception:
                logger.debug(
                    "Codex app-server watchdog status callback failed",
                    exc_info=True,
                )

    # Continuity gate. A native Codex thread stays resumable indefinitely, so a
    # bound thread id proves only that the thread still exists — never that it
    # holds this conversation's current transcript. Any turn answered by another
    # runtime (a deliberate model switch, or the automatic one after a credit
    # limit) advances the Hermes record while the native thread stands still.
    # Settle that question BEFORE deciding what to reuse or resume.
    from agent.runtime_cwd import resolve_agent_cwd

    codex_cwd = getattr(agent, "session_cwd", None) or str(resolve_agent_cwd())
    prior_entries = _codex_dialogue_entries(messages[:-1])
    codex_thread_state = _load_codex_thread_state(agent)
    resume_thread_id, pending_entries, continuity_reason = _codex_resume_plan(
        thread_id=(
            _stored_codex_app_server_thread_id(agent)
            or str(codex_thread_state.get("thread_id") or "").strip()
        ),
        state=codex_thread_state,
        prior_entries=prior_entries,
        cwd=codex_cwd,
    )
    if pending_entries and resume_thread_id:
        logger.info(
            "codex thread %s is behind this conversation by %d messages; "
            "delivering them as a catch-up before the current turn",
            resume_thread_id[:8],
            len(pending_entries),
        )
    resident_codex_session = getattr(agent, "_codex_session", None)
    if resident_codex_session is not None and (
        not resume_thread_id
        or str(getattr(resident_codex_session, "_thread_id", "") or "")
        != resume_thread_id
    ):
        # A warm process is not continuity either. The same AIAgent object
        # survives a model switch, so a resident thread goes stale exactly like
        # a persisted one — and reusing it is the more dangerous case, because
        # it never even reaches the resume path. Retire it and rebuild below.
        logger.warning(
            "retiring resident Codex session: bound thread cannot continue "
            "this transcript (%s)",
            continuity_reason,
        )
        try:
            resident_codex_session.close()
        except Exception:
            logger.debug("codex resident-session cleanup failed", exc_info=True)
        agent._codex_session = None

    # Lazy session: one CodexAppServerSession per AIAgent instance.
    # Spawned on first turn, reused across turns, closed at AIAgent
    # shutdown (see _cleanup hook).
    created_codex_session = False
    if not hasattr(agent, "_codex_session") or agent._codex_session is None:
        created_codex_session = True

        cwd = codex_cwd
        # Approval callback: defer to Hermes' standard prompt flow if a CLI
        # thread has installed one. Gateway/API turns use their existing
        # per-run approval queue; cron or detached contexts without an
        # attached notifier return an explicit ``unavailable`` outcome. That
        # distinction matters because Codex must cancel an unpresented request,
        # not report it as rejected by the user.
        try:
            from tools.terminal_tool import _get_approval_callback
            approval_callback = _get_approval_callback()
        except Exception:
            approval_callback = None
        if approval_callback is None:
            from tools.approval import request_codex_approval

            approval_callback = request_codex_approval

        from hermes_cli.config import load_config

        runtime_cfg = load_config()
        codex_runtime_cfg = (
            runtime_cfg.get("codex_runtime", {})
            if isinstance(runtime_cfg, dict)
            else {}
        )
        no_prompt_cfg = (
            codex_runtime_cfg.get("no_prompt", {})
            if isinstance(codex_runtime_cfg, dict)
            else {}
        )
        bounded_no_prompt = bool(
            getattr(agent, "platform", "") == "api_server"
            and isinstance(no_prompt_cfg, dict)
            and no_prompt_cfg.get("enabled") is True
        )
        from tools.reversible_deletion import ReversibleDeletionPolicy

        reversible_deletion_policy = ReversibleDeletionPolicy.from_config(
            codex_runtime_cfg.get("reversible_deletion", {})
            if isinstance(codex_runtime_cfg, dict)
            else {}
        )
        from tools.workspace_snapshots import WorkspaceSnapshotPolicy

        workspace_snapshot_policy = WorkspaceSnapshotPolicy.from_config(
            codex_runtime_cfg.get("workspace_snapshots", {})
            if isinstance(codex_runtime_cfg, dict)
            else {}
        )

        # When the user has
        # explicitly opted out of Hermes approvals — via `approvals.mode: off`
        # in config, the /yolo session toggle, or --yolo / HERMES_YOLO_MODE —
        # honor that and let codex's own sandbox permission profile
        # (~/.codex/config.toml) be the policy gate instead of double-gating
        # with Hermes. Defaults (manual/smart/unset) stay fail-closed and use
        # the callback above for a genuine round trip when a surface exists.
        auto_approve_requests = bounded_no_prompt
        try:
            from tools.approval import is_approval_bypass_active

            auto_approve_requests = (
                auto_approve_requests or is_approval_bypass_active()
            )
        except Exception:
            logger.debug(
                "codex app-server: approval-bypass lookup failed; "
                "keeping fail-closed default",
                exc_info=True,
            )

        # Bridge codex JSON-RPC notifications (item/started, item/completed,
        # item/agentMessage/delta, ...) into Hermes' gateway UI callbacks
        # (tool_progress_callback, _fire_stream_delta,
        # _emit_interim_assistant_message). Without this, Discord/Telegram
        # users see no live tool-progress or interim commentary while
        # codex_app_server is running — only the final answer (#33200).
        # Supersedes the narrower item/started-only bridge from #38835.
        from agent.reasoning_effort import requested_effort
        model_cfg = runtime_cfg.get("model", {}) if isinstance(runtime_cfg, dict) else {}
        require_exact = bool(
            model_cfg.get("openai_runtime_require_exact", False)
            if isinstance(model_cfg, dict)
            else False
        )
        read_only = bool(getattr(agent, "read_only", False))
        agent._codex_session = CodexAppServerSession(
            cwd=cwd,
            hermes_session_id=str(getattr(agent, "session_id", "") or ""),
            resume_thread_id=resume_thread_id,
            model=getattr(agent, "model", ""),
            effort=requested_effort(getattr(agent, "reasoning_config", None)),
            ultracode=_codex_ultracode(agent),
            parent_provider=str(getattr(agent, "provider", "") or ""),
            opus_worker_enabled=_agent_opus_worker_enabled(agent),
            require_exact=require_exact,
            read_only=read_only,
            project_key=str(getattr(agent, "session_project", "") or ""),
            reversible_deletion_policy=reversible_deletion_policy,
            workspace_snapshot_policy=workspace_snapshot_policy,
            approval_callback=approval_callback,
            request_routing=_ServerRequestRouting(
                auto_approve_exec=auto_approve_requests,
                auto_approve_apply_patch=auto_approve_requests,
                guard_no_prompt_exec=bounded_no_prompt,
                guard_no_prompt_file_changes=bounded_no_prompt,
            ),
            on_event=make_codex_app_server_event_bridge(agent),
            on_watchdog_timeout=_watchdog_timeout,
            first_event_timeout=first_event_timeout,
        )
    else:
        # These callbacks and deadlines belong to the outer run even when the
        # native app-server process is reused across multiple Hermes turns.
        agent._codex_session._on_event = make_codex_app_server_event_bridge(agent)
        agent._codex_session.on_watchdog_timeout = _watchdog_timeout
        agent._codex_session.first_event_timeout = first_event_timeout
        # Effort rides each turn/start, so a resident session adopts this
        # turn's level (including an Ultracode toggle) on the same thread.
        from agent.reasoning_effort import requested_effort

        set_turn_effort = getattr(agent._codex_session, "set_turn_effort", None)
        if callable(set_turn_effort):
            set_turn_effort(
                requested_effort(getattr(agent, "reasoning_config", None)),
                ultracode=_codex_ultracode(agent),
            )

    # NOTE: the user message is ALREADY appended to messages by the
    # standard run_conversation() flow (line ~11823) before the early
    # return reaches us. Do NOT append again — that would duplicate.

    try:
        # Command parsing cannot see an unlink hidden inside an arbitrary
        # executable. A configured workspace snapshot is therefore the broad
        # recovery boundary: it must complete before this turn reaches Codex.
        # This runs for every turn, including turns that reuse a warm app-server
        # session, so files created by an earlier turn are protected too.
        snapshot = None
        if not bool(getattr(agent, "read_only", False)):
            snapshot = agent._codex_session.capture_workspace_snapshot(
                str(effective_task_id or "")
            )
        if snapshot is not None:
            agent._last_workspace_snapshot = snapshot.as_dict()

        thread_id = agent._codex_session.ensure_started()
        if created_codex_session and not getattr(
            agent._codex_session, "_resumed_existing_thread", False
        ):
            # Either nothing could be resumed, or the resume attempt failed and
            # the session fell back to a fresh thread. Both need the whole
            # transcript — the delta was computed for a thread we no longer have.
            handoff_entries = prior_entries
            build_turn_input = _codex_history_handoff
        else:
            handoff_entries = pending_entries
            build_turn_input = _codex_catch_up_handoff
        turn_input = (
            build_turn_input(handoff_entries, user_message)
            if handoff_entries
            else user_message
        )
        if plugin_user_context:
            # The pre_llm_call hook's note (the preflight judge, gateway
            # notices) rides the turn input here as it rides the API copy
            # of the user message on the default loop. It is not part of
            # the durable dialogue entries, so it never replays.
            turn_input = f"{turn_input}\n\n{plugin_user_context}"
        if _codex_ultracode(agent):
            turn_input = (
                [*turn_input, {"type": "text", "text": CODEX_ULTRACODE_NOTE}]
                if isinstance(turn_input, list)
                else f"{turn_input}\n\n{CODEX_ULTRACODE_NOTE}"
            )
        # Record what this thread is about to consume before the turn runs. A
        # turn that dies mid-flight still leaves a thread holding this input,
        # and re-seeding it from scratch next time would duplicate everything.
        # The current user message is deliberately excluded: if delivery failed,
        # under-counting replays one message, while over-counting drops it.
        _persist_codex_thread_state(
            agent, thread_id=thread_id, cwd=codex_cwd, entries=prior_entries
        )
        turn = agent._codex_session.run_turn(user_input=turn_input)
    except Exception as exc:
        logger.exception("codex app-server turn failed")
        # Crash → unconditionally drop the session so the next turn
        # respawns from scratch instead of reusing a dead client.
        try:
            agent._codex_session.close()
        except Exception:
            pass
        agent._codex_session = None
        _user_interrupted = bool(
            getattr(agent, "_interrupt_requested", False)
        )
        _interrupt_message = (
            getattr(agent, "_interrupt_message", None)
            if _user_interrupted
            else None
        )
        if _user_interrupted:
            agent.clear_interrupt()
        return {
            "final_response": (
                f"Codex app-server turn failed: {exc}. "
                f"Fall back to default runtime with `/codex-runtime auto`."
            ),
            "messages": messages,
            "api_calls": 0,
            "completed": False,
            "partial": True,
            "interrupted": _user_interrupted,
            **(
                {"interrupt_message": _interrupt_message}
                if _interrupt_message
                else {}
            ),
            "error": str(exc),
        }

    # This runtime bypasses the normal conversation-loop finalizer. Mirror its
    # interrupt handoff/cleanup so a hard stop cannot poison the next turn and a
    # message-bearing compatibility interrupt can still be replayed by callers.
    _user_interrupted = bool(
        turn.interrupted and getattr(agent, "_interrupt_requested", False)
    )
    _interrupt_message = (
        getattr(agent, "_interrupt_message", None) if _user_interrupted else None
    )
    if _user_interrupted:
        agent.clear_interrupt()

    # If the turn signalled the underlying client is wedged (deadline
    # blown, a turn deadline fired, OAuth refresh died, subprocess
    # exited), retire the session so the next turn respawns codex
    # rather than riding the broken process. Mirrors openclaw beta.8's
    # "retire timed-out app-server clients" fix.
    if getattr(turn, "should_retire", False):
        logger.warning(
            "codex app-server session retired (turn error: %s)",
            turn.error,
        )
        try:
            agent._codex_session.close()
        except Exception:
            pass
        agent._codex_session = None

    # Splice projected messages into the conversation. The projector emits
    # standard {role, content, tool_calls, tool_call_id} entries, which
    # is exactly what curator.py / sessions DB expect.
    # The transcript prefix this thread has consumed, used below to record
    # continuity. It narrows to the pre-projection boundary if the projected
    # rows cannot be persisted, since a later turn will reload a transcript
    # that does not contain them.
    codex_consumed_messages = messages
    if turn.projected_messages:
        from agent.message_metadata import append_message

        pre_projection_len = len(messages)
        for projected_message in turn.projected_messages:
            append_message(messages, projected_message)

        # Persist the newly-projected assistant/tool messages ourselves.
        # This path is an early return that bypasses conversation_loop, whose
        # normal per-step _persist_session() calls would otherwise flush them.
        # The inbound user turn was already flushed at turn start
        # (turn_context.py _persist_session), and _flush_messages_to_session_db
        # is idempotent via the intrinsic _DB_PERSISTED_MARKER — so this writes
        # ONLY the new codex projected rows and does NOT re-write the user turn.
        # Keeping the agent as the sole persister lets us return
        # agent_persisted=True below, so the gateway skips its own DB write and
        # we avoid the #860/#42039 duplicate user-message write (append_message
        # is a raw INSERT with no dedup, so a gateway re-write would duplicate
        # the already-flushed user turn). See gateway/run.py agent_persisted.
        if getattr(agent, "_session_db", None) is not None:
            try:
                _codex_flush_ok = agent._flush_messages_to_session_db(messages)
            except Exception:
                _codex_flush_ok = False
                logger.warning(
                    "codex app-server projected-message flush failed",
                    exc_info=True,
                )
            if _codex_flush_ok is False:
                # Unlike the chat-completions loop (which fails closed BEFORE
                # projection — see conversation_loop session_persistence_failed),
                # codex output has already streamed to the user by the time this
                # flush runs, so there is nothing left to withhold. We cannot
                # flip agent_persisted=False either: the gateway fallback write
                # would re-INSERT the already-flushed user turn (#860/#42039).
                # Surface the durability gap loudly instead of a silent debug.
                logger.warning(
                    "codex app-server turn was delivered but could NOT be "
                    "persisted to the session DB (session=%s) — this turn "
                    "will be missing after restart/resume",
                    getattr(agent, "session_id", None),
                )
                codex_consumed_messages = messages[:pre_projection_len]

    # Hook parity with the default loop (observer hooks + the pre_verify gate),
    # which may extend this turn with bounded follow-up turns on the thread.
    _codex_follow_ups = 0
    try:
        _codex_follow_ups = _codex_hook_parity(
            agent, turn, messages, original_user_message, effective_task_id
        )
    except Exception:
        logger.debug("codex hook parity failed", exc_info=True)
    # This thread has now consumed this turn. Record the boundary so a later
    # runtime switch is measured against the right prefix instead of being
    # waved through by the mere existence of a thread id.
    _persist_codex_thread_state(
        agent,
        thread_id=thread_id,
        cwd=codex_cwd,
        entries=_codex_dialogue_entries(codex_consumed_messages),
    )

    # Counter ticks for the agent-improvement loop.
    # _turns_since_memory and _user_turn_count are ALREADY incremented
    # in the run_conversation() pre-loop block (lines ~11793-11817) so we
    # do NOT touch them here — that would double-count.
    # Only _iters_since_skill needs explicit increment, since the
    # chat_completions loop bumps it per tool iteration (line ~12110)
    # and that loop is bypassed on this path.
    agent._iters_since_skill = (
        getattr(agent, "_iters_since_skill", 0) + turn.tool_iterations
    )
    _record_codex_app_server_compaction(agent, turn)
    usage_result = _record_codex_app_server_usage(agent, turn)
    api_calls = 1 + _codex_follow_ups

    # Now check the skill nudge AFTER iters were incremented — same
    # pattern the chat_completions path uses (line ~15432).
    should_review_skills = False
    if (
        agent._skill_nudge_interval > 0
        and agent._iters_since_skill >= agent._skill_nudge_interval
        and "skill_manage" in agent.valid_tool_names
    ):
        should_review_skills = True
        agent._iters_since_skill = 0

    # A clean turn/completed notification without assistant text is not a
    # successful chat turn. This occurs when Codex cancels an unanswered
    # approval request: the protocol closes normally, but there is no answer
    # to persist or return. Keep that distinct from an authoritative final.
    missing_final = (
        not turn.interrupted
        and turn.error is None
        and (
            not isinstance(turn.final_text, str)
            or not turn.final_text.strip()
        )
    )
    effective_error = turn.error
    if missing_final:
        effective_error = (
            "codex app-server turn completed without final assistant text; "
            "a requested approval may have expired or been unavailable"
        )

    # External memory provider sync (mirrors line ~15439). Skipped on
    # interrupt/error to avoid feeding partial transcripts to memory.
    if not turn.interrupted and effective_error is None:
        try:
            agent._sync_external_memory_for_turn(
                original_user_message=original_user_message,
                final_response=turn.final_text,
                interrupted=False,
                messages=messages,
            )
        except Exception:
            logger.debug("external memory sync raised", exc_info=True)

    # Background review fork — same cadence + signature as the default
    # path (line ~15449). Only fires when a trigger actually tripped AND
    # we have a real final response.
    if (
        turn.final_text
        and not turn.interrupted
        and (should_review_memory or should_review_skills)
    ):
        try:
            agent._spawn_background_review(
                messages_snapshot=list(messages),
                review_memory=should_review_memory,
                review_skills=should_review_skills,
            )
        except Exception:
            logger.debug("background review spawn raised", exc_info=True)

    return {
        "final_response": turn.final_text,
        "messages": messages,
        "api_calls": api_calls,
        "completed": not turn.interrupted and effective_error is None,
        "partial": turn.interrupted or effective_error is not None,
        "interrupted": _user_interrupted,
        **(
            {"interrupt_message": _interrupt_message}
            if _interrupt_message
            else {}
        ),
        "error": effective_error,
        **({"error_code": turn.error_code} if turn.error_code else {}),
        # The codex app-server runtime IS an early-return path that bypasses
        # conversation_loop, but we flush the projected assistant/tool messages
        # ourselves above (see the _flush_messages_to_session_db call after
        # messages.extend). The inbound user turn was already flushed at turn
        # start (turn_context._persist_session) and the flush dedups via
        # _DB_PERSISTED_MARKER, so state.db ends up with each real message
        # exactly once and session_search / conversation-distill see the full
        # gateway conversation. Report agent_persisted=True so the gateway
        # skips its own append_to_transcript DB write — writing again there
        # would re-INSERT the already-flushed user turn (append_message has no
        # dedup), reintroducing the #860 / #42039 duplicate-write bug.
        "agent_persisted": True,
        "codex_thread_id": turn.thread_id,
        "codex_turn_id": turn.turn_id,
        **usage_result,
    }


# ---------------------------------------------------------------------------
# Event-driven Responses streaming
#
# OpenAI ships its consumer Codex backend (chatgpt.com/backend-api/codex) on
# a different schedule from the openai Python SDK.  The high-level
# ``client.responses.stream(...)`` helper reconstructs a typed Response from
# the terminal ``response.completed`` event's ``response.output`` field, and
# when that field drifts to ``null`` (gpt-5.5, May 2026) the SDK raises
# ``TypeError: 'NoneType' object is not iterable`` mid-iteration.
#
# We sidestep the whole class of failure by going one level lower:
# ``client.responses.create(stream=True)`` returns the raw AsyncIterable of
# SSE events, and we assemble the final response object purely from
# ``response.output_item.done`` events as they arrive.  We never read
# ``response.completed.response.output`` for content reconstruction, so the
# backend can return ``null``, ``[]``, a string, or omit the field entirely
# and we don't care.
#
# This mirrors what the OpenClaw TS implementation does for the same backend
# and is structurally immune to the bug class rather than patched.
# ---------------------------------------------------------------------------


_TERMINAL_EVENT_TYPES = frozenset({
    "response.completed",
    "response.incomplete",
    "response.failed",
})


def _event_field(event: Any, name: str, default: Any = None) -> Any:
    """Field access that handles both attr-style (SDK objects) and dict (raw JSON) events."""
    value = getattr(event, name, None)
    if value is None and isinstance(event, dict):
        value = event.get(name, default)
    return value if value is not None else default


def _item_field(item: Any, name: str, default: Any = None) -> Any:
    """Field access for nested Response items (attr-style SDK object or dict)."""
    value = getattr(item, name, None)
    if value is None and isinstance(item, dict):
        value = item.get(name, default)
    return value if value is not None else default


def _raise_stream_error(event: Any) -> None:
    """Raise a ``_StreamErrorEvent`` from a ``type=error`` SSE frame.

    The Responses spec puts the failure details at the top level of the
    frame (``{"type": "error", "code": ..., "message": ..., "param": ...}``),
    but the official OpenAI SDK and several OpenAI-compatible proxies wrap
    them in an HTTP-style nested envelope instead
    (``{"type": "error", "error": {"code": ..., "message": ..., "param": ...}}``).
    Read the top-level fields first, then fall back to the nested envelope so
    the error classifier sees the provider's real code/message (rate-limit vs
    context-overflow vs entitlement) rather than the generic placeholder.
    Port of anomalyco/opencode#36130.

    Imported lazily so this module stays importable from places that don't
    pull in ``run_agent`` (e.g. plugin code, doc tools).
    """
    from run_agent import _StreamErrorEvent

    nested = _event_field(event, "error")

    def _error_field(name: str) -> Any:
        value = _event_field(event, name)
        if value is None and nested is not None:
            value = _item_field(nested, name)
        return value

    raw_message = _error_field("message")
    if raw_message is not None and not isinstance(raw_message, str):
        raw_message = str(raw_message)
    message = (raw_message or "stream emitted error event").strip() or "stream emitted error event"
    raise _StreamErrorEvent(
        message,
        code=_error_field("code"),
        param=_error_field("param"),
    )


def _consume_codex_event_stream(
    event_iter: Any,
    *,
    model: str,
    on_text_delta=None,
    on_reasoning_delta=None,
    on_commentary_message=None,
    on_first_delta=None,
    on_event=None,
    interrupt_check=None,
) -> SimpleNamespace:
    """Consume a Codex Responses SSE event stream and return a final response.

    The returned object is a ``SimpleNamespace`` shaped like the SDK's typed
    ``Response`` for the fields downstream code actually reads:

    * ``output``: list of output items, assembled from ``response.output_item.done``.
      For tool-call turns this contains the function_call items; for plain-text
      turns it contains a synthesized ``message`` item built from streamed deltas
      if no message item was emitted directly.
    * ``output_text``: assembled text from ``response.output_text.delta`` deltas.
    * ``usage``: copied from the terminal event's ``response.usage`` (when present).
    * ``status``: ``completed`` / ``incomplete`` / ``failed`` (or ``completed`` if
      the stream ended without a terminal frame but produced content).
    * ``id``: ``response.id`` when present.
    * ``incomplete_details``: passed through for ``response.incomplete`` frames.
    * ``error``: passed through for ``response.failed`` frames.
    * ``model``: from kwargs (the wire model name is not authoritative).

    Critically, we never read ``response.output`` from the terminal event for
    content reconstruction — only ``usage``, ``status``, ``id``.  That field
    being ``null`` / ``[]`` / missing is fine.

    Callbacks:

    * ``on_text_delta(str)`` — fires per ``response.output_text.delta``, suppressed
      once a function_call event is seen (so tool-call turns don't bleed text
      into the chat).
    * ``on_reasoning_delta(str)`` — fires per ``response.reasoning.*.delta`` and
      ``phase=analysis`` message deltas. When no dedicated commentary callback
      is supplied, commentary also uses this legacy fallback.
    * ``on_commentary_message(str)`` — fires once per completed
      ``phase=commentary`` message, before any following tool item executes.
    * ``on_first_delta()`` — one-shot, fires on the first text delta only.
    * ``on_event(event)`` — fires for every event before any other processing.
      Used for watchdog activity, debug logging, anything wire-shape-agnostic.
    * ``interrupt_check()`` — returns True to break the loop early.
    """
    collected_output_items: List[Any] = []
    collected_text_deltas: List[str] = []
    has_tool_calls = False
    first_delta_fired = False
    active_message_phase: str | None = None
    commentary_text_deltas: List[str] = []
    # Last reasoning summary_index seen. The Responses stream delimits summary
    # parts by this index and gives each part no separator of its own, so a
    # change of index is where the blank line belongs.
    active_summary_index: Any = None
    terminal_status: str = "completed"
    terminal_usage: Any = None
    terminal_response_id: str = None
    terminal_incomplete_details: Any = None
    terminal_error: Any = None
    saw_terminal = False

    for event in event_iter:
        if on_event is not None:
            try:
                on_event(event)
            except (TimeoutError, InterruptedError):
                # Control-flow signals from watchdog/cancellation hooks must
                # propagate, not get swallowed as "debug noise".
                raise
            except Exception:
                # Genuine bugs in third-party debug/log hooks shouldn't break
                # stream consumption.
                logger.debug("Codex stream on_event hook raised", exc_info=True)
        if interrupt_check is not None and interrupt_check():
            break

        event_type = _event_field(event, "type", "")
        if not isinstance(event_type, str):
            event_type = ""

        # ``error`` SSE frames carry the provider's real failure reason
        # (subscription / quota / model-not-available / rejected-reasoning-replay)
        # but never appear in the terminal set.  Surface them as a structured
        # exception so the credential pool + error classifier see the body.
        if event_type == "error":
            _raise_stream_error(event)

        # Track the phase of the active streamed message item.  Codex/Harmony
        # ``commentary``/``analysis`` text is mid-turn preamble/progress
        # narration, never the final answer.  We still collect completed output
        # items for replay, but route those deltas to the reasoning callback so
        # they display like thinking text instead of assistant content.
        if event_type == "response.output_item.added":
            item = _event_field(event, "item")
            item_type = _item_field(item, "type", "")
            if item_type == "message":
                phase = _item_field(item, "phase", None)
                active_message_phase = phase.strip().lower() if isinstance(phase, str) else None
                if active_message_phase == "commentary":
                    commentary_text_deltas = []
            else:
                active_message_phase = None
            if "function_call" in str(item_type):
                has_tool_calls = True
            continue

        if "output_text.delta" in event_type or event_type == "response.output_text.delta":
            delta_text = _event_field(event, "delta", "")
            if delta_text and active_message_phase == "commentary":
                commentary_text_deltas.append(delta_text)
                # Preserve CLI/backward compatibility when no first-class
                # commentary consumer is installed.
                if on_commentary_message is None and on_reasoning_delta is not None:
                    try:
                        on_reasoning_delta(delta_text)
                    except Exception:
                        logger.debug("Codex stream on_reasoning_delta raised", exc_info=True)
            elif delta_text and active_message_phase == "analysis":
                if on_reasoning_delta is not None:
                    try:
                        on_reasoning_delta(delta_text)
                    except Exception:
                        logger.debug("Codex stream on_reasoning_delta raised", exc_info=True)
            elif delta_text:
                collected_text_deltas.append(delta_text)
                if not has_tool_calls:
                    if not first_delta_fired:
                        first_delta_fired = True
                        if on_first_delta is not None:
                            try:
                                on_first_delta()
                            except Exception:
                                logger.debug("Codex stream on_first_delta raised", exc_info=True)
                    if on_text_delta is not None:
                        try:
                            on_text_delta(delta_text)
                        except Exception:
                            logger.debug("Codex stream on_text_delta raised", exc_info=True)
            continue

        if "function_call" in event_type:
            has_tool_calls = True
            # fall through — function_call items still get added on output_item.done

        if "reasoning" in event_type and "delta" in event_type:
            reasoning_text = _event_field(event, "delta", "")
            if reasoning_text and on_reasoning_delta is not None:
                # Summary parts stream one after another with no separator of
                # their own; summary_index is the boundary the wire gives us.
                summary_index = _event_field(event, "summary_index")
                if (
                    summary_index is not None
                    and active_summary_index is not None
                    and summary_index != active_summary_index
                ):
                    reasoning_text = f"\n\n{reasoning_text}"
                if summary_index is not None:
                    active_summary_index = summary_index
                try:
                    on_reasoning_delta(reasoning_text)
                except Exception:
                    logger.debug("Codex stream on_reasoning_delta raised", exc_info=True)
            continue

        if event_type == "response.output_item.done":
            done_item = _event_field(event, "item")
            if done_item is not None:
                collected_output_items.append(done_item)
                done_phase = _item_field(done_item, "phase", None)
                done_phase = done_phase.strip().lower() if isinstance(done_phase, str) else None
                if done_phase == "commentary" and on_commentary_message is not None:
                    commentary_text = "".join(commentary_text_deltas).strip()
                    if not commentary_text:
                        content_parts = _item_field(done_item, "content", [])
                        if isinstance(content_parts, list):
                            commentary_text = "".join(
                                str(_item_field(part, "text", "") or "")
                                for part in content_parts
                                if _item_field(part, "type", "") == "output_text"
                            ).strip()
                    if commentary_text:
                        try:
                            on_commentary_message(commentary_text)
                        except Exception:
                            logger.debug(
                                "Codex stream on_commentary_message raised",
                                exc_info=True,
                            )
                    commentary_text_deltas = []
            continue

        if event_type in _TERMINAL_EVENT_TYPES:
            saw_terminal = True
            resp_obj = _event_field(event, "response")
            if resp_obj is not None:
                terminal_usage = getattr(resp_obj, "usage", None)
                if terminal_usage is None and isinstance(resp_obj, dict):
                    terminal_usage = resp_obj.get("usage")
                rid = getattr(resp_obj, "id", None)
                if rid is None and isinstance(resp_obj, dict):
                    rid = resp_obj.get("id")
                terminal_response_id = rid
                rstatus = getattr(resp_obj, "status", None)
                if rstatus is None and isinstance(resp_obj, dict):
                    rstatus = resp_obj.get("status")
                if isinstance(rstatus, str):
                    terminal_status = rstatus
                if event_type == "response.incomplete":
                    terminal_incomplete_details = getattr(resp_obj, "incomplete_details", None)
                    if terminal_incomplete_details is None and isinstance(resp_obj, dict):
                        terminal_incomplete_details = resp_obj.get("incomplete_details")
                if event_type == "response.failed":
                    terminal_error = getattr(resp_obj, "error", None)
                    if terminal_error is None and isinstance(resp_obj, dict):
                        terminal_error = resp_obj.get("error")
            if event_type == "response.completed":
                terminal_status = terminal_status or "completed"
            elif event_type == "response.incomplete":
                terminal_status = terminal_status or "incomplete"
            elif event_type == "response.failed":
                terminal_status = terminal_status or "failed"
            # Stop on terminal event.
            break

    # Build the final output list.  Prefer items observed via output_item.done;
    # if none arrived but we streamed plain text deltas (no tool calls), synthesize
    # a single message item so downstream normalization has something to work with.
    if collected_output_items:
        output = list(collected_output_items)
    elif collected_text_deltas and not has_tool_calls:
        assembled = "".join(collected_text_deltas)
        output = [SimpleNamespace(
            type="message",
            role="assistant",
            status="completed",
            content=[SimpleNamespace(type="output_text", text=assembled)],
        )]
    else:
        output = []

    # If the stream ended without any terminal event AND produced no usable
    # content (no items, no text deltas), surface that as a RuntimeError so
    # callers can distinguish "stream truncated mid-flight / provider rejected
    # the call" from "stream completed with empty body".  This preserves the
    # signal the SDK's high-level helper used to raise as
    # ``RuntimeError("Didn't receive a `response.completed` event.")``.
    if not saw_terminal and not output:
        raise RuntimeError(
            "Codex Responses stream did not emit a terminal response"
        )

    assembled_text = "".join(collected_text_deltas)

    final = SimpleNamespace(
        output=output,
        output_text=assembled_text,
        usage=terminal_usage,
        status=terminal_status,
        id=terminal_response_id,
        model=model,
        incomplete_details=terminal_incomplete_details,
        error=terminal_error,
    )
    return final


def _sanitize_consumer_codex_request(
    agent: Any,
    request: dict[str, Any],
) -> dict[str, Any]:
    """Drop fields the ChatGPT OAuth Codex endpoint does not accept.

    This guard intentionally lives at the final wire boundary, after Relay or
    other request middleware has had a chance to transform the request. The
    normal transport builder already omits ``prompt_cache_retention`` for this
    endpoint, but a late mutation must not be allowed to turn a valid tool
    follow-up into a non-retryable HTTP 400.

    Explicit ``request_overrides`` are subject to the same endpoint contract:
    unsupported retention is dropped with a warning instead of being sent and
    rejected by the provider. The check covers both the top-level kwarg and a
    nested ``extra_body`` entry — the OpenAI SDK merges ``extra_body`` into
    the outgoing JSON body, so either shape reaches the endpoint.
    """
    sanitized = dict(request)
    # Resolved defensively on purpose: run_codex_stream is also driven with
    # lightweight stand-in agents that carry only the attributes a given path
    # needs (see tests/agent/test_codex_request_transport_diagnostics.py), so a
    # bare agent._is_codex_backend() here would raise AttributeError on them.
    backend_predicate = getattr(agent, "_is_codex_backend", None)
    is_consumer_codex = (
        bool(backend_predicate()) if callable(backend_predicate) else False
    )
    if not is_consumer_codex:
        return sanitized
    dropped_from: list[str] = []
    if "prompt_cache_retention" in sanitized:
        sanitized.pop("prompt_cache_retention")
        dropped_from.append("top-level")
    # The OpenAI SDK merges ``extra_body`` into the outgoing JSON body, so a
    # nested ``extra_body.prompt_cache_retention`` reaches the endpoint just
    # like the top-level field would. Copy before editing — the caller's
    # mapping must not be mutated — and drop the mapping when it empties.
    extra_body = sanitized.get("extra_body")
    if isinstance(extra_body, dict) and "prompt_cache_retention" in extra_body:
        extra_body = dict(extra_body)
        extra_body.pop("prompt_cache_retention")
        if extra_body:
            sanitized["extra_body"] = extra_body
        else:
            sanitized.pop("extra_body")
        dropped_from.append("extra_body")
    if dropped_from:
        logger.warning(
            "Dropped unsupported prompt_cache_retention at consumer Codex "
            "wire boundary (model=%s, via %s).",
            sanitized.get("model", getattr(agent, "model", "unknown")),
            ", ".join(dropped_from),
        )
    return sanitized


def run_codex_stream(agent, api_kwargs: dict, client: Any = None, on_first_delta=None):
    """Execute one streaming Responses API request and return the final response.

    Uses ``responses.create(stream=True)`` (low-level raw event iteration)
    rather than the high-level ``responses.stream(...)`` helper.  This makes
    us structurally immune to backend drift in the ``response.completed``
    payload shape — we never let the SDK reconstruct a typed object from
    the terminal event's ``output`` field.
    """
    import httpx as _httpx
    from openai import APIConnectionError as _APIConnectionError

    from agent import relay_llm

    active_client = client or agent._ensure_primary_openai_client(reason="codex_stream_direct")
    max_stream_retries = 1
    # Accumulate streamed text so callers / compat shims can read it.
    agent._codex_streamed_text_parts: list = []

    def _on_text_delta(text: str) -> None:
        agent._codex_streamed_text_parts.append(text)
        agent._fire_stream_delta(text)

    def _on_reasoning_delta(text: str) -> None:
        agent._fire_reasoning_delta(text)

    def _on_commentary_message(text: str) -> None:
        agent._fire_streamed_codex_commentary(text)

    def _on_event(event: Any) -> None:
        # TTFB watchdog and activity touch — runs once per SSE event.
        agent._codex_stream_last_event_ts = time.time()
        agent._touch_activity("receiving stream response")

    for attempt in range(max_stream_retries + 1):
        if agent._interrupt_requested:
            raise InterruptedError("Agent interrupted before Codex stream retry")

        intercepted_events = []
        writer_token = {"value": None}

        def _open_codex_stream(next_api_kwargs: dict[str, Any]):
            stream_kwargs = _sanitize_consumer_codex_request(
                agent,
                next_api_kwargs,
            )
            stream_kwargs["stream"] = True
            return active_client.responses.create(**stream_kwargs)

        def _codex_stream_created(_raw_stream: Any) -> None:
            # Claim the delta sink for THIS physical attempt. A newer attempt
            # supersedes this token and fences late deltas out of the turn.
            writer_token["value"] = claim_stream_writer(agent)

        def _accept_codex_chunk(_chunk: Any) -> bool:
            token = writer_token["value"]
            if token is None or stream_writer_is_current(agent, token):
                return True
            logger.warning(
                "Codex streaming attempt superseded by a newer stream; "
                "stopping consumption to preserve the single-writer "
                "invariant (model=%s).",
                api_kwargs.get("model", "unknown"),
            )
            return False

        def _finalize_codex_stream() -> Any:
            return _consume_codex_event_stream(
                list(intercepted_events),
                model=api_kwargs.get("model"),
            )

        try:
            event_stream = relay_llm.stream(
                dict(api_kwargs),
                _open_codex_stream,
                session_id=str(getattr(agent, "session_id", "") or ""),
                name=str(getattr(agent, "provider", "") or "codex"),
                model_name=str(api_kwargs.get("model") or ""),
                finalizer=_finalize_codex_stream,
                on_stream_created=_codex_stream_created,
                on_chunk=intercepted_events.append,
                chunk_adapter=lambda chunk: chunk,
                accept_chunk=_accept_codex_chunk,
                completed_response_predicate=lambda response: bool(
                    hasattr(response, "output") and not hasattr(response, "__iter__")
                ),
                metadata={
                    "api_mode": "codex_responses",
                    "api_request_id": getattr(agent, "_current_api_request_id", None),
                    "call_role": (
                        "delegated"
                        if getattr(agent, "is_subagent", False)
                        else "fallback"
                        if int(getattr(agent, "_fallback_index", 0) or 0) > 0
                        else "primary"
                    ),
                    "retry_count": attempt,
                },
                defer_logical_completion=True,
            )
        except (
            _httpx.RemoteProtocolError,
            _httpx.ReadTimeout,
            _httpx.ConnectError,
            ConnectionError,
        ) as exc:
            if attempt < max_stream_retries:
                logger.debug(
                    "Codex Responses stream connect failed (attempt %s/%s); "
                    "retrying. %s error=%s",
                    attempt + 1,
                    max_stream_retries + 1,
                    agent._client_log_context(),
                    exc,
                )
                continue
            _log_codex_request_failure(
                agent,
                exc,
                stream_opened=writer_token["value"] is not None,
            )
            raise
        except _APIConnectionError as exc:
            _log_codex_request_failure(
                agent,
                exc,
                stream_opened=writer_token["value"] is not None,
            )
            raise

        def _interrupt_or_superseded() -> bool:
            return bool(agent._interrupt_requested)

        try:
            try:
                final = _consume_codex_event_stream(
                    event_stream,
                    model=api_kwargs.get("model"),
                    on_text_delta=_on_text_delta,
                    on_reasoning_delta=_on_reasoning_delta,
                    on_commentary_message=(
                        _on_commentary_message
                        if (
                            getattr(agent, "interim_assistant_callback", None) is not None
                            and getattr(agent, "show_commentary", True)
                        )
                        else None
                    ),
                    on_first_delta=on_first_delta,
                    on_event=_on_event,
                    interrupt_check=_interrupt_or_superseded,
                )
            except (_httpx.RemoteProtocolError, _httpx.ReadTimeout, _httpx.ConnectError, ConnectionError) as exc:
                if attempt < max_stream_retries:
                    logger.debug(
                        "Codex Responses stream transport failed mid-iteration "
                        "(attempt %s/%s); retrying. %s error=%s",
                        attempt + 1, max_stream_retries + 1,
                        agent._client_log_context(), exc,
                    )
                    continue
                _log_codex_request_failure(
                    agent,
                    exc,
                    stream_opened=writer_token["value"] is not None,
                )
                raise
            except RuntimeError:
                if event_stream.final_response is not None:
                    return event_stream.final_response
                raise
            except _APIConnectionError as exc:
                _log_codex_request_failure(
                    agent,
                    exc,
                    stream_opened=writer_token["value"] is not None,
                )
                raise

            # A terminal response has already been assembled at this point
            # (``final`` is built), so a transport error while draining the
            # rest of the iterator — done only to let Relay run its response
            # finalizer — must NOT discard it or trigger a new physical
            # request. Record it as a non-fatal finalization warning and
            # still return the already-completed, already-billed response.
            if not agent._interrupt_requested:
                try:
                    for _ignored in event_stream:
                        pass
                except (
                    _httpx.RemoteProtocolError,
                    _httpx.ReadTimeout,
                    _httpx.ConnectError,
                    ConnectionError,
                ) as exc:
                    logger.warning(
                        "Codex Responses stream transport finalization failed "
                        "after a terminal response was already received; "
                        "returning the completed response instead of "
                        "retrying. %s error=%s",
                        agent._client_log_context(), exc,
                    )
                except _APIConnectionError as exc:
                    _log_codex_request_failure(
                        agent,
                        exc,
                        stream_opened=writer_token["value"] is not None,
                    )
                    logger.warning(
                        "Codex Responses stream transport finalization failed "
                        "after a terminal response was already received; "
                        "returning the completed response instead of "
                        "retrying. %s error=%s",
                        agent._client_log_context(), exc,
                    )

            if final.status in {"incomplete", "failed"}:
                logger.warning(
                    "Codex Responses stream terminal status=%s "
                    "(incomplete_details=%s, error=%s, streamed_chars=%d). %s",
                    final.status, final.incomplete_details, final.error,
                    sum(len(p) for p in agent._codex_streamed_text_parts),
                    agent._client_log_context(),
                )

            return final
        finally:
            close_fn = getattr(event_stream, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    # A failed close can leave this response's connection
                    # checked out of the httpx pool while the caller's finally
                    # reports a reuse-reason close (e.g. interrupt_check broke
                    # the event loop with collected output) — caching the
                    # client with the leaked connection. Poison the slot so
                    # that close really closes the pool (owner-thread abort;
                    # mirrors the chat-streaming interrupt-break handling).
                    # ``client is None`` means the shared primary client,
                    # which is never reuse-cached and must not have its
                    # sockets force-shut here.
                    if client is not None:
                        agent._abort_request_openai_client(
                            active_client, reason="codex_stream_close_failed"
                        )


def run_codex_create_stream_fallback(agent, api_kwargs: dict, client: Any = None):
    """Backward-compatible alias for the unified event-driven path.

    Historically this was the fallback when the SDK's high-level
    ``responses.stream(...)`` helper raised on shape drift.  The primary
    path now does exactly what the fallback did, so this just forwards.
    Kept as a public symbol because tests and a small number of call sites
    still reference it by name.
    """
    return run_codex_stream(agent, api_kwargs, client=client)


__all__ = [
    "run_codex_app_server_turn",
    "run_codex_stream",
    "run_codex_create_stream_fallback",
    "_consume_codex_event_stream",
    "make_codex_app_server_event_bridge",
]
