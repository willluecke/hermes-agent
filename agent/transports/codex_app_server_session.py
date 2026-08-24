"""Session adapter for codex app-server runtime.

Owns one Codex thread per Hermes session. Drives `turn/start`, consumes
streaming notifications via CodexEventProjector, handles server-initiated
approval requests (apply_patch, exec command), translates cancellation,
and returns a clean turn result that AIAgent.run_conversation() can splice
into its `messages` list.

Lifecycle:
    session = CodexAppServerSession(cwd="/home/x/proj")
    session.ensure_started()                              # spawns + handshake + thread/start
    result = session.run_turn(user_input="hello")         # blocks until turn/completed
    # result.final_text          → assistant text returned to caller
    # result.projected_messages  → list of {role, content, ...} for messages list
    # result.tool_iterations     → how many tool-shaped items completed (skill nudge counter)
    # result.interrupted         → True if Ctrl+C / interrupt_requested fired mid-turn
    session.close()                                       # tears down subprocess

Threading model: the adapter is single-threaded from the caller's perspective.
The underlying CodexAppServerClient owns its own reader threads but exposes
blocking-with-timeout queues that this adapter polls in a loop, so the run_turn
call is synchronous and behaves like AIAgent's existing chat_completions loop.
"""

from __future__ import annotations

import base64
import binascii
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from agent.codex_responses_adapter import _format_responses_error
from agent.redact import redact_sensitive_text
from agent.transports.codex_app_server import (
    CodexAppServerClient,
    CodexAppServerError,
)
from agent.transports.codex_event_projector import CodexEventProjector

logger = logging.getLogger(__name__)


# How many tailing stderr lines from the codex subprocess to attach to a
# user-facing error when we don't have a more specific classification (OAuth,
# wedge watchdog, etc.). Small enough to keep error messages legible, large
# enough to surface a config/provider/auth diagnostic.
_STDERR_TAIL_LINES = 12


# Permission profile mapping mirrors the docstring in PR proposal:
# Hermes' tools.terminal.security_mode → Codex's permissions profile id.
# Defaults if config is missing → workspace-write (matches Codex's own default).
_HERMES_TO_CODEX_PERMISSION_PROFILE = {
    "auto": "workspace-write",
    "approval-required": "read-only-with-approval",
    "unrestricted": "full-access",
    # Backstop alias used by some skills/tests.
    "yolo": "full-access",
}

_DATA_IMAGE_EXTENSIONS = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/webp": "webp",
    "image/gif": "gif",
}
_MAX_LOCAL_IMAGE_BYTES = 10 * 1024 * 1024

# A reasoning model can legitimately emit no app-server notifications for
# several minutes after consuming a large tool result. Treat silence uniformly
# across the whole turn instead of imposing a shorter post-tool deadline.
_DEFAULT_TURN_INACTIVITY_TIMEOUT = 10 * 60.0
_DEFAULT_ABSOLUTE_TURN_TIMEOUT = 2 * 60 * 60.0


@dataclass
class TurnResult:
    """Result of one user→assistant→tool turn through the codex app-server."""

    final_text: str = ""
    # Set only by an agentMessage carrying phase=final_answer. This is stricter
    # than final_text because phase-less legacy messages are accepted only when
    # Codex also emits turn/completed.
    final_answer_seen: bool = False
    projected_messages: list[dict] = field(default_factory=list)
    tool_iterations: int = 0
    interrupted: bool = False
    error: Optional[str] = None  # Set if turn ended in a non-recoverable error
    turn_id: Optional[str] = None
    thread_id: Optional[str] = None
    token_usage_last: Optional[dict[str, Any]] = None
    token_usage_total: Optional[dict[str, Any]] = None
    model_context_window: Optional[int] = None
    compacted: bool = False
    # Hint to the caller that the underlying codex subprocess is likely
    # wedged (turn-level or absolute timeout fired, or token-refresh failure
    # killed the child). The caller should retire
    # the session so the next turn respawns codex from scratch instead
    # of riding a CPU-spinning or auth-broken process. Mirrors openclaw
    # beta.8's "retire timed-out app-server clients" fix.
    should_retire: bool = False


def _apply_projected_final_text(result: TurnResult, projection: Any) -> None:
    """Keep a phase-qualified final answer canonical once one is observed."""
    if projection.final_text is None:
        return
    if projection.is_final_answer or not result.final_answer_seen:
        result.final_text = projection.final_text
    if projection.is_final_answer:
        result.final_answer_seen = True


# Markers we accept as terminal even when codex never emits turn/completed.
# Some codex versions stream `<turn_aborted>` as raw text in agentMessage
# items when an interrupt or upstream error tears the turn down before the
# normal completion path fires. Mirrors openclaw beta.8 fix.
_TURN_ABORTED_MARKERS = ("<turn_aborted>", "<turn_aborted/>")


def _notification_scope_ids(
    note: dict,
) -> tuple[Optional[str], Optional[str]]:
    """Extract the thread/turn identity carried by a notification."""
    if not isinstance(note, dict):
        return None, None
    params = note.get("params") or {}
    if not isinstance(params, dict):
        return None, None

    nested_turn = params.get("turn") or {}
    nested_item = params.get("item") or {}

    observed_thread_id = params.get("threadId") or params.get("thread_id")
    if observed_thread_id is None and isinstance(nested_turn, dict):
        observed_thread_id = (
            nested_turn.get("threadId")
            or nested_turn.get("thread_id")
        )
    if observed_thread_id is None and isinstance(nested_item, dict):
        observed_thread_id = (
            nested_item.get("threadId")
            or nested_item.get("thread_id")
        )

    observed_turn_id = params.get("turnId") or params.get("turn_id")
    if observed_turn_id is None and isinstance(nested_turn, dict):
        observed_turn_id = nested_turn.get("id") or nested_turn.get("turnId")
    if observed_turn_id is None and isinstance(nested_item, dict):
        observed_turn_id = (
            nested_item.get("turnId")
            or nested_item.get("turn_id")
        )

    return observed_thread_id, observed_turn_id


def _notification_belongs_to_turn(
    note: dict,
    *,
    thread_id: Optional[str],
    turn_id: Optional[str],
) -> bool:
    """Return whether a multiplexed notification belongs to this turn.

    Codex app-server can carry parent and hosted subagent threads over one
    JSON-RPC connection.  An explicitly foreign child or
    stale-turn event must not mutate the active parent's transcript or mark
    its turn complete.  Unscoped notifications remain accepted for protocol
    compatibility.
    """
    if not isinstance(note, dict):
        return False

    observed_thread_id, observed_turn_id = _notification_scope_ids(note)

    if (
        thread_id is not None
        and observed_thread_id is not None
        and str(observed_thread_id) != str(thread_id)
    ):
        return False

    if (
        turn_id is not None
        and observed_turn_id is not None
        and str(observed_turn_id) != str(turn_id)
    ):
        return False

    return True


def _image_url_from_content_part(item: dict[str, Any]) -> str:
    image_value = item.get("image_url")
    if isinstance(image_value, dict):
        image_value = image_value.get("url")
    if not isinstance(image_value, str) or not image_value.strip():
        image_value = item.get("url")
    return image_value.strip() if isinstance(image_value, str) else ""


def _image_bytes_match_type(data: bytes, media_type: str) -> bool:
    if media_type == "image/png":
        return data.startswith(b"\x89PNG\r\n\x1a\n")
    if media_type in {"image/jpeg", "image/jpg"}:
        return data.startswith(b"\xff\xd8\xff")
    if media_type == "image/gif":
        return data.startswith((b"GIF87a", b"GIF89a"))
    if media_type == "image/webp":
        return (
            len(data) >= 12
            and data.startswith(b"RIFF")
            and data[8:12] == b"WEBP"
        )
    return False


def _materialize_data_image(
    url: str,
    *,
    directory: str,
    index: int,
) -> str:
    header, separator, encoded = url.partition(",")
    header_parts = header.split(";")
    if (
        separator != ","
        or len(header_parts) != 2
        or header_parts[1].lower() != "base64"
    ):
        raise ValueError("image data URL must contain a base64 payload")
    media_type = (
        header_parts[0][5:].lower()
        if header_parts[0].lower().startswith("data:")
        else ""
    )
    extension = _DATA_IMAGE_EXTENSIONS.get(media_type)
    if extension is None:
        raise ValueError(f"unsupported image media type: {media_type or 'unknown'}")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image data URL contains invalid base64") from exc
    if not data:
        raise ValueError("image attachment is empty")
    if len(data) > _MAX_LOCAL_IMAGE_BYTES:
        raise ValueError("image attachment exceeds the 10 MiB runtime limit")
    if not _image_bytes_match_type(data, media_type):
        raise ValueError(f"image bytes do not match declared media type {media_type}")

    path = os.path.join(directory, f"attachment-{index}.{extension}")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        raise
    return path


def _prepare_turn_input_items(
    user_input: Any,
) -> tuple[list[dict[str, str]], Optional[tempfile.TemporaryDirectory]]:
    """Translate Hermes/OpenAI content parts to native Codex user inputs.

    Codex app-server accepts remote images as ``image`` items and local files
    as ``localImage`` items. Inline data URLs are decoded into a private
    temporary directory that remains alive for the whole turn.
    """
    if isinstance(user_input, str):
        return [{"type": "text", "text": user_input}], None
    if not isinstance(user_input, list):
        text = "" if user_input is None else str(user_input)
        return [{"type": "text", "text": text}], None

    prepared: list[dict[str, str]] = []
    temp_dir: Optional[tempfile.TemporaryDirectory] = None
    try:
        for item in user_input:
            if isinstance(item, str):
                if item.strip():
                    prepared.append({"type": "text", "text": item})
                continue
            if not isinstance(item, dict):
                if item is not None:
                    prepared.append({"type": "text", "text": str(item)})
                continue

            item_type = str(item.get("type") or "")
            if item_type in {"text", "input_text"}:
                text = item.get("text") or item.get("content") or ""
                if text:
                    prepared.append({"type": "text", "text": str(text)})
                continue
            if item_type not in {"image", "image_url", "input_image"}:
                continue

            image_url = _image_url_from_content_part(item)
            if not image_url:
                raise ValueError("image content part is missing its URL")
            if image_url.lower().startswith("data:"):
                if temp_dir is None:
                    temp_dir = tempfile.TemporaryDirectory(prefix="hermes-codex-image-")
                path = _materialize_data_image(
                    image_url,
                    directory=temp_dir.name,
                    index=len(prepared),
                )
                prepared.append({"type": "localImage", "path": path})
            elif image_url.lower().startswith(("https://", "http://")):
                prepared.append({"type": "image", "url": image_url})
            else:
                raise ValueError("image URL must use http(s) or data:image/...;base64")

        if not prepared:
            prepared.append({"type": "text", "text": "What do you see in this image?"})
        return prepared, temp_dir
    except Exception:
        if temp_dir is not None:
            temp_dir.cleanup()
        raise


# Substrings in codex stderr / JSON-RPC error messages that signal the
# subprocess died because its OAuth credentials are no longer valid.
# Kept conservative: we only redirect users to `codex login` when we're
# reasonably sure that's the actual failure, otherwise we surface the
# original error verbatim. Mirrors openclaw beta.8's auth-refresh
# classification.
_OAUTH_REFRESH_FAILURE_HINTS = (
    "invalid_grant",
    "invalid grant",
    "refresh token",
    "refresh_token",
    "token refresh",
    "token_refresh",
    "token has expired",
    "expired_token",
    "expired token",
    "not authenticated",
    "unauthenticated",
    "unauthorized",
    "401 unauthorized",
    "re-authenticate",
    "reauthenticate",
    "please log in",
    "please login",
    "auth profile",
    "no auth profile",
    "oauth",
)


def _classify_oauth_failure(*parts: str) -> Optional[str]:
    """Return a user-friendly re-auth hint if any of the provided strings
    look like a codex OAuth/token-refresh failure; otherwise None.

    Used for both `turn/start` JSON-RPC errors and post-mortem stderr
    inspection when the subprocess exits unexpectedly. Conservative on
    purpose — we only redirect users to `codex login` when the signal
    is strong, so unrelated runtime failures still surface verbatim.
    """
    haystack = " ".join(p for p in parts if p).lower()
    if not haystack:
        return None
    for needle in _OAUTH_REFRESH_FAILURE_HINTS:
        if needle in haystack:
            return (
                "Codex authentication failed — your ChatGPT/Codex login "
                "looks expired or invalid. Run `codex login` to refresh, "
                "then retry. (Fall back to default runtime with "
                "`/codex-runtime auto` if the issue persists.)"
            )
    return None


@dataclass
class _ServerRequestRouting:
    """Default policies for codex-side approval requests when no interactive
    callback is wired in. These are only used by tests + cron / non-interactive
    contexts; the live CLI path passes an approval_callback that defers to
    tools.approval.prompt_dangerous_approval()."""

    auto_approve_exec: bool = False
    auto_approve_apply_patch: bool = False
    guard_no_prompt_exec: bool = False
    guard_no_prompt_file_changes: bool = False


@dataclass(frozen=True)
class _PendingFileChange:
    summary: str
    kinds: frozenset[str]


class CodexAppServerSession:
    """One Codex thread per Hermes session, lifetime owned by AIAgent.

    Not thread-safe — one caller drives it at a time, matching how AIAgent's
    run_conversation() loop is structured today. The codex client itself can
    handle interleaved reads/writes via its own threads, but the adapter's
    state (projector, thread_id, turn counter) is owned by the caller thread.
    """

    def __init__(
        self,
        *,
        cwd: Optional[str] = None,
        codex_bin: str = "codex",
        codex_home: Optional[str] = None,
        hermes_session_id: Optional[str] = None,
        resume_thread_id: Optional[str] = None,
        model: Optional[str] = None,
        effort: Optional[str] = None,
        require_exact: bool = False,
        permission_profile: Optional[str] = None,
        approval_callback: Optional[Callable[..., str]] = None,
        on_event: Optional[Callable[[dict], None]] = None,
        request_routing: Optional[_ServerRequestRouting] = None,
        client_factory: Optional[Callable[..., CodexAppServerClient]] = None,
    ) -> None:
        self._cwd = cwd or os.getcwd()
        self._codex_bin = codex_bin
        self._codex_home = codex_home
        self._hermes_session_id = str(hermes_session_id or "").strip()
        self._resume_thread_id = str(resume_thread_id or "").strip()
        self._resumed_existing_thread = False
        self._model = str(model or "").strip()
        self._effort = str(effort or "").strip().lower()
        self._require_exact = bool(require_exact)
        self._exact_runtime_validated = False
        self._permission_profile = (
            permission_profile or _HERMES_TO_CODEX_PERMISSION_PROFILE.get(
                os.environ.get("HERMES_TERMINAL_SECURITY_MODE", "auto"),
                "workspace-write",
            )
        )
        self._approval_callback = approval_callback
        self._on_event = on_event  # Display hook (kawaii spinner ticks etc.)
        self._routing = request_routing or _ServerRequestRouting()
        self._client_factory = client_factory or CodexAppServerClient

        self._client: Optional[CodexAppServerClient] = None
        self._thread_id: Optional[str] = None
        self._interrupt_event = threading.Event()
        self._active_turn_id: Optional[str] = None
        self._active_turn_lock = threading.Lock()
        # Pending file-change items, keyed by item id. Populated on
        # item/started for fileChange items; consumed by the approval
        # bridge when codex sends item/fileChange/requestApproval. The
        # approval params don't carry the changeset, so we cache here
        # to surface a real summary in the approval prompt (quirk #4).
        self._pending_file_changes: dict[str, _PendingFileChange] = {}
        self._closed = False

    # ---------- lifecycle ----------

    def _validate_exact_runtime(self) -> None:
        """Fail closed unless the configured model and effort are available."""
        if not self._require_exact or self._exact_runtime_validated:
            return
        if not self._model or not self._effort:
            raise CodexAppServerError(
                code=-32602,
                message=(
                    "exact Codex runtime requires both an explicit model and "
                    "reasoning effort"
                ),
            )
        assert self._client is not None
        cursor: Optional[str] = None
        found: Optional[dict[str, Any]] = None
        for _ in range(20):
            params: dict[str, Any] = {"includeHidden": True}
            if cursor:
                params["cursor"] = cursor
            response = self._client.request("model/list", params, timeout=15)
            for candidate in response.get("data", response.get("models", [])):
                candidate_id = str(
                    candidate.get("id")
                    or candidate.get("model")
                    or candidate.get("slug")
                    or ""
                )
                if candidate_id == self._model:
                    found = candidate
                    break
            if found is not None:
                break
            cursor = response.get("nextCursor")
            if not cursor:
                break
        if found is None:
            raise CodexAppServerError(
                code=-32602,
                message=f"required Codex model {self._model!r} is unavailable",
            )
        efforts = {
            str(item.get("reasoningEffort") or item.get("effort") or "").lower()
            for item in found.get("supportedReasoningEfforts", [])
            if isinstance(item, dict)
        }
        if self._effort not in efforts:
            raise CodexAppServerError(
                code=-32602,
                message=(
                    f"required reasoning effort {self._effort!r} is unavailable "
                    f"for Codex model {self._model!r}"
                ),
            )
        self._exact_runtime_validated = True

    def ensure_started(self) -> str:
        """Spawn the subprocess, do the initialize handshake, and start a
        thread. Returns the codex thread id. Idempotent — repeated calls
        return the same thread id."""
        if self._thread_id is not None:
            return self._thread_id
        if self._client is None:
            client_env = (
                {"HERMES_GATEWAY_SESSION_ID": self._hermes_session_id}
                if self._hermes_session_id
                else None
            )
            client_kwargs: dict[str, Any] = {
                "codex_bin": self._codex_bin,
                "codex_home": self._codex_home,
            }
            if client_env is not None:
                client_kwargs["env"] = client_env
            self._client = self._client_factory(**client_kwargs)
        self._client.initialize(
            client_name="hermes",
            client_title="Hermes Agent",
            client_version=_get_hermes_version(),
        )
        self._validate_exact_runtime()
        # Permission selection is intentionally NOT sent on thread/start.
        # Two reasons (live-tested against codex 0.130.0):
        #   1. `thread/start.permissions` is gated behind the experimentalApi
        #      capability on this codex version — we'd have to opt in during
        #      initialize and accept the unstable surface.
        #   2. Even with experimentalApi declared and the correct shape
        #      (`{"type": "profile", "id": "..."}`, not `{"profileId": ...}`),
        #      codex requires a matching `[permissions]` table in
        #      ~/.codex/config.toml or it fails the request with
        #      'default_permissions requires a [permissions] table'.
        # Letting codex pick its default (`:read-only` unless the user has
        # configured otherwise in their codex config.toml) is the standard
        # codex CLI workflow and avoids fighting codex's own validation.
        # Users who want a write-capable profile configure it in their
        # ~/.codex/config.toml the same way they would for any codex usage.
        params: dict[str, Any] = {"cwd": self._cwd}
        if self._model:
            params["model"] = self._model
        method = "thread/start"
        if self._resume_thread_id:
            method = "thread/resume"
            params["threadId"] = self._resume_thread_id
            try:
                result = self._client.request(method, params, timeout=15)
                self._resumed_existing_thread = True
            except (CodexAppServerError, TimeoutError) as exc:
                logger.warning(
                    "codex thread resume failed for %s; starting a fresh "
                    "thread with durable Hermes history fallback: %s",
                    self._resume_thread_id[:8],
                    exc,
                )
                method = "thread/start"
                params.pop("threadId", None)
                result = self._client.request(method, params, timeout=15)
        else:
            result = self._client.request(method, params, timeout=15)
        # Cross-fill thread.id/sessionId — different codex versions have
        # serialized this under either key. Mirrors openclaw beta.8's
        # tolerance fix so future codex drops/renames don't KeyError us
        # at handshake time.
        thread_obj = result.get("thread") or {}
        thread_id = (
            thread_obj.get("id")
            or thread_obj.get("sessionId")
            or result.get("sessionId")
            or result.get("threadId")
        )
        if not thread_id:
            raise CodexAppServerError(
                code=-32603,
                message=(
                    f"codex {method} returned no thread id "
                    f"(payload keys: {sorted(result.keys())})"
                ),
            )
        self._thread_id = thread_id
        logger.info(
            "codex app-server thread %s: id=%s profile=%s cwd=%s",
            "resumed" if self._resumed_existing_thread else "started",
            self._thread_id[:8],
            self._permission_profile,
            self._cwd,
        )
        return self._thread_id

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._active_turn_lock:
            self._active_turn_id = None
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass
            self._client = None
        self._thread_id = None

    def __enter__(self) -> "CodexAppServerSession":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---------- interrupt ----------

    def request_interrupt(self) -> None:
        """Idempotent: signal the active turn loop to issue turn/interrupt
        and unwind. Called by AIAgent's _interrupt_requested path."""
        self._interrupt_event.set()

    def request_steer(self, text: str) -> bool:
        """Append user guidance to the active Codex turn via ``turn/steer``."""
        cleaned = str(text or "").strip()
        if not cleaned:
            return False
        with self._active_turn_lock:
            turn_id = self._active_turn_id
            thread_id = self._thread_id
            client = self._client
        if not turn_id or not thread_id or client is None:
            return False
        try:
            response = client.request(
                "turn/steer",
                {
                    "threadId": thread_id,
                    "input": [{"type": "text", "text": cleaned}],
                    "expectedTurnId": turn_id,
                },
                timeout=10,
            )
        except (CodexAppServerError, TimeoutError):
            logger.debug("turn/steer rejected for active Codex turn", exc_info=True)
            return False
        accepted_turn_id = response.get("turnId") if isinstance(response, dict) else None
        return accepted_turn_id in {None, turn_id}

    # ---------- diagnostics ----------

    def _format_error_with_stderr(
        self,
        prefix: str,
        exc: Any = "",
        *,
        tail_lines: int = _STDERR_TAIL_LINES,
    ) -> str:
        """Build a user-facing error string for codex failures.

        Appends the last few lines of codex's stderr buffer when available,
        passed through agent.redact with force=True so secrets in provider
        error responses (auth headers, query-string tokens, sk-* keys) never
        leak into chat output or trajectories. The codex CLI's own error
        text ('Internal error', 'turn/start failed: ...') is otherwise
        opaque and forces users to re-run with verbose flags to diagnose
        config / provider / auth-bridge problems.

        Use this for the generic / catch-all branches. Specific
        classifications (OAuth via _classify_oauth_failure, post-tool wedge
        watchdog) already produce a clean hint and should be used instead.
        """
        exc_str = str(exc) if exc != "" and exc is not None else ""
        base = f"{prefix}: {exc_str}" if exc_str else prefix
        if self._client is None:
            return base
        try:
            tail = self._client.stderr_tail(tail_lines)
        except Exception:  # pragma: no cover - diagnostic best-effort
            return base
        if not tail:
            return base
        joined = "\n".join(line.rstrip() for line in tail if line)
        if not joined.strip():
            return base
        redacted = redact_sensitive_text(joined, force=True)
        return f"{base}\ncodex stderr (last {len(tail)} lines):\n{redacted}"

    # ---------- per-turn ----------

    def run_turn(
        self,
        user_input: Any,
        *,
        turn_timeout: float = _DEFAULT_TURN_INACTIVITY_TIMEOUT,
        notification_poll_timeout: float = 0.25,
        absolute_turn_timeout: float = _DEFAULT_ABSOLUTE_TURN_TIMEOUT,
    ) -> TurnResult:
        """Send a user message and block until turn/completed, while
        forwarding server-initiated approval requests and projecting items
        into Hermes' messages shape.

        ``turn_timeout`` is an inactivity limit, not a wall-clock turn budget.
        A long Codex turn may run past it while notifications attributable to
        this turn continue to arrive. Set it to ``0`` to disable the inactivity
        limit. Tool completions use this same limit because a quiet reasoning
        phase after a tool result is normal app-server behavior.

        ``absolute_turn_timeout`` is the wall-clock ceiling for one turn even
        when notifications continue to arrive. Set it to ``0`` to disable the
        ceiling. Process exit, protocol failure, and explicit interruption are
        still handled immediately.
        """
        # Pre-create the result so startup failures (codex subprocess can't
        # spawn, initialize handshake rejects, thread/start blows up) surface
        # the same way per-turn failures do — with a TurnResult.error string
        # the caller can render — instead of bubbling raw codex exceptions
        # up to AIAgent.run_conversation.
        result = TurnResult()
        try:
            self.ensure_started()
        except (CodexAppServerError, TimeoutError) as exc:
            result.error = self._format_error_with_stderr(
                "codex app-server startup failed", exc
            )
            # Subprocess almost certainly unhealthy — retire so the next
            # turn re-spawns cleanly.
            result.should_retire = True
            self._interrupt_event.clear()
            return result
        assert self._client is not None and self._thread_id is not None
        result.thread_id = self._thread_id

        # Do not clear here: a hard stop can arrive while ensure_started() is
        # spawning/initializing the subprocess. Honor it before launching a
        # Codex turn instead of erasing the signal.
        if self._interrupt_event.is_set():
            result.interrupted = True
            self._interrupt_event.clear()
            return result
        projector = CodexEventProjector()

        try:
            turn_input, image_temp_dir = _prepare_turn_input_items(user_input)
        except ValueError as exc:
            result.error = f"invalid image attachment: {exc}"
            self._interrupt_event.clear()
            return result

        # Keep image_temp_dir alive until the terminal notification. Codex
        # app-server receives real image/localImage items rather than a text
        # placeholder, so the model can inspect browser uploads natively.
        try:
            ts = self._client.request(
                "turn/start",
                {
                    "threadId": self._thread_id,
                    "input": turn_input,
                    **({"model": self._model} if self._model else {}),
                    **({"effort": self._effort} if self._effort else {}),
                },
                timeout=10,
            )
        except CodexAppServerError as exc:
            if image_temp_dir is not None:
                image_temp_dir.cleanup()
            # Classify auth/refresh failures so the user gets a clear
            # `codex login` pointer instead of a raw RPC error string.
            stderr_blob = "\n".join(self._client.stderr_tail(40))
            hint = _classify_oauth_failure(exc.message, stderr_blob)
            if hint is not None:
                result.error = hint
                # Subprocess is fine on a JSON-RPC level here, but the
                # token store is broken — retire so the next turn does a
                # clean handshake (and the user has a chance to re-auth
                # via `codex login` between turns).
                result.should_retire = True
            else:
                result.error = self._format_error_with_stderr(
                    "turn/start failed", exc
                )
            self._interrupt_event.clear()
            return result
        except TimeoutError as exc:
            if image_temp_dir is not None:
                image_temp_dir.cleanup()
            # turn/start hanging is a strong signal the subprocess is wedged.
            stderr_blob = "\n".join(self._client.stderr_tail(40))
            hint = _classify_oauth_failure(stderr_blob)
            result.error = hint or self._format_error_with_stderr(
                "turn/start timed out", exc
            )
            result.should_retire = True
            self._interrupt_event.clear()
            return result

        result.turn_id = (ts.get("turn") or {}).get("id")
        with self._active_turn_lock:
            self._active_turn_id = result.turn_id
        inactivity_timeout = turn_timeout if turn_timeout > 0 else None
        last_activity_at = time.monotonic()
        turn_started_at = last_activity_at
        absolute_timeout = (
            absolute_turn_timeout if absolute_turn_timeout > 0 else None
        )
        turn_complete = False

        while not turn_complete:
            now = time.monotonic()
            if (
                absolute_timeout is not None
                and (now - turn_started_at) >= absolute_timeout
            ):
                result.error = self._format_error_with_stderr(
                    f"turn exceeded absolute timeout of "
                    f"{absolute_turn_timeout:g}s"
                )
                break
            if (
                inactivity_timeout is not None
                and (now - last_activity_at) >= inactivity_timeout
            ):
                break
            if self._interrupt_event.is_set():
                self._issue_interrupt(result.turn_id)
                result.interrupted = True
                break

            # Detect a dead subprocess between iterations. If codex exited
            # (e.g. crashed, segfaulted, or its auth refresh thread killed
            # the process), we won't get any more notifications — bail out
            # rather than waiting for the full turn deadline.
            if not self._client.is_alive():
                stderr_blob = "\n".join(self._client.stderr_tail(60))
                hint = _classify_oauth_failure(stderr_blob)
                if hint is not None:
                    result.error = hint
                else:
                    result.error = self._format_error_with_stderr(
                        "codex app-server subprocess exited unexpectedly",
                        tail_lines=20,
                    )
                result.should_retire = True
                break

            # Drain any server-initiated requests (approvals) before
            # reading notifications, so the codex side isn't blocked.
            sreq = self._client.take_server_request(timeout=0)
            if sreq is not None:
                # Drain any pending notifications first so per-turn state
                # (e.g. _pending_file_changes for fileChange approvals) is
                # up to date when we make the approval decision. Bounded
                # to avoid starving the server-request response.
                for _ in range(8):
                    pending = self._client.take_notification(timeout=0)
                    if pending is None:
                        break
                    if not _notification_belongs_to_turn(
                        pending,
                        thread_id=self._thread_id,
                        turn_id=result.turn_id,
                    ):
                        logger.debug(
                            "ignoring foreign codex notification while draining "
                            "server request: method=%s",
                            pending.get("method"),
                        )
                        continue
                    event_at = time.monotonic()
                    last_activity_at = event_at
                    # Mirror the main notification-handling block below so
                    # display events surface and stay in step with projector
                    # state. Without this, item/started / item/completed
                    # events drained as part of the approval-roundtrip
                    # preamble are projected into messages but never reach
                    # the tool-progress display, silently hiding tool
                    # bubbles around approvals.
                    if self._on_event is not None:
                        try:
                            self._on_event(pending)
                        except Exception:  # pragma: no cover - display callback
                            logger.debug(
                                "on_event callback raised", exc_info=True
                            )
                    _apply_token_usage_notification(result, pending)
                    _apply_compaction_notification(result, pending)
                    self._track_pending_file_change(pending)
                    proj = projector.project(pending)
                    if proj.messages:
                        result.projected_messages.extend(proj.messages)
                    if proj.is_tool_iteration:
                        result.tool_iterations += 1
                    if proj.final_text is not None:
                        _apply_projected_final_text(result, proj)
                        if _has_turn_aborted_marker(proj.final_text):
                            turn_complete = True
                            result.interrupted = True
                            result.error = (
                                result.error
                                or "codex reported turn_aborted"
                            )
                self._handle_server_request(sreq)
                # The approval round-trip is live turn activity.
                last_activity_at = time.monotonic()
                continue

            note = self._client.take_notification(
                timeout=notification_poll_timeout
            )
            if note is None:
                continue

            method = note.get("method", "")
            if not _notification_belongs_to_turn(
                note,
                thread_id=self._thread_id,
                turn_id=result.turn_id,
            ):
                logger.debug(
                    "ignoring foreign codex notification: method=%s", method
                )
                continue

            event_at = time.monotonic()
            last_activity_at = event_at
            if self._on_event is not None:
                try:
                    self._on_event(note)
                except Exception:  # pragma: no cover - display callback
                    logger.debug("on_event callback raised", exc_info=True)

            _apply_token_usage_notification(result, note)
            _apply_compaction_notification(result, note)

            # Track in-progress fileChange items so the approval bridge
            # can surface a real change summary when codex requests
            # approval (the approval params themselves don't carry the
            # changeset). Quirk #4 fix.
            self._track_pending_file_change(note)

            # Project into messages
            projection = projector.project(note)
            if projection.messages:
                result.projected_messages.extend(projection.messages)
            if projection.is_tool_iteration:
                result.tool_iterations += 1
            if projection.final_text is not None:
                # Codex can emit multiple agentMessage items in one turn
                # (e.g. partial then final). Once an explicit final answer is
                # seen, a later phase-less legacy item cannot replace it.
                _apply_projected_final_text(result, projection)
                # Some codex builds tear a turn down by emitting a
                # `<turn_aborted>` marker in the agent message text and
                # never sending turn/completed. Treat the marker itself
                # as terminal so we don't burn the full deadline.
                if _has_turn_aborted_marker(projection.final_text):
                    turn_complete = True
                    result.interrupted = True
                    result.error = (
                        result.error or "codex reported turn_aborted"
                    )

            if method == "turn/completed":
                turn_complete = True
                turn_status = (
                    (note.get("params") or {}).get("turn") or {}
                ).get("status")
                if turn_status and turn_status not in {"completed", "interrupted"}:
                    err_obj = (
                        (note.get("params") or {}).get("turn") or {}
                    ).get("error")
                    if err_obj:
                        err_msg = _format_responses_error(err_obj, str(turn_status))
                        # If the turn failed for an auth/refresh reason,
                        # rewrite the error into a re-auth hint AND mark
                        # the session for retirement.
                        stderr_blob = "\n".join(
                            self._client.stderr_tail(40)
                        )
                        hint = _classify_oauth_failure(err_msg, stderr_blob)
                        if hint is not None:
                            result.error = hint
                            result.should_retire = True
                        else:
                            result.error = self._format_error_with_stderr(
                                f"turn ended status={turn_status}", err_msg
                            )

        if (
            not turn_complete
            and not result.interrupted
            and result.final_text
            and result.final_answer_seen
            and result.error is None
        ):
            logger.warning(
                "codex app-server turn reached deadline after a completed "
                "final-answer message but before turn/completed; accepting "
                "the phase-qualified text as the terminal response"
            )
            turn_complete = True

        if not turn_complete and not result.interrupted:
            # Hit the inactivity or absolute deadline. Issue interrupt to stop
            # wasted compute and retire the unfinished session so the next turn
            # does not inherit a potentially wedged process.
            self._issue_interrupt(result.turn_id)
            result.interrupted = True
            if not result.error:
                result.error = self._format_error_with_stderr(
                    f"turn timed out after {turn_timeout}s without activity"
                )
            result.should_retire = True

        with self._active_turn_lock:
            self._active_turn_id = None
        self._interrupt_event.clear()
        if image_temp_dir is not None:
            image_temp_dir.cleanup()
        return result

    def compact_thread(
        self,
        *,
        turn_timeout: float = 600.0,
        notification_poll_timeout: float = 0.25,
    ) -> TurnResult:
        """Trigger Codex-native history compaction for the current thread.

        `thread/compact/start` returns immediately; the actual compaction
        progress streams through the same turn/item notifications as a normal
        turn. We wait for the matching `turn/completed` so callers can treat a
        successful return as a completed compaction boundary.
        """
        result = TurnResult()
        try:
            self.ensure_started()
        except (CodexAppServerError, TimeoutError) as exc:
            result.error = self._format_error_with_stderr(
                "codex app-server startup failed", exc
            )
            result.should_retire = True
            return result

        assert self._client is not None and self._thread_id is not None
        result.thread_id = self._thread_id
        self._interrupt_event.clear()
        projector = CodexEventProjector()

        try:
            self._client.request(
                "thread/compact/start",
                {"threadId": self._thread_id},
                timeout=10,
            )
        except CodexAppServerError as exc:
            stderr_blob = "\n".join(self._client.stderr_tail(40))
            hint = _classify_oauth_failure(exc.message, stderr_blob)
            if hint is not None:
                result.error = hint
                result.should_retire = True
            else:
                result.error = self._format_error_with_stderr(
                    "thread/compact/start failed", exc
                )
            return result
        except TimeoutError as exc:
            stderr_blob = "\n".join(self._client.stderr_tail(40))
            hint = _classify_oauth_failure(stderr_blob)
            result.error = hint or self._format_error_with_stderr(
                "thread/compact/start timed out", exc
            )
            result.should_retire = True
            return result

        deadline = time.monotonic() + turn_timeout
        turn_complete = False

        while time.monotonic() < deadline and not turn_complete:
            if self._interrupt_event.is_set():
                self._issue_interrupt(result.turn_id)
                result.interrupted = True
                break

            if not self._client.is_alive():
                stderr_blob = "\n".join(self._client.stderr_tail(60))
                hint = _classify_oauth_failure(stderr_blob)
                if hint is not None:
                    result.error = hint
                else:
                    result.error = self._format_error_with_stderr(
                        "codex app-server subprocess exited unexpectedly",
                        tail_lines=20,
                    )
                result.should_retire = True
                break

            sreq = self._client.take_server_request(timeout=0)
            if sreq is not None:
                self._handle_server_request(sreq)
                continue

            note = self._client.take_notification(
                timeout=notification_poll_timeout
            )
            if note is None:
                continue

            method = note.get("method", "")
            observed_thread_id, observed_turn_id = _notification_scope_ids(note)
            if result.turn_id is None:
                if method == "turn/started":
                    if (
                        observed_thread_id is not None
                        and str(observed_thread_id) != str(self._thread_id)
                    ):
                        logger.debug(
                            "ignoring foreign compact turn/started: thread=%s",
                            observed_thread_id,
                        )
                        continue
                    if observed_turn_id is None:
                        logger.debug(
                            "ignoring compact turn/started without a turn id"
                        )
                        continue
                    result.turn_id = str(observed_turn_id)
                elif observed_turn_id is not None or method in {
                    "item/completed",
                    "turn/completed",
                }:
                    # thread/compact/start does not return a turn id. Until the
                    # new turn/started arrives, any terminal/projectable event
                    # is stale or cannot be safely attributed to this compaction.
                    logger.debug(
                        "ignoring codex notification before compact turn start: "
                        "method=%s",
                        method,
                    )
                    continue

            if not _notification_belongs_to_turn(
                note,
                thread_id=self._thread_id,
                turn_id=result.turn_id,
            ):
                logger.debug(
                    "ignoring foreign codex notification: method=%s", method
                )
                continue

            if self._on_event is not None:
                try:
                    self._on_event(note)
                except Exception:  # pragma: no cover - display callback
                    logger.debug("on_event callback raised", exc_info=True)

            _apply_token_usage_notification(result, note)
            _apply_compaction_notification(result, note)
            self._track_pending_file_change(note)

            projection = projector.project(note)
            if projection.messages:
                result.projected_messages.extend(projection.messages)
            if projection.is_tool_iteration:
                result.tool_iterations += 1
            if projection.final_text is not None:
                _apply_projected_final_text(result, projection)
                if _has_turn_aborted_marker(projection.final_text):
                    turn_complete = True
                    result.interrupted = True
                    result.error = (
                        result.error or "codex reported turn_aborted"
                    )

            if method == "turn/started":
                turn_obj = (note.get("params") or {}).get("turn") or {}
                result.turn_id = turn_obj.get("id") or result.turn_id
            elif method == "turn/completed":
                turn_complete = True
                turn_obj = (note.get("params") or {}).get("turn") or {}
                result.turn_id = turn_obj.get("id") or result.turn_id
                turn_status = turn_obj.get("status")
                if turn_status == "interrupted":
                    result.interrupted = True
                    result.error = result.error or "compact turn interrupted"
                elif turn_status and turn_status != "completed":
                    err_obj = turn_obj.get("error")
                    err_msg = _format_responses_error(err_obj, str(turn_status))
                    stderr_blob = "\n".join(self._client.stderr_tail(40))
                    hint = _classify_oauth_failure(err_msg, stderr_blob)
                    if hint is not None:
                        result.error = hint
                        result.should_retire = True
                    else:
                        result.error = self._format_error_with_stderr(
                            f"compact turn ended status={turn_status}",
                            err_msg,
                        )

        if not turn_complete and not result.interrupted:
            self._issue_interrupt(result.turn_id)
            result.interrupted = True
            if not result.error:
                result.error = self._format_error_with_stderr(
                    f"compact turn timed out after {turn_timeout}s"
                )
            result.should_retire = True

        return result

    # ---------- internals ----------

    def _issue_interrupt(self, turn_id: Optional[str]) -> None:
        if self._client is None or self._thread_id is None or turn_id is None:
            return
        try:
            self._client.request(
                "turn/interrupt",
                {"threadId": self._thread_id, "turnId": turn_id},
                timeout=5,
            )
        except CodexAppServerError as exc:
            # "no active turn to interrupt" is fine — already done.
            logger.debug("turn/interrupt non-fatal: %s", exc)
        except TimeoutError:
            logger.warning("turn/interrupt timed out")

    def _handle_server_request(self, req: dict) -> None:
        """Translate a codex server request (approval) into Hermes' approval
        flow, then send the response.

        Method names verified live against codex 0.130.0 (Apr 2026):
          item/commandExecution/requestApproval — exec approvals
          item/fileChange/requestApproval       — apply_patch approvals
          item/permissions/requestApproval      — permissions changes
                                                  (we decline; user controls
                                                  permission profile in
                                                  ~/.codex/config.toml).
        """
        if self._client is None:
            return
        method = req.get("method", "")
        rid = req.get("id")
        params = req.get("params") or {}

        if method == "item/commandExecution/requestApproval":
            decision = self._decide_exec_approval(params)
            self._client.respond(rid, {"decision": decision})
        elif method == "item/fileChange/requestApproval":
            decision = self._decide_apply_patch_approval(params)
            self._client.respond(rid, {"decision": decision})
        elif method == "item/permissions/requestApproval":
            # Codex sometimes asks to escalate permissions mid-turn. We
            # always cancel — the user already chose their permission
            # profile in ~/.codex/config.toml and surprise escalations
            # shouldn't be silently accepted. ``cancel`` also avoids falsely
            # presenting this client policy as a user rejection.
            self._client.respond(rid, {"decision": "cancel"})
        elif method == "mcpServer/elicitation/request":
            # Codex's MCP layer asks the user for structured input on
            # behalf of an MCP server (e.g. tool-call confirmation,
            # OAuth, form data). Auto-accept only the two local servers that
            # are part of this runtime contract. Their actual capability
            # surfaces remain bounded by Codex's per-server enabled_tools
            # configuration. For other MCP servers we decline so the user
            # explicitly opts in via codex's own auth flow.
            server_name = params.get("serverName") or ""
            if server_name in {"hermes-tools", "reccli"}:
                self._client.respond(
                    rid,
                    {"action": "accept", "content": None, "_meta": None},
                )
            else:
                self._client.respond(
                    rid,
                    {"action": "decline", "content": None, "_meta": None},
                )
        else:
            # Unknown server request — codex can extend this surface. Reject
            # cleanly so codex doesn't hang waiting for us.
            logger.warning("Unknown codex server request: %s", method)
            self._client.respond_error(
                rid, code=-32601, message=f"Unsupported method: {method}"
            )

    def _decide_exec_approval(self, params: dict) -> str:
        """Decide a Codex exec approval request.

        This is protocol-level routing only — it carries NO Hermes
        approval-mode/timeout logic. The Hermes-side resolution happens
        upstream: ``agent/codex_runtime.py`` derives
        ``auto_approve_exec`` from the canonical
        ``tools.approval.is_approval_bypass_active()`` (which reads
        ``approvals.mode`` via ``tools.approval._get_approval_mode``),
        and ``self._approval_callback`` itself runs the shared approval
        gate (mode + ``approvals.timeout``) in ``tools/approval.py``.
        Keep it that way — do not re-read approval config here.
        """
        if self._routing.auto_approve_exec:
            if self._routing.guard_no_prompt_exec:
                from tools.approval import check_no_prompt_command_guard

                allowed, reason = check_no_prompt_command_guard(
                    str(params.get("command") or "")
                )
                if not allowed:
                    logger.warning(
                        "Codex no-prompt policy declined exec: %s",
                        reason or "unclassified guarded command",
                    )
                    # No human made this decision. ``decline`` is rendered by
                    # Codex as "rejected by user"; ``cancel`` accurately marks
                    # a client-side policy block.
                    return "cancel"
            return "accept"
        command = params.get("command") or ""
        # Codex's CommandExecutionRequestApprovalParams has cwd as Optional —
        # fall back to the session's cwd when codex doesn't include it so the
        # approval prompt is never empty (quirk #10 fix).
        cwd = params.get("cwd") or self._cwd or "<unknown>"
        reason = params.get("reason")
        description = f"Codex requests exec in {cwd}"
        if reason:
            description += f" — {reason}"
        if self._approval_callback is not None:
            try:
                choice = self._approval_callback(
                    command, description, allow_permanent=False
                )
                return _approval_choice_to_codex_decision(choice)
            except Exception:
                logger.exception("approval_callback raised on exec request")
                return "cancel"
        # No callback means no human saw the request. Cancel fail-closed; a
        # decline is reserved for an actual user denial.
        return "cancel"

    def _decide_apply_patch_approval(self, params: dict) -> str:
        """Decide a Codex apply_patch approval request.

        Protocol-level routing only; Hermes approval-mode/timeout
        resolution is delegated to ``tools/approval.py`` upstream — see
        the docstring on ``_decide_exec_approval``.
        """
        if self._routing.auto_approve_apply_patch:
            if self._routing.guard_no_prompt_file_changes:
                item_id = str(params.get("itemId") or "")
                pending = self._pending_file_changes.get(item_id)
                if pending is None:
                    logger.warning(
                        "Codex no-prompt policy declined file change without "
                        "inspectable item metadata"
                    )
                    return "cancel"
                if not pending.kinds or not pending.kinds.issubset({"add", "update"}):
                    logger.warning(
                        "Codex no-prompt policy declined guarded file change "
                        "kinds: %s",
                        ", ".join(sorted(pending.kinds)) or "unknown",
                    )
                    return "cancel"
            return "accept"
        if self._approval_callback is not None:
            # FileChangeRequestApprovalParams gives us reason + grantRoot.
            # The actual changeset lives on the corresponding fileChange
            # item which the projector has already cached for us — look it
            # up by item_id so the user sees what's actually changing.
            reason = params.get("reason")
            grant_root = params.get("grantRoot")
            item_id = params.get("itemId") or ""
            change_summary = self._lookup_pending_file_change(item_id)
            description_parts = []
            if reason:
                description_parts.append(reason)
            if change_summary:
                description_parts.append(change_summary)
            if grant_root:
                description_parts.append(f"grants write to {grant_root}")
            description = (
                "; ".join(description_parts)
                if description_parts
                else "Codex requests to apply a patch"
            )
            command_label = (
                f"apply_patch: {change_summary}" if change_summary
                else f"apply_patch: {reason}" if reason
                else "apply_patch"
            )
            try:
                choice = self._approval_callback(
                    command_label,
                    description,
                    allow_permanent=False,
                )
                return _approval_choice_to_codex_decision(choice)
            except Exception:
                logger.exception("approval_callback raised on apply_patch")
                return "cancel"
        return "cancel"

    def _track_pending_file_change(self, note: dict) -> None:
        """Maintain self._pending_file_changes from item/started + item/completed
        notifications. Lets the apply_patch approval prompt show what's
        actually changing — codex's approval params don't carry the data."""
        method = note.get("method", "")
        params = note.get("params") or {}
        item = params.get("item") or {}
        if item.get("type") != "fileChange":
            return
        item_id = item.get("id") or ""
        if not item_id:
            return
        if method == "item/started":
            changes = item.get("changes") or []
            if not changes:
                self._pending_file_changes[item_id] = _PendingFileChange(
                    summary="1 change pending",
                    kinds=frozenset(),
                )
                return
            kinds: dict[str, int] = {}
            paths: list[str] = []
            for ch in changes:
                if not isinstance(ch, dict):
                    continue
                raw_kind = ch.get("kind") or {}
                kind = (
                    raw_kind.get("type")
                    if isinstance(raw_kind, dict)
                    else str(raw_kind)
                ) or "update"
                kind = str(kind).lower()
                kinds[kind] = kinds.get(kind, 0) + 1
                p = ch.get("path") or ""
                if p:
                    paths.append(p)
            counts = ", ".join(f"{n} {k}" for k, n in sorted(kinds.items()))
            preview = ", ".join(paths[:3])
            if len(paths) > 3:
                preview += f", +{len(paths) - 3} more"
            self._pending_file_changes[item_id] = _PendingFileChange(
                summary=f"{counts}: {preview}" if preview else counts,
                kinds=frozenset(kinds),
            )
        elif method == "item/completed":
            self._pending_file_changes.pop(item_id, None)

    def _lookup_pending_file_change(self, item_id: str) -> Optional[str]:
        """Look up an in-progress fileChange item by id and summarize its
        changes for the approval prompt. Returns None when we don't have
        the item cached (e.g. approval arrived before item/started, or
        fileChange item content not tracked yet)."""
        if not item_id:
            return None
        cached = self._pending_file_changes.get(item_id)
        if not cached:
            return None
        return cached.summary


def _apply_token_usage_notification(result: TurnResult, note: dict) -> None:
    """Capture Codex app-server token usage updates for caller accounting.

    Codex does not put token usage on turn/completed. It emits a separate
    thread/tokenUsage/updated notification containing cumulative totals and
    the latest turn breakdown.
    """
    if not isinstance(note, dict) or note.get("method") != "thread/tokenUsage/updated":
        return
    params = note.get("params") or {}
    token_usage = params.get("tokenUsage") or {}
    if not isinstance(token_usage, dict):
        return
    last = token_usage.get("last")
    total = token_usage.get("total")
    if isinstance(last, dict):
        result.token_usage_last = dict(last)
    if isinstance(total, dict):
        result.token_usage_total = dict(total)
    window = token_usage.get("modelContextWindow")
    if isinstance(window, int) and window > 0:
        result.model_context_window = window


def _apply_compaction_notification(result: TurnResult, note: dict) -> None:
    """Capture Codex-native context compaction boundaries.

    Recent app-server builds expose compaction as a ContextCompaction item.
    Older builds also emit the deprecated thread/compacted notification. Both
    mean the underlying Codex thread history has been compacted.
    """
    if not isinstance(note, dict):
        return
    method = note.get("method") or ""
    params = note.get("params") or {}
    if not isinstance(params, dict):
        return

    if method == "thread/compacted":
        result.compacted = True
        result.thread_id = params.get("threadId") or result.thread_id
        result.turn_id = params.get("turnId") or result.turn_id
        return

    if method not in {"item/started", "item/completed"}:
        return

    item = params.get("item") or {}
    if not isinstance(item, dict) or item.get("type") != "contextCompaction":
        return

    result.compacted = True
    result.thread_id = params.get("threadId") or result.thread_id
    result.turn_id = params.get("turnId") or result.turn_id


def _approval_choice_to_codex_decision(choice: str) -> str:
    """Map Hermes approval choices onto codex's CommandExecutionApprovalDecision
    / FileChangeApprovalDecision wire values.

    Hermes returns 'once', 'session', 'always', or 'deny'.
    Codex expects 'accept', 'acceptForSession', 'decline', or 'cancel'
    (verified against codex-rs/app-server-protocol/src/protocol/v2/item.rs
    on codex 0.130.0).

    This mapping is Codex-protocol-semantic and intentionally lives here,
    NOT in tools/approval.py: the Hermes approval mode/timeout resolution
    and the choice itself come from the shared core (tools/approval.py);
    only the wire-value translation is local.
    """
    if choice in {"once",}:
        return "accept"
    if choice in {"session", "always"}:
        return "acceptForSession"
    if choice == "deny":
        return "decline"
    # Timeout, an unavailable approval channel, callback failure, and unknown
    # outcomes all mean no human denial occurred. ``cancel`` keeps the action
    # fail-closed without fabricating a rejection by the user.
    return "cancel"


def _has_turn_aborted_marker(text: str) -> bool:
    """Return True if `text` contains any of the raw markers codex uses
    to signal a turn was aborted without emitting `turn/completed`.

    Codex emits `<turn_aborted>` (and sometimes `<turn_aborted/>`) as raw
    text inside agentMessage items when an interrupt or upstream error
    tears the turn down before the normal completion path fires. Mirrors
    openclaw beta.8's terminal-marker fix so we don't burn the full turn
    deadline waiting for a turn/completed that never comes.
    """
    if not text:
        return False
    for marker in _TURN_ABORTED_MARKERS:
        if marker in text:
            return True
    return False


def _get_hermes_version() -> str:
    """Best-effort Hermes version string for codex's userAgent line."""
    try:
        from importlib.metadata import version

        return version("hermes-agent")
    except Exception:  # pragma: no cover
        return "0.0.0"
