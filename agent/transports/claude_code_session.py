"""One Hermes turn executed through the signed-in Claude Code CLI.

Unlike the dashboard's native Claude harness, this adapter is owned by an
``AIAgent`` run. Hermes retains the run lifecycle, transcript, streaming event
surface, project selection, and MCP tools while Claude Code supplies the
subscription-authenticated model/tool loop.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Optional
from uuid import uuid4


DEFAULT_INACTIVITY_TIMEOUT = 10 * 60.0
DEFAULT_ABSOLUTE_TIMEOUT = 2 * 60 * 60.0
DEFAULT_RESIDENT_FIRST_EVENT_TIMEOUT = 30.0
DEFAULT_STARTUP_FIRST_EVENT_TIMEOUT = 60.0
_DEBUG_LOG_RETENTION_SECONDS = 7 * 24 * 60 * 60

_ASYNC_AGENT_LAUNCH_MARKER = "async agent launched successfully"
_TASK_NOTIFICATION_TOOL_USE_ID_RE = re.compile(
    r"<tool-use-id>\s*([^<]+?)\s*</tool-use-id>", re.IGNORECASE
)
_TASK_NOTIFICATION_STATUS_RE = re.compile(
    r"<status>\s*([^<]+?)\s*</status>", re.IGNORECASE
)
_TERMINAL_TASK_STATUSES = frozenset(
    {"completed", "failed", "stopped", "cancelled", "canceled"}
)

CLAUDE_AUTH_ERROR_CODE = "claude_authentication_failed"
CLAUDE_AUTH_REMEDIATION = (
    "Run `claude auth login --claudeai` on command-center, then retry."
)

logger = logging.getLogger(__name__)


class ClaudeCodeError(RuntimeError):
    """Controlled Claude Code runtime failure."""

    def __init__(self, message: str, *, error_code: Optional[str] = None) -> None:
        super().__init__(message)
        self.error_code = error_code


@dataclass
class ClaudeCodeTurnResult:
    final_text: str = ""
    session_id: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    tool_iterations: int = 0
    interrupted: bool = False
    error: Optional[str] = None
    error_code: Optional[str] = None
    should_retire: bool = False
    session_confirmed: bool = False
    prompt_acknowledged: bool = False
    watchdog_retries: int = 0


def find_claude_binary() -> str:
    configured = os.environ.get("CLAUDE_CODE_BIN", "").strip()
    candidate = configured or shutil.which("claude") or ""
    if not candidate or not Path(candidate).is_file():
        raise ClaudeCodeError(
            "Claude Code is not installed or is not on PATH for the Hermes gateway"
        )
    return candidate


def claude_subscription_env() -> dict[str, str]:
    """Keep native OAuth, but never inherit an API or third-party route."""
    env = os.environ.copy()
    for key in (
        "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_TOKEN",
        "ANTHROPIC_BASE_URL", "ANTHROPIC_CUSTOM_HEADERS",
        "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_FOUNDRY", "CLAUDE_CODE_SIMPLE",
    ):
        env.pop(key, None)
    return env


def _current_claude_auth_generation() -> str:
    explicit = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "").strip()
    if explicit:
        from agent.claude_auth_lease import claude_auth_generation

        return claude_auth_generation(explicit)
    try:
        from agent.claude_auth_lease import acquire_claude_auth_lease

        return acquire_claude_auth_lease(0).generation
    except Exception:
        logger.debug("Could not fingerprint Claude Code credentials", exc_info=True)
        return ""


def claude_subscription_auth_available(
    *, min_validity_seconds: float = 0
) -> bool:
    """Whether Claude Code has the OAuth material required by this route.

    The Claude subscription transport deliberately strips ambient Anthropic
    API credentials before spawning the CLI.  Prefer the cheap credential
    sources Hermes understands, then ask Claude itself.  The native fallback
    matters because newer Claude Code builds can keep a valid login in a
    platform credential store that is not mirrored to
    ``~/.claude/.credentials.json``.  After a gateway restart there is no
    resident Claude process to bridge that gap, so rejecting the login here
    would incorrectly force the user through OAuth again.

    A native status is accepted only when it explicitly identifies a signed-in
    claude.ai Max subscription.  Anthropic API variables are removed from the
    probe environment so an API key cannot make this subscription-only route
    pass its preflight.
    """
    if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "").strip():
        return True
    try:
        from agent.claude_auth_lease import acquire_claude_auth_lease

        lease = acquire_claude_auth_lease(min_validity_seconds)
    except Exception:
        logger.warning("Claude Code subscription auth preflight failed", exc_info=True)
        lease = None
    if lease and lease.available:
        return True
    if lease and lease.credentials_found:
        return False

    try:
        status = subprocess.run(
            [find_claude_binary(), "auth", "status", "--json"],
            env=claude_subscription_env(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if status.returncode != 0:
            return False
        payload = json.loads(status.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        logger.debug("Claude Code native auth preflight failed", exc_info=True)
        return False
    return bool(
        isinstance(payload, dict)
        and payload.get("loggedIn") is True
        and str(payload.get("authMethod") or "").casefold() == "claude.ai"
        and str(payload.get("subscriptionType") or "").casefold() == "max"
    )


def _auth_failure_detail(event: dict[str, Any]) -> Optional[str]:
    """Extract an authoritative auth failure from Claude stream-json."""
    raw_error = str(event.get("error") or "").strip().lower()
    message = event.get("message") if isinstance(event.get("message"), dict) else {}
    text = "\n".join(
        str(block.get("text") or "").strip()
        for block in (message.get("content") or [])
        if isinstance(block, dict)
        and block.get("type") == "text"
        and str(block.get("text") or "").strip()
    )
    result_text = str(event.get("result") or "").strip()
    detail = text or result_text
    haystack = f"{raw_error} {detail}".lower()
    if raw_error == "authentication_failed" or (
        bool(event.get("isApiErrorMessage") or event.get("is_error"))
        and any(
            marker in haystack
            for marker in (
                "failed to authenticate",
                "authentication failed",
                "oauth session expired",
            )
        )
    ):
        return detail or "Claude Max authentication failed"
    return None


def _merge_usage(
    accumulated: dict[str, Any], current: dict[str, Any]
) -> dict[str, Any]:
    """Combine usage from multiple autonomous Claude result boundaries."""
    if not accumulated:
        return dict(current)
    merged = dict(accumulated)
    for key, value in current.items():
        previous = merged.get(key)
        if (
            isinstance(previous, (int, float))
            and not isinstance(previous, bool)
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            merged[key] = previous + value
        elif isinstance(previous, dict) and isinstance(value, dict):
            merged[key] = _merge_usage(previous, value)
        elif isinstance(previous, list) and isinstance(value, list):
            merged[key] = [*previous, *value]
        else:
            merged[key] = value
    return merged


def _content_text(content: Any) -> str:
    """Flatten Claude message content without interpreting tool metadata."""
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


def _is_prompt_replay(event: dict[str, Any], prompt: str) -> bool:
    """Whether ``event`` is Claude's echo of this exact submitted turn."""
    if event.get("type") != "user":
        return False
    message = event.get("message")
    if not isinstance(message, dict):
        return False
    return _content_text(message.get("content")) == prompt.strip()


def _claude_transcript_root() -> Path:
    configured = os.environ.get("CLAUDE_CONFIG_DIR", "").strip()
    return Path(configured).expanduser() if configured else Path.home() / ".claude"


@dataclass
class _ClaudeTranscriptPromptProbe:
    """Detect this turn's durable Claude transcript append.

    Claude persists streaming-input user records before it necessarily replays
    them on stdout.  The first-event watchdog may trust that append as proof the
    runtime accepted the turn, but output forwarding must still wait for the
    stdout replay so queued autonomous frames cannot claim a later prompt.
    """

    session_id: str
    prompt: str
    offsets: dict[Path, int]

    @classmethod
    def begin(cls, session_id: str, prompt: str) -> "_ClaudeTranscriptPromptProbe":
        probe = cls(session_id=session_id, prompt=prompt, offsets={})
        for path in probe._candidates():
            try:
                probe.offsets[path] = path.stat().st_size
            except OSError:
                continue
        return probe

    def _candidates(self) -> list[Path]:
        projects_dir = _claude_transcript_root() / "projects"
        try:
            project_dirs = list(projects_dir.iterdir())
        except OSError:
            return []
        filename = f"{self.session_id}.jsonl"
        return [path / filename for path in project_dirs if path.is_dir()]

    def acknowledged(self) -> bool:
        for path in self._candidates():
            offset = self.offsets.setdefault(path, 0)
            try:
                size = path.stat().st_size
                if size < offset:
                    # A replacement/truncation is not evidence for this dispatch.
                    self.offsets[path] = size
                    continue
                with path.open("rb") as transcript:
                    transcript.seek(offset)
                    while True:
                        line = transcript.readline()
                        if not line:
                            break
                        if not line.endswith(b"\n"):
                            # Claude is still appending this record. Re-read it on
                            # the next poll rather than advancing past partial JSON.
                            break
                        next_offset = transcript.tell()
                        try:
                            event = json.loads(line)
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            self.offsets[path] = next_offset
                            continue
                        self.offsets[path] = next_offset
                        if isinstance(event, dict) and _is_prompt_replay(
                            event, self.prompt
                        ):
                            return True
            except OSError:
                continue
        return False


@dataclass
class _ClaudeTranscriptTaskCompletionProbe:
    """Read terminal task notifications Claude may omit from stream-json.

    A native background agent can finish while Claude is inside another model
    call.  Claude durably records that notification as a ``queued_command``
    attachment, but that attachment is not always replayed on stdout.  Track
    the transcript independently so an already-finished agent cannot leave the
    enclosing Hermes turn waiting forever or hide its final answer.
    """

    session_id: str
    offsets: dict[Path, int]

    @classmethod
    def begin(cls, session_id: str) -> "_ClaudeTranscriptTaskCompletionProbe":
        probe = cls(session_id=session_id, offsets={})
        for path in probe._candidates():
            try:
                probe.offsets[path] = path.stat().st_size
            except OSError:
                continue
        return probe

    def _candidates(self) -> list[Path]:
        projects_dir = _claude_transcript_root() / "projects"
        try:
            project_dirs = list(projects_dir.iterdir())
        except OSError:
            return []
        filename = f"{self.session_id}.jsonl"
        return [path / filename for path in project_dirs if path.is_dir()]

    def completed_agent_tool_ids(self) -> set[str]:
        completed: set[str] = set()
        for path in self._candidates():
            offset = self.offsets.setdefault(path, 0)
            try:
                size = path.stat().st_size
                if size < offset:
                    # Replaced/truncated transcripts do not prove completion.
                    self.offsets[path] = size
                    continue
                with path.open("rb") as transcript:
                    transcript.seek(offset)
                    while True:
                        line = transcript.readline()
                        if not line:
                            break
                        if not line.endswith(b"\n"):
                            # Re-read a record Claude is still appending.
                            break
                        next_offset = transcript.tell()
                        try:
                            event = json.loads(line)
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            self.offsets[path] = next_offset
                            continue
                        self.offsets[path] = next_offset
                        if isinstance(event, dict):
                            completed.update(_completed_agent_tool_ids(event))
            except OSError:
                continue
        return completed


def _is_async_agent_launch_result(block: dict[str, Any]) -> bool:
    return _ASYNC_AGENT_LAUNCH_MARKER in _content_text(
        block.get("content")
    ).casefold()


def _completed_agent_tool_ids(event: dict[str, Any]) -> set[str]:
    """Extract terminal native-Agent ids from Claude task notifications."""
    text_parts: list[str] = []
    direct_content = event.get("content")
    if isinstance(direct_content, str):
        text_parts.append(direct_content)
    message = event.get("message")
    if isinstance(message, dict):
        message_text = _content_text(message.get("content"))
        if message_text:
            text_parts.append(message_text)
    attachment = event.get("attachment")
    if isinstance(attachment, dict):
        attachment_prompt = attachment.get("prompt")
        if isinstance(attachment_prompt, str):
            text_parts.append(attachment_prompt)
    text = "\n".join(text_parts)
    if "<task-notification>" not in text.casefold():
        return set()
    status_match = _TASK_NOTIFICATION_STATUS_RE.search(text)
    if (
        status_match is None
        or status_match.group(1).strip().casefold() not in _TERMINAL_TASK_STATUSES
    ):
        return set()
    return {
        match.group(1).strip()
        for match in _TASK_NOTIFICATION_TOOL_USE_ID_RE.finditer(text)
        if match.group(1).strip()
    }


def claude_code_args(
    *,
    model: str,
    session_id: str,
    resume: bool = False,
    effort: Optional[str] = None,
    system_prompt: str = "",
    additional_dirs: Optional[list[str]] = None,
    read_only: bool = False,
    no_tools: bool = False,
    debug_file: Optional[str] = None,
) -> list[str]:
    read_only = read_only or no_tools
    repo_root = str(Path(__file__).resolve().parents[2])
    mcp_config = {
        "mcpServers": {
            "hermes-tools": {
                "command": sys.executable,
                "args": ["-m", "agent.transports.hermes_tools_mcp_server"],
                "env": {"PYTHONPATH": repo_root},
            }
        }
    }
    args = [
        "-p",
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--replay-user-messages",
        "--include-partial-messages",
        "--verbose",
        "--permission-mode",
        "plan" if read_only else "bypassPermissions",
        "--setting-sources",
        "user,project",
        "--mcp-config",
        json.dumps(mcp_config, separators=(",", ":")),
        "--disable-slash-commands",
        "--model",
        model,
        "--name",
        "Hermes Chat",
    ]
    if read_only:
        # Ignore user/project MCP registrations as well as the Hermes bridge.
        # Safe mode also disables hooks, plugins, agents, and other local
        # customizations that could perform side effects outside Claude's
        # normal tool permission path. Native Read/Glob/Grep remain available
        # in plan mode.
        args[args.index("--mcp-config") + 1] = '{"mcpServers":{}}'
        args.extend(["--strict-mcp-config", "--safe-mode"])
    if no_tools:
        # An advisor sees only the supplied context. Plan mode alone still
        # exposes file tools, so explicitly remove the entire built-in set.
        args[args.index("--setting-sources") + 1] = ""
        args.extend(["--tools", "", "--no-chrome", "--no-session-persistence"])
    if effort:
        args.extend(["--effort", effort])
    args.extend(["--resume" if resume else "--session-id", session_id])
    if system_prompt.strip():
        args.extend(["--append-system-prompt", system_prompt.strip()])
    for directory in additional_dirs or []:
        args.extend(["--add-dir", directory])
    if debug_file:
        args.extend(["--debug-file", debug_file])
    return args


class ClaudeCodeSession:
    """One durable Claude Code process serving sequential Hermes turns.

    Claude's streaming-input protocol keeps stdin open and emits one ``result``
    record per user message.  The process is the normal continuity path; the
    durable Claude session id is only used to recover after eviction or crash.
    """

    def __init__(
        self,
        *,
        cwd: str,
        model: str,
        session_id: Optional[str] = None,
        resume: bool = False,
        effort: Optional[str] = None,
        system_prompt: str = "",
        additional_dirs: Optional[list[str]] = None,
        read_only: bool = False,
        no_tools: bool = False,
        on_event: Optional[Callable[[dict[str, Any]], None]] = None,
        on_session_id: Optional[Callable[[str], None]] = None,
        on_watchdog_timeout: Optional[Callable[[dict[str, Any]], None]] = None,
        inactivity_timeout: float = DEFAULT_INACTIVITY_TIMEOUT,
        absolute_timeout: float = DEFAULT_ABSOLUTE_TIMEOUT,
        resident_first_event_timeout: Optional[float] = DEFAULT_RESIDENT_FIRST_EVENT_TIMEOUT,
        startup_first_event_timeout: Optional[float] = DEFAULT_STARTUP_FIRST_EVENT_TIMEOUT,
    ) -> None:
        self.cwd = cwd
        self.model = model
        self.session_id = str(session_id or uuid4())
        self.resume = bool(resume and session_id)
        self.effort = effort
        self.system_prompt = system_prompt
        self.additional_dirs = list(additional_dirs or [])
        self.no_tools = bool(no_tools)
        self.read_only = bool(read_only or no_tools)
        self.on_event = on_event
        self.on_session_id = on_session_id
        self.on_watchdog_timeout = on_watchdog_timeout
        self.inactivity_timeout = inactivity_timeout
        self.absolute_timeout = absolute_timeout
        self.resident_first_event_timeout = resident_first_event_timeout
        self.startup_first_event_timeout = startup_first_event_timeout
        self._interrupt = threading.Event()
        self._process: Optional[subprocess.Popen[str]] = None
        self._output_queue: queue.Queue[Optional[str]] = queue.Queue()
        self._stderr_tail: deque[str] = deque(maxlen=40)
        self._debug_file: Optional[str] = None
        self._turn_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._closed = False
        self._confirmed = bool(resume and session_id)
        self._auth_generation = ""

    @property
    def pid(self) -> Optional[int]:
        process = self._process
        return process.pid if process is not None and process.poll() is None else None

    def is_alive(self) -> bool:
        process = self._process
        return bool(process is not None and process.poll() is None and not self._closed)

    def compatible_with(
        self,
        *,
        cwd: str,
        model: str,
        effort: Optional[str],
        system_prompt: str,
        read_only: bool = False,
        no_tools: bool = False,
    ) -> bool:
        """Whether a later turn can safely reuse this frozen CLI process."""
        return bool(
            not self._closed
            and str(Path(self.cwd).expanduser().resolve())
            == str(Path(cwd).expanduser().resolve())
            and self.model == model
            and self.effort == effort
            and self.system_prompt == system_prompt
            and self.read_only == bool(read_only or no_tools)
            and self.no_tools == bool(no_tools)
        )

    def request_interrupt(self) -> None:
        self._interrupt.set()
        self._terminate_process(signal.SIGINT)

    def close(self) -> None:
        with self._lifecycle_lock:
            self._closed = True
            process = self._process
        if process is not None:
            self._retire_process(process)

    def _retire_process(
        self,
        process: subprocess.Popen[str],
        *,
        grace_seconds: float = 5.0,
    ) -> bool:
        """Stop and reap one CLI process without closing the durable session."""
        with self._lifecycle_lock:
            if self._process is not process:
                return True
            reaped = False
            self._terminate_process(signal.SIGTERM)
            try:
                process.wait(timeout=grace_seconds)
                reaped = True
            except subprocess.TimeoutExpired:
                self._terminate_process(signal.SIGKILL)
                try:
                    process.wait(timeout=grace_seconds)
                    reaped = True
                except subprocess.TimeoutExpired:
                    pass
            if reaped and self._process is process:
                self._process = None
            return reaped

    def _terminate_process(self, sig: signal.Signals) -> None:
        process = self._process
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, sig)
        except (ProcessLookupError, PermissionError):
            try:
                process.send_signal(sig)
            except ProcessLookupError:
                pass

    def _next_debug_file(self) -> Optional[str]:
        """Path for this process's Claude debug log, or None when disabled.

        A watchdog kill discards the CLI's stdout and stderr, so a stall before
        Claude's first event leaves no trace of what the CLI was doing.  Its
        own debug log is the only record.  Keep one per process under the
        Hermes logs directory and prune old ones so the directory stays bounded.
        """
        if os.environ.get("HERMES_CLAUDE_CODE_DEBUG", "1").strip().lower() in {"0", "false", "no"}:
            return None
        configured = os.environ.get("HERMES_CLAUDE_CODE_DEBUG_DIR", "").strip()
        directory = (
            Path(configured).expanduser()
            if configured
            else Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser() / "logs" / "claude-code"
        )
        try:
            directory.mkdir(parents=True, exist_ok=True)
            cutoff = time.time() - _DEBUG_LOG_RETENTION_SECONDS
            for entry in directory.glob("*.log"):
                try:
                    if entry.stat().st_mtime < cutoff:
                        entry.unlink()
                except OSError:
                    continue
        except OSError:
            logger.debug("Claude Code debug log directory unavailable", exc_info=True)
            return None
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        self._debug_file = str(directory / f"{self.session_id or 'session'}-{stamp}.log")
        return self._debug_file

    def _start_process(self) -> subprocess.Popen[str]:
        if self._closed:
            raise ClaudeCodeError("Claude Code session is closed")
        binary = find_claude_binary()
        if not claude_subscription_auth_available(
            min_validity_seconds=self.absolute_timeout + 10 * 60
        ):
            raise ClaudeCodeError(
                f"Claude Max authentication is unavailable. {CLAUDE_AUTH_REMEDIATION}",
                error_code=CLAUDE_AUTH_ERROR_CODE,
            )
        args = claude_code_args(
            model=self.model,
            session_id=self.session_id,
            resume=bool(self.resume or self._confirmed),
            effort=self.effort,
            system_prompt=self.system_prompt,
            additional_dirs=self.additional_dirs,
            read_only=self.read_only,
            no_tools=self.no_tools,
            debug_file=self._next_debug_file(),
        )
        # This route must use the signed-in Max subscription. An ambient key
        # would silently turn it into metered API usage.
        env = claude_subscription_env()
        process = subprocess.Popen(
            [binary, *args],
            cwd=self.cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        self._process = process
        self._auth_generation = _current_claude_auth_generation()
        output_queue: queue.Queue[Optional[str]] = queue.Queue()
        stderr_tail: deque[str] = deque(maxlen=40)
        self._output_queue = output_queue
        self._stderr_tail = stderr_tail

        def _read_stdout() -> None:
            try:
                for line in process.stdout:
                    output_queue.put(line)
            finally:
                output_queue.put(None)

        def _read_stderr() -> None:
            for line in process.stderr:
                stderr_tail.append(line.rstrip())

        threading.Thread(
            target=_read_stdout,
            daemon=True,
            name=f"claude-stdout-{process.pid}",
        ).start()
        threading.Thread(
            target=_read_stderr,
            daemon=True,
            name=f"claude-stderr-{process.pid}",
        ).start()
        return process

    def _ensure_process(self) -> subprocess.Popen[str]:
        with self._lifecycle_lock:
            process = self._process
            if process is not None and process.poll() is None:
                return process
            self._process = None
            return self._start_process()

    @staticmethod
    def _user_record(prompt: str) -> str:
        if not isinstance(prompt, str):
            raise ClaudeCodeError("Claude Code prompts must be plain text")
        return json.dumps(
            {
                "type": "user",
                "message": {"role": "user", "content": prompt},
                "parent_tool_use_id": None,
                "session_id": "default",
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def run_turn(self, prompt: str) -> ClaudeCodeTurnResult:
        with self._turn_lock:
            self._interrupt.clear()
            resident_candidate = self._process if self.is_alive() else None
            if resident_candidate is not None and self._auth_generation:
                if not claude_subscription_auth_available(
                    min_validity_seconds=self.absolute_timeout + 10 * 60
                ):
                    raise ClaudeCodeError(
                        f"Claude Max authentication is unavailable. {CLAUDE_AUTH_REMEDIATION}",
                        error_code=CLAUDE_AUTH_ERROR_CODE,
                    )
                current_generation = _current_claude_auth_generation()
                if (
                    current_generation
                    and current_generation != self._auth_generation
                ):
                    # A different Hermes/Claude process rotated the shared OAuth
                    # pair.  Resume through a fresh CLI so this process never
                    # reaches expiry with its stale in-memory refresh token.
                    self._retire_process(resident_candidate)
                    resident_candidate = None
            process = self._ensure_process()
            resident_process = process is resident_candidate
            started_at = time.monotonic()
            result = ClaudeCodeTurnResult(session_id=self.session_id)
            reported_session_ids: set[str] = set()
            attempt = 0
            prompt_accepted = False
            stream_synchronized = False
            transcript_probe = _ClaudeTranscriptPromptProbe.begin(
                result.session_id, prompt
            )
            task_completion_probe = _ClaudeTranscriptTaskCompletionProbe.begin(
                result.session_id
            )
            scheduled_wakeup_tool_ids: set[str] = set()
            scheduled_wakeup_ready = False
            waiting_for_scheduled_wakeup = False
            agent_tool_ids: set[str] = set()
            pending_background_agent_tool_ids: set[str] = set()
            waiting_for_background_agents = False

            def _waiting_for_autonomous_work() -> bool:
                return bool(
                    waiting_for_scheduled_wakeup
                    or waiting_for_background_agents
                    or pending_background_agent_tool_ids
                )

            def _write_prompt(target: subprocess.Popen[str]) -> None:
                assert target.stdin is not None
                try:
                    target.stdin.write(self._user_record(prompt) + "\n")
                    target.stdin.flush()
                except (BrokenPipeError, OSError) as exc:
                    if self._process is target:
                        self._process = None
                    raise ClaudeCodeError("Claude Code input stream closed") from exc

            def _observe_session_id(value: Any) -> None:
                confirmed_id = str(value or "").strip()
                if not confirmed_id:
                    return
                self.session_id = confirmed_id
                self._confirmed = True
                self.resume = True
                result.session_id = confirmed_id

            def _publish_session_id() -> None:
                confirmed_id = str(result.session_id or self.session_id or "").strip()
                if not confirmed_id:
                    return
                result.session_confirmed = True
                if confirmed_id in reported_session_ids:
                    return
                reported_session_ids.add(confirmed_id)
                if self.on_session_id is not None:
                    try:
                        self.on_session_id(confirmed_id)
                    except Exception:
                        logger.warning(
                            "Claude Code session-id callback failed", exc_info=True
                        )

            def _notify_watchdog(*, timeout: float, retrying: bool) -> None:
                payload = {
                    "code": "claude_first_event_timeout",
                    "attempt": attempt + 1,
                    "retrying": retrying,
                    "timeout_seconds": timeout,
                    "session_id": result.session_id,
                }
                logger.warning(
                    "Claude Code first-event watchdog expired: "
                    "attempt=%d retrying=%s timeout_seconds=%.1f session=%s pid=%s "
                    "debug_file=%s stderr_tail=%s",
                    attempt + 1,
                    retrying,
                    timeout,
                    result.session_id,
                    process.pid,
                    self._debug_file,
                    (" | ".join(self._stderr_tail).strip() or "<empty>")[-2000:],
                )
                if self.on_watchdog_timeout is not None:
                    try:
                        self.on_watchdog_timeout(payload)
                    except Exception:
                        logger.warning(
                            "Claude Code watchdog callback failed", exc_info=True
                        )

            def _dispatch(
                target: subprocess.Popen[str], *, resident: bool
            ) -> tuple[float, float, Optional[float]]:
                nonlocal process
                nonlocal prompt_accepted, stream_synchronized, transcript_probe
                process = target
                prompt_accepted = False
                stream_synchronized = False
                transcript_probe = _ClaudeTranscriptPromptProbe.begin(
                    result.session_id, prompt
                )
                _write_prompt(target)
                dispatched_at = time.monotonic()
                timeout = (
                    self.resident_first_event_timeout
                    if resident
                    else self.startup_first_event_timeout
                )
                return dispatched_at, dispatched_at, timeout

            attempt_started_at, last_activity, first_event_timeout = _dispatch(
                process, resident=resident_process
            )

            while True:
                if self._interrupt.is_set():
                    result.interrupted = True
                    result.should_retire = True
                    break
                now = time.monotonic()
                if now - started_at > self.absolute_timeout:
                    result.error = f"Claude Code exceeded the {self.absolute_timeout:g}-second turn limit"
                    result.should_retire = True
                    break
                if not prompt_accepted and transcript_probe.acknowledged():
                    prompt_accepted = True
                    result.prompt_acknowledged = True
                    _publish_session_id()
                wait_timeout = 0.5
                if first_event_timeout is not None and not prompt_accepted:
                    wait_timeout = min(
                        wait_timeout,
                        max(0.01, first_event_timeout - (now - attempt_started_at)),
                    )
                if not _waiting_for_autonomous_work():
                    wait_timeout = min(
                        wait_timeout,
                        max(0.01, self.inactivity_timeout - (now - last_activity)),
                    )
                try:
                    line = self._output_queue.get(timeout=wait_timeout)
                except queue.Empty:
                    now = time.monotonic()
                    if not prompt_accepted and transcript_probe.acknowledged():
                        prompt_accepted = True
                        result.prompt_acknowledged = True
                        _publish_session_id()
                    if (
                        first_event_timeout is not None
                        and not prompt_accepted
                        and now - attempt_started_at >= first_event_timeout
                    ):
                        retrying = attempt == 0
                        _notify_watchdog(
                            timeout=first_event_timeout,
                            retrying=retrying,
                        )
                        if not retrying:
                            result.error_code = "claude_first_event_timeout"
                            result.error = (
                                "Claude Code did not acknowledge the turn after "
                                "the runtime was reset"
                            )
                            result.should_retire = True
                            break
                        reaped = self._retire_process(
                            process, grace_seconds=2.0
                        )
                        if not reaped:
                            result.error_code = "claude_runtime_reap_timeout"
                            result.error = (
                                "Claude Code did not acknowledge the turn and its "
                                "stalled runtime could not be stopped safely"
                            )
                            result.should_retire = True
                            break
                        if self._interrupt.is_set():
                            result.interrupted = True
                            result.should_retire = True
                            break
                        attempt += 1
                        result.watchdog_retries += 1
                        process = self._start_process()
                        attempt_started_at, last_activity, first_event_timeout = _dispatch(
                            process, resident=False
                        )
                        continue
                    if (
                        not _waiting_for_autonomous_work()
                        and now - last_activity >= self.inactivity_timeout
                    ):
                        result.error = "Claude Code produced no activity for ten minutes"
                        result.should_retire = True
                        break
                    continue
                if line is None:
                    code = process.poll()
                    detail = "\n".join(self._stderr_tail).strip()
                    result.error = detail or f"Claude Code exited {code}"
                    result.should_retire = True
                    break
                last_activity = time.monotonic()
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue
                _observe_session_id(event.get("session_id"))
                event_type = event.get("type")
                auth_failure = _auth_failure_detail(event)
                if auth_failure:
                    result.prompt_acknowledged = True
                    result.error_code = CLAUDE_AUTH_ERROR_CODE
                    result.error = f"{auth_failure.rstrip('.')}. {CLAUDE_AUTH_REMEDIATION}"
                    result.should_retire = True
                    break
                if not stream_synchronized:
                    if not _is_prompt_replay(event, prompt):
                        logger.debug(
                            "Quarantining Claude output emitted before the current "
                            "prompt replay: session=%s type=%s",
                            result.session_id,
                            event_type,
                        )
                        continue
                    stream_synchronized = True
                    prompt_accepted = True
                    result.prompt_acknowledged = True
                    _publish_session_id()
                if self.on_event is not None:
                    self.on_event(event)
                message = (
                    event.get("message")
                    if isinstance(event.get("message"), dict)
                    else {}
                )
                blocks = message.get("content") or []
                if not isinstance(blocks, list):
                    blocks = []
                if event_type == "assistant":
                    for block in blocks:
                        if not isinstance(block, dict) or block.get("type") != "tool_use":
                            continue
                        tool_use_id = str(block.get("id") or "").strip()
                        tool_name = str(block.get("name") or "").casefold()
                        if tool_use_id and tool_name == "schedulewakeup":
                            scheduled_wakeup_tool_ids.add(tool_use_id)
                        elif tool_use_id and tool_name == "agent":
                            agent_tool_ids.add(tool_use_id)
                completed_agent_tool_ids = _completed_agent_tool_ids(event)
                completed_agent_tool_ids.update(
                    task_completion_probe.completed_agent_tool_ids()
                )
                if completed_agent_tool_ids:
                    pending_background_agent_tool_ids.difference_update(
                        completed_agent_tool_ids
                    )
                if event_type == "user":
                    for block in blocks:
                        if not isinstance(block, dict) or block.get("type") != "tool_result":
                            continue
                        result.tool_iterations += 1
                        tool_use_id = str(block.get("tool_use_id") or "").strip()
                        if tool_use_id in agent_tool_ids:
                            agent_tool_ids.discard(tool_use_id)
                            if (
                                not block.get("is_error")
                                and _is_async_agent_launch_result(block)
                            ):
                                pending_background_agent_tool_ids.add(tool_use_id)
                        if tool_use_id in scheduled_wakeup_tool_ids:
                            scheduled_wakeup_tool_ids.discard(tool_use_id)
                            if not block.get("is_error"):
                                scheduled_wakeup_ready = True
                if event_type != "result":
                    continue
                result.final_text = str(event.get("result") or "").strip()
                _observe_session_id(event.get("session_id"))
                _publish_session_id()
                result.usage = _merge_usage(
                    result.usage, dict(event.get("usage") or {})
                )
                if event.get("is_error"):
                    result.error = result.final_text or "Claude Code returned an error"
                if not result.error and pending_background_agent_tool_ids:
                    waiting_for_background_agents = True
                    logger.info(
                        "Claude Code has %d native background agent(s) outstanding; "
                        "keeping the current Hermes turn open: session=%s pid=%s",
                        len(pending_background_agent_tool_ids),
                        result.session_id,
                        process.pid,
                    )
                    result.final_text = ""
                    continue
                if waiting_for_background_agents:
                    if not result.final_text:
                        continue
                    waiting_for_background_agents = False
                if result.error or result.final_text:
                    break
                if scheduled_wakeup_ready or waiting_for_scheduled_wakeup:
                    if not waiting_for_scheduled_wakeup:
                        logger.info(
                            "Claude Code scheduled an autonomous continuation; "
                            "keeping the current Hermes turn open: session=%s pid=%s",
                            result.session_id,
                            process.pid,
                        )
                    scheduled_wakeup_ready = False
                    waiting_for_scheduled_wakeup = True
                    continue
                break

            if not result.final_text and not result.error and not result.interrupted:
                result.error = "Claude Code completed without an authoritative final answer"
                result.should_retire = True
            if result.should_retire:
                self.close()
            return result
