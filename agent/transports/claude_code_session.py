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

logger = logging.getLogger(__name__)


class ClaudeCodeError(RuntimeError):
    """Controlled Claude Code runtime failure."""


@dataclass
class ClaudeCodeTurnResult:
    final_text: str = ""
    session_id: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    tool_iterations: int = 0
    interrupted: bool = False
    error: Optional[str] = None
    should_retire: bool = False
    session_confirmed: bool = False


def find_claude_binary() -> str:
    configured = os.environ.get("CLAUDE_CODE_BIN", "").strip()
    candidate = configured or shutil.which("claude") or ""
    if not candidate or not Path(candidate).is_file():
        raise ClaudeCodeError(
            "Claude Code is not installed or is not on PATH for the Hermes gateway"
        )
    return candidate


def claude_code_args(
    *,
    model: str,
    session_id: str,
    resume: bool = False,
    effort: Optional[str] = None,
    system_prompt: str = "",
    additional_dirs: Optional[list[str]] = None,
) -> list[str]:
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
        "bypassPermissions",
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
    if effort:
        args.extend(["--effort", effort])
    args.extend(["--resume" if resume else "--session-id", session_id])
    if system_prompt.strip():
        args.extend(["--append-system-prompt", system_prompt.strip()])
    for directory in additional_dirs or []:
        args.extend(["--add-dir", directory])
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
        on_event: Optional[Callable[[dict[str, Any]], None]] = None,
        on_session_id: Optional[Callable[[str], None]] = None,
        inactivity_timeout: float = DEFAULT_INACTIVITY_TIMEOUT,
        absolute_timeout: float = DEFAULT_ABSOLUTE_TIMEOUT,
    ) -> None:
        self.cwd = cwd
        self.model = model
        self.session_id = str(session_id or uuid4())
        self.resume = bool(resume and session_id)
        self.effort = effort
        self.system_prompt = system_prompt
        self.additional_dirs = list(additional_dirs or [])
        self.on_event = on_event
        self.on_session_id = on_session_id
        self.inactivity_timeout = inactivity_timeout
        self.absolute_timeout = absolute_timeout
        self._interrupt = threading.Event()
        self._process: Optional[subprocess.Popen[str]] = None
        self._output_queue: queue.Queue[Optional[str]] = queue.Queue()
        self._stderr_tail: deque[str] = deque(maxlen=40)
        self._turn_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._closed = False
        self._confirmed = bool(resume and session_id)

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
    ) -> bool:
        """Whether a later turn can safely reuse this frozen CLI process."""
        return bool(
            not self._closed
            and str(Path(self.cwd).expanduser().resolve())
            == str(Path(cwd).expanduser().resolve())
            and self.model == model
            and self.effort == effort
            and self.system_prompt == system_prompt
        )

    def request_interrupt(self) -> None:
        self._interrupt.set()
        self._terminate_process(signal.SIGINT)

    def close(self) -> None:
        with self._lifecycle_lock:
            self._closed = True
            process = self._process
            if process is None:
                return
            self._terminate_process(signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._terminate_process(signal.SIGKILL)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            self._process = None

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

    def _start_process(self) -> subprocess.Popen[str]:
        if self._closed:
            raise ClaudeCodeError("Claude Code session is closed")
        binary = find_claude_binary()
        args = claude_code_args(
            model=self.model,
            session_id=self.session_id,
            resume=bool(self.resume or self._confirmed),
            effort=self.effort,
            system_prompt=self.system_prompt,
            additional_dirs=self.additional_dirs,
        )
        env = os.environ.copy()
        # This route must use the signed-in Max subscription. An ambient key
        # would silently turn it into metered API usage.
        env.pop("ANTHROPIC_API_KEY", None)
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
        self._output_queue = queue.Queue()
        self._stderr_tail.clear()

        def _read_stdout() -> None:
            try:
                for line in process.stdout:
                    self._output_queue.put(line)
            finally:
                self._output_queue.put(None)

        def _read_stderr() -> None:
            for line in process.stderr:
                self._stderr_tail.append(line.rstrip())

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
            process = self._ensure_process()
            assert process.stdin is not None
            try:
                process.stdin.write(self._user_record(prompt) + "\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                self._process = None
                raise ClaudeCodeError("Claude Code input stream closed") from exc

            started_at = last_activity = time.monotonic()
            result = ClaudeCodeTurnResult(session_id=self.session_id)
            reported_session_ids: set[str] = set()

            def _confirm_session_id(value: Any) -> None:
                confirmed_id = str(value or "").strip()
                if not confirmed_id:
                    return
                self.session_id = confirmed_id
                self._confirmed = True
                self.resume = True
                result.session_id = confirmed_id
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

            while True:
                if self._interrupt.is_set():
                    result.interrupted = True
                    result.should_retire = True
                    break
                now = time.monotonic()
                if now - started_at > self.absolute_timeout:
                    result.error = "Claude Code exceeded the two-hour turn limit"
                    result.should_retire = True
                    break
                if now - last_activity > self.inactivity_timeout:
                    result.error = "Claude Code produced no activity for ten minutes"
                    result.should_retire = True
                    break
                try:
                    line = self._output_queue.get(timeout=0.5)
                except queue.Empty:
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
                _confirm_session_id(event.get("session_id"))
                if self.on_event is not None:
                    self.on_event(event)
                if event.get("type") == "user":
                    result.tool_iterations += sum(
                        1
                        for block in ((event.get("message") or {}).get("content") or [])
                        if isinstance(block, dict) and block.get("type") == "tool_result"
                    )
                if event.get("type") != "result":
                    continue
                result.final_text = str(event.get("result") or "").strip()
                _confirm_session_id(event.get("session_id"))
                result.usage = dict(event.get("usage") or {})
                if event.get("is_error"):
                    result.error = result.final_text or "Claude Code returned an error"
                break

            if result.should_retire:
                self.close()
            if not result.final_text and not result.error and not result.interrupted:
                result.error = "Claude Code completed without an authoritative final answer"
            return result
