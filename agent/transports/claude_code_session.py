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
        "--output-format",
        "stream-json",
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
        "--effort",
        "high",
        "--session-id",
        session_id,
        "--name",
        "Hermes Chat",
    ]
    if system_prompt.strip():
        args.extend(["--append-system-prompt", system_prompt.strip()])
    for directory in additional_dirs or []:
        args.extend(["--add-dir", directory])
    return args


class ClaudeCodeSession:
    """Blocking, interruptible wrapper around one Claude Code print turn."""

    def __init__(
        self,
        *,
        cwd: str,
        model: str,
        system_prompt: str = "",
        additional_dirs: Optional[list[str]] = None,
        on_event: Optional[Callable[[dict[str, Any]], None]] = None,
        inactivity_timeout: float = DEFAULT_INACTIVITY_TIMEOUT,
        absolute_timeout: float = DEFAULT_ABSOLUTE_TIMEOUT,
    ) -> None:
        self.cwd = cwd
        self.model = model
        self.system_prompt = system_prompt
        self.additional_dirs = list(additional_dirs or [])
        self.on_event = on_event
        self.inactivity_timeout = inactivity_timeout
        self.absolute_timeout = absolute_timeout
        self._interrupt = threading.Event()
        self._process: Optional[subprocess.Popen[str]] = None

    def request_interrupt(self) -> None:
        self._interrupt.set()
        self._terminate_process(signal.SIGINT)

    def close(self) -> None:
        self._terminate_process(signal.SIGTERM)

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

    def run_turn(self, prompt: str) -> ClaudeCodeTurnResult:
        binary = find_claude_binary()
        session_id = str(uuid4())
        args = claude_code_args(
            model=self.model,
            session_id=session_id,
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
        self._process = process
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        process.stdin.write(prompt)
        process.stdin.close()

        output_queue: queue.Queue[Optional[str]] = queue.Queue()
        stderr_tail: deque[str] = deque(maxlen=40)

        def _read_stdout() -> None:
            try:
                for line in process.stdout:
                    output_queue.put(line)
            finally:
                output_queue.put(None)

        def _read_stderr() -> None:
            for line in process.stderr:
                stderr_tail.append(line.rstrip())

        threading.Thread(target=_read_stdout, daemon=True).start()
        threading.Thread(target=_read_stderr, daemon=True).start()

        started_at = last_activity = time.monotonic()
        result = ClaudeCodeTurnResult(session_id=session_id)
        stdout_closed = False
        timed_out = False
        try:
            while not stdout_closed:
                if self._interrupt.is_set():
                    result.interrupted = True
                    self._terminate_process(signal.SIGINT)
                now = time.monotonic()
                if now - started_at > self.absolute_timeout:
                    timed_out = True
                    result.error = "Claude Code exceeded the two-hour turn limit"
                    break
                if now - last_activity > self.inactivity_timeout:
                    timed_out = True
                    result.error = "Claude Code produced no activity for ten minutes"
                    break
                try:
                    line = output_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                if line is None:
                    stdout_closed = True
                    continue
                last_activity = time.monotonic()
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue
                self.on_event and self.on_event(event)
                if event.get("type") == "user":
                    result.tool_iterations += sum(
                        1
                        for block in ((event.get("message") or {}).get("content") or [])
                        if isinstance(block, dict) and block.get("type") == "tool_result"
                    )
                if event.get("type") == "result":
                    result.final_text = str(event.get("result") or "").strip()
                    result.session_id = str(event.get("session_id") or session_id)
                    result.usage = dict(event.get("usage") or {})
                    if event.get("is_error"):
                        result.error = result.final_text or "Claude Code returned an error"
        finally:
            if timed_out:
                result.should_retire = True
                self._terminate_process(signal.SIGTERM)
            try:
                code = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._terminate_process(signal.SIGKILL)
                code = process.wait(timeout=5)
            self._process = None

        if result.interrupted:
            return result
        if code != 0 and not result.error:
            result.error = "\n".join(stderr_tail).strip() or f"Claude Code exited {code}"
        if not result.final_text and not result.error:
            result.error = "Claude Code completed without an authoritative final answer"
        return result
