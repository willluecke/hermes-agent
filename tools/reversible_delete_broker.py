"""Narrow loopback broker for sandboxed reversible-deletion shims.

Codex tool commands run inside its filesystem sandbox, so a command shim must
not receive write access to Hermes recovery storage. This broker stays in the
gateway process, accepts only run-scoped removal argv over loopback, revalidates
every target against its server-owned scope, and performs the Trash capture.
It never removes or restores files.
"""

from __future__ import annotations

import json
import os
import secrets
import socketserver
import threading
import uuid
from dataclasses import dataclass
from typing import Any

from tools.reversible_deletion import (
    ReversibleDeletionPolicy,
    capture_delete_argv,
    operation_key_for,
)


_MAX_REQUEST_BYTES = 256 * 1024
_MAX_ARGUMENTS = 4096
_MAX_ARGUMENT_BYTES = 64 * 1024


@dataclass(frozen=True)
class ReversibleDeleteBrokerEndpoint:
    host: str
    port: int
    token: str


class _BrokerServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = False
    daemon_threads = True


class _BrokerHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        raw = self.rfile.readline(_MAX_REQUEST_BYTES + 1)
        if len(raw) > _MAX_REQUEST_BYTES:
            response = {"handled": False, "reason": "capture request is too large"}
        else:
            try:
                payload = json.loads(raw.decode("utf-8"))
                response = self.server.broker.capture(payload)  # type: ignore[attr-defined]
            except Exception:
                response = {
                    "handled": False,
                    "reason": "capture request is malformed",
                }
        self.wfile.write(
            json.dumps(response, separators=(",", ":")).encode("utf-8") + b"\n"
        )


class ReversibleDeleteBroker:
    """Run-scoped capture capability hosted outside the Codex sandbox."""

    def __init__(
        self,
        *,
        workspace_root: str,
        project: str,
        run_id: str,
        policy: ReversibleDeletionPolicy,
    ) -> None:
        self._workspace_root = os.path.realpath(os.path.abspath(workspace_root))
        self._project = str(project)
        self._run_id = str(run_id)
        self._policy = policy
        self._token = secrets.token_urlsafe(32)
        self._server = _BrokerServer(("127.0.0.1", 0), _BrokerHandler)
        self._server.broker = self  # type: ignore[attr-defined]
        host, port = self._server.server_address
        self.endpoint = ReversibleDeleteBrokerEndpoint(
            host=str(host),
            port=int(port),
            token=self._token,
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name=f"reversible-delete-{self._run_id[:12]}",
            daemon=True,
        )
        self._closed = False
        self._thread.start()

    def capture(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict) or not secrets.compare_digest(
            str(payload.get("token") or ""), self._token
        ):
            return {"handled": False, "reason": "capture capability is invalid"}

        executable = str(payload.get("executable") or "")
        raw_args = payload.get("args")
        cwd = str(payload.get("cwd") or "")
        if executable not in {"rm", "unlink", "rmdir"}:
            return {"handled": False, "reason": "unsupported removal primitive"}
        if not isinstance(raw_args, list) or len(raw_args) > _MAX_ARGUMENTS:
            return {"handled": False, "reason": "invalid removal argv"}
        args = [str(value) for value in raw_args]
        if any(len(value.encode("utf-8")) > _MAX_ARGUMENT_BYTES for value in args):
            return {"handled": False, "reason": "removal argument is too large"}
        canonical_cwd = os.path.realpath(os.path.abspath(cwd))
        try:
            if os.path.commonpath(
                [canonical_cwd, self._workspace_root]
            ) != self._workspace_root:
                return {
                    "handled": False,
                    "reason": "command cwd is outside the selected workspace",
                }
        except ValueError:
            return {
                "handled": False,
                "reason": "command cwd is outside the selected workspace",
            }

        result = capture_delete_argv(
            executable,
            args,
            cwd=canonical_cwd,
            workspace_root=self._workspace_root,
            project=self._project,
            run_id=self._run_id,
            operation_key=operation_key_for(
                self._run_id,
                executable,
                json.dumps(args, ensure_ascii=True),
                uuid.uuid4().hex,
            ),
            policy=self._policy,
        )
        return {
            "handled": result.handled,
            "reason": result.reason,
            "captured": len(result.items),
            "missing": len(result.missing_targets),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)

    def __enter__(self) -> "ReversibleDeleteBroker":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()
