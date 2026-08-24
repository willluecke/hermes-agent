"""Protected argv-level runner for ``rm``, ``unlink``, and ``rmdir``.

This file is copied into owner-only Hermes runtime storage before Codex starts.
It verifies the deletion module loaded by the dedicated interpreter, captures
every expanded operand, and only then replaces itself with the real OS binary.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys


BLOCK_EXIT = 125


def _block(message: str) -> int:
    sys.stderr.write(f"Hermes reversible deletion blocked command: {message}\n")
    return BLOCK_EXIT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--executable", required=True)
    parser.add_argument("--real-executable", required=True)
    parser.add_argument("--broker-host", required=True)
    parser.add_argument("--broker-port", required=True, type=int)
    parser.add_argument("--broker-token", required=True)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    parsed = parser.parse_args(argv)
    command_args = parsed.args[1:] if parsed.args[:1] == ["--"] else parsed.args

    try:
        request = json.dumps(
            {
                "token": parsed.broker_token,
                "executable": parsed.executable,
                "args": command_args,
                "cwd": os.getcwd(),
            },
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"
        with socket.create_connection(
            (parsed.broker_host, parsed.broker_port), timeout=30
        ) as connection:
            connection.sendall(request)
            response_file = connection.makefile("rb")
            raw_response = response_file.readline(256 * 1024 + 1)
        if not raw_response or len(raw_response) > 256 * 1024:
            return _block("the server capture broker returned no valid response")
        result = json.loads(raw_response.decode("utf-8"))
    except Exception as exc:
        return _block(f"the server capture broker failed: {exc}")
    if not result.get("handled"):
        return _block(result.get("reason") or "capture did not complete")

    os.execv(
        parsed.real_executable,
        [parsed.real_executable, *command_args],
    )
    return BLOCK_EXIT


if __name__ == "__main__":
    raise SystemExit(main())
