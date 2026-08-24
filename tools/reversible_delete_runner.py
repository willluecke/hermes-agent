"""Protected argv-level runner for ``rm``, ``unlink``, and ``rmdir``.

This file is copied into owner-only Hermes runtime storage before Codex starts.
It verifies the deletion module loaded by the dedicated interpreter, captures
every expanded operand, and only then replaces itself with the real OS binary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path


BLOCK_EXIT = 125


def _sha256(path: str | os.PathLike[str]) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _block(message: str) -> int:
    sys.stderr.write(f"Hermes reversible deletion blocked command: {message}\n")
    return BLOCK_EXIT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--executable", required=True)
    parser.add_argument("--real-executable", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--policy-json", required=True)
    parser.add_argument("--module-sha256", required=True)
    parser.add_argument("--constants-sha256", required=True)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    parsed = parser.parse_args(argv)
    command_args = parsed.args[1:] if parsed.args[:1] == ["--"] else parsed.args

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
        import hermes_constants
        from tools import reversible_deletion

        if _sha256(reversible_deletion.__file__) != parsed.module_sha256:
            return _block("the server deletion policy changed during this run")
        if _sha256(hermes_constants.__file__) != parsed.constants_sha256:
            return _block("the server recovery boundary changed during this run")
        policy = reversible_deletion.ReversibleDeletionPolicy.from_config(
            json.loads(parsed.policy_json)
        )
        result = reversible_deletion.capture_delete_argv(
            parsed.executable,
            command_args,
            cwd=os.getcwd(),
            workspace_root=parsed.workspace,
            project=parsed.project,
            run_id=parsed.run_id,
            operation_key=reversible_deletion.operation_key_for(
                parsed.run_id,
                parsed.executable,
                json.dumps(command_args, ensure_ascii=True),
                uuid.uuid4().hex,
            ),
            policy=policy,
        )
    except Exception as exc:
        return _block(str(exc))
    if not result.handled:
        return _block(result.reason or "capture did not complete")

    os.execv(
        parsed.real_executable,
        [parsed.real_executable, *command_args],
    )
    return BLOCK_EXIT


if __name__ == "__main__":
    raise SystemExit(main())
