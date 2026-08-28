from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "ops/command-center/restart-gateway-when-idle.sh"
WORKER = ROOT / "ops/command-center/restart-gateway-broker-worker.sh"


def _executable(path: Path, body: str) -> Path:
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return path


def test_launcher_hands_off_to_transient_unit(tmp_path):
    calls = tmp_path / "calls"
    worker = _executable(tmp_path / "worker", "exit 0\n")
    systemctl = _executable(
        tmp_path / "systemctl",
        'if [[ "$3" == "show" || "$2" == "show" ]]; then echo inactive; fi\n',
    )
    systemd_run = _executable(
        tmp_path / "systemd-run",
        f'printf "%s\\n" "$@" >"{calls}"\n',
    )

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        text=True,
        capture_output=True,
        timeout=5,
        env={
            **os.environ,
            "HERMES_SYSTEMCTL_BIN": str(systemctl),
            "HERMES_SYSTEMD_RUN_BIN": str(systemd_run),
            "HERMES_RESTART_BROKER_WORKER": str(worker),
        },
    )

    assert result.returncode == 0, result.stderr
    argv = calls.read_text(encoding="utf-8").splitlines()
    assert "--user" in argv
    assert "--property=Type=exec" in argv
    assert str(worker) == argv[-1]
    assert "external worker will restart after active work drains" in result.stdout


def test_launcher_timeout_is_fail_closed(tmp_path):
    worker = _executable(tmp_path / "worker", "exit 0\n")
    systemctl = _executable(tmp_path / "systemctl", "echo inactive\n")
    timeout = _executable(tmp_path / "timeout", "exit 124\n")

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        text=True,
        capture_output=True,
        timeout=5,
        env={
            **os.environ,
            "HERMES_SYSTEMCTL_BIN": str(systemctl),
            "HERMES_TIMEOUT_BIN": str(timeout),
            "HERMES_SYSTEMD_RUN_BIN": str(tmp_path / "unused"),
            "HERMES_RESTART_BROKER_WORKER": str(worker),
        },
    )

    assert result.returncode == 124
    assert "timed out" in result.stderr


def _worker_fixture(tmp_path: Path, *, restart_exit: int, recovery_changes_pid: bool):
    state = tmp_path / "pid"
    state.write_text("111\n", encoding="utf-8")
    proc_root = tmp_path / "proc"
    (proc_root / "111").mkdir(parents=True)
    status_file = tmp_path / "runtime/status.json"

    hermes = _executable(
        tmp_path / "hermes",
        (
            f'printf "222\\n" >"{state}"\n'
            f'mv "{proc_root / "111"}" "{proc_root / "old-111"}"\n'
            if restart_exit == 0
            else f"exit {restart_exit}\n"
        ),
    )
    recovery_body = ":"
    if recovery_changes_pid:
        recovery_body = (
            f'printf "222\\n" >"{state}"; '
            f'mv "{proc_root / "111"}" "{proc_root / "old-111"}"'
        )
    systemctl = _executable(
        tmp_path / "systemctl",
        f'''
args="$*"
if [[ "$args" == *"show hermes-gateway.service"* ]]; then
  cat "{state}"
elif [[ "$args" == *"is-active"* ]]; then
  echo active
elif [[ "$args" == *" start hermes-gateway.service"* ]]; then
  {recovery_body}
fi
''',
    )
    curl = _executable(tmp_path / "curl", "exit 0\n")
    timeout = _executable(
        tmp_path / "timeout",
        'if [[ "${1:-}" == "--foreground" ]]; then shift; fi\nshift\nexec "$@"\n',
    )
    env = {
        **os.environ,
        "HERMES_SYSTEMCTL_BIN": str(systemctl),
        "HERMES_CURL_BIN": str(curl),
        "HERMES_TIMEOUT_BIN": str(timeout),
        "HERMES_SLEEP_BIN": "/bin/true",
        "HERMES_BIN": str(hermes),
        "HERMES_RESTART_BROKER_STATUS": str(status_file),
        "HERMES_RESTART_BROKER_PROC_ROOT": str(proc_root),
        "HERMES_RESTART_BROKER_PROBE_SECONDS": "1",
    }
    return env, status_file


def test_worker_verifies_old_pid_replacement_and_health(tmp_path):
    env, status_file = _worker_fixture(
        tmp_path, restart_exit=0, recovery_changes_pid=False
    )

    result = subprocess.run(
        ["bash", str(WORKER)], text=True, capture_output=True, timeout=5, env=env
    )

    assert result.returncode == 0, result.stderr
    status = json.loads(status_file.read_text(encoding="utf-8"))
    assert status == {
        **status,
        "outcome": "success",
        "phase": "verified",
        "oldPid": 111,
        "newPid": 222,
        "restartCommandExit": 0,
        "recoveryAttempted": False,
        "gatewayState": "active",
        "syncState": "active",
        "gatewayHealth": True,
        "syncHealth": True,
        "oldPidExited": True,
    }


def test_worker_recovers_an_inactive_failed_restart(tmp_path):
    env, status_file = _worker_fixture(
        tmp_path, restart_exit=7, recovery_changes_pid=True
    )

    result = subprocess.run(
        ["bash", str(WORKER)], text=True, capture_output=True, timeout=5, env=env
    )

    assert result.returncode == 0, result.stderr
    status = json.loads(status_file.read_text(encoding="utf-8"))
    assert status["outcome"] == "recovered"
    assert status["restartCommandExit"] == 7
    assert status["recoveryAttempted"] is True
    assert status["oldPidExited"] is True
    assert status["newPid"] == 222


def test_worker_recovers_after_restart_command_timeout(tmp_path):
    env, status_file = _worker_fixture(
        tmp_path, restart_exit=124, recovery_changes_pid=True
    )

    result = subprocess.run(
        ["bash", str(WORKER)], text=True, capture_output=True, timeout=5, env=env
    )

    assert result.returncode == 0, result.stderr
    status = json.loads(status_file.read_text(encoding="utf-8"))
    assert status["outcome"] == "recovered"
    assert status["restartCommandExit"] == 124
    assert status["recoveryAttempted"] is True
    assert status["oldPidExited"] is True
    assert status["newPid"] == 222


def test_worker_reports_failed_restart_when_recovery_does_not_replace_pid(tmp_path):
    env, status_file = _worker_fixture(
        tmp_path, restart_exit=7, recovery_changes_pid=False
    )

    result = subprocess.run(
        ["bash", str(WORKER)], text=True, capture_output=True, timeout=5, env=env
    )

    assert result.returncode == 4
    status = json.loads(status_file.read_text(encoding="utf-8"))
    assert status["outcome"] == "failed"
    assert status["phase"] == "verify"
    assert status["oldPid"] == status["newPid"] == 111
    assert status["oldPidExited"] is False
