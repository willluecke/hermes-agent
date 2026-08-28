#!/usr/bin/env bash

# Queue the command-center restart worker in a sibling user-systemd cgroup.
# This launcher is safe to invoke from a gateway-hosted terminal: it returns
# as soon as systemd has exec'd the worker, allowing the originating turn to
# finish before the external worker asks the gateway to drain and restart.

set -euo pipefail

SYSTEMCTL_BIN="${HERMES_SYSTEMCTL_BIN:-systemctl}"
SYSTEMD_RUN_BIN="${HERMES_SYSTEMD_RUN_BIN:-systemd-run}"
TIMEOUT_BIN="${HERMES_TIMEOUT_BIN:-timeout}"
UNIT="${HERMES_RESTART_BROKER_UNIT:-hermes-gateway-restart-broker}"
LAUNCH_TIMEOUT="${HERMES_RESTART_BROKER_LAUNCH_TIMEOUT:-15}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALLED_WORKER="/home/will/.local/libexec/hermes-command-center-restart-worker"
WORKER="${HERMES_RESTART_BROKER_WORKER:-$INSTALLED_WORKER}"

if [[ ! -x "$WORKER" && -x "$SCRIPT_DIR/restart-gateway-broker-worker.sh" ]]; then
  WORKER="$SCRIPT_DIR/restart-gateway-broker-worker.sh"
fi
if [[ ! -x "$WORKER" ]]; then
  echo "Restart broker worker is not executable: $WORKER" >&2
  exit 1
fi

unit_state="$($SYSTEMCTL_BIN --user show "$UNIT.service" \
  --property=ActiveState --value 2>/dev/null || true)"
if [[ "$unit_state" == "active" || "$unit_state" == "activating" || "$unit_state" == "reloading" ]]; then
  echo "Gateway restart already queued in $UNIT.service"
  exit 0
fi
if [[ "$unit_state" == "failed" ]]; then
  "$SYSTEMCTL_BIN" --user reset-failed "$UNIT.service" >/dev/null 2>&1 || true
fi

run_args=(
  --user
  --unit "$UNIT"
  --collect
  --property=Type=exec
  --description="Command-center external gateway restart broker"
)
for name in HERMES_HOME PATH PYTHONPATH VIRTUAL_ENV; do
  value="${!name:-}"
  if [[ -n "$value" ]]; then
    run_args+=("--setenv=$name=$value")
  fi
done

set +e
"$TIMEOUT_BIN" "$LAUNCH_TIMEOUT" "$SYSTEMD_RUN_BIN" "${run_args[@]}" "$WORKER"
launch_rc=$?
set -e
if (( launch_rc != 0 )); then
  unit_state="$($SYSTEMCTL_BIN --user show "$UNIT.service" \
    --property=ActiveState --value 2>/dev/null || true)"
  if [[ "$unit_state" == "active" || "$unit_state" == "activating" || "$unit_state" == "reloading" ]]; then
    echo "Gateway restart queued in $UNIT.service"
    exit 0
  fi
  if (( launch_rc == 124 )); then
    echo "Restart broker launch timed out before systemd accepted the unit." >&2
  else
    echo "Restart broker launch failed with exit code $launch_rc." >&2
  fi
  exit "$launch_rc"
fi

echo "Gateway restart queued in $UNIT.service; the external worker will restart after active work drains."
