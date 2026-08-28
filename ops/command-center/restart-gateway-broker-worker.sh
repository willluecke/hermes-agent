#!/usr/bin/env bash

# External half of the command-center gateway restart contract. This script is
# launched only by a sibling user-systemd transient unit, never as a gateway
# child. It drives Hermes' graceful SIGUSR1/exit-75 restart path, attempts a
# non-destructive start recovery if that command fails, and records exact
# postconditions for the next reattached chat turn to verify.

set -euo pipefail

SYSTEMCTL_BIN="${HERMES_SYSTEMCTL_BIN:-systemctl}"
CURL_BIN="${HERMES_CURL_BIN:-curl}"
TIMEOUT_BIN="${HERMES_TIMEOUT_BIN:-timeout}"
SLEEP_BIN="${HERMES_SLEEP_BIN:-sleep}"
HERMES_BIN="${HERMES_BIN:-/home/will/.local/bin/hermes}"
GATEWAY_SERVICE="${HERMES_GATEWAY_SERVICE:-hermes-gateway.service}"
SYNC_SERVICE="${HERMES_SYNC_SERVICE:-hermes-sync.service}"
GATEWAY_HEALTH="${HERMES_GATEWAY_HEALTH_URL:-http://127.0.0.1:8642/health}"
SYNC_HEALTH="${HERMES_SYNC_HEALTH_URL:-http://127.0.0.1:8643/health}"
RESTART_TIMEOUT="${HERMES_RESTART_BROKER_COMMAND_TIMEOUT:-4000}"
PROBE_SECONDS="${HERMES_RESTART_BROKER_PROBE_SECONDS:-90}"
PROC_ROOT="${HERMES_RESTART_BROKER_PROC_ROOT:-/proc}"
STATUS_FILE="${HERMES_RESTART_BROKER_STATUS:-/home/will/.hermes/runtime/gateway-restart-broker.json}"

mkdir -p "$(dirname "$STATUS_FILE")"
chmod 0700 "$(dirname "$STATUS_FILE")"

old_pid=0
new_pid=0
restart_rc=-1
recovery_attempted=false
gateway_state=unknown
sync_state=unknown
gateway_health=false
sync_health=false
old_pid_exited=false
detail=""

write_status() {
  local outcome="$1"
  local phase="$2"
  local temporary
  temporary="$(mktemp "${STATUS_FILE}.tmp.XXXXXX")"
  jq -n \
    --arg outcome "$outcome" \
    --arg phase "$phase" \
    --arg detail "$detail" \
    --arg gatewayService "$GATEWAY_SERVICE" \
    --arg syncService "$SYNC_SERVICE" \
    --argjson recordedAt "$(date +%s)" \
    --argjson oldPid "$old_pid" \
    --argjson newPid "$new_pid" \
    --argjson restartCommandExit "$restart_rc" \
    --argjson recoveryAttempted "$recovery_attempted" \
    --arg gatewayState "$gateway_state" \
    --arg syncState "$sync_state" \
    --argjson gatewayHealth "$gateway_health" \
    --argjson syncHealth "$sync_health" \
    --argjson oldPidExited "$old_pid_exited" \
    '{version:1, outcome:$outcome, phase:$phase, detail:$detail,
      recordedAt:$recordedAt, gatewayService:$gatewayService,
      syncService:$syncService, oldPid:$oldPid, newPid:$newPid,
      restartCommandExit:$restartCommandExit,
      recoveryAttempted:$recoveryAttempted, gatewayState:$gatewayState,
      syncState:$syncState, gatewayHealth:$gatewayHealth,
      syncHealth:$syncHealth, oldPidExited:$oldPidExited}' \
    >"$temporary"
  chmod 0600 "$temporary"
  mv "$temporary" "$STATUS_FILE"
}

read_main_pid() {
  local value
  value="$($SYSTEMCTL_BIN --user show "$GATEWAY_SERVICE" \
    --property=MainPID --value 2>/dev/null || true)"
  if [[ "$value" =~ ^[0-9]+$ ]]; then
    printf '%s\n' "$value"
  else
    printf '0\n'
  fi
}

old_pid="$(read_main_pid)"
detail="External broker started; waiting for Hermes' graceful restart contract."
write_status running restart

set +e
"$TIMEOUT_BIN" --foreground "$RESTART_TIMEOUT" \
  env -u _HERMES_GATEWAY "$HERMES_BIN" gateway restart
restart_rc=$?
set -e

if (( restart_rc != 0 )); then
  recovery_attempted=true
  "$SYSTEMCTL_BIN" --user reset-failed "$GATEWAY_SERVICE" >/dev/null 2>&1 || true
  "$SYSTEMCTL_BIN" --user start "$GATEWAY_SERVICE" >/dev/null 2>&1 || true
fi

deadline=$((SECONDS + PROBE_SECONDS))
while (( SECONDS <= deadline )); do
  new_pid="$(read_main_pid)"
  gateway_state="$($SYSTEMCTL_BIN --user is-active "$GATEWAY_SERVICE" 2>/dev/null || true)"
  sync_state="$($SYSTEMCTL_BIN --user is-active "$SYNC_SERVICE" 2>/dev/null || true)"

  if (( old_pid <= 0 )) || [[ ! -e "$PROC_ROOT/$old_pid" ]]; then
    old_pid_exited=true
  else
    old_pid_exited=false
  fi
  if "$CURL_BIN" --fail --silent --show-error "$GATEWAY_HEALTH" >/dev/null 2>&1; then
    gateway_health=true
  else
    gateway_health=false
  fi
  if "$CURL_BIN" --fail --silent --show-error "$SYNC_HEALTH" >/dev/null 2>&1; then
    sync_health=true
  else
    sync_health=false
  fi

  if [[ "$gateway_state" == "active" \
      && "$sync_state" == "active" \
      && "$gateway_health" == "true" \
      && "$sync_health" == "true" \
      && "$old_pid_exited" == "true" \
      && "$new_pid" =~ ^[1-9][0-9]*$ \
      && ( "$old_pid" == "0" || "$new_pid" != "$old_pid" ) ]]; then
    if (( restart_rc == 0 )); then
      detail="Old PID exited and a different healthy gateway passed gateway/sync probes."
      write_status success verified
    else
      detail="Restart command failed, but start recovery produced a different healthy gateway and both probes passed."
      write_status recovered verified
    fi
    exit 0
  fi
  "$SLEEP_BIN" 1
done

detail="Restart verification failed: old PID must exit, a different gateway PID must be active, and gateway/sync health probes must pass."
write_status failed verify
exit 4
