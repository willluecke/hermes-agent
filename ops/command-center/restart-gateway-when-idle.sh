#!/usr/bin/env bash

# Drain command-center, prove the persisted active-work count is stably zero,
# and only then restart the user gateway service. A timeout never restarts.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERMES_PYTHON="${HERMES_PYTHON:-/home/will/.hermes/venvs/hermes-command-center/bin/python}"
WAIT_SECONDS="${HERMES_RESTART_IDLE_TIMEOUT:-3600}"
SERVICE="${HERMES_GATEWAY_SERVICE:-hermes-gateway.service}"

if [[ ! -x "$HERMES_PYTHON" ]]; then
  echo "Hermes Python is not executable: $HERMES_PYTHON" >&2
  exit 1
fi

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

clear_drain() {
  "$HERMES_PYTHON" - <<'PY' >/dev/null 2>&1 || true
from gateway.drain_control import clear_drain_request
clear_drain_request()
PY
}
trap clear_drain EXIT INT TERM

"$HERMES_PYTHON" - <<'PY'
from gateway.drain_control import write_drain_request
write_drain_request(principal="command-center-safe-restart")
PY

deadline=$((SECONDS + WAIT_SECONDS))
zero_samples=0
while (( SECONDS < deadline )); do
  read -r gateway_state active_agents < <(
    "$HERMES_PYTHON" - <<'PY'
from gateway.status import parse_active_agents, read_runtime_status
state = read_runtime_status() or {}
print(state.get("gateway_state") or "unknown", parse_active_agents(state.get("active_agents", 0)))
PY
  )

  if [[ "$gateway_state" == "draining" && "$active_agents" == "0" ]]; then
    zero_samples=$((zero_samples + 1))
    if (( zero_samples >= 2 )); then
      break
    fi
  else
    zero_samples=0
  fi
  sleep 1
done

if (( zero_samples < 2 )); then
  echo "Gateway did not become stably idle within ${WAIT_SECONDS}s; restart cancelled." >&2
  exit 2
fi

if [[ "$(systemctl --user is-active "$SERVICE")" != "active" ]]; then
  echo "Gateway service is not active; restart cancelled." >&2
  exit 3
fi

systemctl --user restart "$SERVICE"
clear_drain
trap - EXIT INT TERM

for _ in $(seq 1 60); do
  if [[ "$(systemctl --user is-active "$SERVICE" 2>/dev/null || true)" == "active" ]] \
    && curl --fail --silent --show-error http://127.0.0.1:8642/health >/dev/null; then
    echo "Gateway restarted after a stable idle drain."
    exit 0
  fi
  sleep 1
done

echo "Gateway restart was issued, but health did not recover within 60s." >&2
exit 4
