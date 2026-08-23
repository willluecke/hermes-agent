#!/usr/bin/env bash
set -u

CODEX_BIN=${CODEX_BIN:-/home/will/.local/bin/codex}
CLAUDE_BIN=${CLAUDE_BIN:-/home/will/.local/bin/claude}
failures=()

codex_status=$(timeout 20 "$CODEX_BIN" login status 2>&1)
codex_rc=$?
if [[ $codex_rc -ne 0 ]] || ! grep -Fq "Logged in using ChatGPT" <<<"$codex_status"; then
  failures+=("Codex ChatGPT login is unavailable; run 'codex login' on command-center.")
fi

claude_status=$(timeout 20 "$CLAUDE_BIN" auth status --json 2>&1)
claude_rc=$?
if [[ $claude_rc -ne 0 ]] || ! jq -e \
  '.loggedIn == true and .authMethod == "claude.ai" and .subscriptionType == "max"' \
  >/dev/null 2>&1 <<<"$claude_status"; then
  failures+=("Claude Max login is unavailable; re-authenticate Claude Code on command-center.")
fi

if ! systemctl --user is-active --quiet hermes-subscription-worker.service; then
  failures+=("The command-center subscription worker is not active.")
fi

if [[ ${#failures[@]} -eq 0 ]]; then
  printf '%s\n' '[SILENT]'
  exit 0
fi

printf '%s\n' 'Subscription auth health check requires action:'
for failure in "${failures[@]}"; do
  printf -- '- %s\n' "$failure"
done

