#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR=${SOURCE_DIR:-/home/will/src/hermes-agent-migration/ops/command-center}
HERMES_HOME=${HERMES_HOME:-/home/will/.hermes}
HERMES_BIN=${HERMES_BIN:-/home/will/.local/bin/hermes}
JOB_NAME="Governed Agentic Loop Gate"
SCHEDULE="*/15 6-22 * * *"
MODEL="gpt-5.6-sol"
PROVIDER="openai-codex"
CHECKPOINT_NAMES=(
  "Daily Founder Revenue Dispatcher"
  "Daily Founder Evening Review"
)

install -d -m 0700 "$HERMES_HOME/scripts" "$HERMES_HOME/agentic-loop"
install -m 0700 \
  "$SOURCE_DIR/agentic-loop-gate.py" \
  "$HERMES_HOME/scripts/agentic-loop-gate.py"
install -m 0600 \
  "$SOURCE_DIR/agentic-loop-prompt.md" \
  "$HERMES_HOME/agentic-loop/prompt.md"

if [[ ! -f "$HERMES_HOME/agentic-loop/gate-state.json" ]]; then
  "$HERMES_HOME/scripts/agentic-loop-gate.py" prime >/dev/null
fi

jobs_file="$HERMES_HOME/cron/jobs.json"
job_id=""
if [[ -f "$jobs_file" ]]; then
  job_id=$(jq -r --arg name "$JOB_NAME" \
    '[(.jobs // .)[] | select(.name == $name) | .id] | if length == 1 then .[0] elif length == 0 then "" else error("duplicate governed agentic-loop jobs") end' \
    "$jobs_file")
fi

prompt=$(<"$HERMES_HOME/agentic-loop/prompt.md")
if [[ -n "$job_id" ]]; then
  "$HERMES_BIN" cron edit "$job_id" \
    --name "$JOB_NAME" \
    --schedule "$SCHEDULE" \
    --prompt "$prompt" \
    --script agentic-loop-gate.py \
    --workdir /home/will \
    --provider "$PROVIDER" \
    --model "$MODEL" \
    --reasoning-effort xhigh \
    --deliver local
  "$HERMES_BIN" cron resume "$job_id" >/dev/null
else
  output=$(
    "$HERMES_BIN" cron create "$SCHEDULE" "$prompt" \
      --name "$JOB_NAME" \
      --script agentic-loop-gate.py \
      --workdir /home/will \
      --provider "$PROVIDER" \
      --model "$MODEL" \
      --reasoning-effort xhigh \
      --deliver local
  )
  printf '%s\n' "$output"
  job_id=$(printf '%s\n' "$output" | sed -n 's/^Created job: //p' | head -1)
fi

if [[ -z "$job_id" ]]; then
  printf '%s\n' "Could not resolve the governed agentic-loop job ID" >&2
  exit 1
fi

printf '%s\n' "$job_id" >"$HERMES_HOME/agentic-loop/job-id"
chmod 0600 "$HERMES_HOME/agentic-loop/job-id"
printf 'Governed agentic loop installed: %s (%s)\n' "$job_id" "$SCHEDULE"

for checkpoint_name in "${CHECKPOINT_NAMES[@]}"; do
  checkpoint_id=""
  if [[ -f "$jobs_file" ]]; then
    checkpoint_id=$(jq -r --arg name "$checkpoint_name" \
      '[(.jobs // .)[] | select(.name == $name) | .id] | if length == 1 then .[0] elif length == 0 then "" else error("duplicate named checkpoints") end' \
      "$jobs_file")
  fi
  if [[ -z "$checkpoint_id" ]]; then
    printf 'Named checkpoint not present; skipped: %s\n' "$checkpoint_name"
    continue
  fi
  "$HERMES_BIN" cron edit "$checkpoint_id" \
    --provider "$PROVIDER" \
    --model "$MODEL" \
    --reasoning-effort xhigh >/dev/null
  printf 'Pinned named checkpoint: %s (%s)\n' "$checkpoint_name" "$checkpoint_id"
done
