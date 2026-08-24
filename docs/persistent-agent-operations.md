# Persistent Agent Operations

> **Command-center runbook.** The canonical, reusable persistent-agent pattern
> lives in
> [Build a Process-Driven Persistent Agent](../website/docs/guides/process-driven-persistent-agent.md).

## Service Layout

Host: `command-center` (`will@192.168.1.243`)

- `hermes-gateway.service`: Hermes conversations and streamed run events
- `hermes-sync.service`: Hermes Chat synchronization
- `hermes-subscription-worker.service`: native Codex and Claude Code jobs
- `cloudflared-hermes.service`: public tunnel, when enabled
- `hermes-command-center-backup.timer`: daily critical-state backup
- `hermes-command-center-full-backup.timer`: weekly full-state backup

All agent services run as user `will`. User lingering must remain enabled.

The gateway and sync listeners are loopback-only:

```text
127.0.0.1:8642  Hermes API (Cloudflare: hermes-api.devsession.org)
127.0.0.1:8643  Hermes sync (Cloudflare: hermes-sync.devsession.org)
```

Do not change either listener to `0.0.0.0` for LAN convenience. SSH and the
authenticated Cloudflare routes are the supported remote access paths.

The process-driven agentic loop does not add another resident service. It uses
the existing Hermes cron ticker and a deterministic pre-check script. A
suppressed tick never constructs an agent.

## Required Configuration

`~/.hermes/config.yaml` must contain the following authority contract:

```yaml
model:
  provider: openai-codex
  default: gpt-5.6-sol
  openai_runtime: codex_app_server
  openai_runtime_require_exact: true
agent:
  reasoning_effort: xhigh
fallback_providers: []
approvals:
  mode: guarded_yolo
codex_runtime:
  permission_profile:
    name: command-center-development
    network_access: true
  workspaces:
    default_project: hermes-chat
    projects:
      hermes-chat: /home/will/coding-projects/hermes-chat
      reccli: /home/will/coding-projects/RecCli
      3dcarparts: /home/will/coding-projects/3dcarparts
      reg-watch: /home/will/coding-projects/reg-watch
      closure-engine: /home/will/coding-projects/closure-engine
      llm-view: /home/will/coding-projects/llm-view
      hermes-agent: /home/will/src/hermes-agent
  no_prompt:
    enabled: true
  reversible_deletion:
    enabled: true
  workspace_snapshots:
    enabled: true
    keep_snapshots: 32
    timeout_seconds: 600
```

Do not add an API provider fallback to this path. Auxiliary API-backed tools
must not be treated as decision authority.

This is bounded no-prompt execution, not an unsandboxed Codex profile. Hermes
Chat sends a project key, the gateway resolves it through the server-owned
`workspaces.projects` map, and Codex receives that canonical directory as its
workspace root. Routine commands and add/update patches proceed without a
browser approval round trip. Direct `rm`, `unlink`, and `rmdir` commands are
also no-prompt: an owner-only server shim plus run-scoped Codex
`shell_environment_policy` overrides archive their shell-expanded targets after
login-shell initialization through a capability-scoped loopback broker. The
broker performs capture outside the Codex filesystem sandbox, while Trash and
the other recovery stores remain absent from Codex's writable roots. Only then
can the real binary run. Reviewable guarded-YOLO operations such
as compound or opaque deletion, history changes, and inspectable patch deletes
or renames pause the run and use the existing browser **Allow once** / **Deny**
round trip. Commands Codex executes directly inside the workspace are still
covered by the mandatory pre-turn snapshot.
Hardline commands, `approvals.deny` matches, sudo password
injection, uninspectable or unrecognized patch kinds, and permission escalation
remain non-overridable policy cancellations. An unknown project key is rejected
before agent creation. Policy cancellations must never be presented as a
rejection made by the user.

The reversible-deletion contract in [reversible-deletion.md](reversible-deletion.md)
supersedes the prompt for direct workspace and safe-temp `rm`-family deletion
once that subsystem is enabled. Its workspace snapshot layer creates a
fail-closed recovery point before every native Codex turn and before recognized
opaque destruction. Recovery-store mutation, root/system deletion, and
non-file destruction remain blocked or reviewable under their existing policy.

The generic meaning of `approvals.mode: off` remains full approval bypass and
must not be used on command-center. The narrower behavior above applies only
to API-server native Codex runs when `codex_runtime.no_prompt.enabled` is
exactly `true`. Direct OpenRouter and other non-Codex runtimes continue through
the normal `guarded_yolo` policy instead of inheriting Codex's auto-approval.

Codex still runs as Linux user `will`, so the workspace sandbox is the write
boundary, not a confidentiality boundary: same-user files may remain readable
to subprocesses. A dedicated worker account is the next isolation step if
command-center later stores unrelated private material. Git history and the
tracked command-center backups remain the host-loss recovery layer. Incremental
per-turn workspace snapshots are the local recovery layer for permitted edits,
including replacement and deletion hidden inside an opaque executable.

Claude implementation jobs are pinned by the subscription worker to:

```text
claude --model claude-opus-5 --effort high
```

The worker ignores a queued Claude model override. This prevents old browser
state or a stale job from silently selecting another model.

The standard Hermes path exposes `opus_code_worker` through the Hermes tools
MCP bridge. Its sync client uses the local bearer-gated management service at
`http://127.0.0.1:8643`; the sync credential is read from
`~/.hermes-api-key` and is never passed to either model. The Codex app-server
process receives only a non-secret Hermes session identifier so worker jobs can
be bound to the originating conversation. The managed Codex MCP entry
whitelists `HERMES_GATEWAY_SESSION_ID` through `env_vars`; passing it only to
the parent app-server is insufficient because Codex restricts the environment
of stdio MCP children.

The command-center Git configuration must also have a local author name and
email. Claude does not need permission to commit: after the native session
returns, the subscription worker materializes the review commit itself. A
missing host Git identity therefore turns an otherwise valid implementation
into a failed durable job.

Install the tracked default-path RecCli policy and registry as user `will`:

```bash
install -m 0644 ops/command-center/AGENTS.md /home/will/AGENTS.md
install -d -m 0700 /home/will/.reccli
install -m 0600 ops/command-center/reccli-projects.json \
  /home/will/.reccli/projects.json
```

Register only repositories that exist on command-center. Each registered
project must retain its canonical `*.devproject` file and `devsession/`
history. Synchronize those RecCli artifacts deliberately; do not copy a Mac
registry containing Mac-only paths, and do not overwrite a tracked feature map
with an unreviewed local proposal.

## Agentic Loop Gate

The reusable state-gated cron design is documented in
[Build a Process-Driven Persistent Agent](../website/docs/guides/process-driven-persistent-agent.md).
This section contains only command-center deployment commands.

Install or update the tracked gate idempotently from the deployed canonical
checkout:

```bash
cd /home/will/src/hermes-agent
ops/command-center/install-agentic-loop.sh
```

The installer deploys private gate state and prompt files, primes only a new
installation, creates or updates exactly one cron job, and pins the gate plus
the morning/evening checkpoints to `openai-codex`, `gpt-5.6-sol`, and `xhigh`.

Inspect state without invoking a model:

```bash
~/.hermes/scripts/agentic-loop-gate.py status
```

Queue an explicit process-owned trigger:

```bash
~/.hermes/scripts/agentic-loop-gate.py enqueue \
  --id regwatch-source-audit-20260822 \
  --project reg-watch \
  --priority High \
  --task "Inspect the new source-monitor failure and propose one bounded next action."
```

Run a token-free check directly:

```bash
~/.hermes/scripts/agentic-loop-gate.py check
```

Do not delete `gate-state.json` to force a rerun. Use a new inbox ID for a new
intentional event, and acknowledge a batch manually only after verifying that
its exact events were handled durably.

## Orchestration Runbook

For an implementation request, Hermes should:

1. Resolve the registered project and call `load_project_context` before
   inspecting current code. A failed load blocks durable decisions and worker
   delegation.
2. Resolve material product or architecture choices as Sol 5.6 at `xhigh`.
3. Record a durable decision only when the result must survive the conversation.
4. Call `opus_code_worker` with `action=run`, a registered project key, a
   bounded specification, constraints, and at least one observable acceptance
   check.
5. If the returned status is `queued` or `running`, call `action=status` with
   the returned job ID and a positive bounded wait.
6. If the result contains `DECISION_NEEDED`, resolve it before issuing a new
   attempt. Increment `attempt` only for an intentional rerun of the same
   specification.
7. Inspect the returned worktree and commit directly. Compare the actual diff
   and test evidence with the accepted specification.
8. Present push, merge, deploy, or release as a separate human authorization.

An explicit stop may call `action=cancel`. Closing the browser, losing the SSE
connection, or sleeping the Mac is not a stop; the command-center job continues
and can be recovered by ID.

## Startup Checks

```bash
systemctl --user is-active hermes-gateway.service
systemctl --user is-active hermes-sync.service
systemctl --user is-active hermes-subscription-worker.service
systemctl --user is-active cloudflared-hermes.service
systemctl --user is-active hermes-command-center-backup.timer
systemctl --user is-active hermes-command-center-full-backup.timer
ss -ltn 'sport = :8642 or sport = :8643'
loginctl show-user will -p Linger
git config --global --get user.name
git config --global --get user.email
codex mcp get hermes-tools --json
codex mcp get reccli --json
rg -n '^default_permissions|^\[permissions\.command-center-development\]' \
  /home/will/.codex/config.toml
test -r /home/will/AGENTS.md
jq -e '.projects | length > 0' /home/will/.reccli/projects.json
test -x /home/will/.hermes/scripts/agentic-loop-gate.py
/home/will/.hermes/scripts/agentic-loop-gate.py status
jq -e --arg name 'Governed Agentic Loop Gate' \
  '[(.jobs // .)[] | select(.name == $name and .enabled == true)] | length == 1' \
  /home/will/.hermes/cron/jobs.json
jq -e '
  [(.jobs // .)[]
   | select(.name == "Governed Agentic Loop Gate"
         or .name == "Daily Founder Revenue Dispatcher"
         or .name == "Daily Founder Evening Review")
   | select(.provider == "openai-codex"
         and .model == "gpt-5.6-sol"
         and .reasoning_effort == "xhigh")]
  | length == 3
' /home/will/.hermes/cron/jobs.json
```

All services must be `active`. The MCP record must whitelist
`HERMES_GATEWAY_SESSION_ID`, `HERMES_HOME`, and `PYTHONPATH`.
The RecCli MCP must be enabled, every registry path must exist, and a direct
`load_project_context` call must return context for each registered project.
Its Codex `enabled_tools` list should contain only the default-path memory
surface: context loading, read/search/inspection tools, and
`save_session_notes`. Do not expose RecCli organization launch, approval,
promotion, deletion, recovery, or configuration tools through this automatic
approval path.

The generated Codex config must select `command-center-development`, extend
`:workspace`, and enable network access. It must never select
`:danger-full-access`. A fresh Hermes run should report the selected project in
`GET /v1/runs/<run_id>` and start Codex with that project's canonical path.

The `ss` output must show only `127.0.0.1:8642` and `127.0.0.1:8643`, never
`0.0.0.0` or `[::]` for either listener.

```toml
[mcp_servers.reccli]
command = "/home/will/.local/bin/reccli-mcp"
enabled_tools = [
  "doctor",
  "expand_search_result",
  "inspect_result_id",
  "list_issues",
  "list_sessions",
  "load_project_context",
  "preview_context",
  "save_session_notes",
  "search_by_file",
  "search_by_time",
  "search_history",
]
```

Verify native authentication without displaying token files:

```bash
codex login status
claude auth status
```

Verify the exact Codex model contract through the app-server `model/list`
method after CLI upgrades. Hermes performs this check automatically before it
starts a thread, but the operator should also run a canary turn after an
upgrade.

## Functional Canary

1. Start a fresh standard Hermes conversation and ask a read-only historical
   question about one registered project without instructing Hermes to use
   RecCli.
2. Confirm the visible tool sequence contains `load_project_context` for the
   matching absolute project path before project conclusions or worker calls.
3. Confirm the streamed response reaches Hermes Chat without reconnect gaps.
4. Ask Hermes to list recent decision records.
5. Queue a trivial Claude implementation job against a disposable test branch.
6. Confirm its result identifies `claude-opus-5`, contains checks, and leaves a
   local reviewable commit without push or deployment.
7. In a standard Hermes conversation, request a disposable code change and
   confirm the visible tool call is `opus_code_worker` rather than a direct
   Claude harness conversation.
8. Confirm the returned worker job has the same derived orchestration thread,
   exact model `claude-opus-5`, a local worktree/commit, and no release status.
9. Confirm Hermes continues after the tool result and reviews evidence before
   offering any promotion action.

Do not restart `hermes-gateway.service` while a run is active. Wait for the run
to complete or explicitly cancel it first.

## Backups

Install the tracked backup services from the deployed canonical checkout:

```bash
bash ops/command-center/install-command-center-backups.sh
```

Backups are private, atomic directories under
`~/.local/state/command-center-backups/`:

- `quick-*`: daily Hermes critical-state snapshot plus chat-sync SQLite and
  recovery configuration; 14 retained.
- `full-*`: weekly full Hermes export plus chat-sync SQLite and recovery
  configuration; 4 retained.

The script uses the command-center virtual environment's Python/SQLite runtime.
The Debian `sqlite3` CLI is older and can falsely report Hermes' newer trigram
FTS index as malformed. Override `COMMAND_CENTER_SQLITE_PYTHON` only with a
runtime whose SQLite compatibility has been verified against the live DB.

Run and inspect a manual quick backup:

```bash
systemctl --user start hermes-command-center-backup.service
systemctl --user status hermes-command-center-backup.service --no-pager
```

Run the full mode only when there is enough local disk space:

```bash
systemctl --user start hermes-command-center-full-backup.service
systemctl --user status hermes-command-center-full-backup.service --no-pager
```

For either artifact, run `sha256sum --check SHA256SUMS` inside its directory.
The timer succeeds only after SQLite integrity and ZIP checks pass, but the
manifest check independently detects later storage damage.

These backups share the host NVMe with the live data. Replicate verified
artifacts to separate storage for hardware-loss recovery.

## Decision Review

The ledger is append-only and can be inspected without a model:

```bash
jq -c . ~/.hermes/decision-ledger.jsonl
```

A decision is not complete merely because a record exists. Acceptance evidence
must match the stated scope, and irreversible actions still require human
authorization. Corrections append a new record with `supersedes` set to the old
record ID.

## Failure Modes

Exact Codex model missing or `xhigh` unsupported:

- Hermes returns an exact-runtime startup error.
- Do not weaken `openai_runtime_require_exact` as an incident workaround.
- Inspect `codex app-server` model availability and the installed CLI version.

Codex subscription logged out or usage-limited:

- Re-authenticate with the native CLI or wait for the subscription limit to
  reset.
- Do not paste an API key into the Hermes authority path.

Claude Opus 5 unavailable:

- The implementation job fails and remains reviewable in job history.
- Hermes may revise the plan, but another Claude model is not substituted.

Streaming interruption:

- Reconnect by run ID and replay persisted run events.
- Do not start a duplicate run until the original run status is known.

Gate database unavailable or state unreadable:

- The scheduled check emits `reason: gate_error` with `wakeAgent: false` and
  exits successfully so Hermes cannot turn an infrastructure failure into a
  recurring model call.
- Run the check directly to see its bounded error field. Install-time `prime`
  and interactive mutation commands still exit non-zero on failure.
- Repair the database path or private state file; do not replace the gate with
  an unconditional model schedule.

Gate batch not acknowledged:

- Inspect `agentic-loop-gate.py status` and the corresponding cron execution.
- The gate suppresses duplicates for six hours, then retries within its daily
  budget. A new event changes the batch ID immediately.
- Acknowledge manually only after verifying that the event batch was actually
  handled; clearing state is not acknowledgment.

Daily gate budget exhausted:

- Pending events remain durable and become eligible after the local date rolls
  over. User messages and separately named checkpoints are unaffected.
- Do not raise the budget as an incident workaround. Inspect whether an input
  source is producing unstable semantic state.

## Upgrade Procedure

1. Confirm no active Hermes or subscription-worker jobs.
2. Run and verify a command-center quick backup.
3. Update the `command-center` branch and run targeted tests.
4. Update one native CLI at a time.
5. Reapply the Hermes tools MCP migration to Codex configuration if needed.
6. Run exact-model probes before restarting the gateway.
7. Restart only the changed service and run the functional canary.

## Rollback

Restore a verified command-center backup, then restart the gateway after
confirming no run is active. Source rollback should use the commit recorded in
the backup's `metadata.txt`. Do not alter or stop the Raspberry Pi as part of
this rollback; it remains separate until a later decommission decision.
