# Persistent Agent Operations

## Service Layout

Host: `command-center` (`will@192.168.1.243`)

- `hermes-gateway.service`: Hermes conversations and streamed run events
- `hermes-sync.service`: Hermes Chat synchronization
- `hermes-subscription-worker.service`: native Codex and Claude Code jobs
- `cloudflared-hermes.service`: public tunnel, when enabled

All agent services run as user `will`. User lingering must remain enabled.

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
```

Do not add an API provider fallback to this path. Auxiliary API-backed tools
must not be treated as decision authority.

Claude implementation jobs are pinned by the subscription worker to:

```text
claude --model claude-opus-5 --effort high
```

The worker ignores a queued Claude model override. This prevents old browser
state or a stale job from silently selecting another model.

## Startup Checks

```bash
systemctl --user is-active hermes-gateway.service
systemctl --user is-active hermes-sync.service
systemctl --user is-active hermes-subscription-worker.service
loginctl show-user will -p Linger
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

1. Send a low-risk Hermes message that requires no file changes.
2. Confirm the streamed response reaches Hermes Chat without reconnect gaps.
3. Ask Hermes to list recent decision records.
4. Queue a trivial Claude implementation job against a disposable test branch.
5. Confirm its result identifies `claude-opus-5`, contains checks, and leaves a
   local reviewable commit without push or deployment.

Do not restart `hermes-gateway.service` while a run is active. Wait for the run
to complete or explicitly cancel it first.

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

## Upgrade Procedure

1. Confirm no active Hermes or subscription-worker jobs.
2. Back up `config.yaml`, `SOUL.md`, `PIPELINE.md`, and both decision ledgers.
3. Update source in the migration branch and run targeted tests.
4. Update one native CLI at a time.
5. Reapply the Hermes tools MCP migration to Codex configuration if needed.
6. Run exact-model probes before restarting the gateway.
7. Restart only the changed service and run the functional canary.

## Rollback

Restore the timestamped Hermes configuration and policy files, then restart the
gateway after confirming no run is active. Source rollback should use the last
known-good migration commit. Do not alter or stop the Raspberry Pi as part of
this rollback; it remains separate until a later decommission decision.
