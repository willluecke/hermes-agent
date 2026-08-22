# Persistent Agent Architecture

## Purpose

Hermes is the always-on control layer on `command-center`. It preserves
conversation continuity, tools, scheduled work, and durable decisions while
using the user's existing native CLI subscriptions where their terms and
technical interfaces allow it.

This design deliberately separates judgment from implementation. A model's
output is evidence or a proposal until the designated authority accepts it.

## Runtime Contracts

### Decision Authority

- Provider: `openai-codex`
- Runtime: local Codex App Server over stdio
- Authentication: the native Codex CLI's ChatGPT subscription login
- Model: exact `gpt-5.6-sol`
- Reasoning effort: exact `xhigh`
- Failure policy: fail closed if either exact value is unavailable

Hermes sends the model and effort on every Codex thread and turn. At process
startup it calls `model/list` and verifies that the installed CLI advertises
the exact pair. Aliases, defaults, and silent downgrades are not accepted.

The app-server protocol provides the streamed events, thread history,
approvals, and local execution loop used by Hermes. See the official
[GPT-5.6 Sol model reference](https://developers.openai.com/api/docs/models/gpt-5.6-sol)
and [Codex App Server reference](https://learn.chatgpt.com/docs/app-server).

### Code Worker

- Provider: native Claude Code CLI
- Authentication: the existing Claude Max login
- Model: exact `claude-opus-5`
- Role: implementation in an isolated Git worktree
- Failure policy: fail if Opus 5 cannot be selected; do not fall back to a
  different Claude model

Claude may inspect code, edit files in its worktree, and run proportional
checks. It may not establish product policy, alter the authority contract,
write the decision ledger, push, deploy, merge, access secrets, or perform
irreversible external actions. Ambiguous specifications return to Hermes as a
`decision_needed` result with evidence and options.

## Standard Hermes Orchestration Loop

The standard Hermes chat is the governed path. A coding request remains one
Hermes conversation while execution crosses the native subscription boundary:

```text
user request
  -> Hermes/Sol inspects evidence and resolves decisions
  -> Hermes records a durable decision when warranted
  -> Hermes pins specification, constraints, and acceptance checks
  -> opus_code_worker queues exact Claude Opus 5
  -> Claude edits/tests in an isolated worktree and creates a local commit
  -> the durable result returns to the same Hermes turn
  -> Hermes inspects the commit and test evidence
  -> Hermes reports acceptance, revision, or DECISION_NEEDED
  -> the human separately authorizes push, merge, deploy, or release
```

`opus_code_worker` supports `run`, `status`, and `cancel`. A run receives a
deterministic job ID derived from the Hermes session, specification, and
explicit attempt number. Retrying a lost request therefore recovers the same
job rather than creating duplicate Opus work. Worker ownership is bound to the
originating Hermes session; another conversation cannot inspect or cancel it.

The tool waits for a bounded interval so one slow implementation cannot wedge
the model tool transport. If the job remains queued or running, it returns its
durable ID and progress. Hermes calls `status` to continue waiting in the same
turn. Worker completion is not acceptance: Sol must inspect the worktree commit
and test evidence before recommending promotion.

Direct Codex and Claude chat harnesses remain manual native-session surfaces.
They bypass this automatic decision/worker/review loop and do not make Claude a
decision authority.

## RecCli Context Gate

RecCli is active on the default Hermes path as the project-memory layer. For
substantive work, Hermes resolves the requested project through
`~/.reccli/projects.json` and loads that project's RecCli context before Sol
makes a durable project decision or delegates to `opus_code_worker`. Ordinary
non-project conversation does not trigger a project-selection prompt.

The command-center Codex App Server starts with `/home/will` as its working
directory. The host-level `/home/will/AGENTS.md` therefore carries the context
gate into every new default Hermes Codex session. The deployed file is tracked
as `ops/command-center/AGENTS.md`; the server registry is tracked as
`ops/command-center/reccli-projects.json`.

A context-load failure is fail-closed for durable project decisions and Opus
delegation. Hermes may continue a clearly labeled read-only investigation, but
must disclose that project history was unavailable. RecCli output is advisory:
current code and primary evidence still control verification, Sol owns
judgment, and project memory never grants promotion authority.

Meaningful project outcomes are saved back to RecCli once at the end of work.
A save failure is reported and does not cause an unbounded retry loop.

## Authority Matrix

| Activity | Hermes / Sol 5.6 xhigh | Claude Opus 5 | Human |
| --- | --- | --- | --- |
| Clarify goals and constraints | Owns | May identify ambiguity | Final source |
| Architecture and product decisions | Owns | Proposes only | Can override |
| Durable decision records | Writes | No access | Can amend/supersede |
| Implementation | Specifies and reviews | Owns worktree changes | May intervene |
| Tests and evidence | Defines acceptance; verifies | Runs and reports | Accepts risk |
| Push, deploy, merge, money, external contact | Recommends only | Prohibited | Authorizes |

## Decision Ledger

Durable decisions live in `~/.hermes/decision-ledger.jsonl`. The
`decision_log` tool appends one structured JSON record under an exclusive file
lock and calls `fsync` before returning success. Existing records are never
rewritten. A later decision supersedes an earlier record by ID.

Every record includes:

- decision, rationale, scope, status, and UTC timestamp
- concrete evidence and considered alternatives
- confidence and reversibility
- dissent or unresolved risk
- exact authority provider, model, effort, and runtime

The write path rejects calls unless the active configuration is an exact
OpenAI/Codex app-server authority. Reading remains available for audit and
context recovery.

## Subscription Boundary

The Codex app-server and Claude Code worker authenticate independently through
their native CLI login stores. Hermes does not turn either subscription into a
general third-party API credential. Direct OpenAI or Anthropic SDK calls remain
API-billed paths and are outside this control architecture.

There is no API fallback for the decision authority. If the Codex subscription
is exhausted, logged out, or no longer exposes the exact model contract,
decision turns stop with a visible error. Existing services and ledger data
remain intact.

## Persistence Boundary

Systemd user services and lingering keep Hermes and the subscription worker
alive when SSH disconnects or the Mac sleeps. Persistence does not mean
unbounded autonomy: scheduled triggers may start work, but irreversible or
external promotion steps remain human-owned.

Persistence applies at each orchestration boundary: the sync store owns the
job queue and status, the worker owns worktrees and native Claude session IDs,
Hermes owns conversation and run-event history, RecCli owns project memory, and
the decision ledger owns accepted judgment. No model is continuously thinking
between events.

The Raspberry Pi is not part of this runtime and remains an independent
rollback host until explicitly retired.
