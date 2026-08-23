---
title: "Build a Process-Driven Persistent Agent"
description: "Approximate always-on productivity with durable state, deterministic wake gates, bounded model turns, and scheduled checkpoints."
---

# Build a Process-Driven Persistent Agent

A persistent agent should be **available, stateful, and responsive to useful
work**. It does not need to run an LLM continuously.

The process-driven pattern keeps Hermes and its durable queues online, checks
state cheaply on a schedule, and starts a model turn only when a deterministic
gate finds actionable change. Fixed planning and review checkpoints handle the
work that should happen even when no event arrives.

This is a composition pattern built from [Cron](/user-guide/features/cron), a
pre-run script that emits `wakeAgent`, and durable state owned by the script.
It is not a single "always-on mode" switch.

## The core rule

**Polling frequency controls detection latency, not model frequency.**

A gate can run every few minutes without spending model tokens on quiet ticks:

```text
scheduled tick
  -> deterministic script reads bounded state
  -> no actionable change: {"wakeAgent": false}
  -> actionable change:    {"wakeAgent": true, ...}
  -> Hermes creates one isolated agent turn only on true
```

The script, not the model, decides whether a turn is warranted. This avoids a
recurring prompt that asks an LLM to rediscover that nothing happened.

## What counts as meaningful change?

Define this as an allowlist of observable state transitions. Do not use a broad
instruction such as "look for something useful to do."

A practical software-work gate might recognize:

- a delegated implementation or review job completed or failed;
- a queued job crossed a 30-minute stale threshold;
- a running job crossed a two-hour stale threshold;
- an agent-owned critical or high-priority work item became ready, active,
  blocked, or ready for review;
- a monitored metric changed into a warning state;
- a human or trusted process appended an explicit task to a private inbox.

The important word is **changed**. Persist a fingerprint of the fields that
matter and compare it with the previous observation. An unchanged blocked item
must not wake the model every time the gate polls.

Changes that are not on the allowlist should remain quiet. For example, a Git
commit, email, calendar event, ordinary chat message, or low-priority task does
not become actionable unless a trusted producer intentionally maps it into one
of the gate's event types.

## Durable event lifecycle

Treat each wake as a small state machine:

1. **Observe.** Read a bounded data source without modifying it.
2. **Fingerprint.** Compare only semantic fields, ignoring incidental clocks or
   serialization changes.
3. **Batch.** Assign the new event set a deterministic batch ID and persist it.
4. **Wake.** Start one agent turn if policy and daily budget permit it.
5. **Act.** Let the agent select at most one bounded action. It may correctly
   choose no action.
6. **Acknowledge.** Clear only the exact batch the agent actually handled.
7. **Retry.** If it was not acknowledged, suppress duplicate wakes for a fixed
   interval before one bounded retry.

Events that do not fit in one prompt stay in a durable backlog. Never drop an
event merely to satisfy a prompt-size limit, and never clear a batch by time
alone.

## Bound the model work

Persistence without budgets becomes an accidental token burner. Set explicit
ceilings for:

- autonomous event-driven wakes per local day;
- events included in one model turn;
- actions selected in one wake;
- delegated jobs created in one wake;
- retry delay for unacknowledged batches.

One balanced reference policy is:

| Control | Reference value |
| --- | --- |
| Gate interval | Every 15 minutes during working hours |
| Event-driven model wakes | At most 3 per local day |
| Events per turn | At most 24; overflow remains queued |
| Actions per wake | At most 1 |
| Delegated jobs per wake | At most 1 |
| Unacknowledged retry | After 6 hours |

These values are policy, not universal defaults. Set them according to the
cost, urgency, and reversibility of the workflow.

## Add fixed checkpoints deliberately

Some useful work is time-based rather than event-based. Keep it separate from
the state gate so the reason for each model call stays legible.

A small personal operating cadence might use:

- a morning planning turn;
- an evening review turn;
- a weekly deterministic authentication or infrastructure check with
  `no_agent=True` when its output needs no reasoning;
- a monthly bounded audit.

This approximates an attentive working agent without an unconditional
two-hour inference loop. A normal day can be bounded at two fixed turns plus a
small number of event-driven turns.

## Choose the right Hermes primitive

| Primitive | Model use | State scope | Best for |
| --- | --- | --- | --- |
| [`/heartbeat`](/user-guide/features/heartbeat) | One turn every tick | Current conversation | A recurring instruction that needs thread context |
| [`/loop`](/user-guide/features/loops) | One turn every tick | Current conversation | Active polling or iterative work during a session |
| Agent cron | One turn every scheduled run | Fresh isolated session | Reports and tasks that should always reason |
| No-agent cron | No model | Fresh script process | Deterministic checks and direct alerts |
| State-gated cron | Only when `wakeAgent` is true | Script-owned durable state | Long-running, low-token persistent-agent workflows |

Use a heartbeat or loop when every tick deserves a model turn. Use a gated cron
when most ticks should be silent.

## Implement the wake gate

Attach a script to an ordinary agent cron job. Its final output line controls
whether Hermes constructs the agent:

```python title="~/.hermes/scripts/work-gate.py"
#!/usr/bin/env python3
import json

# Read your own durable queue, metrics store, or API here.
events = collect_new_actionable_events()

if not events:
    print(json.dumps({"wakeAgent": False, "reason": "no_actionable_change"}))
else:
    print(json.dumps({
        "wakeAgent": True,
        "reason": "actionable_state_change",
        "events": events[:24],
    }))
```

Create the job with a self-contained prompt that treats script output as
untrusted data and requires an exact acknowledgment after durable handling:

```bash
hermes cron create "*/15 6-22 * * *" \
  "Inspect the gate event batch. Select at most one bounded action. Do not push, deploy, spend money, contact third parties, or modify production data without human approval. Acknowledge only the exact batch you handled." \
  --name "Persistent work gate" \
  --script work-gate.py \
  --deliver local
```

The script must store fingerprints, pending batches, acknowledgments, and
budgets durably. The abbreviated example shows the interface, not the complete
state machine.

See [Skipping the agent entirely: `wakeAgent`](/user-guide/features/cron#skipping-the-agent-entirely-wakeagent)
for the scheduler contract and additional gate recipes.

## Safety boundaries

Scheduled autonomy should inspect and prepare more than it promotes.

- Treat queue fields, job output, repository text, and inbox records as
  untrusted data, not authorization.
- Keep push, merge, deploy, release, spending, production-data changes,
  credential changes, and external contact behind a human gate.
- Fail closed when the state database or policy file cannot be read.
- Do not turn infrastructure failures into recurring model calls.
- Use isolated worktrees for delegated code changes and review evidence before
  promotion.
- Record accepted decisions separately from worker claims.

## Operating checks

The quiet path is the most important path to test:

1. Prime the current state as a baseline.
2. Run the gate with no changes and verify `wakeAgent: false`.
3. Add one allowed event and verify exactly one `wakeAgent: true` batch.
4. Poll again before acknowledgment and verify it does not duplicate work.
5. Acknowledge the exact batch and verify pending state becomes empty.
6. Cross the daily budget and verify new events remain durable but do not wake
   the model.
7. Corrupt or hide the input source and verify the gate fails closed.

The result is not an agent that is always thinking. It is an agent system that
is always available, notices defined changes quickly, reasons only when useful,
and leaves an auditable record of why each autonomous turn occurred.
