# system-one-preflight: how Jev runs in Hermes

Start here. This is the current shape of the Jev integration, for an agent that
has to change or debug it. The module docstring in `__init__.py` has the full
detail; the dated history of *why* each piece is the way it is lives in
`~/coding-projects/command-center/docs/jev-integration.md`.

## What Jev is

TypeSafe's Jev is a **decisions model, not a chat model**. You send it a piece
of state plus typed questions (`noul` = a yes/no probability, `choice`, level
scores) and get calibrated probabilities back in ~0.3 s. It never writes code or
text, and no text it returns is ever injected into a prompt: every note the
model sees is fixed template text filled with numbers or quoted sentences.

Transport: `tools/typesafe_tool.py` (`ask_jev`). `TYPESAFE_PROVIDER` in
`~/.hermes/.env` orders the providers; it is `openrouter,typesafe` (OpenRouter
decisions endpoint first, TypeSafe direct as fallback). Model id `jev-latest`.
A slow or failed call fails **open**: the turn proceeds without advice.

## Where it lives

| Piece | Path |
|---|---|
| The plugin (all hooks, all Jev questions) | `plugins/system-one-preflight/__init__.py` |
| Evidence ledger, manifest checks, re-runs | `plugins/system-one-preflight/evidence.py` |
| Explicit tool, for a model's own decisions | `tools/typesafe_tool.py` (`typesafe_decide`) |
| Model-facing criteria / results tools | `tools/acceptance_criteria_tool.py`, `tools/report_results_tool.py` |
| Verify loop bound, send-back plumbing | `agent/verify_hooks.py`, `hermes_cli/plugins.py` |
| Claude Code lane parity | `agent/claude_runtime.py` (`_claude_hook_parity`), `hermes_cli/claude_hooks.py` (`/v1/hooks/claude`) |
| Codex lane parity | `agent/codex_runtime.py` (`_codex_hook_parity`, `_with_turn_note`) |
| Tools exposed to native CLIs | `agent/transports/hermes_tools_mcp_server.py` |
| Verdicts onto the run stream | `hermes_cli/turn_events.py` → `judge.verdict` events |
| Chat rows | hermes-chat `src/lib/run-events.ts`, `src/lib/run-replay.ts` |
| Outcome store + `/management/judge/calibration` (no new labels since 2026-09-30) | hermes-chat `command-center/judge-store.mjs` (deployed copy in `~/hermes-sync/`) |

## One turn, hook by hook

1. **`pre_llm_call`: budget read.** One Jev call asks: how hard, how
   objectively checkable, what kind (answer / build / operate / research…),
   how ambiguous, is it a build. In `mode: feedback` an advisory note is added
   to the user message **only** when P(ambiguous) ≥ 0.6, the plan is
   `candidates` (hard + checkable → "make 3 candidates, pick with
   typesafe_decide") or `criteria_only`, or as the once-per-session build
   criteria nudge. Otherwise the row says `note withheld`. The row also shows
   the injection rate over the last 50 turns. Carried criteria open for 3+
   **build** requests (a question between builds does not count) are raised
   here as a fixed line (finish or retire with a reason) on every build
   request they linger; the user sees one one-line `Jev criteria` row per
   set of lingering criteria, not the texts, which the retirements show.
   The previous answer rides along in the state, so a follow-up such as "yes,
   do it" is read with what it refers to. (Until 2026-09-30 the same call also
   rated that answer worked / partly / failed from the follow-up; the labels
   went unused and the read, the chat buttons and the calibration dialog were
   removed.)
2. **`pre_tool_call`: tool guard.** For `terminal`, `write_file`, `patch`, it
   asks about destructive / out-of-scope consequences. `tool_guard: shadow`
   = log only, never hold.
3. **`post_tool_call` / `transform_tool_result`: ledger and criteria.** Every
   terminal call becomes a ledger row (`c1, c2, …`: exit code, runner counts,
   a digest of each repository it ran under, full output kept on disk). A
   command that failed is a row too: on the Claude lane it arrives through
   Claude Code's `PostToolUseFailure` hook (until 2026-10-02 only successful
   calls were reported, so a failing check could not be cited). A row
   is stale only when one of those repositories changed after it; first
   touching another repository later does not count. The ids are internal: the
   model is never shown one, so it never has to cite one. Every Edit/Write
   path is also recorded per session in `edits.json` beside the retained
   output, which survives a restart. A criteria registration
   gets the fidelity check: one noul per criterion, "does the user's request
   entail this"; under 0.4 it is excluded from judging and named back. The
   coverage steer ("add more criteria") is **disabled** by config. Criteria
   retirements (`retire: [{content, reason}]`) each become a visible row.
4. **`pre_verify`: verify judge**, before a turn that edited files, or ran a
   command, finishes. The changed paths are those of the turn's Edit, Write
   and apply_patch calls; when there are none, the edits came through
   commands (a Python heredoc, `sed -i`) and the judge reads them from git in
   the repositories the turn's commands ran in: tracked files that differ
   from the turn's base commit, untracked files written since the turn
   began (`changed_source: git` in the verify record). Until 2026-10-01 such
   turns were never judged: 28 of that week's 54 build turns, whose criteria
   therefore never retired. A turn that changed nothing is not judged (a row
   says so when it had registered work).
   Skipped (`Jev verify: skipped · not a build turn`) when the preflight read
   was not a build, no check ran, and no criteria or manifest exist. Otherwise
   one Jev call over the diff, ledger, failure excerpts, manifest and draft
   answer asks the questions below. The diff runs from the commit each
   repository was at when the turn began (or first touched it), so work the
   turn committed before finishing still shows; `diff_base` in the verify
   record names it:
   - one noul per **this turn's** criterion (unmet below 0.2 → finding; a
     carried criterion rated met is retired, one still unmet stays open but
     never sends back). Criteria are judged with `earlier_work` beside the
     diff: the current content of files the session edited in earlier
     turns that the criteria or request mention, cut to the 40-line windows
     that mention them, under its own 30k budget (this turn's diff can fill
     its 60k cap alone);
   - "do the excerpts show a failure not fixed afterwards";
   - one noul per sentence of the draft answer (≤ 12 sentences, ≥ 6 words),
     "is this supported by the evidence shown"; ≤ 0.25 → quoted back as
     `Unsupported by the evidence: "…"`;
   - whether the changes satisfy the request as a whole.
   The **manifest** (`report_results`: claim, the commands it rests on as
   the model ran them, predicate) is checked by code, not Jev.
   `evidence.resolve_evidence` matches each cited command to its newest run
   (exact form first, `cd X &&` and output filters ignored, then a distinctive
   part of 8+ characters); a cited row id is refused, with a finding telling
   the model to cite the command; a `contains` claim takes the newest output
   that printed its text. Then `evidence.check_assertion`: supported /
   contradicted / stale (ran before a later edit) / missing / insufficient.
   Plain check commands (pytest, tsc, npm test…) are re-run by the gate,
   under bash (`/bin/sh` is dash here, which has no `time`).
5. **Send-back.** At most `verify_max_send_backs` (1) per turn, and only if
   the *evidence* (diff + commands + outputs) changed since the last attempt.
   Rewording the answer does not buy another round. After that the answer
   ships with `decision.action: ship_flagged` and the flagged sentences
   quoted on the row. `agent.max_verify_nudges: 2` is the outer loop bound.
6. **`post_llm_call`: record.** Logs what the model did (tool calls, tools
   used, answer length) for the turn's `turn_end` record.

## Labels, precision, tuning

A flagged sentence is joined to a manifest item by token containment and
labelled from **execution**, never from Jev or a human: `false_flag` (gate
re-ran it and it held), `overclaim` (contradicted / stale / missing),
`overclaim_unregistered` (manifest exists, nothing covers the sentence),
`unlabelled` (no manifest, agent-only row, or insufficient). Precision over 7
days = real / (real + false) and shows on the verify row. `maybe_tune` runs
hourly and moves `verify_fail_threshold` / `claims_flag_threshold` within
bounds only when that threshold's own basis reaches 50 labels; it writes
`~/.hermes/system-one-preflight.tuning.json` only when a threshold moves.
Each pass is a `tune` log record with the basis counts and `changes`. The
basis is outcome labels, and none arrive since 2026-09-30 (it stood at 18),
so the tuning cannot move a threshold unless labels come back.

## Settings (`plugins.entries.system-one-preflight.settings` in `~/.hermes/config.yaml`)

| Key | Live | Meaning |
|---|---|---|
| `mode` | `feedback` | `off` / `shadow` (ask + log, inject nothing) / `feedback` / `jev` / `always` / `trial` |
| `threshold` | 0.7 | P(missing verification) cutoff, used by the `jev` mode only |
| `timeout_seconds` | 1.5 | Preflight Jev call timeout |
| `tool_guard` | `shadow` | `feedback` would hold a risky call once |
| `verify_judge` | `on` | The `pre_verify` gate |
| `verify_send_back` | `on` | `off` = flag-only: one verdict row, never a send-back |
| `verify_max_send_backs` | 1 | Send-backs per turn |
| `criteria_nudge` | `on` | Once-per-session "register criteria" note on build turns |
| `drift_check` | `off` | Every N tool calls: which criterion is this work for? |
| `fidelity_check` | `on` | Entailment check on registered criteria |
| `fidelity_coverage_threshold` | 0 | 0 disables the "add more criteria" steer |
| `manifest_required` | `on` | A build turn that ran checks and over-claims must register a manifest |
| `controller_reruns` | `on` | Gate re-runs stale check commands itself |

Settings are re-read on every hook call (mtime-cached): **no restart needed**.
Write `"off"` quoted; YAML reads bare `off` as `false` (handled by `_switch`,
but quote it anyway). **Code** changes need a gateway restart, which kills
live runs: use `~/coding-projects/command-center/bin/gateway-idle-restart`.

## Checking it works

- Log: `~/.hermes/logs/system-one-preflight.jsonl`, one JSON line per event
  (`preflight`, `verify`, `manifest`, `criteria`, `fidelity`, `tool_guard`,
  `turn_end`, `tune`).
  Newest verdict: `tail -n 50 … | jq 'select(.event=="verify")'`. Its
  `citations` counts how claims were matched (exact / partial / text /
  row_id / unmatched), each assertion carries its own `citations`, and
  `earlier_work` names the earlier files the judge saw.
- In the chat: `Jev budget: …` and `Jev verify (attempt N): …` rows.
- Calibration: `/management/judge/calibration` on the sync store (labels
  recorded before 2026-09-30 only).
- Tests: `tests/plugins/test_system_one_preflight.py`,
  `tests/plugins/test_system_one_evidence.py`, `tests/agent/test_verify_hooks.py`,
  `tests/gateway/test_claude_hook_endpoint.py`,
  `tests/run_agent/test_claude_hook_parity.py`,
  `tests/run_agent/test_codex_hook_parity.py`,
  `tests/run_agent/test_verification_continuation_budget.py`.

## Known limits

- Lanes differ: the tool guard cannot hold calls on the Claude or Codex lanes.
  On Codex, fidelity notes arrive as verify findings, and drift steers are
  replayed after the fact and never judged.
- Sentences no command can settle ("the design is cleaner") stay unlabelled.
- Studies (2026-09-20) found no repeatable decision-quality gain from the
  per-turn loop; the evidence-support read itself was accurate. Treat it as
  a cheap check on over-claiming, not a proven quality lever.
