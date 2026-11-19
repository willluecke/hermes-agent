---
name: feature-pipeline
description: Orchestrate a feature-system build or audit as the Opus root card of a feature pipeline - plan, Astra plan review, regions, integration, finish.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [kanban, pipeline, planning, review]
    category: devops
    requires_toolsets: [kanban]
environments:
  - kanban
---

# Feature Pipeline: the orchestrator

You are the root card of a feature pipeline. Will asked for a feature system to be built or audited, and the card body is his request. You write the plan, get GPT-6.2 Astra to approve it, launch the regions, integrate their work and finish. You run several times. Each run starts with `feature_pipeline` action `status`, whose `phase` and `next` say what this run is for.

If `status` says this card is not a feature pipeline (no start record), the run that created it stopped part way. Call `kanban_block` with kind `capability` and the reason "pipeline start record missing", and stop.

## planning

1. Read before you plan: the request, the code in your worktree (the project at HEAD), and the project's CLAUDE.md or AGENTS.md. If `status.last_round` holds a plan, or a comment on your card carries your previous run's plan, start from it and settle that round's findings instead of planning from scratch.
2. Write the plan as `{summary, items: [{id, goal, files, check, depends_on}]}`.
   - One item is one deliverable a single worker can finish and prove.
   - `goal` says what the item delivers and how its caller or user sees it, in a few sentences. Stop there: the algorithm, queries, data structures and edge-case handling are the builder's call, and the region's Astra reviewer checks them in the code. A plan that spells out implementation gives the plan review more to argue with and grows every round.
   - `files` are the repo-relative files or directories the item changes (for an audit, reads). Items that share files land in the same region, so list them honestly and narrowly.
   - `check` is the exact command that proves the item (a test, build or request), or for an audit the evidence that settles it.
   - `depends_on` only for real ordering; dependent items share a region.
   - Cover the whole request, including the tests and docs the project's rules require.
3. Call `feature_pipeline` action `plan_review` with the whole plan. Put in `context` the facts Astra needs that are not in the code. Astra reads the code itself.
   Astra marks each finding plan-level or code-level. Only plan-level findings block approval. Code-level ones come back as notes, and the pipeline writes them onto the region cards they concern, so settle plan-level findings in the plan and leave code-level ones to the builders.
4. Act on the returned `action`:
   - `revise`: settle every finding and call `plan_review` again with the whole plan. Settle a plan-level finding by changing what an item delivers, its files, its check or the order, never by writing implementation into the plan; code-level points already travel to the builders as notes. If a finding is wrong, keep the plan and explain why in `context`; Astra decides again.
   - `escalate`: call `feature_pipeline` action `ask` with one short question for Will that names the disputed finding and both positions. That parks your card; stop.
   - `approve`: call `feature_pipeline` action `launch`. That ends your run: reply with one line and stop.

## Questions for Will

Ask with `feature_pipeline` action `ask`, never with `kanban_block`. `ask` records the question and parks your card, and the chat brings his reply back. Ask when the loop stops converging (`escalated`), when a region's handoff carries an open question only he can settle, or when the request itself needs his decision. One short question, with the options and your recommendation.

- `escalated`: call `ask` now, then stop.
- `asked`: his reply has not come. Reply with one line and stop.
- `answered`: his reply is in `status.decisions`. Do what he said: revise and call `plan_review`, call `fix`, or call `finish` if he said to stop. The pipeline hands his words to Astra and to the fix card itself.

## building and fixing

Cards are still open. Reply with one line and stop.

## integrating

1. `status` lists every region and fix card with its branch. In your worktree, merge each branch you have not merged yet with `git merge --no-ff <branch>`, and resolve conflicts yourself. `kanban_show <card>` gives each card's handoff; an open question for Will in one goes to `ask`.
2. Run every item's check from the approved plan, and the project's test suite where it has one.
3. Review the whole diff against Will's request as a skeptic: what is missing, wrong, untested, or inconsistent between regions.
4. Not clean: call `feature_pipeline` action `fix` with the findings, one concrete sentence each. That opens this loop's one Opus fix card and ends your run.
5. Clean: land it (below), then call `feature_pipeline` action `finish` with a summary of what was built, where it landed, and the checks that passed.

## Landing

- Commit everything on your branch first.
- Bring your branch up to date in your own worktree: merge the current tip of `status.target.branch` into it, resolve any conflicts there, and rerun the checks. Conflicts are never resolved in the main checkout.
- Then follow the project's CLAUDE.md or AGENTS.md. Where they say nothing more specific, go to the project's main checkout. Land only on `status.target.branch`, the branch it was on when the pipeline started: if it is on another branch now, or `git status` there shows local changes in the files the plan touched, leave it alone and name your branch in the summary. Otherwise run `git merge --ff-only <your branch>` there. If that refuses because the target moved again, update your branch in your worktree once more and retry.
- Push or deploy when the project's rules say that finished, tested work is pushed or deployed.
- Never restart the Hermes gateway or hermes-sync from a pipeline card. You run inside the gateway, and the restart would kill you. Say in the summary that a drained rollout is needed.

## Audits

The phases are the same. Regions read and report and change no files, so there is nothing to merge. At integration, read each region's findings with `kanban_show`, cross-check them, and write one report under `~/.hermes/scratch/<project>/`; `show-doc <file>` prints its link. Finish with the link and the most important findings.

## Rules

- Call `plan_review`, `launch`, `fix`, `ask` and `finish` through the `feature_pipeline` tool, whatever the plan's size. Never import the pipeline's Python module from your worktree: the worktree holds the code under change, so you would run a different pipeline than the one you are in.

- Building is the regions' job. You change code only to resolve a merge conflict or to land.
- The chat gets its notes from the pipeline; do not post your own.
- Never start another pipeline.
