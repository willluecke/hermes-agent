---
name: feature-pipeline-region
description: Build or audit one region of a feature pipeline as an Opus kanban worker, then hand it to GPT-6.2 Astra's review.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [kanban, pipeline, implementation]
    category: devops
    requires_toolsets: [kanban]
environments:
  - kanban
---

# Feature Pipeline: one region

You build one region of an approved plan: the items listed on your card. An Opus orchestrator owns the whole, and GPT-6.2 Astra reviews your work before it counts as done.

## If you are the reviewer

The review lane loads this skill next to `sdlc-review`. If you were claimed from review, you are the reviewer: follow `sdlc-review`, not the build steps below, and never build or commit. Check that the commits on the card's branch deliver every item on the card, run each item's check yourself, and confirm that nothing outside the items' files changed without a reason in the handoff. Check that every note from Astra's plan review on the card was handled. For an audit, check that each finding's evidence holds. Then approve with `kanban_complete`, or request changes with `kanban_request_changes` and one concrete point per problem. If only Will can settle something, do not block: approve what is right and put the open question in your summary, and the orchestrator asks him.

## Build

1. Call `kanban_show` for your card. It has your items (goal, files, check), the whole plan's summary, and your earlier runs. If Astra requested changes, the review is in that run history: settle every point.
   If the card lists notes from Astra's plan review, handle each one in the code and say how in your handoff.
2. Work only in your worktree: your current directory, on the card's own branch (`git branch --show-current`), which starts from the commit the plan was reviewed against. Stay inside your items' files. If an item cannot be done without touching another file, say so in the handoff.
3. Run each item's check and fix until it passes.
4. Commit on your branch with `git add -A` and `git commit`. Uncommitted work never reaches the integration.
5. Call `kanban_request_review` with a summary of what you did per item, and metadata `{changed_files, checks: [{command, result}]}`. That ends your run.

## Fix cards

A fix card names the integration branch. Merge it into your worktree first, then fix each finding on top of it. The rest is the same as a build.

## Audits

Change no files and make no commits. For each item, inspect the code and report what you find, with evidence (file:line, or command output). Call `kanban_request_review` with the findings as the summary, and metadata `{findings: [{item, finding, evidence, severity}]}`. Astra checks your evidence.

## Questions for Will

If an item needs a decision only Will can make, do not guess and do not block. Build everything that does not depend on it, and put the open question, with the options and your recommendation, at the top of your review handoff. The orchestrator asks Will and sends his answer back as a fix card.

## Never

- Start a pipeline or create cards.
- Merge into the project's main checkout, push, deploy, or restart a service. The orchestrator lands the whole.
