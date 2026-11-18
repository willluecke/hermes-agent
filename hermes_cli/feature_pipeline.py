"""The feature pipeline: planned, reviewed, region-by-region builds and audits.

Will, 2026-11-18: "Always have it as a plan that is executed, not just an
agent toddling across the map." Every build or audit of a *feature system*
(:data:`FEATURE_SYSTEM_DEFINITION`) runs as one pipeline on the project's
kanban board:

1. **Plan.** An Opus orchestrator (the root card) writes a structured plan:
   items with a goal, the files they touch, and a check that proves them.
2. **Plan consensus.** GPT-6.2 Astra reviews the plan in a read-only session
   inside the project checkout, so it can open the files the plan names
   (:func:`plan_review`). The orchestrator revises until Astra approves.
   There is no round cap; the loop stops when it stops making progress (the
   plan comes back unchanged and Astra still objects, or Astra raises the
   same finding three rounds running), and then the orchestrator asks Will
   (:func:`ask`). His reply (:func:`answer`) goes to Astra as a decision.
3. **Regions.** Approved items are grouped into regions: items whose files
   overlap, or that depend on each other, share one (:func:`group_regions`).
   Each region is one card, built by an Opus worker in its own git worktree
   and reviewed by an Astra worker (the card's review model, through the
   kanban review lane). Requested changes return to the same Opus card with
   Astra's review stack.
4. **Integration.** When every region is done the root wakes, merges the
   region branches into its own worktree, runs every check and reviews the
   whole. Clean: it lands the result and finishes. Not clean: one Opus fix
   card (:func:`fix`), Astra-reviewed like a region, and the loop repeats.

An audit runs the same pipeline with read-only regions whose findings Astra
verifies, and an integration that writes one report.

State lives on the root card as structured comments (``[pipeline] {...}``),
like the swarm blackboard: no new service and no new table. Every step that
creates cards is safe to retry: cards carry idempotency keys and a step that
stopped half way resumes instead of duplicating work. The chat that started
the pipeline follows its cards. The chat digest drops the regions' routine
progress (``gateway/kanban_chat_digest.py``), so what reaches Will is the
pipeline's own short notes, a question, a stuck card, and the finish.
"""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import re
import secrets
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

ORCHESTRATOR = ("claude-opus-5-5", "claude-code")
BUILDER = ("claude-opus-5-5", "claude-code")
REVIEWER = ("gpt-6.2-astra", "openai-codex")
#: Thinking depth for every pipeline card. The review lane reuses the card's
#: effort for its reviewer, and Astra's own default is low.
EFFORT = "high"
ORCHESTRATOR_SKILL = "feature-pipeline"
REGION_SKILL = "feature-pipeline-region"
WORKER_PROFILE = "default"
STATE_PREFIX = "[pipeline] "
STATE_AUTHOR = "feature-pipeline"
KINDS = ("build", "audit")
VERDICTS = ("approve", "changes")
MAX_ITEMS = 40
MAX_FINDINGS = 30
#: Astra must see the whole plan and request, so oversized ones are refused
#: rather than cut.
MAX_PLAN_CHARS = 40_000
MAX_REQUEST_CHARS = 20_000
MAX_ANSWER_CHARS = 20_000
#: The chat digest keeps 600 characters of a block (kanban_chat_digest), so a
#: question whose one-line notice fits that reaches Will whole even if the
#: note cannot be posted.
MAX_NOTICE_CHARS = 600
QUESTION_PREFIX = "Question for Will: "
EARLIER_ROUNDS_SHOWN = 6
REPEAT_SIMILARITY = 0.8
#: Within the Codex lane's 600-second cap on a hermes-tools call.
REVIEW_TIMEOUT_SECONDS = 600
REVIEW_INACTIVITY_SECONDS = 300
TEST_DIRS = frozenset({"test", "tests", "__tests__", "spec", "specs", "e2e"})
CLOSED_STATUSES = frozenset({"done", "archived"})
#: Events that settle what the root is doing; see :func:`phase`.
MARKERS = ("ask", "plan_round", "launch", "fix", "fix_escalate", "finish")

FEATURE_SYSTEM_DEFINITION = (
    "A feature system is a capability Will would name (the notes page, message "
    "times, checkout, the HR workspace) when the work builds or audits it as a "
    "whole: it spans two or more layers (interface, server or API, data store, "
    "background jobs, integrations) or two or more repos, or its plan has three "
    "or more items that touch separate files. An individual aspect is a change "
    "inside one layer of one feature: a bug fix, a control, a label, a "
    "threshold, one endpoint; it is done directly, without the pipeline. A "
    "message that starts with 'system:' always uses the pipeline; one that "
    "starts with 'aspect:' never does."
)


class PipelineError(ValueError):
    """A pipeline request that cannot be carried out as asked."""


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

def _escapes_repo(raw: str) -> bool:
    text = str(raw or "").strip().replace("\\", "/")
    return text.startswith(("/", "~")) or ".." in text.split("/")


def normalize_path(path: Any) -> str:
    """A repo-relative path in canonical form; ``.`` is the whole repo."""
    text = str(path or "").strip().replace("\\", "/")
    if not text:
        return ""
    return posixpath.normpath(text)


def paths_overlap(a: str, b: str) -> bool:
    """Same file, or one contains the other; the repo root contains everything."""
    a, b = normalize_path(a), normalize_path(b)
    if not a or not b:
        return False
    if "." in (a, b):
        return True
    return a == b or a.startswith(b + "/") or b.startswith(a + "/")


def _topological(items: Sequence[Dict[str, Any]]) -> Optional[List[str]]:
    """Item ids with every dependency first (plan order otherwise), or None on a cycle."""
    order = {item["id"]: index for index, item in enumerate(items)}
    deps = {item["id"]: [d for d in item["depends_on"] if d in order] for item in items}
    done: List[str] = []
    state: Dict[str, int] = {}

    def visit(node: str) -> bool:
        if state.get(node) == 2:
            return True
        if state.get(node) == 1:
            return False
        state[node] = 1
        for dep in sorted(deps[node], key=order.get):
            if not visit(dep):
                return False
        state[node] = 2
        done.append(node)
        return True

    for item in items:
        if not visit(item["id"]):
            return None
    return done


def validate_plan(plan: Any) -> Dict[str, Any]:
    """The plan in canonical form, or :class:`PipelineError` listing every problem."""
    if isinstance(plan, str):
        try:
            plan = json.loads(plan)
        except ValueError as exc:
            raise PipelineError(f"plan is not valid JSON: {exc}") from None
    if not isinstance(plan, dict):
        raise PipelineError("plan must be an object with summary and items")
    problems: List[str] = []
    summary = str(plan.get("summary") or "").strip()
    if not summary:
        problems.append("summary is required: one sentence on what the whole plan delivers")
    raw_items = plan.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        problems.append("items must be a non-empty list")
        raw_items = []
    if len(raw_items) > MAX_ITEMS:
        problems.append(f"at most {MAX_ITEMS} items")
    items: List[Dict[str, Any]] = []
    seen: set = set()
    for index, raw in enumerate(raw_items, 1):
        if not isinstance(raw, dict):
            problems.append(f"item {index} must be an object")
            continue
        item_id = str(raw.get("id") or f"i{index}").strip()
        if item_id in seen:
            problems.append(f"item id {item_id} is used twice")
        seen.add(item_id)
        goal = str(raw.get("goal") or "").strip()
        raw_files = [str(f) for f in (raw.get("files") or []) if str(f or "").strip()]
        for path in raw_files:
            if _escapes_repo(path):
                problems.append(
                    f"item {item_id} path {path.strip()} must be inside the project's repo; "
                    "work in another repo runs as its own pipeline on that project"
                )
        files = list(dict.fromkeys(normalize_path(f) for f in raw_files if not _escapes_repo(f)))
        check = str(raw.get("check") or "").strip()
        if not goal:
            problems.append(f"item {item_id} needs a goal")
        if not raw_files:
            problems.append(f"item {item_id} needs the files or directories it touches")
        if not check:
            problems.append(f"item {item_id} needs a check that proves it (a command, or for an audit the evidence)")
        depends = list(dict.fromkeys(str(d).strip() for d in (raw.get("depends_on") or []) if str(d).strip()))
        items.append({"id": item_id, "goal": goal, "files": files, "check": check, "depends_on": depends})
    known = {item["id"] for item in items}
    for item in items:
        for dep in item["depends_on"]:
            if dep == item["id"]:
                problems.append(f"item {item['id']} depends on itself")
            elif dep not in known:
                problems.append(f"item {item['id']} depends on unknown item {dep}")
    if not problems and _topological(items) is None:
        problems.append("the items' depends_on form a cycle")
    canonical = {"summary": summary, "items": items}
    if len(json.dumps(canonical)) > MAX_PLAN_CHARS:
        problems.append(f"the plan is over {MAX_PLAN_CHARS} characters; Astra must see all of it, so tighten the "
                        "items or split the feature into pipelines")
    if problems:
        raise PipelineError("; ".join(problems))
    return canonical


def plan_digest(plan: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(plan, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _subsystem(path: str) -> Optional[str]:
    """The layer a path belongs to: its first two directories, tests excluded."""
    parts = [p for p in normalize_path(path).split("/") if p and p != "."]
    if not parts or parts[0] in TEST_DIRS or any(p in TEST_DIRS for p in parts[:-1]):
        return None
    dirs = parts[:-1] if "." in parts[-1] else parts
    if not dirs:
        return "(root)"
    return "/".join(dirs[:2])


def feature_system(plan: Dict[str, Any]) -> Dict[str, Any]:
    """The definition measured on a plan: layers spanned and separate-file items."""
    subsystems = sorted({s for item in plan["items"] for s in (_subsystem(f) for f in item["files"]) if s})
    separate: List[set] = []
    for item in plan["items"]:
        files = set(item["files"])
        if not any(any(paths_overlap(a, b) for a in files for b in other) for other in separate):
            separate.append(files)
    return {
        "feature_system": len(subsystems) >= 2 or len(separate) >= 3,
        "subsystems": subsystems,
        "items": len(plan["items"]),
        "separate_file_items": len(separate),
    }


def group_regions(items: Sequence[Dict[str, Any]]) -> List[List[str]]:
    """Item ids grouped into regions: overlapping files or a dependency join a region.

    Regions are ordered by their first item in the plan. Inside a region,
    every item comes after the items it depends on.
    """
    ids = [item["id"] for item in items]
    order = {item_id: index for index, item_id in enumerate(ids)}
    parent = {i: i for i in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            if order[ra] < order[rb]:
                parent[rb] = ra
            else:
                parent[ra] = rb

    for i, a in enumerate(items):
        for b in items[i + 1:]:
            if any(paths_overlap(x, y) for x in a["files"] for y in b["files"]):
                union(a["id"], b["id"])
        for dep in a.get("depends_on") or []:
            if dep in parent:
                union(a["id"], dep)
    sequence = _topological([{"id": i["id"], "depends_on": i.get("depends_on") or []} for i in items]) or ids
    regions: Dict[str, List[str]] = {}
    for item_id in sequence:
        regions.setdefault(find(item_id), []).append(item_id)
    return sorted(regions.values(), key=lambda region: min(order[i] for i in region))


# ---------------------------------------------------------------------------
# Findings and the stop rule
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOP = frozenset({"the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "it",
                   "be", "with", "that", "this", "as", "at", "by"})


def _words(text: str) -> set:
    return {w for w in _WORD_RE.findall(str(text or "").lower()) if w not in _STOP}


def finding_text(finding: Any) -> str:
    if isinstance(finding, dict):
        return " ".join(str(finding.get(k) or "") for k in ("item", "issue")).strip()
    return str(finding or "").strip()


def _same(a: Any, b: Any) -> bool:
    wa, wb = _words(finding_text(a)), _words(finding_text(b))
    return bool(wa and wb) and len(wa & wb) / len(wa | wb) >= REPEAT_SIMILARITY


def stuck_findings(earlier: Sequence[Sequence[Any]], current: Iterable[Any]) -> List[str]:
    """Current findings raised in every one of the ``earlier`` rounds too.

    With the last two rounds as ``earlier`` this is a finding raised three
    rounds running: two revisions failed to settle it, so the loop is not
    converging and Will decides.
    """
    if not earlier:
        return []
    return [
        finding_text(finding) for finding in current
        if all(any(_same(finding, old) for old in round_findings) for round_findings in earlier)
    ]


REVIEW_INSTRUCTIONS = (
    "You are GPT-6.2 Astra, reviewing a plan that Claude Opus wrote for Will's "
    "request. Your working directory is a read-only checkout of the project at "
    "the state the plan starts from: open the files the plan names, and any "
    "others you need, to check the plan against the real code. You cannot "
    "change anything. Approve only if the plan, carried out as written, fully "
    "delivers the request, every item's check would prove that item, and the "
    "order and dependencies are sound. Otherwise list each problem. Raise only "
    "problems that would make the result wrong, incomplete or unverifiable; "
    "style preferences and optional extras are not findings. Do not raise a "
    "problem an earlier round settled, and treat Will's decisions as final."
)
AUDIT_REVIEW_INSTRUCTIONS = (
    "You are GPT-6.2 Astra, reviewing an audit plan that Claude Opus wrote for "
    "Will's request. Your working directory is a read-only checkout of the "
    "project: look at the code to check what the plan covers. You cannot change "
    "anything. Approve only if the plan covers every part of the feature system "
    "the request names, and every item names the evidence that would establish "
    "its findings. Otherwise list each gap. Do not raise a gap an earlier round "
    "settled, and treat Will's decisions as final."
)
VERDICT_FORMAT = (
    "End your answer with the verdict as JSON, nothing after it: "
    "{\"verdict\": \"approve\" | \"changes\", \"findings\": [{\"item\": "
    "\"<item id or plan>\", \"issue\": \"<what is wrong>\", \"fix\": "
    "\"<what to change>\"}]}"
)


def _plan_rounds(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [e for e in events if e.get("type") == "plan_round"]


def decisions(events: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Will's answers, each with its question: the root's questions and those
    of stuck region or fix cards.

    An amended answer replaces the earlier one to the same question; the
    event history keeps both. Answers to different questions never replace
    each other.
    """
    pairs: List[Dict[str, str]] = []
    slot: Dict[tuple, int] = {}
    question, asked_at = "", -1
    for index, event in enumerate(events):
        if event.get("type") == "ask":
            question, asked_at = str(event.get("question") or ""), index
        elif event.get("type") == "will_answer" and event.get("answer"):
            card = event.get("card")
            if card:
                recorded = str(event.get("question") or "")
                # A reply recorded without its question cannot be matched to
                # another, so it keeps a slot of its own.
                key = ("card", card, recorded) if recorded else ("card-event", index)
                pair = {"question": f"On card {card}: {recorded or 'its question'}", "answer": str(event["answer"])}
            else:
                key = ("ask", asked_at)
                pair = {"question": question, "answer": str(event["answer"])}
            if key in slot:
                pairs[slot[key]] = pair
            else:
                slot[key] = len(pairs)
                pairs.append(pair)
    return pairs


def review_prompt(*, kind: str, request: str, plan: Dict[str, Any], events: List[Dict[str, Any]],
                  context: str = "") -> str:
    """Astra's prompt for one plan round: the whole request and the whole plan."""
    instructions = AUDIT_REVIEW_INSTRUCTIONS if kind == "audit" else REVIEW_INSTRUCTIONS
    rounds = _plan_rounds(events)
    parts = [instructions, VERDICT_FORMAT, f"Will's request:\n{request.strip()}"]
    if rounds:
        shown = rounds[-EARLIER_ROUNDS_SHOWN:]
        earlier = [{"round": r.get("round"), "verdict": r.get("verdict"), "findings": r.get("findings") or []}
                   for r in shown]
        parts.append(f"Earlier rounds ({len(shown)} of {len(rounds)}):\n" + json.dumps(earlier, indent=1))
    answered = decisions(events)
    if answered:
        parts.append("Will's decisions (final):\n" + "\n".join(
            (f"- Asked: {d['question']}\n  Will: {d['answer']}" if d["question"] else f"- Will: {d['answer']}")
            for d in answered))
    if context.strip():
        parts.append("Notes from the orchestrator:\n" + context.strip()[:12000])
    parts.append(f"Plan (round {len(rounds) + 1}):\n{json.dumps(plan, indent=1)}")
    return "\n\n".join(parts)


def parse_verdict(text: str) -> Dict[str, Any]:
    """Astra's verdict: the last JSON object in its answer that carries one."""
    raw = str(text or "")
    data = None
    decoder = json.JSONDecoder()
    for start in [m.start() for m in re.finditer(r"\{", raw)][::-1]:
        try:
            candidate, _ = decoder.raw_decode(raw[start:])
        except ValueError:
            continue
        if isinstance(candidate, dict) and "verdict" in candidate:
            data = candidate
            break
    if data is None:
        raise PipelineError("the reviewer did not return a JSON verdict")
    verdict = str(data.get("verdict") or "").strip().lower()
    if verdict not in VERDICTS:
        raise PipelineError(f"the reviewer's verdict must be approve or changes, got {verdict!r}")
    findings = []
    for finding in (data.get("findings") or [])[:MAX_FINDINGS]:
        if isinstance(finding, dict) and str(finding.get("issue") or "").strip():
            findings.append({
                "item": str(finding.get("item") or "plan").strip()[:40],
                "issue": str(finding.get("issue")).strip()[:600],
                "fix": str(finding.get("fix") or "").strip()[:600],
            })
        elif isinstance(finding, str) and finding.strip():
            findings.append({"item": "plan", "issue": finding.strip()[:600], "fix": ""})
    if verdict == "changes" and not findings:
        raise PipelineError("the reviewer asked for changes without naming any")
    return {"verdict": verdict, "findings": findings if verdict == "changes" else []}


def decide_round(events: List[Dict[str, Any]], digest: str, verdict: Dict[str, Any],
                 *, will_answered: bool = False) -> Dict[str, str]:
    """approve, revise, or escalate to Will (no fixed round cap)."""
    if verdict["verdict"] == "approve":
        return {"action": "approve", "reason": "Astra approved the plan"}
    rounds = _plan_rounds(events)
    if not will_answered:
        previous = rounds[-1] if rounds else None
        if previous and previous.get("verdict") == "changes" and previous.get("digest") == digest:
            return {"action": "escalate",
                    "reason": "the plan came back unchanged and Astra still asks for changes"}
        earlier = [r.get("findings") or [] for r in rounds[-2:]] if len(rounds) >= 2 else []
        stuck = stuck_findings(earlier, verdict["findings"])
        if stuck:
            return {"action": "escalate",
                    "reason": "Astra has raised the same finding three rounds running: " + "; ".join(stuck[:3])}
    return {"action": "revise", "reason": f"Astra asked for {len(verdict['findings'])} change(s)"}


# ---------------------------------------------------------------------------
# State on the root card
# ---------------------------------------------------------------------------

def read_events(conn: Any, root_id: str) -> List[Dict[str, Any]]:
    from hermes_cli import kanban_db as kb

    events: List[Dict[str, Any]] = []
    for comment in kb.list_comments(conn, root_id):
        body = getattr(comment, "body", "") or ""
        if getattr(comment, "author", "") != STATE_AUTHOR or not body.startswith(STATE_PREFIX):
            continue
        try:
            event = json.loads(body[len(STATE_PREFIX):])
        except ValueError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def record(conn: Any, root_id: str, event: Dict[str, Any]) -> None:
    from hermes_cli import kanban_db as kb

    kb.add_comment(conn, root_id, STATE_AUTHOR, STATE_PREFIX + json.dumps(event, sort_keys=True))


def _start_event(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    for event in events:
        if event.get("type") == "start":
            return event
    raise PipelineError("this card is not a feature pipeline (no start record)")


def _approved_plan(events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for event in reversed(_plan_rounds(events)):
        return event.get("plan") if event.get("action") == "approve" else None
    return None


def _launched(events: List[Dict[str, Any]]) -> bool:
    return any(e.get("type") == "launch" for e in events)


def _work_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [e for e in events if e.get("type") in ("launch", "fix")]


def _work_cards(events: List[Dict[str, Any]]) -> List[str]:
    return [card for event in _work_events(events) for card in event.get("cards") or []]


def _open_cards(conn: Any, events: List[Dict[str, Any]]) -> List[str]:
    from hermes_cli import kanban_db as kb

    open_ids = []
    for card in _work_cards(events):
        task = kb.get_task(conn, card)
        if task is not None and task.status not in CLOSED_STATUSES:
            open_ids.append(card)
    return open_ids


def _last_marker(events: List[Dict[str, Any]]) -> Optional[int]:
    for index in range(len(events) - 1, -1, -1):
        if events[index].get("type") in MARKERS:
            return index
    return None


def open_answer(events: List[Dict[str, Any]]) -> str:
    """Will's answer to the root's current question, or ''."""
    index = _last_marker(events)
    if index is None or events[index].get("type") != "ask":
        return ""
    answers = [e.get("answer") for e in events[index + 1:]
               if e.get("type") == "will_answer" and not e.get("card") and e.get("answer")]
    return str(answers[-1]) if answers else ""


def phase(conn: Any, root_id: str, events: List[Dict[str, Any]]) -> str:
    """Where the pipeline is.

    planning, approved, building, fixing and integrating are the normal path.
    escalated: the plan or fix loop stopped converging and the orchestrator
    must ask Will. asked: it did and is waiting. answered: Will replied and
    the orchestrator acts on it. done: the root card is complete.
    """
    from hermes_cli import kanban_db as kb

    root = kb.get_task(conn, root_id)
    types = [e.get("type") for e in events]
    if "finish" in types or (root is not None and root.status == "done"):
        return "done"
    launched = "launch" in types
    if launched and _open_cards(conn, events):
        return "fixing" if _work_events(events)[-1].get("type") == "fix" else "building"
    index = _last_marker(events)
    if index is not None:
        last = events[index]
        if last.get("type") == "ask":
            if open_answer(events):
                return "answered"
            # Parking the card is what tells Will. A question whose card is not
            # parked (the asking run stopped first, or was reclaimed) has not
            # reached him: the orchestrator asks again.
            return "asked" if root is not None and root.status == "blocked" else "escalated"
        if last.get("type") == "fix_escalate" or (last.get("type") == "plan_round" and last.get("action") == "escalate"):
            return "escalated"
    if not launched:
        return "approved" if _approved_plan(events) is not None else "planning"
    return "integrating"


def _branch(task: Any, card: str) -> str:
    return (getattr(task, "branch_name", None) or "").strip() or f"wt/{card}"


def _expected_run(conn: Any, root_id: str, run_id: Optional[int]) -> Optional[int]:
    """The run a root mutation must belong to: the caller's when known.

    A worker passes its own dispatched run id, so a worker the dispatcher has
    since reclaimed cannot act for the run that replaced it.
    """
    from hermes_cli import kanban_db as kb

    root = kb.get_task(conn, root_id)
    if root is None:
        raise PipelineError(f"card {root_id} does not exist")
    if run_id is not None and (root.status != "running" or root.current_run_id is None
                               or int(root.current_run_id) != int(run_id)):
        raise PipelineError(f"run {run_id} is no longer card {root_id}'s current run; stop")
    return int(run_id) if run_id is not None else root.current_run_id


# ---------------------------------------------------------------------------
# Chat thread and boards
# ---------------------------------------------------------------------------

def chat_session(conversation_id: str) -> str:
    conversation_id = str(conversation_id or "").strip()
    if conversation_id.startswith("hermes-chat-"):
        return conversation_id
    return f"hermes-chat-{conversation_id}" if conversation_id else ""


def follow(conn: Any, task_id: str, conversation_id: str) -> None:
    """The chat follows the card: its finish, its questions and a stuck card reach Will."""
    from hermes_cli import kanban_db as kb

    session = chat_session(conversation_id)
    if session:
        kb.add_notify_sub(conn, task_id=task_id, platform="api_server", chat_id=session,
                          notifier_profile=WORKER_PROFILE, delivery_mode="notify+wake")


def post_note(conversation_id: str, text: str, *, board: str) -> Optional[int]:
    """A short progress note in the chat; it starts no agent turn."""
    from hermes_cli.kanban_discovery import post_report_to_session

    session = chat_session(conversation_id)
    if not session or not text.strip():
        return None
    return post_report_to_session(session, text.strip(), board=board)


def _project_roots() -> Dict[str, str]:
    path = Path.home() / ".hermes" / "subscription-projects.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if isinstance(data, dict) and isinstance(data.get("projects"), dict):
        data = data["projects"]
    return {str(k): str(v) for k, v in data.items() if isinstance(v, str)} if isinstance(data, dict) else {}


def resolve_board(project: str) -> str:
    """The project's board, created with the project's checkout as its working
    directory when it is missing one (region worktrees need a repo)."""
    from hermes_cli import kanban_db as kb

    project = str(project or "").strip()
    if not project:
        raise PipelineError("project is required to pick a board")
    boards = {b.get("slug"): b for b in kb.list_boards(include_archived=False)}
    board = boards.get(project)
    if board and board.get("default_workdir"):
        return project
    root = _project_roots().get(project)
    if not root:
        raise PipelineError(
            f"project {project!r} has no board with a working directory and is not in "
            "~/.hermes/subscription-projects.json"
        )
    if board:
        kb.write_board_metadata(project, default_workdir=root)
    else:
        kb.create_board(project, name=project, default_workdir=root)
    return project


def find_board(root_id: str) -> Optional[str]:
    """The board holding ``root_id``; None inside a worker, whose board is pinned."""
    from hermes_cli import kanban_db as kb

    if os.environ.get("HERMES_KANBAN_DB"):
        return None
    for meta in kb.list_boards(include_archived=False):
        slug = meta.get("slug")
        if not slug or not Path(kb.kanban_db_path(board=slug)).exists():
            continue
        with kb.connect_closing(board=slug) as conn:
            if kb.get_task(conn, root_id) is not None:
                return slug
    raise PipelineError(f"no board holds card {root_id}")


def _git(workdir: str, *args: str) -> str:
    if not workdir:
        return ""
    try:
        result = subprocess.run(["git", "-C", workdir, *args], capture_output=True, text=True, timeout=20,
                                check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _board_repo(board: Optional[str]) -> str:
    from hermes_cli import kanban_db as kb

    workdir = (kb.read_board_metadata(board or kb.get_current_board()).get("default_workdir") or "").strip()
    return _git(workdir, "rev-parse", "--show-toplevel") if workdir else ""


def pin_branch(repo: str, branch: str, commit: str) -> None:
    """Create ``branch`` at ``commit``, before the card that uses it exists.

    The dispatcher makes a card's worktree from the card's branch when that
    branch exists, and from the checkout's HEAD otherwise. Every pipeline card
    is created with a branch pinned here first, so it starts from the commit
    the plan was reviewed against even if the checkout moves meanwhile, and no
    card ever needs a staging state.
    """
    expected = _git(repo, "rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}") if repo and commit else ""
    if not expected:
        raise PipelineError(f"commit {commit[:12] or '(none)'} is not available in {repo or '(no repository)'}")
    ref = f"refs/heads/{branch}"
    if not _git(repo, "rev-parse", "--verify", "--quiet", ref):
        _git(repo, "branch", branch, expected)
    if _git(repo, "rev-parse", "--verify", "--quiet", ref) != expected:
        raise PipelineError(f"could not start branch {branch} at {expected[:12]} in {repo}")


def _card_for_key(conn: Any, key: str) -> Optional[str]:
    """The card an earlier, interrupted attempt made for ``key`` (create_task's own lookup)."""
    row = conn.execute(
        "SELECT id FROM tasks WHERE idempotency_key = ? AND status != 'archived' ORDER BY created_at DESC LIMIT 1",
        (key,),
    ).fetchone()
    return row["id"] if row else None


# ---------------------------------------------------------------------------
# Astra's plan review
# ---------------------------------------------------------------------------

def astra_review(prompt: str, *, cwd: str, timeout_seconds: int = REVIEW_TIMEOUT_SECONDS) -> Dict[str, Any]:
    """One read-only GPT-6.2 Astra turn in ``cwd`` on the ChatGPT subscription.

    The sandbox allows reads and read-only shell inspection and denies every
    change, even inside a kanban worker (agent/transports/codex_app_server.py
    lets an explicit read-only sandbox win over the worker default). MCP
    servers and plugins are off. There is no API-key fallback.
    """
    from agent.transports.codex_app_server_session import CodexAppServerSession

    session = CodexAppServerSession(
        cwd=cwd, model=REVIEWER[0], effort=EFFORT, read_only=True,
        require_exact=True, project_key="general", first_event_timeout=60,
    )
    try:
        result = session.run_turn(
            prompt, turn_timeout=REVIEW_INACTIVITY_SECONDS,
            absolute_turn_timeout=timeout_seconds, post_compaction_grace=0,
        )
    except Exception:
        raise PipelineError("Astra's review could not start; check the Codex login (no API fallback was tried)") from None
    finally:
        session.close()
    if result.interrupted or result.error:
        raise PipelineError("Astra's review failed (login, model access, quota or timeout); no API fallback was tried")
    return {"response": (result.final_text or "").strip(),
            "usage": result.token_usage_last or result.token_usage_total or {}}


def _review_dir(task: Any, board: Optional[str]) -> str:
    from hermes_cli import kanban_db as kb

    path = (getattr(task, "workspace_path", None) or "").strip()
    if path and Path(path).is_dir():
        return path
    workdir = (kb.read_board_metadata(board or kb.get_current_board()).get("default_workdir") or "").strip()
    if workdir and Path(workdir).is_dir():
        return workdir
    return tempfile.gettempdir()


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def _short(text: str, limit: int = 70) -> str:
    line = " ".join(str(text or "").split())
    return line if len(line) <= limit else line[: limit - 1].rstrip() + "…"


def _require(current: str, allowed: Sequence[str], action: str) -> None:
    if current not in allowed:
        hint = {
            "escalated": "the loop stopped converging: call ask with a question for Will first",
            "asked": "the pipeline is waiting on Will's answer",
            "building": "regions are still open",
            "fixing": "the fix card is still open",
            "done": "the pipeline is finished",
        }.get(current, f"it does not apply in phase {current}")
        raise PipelineError(f"{action}: {hint}")


def start(*, project: str, request: str, conversation_id: str, kind: str = "build") -> Dict[str, Any]:
    """Create the root card: an Opus orchestrator on the project's board."""
    from hermes_cli import kanban_db as kb

    kind = str(kind or "build").strip().lower()
    if kind not in KINDS:
        raise PipelineError("kind must be build or audit")
    request = str(request or "").strip()
    if not request:
        raise PipelineError("request is required: Will's words, in full")
    if len(request) > MAX_REQUEST_CHARS:
        raise PipelineError(f"the request is over {MAX_REQUEST_CHARS} characters; Astra must see all of it")
    if not kb.review_dispatch_enabled():
        raise PipelineError("the kanban review lane is off (kanban.review_dispatch), and every region needs Astra's review")
    board = resolve_board(project)
    workdir = (kb.read_board_metadata(board).get("default_workdir") or "").strip()
    if not workdir or not Path(workdir).is_absolute():
        raise PipelineError(f"project {project!r} needs an absolute checkout path, not {workdir!r}")
    repo = _board_repo(board)
    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    target = {"branch": "" if branch == "HEAD" else branch, "base": _git(repo, "rev-parse", "HEAD")}
    if not repo or not target["base"]:
        raise PipelineError(f"project {project!r} has no git checkout with a commit; region worktrees need one")
    if kind == "build" and not target["branch"]:
        raise PipelineError(f"project {project!r} is on a detached HEAD; a build needs a branch to land on "
                            "(an audit can run from it)")
    linked = _git(repo, "rev-parse", "--path-format=absolute", "--git-dir") != _git(
        repo, "rev-parse", "--path-format=absolute", "--git-common-dir")
    if linked or Path(workdir).resolve() != Path(repo).resolve():
        # The dispatcher makes per-card worktrees only from a main checkout's
        # top level; a linked worktree or a subdirectory would be reused as is.
        raise PipelineError(f"project {project!r} must point at its repository's main checkout ({workdir} is "
                            "a linked worktree or a subdirectory)")
    body = (
        f"{request}\n\n---\nFeature pipeline ({kind}). You are the Opus orchestrator. "
        "Follow the feature-pipeline skill; call feature_pipeline status first, it "
        "tells you which phase you are in."
    )
    token = secrets.token_hex(4)
    integration = f"fp/{token}/integration"
    pin_branch(repo, integration, target["base"])
    with kb.connect_closing(board=board) as conn:
        # The checkout is passed explicitly, for every card: the branch was
        # pinned in it, and a project-scoped board could otherwise route the
        # worktree to another clone.
        root_id = kb.create_task(
            conn, title=f"Feature pipeline ({kind}): {_short(request)}", body=body,
            assignee=WORKER_PROFILE, created_by=STATE_AUTHOR, workspace_kind="worktree",
            workspace_path=repo, branch_name=integration, skills=[ORCHESTRATOR_SKILL],
            model_override=ORCHESTRATOR[0], provider_override=ORCHESTRATOR[1], reasoning_effort=EFFORT,
            board=board,
        )
        record(conn, root_id, {
            "type": "start", "kind": kind, "project": project, "board": board, "token": token,
            "repo": repo, "conversation_id": str(conversation_id or ""), "request": request, "target": target,
        })
        follow(conn, root_id, conversation_id)
    return {"ok": True, "board": board, "root": root_id, "kind": kind,
            "next": "Tell Will it started. The Opus orchestrator writes the plan and Astra reviews it "
                    "before anything is built; notes arrive in this chat."}


def plan_review(*, root_id: str, plan: Any, context: str = "", board: Optional[str] = None,
                run_id: Optional[int] = None,
                reviewer: Optional[Callable[..., Dict[str, Any]]] = None) -> Dict[str, Any]:
    """One plan round: Astra's verdict, recorded, then approve, revise or escalate."""
    from hermes_cli import kanban_db as kb

    canonical = validate_plan(plan)
    digest = plan_digest(canonical)
    with kb.connect_closing(board=board) as conn:
        events = read_events(conn, root_id)
        begin = _start_event(events)
        current = phase(conn, root_id, events)
        if _launched(events):
            raise PipelineError("plan_review: the plan is launched; use fix for changes")
        _require(current, ("planning", "approved", "answered"), "plan_review")
        expected = _expected_run(conn, root_id, run_id)
        prompt = review_prompt(kind=begin.get("kind", "build"), request=begin.get("request", ""),
                               plan=canonical, events=events, context=context)
        cwd = _review_dir(kb.get_task(conn, root_id), begin.get("board") or board)
    answer = (reviewer or astra_review)(prompt, cwd=cwd)
    verdict = parse_verdict(answer.get("response", ""))
    decision = decide_round(events, digest, verdict, will_answered=current == "answered")
    measured = feature_system(canonical)
    with kb.connect_closing(board=board) as conn, kb.write_txn(conn):
        # Astra can take minutes: the run that asked must still own the card,
        # and nothing may have moved the pipeline on in the meantime. Checked
        # and appended in one immediate transaction, so an answer Will amends
        # meanwhile lands either before (and voids this round) or after.
        _expected_run(conn, root_id, expected)
        latest = read_events(conn, root_id)
        if len(latest) != len(events):
            raise PipelineError("plan_review: the pipeline changed while Astra was reviewing; call status, "
                                "then plan_review again")
        round_no = len(_plan_rounds(latest)) + 1
        record(conn, root_id, {
            "type": "plan_round", "round": round_no, "digest": digest,
            "verdict": verdict["verdict"], "findings": verdict["findings"],
            "action": decision["action"], "reason": decision["reason"],
            "plan": canonical if decision["action"] == "approve" else None,
            "feature_system": measured, "usage": answer.get("usage") or {},
        })
    if decision["action"] == "approve":
        regions = group_regions(canonical["items"])
        aspect = "" if measured["feature_system"] else " Measured on the plan, this is one aspect, not a feature system."
        post_note(begin.get("conversation_id", ""),
                  f"Feature pipeline: Astra approved the plan in round {round_no}: "
                  f"{len(canonical['items'])} items in {len(regions)} region(s). {canonical['summary']}{aspect}",
                  board=begin.get("board", board or ""))
    guidance = {
        "approve": "Call feature_pipeline launch.",
        "revise": "Revise the plan to settle every finding, then call plan_review again with the whole plan. "
                  "If you think a finding is wrong, say why in context instead of changing the plan.",
        "escalate": "Call feature_pipeline ask with one short question for Will that names the disputed finding "
                    "and both positions, then stop. His answer comes back to you.",
    }[decision["action"]]
    return {"ok": True, "round": round_no, "verdict": verdict["verdict"], "findings": verdict["findings"],
            "action": decision["action"], "reason": decision["reason"], "feature_system": measured,
            "next": guidance}


def _decision_lines(events: List[Dict[str, Any]]) -> List[str]:
    """Will's answers so far, for a worker's card: they override the request where they differ."""
    answered = decisions(events)
    if not answered:
        return []
    lines = ["", "Will's decisions since the request (they win where they differ from it):"]
    for pair in answered:
        lines += ([f"- Asked: {pair['question']}", f"  Will: {pair['answer']}"] if pair["question"]
                  else [f"- Will: {pair['answer']}"])
    return lines


def _region_body(kind: str, request: str, plan: Dict[str, Any], region: List[str], number: int, total: int,
                 events: Optional[List[Dict[str, Any]]] = None) -> str:
    by_id = {item["id"]: item for item in plan["items"]}
    lines = [f"Region {number} of {total} in a feature pipeline ({kind}). Will's request:",
             request.strip(), *_decision_lines(events or []), "", "Your items, in order:"]
    for item_id in region:
        item = by_id[item_id]
        lines += [f"- {item['id']}: {item['goal']}",
                  f"  files: {', '.join(item['files'])}",
                  f"  check: {item['check']}"]
        if item["depends_on"]:
            lines.append(f"  after: {', '.join(item['depends_on'])}")
    lines += ["", f"The whole plan: {plan['summary']}", "", "Follow the feature-pipeline-region skill."]
    if kind == "audit":
        lines.append("This is an audit: change no files. Report findings with evidence (file:line or command output).")
    return "\n".join(lines)


def _work_card(conn: Any, *, key: str, repo: str, branch: str, base: str, title: str, body: str, root: Any,
               conversation_id: str, board: Optional[str]) -> str:
    """An Opus card that Astra reviews and the root waits for.

    Safe to repeat: an interrupted attempt's card is found by its key and its
    remaining steps are idempotent. The root is linked last, so it is never
    left waiting on a card whose setup did not finish.
    """
    from hermes_cli import kanban_db as kb

    card = _card_for_key(conn, key)
    if card is None:
        pin_branch(repo, branch, base)
        card = kb.create_task(
            conn, title=title, body=body, assignee=WORKER_PROFILE, created_by=STATE_AUTHOR,
            workspace_kind="worktree", workspace_path=repo, branch_name=branch, skills=[REGION_SKILL],
            model_override=BUILDER[0], provider_override=BUILDER[1], reasoning_effort=EFFORT,
            review_model_override=REVIEWER[0], review_provider_override=REVIEWER[1],
            idempotency_key=key, tenant=getattr(root, "tenant", None), board=board,
        )
    else:
        kb.set_review_model_override(conn, card, REVIEWER[0], provider=REVIEWER[1])
    follow(conn, card, conversation_id)
    kb.link_tasks(conn, card, root.id)
    return card


def _park(conn: Any, root_id: str, reason: str, expected: Optional[int]) -> None:
    """The root waits on its open cards; the dispatcher wakes it when they are done."""
    from hermes_cli import kanban_db as kb

    root = kb.get_task(conn, root_id)
    if root is not None and root.status == "running":
        if not kb.block_task(conn, root_id, reason=reason, kind="dependency", expected_run_id=expected):
            raise PipelineError("the root card could not wait on its cards; call status and retry")


def launch(*, root_id: str, board: Optional[str] = None, run_id: Optional[int] = None) -> Dict[str, Any]:
    """One card per region (Opus builds, Astra reviews); the root waits for them."""
    from hermes_cli import kanban_db as kb

    with kb.connect_closing(board=board) as conn:
        events = read_events(conn, root_id)
        begin = _start_event(events)
        expected = _expected_run(conn, root_id, run_id)
        earlier = next((e for e in events if e.get("type") == "launch"), None)
        if earlier is not None:
            outstanding = (kb.get_task(conn, root_id).status == "running" and _open_cards(conn, [earlier])
                           and events[_last_marker(events)] is earlier)
            if not outstanding:
                raise PipelineError("launch: regions are already launched")
            cards = list(earlier.get("cards") or [])  # a launch that stopped before parking the root
            regions = list(earlier.get("regions") or [])
        else:
            _require(phase(conn, root_id, events), ("approved",), "launch")
            plan = _approved_plan(events)
            root = kb.get_task(conn, root_id)
            regions = group_regions(plan["items"])
            kind = begin.get("kind", "build")
            digest = plan_digest(plan)
            cards = []
            for number, region in enumerate(regions, 1):
                first = next(item for item in plan["items"] if item["id"] == region[0])
                cards.append(_work_card(
                    conn, key=f"{STATE_AUTHOR}:{root_id}:{digest}:region:{number}",
                    repo=begin.get("repo") or _board_repo(board),
                    branch=f"fp/{begin.get('token') or root_id}/region-{number}-{digest[:8]}",
                    base=(begin.get("target") or {}).get("base", ""),
                    title=f"Region {number}/{len(regions)}: {_short(first['goal'], 60)}",
                    body=_region_body(kind, begin.get("request", ""), plan, region, number, len(regions), events),
                    root=root, conversation_id=begin.get("conversation_id", ""), board=board,
                ))
            record(conn, root_id, {"type": "launch", "cards": cards, "regions": regions})
        _park(conn, root_id, f"waiting for {len(cards)} region(s): {', '.join(cards)}", expected)
    if earlier is None:
        post_note(begin.get("conversation_id", ""),
                  f"Feature pipeline: {len(cards)} region(s) started. Opus builds each one and Astra reviews it "
                  "before it counts as done.",
                  board=begin.get("board", board or ""))
    return {"ok": True, "regions": [{"card": c, "items": r} for c, r in zip(cards, regions)],
            "next": "Your run is over: reply with one line and stop. You wake when every region is done."}


def fix(*, root_id: str, findings: Any, board: Optional[str] = None, run_id: Optional[int] = None) -> Dict[str, Any]:
    """The one Opus fix card for this integration loop; the root waits for it."""
    from hermes_cli import kanban_db as kb

    if isinstance(findings, str):
        findings = [findings]
    findings = [str(f).strip() for f in (findings or []) if str(f).strip()]
    if not findings:
        raise PipelineError("findings are required: what the integration review found")
    with kb.connect_closing(board=board) as conn:
        events = read_events(conn, root_id)
        begin = _start_event(events)
        current = phase(conn, root_id, events)
        if not _launched(events):
            raise PipelineError("fix: nothing is built yet")
        expected = _expected_run(conn, root_id, run_id)
        loops = [e for e in events if e.get("type") == "fix"]
        root = kb.get_task(conn, root_id)
        last = loops[-1] if loops else None
        if last is not None and current == "fixing" and root.status == "running":
            # A fix that stopped before parking the root.
            _park(conn, root_id, f"waiting for integration fix {last['cards'][0]}", expected)
            return {"ok": True, "action": "fix", "card": last["cards"][0], "loop": last["loop"],
                    "next": "Your run is over: reply with one line and stop."}
        _require(current, ("integrating", "answered"), "fix")
        will = open_answer(events) if current == "answered" else ""
        if not will:
            earlier = [e.get("findings") or [] for e in loops[-2:]] if len(loops) >= 2 else []
            stuck = stuck_findings(earlier, findings)
            if stuck:
                record(conn, root_id, {"type": "fix_escalate", "findings": findings, "stuck": stuck})
                return {"ok": True, "action": "escalate",
                        "reason": "the same finding survived two fix loops: " + "; ".join(stuck[:3]),
                        "next": "Call feature_pipeline ask with one short question for Will, then stop. "
                                "His answer comes back to you."}
        loop = len(loops) + 1
        integration = _branch(root, root_id)
        decided = decisions(events)[-1] if will else None
        body = "\n".join([
            f"Integration fix {loop} in a feature pipeline ({begin.get('kind', 'build')}). Will's request:",
            begin.get("request", ""), *_decision_lines(events), "",
            f"Your branch starts from {integration}, the integrated result (merge it first if it is ahead), then fix:",
            *[f"- {f}" for f in findings],
            *(["", f"Will was asked: {decided['question']}", f"Will decided: {decided['answer']}"] if decided else []),
            "", "Follow the feature-pipeline-region skill.",
        ])
        repo = begin.get("repo") or _board_repo(board)
        card = _work_card(conn, key=f"{STATE_AUTHOR}:{root_id}:fix:{loop}", repo=repo,
                          branch=f"fp/{begin.get('token') or root_id}/fix-{loop}",
                          base=_git(repo, "rev-parse", "--verify", "--quiet", f"refs/heads/{integration}"),
                          title=f"Integration fix {loop}: {_short(findings[0], 60)}", body=body, root=root,
                          conversation_id=begin.get("conversation_id", ""), board=board)
        record(conn, root_id, {"type": "fix", "loop": loop, "cards": [card], "findings": findings})
        _park(conn, root_id, f"waiting for integration fix {card}", expected)
    return {"ok": True, "action": "fix", "card": card, "loop": loop,
            "next": "Your run is over: reply with one line and stop. You wake when the fix is reviewed and done."}


def ask(*, root_id: str, question: str, board: Optional[str] = None, run_id: Optional[int] = None) -> Dict[str, Any]:
    """The orchestrator's question for Will: recorded, and the root parked until he answers."""
    from hermes_cli import kanban_db as kb

    question = str(question or "").strip()
    if not question:
        raise PipelineError("question is required: one short question for Will")
    notice = QUESTION_PREFIX + " / ".join(line.strip() for line in question.splitlines() if line.strip())
    if len(notice) > MAX_NOTICE_CHARS:
        raise PipelineError(f"the question is too long for the chat notice ({len(notice)} of {MAX_NOTICE_CHARS} "
                            "characters with its prefix); ask one short question")
    with kb.connect_closing(board=board) as conn:
        events = read_events(conn, root_id)
        begin = _start_event(events)
        _require(phase(conn, root_id, events), ("planning", "approved", "escalated", "integrating", "answered"), "ask")
        expected = _expected_run(conn, root_id, run_id)
        record(conn, root_id, {"type": "ask", "question": question})
        # Parking the card is what tells the chat. Will answers every question
        # himself, so the unblock-loop breaker (built for a cron that keeps
        # unblocking) must not route a second question to triage. If this run
        # stops before parking, phase() reads the question as unasked and the
        # next run asks again.
        # One line that fits the digest, so the durable notice carries all of it.
        if not kb.block_task(conn, root_id, reason=notice, kind="needs_input",
                             expected_run_id=expected, count_recurrence=False):
            raise PipelineError("ask: the root card could not be parked for Will; call status")
    noted = post_note(begin.get("conversation_id", ""), f"Feature pipeline question for Will: {question}",
                      board=begin.get("board", board or ""))
    return {"ok": True, "noted": noted is not None,
            "next": "Your run is over: reply with one line and stop. You wake with his answer."}


def answer(*, root_id: str, answer: str, card: Optional[str] = None, board: Optional[str] = None) -> Dict[str, Any]:
    """Will's reply from the chat: to the root's question, or to a stuck region or fix card.

    The reply is recorded and the card it answers is unblocked, so its worker
    resumes with it.
    """
    from hermes_cli import kanban_db as kb

    text = str(answer or "").strip()
    if not text:
        raise PipelineError("answer is required: Will's words, verbatim")
    if len(text) > MAX_ANSWER_CHARS:
        # Never split it: a later answer replaces an earlier one.
        raise PipelineError(f"the answer is over {MAX_ANSWER_CHARS} characters; ask Will for one complete answer "
                            "within that length (the card stays parked until it comes)")
    with kb.connect_closing(board=board) as conn:
        events = read_events(conn, root_id)
        _start_event(events)
        if card:
            # Membership by the board's links: a card the root waits on is the
            # pipeline's, even if the launch that made it never got recorded.
            if card not in set(_work_cards(events)) | set(kb.parent_ids(conn, root_id)):
                raise PipelineError(f"card {card} is not one of this pipeline's region or fix cards")
            task = kb.get_task(conn, card)
            if task is not None and task.status == "triage":
                raise PipelineError(f"card {card} went to triage after repeated blocks; Will should look at it "
                                    "on the board")
            if task is None or task.status != "blocked":
                raise PipelineError(f"card {card} is not waiting on Will")
            # The current block's own reason, from its blocked event (a block
            # of a card that is not running opens no run). A block without a
            # reason gives the reply a slot of its own rather than an earlier
            # block's question.
            blocks = [e for e in kb.list_events(conn, card) if getattr(e, "kind", "") == "blocked"]
            asked = str(((getattr(blocks[-1], "payload", None) or {}).get("reason") or "")) if blocks else ""
            kb.add_comment(conn, card, STATE_AUTHOR, f"Will (through the chat): {text}")
            record(conn, root_id, {"type": "will_answer", "card": card, "question": asked, "answer": text})
            resumed = kb.unblock_task(conn, card)  # back to the lane it was blocked in
            return {"ok": True, "resumed": bool(resumed),
                    "next": "Tell Will his answer went to that card; its worker resumes with it."}
        with kb.write_txn(conn):
            current = phase(conn, root_id, read_events(conn, root_id))
            if current not in ("asked", "answered"):
                raise PipelineError(f"the pipeline is not waiting on Will (phase {current})")
            record(conn, root_id, {"type": "will_answer", "answer": text})
        task = kb.get_task(conn, root_id)
        resumed = bool(task is not None and task.status == "blocked" and kb.unblock_task(conn, root_id))
    return {"ok": True, "resumed": resumed,
            "next": "Tell Will his answer went to the pipeline; its orchestrator resumes with it."}


def finish(*, root_id: str, summary: str, board: Optional[str] = None, run_id: Optional[int] = None) -> Dict[str, Any]:
    """Record the result, tell Will in the chat, and complete the root card.

    Allowed after integration, or whenever Will's answer said to stop.
    """
    from hermes_cli import kanban_db as kb

    summary = str(summary or "").strip()
    if not summary:
        raise PipelineError("summary is required: what was built or found, where it landed, and the checks that passed")
    with kb.connect_closing(board=board) as conn:
        events = read_events(conn, root_id)
        begin = _start_event(events)
        current = phase(conn, root_id, events)
        if current != "answered" and not (_launched(events) and current == "integrating"):
            _require(current, ("integrating",) if _launched(events) else (), "finish")
        expected = _expected_run(conn, root_id, run_id)
        if not kb.complete_task(conn, root_id, result=summary[:8000], summary=_short(summary, 200),
                                expected_run_id=expected):
            raise PipelineError("finish: the root card could not be completed; a card may have reopened")
        record(conn, root_id, {"type": "finish", "summary": summary[:8000]})
    post_note(begin.get("conversation_id", ""), f"Feature pipeline finished. {summary}",
              board=begin.get("board", board or ""))
    return {"ok": True, "next": "Reply with one line and stop."}


NEXT_STEP = {
    "planning": "Write the plan (or revise it to settle the last round's findings), then call plan_review.",
    "approved": "Astra approved the plan: call feature_pipeline launch.",
    "escalated": "The loop stopped converging: call feature_pipeline ask with one short question for Will, then stop.",
    "asked": "Waiting on Will's answer: reply with one line and stop.",
    "answered": "Will answered (see decisions). Act on it: revise and call plan_review, or call fix or finish, "
                "as he said. The pipeline passes his words on to Astra and the fix card.",
    "building": "Regions are still open: reply with one line and stop.",
    "fixing": "The fix card is still open: reply with one line and stop.",
    "integrating": "Merge every branch in cards that you have not merged yet into your worktree, run every "
                   "item's check, and review the whole diff. Read each card's handoff for open questions; "
                   "ask Will if one needs him. Clean: land it on target.branch, then call finish. "
                   "Not clean: call fix.",
    "done": "The pipeline is finished.",
}


def status(*, root_id: str, board: Optional[str] = None) -> Dict[str, Any]:
    from hermes_cli import kanban_db as kb

    with kb.connect_closing(board=board) as conn:
        events = read_events(conn, root_id)
        begin = _start_event(events)
        current = phase(conn, root_id, events)
        cards = []
        for event in _work_events(events):
            for card in event.get("cards") or []:
                task = kb.get_task(conn, card)
                cards.append({"card": card, "kind": event["type"], "status": task.status if task else "missing",
                              "branch": _branch(task, card), "title": getattr(task, "title", "")})
        rounds = _plan_rounds(events)
        asked = [e for e in events if e.get("type") == "ask"]
    return {"ok": True, "phase": current, "kind": begin.get("kind"), "board": begin.get("board"),
            "conversation_id": begin.get("conversation_id"), "request": begin.get("request", ""),
            "target": begin.get("target") or {}, "rounds": len(rounds),
            "last_round": rounds[-1] if rounds else None, "approved_plan": _approved_plan(events),
            "cards": cards, "fix_loops": sum(1 for e in events if e.get("type") == "fix"),
            "question": asked[-1].get("question") if asked and current in ("asked", "answered") else None,
            "decisions": decisions(events), "next": NEXT_STEP[current]}
