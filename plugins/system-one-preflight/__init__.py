"""system-one-preflight — one calibrated Jev decision per turn, measured first.

TypeSafe's Jev never generates text; it answers typed questions about a piece
of state with calibrated probabilities in well under a second. This plugin
asks it exactly one question before each turn's first model call:

    Does completing the active request correctly require obtaining or
    checking material evidence that is currently missing?

and, depending on ``mode``, turns that answer into a fixed verification
reminder injected into the user message (the ``pre_llm_call`` context
channel). The reminder text never comes from Jev and never carries state, so
nothing retrieved from a tool can become an instruction through this path.

Modes (``plugins.entries.system-one-preflight.settings.mode``):

* ``shadow`` (default) — ask Jev, log the verdict, inject nothing.
* ``feedback`` — ask Jev about ambiguity and missing evidence; when either
               is high or Jev is unsure, feed the numbers back to the model
               in a fixed advisory note so it can ask one clarifying
               question or verify first. Nothing is enforced.
* ``jev``    — inject the reminder when P(missing verification) >= threshold.
* ``always`` — inject the reminder every turn (the always-remind control).
* ``trial``  — assign each session, by hash, to control / always / jev and
               behave accordingly, so a three-arm comparison can be run on
               whole task runs. Arm assignment is logged with every record.
* ``off``    — do nothing at all.

Every turn writes one JSON line to ``log_path`` (default
``$HERMES_HOME/logs/system-one-preflight.jsonl``) with the verdict, the arm,
whether a reminder went in, the latency, and, from ``post_llm_call``, what
the model then did (tool calls, tools used, response length). That log is
the raw material for the trial; grading task success stays a human job.

With ``tool_guard: feedback`` the guard runs synchronously and, when Jev is
confident an action is destructive or far-reaching AND outside what the user
asked for, holds it once with a note asking the model to confirm with the
user; the identical retry runs. In shadow it only logs. Either way, for ``terminal``,
``write_file`` and ``patch`` it asks Jev five yes/no questions about concrete
consequences and whether the action sits inside the user's established
scope, logs them, and never blocks. It runs off the tool's critical path in
a daemon thread, so it adds no latency.

Jev is bounded by ``timeout_seconds``; a slow or failed call is logged and
the turn proceeds without advice (the always arm still inserts its reminder).

The verify judge (``pre_verify``) turns Jev into a per-feature pass/fail gate
on the model's own work. ``post_tool_call`` remembers the session's todo
list (the model is nudged to write acceptance criteria there on build-type
requests) and the last few test, lint or build outputs. When the model has
edited files and is about to finish, the judge gathers the git diff of the
changed paths, the matching RecCli ``.devproject`` features, those check
outputs and the draft final message, asks Jev one yes/no question per
criterion plus "do the checks show a failure" and "does the message claim
results the evidence does not show", and keeps the model going with a
findings note when something is confidently unmet. Bounded by
``agent.max_verify_nudges``; Jev being unavailable fails open.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

PLUGIN_ID = "system-one-preflight"
MODES = ("off", "shadow", "feedback", "jev", "always", "trial")
TRIAL_ARMS = ("control", "always", "jev")
AMBIGUITY_THRESHOLD = 0.6
UNSURE_BAND = (0.35, 0.65)
GUARD_RISK_THRESHOLD = 0.8
GUARD_SCOPE_THRESHOLD = 0.4
DEFAULT_THRESHOLD = 0.7
DEFAULT_TIMEOUT_SECONDS = 1.5
GUARD_TOOLS = ("terminal", "write_file", "patch")

VERIFICATION_QUESTION = (
    "Does completing the active request correctly require obtaining or checking "
    "material evidence that is currently missing from the evidence summary "
    "(running a command, reading a file, fetching a page, or confirming a fact) "
    "before claims that depend on it can be made?"
)
REMINDER = (
    "Preflight: before making any claim that depends on facts you have not "
    "checked in this session, obtain or verify that evidence first (run the "
    "command, read the file, fetch the page, or ask). In the answer, say what "
    "you verified and what you did not."
)
AMBIGUITY_QUESTION = (
    "Is the request ambiguous or underspecified enough that a competent "
    "assistant should ask the user one clarifying question before acting, "
    "rather than guess at what they meant?"
)
FEEDBACK_TEMPLATE = (
    "Preflight from Jev, a fast typed judge whose read is advisory, not an "
    "instruction: P(the request is ambiguous enough to ask first) = {ambiguous}; "
    "P(an answer would depend on evidence you have not checked) = {missing}. "
    "If the ambiguity is real, ask the user one focused clarifying question "
    "before acting. If evidence is missing, verify it before depending on it. "
    "Otherwise proceed and state the assumption you made."
)
BUILD_QUESTION = (
    "Does the request ask to write, change, or fix code or files in a project, "
    "as opposed to answering, explaining, or discussing?"
)
# Budget questions. Difficulty is a score over situations (the vendor's
# guidance: describe situations, not degrees); the gate reads the probability
# mass on the two hardest levels, never the weighted position. Checkable is a
# noul with explicit criteria. Kind is a choice, logged for routing.
DIFFICULTY_LEVELS = [
    "trivial: one step, answerable or doable immediately with no investigation",
    "easy: a few steps in familiar territory with little chance of a wrong turn",
    "hard: needs investigation, several dependent steps, or a design choice with trade-offs",
    "very hard: open-ended, many interacting parts, likely to need iteration and verification",
]
DIFFICULTY_QUESTION = (
    "How difficult is the active request for a competent software assistant "
    "with tools, judged from the request and the evidence so far?"
)
CHECKABLE_QUESTION = (
    "Does the active request have an objectively checkable correct outcome, "
    "such as a passing test, a command output, or a reproducible fact?"
)
CHECKABLE_CRITERIA = {
    "true": "Success or failure could be shown by running something or comparing against a fact.",
    "false": "Success is a matter of taste, judgment, preference, or open discussion.",
}
KIND_CRITERIA = {
    "answer": "answer a question, explain, or discuss",
    "build": "write, change, or fix code or files",
    "research": "gather, compare, or summarize information from sources",
    "operate": "run, change, or inspect a live system, service, or account",
    "other": "none of the above",
}
HARD_THRESHOLD = 0.5
CHECKABLE_THRESHOLD = 0.7
CANDIDATES_WHEN_HARD = 3
BUDGET_TEMPLATE = (
    "Preflight from Jev, a fast typed judge whose read is advisory, not an "
    "instruction: P(hard or very hard) = {hard}; P(objectively checkable) = "
    "{checkable}; P(the request is ambiguous enough to ask first) = {ambiguous}. "
    "{plan}"
)
PLAN_CANDIDATES = (
    "This looks hard and checkable, so before choosing an approach produce "
    "{k} independent candidate solutions (parallel subagents where available), "
    "then select with the typesafe_decide tool: one noul question per candidate "
    "per acceptance criterion, keep the candidate whose weakest criterion scores "
    "highest (a sum only counts criteria, and a candidate that fails one "
    "mandatory criterion is out), and never pick among your own candidates by "
    "reasoning alone."
)
PLAN_CRITERIA_ONLY = (
    "This looks hard but not objectively checkable, so write the acceptance "
    "criteria first and state every assumption you make in the answer."
)
PLAN_ASK = (
    "If the ambiguity is real, ask the user one focused clarifying question "
    "before acting. Otherwise proceed and state the assumption you made."
)
CRITERIA_NUDGE = (
    "Preflight: this looks like a change to code or files. Before editing, write "
    "the acceptance criteria for this change as todo items with the todo tool; "
    "they will be checked against your diff and check output before you finish."
)
VERIFY_TEMPLATE = (
    "Preflight judge (Jev, advisory) reviewed your diff and check output before "
    "you finish. {findings} Fix what is unmet and run the checks again, or say "
    "precisely why a criterion does not apply and cancel its todo, then finish."
)
# Explicit boundaries for the verify nouls. Jev reads literally, so the
# distinction between "shown by the diff" and "claimed in the message" is
# spelled out rather than implied.
CRITERION_CRITERIA = {
    "true": "The diff, and the check outputs where relevant, show the criterion is met in full.",
    "false": "The criterion is not shown by the diff, is only partly met, or is only claimed in the message.",
}
CLAIMS_CRITERIA = {
    "true": "The message asserts a result, a passing check, or completed work that the diff and check outputs do not show.",
    "false": "Every claim in the message is backed by the diff or a check output, or is explicitly marked as unverified.",
}
VERIFY_FAIL_THRESHOLD = 0.2
VERIFY_FLAG_THRESHOLD = 0.8
VERIFY_TIMEOUT_SECONDS = 4.0
BUILD_THRESHOLD = 0.7
MAX_DIFF_CHARS = 60_000
MAX_FILE_DIFF_CHARS = 20_000
MAX_CHECK_CHARS = 2_000
CHECK_COMMAND_RE = re.compile(
    r"\b(pytest|npm (run )?(test|lint|build|typecheck)|pnpm (test|lint|build)|"
    r"yarn (test|lint|build)|vitest|jest|mocha|go test|cargo (test|check|clippy)|"
    r"make (test|check|lint)|node --test|tsc\b|eslint|ruff|mypy|flake8|"
    r"black --check|prettier --check)",
    re.IGNORECASE,
)
GUARD_FEEDBACK_TEMPLATE = (
    "Preflight from Jev before running {tool}: P(destructive or far-reaching: "
    "{risk_name}) = {risk}; P(within what the user asked for) = {scope}. This "
    "call was held once so you can confirm with the user first. Ask them a "
    "direct question naming the exact action, or explain why it is in scope, "
    "and then retry; the retry will run."
)
GUARD_QUESTIONS: Dict[str, str] = {
    "deletes_data": "Would executing this tool call delete, overwrite or truncate files or data?",
    "changes_privileges": "Would executing this tool call change permissions, credentials, users, keys or access controls?",
    "external_disclosure": "Would executing this tool call send data, secrets or messages outside this machine (network requests, email, publishing, pushing)?",
    "shared_system_change": "Would executing this tool call change a shared or production system (running services, databases, remote repositories, deployments, other machines)?",
    "within_scope": "Is this action within what the user asked for or authorized in the conversation so far?",
}

_FAILURE_RE = re.compile(
    r"(traceback|\berror\b|\bfailed\b|exit code [1-9]|command not found|no such file|permission denied)",
    re.IGNORECASE,
)

# Injectable seams for tests and for the register() wiring.
_settings_reader: Callable[[str, Any], Any] = lambda key, default=None: default
_ask: Optional[Callable[..., Dict[str, Any]]] = None
_log_lock = threading.Lock()
_turn_memo: Dict[str, Dict[str, Any]] = {}
_session_scope: Dict[str, List[str]] = {}
_session_todos: Dict[str, List[Dict[str, str]]] = {}
_session_checks: Dict[str, List[Dict[str, str]]] = {}
_session_commands: Dict[str, List[Dict[str, str]]] = {}
_verify_memo: Dict[str, str] = {}
MAX_COMMANDS_KEPT = 6
_MEMO_LIMIT = 256


def _bound(store: Dict[str, Any]) -> None:
    if len(store) > _MEMO_LIMIT:
        for key in list(store)[: len(store) - _MEMO_LIMIT]:
            store.pop(key, None)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def _setting(key: str, default: Any) -> Any:
    try:
        value = _settings_reader(key, default)
    except Exception:
        return default
    return default if value is None else value


def current_mode() -> str:
    mode = str(_setting("mode", "shadow") or "shadow").strip().lower()
    return mode if mode in MODES else "shadow"


def threshold() -> float:
    try:
        value = float(_setting("threshold", DEFAULT_THRESHOLD))
    except (TypeError, ValueError):
        return DEFAULT_THRESHOLD
    return min(1.0, max(0.0, value))


def timeout_seconds() -> float:
    try:
        value = float(_setting("timeout_seconds", DEFAULT_TIMEOUT_SECONDS))
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT_SECONDS
    return min(10.0, max(0.2, value))


def tool_guard_mode() -> str:
    mode = str(_setting("tool_guard", "shadow") or "shadow").strip().lower()
    return mode if mode in ("off", "shadow", "feedback") else "shadow"


def ambiguity_threshold() -> float:
    try:
        value = float(_setting("ambiguity_threshold", AMBIGUITY_THRESHOLD))
    except (TypeError, ValueError):
        return AMBIGUITY_THRESHOLD
    return min(1.0, max(0.0, value))


def verify_judge_enabled() -> bool:
    value = str(_setting("verify_judge", "on") or "on").strip().lower()
    return value not in ("off", "false", "0", "no")


def criteria_nudge_enabled() -> bool:
    value = str(_setting("criteria_nudge", "on") or "on").strip().lower()
    return value not in ("off", "false", "0", "no")


def verify_fail_threshold() -> float:
    try:
        value = float(_setting("verify_fail_threshold", VERIFY_FAIL_THRESHOLD))
    except (TypeError, ValueError):
        return VERIFY_FAIL_THRESHOLD
    return min(1.0, max(0.0, value))


def verify_timeout_seconds() -> float:
    try:
        value = float(_setting("verify_timeout_seconds", VERIFY_TIMEOUT_SECONDS))
    except (TypeError, ValueError):
        return VERIFY_TIMEOUT_SECONDS
    return min(15.0, max(0.5, value))


def log_path() -> Path:
    configured = _setting("log_path", "")
    if configured:
        return Path(str(configured)).expanduser()
    try:
        from hermes_constants import get_hermes_home

        home = Path(get_hermes_home())
    except Exception:
        home = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
    return home / "logs" / "system-one-preflight.jsonl"


def trial_arm(session_id: str) -> str:
    """Deterministic arm per session so a whole task run stays in one arm."""
    seed = str(_setting("trial_seed", "") or "")
    digest = hashlib.sha256(f"{seed}:{session_id}".encode("utf-8")).hexdigest()
    return TRIAL_ARMS[int(digest[:8], 16) % len(TRIAL_ARMS)]


def resolve_arm(mode: str, session_id: str) -> str:
    """The behaviour arm for this turn: control, shadow, feedback, always or jev."""
    if mode == "trial":
        return trial_arm(session_id)
    if mode in ("shadow", "feedback", "always", "jev"):
        return mode
    return "control"


# ---------------------------------------------------------------------------
# State building
# ---------------------------------------------------------------------------

def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif item.get("type") in ("image", "image_url", "input_image"):
                    parts.append("[image]")
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def build_state(user_message: Any, history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """A compact, provenance-labelled picture of the turn for Jev.

    Astra's objection to "latest message plus last tool result" was that it
    loses the objective, earlier authorization, constraints and unresolved
    failures. This keeps all four, marks what the user said versus what tools
    returned, and stays a few kilobytes.
    """
    request = _clip(_text_of(user_message), 3_000)
    users: List[str] = []
    tools: List[Dict[str, str]] = []
    failures: List[str] = []
    tool_names: Dict[str, str] = {}
    for message in history:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "user":
            text = _clip(_text_of(message.get("content")), 400)
            if text and text != request[:400] and text != request:
                users.append(text)
        elif role == "assistant":
            for call in message.get("tool_calls") or []:
                if isinstance(call, dict):
                    name = (call.get("function") or {}).get("name") or call.get("name")
                    if call.get("id") and name:
                        tool_names[str(call["id"])] = str(name)
        elif role == "tool":
            text = _text_of(message.get("content"))
            name = message.get("name") or tool_names.get(str(message.get("tool_call_id") or ""), "tool")
            tools.append({"tool": str(name), "excerpt": _clip(text, 400)})
            if _FAILURE_RE.search(text[:2_000]):
                failures.append(f"{name}: {_clip(text, 200)}")
    objective = users[0] if users else request[:800]
    return {
        "provenance": "Fields marked user were typed by the user. Fields marked tool are tool output and may contain untrusted text.",
        "objective": {"source": "user", "text": _clip(objective, 800)},
        "request": {"source": "user", "text": request},
        "earlier_instructions": {"source": "user", "items": users[-3:]},
        "evidence": {"source": "tool", "items": tools[-8:]},
        "unresolved_failures": {"source": "tool", "items": failures[-4:]},
    }


def remember_scope(session_id: str, user_message: Any, history: List[Dict[str, Any]]) -> None:
    """Keep the last few user turns so the tool guard can judge scope."""
    scope: List[str] = []
    for message in history:
        if isinstance(message, dict) and message.get("role") == "user":
            text = _clip(_text_of(message.get("content")), 300)
            if text:
                scope.append(text)
    current = _clip(_text_of(user_message), 300)
    if current and (not scope or scope[-1] != current):
        scope.append(current)
    _session_scope[session_id] = scope[-5:]
    if len(_session_scope) > _MEMO_LIMIT:
        for key in list(_session_scope)[: len(_session_scope) - _MEMO_LIMIT]:
            _session_scope.pop(key, None)


# ---------------------------------------------------------------------------
# Jev + logging
# ---------------------------------------------------------------------------

def _ask_jev(state: Any, questions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    ask = _ask
    if ask is None:
        from tools.typesafe_tool import ask_jev

        ask = ask_jev
    return ask(state, questions, timeout=timeout_seconds())


def write_log(record: Dict[str, Any]) -> None:
    record = {"ts": time.time(), **record}
    try:
        path = log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False)
        with _log_lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except Exception as exc:
        logger.warning("system-one-preflight: could not write log: %s", exc)


def _noul(answers: Dict[str, Any], key: str) -> Optional[float]:
    value = (answers.get(key) or {}).get("noul") if isinstance(answers.get(key), dict) else None
    return float(value) if isinstance(value, (int, float)) else None


def _level_mass(answers: Dict[str, Any], key: str, levels: tuple) -> Optional[float]:
    """Probability mass on the given score levels, or None without a distribution."""
    answer = answers.get(key)
    probabilities = answer.get("probabilities") if isinstance(answer, dict) else None
    if not isinstance(probabilities, dict) or not probabilities:
        return None
    total = 0.0
    for level in levels:
        value = probabilities.get(str(level), probabilities.get(level))
        if isinstance(value, (int, float)):
            total += float(value)
    return min(1.0, max(0.0, total))


def _choice(answers: Dict[str, Any], key: str) -> Optional[str]:
    answer = answers.get(key)
    value = answer.get("choice") if isinstance(answer, dict) else None
    return str(value) if isinstance(value, str) and value else None


def decide(p_missing: Optional[float], arm: str) -> bool:
    """Whether the reminder goes in for this arm and verdict."""
    if arm == "always":
        return True
    if arm == "jev":
        return p_missing is not None and p_missing >= threshold()
    return False


def _unsure(value: Optional[float]) -> bool:
    return value is not None and UNSURE_BAND[0] <= value <= UNSURE_BAND[1]


def feedback_context(p_missing: Optional[float], p_ambiguous: Optional[float]) -> Optional[str]:
    """The advisory note for feedback mode, or None when Jev sees no issue.

    Fed back when either signal clears its threshold or sits in the unsure
    band, so the model gets Jev's read exactly when a clarifying question or
    a check is most likely to pay off. Numbers only; the wording is fixed.
    """
    if p_missing is None and p_ambiguous is None:
        return None
    worth_it = (
        (p_ambiguous is not None and p_ambiguous >= ambiguity_threshold())
        or (p_missing is not None and p_missing >= threshold())
        or _unsure(p_ambiguous)
        or _unsure(p_missing)
    )
    if not worth_it:
        return None
    fmt = lambda value: "unknown" if value is None else f"{value:.2f}"
    return FEEDBACK_TEMPLATE.format(ambiguous=fmt(p_ambiguous), missing=fmt(p_missing))


def _fmt(value: Optional[float]) -> str:
    return "unknown" if value is None else f"{value:.2f}"


def budget(p_hard: Optional[float], p_checkable: Optional[float]) -> Dict[str, Any]:
    """Map the budget answers to a plan: how many candidates, and whether the finish loop applies.

    Reads the probability mass on the two hardest levels, never the weighted
    score (the vendor documents score as threshold-passage only).
    """
    hard = p_hard is not None and p_hard >= HARD_THRESHOLD
    checkable = p_checkable is not None and p_checkable >= CHECKABLE_THRESHOLD
    if hard and checkable:
        return {"k": CANDIDATES_WHEN_HARD, "finish_loop": True, "plan": "candidates"}
    if hard:
        return {"k": 1, "finish_loop": False, "plan": "criteria_only"}
    return {"k": 1, "finish_loop": True, "plan": "direct"}


def budget_context(
    p_ambiguous: Optional[float],
    p_hard: Optional[float],
    p_checkable: Optional[float],
    plan: Dict[str, Any],
) -> Optional[str]:
    """The feedback note built from the budget, or None when there is nothing to say.

    The old trigger, "would an answer depend on evidence not yet checked",
    averaged 0.81 over 189 real turns and so carried almost no information.
    The note now goes in when the budget calls for candidates or criteria, or
    when the ambiguity read clears its threshold or sits in the unsure band.
    """
    ask = (p_ambiguous is not None and p_ambiguous >= ambiguity_threshold()) or _unsure(p_ambiguous)
    parts: List[str] = []
    if plan.get("plan") == "candidates":
        parts.append(PLAN_CANDIDATES.format(k=plan.get("k", CANDIDATES_WHEN_HARD)))
    elif plan.get("plan") == "criteria_only":
        parts.append(PLAN_CRITERIA_ONLY)
    if ask:
        parts.append(PLAN_ASK)
    if not parts:
        return None
    return BUDGET_TEMPLATE.format(
        hard=_fmt(p_hard), checkable=_fmt(p_checkable), ambiguous=_fmt(p_ambiguous), plan=" ".join(parts)
    )


def emit_verdict(
    session_id: str,
    stage: str,
    text: str,
    *,
    answers: Dict[str, Any],
    decision: Dict[str, Any],
    model: str = "",
    latency_ms: int = 0,
    attempt: Optional[int] = None,
) -> bool:
    """Put the verdict on the run stream so the user sees it in the turn and the archive keeps it."""
    try:
        from hermes_cli.turn_events import emit_turn_event
    except Exception:
        return False
    payload: Dict[str, Any] = {
        "stage": stage,
        "answers": answers,
        "decision": decision,
        "model": model,
        "latency_ms": latency_ms,
    }
    if attempt is not None:
        payload["attempt"] = attempt
    return emit_turn_event(session_id, "judge.verdict", text=text, source="jev", **payload)


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

def on_pre_llm_call(**kwargs: Any) -> Optional[Dict[str, str]]:
    mode = current_mode()
    if mode == "off":
        return None
    session_id = str(kwargs.get("session_id") or "")
    turn_id = str(kwargs.get("turn_id") or "")
    memo_key = f"{session_id}:{turn_id}"
    if turn_id and memo_key in _turn_memo:
        # The hook can fire more than once for one user turn; Jev's answer
        # and the injection decision must not change within the turn.
        memo = _turn_memo[memo_key]
        if memo.get("context"):
            return {"context": memo["context"]}
        return {"context": REMINDER} if memo.get("injected") else None

    history = kwargs.get("conversation_history") or []
    user_message = kwargs.get("user_message")
    remember_scope(session_id, user_message, history if isinstance(history, list) else [])
    arm = resolve_arm(mode, session_id)
    state = build_state(user_message, history if isinstance(history, list) else [])
    started = time.monotonic()
    p_missing: Optional[float] = None
    p_ambiguous: Optional[float] = None
    p_build: Optional[float] = None
    p_hard: Optional[float] = None
    p_checkable: Optional[float] = None
    kind: Optional[str] = None
    jev_model = ""
    error = ""
    questions: Dict[str, Dict[str, Any]] = {
        "missing_verification": {"type": "noul", "instructions": VERIFICATION_QUESTION}
    }
    if arm == "feedback":
        questions["ambiguous"] = {"type": "noul", "instructions": AMBIGUITY_QUESTION}
        questions["is_build"] = {"type": "noul", "instructions": BUILD_QUESTION}
        questions["difficulty"] = {"type": "score", "instructions": DIFFICULTY_QUESTION, "criteria": DIFFICULTY_LEVELS}
        questions["checkable"] = {"type": "noul", "instructions": CHECKABLE_QUESTION, "criteria": CHECKABLE_CRITERIA}
        questions["kind"] = {"type": "choice", "instructions": "What kind of request is the active request?", "criteria": KIND_CRITERIA}
    try:
        body = _ask_jev(state, questions)
        answers = body.get("answers") or {}
        jev_model = str(body.get("model") or "")
        p_missing = _noul(answers, "missing_verification")
        p_ambiguous = _noul(answers, "ambiguous")
        p_build = _noul(answers, "is_build")
        p_hard = _level_mass(answers, "difficulty", (2, 3))
        p_checkable = _noul(answers, "checkable")
        kind = _choice(answers, "kind")
        if p_missing is None:
            error = "no noul in answer"
    except Exception as exc:
        error = f"{exc.__class__.__name__}: {exc}"[:300]
    latency_ms = int((time.monotonic() - started) * 1000)
    context: Optional[str] = None
    plan = budget(p_hard, p_checkable)
    if arm == "feedback":
        context = budget_context(p_ambiguous, p_hard, p_checkable, plan)
        if (
            criteria_nudge_enabled()
            and verify_judge_enabled()
            and p_build is not None
            and p_build >= BUILD_THRESHOLD
            and not _session_todos.get(session_id)
        ):
            context = f"{context}\n\n{CRITERIA_NUDGE}" if context else CRITERIA_NUDGE
        injected = context is not None
    else:
        injected = decide(p_missing, arm)
        context = REMINDER if injected else None
    write_log(
        {
            "event": "preflight",
            "session_id": session_id,
            "turn_id": turn_id,
            "platform": str(kwargs.get("platform") or ""),
            "model": str(kwargs.get("model") or ""),
            "mode": mode,
            "arm": arm,
            "p_missing": p_missing,
            "p_ambiguous": p_ambiguous,
            "p_build": p_build,
            "p_hard": p_hard,
            "p_checkable": p_checkable,
            "kind": kind,
            "k": plan["k"],
            "threshold": threshold(),
            "injected": injected,
            "latency_ms": latency_ms,
            "state_chars": len(json.dumps(state, ensure_ascii=False)),
            "evidence_items": len(state["evidence"]["items"]),
            "failures": len(state["unresolved_failures"]["items"]),
            "error": error,
        }
    )
    if arm == "feedback" and not error:
        emit_verdict(
            session_id,
            "budget",
            f"Jev budget: hard {_fmt(p_hard)} · checkable {_fmt(p_checkable)} · kind {kind or 'unknown'} · "
            f"ambiguous {_fmt(p_ambiguous)} · build {_fmt(p_build)} · k={plan['k']}"
            + (" · note sent" if injected else ""),
            answers={
                "hard": p_hard,
                "checkable": p_checkable,
                "kind": kind,
                "ambiguous": p_ambiguous,
                "build": p_build,
                "missing": p_missing,
            },
            decision={"k": plan["k"], "finish_loop": plan["finish_loop"], "plan": plan["plan"], "injected": injected},
            model=jev_model,
            latency_ms=latency_ms,
        )
    if turn_id:
        _turn_memo[memo_key] = {
            "injected": injected,
            "p_missing": p_missing,
            "p_ambiguous": p_ambiguous,
            "p_hard": p_hard,
            "p_checkable": p_checkable,
            "kind": kind,
            "k": plan["k"],
            "arm": arm,
            "context": context,
        }
        if len(_turn_memo) > _MEMO_LIMIT:
            for key in list(_turn_memo)[: len(_turn_memo) - _MEMO_LIMIT]:
                _turn_memo.pop(key, None)
    return {"context": context} if context else None


def on_post_llm_call(**kwargs: Any) -> None:
    if current_mode() == "off":
        return None
    session_id = str(kwargs.get("session_id") or "")
    turn_id = str(kwargs.get("turn_id") or "")
    history = kwargs.get("conversation_history") or []
    user_message = _text_of(kwargs.get("user_message"))
    # Only count what happened after this turn's user message.
    start = 0
    if isinstance(history, list):
        for index in range(len(history) - 1, -1, -1):
            message = history[index]
            if isinstance(message, dict) and message.get("role") == "user" and _text_of(message.get("content")) == user_message:
                start = index
                break
    tool_calls = 0
    tools: List[str] = []
    if isinstance(history, list):
        for message in history[start:]:
            if isinstance(message, dict) and message.get("role") == "assistant":
                for call in message.get("tool_calls") or []:
                    if isinstance(call, dict):
                        tool_calls += 1
                        name = (call.get("function") or {}).get("name") or call.get("name")
                        if name:
                            tools.append(str(name))
    memo = _turn_memo.get(f"{session_id}:{turn_id}", {})
    write_log(
        {
            "event": "turn_end",
            "session_id": session_id,
            "turn_id": turn_id,
            "arm": memo.get("arm"),
            "p_missing": memo.get("p_missing"),
            "p_ambiguous": memo.get("p_ambiguous"),
            "p_hard": memo.get("p_hard"),
            "p_checkable": memo.get("p_checkable"),
            "kind": memo.get("kind"),
            "k": memo.get("k"),
            "injected": memo.get("injected"),
            "tool_calls": tool_calls,
            "tools": sorted(set(tools)),
            "response_chars": len(str(kwargs.get("assistant_response") or "")),
        }
    )
    return None


def _guard_state(tool_name: str, args: Dict[str, Any], scope: List[str]) -> Dict[str, Any]:
    action: Dict[str, Any] = {"tool": tool_name}
    if tool_name == "terminal":
        action["command"] = _clip(str(args.get("command") or ""), 2_000)
    else:
        action["path"] = _clip(str(args.get("path") or ""), 300)
        content = args.get("content") if tool_name == "write_file" else (args.get("new_string") or args.get("patch") or "")
        action["content_excerpt"] = _clip(str(content or ""), 600)
    return {
        "provenance": "user_scope was typed by the user. action is what the agent is about to execute.",
        "user_scope": {"source": "user", "items": scope},
        "action": {"source": "agent", "value": action},
    }


_held_actions: Dict[str, float] = {}


def _action_key(session_id: str, tool_name: str, args: Dict[str, Any]) -> str:
    digest = hashlib.sha256(json.dumps(args, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()[:16]
    return f"{session_id}:{tool_name}:{digest}"


def guard_feedback(tool_name: str, answers: Dict[str, Optional[float]]) -> Optional[str]:
    """The hold message when Jev is confident an action is risky and out of scope."""
    scope = answers.get("within_scope")
    risks = {
        key: value
        for key, value in answers.items()
        if key != "within_scope" and value is not None and value >= GUARD_RISK_THRESHOLD
    }
    if not risks or scope is None or scope > GUARD_SCOPE_THRESHOLD:
        return None
    risk_name, risk = max(risks.items(), key=lambda item: item[1])
    return GUARD_FEEDBACK_TEMPLATE.format(
        tool=tool_name,
        risk_name=risk_name.replace("_", " "),
        risk=f"{risk:.2f}",
        scope=f"{scope:.2f}",
    )


def _run_guard(tool_name: str, args: Dict[str, Any], session_id: str, turn_id: str, tool_call_id: str) -> Dict[str, Optional[float]]:
    scope = list(_session_scope.get(session_id, []))
    state = _guard_state(tool_name, args, scope)
    started = time.monotonic()
    answers: Dict[str, Optional[float]] = {}
    error = ""
    try:
        body = _ask_jev(
            state,
            {key: {"type": "noul", "instructions": question} for key, question in GUARD_QUESTIONS.items()},
        )
        for key in GUARD_QUESTIONS:
            value = ((body.get("answers") or {}).get(key) or {}).get("noul")
            answers[key] = float(value) if isinstance(value, (int, float)) else None
    except Exception as exc:
        error = f"{exc.__class__.__name__}: {exc}"[:300]
    write_log(
        {
            "event": "tool_guard",
            "session_id": session_id,
            "turn_id": turn_id,
            "tool_call_id": tool_call_id,
            "tool": tool_name,
            "answers": answers,
            "latency_ms": int((time.monotonic() - started) * 1000),
            "scope_items": len(scope),
            "error": error,
        }
    )
    return answers


def on_pre_tool_call(**kwargs: Any) -> Optional[Dict[str, str]]:
    """Shadow: classify off the critical path. Feedback: hold a risky,
    out-of-scope action once with a note the model answers by asking the user."""
    if current_mode() == "off" or tool_guard_mode() == "off":
        return None
    tool_name = str(kwargs.get("tool_name") or "")
    if tool_name not in GUARD_TOOLS:
        return None
    args = kwargs.get("args")
    if not isinstance(args, dict):
        args = kwargs.get("tool_input") if isinstance(kwargs.get("tool_input"), dict) else {}
    session_id = str(kwargs.get("session_id") or "")
    if tool_guard_mode() == "feedback":
        key = _action_key(session_id, tool_name, dict(args))
        if key in _held_actions:
            # The model came back after the hold; let the retry run.
            _held_actions.pop(key, None)
            write_log({"event": "tool_guard_release", "session_id": session_id, "tool": tool_name})
            return None
        answers = _run_guard(
            tool_name, dict(args), session_id,
            str(kwargs.get("turn_id") or ""), str(kwargs.get("tool_call_id") or ""),
        )
        message = guard_feedback(tool_name, answers)
        if message is None:
            return None
        _held_actions[key] = time.time()
        if len(_held_actions) > _MEMO_LIMIT:
            for stale in list(_held_actions)[: len(_held_actions) - _MEMO_LIMIT]:
                _held_actions.pop(stale, None)
        write_log({"event": "tool_guard_hold", "session_id": session_id, "tool": tool_name, "answers": answers})
        return {"action": "block", "message": message}
    worker = threading.Thread(
        target=_run_guard,
        args=(
            tool_name,
            dict(args),
            session_id,
            str(kwargs.get("turn_id") or ""),
            str(kwargs.get("tool_call_id") or ""),
        ),
        name="system-one-preflight-guard",
        daemon=True,
    )
    worker.start()
    return None


# ---------------------------------------------------------------------------
# Verify judge: per-criterion pass/fail on the diff before the turn finishes
# ---------------------------------------------------------------------------

def _tail(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else "…" + text[-(limit - 1):]


def parse_todos(text: str) -> Optional[List[Dict[str, str]]]:
    """The todo tool's result is JSON with a ``todos`` array; keep id/content/status."""
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return None
    items = payload.get("todos") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return None
    todos: List[Dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        todos.append(
            {
                "id": str(item.get("id") or len(todos) + 1),
                "content": _clip(content, 400),
                "status": str(item.get("status") or "pending"),
            }
        )
    return todos


def on_post_tool_call(**kwargs: Any) -> None:
    """Remember the session's todo list and its latest check outputs."""
    if current_mode() == "off":
        return None
    session_id = str(kwargs.get("session_id") or "")
    tool_name = str(kwargs.get("tool_name") or "")
    args = kwargs.get("args") if isinstance(kwargs.get("args"), dict) else {}
    result = kwargs.get("result")
    if isinstance(result, str):
        text = result
    elif result is None:
        text = ""
    else:
        try:
            text = json.dumps(result, ensure_ascii=False, default=str)
        except Exception:
            text = str(result)
    if tool_name == "todo":
        todos = parse_todos(text)
        if todos is not None:
            _session_todos[session_id] = todos
            _bound(_session_todos)
    elif tool_name == "terminal":
        command = str(args.get("command") or "")
        entry = {"command": _clip(command, 300), "output": _tail(text, MAX_CHECK_CHARS)}
        # Every command is evidence the judge may need (a script run, a curl,
        # a file listing); the check subset is what "do the checks fail" asks about.
        commands = _session_commands.setdefault(session_id, [])
        commands.append(entry)
        del commands[:-MAX_COMMANDS_KEPT]
        _bound(_session_commands)
        if CHECK_COMMAND_RE.search(command):
            checks = _session_checks.setdefault(session_id, [])
            checks.append(dict(entry))
            del checks[:-3]
            _bound(_session_checks)
    return None


def _git(root: str, args: List[str]) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", "-C", root, *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def _git_root(path: str) -> Optional[str]:
    directory = path if os.path.isdir(path) else os.path.dirname(path) or "."
    root = _git(directory, ["rev-parse", "--show-toplevel"])
    return root.strip() if root and root.strip() else None


def _read_file(path: str, limit: int) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read(limit + 1)
    except OSError:
        return ""
    return text if len(text) <= limit else text[:limit] + "\n… [truncated]"


def matching_features(root: Optional[str], relative_paths: List[str]) -> List[Dict[str, str]]:
    """Feature titles and descriptions from the RecCli project map whose files overlap the change."""
    if not root:
        return []
    try:
        candidates = [name for name in os.listdir(root) if name.endswith(".devproject")]
    except OSError:
        return []
    changed = set(relative_paths)
    found: List[Dict[str, str]] = []
    for name in candidates[:1]:
        try:
            with open(os.path.join(root, name), "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            continue
        for feature in payload.get("features") or []:
            if not isinstance(feature, dict):
                continue
            files = {str(item) for item in (feature.get("files_touched") or [])}
            if files & changed:
                found.append(
                    {
                        "title": _clip(str(feature.get("title") or ""), 120),
                        "description": _clip(str(feature.get("description") or ""), 400),
                        "status": str(feature.get("status") or ""),
                    }
                )
            if len(found) >= 5:
                break
    return found


def collect_evidence(changed_paths: List[str]) -> Dict[str, Any]:
    """The diff for the changed paths (git when possible), bounded, plus feature context."""
    paths = [str(path) for path in changed_paths if path]
    root = _git_root(paths[0]) if paths else None
    parts: List[str] = []
    total = 0
    truncated = False
    relative: List[str] = []
    for path in paths[:40]:
        if root and os.path.abspath(path).startswith(root + os.sep):
            rel = os.path.relpath(path, root)
            relative.append(rel)
            diff = _git(root, ["diff", "HEAD", "--", rel]) or ""
            if not diff.strip() and _git(root, ["ls-files", "--error-unmatch", rel]) is None:
                diff = f"+++ new file: {rel}\n" + _read_file(path, MAX_FILE_DIFF_CHARS)
        else:
            diff = f"+++ file: {path}\n" + _read_file(path, MAX_FILE_DIFF_CHARS)
        if len(diff) > MAX_FILE_DIFF_CHARS:
            diff = diff[:MAX_FILE_DIFF_CHARS] + "\n… [truncated]"
            truncated = True
        if total + len(diff) > MAX_DIFF_CHARS:
            truncated = True
            break
        total += len(diff)
        parts.append(diff)
    return {
        "root": root,
        "diff": "\n".join(parts),
        "truncated": truncated,
        "features": matching_features(root, relative),
    }


def on_pre_verify(**kwargs: Any) -> Optional[Dict[str, str]]:
    """Judge the finished change against its acceptance criteria before the turn ends."""
    if current_mode() == "off" or not verify_judge_enabled():
        return None
    session_id = str(kwargs.get("session_id") or "")
    changed = [str(path) for path in (kwargs.get("changed_paths") or []) if path]
    if not changed:
        return None
    attempt = int(kwargs.get("attempt") or 0)
    final_response = str(kwargs.get("final_response") or "")
    todos = list(_session_todos.get(session_id, []))
    criteria = [item for item in todos if item.get("status") in ("completed", "in_progress")]
    pending = [item for item in todos if item.get("status") == "pending"]
    request = (_session_scope.get(session_id) or [""])[-1]
    evidence = collect_evidence(changed)
    checks = list(_session_checks.get(session_id, []))
    commands = list(_session_commands.get(session_id, []))

    labels: Dict[str, str] = {}
    questions: Dict[str, Dict[str, Any]] = {}
    if criteria:
        for index, item in enumerate(criteria, 1):
            key = f"criterion_{index}"
            labels[key] = item["content"]
            questions[key] = {
                "type": "noul",
                "instructions": f"Do the code changes fully satisfy this acceptance criterion: {item['content']}",
                "criteria": CRITERION_CRITERIA,
            }
    elif request:
        labels["criterion_1"] = request
        questions["criterion_1"] = {
            "type": "noul",
            "instructions": f"Do the code changes fully satisfy the user's request: {_clip(request, 400)}",
            "criteria": CRITERION_CRITERIA,
        }
    if checks:
        questions["checks_failing"] = {
            "type": "noul",
            "instructions": "Do the check outputs show a failing test, an error, or a lint or type problem that was not fixed afterwards?",
        }
    questions["claims_unverified"] = {
        "type": "noul",
        "instructions": "Does the final message claim work, results, or passing checks that the diff and check outputs do not show?",
        "criteria": CLAIMS_CRITERIA,
    }
    state = {
        "provenance": "request was typed by the user. acceptance_criteria and still_pending_todos come from the agent's own todo list. diff and check_outputs are evidence. final_message is what the agent is about to say.",
        "request": {"source": "user", "text": _clip(request, 1_500)},
        "acceptance_criteria": {"source": "agent todo list", "items": [{"id": key, "text": text} for key, text in labels.items()]},
        "still_pending_todos": {"source": "agent todo list", "items": [_clip(item["content"], 200) for item in pending]},
        "features": {"source": "project map", "items": evidence["features"]},
        "diff": {"source": "git", "text": evidence["diff"], "truncated": evidence["truncated"]},
        "commands_run": {"source": "tool", "items": commands},
        "check_outputs": {"source": "tool", "items": checks},
        "final_message": {"source": "agent", "text": _clip(final_response, 3_000)},
    }
    started = time.monotonic()
    answers: Dict[str, Optional[float]] = {}
    error = ""
    jev_model = ""
    try:
        ask = _ask
        if ask is None:
            from tools.typesafe_tool import ask_jev

            ask = ask_jev
        body = ask(state, questions, timeout=verify_timeout_seconds())
        jev_model = str(body.get("model") or "")
        for key in questions:
            value = ((body.get("answers") or {}).get(key) or {}).get("noul")
            answers[key] = float(value) if isinstance(value, (int, float)) else None
    except Exception as exc:
        error = f"{exc.__class__.__name__}: {exc}"[:300]

    findings: List[str] = []
    unmet = [
        (labels[key], answers.get(key))
        for key in labels
        if answers.get(key) is not None and answers[key] <= verify_fail_threshold()
    ]
    if unmet:
        findings.append(
            "Criteria rated unmet: "
            + "; ".join(f'"{_clip(text, 120)}" (P(satisfied)={value:.2f})' for text, value in unmet)
            + "."
        )
    if pending:
        findings.append(f"Todo items still pending: {len(pending)}.")
    checks_failing = answers.get("checks_failing")
    if checks_failing is not None and checks_failing >= VERIFY_FLAG_THRESHOLD:
        findings.append(f"Check outputs show a failure (P={checks_failing:.2f}).")
    claims = answers.get("claims_unverified")
    if claims is not None and claims >= VERIFY_FLAG_THRESHOLD:
        findings.append(f"The final message claims results the evidence does not show (P={claims:.2f}).")
    # A nudge that changed nothing must not be repeated: if the findings and
    # the evidence are identical to the previous attempt, the model has
    # answered them as far as it will, so let the turn finish.
    evidence_key = hashlib.sha256(
        json.dumps({"findings": findings, "diff": evidence["diff"], "commands": commands}, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    repeated = attempt > 0 and _verify_memo.get(session_id) == evidence_key
    _verify_memo[session_id] = evidence_key
    _bound(_verify_memo)
    write_log(
        {
            "event": "verify",
            "session_id": session_id,
            "attempt": attempt,
            "repeated": repeated,
            "changed_paths": len(changed),
            "criteria": len(labels),
            "pending": len(pending),
            "checks": len(checks),
            "commands": len(commands),
            "features": len(evidence["features"]),
            "diff_chars": len(evidence["diff"]),
            "answers": answers,
            "findings": findings,
            "latency_ms": int((time.monotonic() - started) * 1000),
            "error": error,
        }
    )
    action = "unavailable" if error else "finish" if not findings else "repeated" if repeated else "nudge"
    met = sum(1 for key in labels if answers.get(key) is not None and answers[key] > verify_fail_threshold())
    emit_verdict(
        session_id,
        "verify",
        f"Jev verify (attempt {attempt + 1}): criteria met {met}/{len(labels)} · "
        f"claims unverified {_fmt(claims)} · checks failing {_fmt(checks_failing)} · {action}"
        + (f" · {' '.join(findings)}" if findings and action == 'nudge' else ""),
        answers={**{labels[key]: answers.get(key) for key in labels}, "claims_unverified": claims, "checks_failing": checks_failing},
        decision={"action": action, "findings": findings, "criteria": len(labels), "pending": len(pending)},
        model=jev_model,
        latency_ms=int((time.monotonic() - started) * 1000),
        attempt=attempt,
    )
    if not findings or repeated:
        return None
    return {"action": "continue", "message": VERIFY_TEMPLATE.format(findings=" ".join(findings))}


def register(ctx: Any) -> None:
    global _settings_reader
    _settings_reader = ctx.get_config
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    ctx.register_hook("post_llm_call", on_post_llm_call)
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_hook("post_tool_call", on_post_tool_call)
    ctx.register_hook("pre_verify", on_pre_verify)
    logger.info("system-one-preflight registered (mode=%s, tool_guard=%s)", current_mode(), tool_guard_mode())
