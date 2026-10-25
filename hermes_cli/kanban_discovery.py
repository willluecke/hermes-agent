"""Read-only discovery sweep for the per-project orchestrator.

The orchestrator's first move on a board with nothing to route is to *read*:
sweep the project's repository read-only from several angles, assemble a
ranked list of candidate goals/issues with evidence, and park each candidate
on the board where nothing dispatches until a human accepts it. Acceptance
(``hermes kanban accept``) moves a candidate into ``triage``, where the normal
specify/decompose/dispatch pipeline takes over; rejection archives it.

Why candidates are parked as ``blocked``/``needs_input`` and not ``triage``:
the gateway auto-decomposer fans out every ``triage`` task within a tick,
which is how a 2026-10-18 sweep's candidates were dispatched before anyone
picked them. ``needs_input`` is the typed block for "waiting on a human"; the
dispatcher never spawns blocked tasks and the orchestrator never reopens that
kind (:data:`hermes_cli.kanban_orchestrator.HUMAN_GATED_BLOCK_KINDS`).

The sweep itself runs through ``hermes chat`` with the ``read_only`` toolset
(file reads and searches only: no terminal, no writes, no dispatch), one
subprocess per angle, in parallel. The runner is injectable so the
parse/merge/rank/post logic is unit-tested without a model.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from hermes_cli import kanban_db as kb

CANDIDATE_CREATOR = "discovery-sweep"
CANDIDATE_BLOCK_KIND = "needs_input"
CANDIDATE_BLOCK_REASON = "awaiting acceptance (discovery candidate)"
DEFAULT_MAX_CANDIDATES = 14
DEFAULT_SWEEP_TIMEOUT = 1200  # seconds, per angle
PER_ANGLE_LIMIT = 8
CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1}
SKIP_DIRS = (
    "node_modules", ".git", "old-code", "dist", "build", ".next",
    ".codex-worktrees", "coverage", "__pycache__",
)
_STOP_WORDS = frozenset({
    "the", "and", "for", "with", "that", "this", "from", "into", "not",
    "are", "can", "when", "has", "have", "but", "its", "via", "per",
})
_FENCE_RE = re.compile(r"```(?:json)?", re.IGNORECASE)
_EVIDENCE_RE = re.compile(r"^(?P<path>[^:\s]+):(?P<line>\d+)")


@dataclass(frozen=True)
class Angle:
    key: str
    title: str
    focus: str


ANGLES: tuple[Angle, ...] = (
    Angle(
        "half-built",
        "Half-built features",
        "features that are started but not finished: UI without a backend, "
        "routes that return placeholders, TODO/FIXME-marked flows, flags that "
        "are never read, and dead ends a user can actually reach.",
    ),
    Angle(
        "promises",
        "Promises vs reality",
        "what the README, docs, UI copy, and config comments promise versus "
        "what the code does: documented behaviour that is missing, wrong, or "
        "silently different.",
    ),
    Angle(
        "reliability",
        "Reliability and test gaps",
        "ways the product fails for a user or operator: swallowed errors, "
        "missing timeouts and retries, unvalidated input on critical paths, "
        "race conditions, and important paths with no tests or tests that "
        "never run.",
    ),
)


@dataclass
class Candidate:
    title: str
    impact: str = ""
    confidence: str = "medium"
    evidence: list[str] = field(default_factory=list)
    detail: str = ""
    angles: list[str] = field(default_factory=list)
    id: Optional[str] = None

    def score(self) -> tuple:
        """Sort key: stronger confidence, seen from more angles, more evidence."""
        return (
            -CONFIDENCE_RANK.get(self.confidence, 0),
            -len(self.angles),
            -min(len(self.evidence), 6),
            self.title.lower(),
        )

    def body(self, *, rank: int, total: int, source: str) -> str:
        lines = [f"confidence: {self.confidence}", f"rank: {rank}/{total}"]
        if self.impact:
            lines.append(f"impact: {self.impact}")
        if self.detail:
            lines.extend(["", self.detail])
        if self.evidence:
            lines.extend(["", "Evidence:"] + [f"- {e}" for e in self.evidence])
        lines.extend([
            "",
            f"angles: {', '.join(self.angles) or '-'}",
            f"source: {source} (candidate, not verified)",
            "Accept with `hermes kanban accept <id>`; reject with "
            "`hermes kanban reject <id> <reason>`.",
        ])
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "impact": self.impact,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "detail": self.detail,
            "angles": list(self.angles),
        }


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_sweep_prompt(
    angle: Angle,
    project_root: str,
    known_titles: Iterable[str] = (),
    *,
    limit: int = PER_ANGLE_LIMIT,
    goal: Optional[str] = None,
) -> str:
    """One angle's read-only sweep prompt.

    With a ``goal`` (``hermes kanban orchestrate --focus``), the sweep looks
    only for what stands between the current code and that goal: missing or
    half-built pieces it needs, promises around it the code does not keep,
    and failures that would hurt it. Unrelated problems are left out.
    """
    known = "\n".join(f"- {t}" for t in known_titles) or "- (none)"
    if goal:
        opening = (
            "You are a read-only code auditor. Sweep the repository at "
            f"{project_root} for everything that stands between the current "
            f"code and this goal: {goal}\n\n"
            "Report only problems and gaps that block, endanger or undermine "
            "that goal. Ignore unrelated issues, however real.\n\n"
        )
    else:
        opening = (
            "You are a read-only code auditor. Sweep the repository at "
            f"{project_root} and report candidate problems worth fixing.\n\n"
        )
    return (
        opening
        + f"Angle: {angle.title}. Look for {angle.focus}\n\n"
        "Rules:\n"
        "- Read only. Use search_files and read_file. You have no terminal "
        "and must not propose edits.\n"
        f"- Skip these directories: {', '.join(SKIP_DIRS)}.\n"
        f"- Report at most {limit} candidates, highest user impact first.\n"
        "- Every candidate needs evidence: file paths with line numbers you "
        "actually read. No claim without a citation.\n"
        "- Candidates are problems and gaps, not feature ideas.\n"
        "- Do not report anything already on this list (the same problem "
        "under other wording counts as listed):\n"
        f"{known}\n\n"
        "Return ONLY a JSON array, no prose and no code fence, of objects "
        "with this shape:\n"
        '{"title": "one line, under 90 characters", '
        '"impact": "what a user or operator experiences, one sentence", '
        '"confidence": "high|medium|low", '
        '"evidence": ["path:line", "path:line"], '
        '"detail": "two or three sentences on what is wrong and why"}\n'
    )


# ---------------------------------------------------------------------------
# Parsing, merging, ranking (pure)
# ---------------------------------------------------------------------------

def parse_candidates(raw: Optional[str], angle_key: str) -> list[Candidate]:
    """Lenient parse of one sweep's output into candidates.

    Accepts a bare JSON array, a fenced one, or an object with a
    ``candidates`` key. Anything unparseable yields an empty list rather than
    an exception: one bad angle must not sink the sweep.
    """
    if not raw:
        return []
    text = _FENCE_RE.sub("", raw).strip()
    data: Any = None
    first, last = text.find("["), text.rfind("]")
    if first != -1 and last > first:
        try:
            data = json.loads(text[first:last + 1])
        except ValueError:
            data = None
    if not isinstance(data, list):
        first, last = text.find("{"), text.rfind("}")
        if first != -1 and last > first:
            try:
                obj = json.loads(text[first:last + 1])
            except ValueError:
                obj = None
            if isinstance(obj, dict):
                data = obj.get("candidates")
    if not isinstance(data, list):
        return []
    out: list[Candidate] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        evidence = item.get("evidence") or []
        if isinstance(evidence, str):
            evidence = [evidence]
        if not isinstance(evidence, list):
            evidence = []
        confidence = str(item.get("confidence") or "medium").strip().lower()
        if confidence not in CONFIDENCE_RANK:
            confidence = "medium"
        out.append(
            Candidate(
                title=title[:200],
                impact=str(item.get("impact") or "").strip(),
                confidence=confidence,
                evidence=[str(e).strip() for e in evidence if str(e).strip()][:12],
                detail=str(item.get("detail") or "").strip(),
                angles=[angle_key],
            )
        )
    return out


def normalize_title(title: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).split())


def _title_tokens(title: str) -> set[str]:
    return {
        t for t in normalize_title(title).split()
        if len(t) > 2 and t not in _STOP_WORDS
    }


def _evidence_keys(evidence: Iterable[str]) -> set[str]:
    keys: set[str] = set()
    for e in evidence:
        m = _EVIDENCE_RE.match(e.strip())
        if m:
            keys.add(f"{m.group('path')}:{m.group('line')}")
    return keys


def titles_similar(a: str, b: str, *, threshold: float = 0.6) -> bool:
    if normalize_title(a) == normalize_title(b):
        return True
    ta, tb = _title_tokens(a), _title_tokens(b)
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= threshold


def _same_candidate(a: Candidate, b: Candidate) -> bool:
    if titles_similar(a.title, b.title):
        return True
    return bool(_evidence_keys(a.evidence) & _evidence_keys(b.evidence))


def merge_candidates(groups: Iterable[list[Candidate]]) -> list[Candidate]:
    """Fold per-angle lists into one list, merging duplicates across angles.

    A duplicate keeps the stronger confidence and the union of its evidence
    and angles, so a problem three sweeps found independently outranks one
    that a single sweep found.
    """
    merged: list[Candidate] = []
    for group in groups:
        for c in group:
            for m in merged:
                if not _same_candidate(m, c):
                    continue
                if CONFIDENCE_RANK[c.confidence] > CONFIDENCE_RANK[m.confidence]:
                    m.confidence = c.confidence
                for e in c.evidence:
                    if e not in m.evidence:
                        m.evidence.append(e)
                for a in c.angles:
                    if a not in m.angles:
                        m.angles.append(a)
                if not m.impact:
                    m.impact = c.impact
                if len(c.detail) > len(m.detail):
                    m.detail = c.detail
                break
            else:
                merged.append(
                    Candidate(
                        title=c.title, impact=c.impact, confidence=c.confidence,
                        evidence=list(c.evidence), detail=c.detail,
                        angles=list(c.angles),
                    )
                )
    return merged


def rank_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    return sorted(candidates, key=lambda c: c.score())


def exclude_known(
    candidates: Iterable[Candidate], known_titles: Iterable[str]
) -> list[Candidate]:
    known = list(known_titles)
    return [c for c in candidates if not any(titles_similar(c.title, k) for k in known)]


# ---------------------------------------------------------------------------
# Board I/O
# ---------------------------------------------------------------------------

def known_titles(conn) -> list[str]:
    """Titles already on the board, archived included: a rejected candidate
    must not come back on the next sweep, and finished work is not a gap."""
    return [t.title for t in kb.list_tasks(conn, include_archived=True)]


def is_candidate(task) -> bool:
    return (
        task.status == "blocked"
        and task.block_kind == CANDIDATE_BLOCK_KIND
        and task.created_by == CANDIDATE_CREATOR
    )


_RANK_RE = re.compile(r"^rank:\s*(\d+)/(\d+)", re.MULTILINE)


def candidate_rank(task) -> int:
    """The rank a sweep stamped into the body (``rank: 3/14``), or a large
    number for a body without one."""
    m = _RANK_RE.search(task.body or "")
    return int(m.group(1)) if m else 10_000


def list_candidates(conn) -> list:
    """Parked candidates on the board, oldest sweep first, then by rank."""
    rows = [t for t in kb.list_tasks(conn, status="blocked") if is_candidate(t)]
    rows.sort(key=lambda t: (t.created_at, candidate_rank(t), t.id))
    return rows


def post_candidates(
    conn,
    candidates: list[Candidate],
    *,
    source: str,
) -> tuple[list[str], list[dict[str, str]]]:
    """Park each candidate on the board as blocked/needs_input.

    Returns ``(posted_ids, skipped)``. A candidate whose title already exists
    on the board (any status, archived included) is skipped, so re-running a
    sweep never duplicates or resurrects a decision already taken.
    """
    existing = {
        normalize_title(t.title): t.id
        for t in kb.list_tasks(conn, include_archived=True)
    }
    posted: list[str] = []
    skipped: list[dict[str, str]] = []
    total = len(candidates)
    for rank, c in enumerate(candidates, 1):
        norm = normalize_title(c.title)
        if norm in existing:
            skipped.append({
                "title": c.title, "reason": "already on board",
                "id": existing[norm],
            })
            continue
        tid = kb.create_task(
            conn,
            title=c.title,
            body=c.body(rank=rank, total=total, source=source),
            created_by=CANDIDATE_CREATOR,
            initial_status="blocked",
            idempotency_key=f"discovery:{norm[:160]}",
        )
        if not kb.park_candidate(conn, tid, reason=CANDIDATE_BLOCK_REASON):
            # create_task returned an existing (non-blocked) task for the
            # idempotency key; leave it alone.
            skipped.append({"title": c.title, "reason": "already on board", "id": tid})
            continue
        c.id = tid
        existing[norm] = tid
        posted.append(tid)
    return posted, skipped


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

SweepRunner = Callable[[str, str, int], str]


def _hermes_argv() -> list[str]:
    exe = shutil.which("hermes")
    if exe:
        return [exe]
    return [
        sys.executable, "-c",
        "import sys; from hermes_cli.main import main; sys.exit(main())",
    ]


def hermes_chat_runner(prompt: str, project_root: str, timeout: int) -> str:
    """One read-only sweep: ``hermes chat -Q -t read_only -q <prompt>``.

    The ``read_only`` toolset has no terminal and no file-writing tools, so
    the sweep cannot touch the repository. The subprocess never inherits a
    kanban task identity, so it cannot act as a worker either.
    """
    env = dict(os.environ)
    env.pop("HERMES_KANBAN_TASK", None)
    argv = _hermes_argv() + ["chat", "-Q", "-t", "read_only", "-q", prompt]
    proc = subprocess.run(
        argv, cwd=project_root, env=env, capture_output=True, text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip()[-400:]
        raise RuntimeError(f"sweep exited {proc.returncode}: {tail}")
    return proc.stdout or ""


# ---------------------------------------------------------------------------
# Orchestration of one sweep
# ---------------------------------------------------------------------------

def discover(
    conn,
    *,
    board: str,
    project_root: str,
    runner: Optional[SweepRunner] = None,
    angles: Iterable[Angle] = ANGLES,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    timeout: int = DEFAULT_SWEEP_TIMEOUT,
    log: Optional[Callable[[str], None]] = None,
    parallel: bool = True,
    goal: Optional[str] = None,
) -> dict[str, Any]:
    """Sweep ``project_root`` read-only and park the ranked candidates.

    ``goal`` aims every angle at one outcome (see :func:`build_sweep_prompt`)
    and is recorded in each candidate's source line.

    Returns the sweep report: what each angle produced, which angles failed,
    the ranked candidates (with the ids they were posted under), and what was
    skipped as already on the board.
    """
    log = log or (lambda _msg: None)
    runner = runner or hermes_chat_runner
    angles = tuple(angles)
    root = os.path.abspath(os.path.expanduser(project_root))
    if not os.path.isdir(root):
        raise ValueError(f"project root is not a directory: {root}")
    known = known_titles(conn)
    goal = (goal or "").strip() or None
    prompts = {a.key: build_sweep_prompt(a, root, known, goal=goal) for a in angles}
    log(
        f"discovery: sweeping {root} read-only from {len(angles)} angle(s)"
        + (f" for: {goal}" if goal else "")
    )

    def _one(angle: Angle) -> tuple[str, str, Optional[str]]:
        started = time.time()
        try:
            out = runner(prompts[angle.key], root, timeout)
        except Exception as exc:  # one bad angle must not sink the sweep
            log(f"discovery: {angle.key} failed: {exc}")
            return angle.key, "", str(exc)
        log(f"discovery: {angle.key} done in {int(time.time() - started)}s")
        return angle.key, out, None

    if parallel and len(angles) > 1:
        with ThreadPoolExecutor(max_workers=len(angles)) as pool:
            results = list(pool.map(_one, angles))
    else:
        results = [_one(a) for a in angles]

    parsed: dict[str, int] = {}
    errors: dict[str, str] = {}
    groups: list[list[Candidate]] = []
    for key, out, err in results:
        if err:
            errors[key] = err
        group = parse_candidates(out, key)
        parsed[key] = len(group)
        groups.append(group)
    merged = exclude_known(rank_candidates(merge_candidates(groups)), known)
    dropped = max(0, len(merged) - max_candidates)
    merged = merged[:max_candidates]
    source = f"discovery sweep {time.strftime('%Y-%m-%d')} on {board}"
    if goal:
        source += f" for: {goal}"
    posted, skipped = post_candidates(conn, merged, source=source)
    log(
        f"discovery: {len(posted)} candidate(s) parked awaiting acceptance, "
        f"{len(skipped)} already on board, {dropped} over the cap"
    )
    return {
        "board": board,
        "project_root": root,
        "goal": goal,
        "angles": [a.key for a in angles],
        "parsed": parsed,
        "errors": errors,
        "candidates": [c.as_dict() for c in merged],
        "posted": posted,
        "skipped": skipped,
        "dropped_over_cap": dropped,
    }


# ---------------------------------------------------------------------------
# The report the user reads
# ---------------------------------------------------------------------------

#: What an agent relaying the sweep is told. Hermes Chat puts a note field
#: under every line that starts with a task id, so a reworded list (ids moved
#: into parentheses mid-line, say) loses that. Will, 2026-10-24: every
#: orchestrator report should read like the 3dcarparts one.
RELAY_INSTRUCTION = (
    "Show report_markdown to the user as written. Do not reword, regroup, "
    "renumber or reorder it: every candidate line starts with its task id, "
    "which is what lets Hermes Chat put a note field under each one. Put any "
    "remarks of your own before or after it, never inside it."
)

_ANGLE_TITLES = {a.key: a.title for a in ANGLES}
_COUNT_WORDS = {2: "two", 3: "three", 4: "four", 5: "five"}


def _body_field(body: str, name: str) -> str:
    m = re.search(rf"^{name}:\s*(.+)$", body or "", re.MULTILINE)
    return m.group(1).strip() if m else ""


def _sentence(text: str) -> str:
    text = " ".join((text or "").split())
    if not text:
        return ""
    return text if text[-1] in ".!?" else text + "."


def _candidate_entry(row) -> dict[str, Any]:
    """id, title, impact and angles from a parked task row or a candidate dict."""
    if isinstance(row, dict):
        return {
            "id": row.get("id") or "-",
            "title": row.get("title") or "",
            "impact": row.get("impact") or "",
            "angles": [a for a in (row.get("angles") or []) if a],
        }
    body = row.body or ""
    angles = [
        a.strip() for a in _body_field(body, "angles").split(",")
        if a.strip() and a.strip() != "-"
    ]
    return {
        "id": row.id,
        "title": row.title,
        "impact": _body_field(body, "impact"),
        "angles": angles,
    }


def _bullet(entry: dict[str, Any]) -> str:
    text = _sentence(entry["title"])
    if entry["impact"]:
        text += " " + _sentence(entry["impact"])
    if len(entry["angles"]) > 1:
        count = len(entry["angles"])
        text += f" Found by {_COUNT_WORDS.get(count, str(count))} angles."
    return f"- {entry['id']} {text}"


def format_candidates_markdown(
    rows: Iterable,
    *,
    board: Optional[str] = None,
    report: Optional[dict[str, Any]] = None,
    new_ids: Optional[Iterable[str]] = None,
    focus: Optional[str] = None,
) -> str:
    """The candidate report as the user should read it in chat.

    One bullet per candidate, starting with its task id, then its title and
    impact as sentences, grouped under the sweep angle that found it (the
    strongest group first, ranked order within each), and a fenced block of
    board-scoped accept/reject commands.

    ``new_ids`` are the candidates a sweep just parked: they lead, and every
    other waiting candidate is listed after them under "Still waiting from
    earlier sweeps". ``focus`` is that sweep's goal. ``report`` is its sweep
    report: it adds what fell over the cap and which angles failed.
    """
    entries = [_candidate_entry(r) for r in rows]
    where = f" on {board}" if board else ""
    caveat = (
        " Nothing dispatches until you accept one. Each is a lead backed by "
        "file-and-line evidence, not a verified bug."
    )
    if new_ids is None:
        fresh, earlier = entries, []
        if not entries:
            return f"No discovery candidates are waiting{where}."
        n = len(entries)
        lead = f"The read-first sweep has {n} candidate{'s' if n != 1 else ''} waiting{where}." + caveat
    else:
        wanted = set(new_ids)
        fresh = [e for e in entries if e["id"] in wanted]
        earlier = [e for e in entries if e["id"] not in wanted]
        n = len(fresh)
        aim = f" for: {focus.strip()}" if focus and focus.strip() else ""
        if n:
            lead = (
                f"The read-first sweep{aim} parked {n} new candidate"
                f"{'s' if n != 1 else ''}{where}." + caveat
            )
        else:
            lead = f"The read-first sweep{aim} found nothing new{where}."
    if report:
        dropped = int(report.get("dropped_over_cap") or 0)
        if dropped:
            lead += f" {dropped} more fell over the cap and were not posted."
        failed = sorted((report.get("errors") or {}).keys())
        if failed:
            names = ", ".join(_ANGLE_TITLES.get(k, k) for k in failed)
            lead += f" These angles failed and found nothing: {names}."
    groups: dict[str, list[dict[str, Any]]] = {}
    for entry in fresh:
        groups.setdefault(entry["angles"][0] if entry["angles"] else "", []).append(entry)
    lines = [lead]
    for key, group in groups.items():
        heading = _ANGLE_TITLES.get(key) or (key.replace("-", " ").capitalize() if key else "Other")
        lines += ["", f"**{heading}**", ""]
        lines += [_bullet(entry) for entry in group]
    if earlier:
        lines += ["", "**Still waiting from earlier sweeps**", ""]
        lines += [_bullet(entry) for entry in earlier]
    if not entries:
        return lines[0]
    flag = f"--board {board} " if board else ""
    lines += [
        "",
        "Accept or reject by id. Accepted items enter triage and the normal pipeline; rejected ones are archived and never proposed again.",
        "",
        "```",
        f"hermes kanban {flag}accept <id> [--ids <id> ...]",
        f"hermes kanban {flag}reject <id> <reason>",
        f"hermes kanban {flag}show <id>",
        "```",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Delivery to the conversation that asked
# ---------------------------------------------------------------------------

#: Told to an agent whose sweep went to the background. The orchestrator
#: posts the list itself, so what the user reads never depends on how the
#: agent reads or rewords the output (reg-watch, 2026-10-25).
BACKGROUND_NOTE = (
    "The sweep is running in the background; it usually takes 5 to 10 "
    "minutes. When it finishes, the orchestrator posts its candidate list to "
    "this conversation itself, in the standard format with a note field "
    "under each item. Tell the user it has started. Do not wait for it, poll "
    "it, or restate the list when it arrives; if you comment on candidates, "
    "refer to them by task id."
)
RUNNING_NOTE = (
    "A sweep of this board is already running. Its list posts to the "
    "conversation that started it when it finishes; do not start another."
)
REPORT_SOURCE = "kanban_orchestrator"
SWEEP_GRACE_SECONDS = 600


def chat_session_id() -> Optional[str]:
    """The Hermes Chat session this process acts for, or None.

    Set for gateway turns (context variables) and inherited by the shell of a
    resident agent (environment), so both the agent tool and a CLI call made
    from a chat know where to post.
    """
    try:
        from gateway.session_context import get_session_env

        platform = get_session_env("HERMES_SESSION_PLATFORM", "")
        chat_id = get_session_env("HERMES_SESSION_CHAT_ID", "")
    except Exception:
        platform = os.environ.get("HERMES_SESSION_PLATFORM", "")
        chat_id = os.environ.get("HERMES_SESSION_CHAT_ID", "")
    chat_id = (chat_id or "").strip()
    if platform == "api_server" and chat_id.startswith("hermes-chat-"):
        return chat_id
    return None


def sync_store_ping(session_id: str) -> None:
    """Tell Hermes Chat's sync store the session has a row to adopt. Best
    effort: the store also sweeps on a timer."""
    import urllib.error
    import urllib.request
    from pathlib import Path

    base = os.environ.get("HERMES_SYNC_URL", "http://127.0.0.1:8643").rstrip("/")
    key_file = os.environ.get("HERMES_SYNC_KEY_FILE", "").strip()
    try:
        key = Path(key_file or Path.home() / ".hermes-api-key").read_text(encoding="utf-8").strip()
    except OSError:
        return
    if not key:
        return
    request = urllib.request.Request(
        f"{base}/adoptable-messages/changed",
        data=json.dumps({"sessionId": session_id}).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5.0) as response:  # noqa: S310
            response.read()
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        pass


def post_report_to_session(
    session_id: Optional[str],
    text: str,
    *,
    board: str,
    ok: bool = True,
    db: Any = None,
    ping: Optional[Callable[[str], None]] = None,
) -> Optional[int]:
    """Post ``text`` into a Hermes Chat conversation as the orchestrator.

    The row is an assistant late answer in the gateway's session store, the
    same path an answer written after its turn takes, so Hermes Chat adopts
    it into the conversation as its own message. Returns the row id, or None
    when there is no chat session or the write fails.
    """
    text = (text or "").strip()
    if not session_id or not session_id.startswith("hermes-chat-") or not text:
        return None
    try:
        from hermes_state import LATE_ANSWER_DISPLAY_KIND, SessionDB

        owned = db is None
        store = db if db is not None else SessionDB()
        try:
            row_id = store.append_message(
                session_id,
                "assistant",
                text,
                finish_reason="stop",
                display_kind=LATE_ANSWER_DISPLAY_KIND,
                display_metadata={
                    "source": REPORT_SOURCE,
                    "origin": "discovery-sweep",
                    "board": board,
                    "complete": True,
                    "is_error": not ok,
                    "captured_at": time.time(),
                },
            )
        finally:
            if owned:
                try:
                    store.close()
                except Exception:
                    pass
    except Exception:
        return None
    (ping or sync_store_ping)(session_id)
    return row_id


# ---------------------------------------------------------------------------
# One sweep at a time per board
# ---------------------------------------------------------------------------

def sweep_marker_path(board: str):
    return kb.board_dir(board) / "discovery" / "running.json"


def running_sweep(board: str) -> Optional[dict[str, Any]]:
    """The live sweep of ``board``, or None. A marker past its deadline, or
    whose process is gone, is stale."""
    path = sweep_marker_path(board)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    if time.time() > float(data.get("deadline") or 0):
        return None
    pid = data.get("pid")
    if pid and kb._worker_pid_probe_trusted(data.get("pid_ns")) and not kb._pid_alive(int(pid)):
        return None
    return data


def _write_marker(board: str, data: dict[str, Any]) -> None:
    path = sweep_marker_path(board)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(path)


def _clear_marker(board: str, token: str) -> None:
    path = sweep_marker_path(board)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if isinstance(data, dict) and data.get("token") == token:
        try:
            path.unlink()
        except OSError:
            pass


def _resolve_root(slug: str, project_root: Optional[str]) -> Optional[str]:
    return project_root or kb.read_board_metadata(slug).get("default_workdir") or None


def _no_root_hint(slug: str) -> str:
    return (
        f"board {slug!r} has no default workdir to sweep; set one with "
        f"`hermes kanban boards set-default-workdir {slug} <path>` "
        "or pass a project root"
    )


# ---------------------------------------------------------------------------
# Running a sweep: foreground, or detached with the report posted
# ---------------------------------------------------------------------------

def run_sweep(
    *,
    board: Optional[str],
    goal: Optional[str] = None,
    project_root: Optional[str] = None,
    max_candidates: Optional[int] = None,
    timeout: Optional[int] = None,
    deliver_to: Optional[str] = None,
    log: Optional[Callable[[str], None]] = None,
    ping: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    """Sweep now, in this process, holding the board's sweep marker.

    Returns the sweep info (the sweep report plus ``report_markdown`` and
    ``relay``). With ``deliver_to`` the finished report, or a short failure
    note, is also posted into that Hermes Chat session.
    """
    log = log or (lambda _msg: None)
    slug = board or kb.get_current_board()
    goal = (goal or "").strip() or None
    token = os.environ.get("HERMES_DISCOVERY_SWEEP_TOKEN") or f"fg-{os.getpid()}-{time.time()}"
    live = running_sweep(slug)
    if live and live.get("token") != token:
        return {"ran": False, "running": True, "since": live.get("started_at"),
                "delivers_to": live.get("session"), "note": RUNNING_NOTE}
    info: dict[str, Any] = {}

    def _fail(reason: str, **extra: Any) -> dict[str, Any]:
        info.update(ran=False, **extra)
        if deliver_to:
            note = f"The read-first sweep on {slug} could not run: {reason}."
            info["delivered_row"] = post_report_to_session(
                deliver_to, note, board=slug, ok=False, ping=ping,
            )
        return info

    root = _resolve_root(slug, project_root)
    if not root:
        return _fail(_no_root_hint(slug), skipped="no project root", hint=_no_root_hint(slug))
    budget = int(timeout or DEFAULT_SWEEP_TIMEOUT)
    _write_marker(slug, {
        "token": token, "pid": os.getpid(), "pid_ns": kb._pid_namespace_id(),
        "started_at": time.time(), "deadline": time.time() + budget + SWEEP_GRACE_SECONDS,
        "session": deliver_to, "focus": goal,
    })
    try:
        kwargs: dict[str, Any] = {"timeout": budget}
        if max_candidates:
            kwargs["max_candidates"] = int(max_candidates)
        if goal:
            kwargs["goal"] = goal
        try:
            with kb.connect_closing(board=board) as conn:
                report = discover(conn, board=slug, project_root=root, log=log, **kwargs)
        except ValueError as exc:
            return _fail(str(exc), error=str(exc))
        except Exception as exc:  # a crashed sweep still tells the conversation
            return _fail(f"{type(exc).__name__}: {exc}", error=str(exc))
        info["ran"] = True
        info.update(report)
        with kb.connect_closing(board=board) as conn:
            rows = list_candidates(conn)
        info["waiting"] = len(rows)
        info["report_markdown"] = format_candidates_markdown(
            rows, board=slug, report=report, new_ids=report.get("posted") or [], focus=goal,
        )
        info["relay"] = RELAY_INSTRUCTION
        if deliver_to:
            info["delivered_row"] = post_report_to_session(
                deliver_to, info["report_markdown"], board=slug, ping=ping,
            )
        return info
    finally:
        _clear_marker(slug, token)


def _spawn_detached(argv: list[str], *, session: str, title: str, log_path, env: dict) -> str:
    """Start ``argv`` outside this process's lifetime. Prefers ``hermes-bg``
    (a systemd user unit registered to the conversation, which survives a
    gateway restart and shows in Hermes Chat's background indicator), and
    falls back to a detached process. Returns which launcher ran it."""
    shell_cmd = ["/bin/sh", "-c", 'exec "$@" >>"$0" 2>&1', str(log_path)] + argv
    home = env.get("HERMES_HOME")
    if home:
        shell_cmd = ["/usr/bin/env", f"HERMES_HOME={home}",
                     f"HERMES_DISCOVERY_SWEEP_TOKEN={env['HERMES_DISCOVERY_SWEEP_TOKEN']}"] + shell_cmd
    else:
        shell_cmd = ["/usr/bin/env",
                     f"HERMES_DISCOVERY_SWEEP_TOKEN={env['HERMES_DISCOVERY_SWEEP_TOKEN']}"] + shell_cmd
    bg = shutil.which("hermes-bg")
    if bg:
        try:
            done = subprocess.run(
                [bg, "--session", session, "--title", title, "--"] + shell_cmd,
                env=env, capture_output=True, text=True, timeout=30,
            )
            if done.returncode == 0:
                return "hermes-bg"
        except (OSError, subprocess.SubprocessError):
            pass
    with open(log_path, "ab") as out:
        subprocess.Popen(
            argv, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return "process"


def launch_background_sweep(
    *,
    board: Optional[str],
    session: str,
    goal: Optional[str] = None,
    project_root: Optional[str] = None,
    max_candidates: Optional[int] = None,
    timeout: Optional[int] = None,
) -> dict[str, Any]:
    """Start ``hermes kanban sweep`` detached; it posts its report to
    ``session`` when it finishes. Returns at once."""
    slug = board or kb.get_current_board()
    goal = (goal or "").strip() or None
    live = running_sweep(slug)
    if live:
        return {"ran": False, "background": True, "running": True,
                "since": live.get("started_at"), "delivers_to": live.get("session"),
                "note": RUNNING_NOTE}
    root = _resolve_root(slug, project_root)
    if not root:
        return {"ran": False, "skipped": "no project root", "hint": _no_root_hint(slug)}
    budget = int(timeout or DEFAULT_SWEEP_TIMEOUT)
    token = f"bg-{os.getpid()}-{time.time()}"
    log_path = sweep_marker_path(slug).parent / f"sweep-{time.strftime('%Y%m%d-%H%M%S')}.log"
    argv = _hermes_argv() + [
        "kanban", "--board", slug, "sweep", slug, "--deliver-to", session,
        "--project-root", root, "--sweep-timeout", str(budget),
    ]
    if goal:
        argv += ["--focus", goal]
    if max_candidates:
        argv += ["--max-candidates", str(int(max_candidates))]
    # Claimed before the spawn, so a second call right behind this one sees
    # the sweep; the child owns the marker through the shared token.
    _write_marker(slug, {
        "token": token, "pid": None, "started_at": time.time(),
        "deadline": time.time() + budget + SWEEP_GRACE_SECONDS,
        "session": session, "focus": goal, "log": str(log_path),
    })
    env = dict(os.environ)
    env.pop("HERMES_KANBAN_TASK", None)
    env["HERMES_DISCOVERY_SWEEP_TOKEN"] = token
    try:
        launcher = _spawn_detached(
            argv, session=session, title=f"Discovery sweep: {slug}", log_path=log_path, env=env,
        )
    except Exception as exc:
        _clear_marker(slug, token)
        return {"ran": False, "background": True, "started": False, "error": str(exc)}
    return {
        "ran": False, "background": True, "started": True, "launcher": launcher,
        "delivers_to": session, "log": str(log_path), "focus": goal, "note": BACKGROUND_NOTE,
    }


def read_first_step(
    *,
    board: Optional[str],
    recommended: bool,
    force: bool = False,
    skip: bool = False,
    goal: Optional[str] = None,
    project_root: Optional[str] = None,
    max_candidates: Optional[int] = None,
    timeout: Optional[int] = None,
    log: Optional[Callable[[str], None]] = None,
    background_session: Optional[str] = None,
) -> dict[str, Any]:
    """The read-first step of an orchestrate cycle, shared by the CLI and the
    ``kanban_orchestrate`` agent tool.

    Sweeps when the cycle recommends it (nothing to route and no candidates
    already waiting) or when ``force`` is set; ``skip`` suppresses it. With a
    ``background_session`` (a Hermes Chat conversation) the sweep is started
    detached and posts its own report there when it finishes; without one it
    runs here. A board already being swept is not swept twice. Whenever the
    step does not hand off to a background sweep and candidates are waiting,
    the ready-to-relay ``report_markdown`` and the ``relay`` instruction are
    included. Returns the keys to merge into ``result["discovery"]``.
    """
    slug = board or kb.get_current_board()
    info: dict[str, Any] = {}
    due = not skip and (force or recommended)
    if skip:
        info.update(ran=False, skipped="--no-discover")
    elif not due:
        info["ran"] = False
    else:
        live = running_sweep(slug)
        if live:
            info.update(ran=False, running=True, since=live.get("started_at"),
                        delivers_to=live.get("session"), note=RUNNING_NOTE)
        elif background_session:
            launched = launch_background_sweep(
                board=board, session=background_session, goal=goal,
                project_root=project_root, max_candidates=max_candidates, timeout=timeout,
            )
            info.update(launched)
            if launched.get("started"):
                # The posted report lists every waiting candidate; showing
                # them now as well would only invite a second, reworded list.
                return info
        else:
            info.update(run_sweep(
                board=board, goal=goal, project_root=project_root,
                max_candidates=max_candidates, timeout=timeout, log=log,
            ))
            if info.get("report_markdown"):
                return info
    with kb.connect_closing(board=board) as conn:
        rows = list_candidates(conn)
    info["waiting"] = len(rows)
    if rows:
        info["report_markdown"] = format_candidates_markdown(rows, board=slug)
        info["relay"] = RELAY_INSTRUCTION
    return info


def format_candidates(rows: Iterable, *, board: Optional[str] = None) -> str:
    """Human-readable table of parked candidates (task rows or dicts)."""
    lines: list[str] = []
    items = list(rows)
    if not items:
        return "No discovery candidates awaiting acceptance."
    head = "Discovery candidates awaiting acceptance"
    if board:
        head += f" on {board}"
    lines.append(head + ":")
    for i, r in enumerate(items, 1):
        if isinstance(r, dict):
            tid, title = r.get("id") or "-", r.get("title") or ""
            conf = r.get("confidence") or ""
        else:
            tid, title = r.id, r.title
            m = re.search(r"^confidence:\s*(\w+)", r.body or "", re.MULTILINE)
            conf = m.group(1) if m else ""
        lines.append(f"  {i:>2}. {tid}  [{conf or '-'}]  {title}")
    lines.append(
        "Accept: `hermes kanban accept <id> [--ids ...]`   "
        "Reject: `hermes kanban reject <id> <reason>`   "
        "Details: `hermes kanban show <id>`"
    )
    return "\n".join(lines)
