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
