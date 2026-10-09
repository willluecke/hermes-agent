"""Evidence for the verify judge, built by code from every command the turn ran.

The verify judge used to see the last six terminal commands of the session
as raw output tails. That window dropped the checks a long turn cited and
then penalised their absence; the model's answer was to re-run everything
so it would fit. This module replaces the window with a ledger:

* one row per terminal call this turn, built here from the tool result:
  id, command, exit code (or unknown, on the Claude lane whose hook payload
  carries none), what the runner reported (parsed by the extractors below,
  ``unknown`` when nothing is recognised), a failure flag, a digest of the
  output and the workspace digest the command ran under;
* the full output retained on disk per row, so the judge gets the excerpt
  around the first failure instead of a tail;
* a workspace digest per repository, from ``git status`` plus the size and
  mtime of every changed path, so a check that ran before a later edit is
  stale by code rather than by judgment; a row is compared only on the
  repositories it ran under, so first touching another repository later in
  the turn does not make it stale;
* the result manifest the model registers with ``report_results``: each
  claim names the commands it rests on, as the model ran them, and a
  predicate code can compare exactly; code matches each cited command to
  its newest run (the model never sees a row id, so it is never asked for
  one), and the verdict is ``supported``, ``contradicted``, ``stale``,
  ``insufficient`` or ``missing``;
* controller re-runs: a cited check that is stale or whose exit is unknown
  is run again by the gate itself, never by the model, when the command is
  a plain check runner and the cwd is known.

Nothing here calls Jev. Everything is deterministic given the tool results,
and every limit is stated as a constant.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

PARSER_VERSION = 1
MAX_ROWS = 200
MAX_PREVIOUS_ROWS = 50
MAX_OUTPUT_CHARS = 200_000
MAX_EXCERPT_CHARS = 1_200
EXCERPT_BEFORE = 400
MAX_EXCERPTS = 4
MAX_MANIFEST_ITEMS = 16
MAX_CLAIM_CHARS = 300
# A cited command is matched, not run, so a pasted heredoc is kept whole
# only up to this bound; a partial citation shorter than the minimum could
# match almost any command, so it must match exactly.
MAX_CITED_CHARS = 2_000
MIN_PARTIAL_CITATION_CHARS = 8
# The judge's state carries a preview of each command; the row keeps the
# whole command so the gate re-runs what the agent ran, not a prefix of it
# (a 500-character pytest clipped at 300 re-ran as "0 tests").
MAX_COMMAND_CHARS = 300
MAX_COMMAND_STORE_CHARS = 20_000
STATUSES = ("pass", "fail", "unknown")
VERDICTS = ("supported", "contradicted", "stale", "insufficient", "missing")
PREDICATES = ("passed", "count", "exit_zero", "contains", "ran")
# A decisive claim is one the gate must back with its own run: the agent's
# row is advisory, because a command can print the summary that clears it.
DECISIVE = ("passed", "count", "exit_zero")

CHECK_COMMAND_RE = re.compile(
    r"\b(pytest|npm (run )?(test|lint|build|typecheck|type-check|check|verify)|pnpm (run )?(test|lint|build|typecheck|type-check|check)|"
    r"yarn (run )?(test|lint|build|typecheck|type-check|check)|vitest|jest|mocha|playwright test|go test|cargo (test|check|clippy)|"
    r"make (test|check|lint)|node --test|node (--[\w-]+ )*--test|tsc\b|eslint|ruff|mypy|flake8|"
    r"black --check|prettier --check)",
    re.IGNORECASE,
)
FAILURE_RE = re.compile(
    r"(traceback|\berror\b|\bfailed\b|exit code [1-9]|command not found|no such file|permission denied)",
    re.IGNORECASE,
)
_EXIT_LINE_RE = re.compile(r"(?im)^\s*exit code:?\s*(-?\d+)\s*$")
_CD_PREFIX_RE = re.compile(r"""^\s*cd\s+("[^"]+"|'[^']+'|\S+)\s*(?:&&|;)""")
_TRUNCATED_RE = re.compile(r"Full output \([\d,]+ chars\) saved to|\[truncated\]|…$")

# Paths whose change means the verification machinery itself moved. A flag
# for the human, never a finding: adding a test is the normal case.
MACHINERY_RE = re.compile(
    r"(^|/)(tests?|__tests__|spec)/|(^|/)test_[^/]*\.py$|_test\.py$|\.(test|spec)\.[cm]?[jt]sx?$|"
    r"(^|/)conftest\.py$|(^|/)(pytest\.ini|pyproject\.toml|setup\.cfg|tox\.ini|package\.json|"
    r"jest\.config\.[cm]?[jt]s|vitest\.config\.[cm]?[jt]s|\.eslintrc[^/]*|eslint\.config\.[cm]?[jt]s|"
    r"tsconfig[^/]*\.json|Makefile)$|(^|/)\.github/workflows/",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Tool results -> exit code and output
# ---------------------------------------------------------------------------

def split_result(text: str) -> Tuple[str, Optional[int]]:
    """(output, exit_code) from a terminal tool result.

    The default loop's terminal tool answers with JSON carrying ``output``
    and ``exit_code``. The Claude lane forwards stdout and stderr as text
    with no exit code, so it stays ``None`` there unless the runtime printed
    an ``Exit code: N`` line of its own.
    """
    text = text if isinstance(text, str) else ""
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except (TypeError, ValueError):
            payload = None
        if isinstance(payload, dict) and ("output" in payload or "exit_code" in payload):
            output = payload.get("output")
            output = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False, default=str) if output is not None else ""
            code = payload.get("exit_code")
            return output, int(code) if isinstance(code, int) and not isinstance(code, bool) else None
    match = None
    for match in _EXIT_LINE_RE.finditer(text):
        pass
    if match is not None:
        try:
            return text, int(match.group(1))
        except ValueError:
            return text, None
    return text, None


# ---------------------------------------------------------------------------
# Extractors: what the runner reported, or unknown
# ---------------------------------------------------------------------------

_PYTEST_COUNT_RE = re.compile(r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed)\b")
_NODE_COUNT_RE = re.compile(r"(?m)^\s*(?:#|ℹ)\s*(pass|fail|tests|skipped|todo|cancelled)\s+(\d+)\s*$")
_JEST_LINE_RE = re.compile(r"(?m)^\s*Tests:?\s+(.+)$")
_JEST_COUNT_RE = re.compile(r"(\d+) (passed|failed|skipped|todo)\b")
_TSC_ERROR_RE = re.compile(r"(?m)error TS\d+")
_TSC_FOUND_RE = re.compile(r"Found (\d+) errors?")
_ESLINT_PROBLEMS_RE = re.compile(r"✖ (\d+) problems? \((\d+) errors?, (\d+) warnings?\)")
_ESLINT_CLEAN_RE = re.compile(r"✔ No ESLint warnings or errors")
_RUFF_FOUND_RE = re.compile(r"Found (\d+) errors?")
_RUFF_CLEAN_RE = re.compile(r"All checks passed!")


def _by_exit(exit_code: Optional[int]) -> str:
    if exit_code is None:
        return "unknown"
    return "pass" if exit_code == 0 else "fail"


def detect_kind(command: str, output: str) -> str:
    """The runner that produced this output, from its signature first and the command second.

    Only a command that is itself a check runner (or a script runner such
    as ``npm test`` that wraps one) is read by its output signature. An
    ``echo`` or a ``cat`` that prints a runner's summary line is not a
    check, whatever the line says: a live probe showed the real Claude CLI
    running ``echo '3 passed in 0.1s'`` and the extractor believing it.
    """
    if not CHECK_COMMAND_RE.search(command or ""):
        return "other"
    tail = output[-6_000:]
    if _NODE_COUNT_RE.search(tail):
        return "node-test"
    if _JEST_LINE_RE.search(tail) and _JEST_COUNT_RE.search(tail):
        return "jest"
    if _PYTEST_COUNT_RE.search(tail) and re.search(r"(?m)^=+ .* =+\s*$|\bin [\d.]+s\b", tail):
        return "pytest"
    if _TSC_ERROR_RE.search(tail):
        return "tsc"
    if _ESLINT_PROBLEMS_RE.search(tail) or _ESLINT_CLEAN_RE.search(tail):
        return "eslint"
    if _RUFF_CLEAN_RE.search(tail) or (re.search(r"\bruff\b", command) and _RUFF_FOUND_RE.search(tail)):
        return "ruff"
    lowered = command.lower()
    if "pytest" in lowered:
        return "pytest"
    if re.search(r"node\b.*--test\b", lowered):
        return "node-test"
    if re.search(r"\b(vitest|jest)\b", lowered):
        return "jest"
    if re.search(r"\btsc\b", lowered):
        return "tsc"
    if re.search(r"\beslint\b|next lint", lowered):
        return "eslint"
    if re.search(r"\bruff\b", lowered):
        return "ruff"
    return "other"


def parse_observation(command: str, output: str, exit_code: Optional[int]) -> Dict[str, Any]:
    """What the check reported: ``{"kind", "status", "counts", "recognized", "parser"}``.

    The runner's own summary wins over the exit code, because a pipe masks
    the exit code and a summary never lies about itself. Without a summary
    the exit code decides, and without either the status is ``unknown``.
    """
    kind = detect_kind(command, output or "")
    tail = (output or "")[-6_000:]
    counts: Dict[str, int] = {}
    status = "unknown"
    recognized = False
    if kind == "pytest":
        last: Dict[str, int] = {}
        for value, label in _PYTEST_COUNT_RE.findall(tail):
            key = "errors" if label.startswith("error") else label
            last[key] = int(value)
        if last:
            counts = last
            recognized = True
            failed = counts.get("failed", 0) + counts.get("errors", 0)
            status = "fail" if failed else ("pass" if counts.get("passed", 0) > 0 else "unknown")
        elif "no tests ran" in tail:
            recognized = True
            counts = {"passed": 0}
            status = "unknown"
        else:
            status = _by_exit(exit_code)
    elif kind == "node-test":
        for label, value in _NODE_COUNT_RE.findall(tail):
            counts[{"pass": "passed", "fail": "failed"}.get(label, label)] = int(value)
        if "passed" in counts or "failed" in counts:
            recognized = True
            status = "fail" if counts.get("failed", 0) else ("pass" if counts.get("passed", 0) > 0 else "unknown")
        else:
            status = _by_exit(exit_code)
    elif kind == "jest":
        line = None
        for line in _JEST_LINE_RE.finditer(tail):
            pass
        if line is not None:
            for value, label in _JEST_COUNT_RE.findall(line.group(1)):
                counts[label] = int(value)
        if "passed" in counts or "failed" in counts:
            recognized = True
            status = "fail" if counts.get("failed", 0) else ("pass" if counts.get("passed", 0) > 0 else "unknown")
        else:
            status = _by_exit(exit_code)
    elif kind == "tsc":
        found = _TSC_FOUND_RE.search(tail)
        errors = int(found.group(1)) if found else len(_TSC_ERROR_RE.findall(tail))
        counts = {"errors": errors}
        if errors:
            recognized = True
            status = "fail"
        elif exit_code in (0, None) and not FAILURE_RE.search(tail):
            # tsc prints nothing on success; no error lines and no failing exit is a pass.
            recognized = exit_code == 0
            status = "pass"
        else:
            status = _by_exit(exit_code)
    elif kind == "eslint":
        problems = _ESLINT_PROBLEMS_RE.search(tail)
        if problems:
            counts = {"problems": int(problems.group(1)), "errors": int(problems.group(2)), "warnings": int(problems.group(3))}
            recognized = True
            status = "fail" if counts["errors"] else "pass"
        elif _ESLINT_CLEAN_RE.search(tail):
            counts = {"errors": 0, "warnings": 0}
            recognized = True
            status = "pass"
        else:
            status = _by_exit(exit_code)
    elif kind == "ruff":
        found = _RUFF_FOUND_RE.search(tail)
        if found:
            counts = {"errors": int(found.group(1))}
            recognized = True
            status = "fail" if counts["errors"] else "pass"
        elif _RUFF_CLEAN_RE.search(tail):
            counts = {"errors": 0}
            recognized = True
            status = "pass"
        else:
            status = _by_exit(exit_code)
    else:
        status = _by_exit(exit_code)
    return {"kind": kind, "status": status, "counts": counts, "recognized": recognized, "parser": PARSER_VERSION}


# ---------------------------------------------------------------------------
# Workspace digest: what state a command ran under
# ---------------------------------------------------------------------------

_root_cache: Dict[str, Optional[str]] = {}


def _git(root: str, args: List[str], timeout: float = 5.0) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", "-C", root, *args], capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout if completed.returncode == 0 else None


def git_root(path: str) -> Optional[str]:
    """The repository root containing ``path`` (a file or directory), cached per directory."""
    if not path:
        return None
    try:
        candidate = Path(path).expanduser()
        directory = candidate if candidate.is_dir() else candidate.parent
        directory = str(directory.resolve())
    except (OSError, RuntimeError):
        return None
    if directory in _root_cache:
        return _root_cache[directory]
    if not os.path.isdir(directory):
        _root_cache[directory] = None
        return None
    top = _git(directory, ["rev-parse", "--show-toplevel"])
    root = top.strip() if top else None
    if len(_root_cache) > 512:
        _root_cache.clear()
    _root_cache[directory] = root
    return root


def git_head(root: str) -> Optional[str]:
    """The commit a repository is at, or None (no repository, or no commit yet)."""
    head = _git(root, ["rev-parse", "HEAD"])
    return head.strip() if head and head.strip() else None


def cd_prefix(command: str) -> Optional[str]:
    """The directory a ``cd X && ...`` command starts in, or None."""
    match = _CD_PREFIX_RE.match(command or "")
    if not match:
        return None
    return match.group(1).strip("\"'")


def workspace_digests(roots: List[str]) -> Dict[str, str]:
    """One ``workspace_digest`` per repository, for the repositories git can read."""
    digests: Dict[str, str] = {}
    for root in sorted({root for root in roots if root}):
        digest = workspace_digest([root])
        if digest:
            digests[root] = digest
    return digests


def combine_digests(digests: Dict[str, str]) -> Optional[str]:
    """One digest for a whole per-repository map, or None when it is empty."""
    if not digests:
        return None
    return hashlib.sha256(json.dumps(sorted(digests.items())).encode("utf-8")).hexdigest()[:16]


def row_is_stale(row: Dict[str, Any], final: Dict[str, str]) -> Optional[bool]:
    """Whether a repository the row ran under changed after it ran; None when unknown.

    Only the repositories in the row's own map count. With one digest over
    every repository the turn had touched, a command run in a second
    repository made every earlier row read "ran before later edits" with
    nothing edited (2026-09-30). A repository missing from ``final`` (no
    longer readable, or past the root cap) is not evidence of a change.
    """
    mine = row.get("workspaces")
    if not isinstance(mine, dict) or not mine or not final:
        return None
    shared = [root for root in mine if root in final]
    if not shared:
        return None
    return any(final[root] != mine[root] for root in shared)


def workspace_digest(roots: List[str]) -> Optional[str]:
    """A digest of every repository's HEAD, status, and the size and mtime of each changed path.

    ``git status`` alone is not enough: two edits to the same file leave the
    same status line, so the stat of every listed path goes in. Untracked
    files count (a new module is exactly what an edit creates); ignored
    files do not, which keeps test caches out when the repository ignores
    them. Returns None when no root is known or git fails.
    """
    hasher = hashlib.sha256()
    seen = False
    for root in sorted({root for root in roots if root}):
        head = _git(root, ["rev-parse", "HEAD"])
        status = _git(root, ["status", "--porcelain=v2", "--untracked-files=all", "--no-renames"])
        if status is None:
            continue
        seen = True
        hasher.update(root.encode("utf-8", "replace"))
        hasher.update((head or "").encode("utf-8"))
        for line in status.splitlines():
            hasher.update(line.encode("utf-8", "replace"))
            parts = line.split(" ")
            if not parts:
                continue
            if parts[0] == "1" and len(parts) >= 9:
                rel = " ".join(parts[8:])
            elif parts[0] == "?" and len(parts) >= 2:
                rel = " ".join(parts[1:])
            elif parts[0] == "u" and len(parts) >= 11:
                rel = " ".join(parts[10:])
            else:
                continue
            try:
                stat = os.stat(os.path.join(root, rel))
                hasher.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode("ascii"))
            except OSError:
                hasher.update(b"missing")
    return hasher.hexdigest()[:16] if seen else None


def _status_path(line: str) -> Optional[Tuple[str, str]]:
    """``(kind, relative path)`` from one ``git status --porcelain=v2`` line: kind is ``tracked`` or ``untracked``."""
    parts = line.split(" ")
    if not parts:
        return None
    if parts[0] == "1" and len(parts) >= 9:
        return "tracked", " ".join(parts[8:])
    if parts[0] == "u" and len(parts) >= 11:
        return "tracked", " ".join(parts[10:])
    if parts[0] == "?" and len(parts) >= 2:
        return "untracked", " ".join(parts[1:])
    return None


MAX_DERIVED_PATHS = 200


def changed_paths_since(root: str, base: Optional[str], since: float) -> List[str]:
    """The files a turn changed in ``root``, read from git rather than from file-change tool calls.

    A model that edits through shell commands (a Python heredoc, ``sed -i``,
    ``git apply``) makes no Edit, Write or apply_patch call, so the lanes
    handed the verify judge no changed paths and it never ran: 28 of the 54
    build turns of the week to 2026-10-01 went unjudged that way, and their
    criteria were never retired. Tracked files are those that differ from
    ``base``, the commit the repository was at when the turn began, so a
    commit the turn made still shows; without a base, the working tree's
    modified files. Untracked files count when written at or after
    ``since``, the turn's start: a scratch file left from an earlier turn is
    not this turn's change. Ignored files never appear. Absolute paths,
    sorted, at most ``MAX_DERIVED_PATHS``; empty when git cannot read the
    repository.
    """
    status = _git(root, ["status", "--porcelain=v2", "--untracked-files=all", "--no-renames"])
    if status is None:
        return []
    tracked: set = set()
    untracked: set = set()
    for line in status.splitlines():
        entry = _status_path(line)
        if entry is None:
            continue
        kind, rel = entry
        (tracked if kind == "tracked" else untracked).add(rel)
    if base:
        diffed = _git(root, ["diff", "--name-only", base, "--"])
        if diffed is not None:
            tracked = {rel for rel in diffed.splitlines() if rel.strip()}
    changed: List[str] = [os.path.join(root, rel) for rel in tracked]
    for rel in untracked:
        path = os.path.join(root, rel)
        try:
            if os.stat(path).st_mtime >= since:
                changed.append(path)
        except OSError:
            continue
    return sorted(set(changed))[:MAX_DERIVED_PATHS]


# ---------------------------------------------------------------------------
# Ledger rows and retained output
# ---------------------------------------------------------------------------

def _clip(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def make_row(
    n: int,
    command: str,
    result_text: str,
    *,
    cwd: str = "",
    workspace: Optional[str] = None,
    source: str = "agent",
    prefix: str = "c",
) -> Tuple[Dict[str, Any], str]:
    """(row, output) for one terminal call. The output is returned so the caller can retain it."""
    output, exit_code = split_result(result_text)
    output = output if len(output) <= MAX_OUTPUT_CHARS else output[:MAX_OUTPUT_CHARS]
    observation = parse_observation(command, output, exit_code)
    row = {
        "id": f"{prefix}{n}",
        "n": n,
        "command": command.strip()[:MAX_COMMAND_STORE_CHARS],
        "command_preview": _clip(command, MAX_COMMAND_CHARS),
        "cwd": cwd or "",
        "exit": exit_code,
        "status": observation["status"],
        "kind": observation["kind"],
        "counts": observation["counts"],
        "recognized": observation["recognized"],
        "failure": bool(FAILURE_RE.search(output[:20_000])),
        "digest": hashlib.sha256(output.encode("utf-8", "replace")).hexdigest()[:12],
        "chars": len(output),
        "capture_complete": not bool(_TRUNCATED_RE.search(output[-400:])),
        "workspace": workspace,
        "source": source,
        "check": bool(CHECK_COMMAND_RE.search(command)),
        "at": time.time(),
    }
    return row, output


def row_summary(row: Dict[str, Any]) -> Dict[str, Any]:
    """The compact form of a row for the judge's state: identity, result, freshness."""
    summary: Dict[str, Any] = {
        "id": row["id"],
        "command": row.get("command_preview") or _clip(row["command"], MAX_COMMAND_CHARS),
        "exit": row["exit"] if row["exit"] is not None else "unknown",
        "status": row["status"],
        "kind": row["kind"],
        "source": row["source"],
    }
    if row.get("counts"):
        summary["counts"] = row["counts"]
    if row.get("failure"):
        summary["failure_text"] = True
    if not row.get("capture_complete", True):
        summary["capture_complete"] = False
    if row.get("fresh") is not None:
        summary["fresh"] = row["fresh"]
    return summary


def failure_excerpt(output: str, *, before: int = EXCERPT_BEFORE, limit: int = MAX_EXCERPT_CHARS) -> str:
    """The text around the first failure match, or the tail when nothing matches."""
    text = output or ""
    match = FAILURE_RE.search(text)
    if match is None:
        return _clip(text[-limit:], limit) if len(text) > limit else text.strip()
    start = max(0, match.start() - before)
    excerpt = text[start : start + limit]
    return ("…" if start else "") + excerpt.strip() + ("…" if start + limit < len(text) else "")


def retain_output(directory: Path, row_id: str, output: str) -> Optional[str]:
    """Write the full output for a row to disk; returns the path or None."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{row_id}.txt"
        path.write_text(output, encoding="utf-8", errors="replace")
        return str(path)
    except OSError:
        return None


def read_output(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def machinery_paths(changed_paths: List[str], root: Optional[str] = None) -> List[str]:
    """Changed paths that are tests or runner configuration."""
    found: List[str] = []
    for path in changed_paths:
        rel = path
        if root and os.path.abspath(path).startswith(root + os.sep):
            rel = os.path.relpath(path, root)
        if MACHINERY_RE.search(rel.replace(os.sep, "/")):
            found.append(rel)
    return found


# ---------------------------------------------------------------------------
# The result manifest and its code-side verdicts
# ---------------------------------------------------------------------------

def tool_result_payload(text: str) -> Optional[Dict[str, Any]]:
    """Decode native results and the MCP envelopes replayed by external lanes.

    Prefer structured content when present. Otherwise require exactly one
    JSON text block; do not search arbitrary tool prose for a registration.
    Invalid/error results leave the existing registration unchanged.
    """
    payload: Any = text
    for _ in range(8):
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except (TypeError, ValueError):
                return None
        if not isinstance(payload, dict) or payload.get("isError") or payload.get("is_error"):
            return None
        if payload.get("structuredContent") is not None:
            payload = payload["structuredContent"]
        elif isinstance(payload.get("result"), str):
            payload = payload["result"]
        elif isinstance(payload.get("content"), list):
            blocks = [
                item["text"] for item in payload["content"]
                if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str)
            ]
            if len(blocks) != 1:
                return None
            payload = blocks[0]
        else:
            return payload
    return None


def parse_manifest(text: str) -> Optional[List[Dict[str, Any]]]:
    """The ``report_results`` registration, raw or wrapped by the MCP bridge."""
    payload = tool_result_payload(text)
    items = payload.get("manifest") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return None
    manifest: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim") or "").strip()
        if not claim:
            continue
        # The tool normalizes the key, but a registration from an older tool
        # build (or a bridge that passed the raw call through) may still carry
        # the model's spelling.
        evidence = next(
            (item[key] for key in ("evidence", "commands", "command", "rows") if isinstance(item.get(key), list) and item[key]),
            [],
        )
        if isinstance(item.get("evidence"), str):
            evidence = [item["evidence"]]
        expected = item.get("expected") if isinstance(item.get("expected"), dict) else {}
        manifest.append(
            {
                "id": str(item.get("id") or f"r{len(manifest) + 1}"),
                "criterion": str(item.get("criterion") or ""),
                "claim": _clip(claim, MAX_CLAIM_CHARS),
                "evidence": [str(value).strip()[:MAX_CITED_CHARS] for value in evidence if str(value).strip()][:8],
                "predicate": str(item.get("predicate") or "passed"),
                "expected": expected,
            }
        )
        if len(manifest) >= MAX_MANIFEST_ITEMS:
            break
    return manifest


# ---------------------------------------------------------------------------
# Citations: the commands a claim rests on, matched to ledger rows by code
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")
_ROW_ID_RE = re.compile(r"p?[ck]\d+", re.IGNORECASE)
_PROMPT_RE = re.compile(r"^\$\s+")
_SHELL_WRAPPER_RE = re.compile(r"^(?:\S*/)?(?:ba|z|da)?sh\s+-l?c\s")


def cite(row: Dict[str, Any]) -> str:
    """How a row is named in text the model reads: by its command, never by
    its id, because the model is never shown ids and cites what it reads."""
    command = " ".join(str(row.get("command_preview") or row.get("command") or "").split())
    named = f"`{_clip(command, 80)}`" if command else "a command with no text"
    return f"the gate's run of {named}" if row.get("source") == "controller" else named


def unwrap_shell(command: str) -> str:
    """The inner command of a ``/bin/bash -lc "..."`` wrapper, else the command.

    The Codex lane records every command in that wrapper with the inner
    quotes escaped, while the model cites what it wrote inside it.
    """
    text = (command or "").strip()
    if not _SHELL_WRAPPER_RE.match(text):
        return text
    try:
        parts = shlex.split(text)
    except ValueError:
        return text
    return parts[2] if len(parts) == 3 else text


@functools.lru_cache(maxsize=4096)
def command_key(command: str) -> str:
    """The form commands are matched in: a shell wrapper unwrapped, whitespace
    collapsed, a leading ``cd X &&`` and trailing output filters dropped."""
    text = _WHITESPACE_RE.sub(" ", unwrap_shell(command)).strip()
    prefix = _CD_PREFIX_RE.match(text)
    if prefix:
        text = text[prefix.end():].strip()
    return strip_filters(text)


def is_row_id(text: str) -> bool:
    """Whether a citation is a ledger row id (c7, k58, pc3) rather than a command."""
    return bool(_ROW_ID_RE.fullmatch((text or "").strip()))


def match_command(cited: str, rows: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], str]:
    """(row, how) for one cited command, searching ``rows`` in order (newest first).

    An exact match of the matching form wins wherever it is; failing that,
    the first row whose command contains the citation, when the citation is
    long enough to be distinctive. Backticks and a ``$ `` prompt around the
    citation are ignored. ``how`` is ``exact``, ``partial`` or ``unmatched``.
    """
    key = command_key(_PROMPT_RE.sub("", (cited or "").strip().strip("`").strip()))
    if not key:
        return None, "unmatched"
    keyed = [(row, command_key(str(row.get("command") or ""))) for row in rows]
    for row, candidate in keyed:
        if candidate == key:
            return row, "exact"
    if len(key) >= MIN_PARTIAL_CITATION_CHARS:
        for row, candidate in keyed:
            if key in candidate:
                return row, "partial"
    return None, "unmatched"


def resolve_evidence(
    item: Dict[str, Any], rows: List[Dict[str, Any]], read: Callable[[Dict[str, Any]], Optional[str]],
) -> Dict[str, Any]:
    """The rows one manifest item rests on: ``{"ids", "citations", "problem", "row_ids"}``.

    ``rows`` is the search order, newest first: this turn's rows, then the
    previous turn's, with the gate's own rows left out. Each citation is a
    command as the agent ran it, or a distinctive part of one
    (``match_command``), so the newest run of a re-run check is the one
    that counts. A row id is refused: the agent is never shown one, so an
    id it cites is a guess, and guesses one row off made correct claims
    read as contradicted (2026-09-30). A ``contains`` claim whose text is in
    no matched row's output takes the newest row that printed it, because
    the quoted text is the evidence itself. ``problem`` is the verdict when
    no row can be judged: ``missing`` for a command that did not run or a
    text no output shows, ``insufficient`` for row ids alone.
    """
    ids: List[str] = []
    citations: List[Dict[str, Any]] = []
    unmatched: List[str] = []
    row_ids: List[str] = []
    for cited in item.get("evidence") or []:
        cited = str(cited).strip()
        if is_row_id(cited):
            row_ids.append(cited)
            citations.append({"cited": cited, "row": None, "how": "row_id"})
            continue
        row, how = match_command(cited, rows)
        citations.append({"cited": _clip(cited, 120), "row": row["id"] if row else None, "how": how})
        if row is None:
            unmatched.append(cited)
        elif row["id"] not in ids:
            ids.append(row["id"])
    needle = str((item.get("expected") or {}).get("text") or "").strip()
    if str(item.get("predicate") or "passed") == "contains" and needle:
        by_id = {row["id"]: row for row in rows}
        showing = [row_id for row_id in ids if needle in (read(by_id[row_id]) or "")]
        if not showing:
            located = next((row for row in rows if needle in (read(row) or "")), None)
            if located is not None:
                showing = [located["id"]]
                citations.append({"cited": _clip(needle, 120), "row": located["id"], "how": "text"})
            elif not ids:
                missing = {"verdict": "missing", "detail": f"no command output this turn contains {_clip(needle, 80)!r}"}
                return {"ids": [], "citations": citations, "problem": missing, "row_ids": row_ids}
        if showing:
            # The text is shown: a citation that missed does not matter.
            ids, unmatched, row_ids = showing, [], []
    problem: Optional[Dict[str, str]] = None
    if unmatched:
        problem = {"verdict": "missing", "detail": "no command this turn matches " + ", ".join(f"`{_clip(' '.join(text.split()), 80)}`" for text in unmatched)}
    elif row_ids and not ids:
        problem = {"verdict": "insufficient", "detail": f"cites ledger row ids ({', '.join(row_ids)}), which are never shown to the agent, instead of commands"}
    return {"ids": ids, "citations": citations, "problem": problem, "row_ids": row_ids}


def check_assertion(
    item: Dict[str, Any], rows: Dict[str, Dict[str, Any]], final_digest: Optional[str], stale_ids: Optional[set] = None,
) -> Dict[str, Any]:
    """Compare one manifest item with its cited rows: ``{"verdict", "detail", "rows", "basis"}``.

    Missing rows and stale rows are decided before the predicate, because
    no comparison against evidence that is absent or superseded means
    anything. ``insufficient`` is the honest answer when the row's status
    is unknown; it is never treated as a contradiction. A ``passed`` or
    ``count`` claim needs rows that are runner-shaped commands with a
    recognised runner summary: an exit-zero ``echo`` or wrapper script is
    insufficient, never supported. ``basis`` is ``gate`` when every cited
    row was produced by the gate's own re-run, else ``agent``.
    """
    verdict = _decide_assertion(item, rows, final_digest, stale_ids)
    ids = [row_id for row_id in verdict.get("rows") or [] if row_id in rows]
    verdict["basis"] = "gate" if ids and all(rows[row_id].get("source") == "controller" for row_id in ids) else "agent"
    return verdict


def _decide_assertion(
    item: Dict[str, Any], rows: Dict[str, Dict[str, Any]], final_digest: Optional[str], stale_ids: Optional[set] = None,
) -> Dict[str, Any]:
    ids = list(item.get("evidence") or [])
    predicate = str(item.get("predicate") or "passed")
    if predicate not in PREDICATES:
        return {"verdict": "insufficient", "detail": f"unknown predicate {predicate!r}", "rows": ids}
    if not ids:
        return {"verdict": "insufficient", "detail": "no command cited", "rows": []}
    missing = [row_id for row_id in ids if row_id not in rows]
    if missing:
        return {"verdict": "missing", "detail": f"cited rows not in this turn's ledger: {', '.join(missing)}", "rows": ids}
    cited = [rows[row_id] for row_id in ids]
    if predicate != "ran" and (stale_ids is not None or final_digest):
        # The caller decides per row (row_is_stale) when it can; a single
        # combined digest is the fallback for rows without a per-repository map.
        if stale_ids is not None:
            stale = [row for row in cited if row["id"] in stale_ids]
        else:
            stale = [row for row in cited if row.get("workspace") and row["workspace"] != final_digest]
        if stale:
            return {"verdict": "stale", "detail": f"ran before later edits: {', '.join(cite(row) for row in stale)}", "rows": ids}
    if predicate == "ran":
        return {"verdict": "supported", "detail": "rows exist", "rows": ids}
    if predicate in ("passed", "count"):
        not_checks = [row for row in cited if not row.get("check")]
        if not_checks:
            return {"verdict": "insufficient", "detail": f"not a check runner: {', '.join(cite(row) for row in not_checks)}", "rows": ids}
    if predicate == "passed":
        failed = [row for row in cited if row["status"] == "fail"]
        if failed:
            return {"verdict": "contradicted", "detail": f"reported failure: {', '.join(cite(row) for row in failed)}", "rows": ids}
        if all(row["status"] == "pass" for row in cited):
            unrecognized = [row for row in cited if not row.get("recognized")]
            if unrecognized:
                return {"verdict": "insufficient", "detail": f"no runner summary recognised: {', '.join(cite(row) for row in unrecognized)}", "rows": ids}
            return {"verdict": "supported", "detail": "every cited row reports a pass", "rows": ids}
        unknown = [row for row in cited if row["status"] == "unknown"]
        return {"verdict": "insufficient", "detail": f"status unknown: {', '.join(cite(row) for row in unknown)}", "rows": ids}
    if predicate == "exit_zero":
        bad = [f"{cite(row)} exited {row.get('exit')}" for row in cited if row.get("exit") not in (0, None)]
        if bad:
            return {"verdict": "contradicted", "detail": f"non-zero exit: {', '.join(bad)}", "rows": ids}
        if all(row.get("exit") == 0 for row in cited):
            return {"verdict": "supported", "detail": "exit 0", "rows": ids}
        return {"verdict": "insufficient", "detail": "exit code unknown on this lane", "rows": ids}
    if predicate == "count":
        expected = {key: value for key, value in (item.get("expected") or {}).items() if isinstance(value, int) and not isinstance(value, bool)}
        if not expected:
            return {"verdict": "insufficient", "detail": "count predicate without expected counts", "rows": ids}
        if not all(row.get("recognized") for row in cited):
            return {"verdict": "insufficient", "detail": "no runner summary was recognised in the cited output", "rows": ids}
        totals: Dict[str, int] = {}
        for row in cited:
            for key, value in (row.get("counts") or {}).items():
                totals[key] = totals.get(key, 0) + int(value)
        mismatched = [f"{key}: claimed {value}, ledger {totals.get(key, 0)}" for key, value in expected.items() if totals.get(key, 0) != value]
        if mismatched:
            return {"verdict": "contradicted", "detail": "; ".join(mismatched), "rows": ids}
        return {"verdict": "supported", "detail": "counts match", "rows": ids}
    if predicate == "contains":
        needle = str((item.get("expected") or {}).get("text") or "").strip()
        if not needle:
            return {"verdict": "insufficient", "detail": "contains predicate without expected.text", "rows": ids}
        outputs = [read_output(row.get("file")) for row in cited]
        if any(output is None for output in outputs):
            return {"verdict": "insufficient", "detail": "retained output unavailable", "rows": ids}
        if any(needle in (output or "") for output in outputs):
            return {"verdict": "supported", "detail": "text found in retained output", "rows": ids}
        return {"verdict": "contradicted", "detail": f"text not in retained output of {', '.join(cite(row) for row in cited)}: {_clip(needle, 80)!r}", "rows": ids}
    return {"verdict": "insufficient", "detail": "not checked", "rows": ids}


# ---------------------------------------------------------------------------
# Controller re-runs: the gate runs a plain check itself
# ---------------------------------------------------------------------------

_SEGMENT_SPLIT_RE = re.compile(r"\s*(?:&&|\|\||;|\|)\s*")
_SAFE_SEGMENT_RE = re.compile(
    r"^(?:[A-Za-z_][A-Za-z0-9_]*=\S*\s+)*"
    r"(?:time\s+)?(?:timeout\s+\d+[smh]?\s+)?"
    r"(?:"
    r"cd\s+\S+"
    r"|(?:\S*/)?(?:python3?|py)(?:\s+-[a-zA-Z]+)*\s+-m\s+pytest\b.*"
    r"|(?:\S*/)?pytest\b.*"
    r"|npm\s+(?:run\s+)?(?:test|lint|build|typecheck|type-check|check|verify)(?::[\w.-]+)?(?:\s+--prefix\s+\S+)?(?:\s+--\s+.*)?(?:\s+--[\w=-]+)*"
    r"|pnpm\s+(?:run\s+)?(?:test|lint|build|typecheck|type-check|check)\b.*"
    r"|yarn\s+(?:run\s+)?(?:test|lint|build|typecheck|type-check|check)\b.*"
    r"|npx\s+(?:vitest|jest|tsc|eslint|playwright\s+test|prettier\s+--check)\b.*"
    r"|node\s+(?:--[\w-]+(?:=\S+)?\s+)*--test\b.*"
    r"|(?:\S*/)?(?:vitest|jest|tsc|eslint|ruff|mypy|flake8)\b.*"
    r"|go\s+test\b.*"
    r"|cargo\s+(?:test|check|clippy)\b.*"
    r"|make\s+(?:test|check|lint)\b"
    r"|(?:tail|head)\s+(?:-n\s*)?-?\d+"
    r"|grep\s+(?:-[a-zA-Z]+\s+)*(?:\"[^\"]*\"|'[^']*'|\S+)"
    r"|wc\s+-[lwc]"
    r"|sort|uniq"
    r")$"
)
_UNSAFE_RE = re.compile(r"`|\$\(|(?<![2&])>(?!&1|/dev/null)|<<")


_PIPE_SPLIT_RE = re.compile(r"\s*(?<!\|)\|(?!\|)\s*")
_FILTER_SEGMENT_RE = re.compile(
    r"^(?:(?:tail|head)\s+(?:-n\s*)?-?\d+|grep\s+(?:-[a-zA-Z]+\s+)*(?:\"[^\"]*\"|'[^']*'|\S+)|wc\s+-[lwc]|sort|uniq)$"
)


def strip_filters(command: str) -> str:
    """The command without trailing output filters, so a re-run sees the whole summary.

    ``pytest -q | grep -v failed`` becomes ``pytest -q``. A pipe into anything
    that is not a plain filter is left alone (and is not re-runnable anyway).
    """
    command = (command or "").strip()
    parts = _PIPE_SPLIT_RE.split(command)
    if len(parts) < 2:
        return command
    if all(_FILTER_SEGMENT_RE.match(part.strip()) for part in parts[1:]):
        return parts[0].strip()
    return command


def tests_run(row: Dict[str, Any]) -> Optional[int]:
    """How many tests a recognised test-runner row reports having run, or None."""
    if not row.get("recognized") or row.get("kind") not in ("pytest", "node-test", "jest"):
        return None
    counts = row.get("counts") or {}
    if row.get("kind") == "node-test" and isinstance(counts.get("tests"), int):
        return int(counts["tests"])
    keys = ("passed", "failed", "errors", "skipped", "xfailed", "xpassed")
    if not any(key in counts for key in keys):
        return None
    return sum(int(counts.get(key, 0)) for key in keys)


_TEST_FILE_RE = re.compile(
    r"(^|/)(tests?|__tests__|spec)/|(^|/)test_[^/]*\.py$|_test\.py$|\.(test|spec)\.[cm]?[jt]sx?$|(^|/)conftest\.py$|_test\.go$",
    re.IGNORECASE,
)
_REMOVED_TEST_LINE_RE = re.compile(
    r"^-\s*(?:assert\b|self\.assert\w*\(|def test_|async def test_|it\(|test\(|expect\(|assert_eq!|assert!|t\.Errorf|t\.Fatal)"
)
_ADDED_SKIP_RE = re.compile(
    r"^\+.*(?:pytest\.mark\.skip|pytest\.mark\.xfail|@unittest\.skip|pytest\.skip\(|\.skip\(|\.only\(|\bxit\(|\bxdescribe\(|\bxtest\(|\btest\.skip|\bit\.skip|\bdescribe\.skip|t\.Skip\(|#\[ignore\])"
)


def _git_ok(root: str, args: List[str]) -> bool:
    try:
        return subprocess.run(["git", "-C", root, *args], capture_output=True, text=True, timeout=5.0, check=False).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def weakening_signals(root: Optional[str], changed_paths: List[str], base: str = "HEAD") -> Dict[str, Any]:
    """Removed assertion or test lines, and added skip markers, in test files that existed at ``base``.

    ``base`` is the commit the turn started from, so a weakening the turn
    committed before finishing is still counted; HEAD when it is unknown.

    Deterministic and coarse: it catches a deleted assertion and a skipped
    test, not a loosened expected value. The count is the net loss of
    assertion or test lines, so a moved test does not count. New test files
    are never counted, because adding tests is the normal case.
    """
    result: Dict[str, Any] = {"files": [], "removed": 0, "skips": 0}
    if not root:
        return result
    for path in changed_paths[:40]:
        if not os.path.abspath(path).startswith(root + os.sep):
            continue
        rel = os.path.relpath(path, root)
        if not _TEST_FILE_RE.search(rel.replace(os.sep, "/")):
            continue
        if not _git_ok(root, ["cat-file", "-e", f"{base}:{rel}"]):
            continue
        diff = _git(root, ["diff", base, "--", rel]) or ""
        lines = diff.splitlines()
        # Net loss: a line the diff algorithm removes and re-adds (a moved
        # test) is not a weakening, so added assertion lines cancel removed ones.
        gone = sum(1 for line in lines if line.startswith("-") and not line.startswith("---") and _REMOVED_TEST_LINE_RE.match(line))
        back = sum(1 for line in lines if line.startswith("+") and not line.startswith("+++") and _REMOVED_TEST_LINE_RE.match("-" + line[1:]))
        removed = max(0, gone - back)
        skips = sum(1 for line in lines if line.startswith("+") and not line.startswith("+++") and _ADDED_SKIP_RE.match(line))
        if removed or skips:
            result["files"].append({"path": rel, "removed": removed, "skips": skips})
            result["removed"] += removed
            result["skips"] += skips
    return result


def rerunnable(command: str) -> bool:
    """Whether the gate may run this command itself: every segment is a check runner, a cd, or a harmless filter."""
    command = (command or "").strip()
    if not command or _UNSAFE_RE.search(command):
        return False
    segments = [segment for segment in _SEGMENT_SPLIT_RE.split(command) if segment]
    if not segments or not any(CHECK_COMMAND_RE.search(segment) for segment in segments):
        return False
    return all(_SAFE_SEGMENT_RE.match(segment) for segment in segments)


# The agent's commands ran under bash, so the gate re-runs them under bash
# too. /bin/sh is dash on Debian: it has no ``time`` keyword, and with no
# /usr/bin/time a whitelisted ``time pytest`` exited 127 there, which the
# ledger would have read as a failing check.
GATE_SHELL: Optional[str] = "/bin/bash" if os.path.exists("/bin/bash") else None


def run_check(command: str, cwd: str, timeout: float) -> Dict[str, Any]:
    """Run one check for the gate: ``{"output", "exit_code", "timed_out", "seconds"}``."""
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command, shell=True, executable=GATE_SHELL, cwd=cwd or None, capture_output=True, text=True, timeout=timeout,
            check=False, env={**os.environ, "HERMES_CONTROLLER_RERUN": "1"},
        )
        output = (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")
        return {"output": output, "exit_code": completed.returncode, "timed_out": False, "seconds": time.monotonic() - started}
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")) or ""
        return {"output": output, "exit_code": None, "timed_out": True, "seconds": time.monotonic() - started}
    except OSError as exc:
        return {"output": f"{exc.__class__.__name__}: {exc}", "exit_code": None, "timed_out": False, "seconds": time.monotonic() - started}


def manifest_counts(verdicts: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {name: 0 for name in VERDICTS}
    for item in verdicts:
        verdict = str(item.get("verdict") or "insufficient")
        counts[verdict] = counts.get(verdict, 0) + 1
    return counts
