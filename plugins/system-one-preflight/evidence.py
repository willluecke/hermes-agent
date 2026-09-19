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
* a workspace digest, from ``git status`` plus the size and mtime of every
  changed path, so a check that ran before a later edit is stale by code
  rather than by judgment;
* the result manifest the model registers with ``report_results``: each
  claim names the rows it rests on and a predicate code can compare
  exactly, and the verdict is ``supported``, ``contradicted``, ``stale``,
  ``insufficient`` or ``missing``;
* controller re-runs: a cited check that is stale or whose exit is unknown
  is run again by the gate itself, never by the model, when the command is
  a plain check runner and the cwd is known.

Nothing here calls Jev. Everything is deterministic given the tool results,
and every limit is stated as a constant.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PARSER_VERSION = 1
MAX_ROWS = 200
MAX_PREVIOUS_ROWS = 50
MAX_OUTPUT_CHARS = 200_000
MAX_EXCERPT_CHARS = 1_200
EXCERPT_BEFORE = 400
MAX_EXCERPTS = 4
MAX_MANIFEST_ITEMS = 16
MAX_CLAIM_CHARS = 300
MAX_COMMAND_CHARS = 300
STATUSES = ("pass", "fail", "unknown")
VERDICTS = ("supported", "contradicted", "stale", "insufficient", "missing")
PREDICATES = ("passed", "count", "exit_zero", "contains", "ran")
# A decisive claim is one the gate must back with its own run: the agent's
# row is advisory, because a command can print the summary that clears it.
DECISIVE = ("passed", "count", "exit_zero")

CHECK_COMMAND_RE = re.compile(
    r"\b(pytest|npm (run )?(test|lint|build|typecheck)|pnpm (test|lint|build)|"
    r"yarn (test|lint|build)|vitest|jest|mocha|go test|cargo (test|check|clippy)|"
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


def cd_prefix(command: str) -> Optional[str]:
    """The directory a ``cd X && ...`` command starts in, or None."""
    match = _CD_PREFIX_RE.match(command or "")
    if not match:
        return None
    return match.group(1).strip("\"'")


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
        "command": _clip(command, MAX_COMMAND_CHARS),
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
        "command": row["command"],
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

def parse_manifest(text: str) -> Optional[List[Dict[str, Any]]]:
    """The ``report_results`` tool's result, raw or wrapped by the MCP bridge as {"result": "<json>"}."""
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return None
    if isinstance(payload, dict) and "manifest" not in payload and isinstance(payload.get("result"), str):
        try:
            payload = json.loads(payload["result"])
        except (TypeError, ValueError):
            return None
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
        evidence = item.get("evidence") if isinstance(item.get("evidence"), list) else []
        expected = item.get("expected") if isinstance(item.get("expected"), dict) else {}
        manifest.append(
            {
                "id": str(item.get("id") or f"r{len(manifest) + 1}"),
                "criterion": str(item.get("criterion") or ""),
                "claim": _clip(claim, MAX_CLAIM_CHARS),
                "evidence": [str(value) for value in evidence if str(value).strip()][:8],
                "predicate": str(item.get("predicate") or "passed"),
                "expected": expected,
            }
        )
        if len(manifest) >= MAX_MANIFEST_ITEMS:
            break
    return manifest


def check_assertion(item: Dict[str, Any], rows: Dict[str, Dict[str, Any]], final_digest: Optional[str]) -> Dict[str, Any]:
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
    verdict = _decide_assertion(item, rows, final_digest)
    ids = [row_id for row_id in verdict.get("rows") or [] if row_id in rows]
    verdict["basis"] = "gate" if ids and all(rows[row_id].get("source") == "controller" for row_id in ids) else "agent"
    return verdict


def _decide_assertion(item: Dict[str, Any], rows: Dict[str, Dict[str, Any]], final_digest: Optional[str]) -> Dict[str, Any]:
    ids = list(item.get("evidence") or [])
    predicate = str(item.get("predicate") or "passed")
    if predicate not in PREDICATES:
        return {"verdict": "insufficient", "detail": f"unknown predicate {predicate!r}", "rows": ids}
    if not ids:
        return {"verdict": "insufficient", "detail": "no evidence rows cited", "rows": []}
    missing = [row_id for row_id in ids if row_id not in rows]
    if missing:
        return {"verdict": "missing", "detail": f"cited rows not in this turn's ledger: {', '.join(missing)}", "rows": ids}
    cited = [rows[row_id] for row_id in ids]
    if predicate != "ran" and final_digest:
        stale = [row["id"] for row in cited if row.get("workspace") and row["workspace"] != final_digest]
        if stale:
            return {"verdict": "stale", "detail": f"ran before later edits: {', '.join(stale)}", "rows": ids}
    if predicate == "ran":
        return {"verdict": "supported", "detail": "rows exist", "rows": ids}
    if predicate in ("passed", "count"):
        not_checks = [row["id"] for row in cited if not row.get("check")]
        if not_checks:
            return {"verdict": "insufficient", "detail": f"not a check runner: {', '.join(not_checks)}", "rows": ids}
    if predicate == "passed":
        statuses = {row["id"]: row["status"] for row in cited}
        if any(status == "fail" for status in statuses.values()):
            failed = [row_id for row_id, status in statuses.items() if status == "fail"]
            return {"verdict": "contradicted", "detail": f"reported failure: {', '.join(failed)}", "rows": ids}
        if all(status == "pass" for status in statuses.values()):
            unrecognized = [row["id"] for row in cited if not row.get("recognized")]
            if unrecognized:
                return {"verdict": "insufficient", "detail": f"no runner summary recognised: {', '.join(unrecognized)}", "rows": ids}
            return {"verdict": "supported", "detail": "every cited row reports a pass", "rows": ids}
        unknown = [row_id for row_id, status in statuses.items() if status == "unknown"]
        return {"verdict": "insufficient", "detail": f"status unknown: {', '.join(unknown)}", "rows": ids}
    if predicate == "exit_zero":
        codes = {row["id"]: row.get("exit") for row in cited}
        if any(code not in (0, None) for code in codes.values()):
            bad = [f"{row_id}={code}" for row_id, code in codes.items() if code not in (0, None)]
            return {"verdict": "contradicted", "detail": f"non-zero exit: {', '.join(bad)}", "rows": ids}
        if all(code == 0 for code in codes.values()):
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
        return {"verdict": "contradicted", "detail": f"text not in retained output: {_clip(needle, 80)!r}", "rows": ids}
    return {"verdict": "insufficient", "detail": "not checked", "rows": ids}


# ---------------------------------------------------------------------------
# Controller re-runs: the gate runs a plain check itself
# ---------------------------------------------------------------------------

_SEGMENT_SPLIT_RE = re.compile(r"\s*(?:&&|\|\||;|\|)\s*")
_SAFE_SEGMENT_RE = re.compile(
    r"^(?:[A-Za-z_][A-Za-z0-9_]*=\S*\s+)*"
    r"(?:timeout\s+\d+[smh]?\s+)?"
    r"(?:"
    r"cd\s+\S+"
    r"|(?:\S*/)?(?:python3?|py)(?:\s+-[a-zA-Z]+)*\s+-m\s+pytest\b.*"
    r"|(?:\S*/)?pytest\b.*"
    r"|npm\s+(?:run\s+)?(?:test|lint|build|typecheck)(?::[\w.-]+)?(?:\s+--\s+.*)?(?:\s+--[\w=-]+)*"
    r"|pnpm\s+(?:test|lint|build)\b.*"
    r"|yarn\s+(?:test|lint|build)\b.*"
    r"|npx\s+(?:vitest|jest|tsc|eslint|prettier\s+--check)\b.*"
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


def weakening_signals(root: Optional[str], changed_paths: List[str]) -> Dict[str, Any]:
    """Removed assertion or test lines, and added skip markers, in test files that existed at HEAD.

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
        if not _git_ok(root, ["cat-file", "-e", f"HEAD:{rel}"]):
            continue
        diff = _git(root, ["diff", "HEAD", "--", rel]) or ""
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


def run_check(command: str, cwd: str, timeout: float) -> Dict[str, Any]:
    """Run one check for the gate: ``{"output", "exit_code", "timed_out", "seconds"}``."""
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command, shell=True, cwd=cwd or None, capture_output=True, text=True, timeout=timeout, check=False,
            env={**os.environ, "HERMES_CONTROLLER_RERUN": "1"},
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
