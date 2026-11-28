"""Kanban Work Budget & Time-Aware Multi-Agent Sizing (w / x / y).

Computes work quantity ``w``, calibrates empirical shard timings from a board's
real SQLite ``task_runs`` history, solves for the multi-agent worker count ``x``
needed to complete ``w`` within a target time budget ``y`` (reserving a
protected synthesis/verification tail), and tracks closed-loop budget burn.

Three workloads are modeled cleanly without claiming precision the data does not
support:

1. **Empirical board calibration** (:func:`calibrate_board`): reads active run
   durations from ``task_runs`` (never conflated with wall-clock human wait) and
   reports quantiles (p50, p85, p90) per task class (e.g. ``auto-decomposer``,
   ``discovery-sweep``, ``feature-pipeline:region``). Planning uses tail
   quantiles (p85 by default) because parallel fan-out finishes at the slowest
   straggler, not the mean worker.
2. **Surface-sweep sizing** (:func:`plan_surface_swarm`): partitions an
   explicitly enumerable surface (URLs, files, items) into ``x`` balanced worker
   shards to fit inside ``target_minutes`` after carving out a protected
   verification + synthesis tail reserve, capped by concurrency/rate limits.
   Reports honest feasibility (``feasible``, ``bottleneck``, and cut-list scope
   if the ceiling cannot cover the full surface in ``y``).
3. **Codebase DAG sizing & wave scheduling** (:func:`plan_dag_budget`,
   :func:`blast_radius_for_paths`): takes a DAG of plan items with explicit
   file lists and ``depends_on`` edges, keeps file-overlapping items inside one
   region (to avoid merge conflicts), and arranges *disjoint* dependent regions
   into topological execution waves rather than collapsing the whole connected
   component into one serial worker. Computes ex-ante reverse-import blast
   radius (JS/TS resolved aliases + Python module imports, ignoring mirror trees
   and build artifacts), critical-path wave depth, required concurrent workers
   ``x``, and per-region ``max_runtime_seconds`` cutoffs.
4. **Closed-loop reprojection** (:func:`reproject_budget`): given completed vs
   remaining work and elapsed wall time, updates the empirical burn rate and
   returns an explicit control action (``on_track``, ``scale_up``,
   ``tighten_scope``, or ``harvest_now``).

Pure/read-only by design: this module never mutates repositories or dispatches
workers on its own.
"""

from __future__ import annotations

import collections
import json
import math
import os
import re
import sqlite3
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Default fallback shard timings (minutes) anchored to measured Hermes Kanban
# history (2026-11-27 live calibration across 109 leaf tasks: auto-decomposer
# p50=7.15m, p85=9.83m, first-run p50=5.02m, p85=7.32m).
DEFAULT_SHARD_P50_MIN = 7.2
DEFAULT_SHARD_P85_MIN = 9.8
DEFAULT_FIRST_RUN_P50_MIN = 5.0
DEFAULT_FIRST_RUN_P85_MIN = 7.3
DEFAULT_SYNTHESIS_RESERVE_FRACTION = 0.25
MIN_RESERVE_MIN = 1.5
MIN_WORKER_RUNTIME_SECONDS = 90
# Extra buffer so a worker hits its partial-handoff rule cleanly before kill.
HANDOFF_GRACE_SECONDS = 45

GRAPH_EXTS = frozenset({".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".py", ".go", ".rs"})
GRAPH_SKIP_DIRS = frozenset({
    ".git",
    "node_modules",
    "dist",
    "build",
    ".next",
    "__pycache__",
    ".venv",
    "venv",
    "vendor",
    "coverage",
    ".worktrees",
    ".codex-worktrees",
    "devsession",
    ".archive",
})
TEST_PATH_RE = re.compile(
    r"(^|/)(tests?|__tests__|spec|specs|e2e)(/|$)|\.(test|spec)\.|_test\.(go|py)$|test_.*\.py$",
    re.I,
)
ROUTE_PATH_RE = re.compile(
    r"(^|/)(app|pages|routes|api)(/|$)|route\.(ts|tsx|js)$|urls\.py$",
    re.I,
)
MIGRATION_PATH_RE = re.compile(
    r"(^|/)(migrations?|supabase/migrations|db/migrate)(/|$)|\.sql$",
    re.I,
)
BODY_PATH_RE = re.compile(
    r"(?:src|tests?|scripts|docs|app|pages|lib|components|supabase|hermes_cli|tools|extensions|packages|website|site)/"
    r"[A-Za-z0-9_./-]+\.[A-Za-z0-9]+"
)
WORD_RE = re.compile(r"[A-Za-z0-9_]+")


# ---------------------------------------------------------------------------
# Pure quantile / statistics helpers
# ---------------------------------------------------------------------------

def _quantile(sorted_vals: Sequence[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = max(0.0, min(1.0, q)) * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def summarize_durations(values: Sequence[float]) -> Dict[str, float]:
    clean = sorted(float(v) for v in values if v is not None and float(v) >= 0.0)
    if not clean:
        return {"n": 0}
    return {
        "n": len(clean),
        "p50": round(_quantile(clean, 0.50), 2),
        "p75": round(_quantile(clean, 0.75), 2),
        "p85": round(_quantile(clean, 0.85), 2),
        "p90": round(_quantile(clean, 0.90), 2),
        "max": round(clean[-1], 2),
    }


# ---------------------------------------------------------------------------
# Repo reverse-import blast radius (primary tree only, read-only)
# ---------------------------------------------------------------------------

def _strip_jsonc_comments(text: str) -> str:
    """Strip // and /* */ comments while preserving string literals (e.g. URLs)."""
    out: List[str] = []
    i = 0
    n = len(text)
    in_str = False
    esc = False
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and nxt == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if ch == "/" and nxt == "*":
            end = text.find("*/", i + 2)
            i = n if end == -1 else end + 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _load_alias_mappings(repo: Path) -> List[Tuple[str, str, str]]:
    mappings: List[Tuple[str, str, str]] = []
    for name in ("tsconfig.json", "jsconfig.json"):
        fp = repo / name
        if not fp.is_file():
            continue
        try:
            raw = fp.read_text(encoding="utf-8", errors="ignore")
            data = json.loads(_strip_jsonc_comments(raw))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        compiler = data.get("compilerOptions") or {}
        if not isinstance(compiler, dict):
            continue
        paths = compiler.get("paths") or {}
        base_url = str(compiler.get("baseUrl") or ".")
        if not isinstance(paths, dict):
            continue
        for pat, targets in paths.items():
            if not isinstance(pat, str) or not isinstance(targets, list) or not targets:
                continue
            tgt = targets[0]
            if not isinstance(tgt, str):
                continue
            if pat.endswith("*") and tgt.endswith("*"):
                mappings.append(("prefix", pat[:-1], os.path.normpath(os.path.join(base_url, tgt[:-1]))))
            elif "*" not in pat and "*" not in tgt:
                mappings.append(("exact", pat, os.path.normpath(os.path.join(base_url, tgt))))
    return mappings


def _resolve_alias(spec: str, mappings: Sequence[Tuple[str, str, str]]) -> Optional[str]:
    for kind, pat, tgt in mappings:
        if kind == "prefix" and spec.startswith(pat):
            return os.path.normpath(os.path.join(tgt, spec[len(pat):]))
        if kind == "exact" and spec == pat:
            return os.path.normpath(tgt)
    return None


def build_import_graph(repo: str | Path, *, max_files: int = 20_000) -> Dict[str, List[str]]:
    """Build forward local import edges for a repository's primary tree only."""
    root = Path(repo).resolve()
    if not root.is_dir():
        return {}
    aliases = _load_alias_mappings(root)
    edges: Dict[str, List[str]] = {}
    scanned = 0
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in GRAPH_SKIP_DIRS)
        for fn in sorted(files):
            ext = os.path.splitext(fn)[1].lower()
            if ext not in GRAPH_EXTS:
                continue
            scanned += 1
            if scanned > max_files:
                return edges
            fp = os.path.join(dirpath, fn)
            rel = os.path.relpath(fp, root).replace("\\", "/")
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
                    text = fh.read(200_000)
            except OSError:
                continue
            cands: List[str] = []
            if ext == ".py":
                for m in re.finditer(r"^\s*(?:from|import)\s+([A-Za-z0-9_.]+)", text, re.M):
                    mod = m.group(1)
                    cands.append(mod.replace(".", "/") + ".py")
                    cands.append(mod.replace(".", "/") + "/__init__.py")
            elif ext in (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"):
                for m in re.finditer(r"""(?:import|require|from)\s*\(?\s*['"]([^'"]+)['"]""", text):
                    spec = m.group(1)
                    if spec.startswith("."):
                        base = os.path.normpath(os.path.join(os.path.dirname(rel), spec))
                    else:
                        alias = _resolve_alias(spec, aliases)
                        if alias is None:
                            continue
                        base = alias
                    for suf in ("", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", "/index.ts", "/index.tsx", "/index.js"):
                        cands.append(os.path.normpath(base + suf).replace("\\", "/"))
            resolved = [c for c in dict.fromkeys(cands) if not c.startswith("..") and (root / c).is_file()]
            if resolved:
                edges[rel] = resolved
    return edges


def blast_radius_for_paths(
    repo: str | Path,
    seeds: Sequence[str],
    *,
    max_depth: int = 4,
    max_nodes: int = 20_000,
    edges: Optional[Mapping[str, Sequence[str]]] = None,
) -> Dict[str, Any]:
    """Reverse import closure within ``max_depth`` hops of ``seeds``.

    Accepts files or directory prefixes in ``seeds`` (a directory seed expands
    to all primary-tree source files under that prefix).
    """
    root = Path(repo).resolve()
    clean_seeds = [os.path.normpath(str(s).strip()).replace("\\", "/") for s in seeds if str(s or "").strip()]
    clean_seeds = [s for s in clean_seeds if s and not s.startswith("..")]
    if not clean_seeds or not root.is_dir():
        return {
            "seed_files": 0,
            "blast_radius_files": 0,
            "blast_radius_tests": 0,
            "blast_radius_routes": 0,
            "blast_radius_migrations": 0,
            "graph_files_scanned": 0,
            "graph_basis": "none",
        }
    graph = edges if edges is not None else build_import_graph(root)
    reverse: Dict[str, set] = {}
    all_nodes: set = set(graph.keys())
    for src, deps in graph.items():
        for dep in deps:
            all_nodes.add(dep)
            reverse.setdefault(dep, set()).add(src)

    expanded_seeds: set = set()
    for s in clean_seeds:
        if s == ".":
            expanded_seeds.update(all_nodes)
            continue
        if (root / s).is_dir():
            prefix = s.rstrip("/") + "/"
            matches = {n for n in all_nodes if n.startswith(prefix)}
            expanded_seeds.update(matches or {s})
        else:
            expanded_seeds.add(s)

    seen = set(expanded_seeds)
    frontier = list(expanded_seeds)
    depth = 0
    while frontier and depth < max_depth and len(seen) < max_nodes:
        nxt: set = set()
        for node in frontier:
            for parent in reverse.get(node, ()):
                if parent not in seen:
                    nxt.add(parent)
                    seen.add(parent)
        frontier = list(nxt)
        depth += 1

    return {
        "seed_files": len(expanded_seeds),
        "blast_radius_files": len(seen),
        "blast_radius_tests": sum(1 for p in seen if TEST_PATH_RE.search(p)),
        "blast_radius_routes": sum(1 for p in seen if ROUTE_PATH_RE.search(p)),
        "blast_radius_migrations": sum(1 for p in seen if MIGRATION_PATH_RE.search(p)),
        "graph_files_scanned": len(graph),
        "graph_basis": "resolved-imports(js/ts,py) / conservative(others)",
    }


def extract_body_paths(title: str, body: str) -> List[str]:
    return sorted(set(BODY_PATH_RE.findall(f"{title or ''}\n{body or ''}")))


# ---------------------------------------------------------------------------
# Live board calibration from SQLite task_runs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BoardCalibration:
    """Empirical active-runtime calibration derived from a board's completed runs."""

    board: str
    sample_count: int
    shard_p50_min: float
    shard_p85_min: float
    first_run_p50_min: float
    first_run_p85_min: float
    median_runs_per_task: float
    review_multiplier: float
    by_class: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    used_fallback: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "board": self.board,
            "sample_count": self.sample_count,
            "shard_p50_min": round(self.shard_p50_min, 2),
            "shard_p85_min": round(self.shard_p85_min, 2),
            "first_run_p50_min": round(self.first_run_p50_min, 2),
            "first_run_p85_min": round(self.first_run_p85_min, 2),
            "median_runs_per_task": round(self.median_runs_per_task, 2),
            "review_multiplier": round(self.review_multiplier, 2),
            "by_class": self.by_class,
            "used_fallback": self.used_fallback,
        }


def _classify_task_row(title: str, created_by: str) -> str:
    if title.startswith("Region "):
        return "feature-pipeline:region"
    if title.startswith("Integration fix "):
        return "feature-pipeline:fix"
    if title.startswith("Feature pipeline ("):
        return "feature-pipeline:root"
    return (created_by or "unknown").strip() or "unknown"


def calibrate_board(
    conn: sqlite3.Connection,
    *,
    board: str = "default",
    min_samples: int = 5,
    preferred_classes: Sequence[str] = ("auto-decomposer", "discovery-sweep", "codex-orchestrator"),
) -> BoardCalibration:
    """Measure active-run shard timing quantiles from a board's completed tasks.

    Excludes ``assignee='jim'`` and pipeline root orchestrator cards (which sit
    waiting on children) so the shard distribution reflects actual worker runs.
    Falls back to measured cross-board defaults when fewer than ``min_samples``
    completed leaf tasks exist on the board.
    """
    rows = conn.execute(
        """
        SELECT id, title, assignee, created_by, started_at, completed_at
          FROM tasks
         WHERE started_at IS NOT NULL
           AND completed_at IS NOT NULL
           AND completed_at >= started_at
           AND status = 'done'
        """
    ).fetchall()

    by_class_act: Dict[str, List[float]] = collections.defaultdict(list)
    by_class_first: Dict[str, List[float]] = collections.defaultdict(list)
    by_class_wall: Dict[str, List[float]] = collections.defaultdict(list)
    by_class_runs: Dict[str, List[int]] = collections.defaultdict(list)

    all_leaf_act: List[float] = []
    all_leaf_first: List[float] = []
    all_leaf_runs: List[int] = []

    for row in rows:
        tid = row["id"]
        title = row["title"] or ""
        assignee = (row["assignee"] or "").strip().lower()
        created_by = row["created_by"] or ""
        cls = _classify_task_row(title, created_by)
        if assignee == "jim" or cls == "feature-pipeline:root":
            continue
        runs = conn.execute(
            "SELECT started_at, ended_at FROM task_runs WHERE task_id = ? ORDER BY started_at ASC",
            (tid,),
        ).fetchall()
        durs = [
            (r["ended_at"] - r["started_at"]) / 60.0
            for r in runs
            if r["started_at"] is not None and r["ended_at"] is not None and r["ended_at"] >= r["started_at"]
        ]
        if not durs:
            continue
        act = sum(durs)
        first = durs[0]
        wall = (row["completed_at"] - row["started_at"]) / 60.0
        by_class_act[cls].append(act)
        by_class_first[cls].append(first)
        by_class_wall[cls].append(wall)
        by_class_runs[cls].append(len(durs))
        all_leaf_act.append(act)
        all_leaf_first.append(first)
        all_leaf_runs.append(len(durs))

    by_class: Dict[str, Dict[str, Any]] = {}
    for cls, vals in sorted(by_class_act.items(), key=lambda kv: -len(kv[1])):
        by_class[cls] = {
            "n": len(vals),
            "active_min": summarize_durations(vals),
            "first_run_min": summarize_durations(by_class_first[cls]),
            "wall_min": summarize_durations(by_class_wall[cls]),
            "runs_median": round(statistics.median(by_class_runs[cls]), 2),
        }

    # Prefer decomposed-shard classes for shard duration if enough samples exist,
    # otherwise use all leaf tasks on the board, otherwise measured defaults.
    shard_act: List[float] = []
    shard_first: List[float] = []
    shard_runs: List[int] = []
    for cls in preferred_classes:
        shard_act.extend(by_class_act.get(cls, ()))
        shard_first.extend(by_class_first.get(cls, ()))
        shard_runs.extend(by_class_runs.get(cls, ()))

    if len(shard_act) >= min_samples:
        src_act, src_first, src_runs = shard_act, shard_first, shard_runs
        used_fallback = False
    elif len(all_leaf_act) >= min_samples:
        src_act, src_first, src_runs = all_leaf_act, all_leaf_first, all_leaf_runs
        used_fallback = False
    else:
        return BoardCalibration(
            board=board,
            sample_count=len(all_leaf_act),
            shard_p50_min=DEFAULT_SHARD_P50_MIN,
            shard_p85_min=DEFAULT_SHARD_P85_MIN,
            first_run_p50_min=DEFAULT_FIRST_RUN_P50_MIN,
            first_run_p85_min=DEFAULT_FIRST_RUN_P85_MIN,
            median_runs_per_task=2.0,
            review_multiplier=round(DEFAULT_SHARD_P50_MIN / DEFAULT_FIRST_RUN_P50_MIN, 2),
            by_class=by_class,
            used_fallback=True,
        )

    s_act = sorted(src_act)
    s_first = sorted(src_first)
    p50 = max(0.5, _quantile(s_act, 0.50))
    p85 = max(p50, _quantile(s_act, 0.85))
    fp50 = max(0.5, _quantile(s_first, 0.50))
    fp85 = max(fp50, _quantile(s_first, 0.85))
    review_mult = max(1.0, min(3.0, p50 / fp50))
    return BoardCalibration(
        board=board,
        sample_count=len(src_act),
        shard_p50_min=p50,
        shard_p85_min=p85,
        first_run_p50_min=fp50,
        first_run_p85_min=fp85,
        median_runs_per_task=float(statistics.median(src_runs)),
        review_multiplier=review_mult,
        by_class=by_class,
        used_fallback=used_fallback,
    )


# ---------------------------------------------------------------------------
# Workload 1: Surface-sweep sizing (w items / y minutes -> x agents)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SurfaceSwarmPlan:
    """Result of solving x agents to ingest w work units within y minutes."""

    total_units: int
    target_minutes: float
    reserve_minutes: float
    exec_minutes: float
    serial_setup_minutes: float
    unit_minutes: float
    straggler_factor: float
    effective_units_per_agent: int
    required_agents: int
    recommended_agents: int
    max_agents: int
    feasible: bool
    covered_units: int
    deferred_units: int
    worker_max_runtime_seconds: int
    verifier_max_runtime_seconds: int
    synthesizer_max_runtime_seconds: int
    predicted_wall_minutes: float
    bottleneck: str
    shards: Tuple[Tuple[int, int], ...] = ()  # (start_idx_inclusive, end_idx_exclusive)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_units": self.total_units,
            "target_minutes": round(self.target_minutes, 2),
            "reserve_minutes": round(self.reserve_minutes, 2),
            "exec_minutes": round(self.exec_minutes, 2),
            "serial_setup_minutes": round(self.serial_setup_minutes, 2),
            "unit_minutes": round(self.unit_minutes, 3),
            "straggler_factor": round(self.straggler_factor, 2),
            "effective_units_per_agent": self.effective_units_per_agent,
            "required_agents": self.required_agents,
            "recommended_agents": self.recommended_agents,
            "max_agents": self.max_agents,
            "feasible": self.feasible,
            "covered_units": self.covered_units,
            "deferred_units": self.deferred_units,
            "worker_max_runtime_seconds": self.worker_max_runtime_seconds,
            "verifier_max_runtime_seconds": self.verifier_max_runtime_seconds,
            "synthesizer_max_runtime_seconds": self.synthesizer_max_runtime_seconds,
            "predicted_wall_minutes": round(self.predicted_wall_minutes, 2),
            "bottleneck": self.bottleneck,
            "shard_sizes": [end - start for start, end in self.shards],
        }


def partition_items(items: Sequence[Any], num_shards: int) -> List[List[Any]]:
    """Evenly partition ``items`` into ``min(num_shards, len(items))`` disjoint lists."""
    n = len(items)
    if n == 0 or num_shards <= 0:
        return []
    k = min(n, num_shards)
    base, rem = divmod(n, k)
    shards: List[List[Any]] = []
    offset = 0
    for i in range(k):
        size = base + (1 if i < rem else 0)
        shards.append(list(items[offset : offset + size]))
        offset += size
    return shards


def plan_surface_swarm(
    total_units: int,
    target_minutes: float,
    *,
    unit_minutes: float = 0.45,
    serial_setup_minutes: float = 1.0,
    reserve_minutes: Optional[float] = None,
    max_agents: int = 12,
    straggler_factor: float = 1.25,
    min_units_per_agent: int = 1,
) -> SurfaceSwarmPlan:
    """Calculate agent count ``x`` to cover ``total_units`` within ``target_minutes``.

    Models:
      - Protected tail reserve (verification + synthesis) carved out upfront so
        execution never starves final packaging.
      - Per-agent serial overhead (boot + context load) subtracted from the
        execution window.
      - Tail/straggler inflation (``straggler_factor``, default 1.25 for p85
        parallel stragglers) so N workers don't plan to the optimistic mean.
      - Explicit concurrency ceiling (``max_agents``) with honest reporting of
        ``covered_units`` vs ``deferred_units`` when the cap binds.
    """
    if total_units <= 0:
        raise ValueError("total_units must be positive")
    if target_minutes <= 0:
        raise ValueError("target_minutes must be positive")
    if unit_minutes <= 0:
        raise ValueError("unit_minutes must be positive")
    if max_agents <= 0:
        raise ValueError("max_agents must be positive")
    if straggler_factor < 1.0:
        raise ValueError("straggler_factor must be >= 1.0")

    reserve = (
        float(reserve_minutes)
        if reserve_minutes is not None
        else max(MIN_RESERVE_MIN, round(target_minutes * DEFAULT_SYNTHESIS_RESERVE_FRACTION, 2))
    )
    # Never let reserve consume the entire target window.
    reserve = min(reserve, target_minutes * 0.45)
    exec_minutes = max(0.0, target_minutes - reserve)
    usable_unit_budget = max(0.0, (exec_minutes - serial_setup_minutes) / straggler_factor)

    if usable_unit_budget < unit_minutes:
        # Even one unit per worker cannot fit comfortably inside the exec window.
        eff_per_agent = 0
        required_agents = total_units
        recommended = min(max_agents, total_units)
        feasible = False
        covered = 0
        deferred = total_units
        bottleneck = "serial_overhead_exceeds_exec_window"
    else:
        eff_per_agent = max(min_units_per_agent, int(math.floor(usable_unit_budget / unit_minutes)))
        required_agents = int(math.ceil(total_units / eff_per_agent))
        recommended = min(max_agents, required_agents, total_units)
        if required_agents <= max_agents:
            feasible = True
            covered = total_units
            deferred = 0
            bottleneck = "none"
        else:
            feasible = False
            covered = min(total_units, recommended * eff_per_agent)
            deferred = max(0, total_units - covered)
            bottleneck = "max_agents_cap"

    # Partition covered units (or all units if planning best-effort when infeasible)
    units_to_slice = covered if covered > 0 else total_units
    k = max(1, min(recommended, units_to_slice))
    base, rem = divmod(units_to_slice, k)
    shards: List[Tuple[int, int]] = []
    pos = 0
    max_shard_len = 0
    for i in range(k):
        sz = base + (1 if i < rem else 0)
        shards.append((pos, pos + sz))
        max_shard_len = max(max_shard_len, sz)
        pos += sz

    predicted_exec = serial_setup_minutes + max_shard_len * unit_minutes * straggler_factor
    predicted_wall = predicted_exec + reserve

    worker_sec = max(MIN_WORKER_RUNTIME_SECONDS, int(round(exec_minutes * 60.0)))
    verifier_sec = max(MIN_WORKER_RUNTIME_SECONDS, int(round(reserve * 0.40 * 60.0)))
    synthesizer_sec = max(MIN_WORKER_RUNTIME_SECONDS, int(round(reserve * 0.60 * 60.0)))

    return SurfaceSwarmPlan(
        total_units=total_units,
        target_minutes=target_minutes,
        reserve_minutes=reserve,
        exec_minutes=exec_minutes,
        serial_setup_minutes=serial_setup_minutes,
        unit_minutes=unit_minutes,
        straggler_factor=straggler_factor,
        effective_units_per_agent=eff_per_agent,
        required_agents=required_agents,
        recommended_agents=recommended,
        max_agents=max_agents,
        feasible=feasible,
        covered_units=covered,
        deferred_units=deferred,
        worker_max_runtime_seconds=worker_sec,
        verifier_max_runtime_seconds=verifier_sec,
        synthesizer_max_runtime_seconds=synthesizer_sec,
        predicted_wall_minutes=predicted_wall,
        bottleneck=bottleneck,
        shards=tuple(shards),
    )


# ---------------------------------------------------------------------------
# Workload 2: Codebase DAG / Feature Pipeline wave scheduling & w/x/y budget
# ---------------------------------------------------------------------------

def _norm_path(path: Any) -> str:
    text = str(path or "").strip().replace("\\", "/")
    if not text:
        return ""
    return os.path.normpath(text).replace("\\", "/")


def _paths_overlap(a: str, b: str) -> bool:
    a, b = _norm_path(a), _norm_path(b)
    if not a or not b:
        return False
    if "." in (a, b):
        return True
    return a == b or a.startswith(b + "/") or b.startswith(a + "/")


@dataclass(frozen=True)
class RegionNode:
    """One file-disjoint work region in a wave-scheduled DAG."""

    index: int  # 0-based stable index
    item_ids: Tuple[str, ...]
    files: Tuple[str, ...]
    depends_on_regions: Tuple[int, ...]
    wave: int  # 0-based execution wave (critical-path depth)
    work_shards: float  # estimated work units in standard shards
    est_p50_min: float
    est_p85_min: float
    blast_radius_files: int = 0
    blast_radius_tests: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "item_ids": list(self.item_ids),
            "files": list(self.files),
            "depends_on_regions": list(self.depends_on_regions),
            "wave": self.wave,
            "work_shards": round(self.work_shards, 2),
            "est_p50_min": round(self.est_p50_min, 2),
            "est_p85_min": round(self.est_p85_min, 2),
            "blast_radius_files": self.blast_radius_files,
            "blast_radius_tests": self.blast_radius_tests,
        }


def _contract_region_cycles(
    groups: List[List[str]],
    item_deps: Mapping[str, Sequence[str]],
    order: Mapping[str, int],
) -> Tuple[List[List[str]], List[List[int]]]:
    """Contract any cycle introduced when file-overlapping items straddle deps.

    File overlap is undirected, while ``depends_on`` is directed. If A overlaps
    C and A -> B -> C, merging {A, C} creates a cycle with B; Tarjan's SCC
    contracts {A, B, C} into a single region so the region graph is always a
    valid DAG while preserving maximum safe parallelism everywhere else.
    """
    node_to_grp: Dict[str, int] = {}
    for gi, grp in enumerate(groups):
        for item_id in grp:
            node_to_grp[item_id] = gi

    n = len(groups)
    adj: List[set] = [set() for _ in range(n)]
    for gi, grp in enumerate(groups):
        for item_id in grp:
            for dep in item_deps.get(item_id, ()):
                dg = node_to_grp.get(dep)
                if dg is not None and dg != gi:
                    adj[gi].add(dg)

    # Tarjan's strongly connected components.
    index_counter = 0
    stack: List[int] = []
    on_stack: set = set()
    indices: Dict[int, int] = {}
    lowlink: Dict[int, int] = {}
    sccs: List[List[int]] = []

    def strongconnect(v: int) -> None:
        nonlocal index_counter
        indices[v] = lowlink[v] = index_counter
        index_counter += 1
        stack.append(v)
        on_stack.add(v)
        for w in sorted(adj[v]):
            if w not in indices:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], indices[w])
        if lowlink[v] == indices[v]:
            comp: List[int] = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                comp.append(w)
                if w == v:
                    break
            sccs.append(comp)

    for v in range(n):
        if v not in indices:
            strongconnect(v)

    # Merge groups inside each SCC and topologically order items inside each region.
    merged_groups: List[List[str]] = []
    for comp in sccs:
        members: List[str] = []
        for gi in comp:
            members.extend(groups[gi])
        merged_groups.append(members)

    # Sort regions deterministically by earliest plan item order.
    merged_groups.sort(key=lambda g: min(order[i] for i in g))

    # Topologically order items within each region.
    final_groups: List[List[str]] = []
    for grp in merged_groups:
        gset = set(grp)
        done: List[str] = []
        state: Dict[str, int] = {}

        def visit(u: str) -> None:
            if state.get(u) == 2:
                return
            state[u] = 1
            for dep in sorted((d for d in item_deps.get(u, ()) if d in gset), key=lambda x: order[x]):
                if state.get(dep, 0) == 0:
                    visit(dep)
            state[u] = 2
            done.append(u)

        for u in sorted(grp, key=lambda x: order[x]):
            visit(u)
        final_groups.append(done)

    # Rebuild region-level dependency edges (guaranteed acyclic).
    final_owner: Dict[str, int] = {}
    for ri, grp in enumerate(final_groups):
        for item_id in grp:
            final_owner[item_id] = ri

    region_deps: List[List[int]] = []
    for ri, grp in enumerate(final_groups):
        deps_set = {
            final_owner[dep]
            for item_id in grp
            for dep in item_deps.get(item_id, ())
            if dep in final_owner and final_owner[dep] != ri
        }
        region_deps.append(sorted(deps_set))

    return final_groups, region_deps


def group_regions_dag(
    items: Sequence[Mapping[str, Any]],
) -> Tuple[List[List[str]], List[List[int]], List[int]]:
    """Group items by file overlap ONLY, keeping disjoint dependencies as DAG waves.

    Returns ``(regions, region_deps, region_waves)`` where:
      - ``regions[r]`` is the ordered list of item IDs in region ``r``.
      - ``region_deps[r]`` is the sorted list of parent region indices ``r``
        depends on (0-based, guaranteed acyclic via SCC contraction).
      - ``region_waves[r]`` is the 0-based execution wave (longest dependency
        chain depth from wave 0).

    Unlike naive connected-component grouping over both files and ``depends_on``,
    this keeps ``schema -> [api, ui, docs]`` as wave 0 (``schema``) followed by
    wave 1 (``api``, ``ui``, ``docs`` running concurrently in parallel worktrees).
    """
    if not items:
        return [], [], []
    ids = [str(item["id"]) for item in items]
    order = {item_id: idx for idx, item_id in enumerate(ids)}
    item_deps = {
        str(item["id"]): [str(d) for d in (item.get("depends_on") or []) if str(d) in order]
        for item in items
    }

    parent = {item_id: item_id for item_id in ids}

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

    # Union ONLY on file/directory overlap, NOT on depends_on.
    for i, a in enumerate(items):
        a_files = a.get("files") or []
        for b in items[i + 1 :]:
            b_files = b.get("files") or []
            if any(_paths_overlap(x, y) for x in a_files for y in b_files):
                union(str(a["id"]), str(b["id"]))

    buckets: Dict[str, List[str]] = {}
    for item_id in ids:
        buckets.setdefault(find(item_id), []).append(item_id)
    raw_groups = sorted(buckets.values(), key=lambda g: min(order[i] for i in g))

    regions, region_deps = _contract_region_cycles(raw_groups, item_deps, order)

    # Compute 0-based wave depth per region.
    memo: Dict[int, int] = {}

    def wave_of(r: int) -> int:
        if r in memo:
            return memo[r]
        deps = region_deps[r]
        w = 0 if not deps else 1 + max(wave_of(d) for d in deps)
        memo[r] = w
        return w

    waves = [wave_of(r) for r in range(len(regions))]
    return regions, region_deps, waves


@dataclass(frozen=True)
class DagBudgetPlan:
    """Work-quantity (w), concurrency (x), and duration (y) plan for a code DAG."""

    regions: Tuple[RegionNode, ...]
    wave_count: int
    max_wave_width: int
    total_work_shards: float
    critical_path_p50_min: float
    critical_path_p85_min: float
    target_minutes: Optional[float]
    reserve_minutes: float
    max_concurrent_workers: int
    recommended_workers: int
    predicted_p50_min: float
    predicted_p85_min: float
    feasible: bool
    bottleneck: str
    region_max_runtime_seconds: Tuple[int, ...]
    calibration: BoardCalibration

    def as_dict(self) -> Dict[str, Any]:
        return {
            "regions": [r.as_dict() for r in self.regions],
            "wave_count": self.wave_count,
            "max_wave_width": self.max_wave_width,
            "total_work_shards": round(self.total_work_shards, 2),
            "critical_path_p50_min": round(self.critical_path_p50_min, 2),
            "critical_path_p85_min": round(self.critical_path_p85_min, 2),
            "target_minutes": round(self.target_minutes, 2) if self.target_minutes is not None else None,
            "reserve_minutes": round(self.reserve_minutes, 2),
            "max_concurrent_workers": self.max_concurrent_workers,
            "recommended_workers": self.recommended_workers,
            "predicted_p50_min": round(self.predicted_p50_min, 2),
            "predicted_p85_min": round(self.predicted_p85_min, 2),
            "feasible": self.feasible,
            "bottleneck": self.bottleneck,
            "region_max_runtime_seconds": list(self.region_max_runtime_seconds),
            "calibration": self.calibration.as_dict(),
        }


def _simulate_dag_makespan(
    durations: Sequence[float],
    region_deps: Sequence[Sequence[int]],
    workers: int,
) -> float:
    """Greedy list-scheduling simulation of a region DAG on ``workers`` slots."""
    n = len(durations)
    if n == 0:
        return 0.0
    k = max(1, workers)
    # Critical-path remaining time priority (longest path from node to sink).
    children: List[List[int]] = [[] for _ in range(n)]
    in_deg = [0] * n
    for r, deps in enumerate(region_deps):
        for d in deps:
            children[d].append(r)
            in_deg[r] += 1

    tail_memo: Dict[int, float] = {}

    def tail_len(u: int) -> float:
        if u in tail_memo:
            return tail_memo[u]
        val = durations[u] + (max((tail_len(c) for c in children[u]), default=0.0))
        tail_memo[u] = val
        return val

    for r in range(n):
        tail_len(r)

    # Event-driven simulation over worker finish times.
    worker_free = [0.0] * k
    finish_time: Dict[int, float] = {}
    remaining = set(range(n))

    while remaining:
        ready = [r for r in remaining if all(d in finish_time for d in region_deps[r])]
        ready.sort(key=lambda r: (-tail_memo[r], r))
        # Pick earliest-available ready task that can run on the earliest free worker.
        best_r = None
        best_start = None
        best_w = 0
        for r in ready:
            dep_ready = max((finish_time[d] for d in region_deps[r]), default=0.0)
            # Earliest worker slot
            w_idx = min(range(k), key=lambda idx: worker_free[idx])
            start = max(dep_ready, worker_free[w_idx])
            if best_r is None or best_start is None or start < best_start - 1e-9 or (
                abs(start - best_start) <= 1e-9 and tail_memo[r] > tail_memo[best_r]
            ):
                best_start = start
                best_r = r
                best_w = w_idx
        assert best_r is not None and best_start is not None
        end = best_start + durations[best_r]
        worker_free[best_w] = end
        finish_time[best_r] = end
        remaining.remove(best_r)

    return max(finish_time.values(), default=0.0)


def _estimate_region_shards(
    item_count: int,
    file_count: int,
    blast_files: int,
    has_migration: bool,
) -> float:
    """Convert structural work features into standard-shard equivalents (w).

    A single well-scoped item touching 1-4 files is ~1.0 shard. Additional items
    in the same file-locked region add sub-linear marginal effort (0.65 each
    because context is already loaded), high blast radius adds test/verification
    overhead, and migrations add a fixed safety increment.
    """
    base = 1.0 + max(0, item_count - 1) * 0.65
    file_extra = max(0, file_count - 3) * 0.08
    blast_extra = 0.0
    if blast_files > 15:
        blast_extra = min(1.5, math.log2(blast_files / 15.0) * 0.35)
    migr_extra = 0.35 if has_migration else 0.0
    return round(base + file_extra + blast_extra + migr_extra, 2)


def plan_dag_budget(
    items: Sequence[Mapping[str, Any]],
    *,
    calibration: Optional[BoardCalibration] = None,
    repo: Optional[str | Path] = None,
    target_minutes: Optional[float] = None,
    max_concurrent_workers: int = 4,
    reserve_minutes: Optional[float] = None,
    edges: Optional[Mapping[str, Sequence[str]]] = None,
) -> DagBudgetPlan:
    """Compute wave-scheduled regions, work ``w``, minimum workers ``x``, and runtime caps."""
    cal = calibration or BoardCalibration(
        board="default",
        sample_count=0,
        shard_p50_min=DEFAULT_SHARD_P50_MIN,
        shard_p85_min=DEFAULT_SHARD_P85_MIN,
        first_run_p50_min=DEFAULT_FIRST_RUN_P50_MIN,
        first_run_p85_min=DEFAULT_FIRST_RUN_P85_MIN,
        median_runs_per_task=2.0,
        review_multiplier=round(DEFAULT_SHARD_P50_MIN / DEFAULT_FIRST_RUN_P50_MIN, 2),
        used_fallback=True,
    )
    if max_concurrent_workers <= 0:
        raise ValueError("max_concurrent_workers must be positive")

    regions_ids, region_deps, region_waves = group_regions_dag(items)
    by_id = {str(it["id"]): it for it in items}
    precomputed_edges = edges
    if precomputed_edges is None and repo and Path(repo).is_dir():
        precomputed_edges = build_import_graph(repo)

    region_nodes: List[RegionNode] = []
    for idx, item_ids in enumerate(regions_ids):
        r_files = tuple(dict.fromkeys(
            _norm_path(f) for iid in item_ids for f in (by_id[iid].get("files") or []) if _norm_path(f)
        ))
        blast = (
            blast_radius_for_paths(repo, r_files, edges=precomputed_edges)
            if repo and r_files
            else {"blast_radius_files": 0, "blast_radius_tests": 0, "blast_radius_migrations": 0}
        )
        has_migr = any(MIGRATION_PATH_RE.search(f) for f in r_files) or blast.get("blast_radius_migrations", 0) > 0
        shards = _estimate_region_shards(
            len(item_ids),
            len(r_files),
            int(blast.get("blast_radius_files", 0)),
            has_migr,
        )
        est_p50 = round(shards * cal.shard_p50_min, 2)
        est_p85 = round(shards * cal.shard_p85_min, 2)
        region_nodes.append(
            RegionNode(
                index=idx,
                item_ids=tuple(item_ids),
                files=r_files,
                depends_on_regions=tuple(region_deps[idx]),
                wave=region_waves[idx],
                work_shards=shards,
                est_p50_min=est_p50,
                est_p85_min=est_p85,
                blast_radius_files=int(blast.get("blast_radius_files", 0)),
                blast_radius_tests=int(blast.get("blast_radius_tests", 0)),
            )
        )

    wave_count = (max(region_waves) + 1) if region_waves else 0
    wave_widths = collections.Counter(region_waves)
    max_wave_width = max(wave_widths.values(), default=0)
    total_shards = sum(r.work_shards for r in region_nodes)

    durs_p50 = [r.est_p50_min for r in region_nodes]
    durs_p85 = [r.est_p85_min for r in region_nodes]
    # Unbounded concurrency gives the pure critical-path lower bound.
    unbounded_k = max(1, len(region_nodes))
    cp_p50 = _simulate_dag_makespan(durs_p50, region_deps, unbounded_k)
    cp_p85 = _simulate_dag_makespan(durs_p85, region_deps, unbounded_k)

    # Integration + verification reserve: scales mildly with number of regions to merge.
    default_reserve = max(
        cal.first_run_p50_min,
        round((0.5 + 0.15 * max(0, len(region_nodes) - 1)) * cal.shard_p50_min, 2),
    )
    if reserve_minutes is not None:
        reserve = float(reserve_minutes)
    elif target_minutes is not None:
        reserve = min(default_reserve, round(target_minutes * DEFAULT_SYNTHESIS_RESERVE_FRACTION, 2))
        reserve = max(MIN_RESERVE_MIN, reserve)
    else:
        reserve = default_reserve

    cap = max(1, min(max_concurrent_workers, max(1, max_wave_width)))
    if target_minutes is None:
        rec_workers = cap
        pred_p50 = _simulate_dag_makespan(durs_p50, region_deps, rec_workers) + reserve
        pred_p85 = _simulate_dag_makespan(durs_p85, region_deps, rec_workers) + reserve
        feasible = True
        bottleneck = "none"
    else:
        exec_window = max(0.0, target_minutes - reserve)
        rec_workers = cap
        feasible = False
        for k in range(1, cap + 1):
            mk = _simulate_dag_makespan(durs_p85, region_deps, k)
            if mk <= exec_window + 1e-9:
                rec_workers = k
                feasible = True
                break
        pred_p50 = _simulate_dag_makespan(durs_p50, region_deps, rec_workers) + reserve
        pred_p85 = _simulate_dag_makespan(durs_p85, region_deps, rec_workers) + reserve
        if feasible:
            bottleneck = "none"
        elif cp_p85 > exec_window:
            bottleneck = "critical_path_exceeds_target"
        else:
            bottleneck = "max_concurrent_workers_cap"

    # Compute per-region max_runtime_seconds.
    # If target_minutes is given, allocate proportionally along the critical path
    # within exec_window so no single region starves downstream waves or synthesis.
    region_caps: List[int] = []
    if target_minutes is not None and region_nodes:
        exec_window = max(1.0, target_minutes - reserve)
        scale = min(1.0, exec_window / max(cp_p85, 0.1))
        for r in region_nodes:
            allocated_min = max(1.5, r.est_p85_min * scale)
            region_caps.append(max(MIN_WORKER_RUNTIME_SECONDS, int(round(allocated_min * 60.0))))
    else:
        for r in region_nodes:
            # Generous safety cap (~1.5x p85) when no strict deadline was set,
            # preventing 60-run infinite loops without cutting off healthy runs.
            region_caps.append(max(300, int(round(r.est_p85_min * 1.5 * 60.0))))

    return DagBudgetPlan(
        regions=tuple(region_nodes),
        wave_count=wave_count,
        max_wave_width=max_wave_width,
        total_work_shards=total_shards,
        critical_path_p50_min=cp_p50,
        critical_path_p85_min=cp_p85,
        target_minutes=target_minutes,
        reserve_minutes=reserve,
        max_concurrent_workers=max_concurrent_workers,
        recommended_workers=rec_workers,
        predicted_p50_min=pred_p50,
        predicted_p85_min=pred_p85,
        feasible=feasible,
        bottleneck=bottleneck,
        region_max_runtime_seconds=tuple(region_caps),
        calibration=cal,
    )


# ---------------------------------------------------------------------------
# Closed-loop runtime controller (reproject burn rate against wall-clock budget)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BudgetReprojection:
    """Closed-loop control verdict midway through a time-budgeted run."""

    target_minutes: float
    elapsed_minutes: float
    remaining_minutes: float
    reserve_minutes: float
    remaining_exec_minutes: float
    completed_units: float
    remaining_units: float
    observed_minutes_per_unit: float
    active_workers: int
    max_workers: int
    projected_remaining_exec_minutes: float
    projected_total_minutes: float
    action: str  # "on_track" | "scale_up" | "tighten_scope" | "harvest_now"
    recommended_workers: int
    reason: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "target_minutes": round(self.target_minutes, 2),
            "elapsed_minutes": round(self.elapsed_minutes, 2),
            "remaining_minutes": round(self.remaining_minutes, 2),
            "reserve_minutes": round(self.reserve_minutes, 2),
            "remaining_exec_minutes": round(self.remaining_exec_minutes, 2),
            "completed_units": round(self.completed_units, 2),
            "remaining_units": round(self.remaining_units, 2),
            "observed_minutes_per_unit": round(self.observed_minutes_per_unit, 3),
            "active_workers": self.active_workers,
            "max_workers": self.max_workers,
            "projected_remaining_exec_minutes": round(self.projected_remaining_exec_minutes, 2),
            "projected_total_minutes": round(self.projected_total_minutes, 2),
            "action": self.action,
            "recommended_workers": self.recommended_workers,
            "reason": self.reason,
        }


def reproject_budget(
    *,
    target_minutes: float,
    elapsed_minutes: float,
    reserve_minutes: float,
    completed_units: float,
    remaining_units: float,
    completed_worker_minutes: float,
    prior_minutes_per_unit: float,
    active_workers: int,
    max_workers: int,
    parallelizable_remaining: Optional[int] = None,
    straggler_factor: float = 1.2,
) -> BudgetReprojection:
    """Re-project completion time from observed burn rate and return a control action.

    Uses empirical Bayes shrinkage between the prior rate and observed worker
    burn rate so early noisy completions do not cause wild thrashing.
    """
    remaining_total = max(0.0, target_minutes - elapsed_minutes)
    remaining_exec = max(0.0, remaining_total - reserve_minutes)

    # Shrink observed rate toward prior with pseudo-count weight = 2 units.
    prior_weight = 2.0
    if completed_units > 0 and completed_worker_minutes > 0:
        obs_rate = (completed_worker_minutes + prior_weight * prior_minutes_per_unit) / (
            completed_units + prior_weight
        )
    else:
        obs_rate = prior_minutes_per_unit

    eff_active = max(1, active_workers)
    if remaining_units <= 0:
        return BudgetReprojection(
            target_minutes=target_minutes,
            elapsed_minutes=elapsed_minutes,
            remaining_minutes=remaining_total,
            reserve_minutes=reserve_minutes,
            remaining_exec_minutes=remaining_exec,
            completed_units=completed_units,
            remaining_units=0.0,
            observed_minutes_per_unit=obs_rate,
            active_workers=active_workers,
            max_workers=max_workers,
            projected_remaining_exec_minutes=0.0,
            projected_total_minutes=elapsed_minutes + reserve_minutes,
            action="harvest_now",
            recommended_workers=0,
            reason="all work units completed; enter synthesis/verification reserve",
        )

    if remaining_exec <= 0.0:
        return BudgetReprojection(
            target_minutes=target_minutes,
            elapsed_minutes=elapsed_minutes,
            remaining_minutes=remaining_total,
            reserve_minutes=reserve_minutes,
            remaining_exec_minutes=0.0,
            completed_units=completed_units,
            remaining_units=remaining_units,
            observed_minutes_per_unit=obs_rate,
            active_workers=active_workers,
            max_workers=max_workers,
            projected_remaining_exec_minutes=(remaining_units * obs_rate * straggler_factor) / eff_active,
            projected_total_minutes=elapsed_minutes + reserve_minutes,
            action="harvest_now",
            recommended_workers=0,
            reason="execution cutoff reached; stop dispatching and synthesize partial results within reserve",
        )

    proj_rem_exec = (remaining_units * obs_rate * straggler_factor) / eff_active
    proj_total = elapsed_minutes + proj_rem_exec + reserve_minutes

    if proj_rem_exec <= remaining_exec + 1e-9:
        return BudgetReprojection(
            target_minutes=target_minutes,
            elapsed_minutes=elapsed_minutes,
            remaining_minutes=remaining_total,
            reserve_minutes=reserve_minutes,
            remaining_exec_minutes=remaining_exec,
            completed_units=completed_units,
            remaining_units=remaining_units,
            observed_minutes_per_unit=obs_rate,
            active_workers=active_workers,
            max_workers=max_workers,
            projected_remaining_exec_minutes=proj_rem_exec,
            projected_total_minutes=proj_total,
            action="on_track",
            recommended_workers=eff_active,
            reason="projected execution fits inside remaining execution window",
        )

    # Slipping: check if scaling up workers within parallelism + host ceiling recovers deadline.
    needed_workers = int(math.ceil((remaining_units * obs_rate * straggler_factor) / remaining_exec))
    width_cap = max_workers
    if parallelizable_remaining is not None:
        width_cap = min(width_cap, max(1, parallelizable_remaining))

    if needed_workers <= width_cap and needed_workers > eff_active:
        new_proj_exec = (remaining_units * obs_rate * straggler_factor) / needed_workers
        return BudgetReprojection(
            target_minutes=target_minutes,
            elapsed_minutes=elapsed_minutes,
            remaining_minutes=remaining_total,
            reserve_minutes=reserve_minutes,
            remaining_exec_minutes=remaining_exec,
            completed_units=completed_units,
            remaining_units=remaining_units,
            observed_minutes_per_unit=obs_rate,
            active_workers=active_workers,
            max_workers=max_workers,
            projected_remaining_exec_minutes=new_proj_exec,
            projected_total_minutes=elapsed_minutes + new_proj_exec + reserve_minutes,
            action="scale_up",
            recommended_workers=needed_workers,
            reason=f"burn rate slipping; scale workers from {eff_active} to {needed_workers} to hold target",
        )

    return BudgetReprojection(
        target_minutes=target_minutes,
        elapsed_minutes=elapsed_minutes,
        remaining_minutes=remaining_total,
        reserve_minutes=reserve_minutes,
        remaining_exec_minutes=remaining_exec,
        completed_units=completed_units,
        remaining_units=remaining_units,
        observed_minutes_per_unit=obs_rate,
        active_workers=active_workers,
        max_workers=max_workers,
        projected_remaining_exec_minutes=proj_rem_exec,
        projected_total_minutes=proj_total,
        action="tighten_scope",
        recommended_workers=min(width_cap, max(eff_active, needed_workers)),
        reason=(
            f"need {needed_workers} workers at observed rate ({obs_rate:.2f}m/unit) but ceiling is {width_cap}; "
            "shed optional scope or lower reasoning effort"
        ),
    )


def format_deadline_directive(max_runtime_seconds: Optional[int]) -> str:
    """Short prompt directive telling a worker its hard cutoff & partial-handoff rule."""
    if max_runtime_seconds is None or max_runtime_seconds <= 0:
        return ""
    mins = max_runtime_seconds / 60.0
    cutoff_sec = max(30, max_runtime_seconds - HANDOFF_GRACE_SECONDS)
    cutoff_mins = cutoff_sec / 60.0
    return (
        f"Time budget: {mins:.1f} min ({max_runtime_seconds}s hard limit). "
        f"If unfinished at ~{cutoff_mins:.1f} min ({cutoff_sec}s), stop expanding scope, "
        "commit or record partial structured findings, and hand off cleanly rather than timing out."
    )
