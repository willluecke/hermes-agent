"""Tests for w/x/y work-budget calibration, surface-sweep sizing, DAG wave scheduling,
closed-loop budget reprojection, feature_pipeline wave parallelism, and CLI surfaces.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pytest

from hermes_cli import feature_pipeline as fp
from hermes_cli import kanban as kanban_cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_swarm as ks
from hermes_cli import kanban_work_budget as wb


# ---------------------------------------------------------------------------
# 1. Surface-sweep w / y -> x calculation
# ---------------------------------------------------------------------------

def test_surface_swarm_sizes_exact_workers_and_reserves_tail():
    # Example from Will's SEO audit surface: 113 URLs in 10 minutes
    plan = wb.plan_surface_swarm(
        113,
        10.0,
        unit_minutes=0.45,
        serial_setup_minutes=1.0,
        max_agents=12,
        straggler_factor=1.25,
    )
    assert plan.reserve_minutes == 2.5
    assert plan.exec_minutes == 7.5
    assert plan.feasible is True
    assert plan.bottleneck == "none"
    assert plan.covered_units == 113
    assert plan.deferred_units == 0
    assert plan.recommended_agents == plan.required_agents
    assert 1 <= plan.recommended_agents <= 12
    assert sum(end - start for start, end in plan.shards) == 113
    assert plan.predicted_wall_minutes <= 10.0 + 0.05
    assert plan.worker_max_runtime_seconds == 450
    assert plan.verifier_max_runtime_seconds >= wb.MIN_WORKER_RUNTIME_SECONDS
    assert plan.synthesizer_max_runtime_seconds >= wb.MIN_WORKER_RUNTIME_SECONDS


def test_surface_swarm_reports_infeasibility_and_cut_list_under_cap():
    plan = wb.plan_surface_swarm(
        200,
        6.0,
        unit_minutes=0.5,
        serial_setup_minutes=1.0,
        max_agents=4,
        straggler_factor=1.25,
    )
    assert plan.feasible is False
    assert plan.bottleneck == "max_agents_cap"
    assert plan.recommended_agents == 4
    assert plan.covered_units < 200
    assert plan.covered_units + plan.deferred_units == 200
    assert sum(end - start for start, end in plan.shards) == plan.covered_units


def test_surface_swarm_handles_impossible_serial_overhead():
    plan = wb.plan_surface_swarm(
        10,
        2.0,
        unit_minutes=1.0,
        serial_setup_minutes=2.0,
        reserve_minutes=0.8,
        max_agents=10,
    )
    assert plan.feasible is False
    assert plan.bottleneck == "serial_overhead_exceeds_exec_window"
    assert plan.covered_units == 0
    assert plan.deferred_units == 10


# ---------------------------------------------------------------------------
# 2. DAG wave grouping & cycle contraction (fixes Region 1/1 collapse)
# ---------------------------------------------------------------------------

def test_group_regions_dag_keeps_disjoint_dependents_parallel_across_waves():
    # Classic feature pattern that previously collapsed into Region 1/1:
    # schema -> [api, ui, docs], and tests depends on api + ui.
    items = [
        {"id": "schema", "files": ["src/db/schema.ts"], "depends_on": []},
        {"id": "api", "files": ["src/app/api/items/route.ts"], "depends_on": ["schema"]},
        {"id": "ui", "files": ["src/components/items.tsx"], "depends_on": ["schema"]},
        {"id": "docs", "files": ["docs/items.md"], "depends_on": ["schema"]},
        {"id": "tests", "files": ["tests/items.test.ts"], "depends_on": ["api", "ui"]},
    ]
    # Legacy default still collapses connected components for backward compat
    assert fp.group_regions(items) == [["schema", "api", "ui", "docs", "tests"]]

    # Wave DAG keeps file-disjoint regions separate across 3 waves (width 3 in wave 1!)
    regions, deps, waves = wb.group_regions_dag(items)
    assert regions == [["schema"], ["api"], ["ui"], ["docs"], ["tests"]]
    assert deps == [[], [0], [0], [0], [1, 2]]
    assert waves == [0, 1, 1, 1, 2]


def test_group_regions_dag_contracts_cycles_when_file_overlap_straddles_dependency():
    # i1 -> i2 -> i3, and i1 shares a file with i3.
    # Merging {i1, i3} creates a cycle with i2, so SCC contraction merges all three safely.
    items = [
        {"id": "i1", "files": ["src/shared.ts"], "depends_on": []},
        {"id": "i2", "files": ["src/mid.ts"], "depends_on": ["i1"]},
        {"id": "i3", "files": ["src/shared.ts"], "depends_on": ["i2"]},
        {"id": "i4", "files": ["src/independent.ts"], "depends_on": []},
    ]
    regions, deps, waves = wb.group_regions_dag(items)
    assert regions == [["i1", "i2", "i3"], ["i4"]]
    assert deps == [[], []]
    assert waves == [0, 0]


# ---------------------------------------------------------------------------
# 3. Blast radius (primary tree only, tsconfig aliases, skips mirror dirs)
# ---------------------------------------------------------------------------

def test_blast_radius_resolves_aliases_and_skips_worktrees_and_node_modules(tmp_path: Path):
    (tmp_path / "tsconfig.json").write_text(
        json.dumps(
            {
                "compilerOptions": {
                    "baseUrl": ".",
                    "paths": {"@/*": ["src/*"]},
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "src" / "lib").mkdir(parents=True)
    (tmp_path / "src" / "app" / "api").mkdir(parents=True)
    (tmp_path / "tests").mkdir(parents=True)
    (tmp_path / ".worktrees" / "mirror" / "src").mkdir(parents=True)

    (tmp_path / "src" / "lib" / "core.ts").write_text("export const X = 1;\n", encoding="utf-8")
    (tmp_path / "src" / "app" / "api" / "route.ts").write_text(
        'import { X } from "@/lib/core";\nexport const y = X;\n', encoding="utf-8"
    )
    (tmp_path / "tests" / "core.test.ts").write_text(
        'import { y } from "@/app/api/route";\nconsole.log(y);\n', encoding="utf-8"
    )
    # Mirror copy inside .worktrees must NOT be counted
    (tmp_path / ".worktrees" / "mirror" / "src" / "copy.ts").write_text(
        'import { X } from "@/lib/core";\n', encoding="utf-8"
    )

    blast = wb.blast_radius_for_paths(tmp_path, ["src/lib/core.ts"])
    assert blast["seed_files"] == 1
    assert blast["blast_radius_files"] == 3
    assert blast["blast_radius_tests"] == 1
    assert blast["blast_radius_routes"] == 1
    assert blast["graph_files_scanned"] == 2  # route.ts and core.test.ts have resolved edges


# ---------------------------------------------------------------------------
# 4. Empirical calibration from live board task_runs + DAG budget solver
# ---------------------------------------------------------------------------

def _seed_completed_tasks(conn, durations_min: list[float], *, created_by: str = "auto-decomposer"):
    base_ts = 1_795_000_000
    for idx, dur in enumerate(durations_min):
        tid = f"t_seed_{idx:03d}"
        sa = base_ts + idx * 1000
        ca = sa + int(dur * 60)
        conn.execute(
            "INSERT INTO tasks (id, title, body, assignee, status, created_by, created_at, started_at, completed_at) "
            "VALUES (?, ?, ?, 'default', 'done', ?, ?, ?, ?)",
            (tid, f"Seed task {idx}", "body", created_by, sa - 10, sa, ca),
        )
        conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, started_at, ended_at, outcome) "
            "VALUES (?, 'default', 'done', ?, ?, 'completed')",
            (tid, sa, ca),
        )
    conn.commit()


def test_calibrate_board_reads_active_quantiles_and_plans_dag(tmp_path: Path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        # Fewer than 5 tasks -> fallback defaults
        cal_empty = wb.calibrate_board(conn, board="empty")
        assert cal_empty.used_fallback is True
        assert cal_empty.shard_p50_min == wb.DEFAULT_SHARD_P50_MIN

        # Seed 8 auto-decomposer tasks with known durations (4m..11m)
        _seed_completed_tasks(conn, [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
        cal = wb.calibrate_board(conn, board="seeded")
        assert cal.used_fallback is False
        assert cal.sample_count == 8
        assert 7.0 <= cal.shard_p50_min <= 8.0
        assert cal.shard_p85_min >= 9.5

        items = [
            {"id": "s", "files": ["src/schema.ts"], "depends_on": []},
            {"id": "a", "files": ["src/a.ts"], "depends_on": ["s"]},
            {"id": "b", "files": ["src/b.ts"], "depends_on": ["s"]},
            {"id": "c", "files": ["src/c.ts"], "depends_on": ["s"]},
        ]
        # 2 waves: wave 0 has 1 shard (~10m p85), wave 1 has 3 parallel shards (~10m p85).
        # With 3 workers and 26m target (reserve ~6.5m -> exec ~19.5m), 1 worker fails but 3 succeeds.
        plan_tight = wb.plan_dag_budget(items, calibration=cal, target_minutes=28.0, max_concurrent_workers=4)
        assert plan_tight.wave_count == 2
        assert plan_tight.max_wave_width == 3
        assert plan_tight.feasible is True
        assert plan_tight.recommended_workers == 3

        # Target below critical path -> honestly infeasible
        plan_impossible = wb.plan_dag_budget(items, calibration=cal, target_minutes=10.0, max_concurrent_workers=4)
        assert plan_impossible.feasible is False
        assert plan_impossible.bottleneck == "critical_path_exceeds_target"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 5. Closed-loop budget reprojection
# ---------------------------------------------------------------------------

def test_reproject_budget_transitions_on_track_scale_up_tighten_and_harvest():
    ok = wb.reproject_budget(
        target_minutes=10.0, elapsed_minutes=2.0, reserve_minutes=2.5,
        completed_units=30, remaining_units=50, completed_worker_minutes=10.0,
        prior_minutes_per_unit=0.4, active_workers=6, max_workers=12,
    )
    assert ok.action == "on_track"

    scale = wb.reproject_budget(
        target_minutes=10.0, elapsed_minutes=2.5, reserve_minutes=2.5,
        completed_units=20, remaining_units=45, completed_worker_minutes=14.0,
        prior_minutes_per_unit=0.5, active_workers=4, max_workers=12,
    )
    assert scale.action == "scale_up"
    assert 4 < scale.recommended_workers <= 12

    tighten = wb.reproject_budget(
        target_minutes=10.0, elapsed_minutes=5.0, reserve_minutes=2.5,
        completed_units=10, remaining_units=80, completed_worker_minutes=12.0,
        prior_minutes_per_unit=0.8, active_workers=4, max_workers=4,
    )
    assert tighten.action == "tighten_scope"

    harvest = wb.reproject_budget(
        target_minutes=10.0, elapsed_minutes=7.6, reserve_minutes=2.5,
        completed_units=70, remaining_units=20, completed_worker_minutes=25.0,
        prior_minutes_per_unit=0.4, active_workers=4, max_workers=8,
    )
    assert harvest.action == "harvest_now"
    assert harvest.recommended_workers == 0


# ---------------------------------------------------------------------------
# 6. Feature pipeline wave launch + budget propagation + parent branch handoff
# ---------------------------------------------------------------------------

@pytest.fixture
def pipeline_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    hermes = tmp_path / ".hermes"
    hermes.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for name in (
        "HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_TASK",
        "HERMES_KANBAN_RUN_ID", "HERMES_KANBAN_WORKSPACES_ROOT", "HERMES_GATEWAY_SESSION_ID",
    ):
        monkeypatch.delenv(name, raising=False)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    repo = tmp_path / "proj"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q",
         "--allow-empty", "-m", "base"],
        check=True,
    )
    (hermes / "subscription-projects.json").write_text(json.dumps({"proj": str(repo)}))
    notes = []
    monkeypatch.setattr(
        "hermes_cli.kanban_discovery.post_report_to_session",
        lambda session, text, board, **kw: notes.append((session, text)) or len(notes),
    )
    return {"repo": repo, "notes": notes}


def test_feature_pipeline_launches_dependency_waves_in_parallel_with_budget(pipeline_home):
    plan = {
        "summary": "Multi-wave feature.",
        "items": [
            {"id": "schema", "goal": "Add schema", "files": ["src/schema.ts"], "check": "pytest -q", "depends_on": []},
            {"id": "api", "goal": "Add API", "files": ["src/api.ts"], "check": "pytest -q", "depends_on": ["schema"]},
            {"id": "ui", "goal": "Add UI", "files": ["src/ui.tsx"], "check": "pytest -q", "depends_on": ["schema"]},
        ],
    }
    started = fp.start(
        project="proj",
        request="Build multi-wave feature.",
        conversation_id="c1",
        budget_minutes=45.0,
    )
    root = started["root"]
    assert started["budget_minutes"] == 45.0

    with kb.connect_closing(board="proj") as conn:
        kb.recompute_ready(conn)
        run = kb.claim_task(conn, root, claimer="opus").current_run_id

    approve_json = json.dumps({"verdict": "approve", "findings": []})
    fp.plan_review(
        root_id=root, plan=plan, board="proj", run_id=run,
        reviewer=lambda prompt, *, cwd: {"response": approve_json, "usage": {}},
    )
    launched = fp.launch(root_id=root, board="proj", run_id=run)
    assert len(launched["regions"]) == 3
    assert [r["wave"] for r in launched["regions"]] == [0, 1, 1]
    assert launched["budget"]["wave_count"] == 2
    assert launched["budget"]["max_wave_width"] == 2

    c_schema, c_api, c_ui = [r["card"] for r in launched["regions"]]
    with kb.connect_closing(board="proj") as conn:
        t_schema = kb.get_task(conn, c_schema)
        t_api = kb.get_task(conn, c_api)
        t_ui = kb.get_task(conn, c_ui)
        assert t_schema.status == "ready"
        assert t_api.status == "todo"
        assert t_ui.status == "todo"
        assert kb.parent_ids(conn, c_api) == [c_schema]
        assert kb.parent_ids(conn, c_ui) == [c_schema]
        # Wave 2 cards instruct workers to merge wave 1's reviewed branch and carry time budgets
        assert f"git merge --no-ff {t_schema.branch_name}" in (t_api.body or "")
        assert "Time budget:" in (t_schema.body or "")
        assert t_schema.max_runtime_seconds is not None and t_schema.max_runtime_seconds >= 90
        assert t_api.max_runtime_seconds is not None and t_api.max_runtime_seconds >= 90

        # Completing wave 1 promotes BOTH wave 2 cards to ready in parallel!
        r1 = kb.claim_task(conn, c_schema, claimer="opus")
        kb.request_review(conn, c_schema, summary="done", expected_run_id=r1.current_run_id)
        rev1 = kb.claim_review_task(conn, c_schema, claimer="astra")
        kb.complete_task(conn, c_schema, result="ok", expected_run_id=rev1.current_run_id)
        assert kb.get_task(conn, c_api).status == "ready"
        assert kb.get_task(conn, c_ui).status == "ready"
        assert kb.get_task(conn, root).status == "todo"

    # Closed-loop budget_reprojection appears on status() with remaining work & verdict
    st = fp.status(root_id=root, board="proj")
    assert st["budget_minutes"] == 45.0
    assert "budget_reprojection" in st
    assert st["budget_reprojection"]["completed_units"] > 0
    assert st["budget_reprojection"]["remaining_units"] > 0
    assert st["budget_reprojection"]["action"] in ("on_track", "scale_up", "tighten_scope", "harvest_now")


# ---------------------------------------------------------------------------
# 7. Time-boxed surface swarm creation, decompose budget inheritance & CLI
# ---------------------------------------------------------------------------

def test_feature_pipeline_launch_refuses_non_positive_budget_and_tool_passes_budget(pipeline_home, monkeypatch: pytest.MonkeyPatch):
    from tools.feature_pipeline_tool import feature_pipeline_tool

    bad = json.loads(feature_pipeline_tool(
        action="start", project="proj", request="Build it", conversation_id="c1", budget_minutes=-5
    ))
    assert bad["ok"] is False
    assert "positive" in bad["error"]

    good = json.loads(feature_pipeline_tool(
        action="start", project="proj", request="Build it", conversation_id="c1", budget_minutes=30
    ))
    assert good["ok"] is True
    assert good["budget_minutes"] == 30.0


def test_create_timeboxed_surface_swarm_and_decompose_budget_inheritance(tmp_path: Path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        urls = [f"https://example.com/page-{i}" for i in range(40)]
        created, budget = ks.create_timeboxed_surface_swarm(
            conn,
            goal="Full SEO audit",
            surface_items=urls,
            worker_profile="researcher",
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
            target_minutes=10.0,
            unit_minutes=0.45,
            max_agents=8,
        )
        assert budget["feasible"] is True
        assert len(created.worker_ids) == budget["recommended_agents"]
        # Every URL appears in exactly one worker shard
        seen_urls = []
        for wid in created.worker_ids:
            w = kb.get_task(conn, wid)
            assert w.max_runtime_seconds == budget["worker_max_runtime_seconds"]
            assert "Time budget:" in (w.body or "")
            seen_urls.extend(line[2:] for line in (w.body or "").splitlines() if line.startswith("- https://"))
        assert seen_urls == urls
        assert kb.get_task(conn, created.verifier_id).max_runtime_seconds == budget["verifier_max_runtime_seconds"]
        assert kb.get_task(conn, created.synthesizer_id).max_runtime_seconds == budget["synthesizer_max_runtime_seconds"]
        bb = ks.latest_blackboard(conn, created.root_id)
        assert bb["budget"]["total_units"] == 40

        # Decompose inherits root max_runtime_seconds divided across waves (80% exec / 2 waves)
        root_id = kb.create_task(
            conn, title="Big triage task", body="do a and b", triage=True, max_runtime_seconds=1200
        )
        children = kb.decompose_triage_task(
            conn,
            root_id,
            root_assignee="default",
            children=[
                {"title": "Step 1", "body": "a", "assignee": "default", "parents": []},
                {"title": "Step 2", "body": "b", "assignee": "default", "parents": [0]},
            ],
        )
        assert children is not None and len(children) == 2
        assert kb.get_task(conn, children[0]).max_runtime_seconds == 480
        assert kb.get_task(conn, children[1]).max_runtime_seconds == 480
    finally:
        conn.close()


def test_kanban_cli_calibrate_estimate_and_timeboxed_swarm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    db_path = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with kb.connect_closing() as conn:
        _seed_completed_tasks(conn, [5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    kanban_cli.build_parser(sub)

    # calibrate --json
    args = parser.parse_args(["kanban", "calibrate", "--json"])
    assert kanban_cli.kanban_command(args) == 0
    cal_out = json.loads(capsys.readouterr().out)
    assert cal_out["sample_count"] == 6
    assert cal_out["used_fallback"] is False

    # estimate --units 113 --budget-minutes 10 --json
    args = parser.parse_args(["kanban", "estimate", "--units", "113", "--budget-minutes", "10", "--json"])
    assert kanban_cli.kanban_command(args) == 0
    est_out = json.loads(capsys.readouterr().out)
    assert est_out["feasible"] is True
    assert est_out["covered_units"] == 113

    # swarm --surface-file ... --budget-minutes 10 --json
    sf = tmp_path / "urls.txt"
    sf.write_text("\n".join(f"https://regwatch.us/p/{i}" for i in range(25)) + "\n", encoding="utf-8")
    args = parser.parse_args([
        "kanban", "swarm", "Audit 25 pages",
        "--surface-file", str(sf),
        "--budget-minutes", "10",
        "--verifier", "reviewer",
        "--synthesizer", "writer",
        "--json",
    ])
    assert kanban_cli.kanban_command(args) == 0
    sw_out = json.loads(capsys.readouterr().out)
    assert sw_out["budget"]["total_units"] == 25
    assert len(sw_out["worker_ids"]) == sw_out["budget"]["recommended_agents"]
