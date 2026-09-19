"""The evidence module: code-built ledger rows, extractors, workspace digest, manifest checks, safe re-runs."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

_MODULE_FILE = Path(__file__).resolve().parents[2] / "plugins" / "system-one-preflight" / "evidence.py"
_spec = importlib.util.spec_from_file_location("system_one_evidence_under_test", _MODULE_FILE)
evidence = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(evidence)


# ---------------------------------------------------------------------------
# Tool results
# ---------------------------------------------------------------------------

def test_split_result_reads_the_native_json_and_leaves_text_with_no_exit_code():
    assert evidence.split_result(json.dumps({"output": "3 passed", "exit_code": 0})) == ("3 passed", 0)
    assert evidence.split_result(json.dumps({"output": "boom", "exit_code": 1})) == ("boom", 1)
    assert evidence.split_result("3 passed in 0.2s") == ("3 passed in 0.2s", None), "the Claude lane carries no exit code"
    assert evidence.split_result("stuff\nExit code: 2\n") == ("stuff\nExit code: 2\n", 2), "a printed exit line is read"
    assert evidence.split_result("") == ("", None)
    assert evidence.split_result('{"not": "a terminal result"}') == ('{"not": "a terminal result"}', None)


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

PYTEST_OK = "............\n============ 12 passed in 0.31s ============\n"
PYTEST_FAIL = "F...\n=========== 1 failed, 3 passed, 2 warnings in 0.5s ===========\n"
PYTEST_ERRORS = "==== 2 errors in 0.1s ====\n"
NODE_OK = "▶ suite\n  ✔ works (1ms)\nℹ tests 4\nℹ suites 1\nℹ pass 4\nℹ fail 0\nℹ cancelled 0\n"
NODE_TAP_FAIL = "TAP version 13\nnot ok 1 - x\n# tests 3\n# pass 2\n# fail 1\n"
JEST_OK = "PASS src/a.test.ts\n\nTests:       7 passed, 7 total\nSnapshots:   0 total\nTime:        1.2 s\n"
VITEST_FAIL = " ❯ src/b.test.ts (2)\n\n Test Files  1 failed (1)\n      Tests  1 failed | 1 passed (2)\n   Start at  10:00:00\n"
TSC_FAIL = "src/x.ts(3,5): error TS2322: Type 'string' is not assignable to type 'number'.\nsrc/y.ts(1,1): error TS1005: ';' expected.\n\nFound 2 errors in 2 files.\n"
ESLINT_FAIL = "\n/app/src/a.ts\n  3:1  error  Unexpected var  no-var\n\n✖ 1 problem (1 error, 0 warnings)\n"
ESLINT_CLEAN = "✔ No ESLint warnings or errors\n"
RUFF_FAIL = "a.py:1:1: F401 `os` imported but unused\nFound 1 error.\n"
RUFF_CLEAN = "All checks passed!\n"


@pytest.mark.parametrize(
    "command, output, exit_code, kind, status, counts",
    [
        ("pytest -q", PYTEST_OK, 0, "pytest", "pass", {"passed": 12}),
        ("pytest -q", PYTEST_FAIL, None, "pytest", "fail", {"failed": 1, "passed": 3}),
        ("python -m pytest tests/", PYTEST_ERRORS, 1, "pytest", "fail", {"errors": 2}),
        ("pytest -q | tail -3", PYTEST_FAIL, 0, "pytest", "fail", {"failed": 1, "passed": 3}),  # the pipe masks the exit code; the summary wins
        ("pytest -q", "no tests ran in 0.01s", 5, "pytest", "unknown", {"passed": 0}),
        ("pytest -q", "", 0, "pytest", "pass", {}),
        ("pytest -q", "", None, "pytest", "unknown", {}),
        ("npm test", NODE_OK, 0, "node-test", "pass", {"tests": 4, "passed": 4, "failed": 0, "cancelled": 0}),
        ("node --test test/", NODE_TAP_FAIL, 1, "node-test", "fail", {"tests": 3, "passed": 2, "failed": 1}),
        ("npx jest", JEST_OK, 0, "jest", "pass", {"passed": 7}),
        ("npx vitest run", VITEST_FAIL, 1, "jest", "fail", {"failed": 1, "passed": 1}),
        ("npx tsc --noEmit", TSC_FAIL, 2, "tsc", "fail", {"errors": 2}),
        ("npx tsc --noEmit", "", 0, "tsc", "pass", {"errors": 0}),
        ("npx tsc --noEmit", "", None, "tsc", "pass", {"errors": 0}),
        ("npx tsc --noEmit", "error: cannot find module", None, "tsc", "unknown", {"errors": 0}),
        ("npx eslint src", ESLINT_FAIL, 1, "eslint", "fail", {"problems": 1, "errors": 1, "warnings": 0}),
        ("npm run lint", ESLINT_CLEAN, 0, "eslint", "pass", {"errors": 0, "warnings": 0}),
        ("ruff check .", RUFF_FAIL, 1, "ruff", "fail", {"errors": 1}),
        ("ruff check .", RUFF_CLEAN, 0, "ruff", "pass", {"errors": 0}),
        ("ls -la", "total 0", 0, "other", "pass", {}),
        ("echo '3 passed in 0.1s'", "3 passed in 0.1s", None, "other", "unknown", {}),  # a printed summary is not a check
        ("cat results.txt", PYTEST_OK, 0, "other", "pass", {}),  # a pass by exit code only; no counts are believed
        ("bash run_tests.sh", PYTEST_FAIL, 1, "other", "fail", {}),
        ("python3 app.py", "hello", None, "other", "unknown", {}),
        ("cat missing", "cat: missing: No such file", 1, "other", "fail", {}),
    ],
)
def test_extractors_read_the_runner_summary_and_stay_unknown_otherwise(command, output, exit_code, kind, status, counts):
    observation = evidence.parse_observation(command, output, exit_code)
    assert observation["kind"] == kind
    assert observation["status"] == status
    assert observation["counts"] == counts
    assert observation["parser"] == evidence.PARSER_VERSION


def test_recognized_marks_a_runner_summary_not_a_guess():
    assert evidence.parse_observation("pytest", PYTEST_OK, None)["recognized"] is True
    assert evidence.parse_observation("pytest", "", 0)["recognized"] is False
    assert evidence.parse_observation("ls", "x", 0)["recognized"] is False
    assert evidence.parse_observation("npx tsc", "", 0)["recognized"] is True
    assert evidence.parse_observation("npx tsc", "", None)["recognized"] is False, "silence with no exit code is not a recognised pass"


def test_a_printed_summary_is_only_read_when_the_command_is_a_runner():
    # Seen live: the real Claude CLI ran `echo '3 passed in 0.1s'` and the signature alone read it as pytest.
    observation = evidence.parse_observation("echo '3 passed in 0.1s'", "3 passed in 0.1s", None)
    assert observation == {"kind": "other", "status": "unknown", "counts": {}, "recognized": False, "parser": evidence.PARSER_VERSION}
    assert evidence.parse_observation('echo "100 passed"', "100 passed", 0)["kind"] == "other"
    # A runner command trusts its own summary line, banner or not.
    assert evidence.parse_observation("pytest -q", "100 passed", None)["counts"] == {"passed": 100}
    # A script runner that wraps a runner is read by the output's signature.
    assert evidence.parse_observation("npm test", NODE_OK, 0)["kind"] == "node-test"
    assert evidence.parse_observation("make test", PYTEST_OK, 0)["kind"] == "pytest"


# ---------------------------------------------------------------------------
# Rows, retained output, excerpts
# ---------------------------------------------------------------------------

def test_make_row_carries_identity_result_digest_and_freshness_inputs(tmp_path):
    row, output = evidence.make_row(3, "cd /app && pytest -q", json.dumps({"output": PYTEST_FAIL, "exit_code": 1}), cwd="/app", workspace="ws1")
    assert row["id"] == "c3" and row["n"] == 3 and row["command"] == "cd /app && pytest -q"
    assert row["exit"] == 1 and row["status"] == "fail" and row["kind"] == "pytest" and row["counts"] == {"failed": 1, "passed": 3}
    assert row["failure"] is True and row["check"] is True and row["workspace"] == "ws1" and row["source"] == "agent" and row["cwd"] == "/app"
    assert len(row["digest"]) == 12 and row["chars"] == len(PYTEST_FAIL) and row["capture_complete"] is True
    assert output == PYTEST_FAIL
    row, _ = evidence.make_row(4, "ls", "a\nb", prefix="k", source="controller")
    assert row["id"] == "k4" and row["exit"] is None and row["status"] == "unknown" and row["check"] is False and row["source"] == "controller"
    row, _ = evidence.make_row(5, "pytest", "x" * 100 + "\nFull output (120,000 chars) saved to /tmp/x.txt")
    assert row["capture_complete"] is False
    row, _ = evidence.make_row(6, "pytest", "x" * 100 + "…")
    assert row["capture_complete"] is False, "the hook script's clip marker"


def test_row_summary_is_compact_and_says_unknown_out_loud():
    row, _ = evidence.make_row(1, "pytest -q", PYTEST_OK, workspace="w")
    row["fresh"] = False
    summary = evidence.row_summary(row)
    assert summary == {"id": "c1", "command": "pytest -q", "exit": "unknown", "status": "pass", "kind": "pytest", "source": "agent", "counts": {"passed": 12}, "fresh": False}
    row, _ = evidence.make_row(2, "ls", json.dumps({"output": "Error: x", "exit_code": 1}))
    assert evidence.row_summary(row)["exit"] == 1 and evidence.row_summary(row)["failure_text"] is True


def test_failure_excerpt_centres_on_the_first_failure_not_the_tail():
    output = "ok\n" * 500 + "Traceback (most recent call last):\n  boom\n" + "later\n" * 500
    excerpt = evidence.failure_excerpt(output)
    assert "Traceback" in excerpt and excerpt.startswith("…") and excerpt.endswith("…")
    assert len(excerpt) <= evidence.MAX_EXCERPT_CHARS + 2
    assert evidence.failure_excerpt("all fine") == "all fine"
    tail = evidence.failure_excerpt("x" * 5_000)
    assert len(tail) <= evidence.MAX_EXCERPT_CHARS


def test_retained_output_round_trips_and_missing_files_read_as_none(tmp_path):
    path = evidence.retain_output(tmp_path / "ev", "c1", "hello\nworld")
    assert path and evidence.read_output(path) == "hello\nworld"
    assert evidence.read_output(str(tmp_path / "nope.txt")) is None
    assert evidence.read_output(None) is None


# ---------------------------------------------------------------------------
# Workspace digest
# ---------------------------------------------------------------------------

@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()

    def git(*args):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True)

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (root / "app.py").write_text("print('hi')\n")
    (root / ".gitignore").write_text("__pycache__/\n.cache/\n")
    git("add", ".")
    git("commit", "-q", "-m", "init")
    return root


def test_git_root_resolves_files_and_directories_and_caches(repo, tmp_path):
    evidence._root_cache.clear()
    assert evidence.git_root(str(repo / "app.py")) == str(repo)
    assert evidence.git_root(str(repo)) == str(repo)
    assert evidence.git_root(str(repo / "missing" / "deep.py")) is None
    assert evidence.git_root("") is None
    outside = tmp_path / "plain"
    outside.mkdir()
    assert evidence.git_root(str(outside)) in (None, evidence.git_root(str(outside))), "a non-repo directory answers consistently"


def test_workspace_digest_moves_on_every_edit_and_ignores_ignored_files(repo):
    first = evidence.workspace_digest([str(repo)])
    assert first and len(first) == 16
    assert evidence.workspace_digest([str(repo)]) == first, "stable while nothing changes"
    (repo / "app.py").write_text("print('hello')\n")
    second = evidence.workspace_digest([str(repo)])
    assert second != first, "a tracked edit"
    os.utime(repo / "app.py", ns=(time.time_ns(), time.time_ns() + 1_000_000))
    (repo / "app.py").write_text("print('hello!')\n")
    third = evidence.workspace_digest([str(repo)])
    assert third != second, "a second edit to the same file (same status line) still moves the digest"
    (repo / "new_module.py").write_text("x = 1\n")
    fourth = evidence.workspace_digest([str(repo)])
    assert fourth != third, "an untracked file counts"
    cache = repo / ".cache"
    cache.mkdir()
    (cache / "junk").write_text("z")
    assert evidence.workspace_digest([str(repo)]) == fourth, "an ignored file does not"
    assert evidence.workspace_digest([]) is None
    assert evidence.workspace_digest(["/definitely/not/a/repo"]) is None


def test_cd_prefix_reads_the_directory_a_command_starts_in():
    assert evidence.cd_prefix("cd /app && pytest -q") == "/app"
    assert evidence.cd_prefix("cd '/my dir'; npm test") == "/my dir"
    assert evidence.cd_prefix("pytest -q") is None
    assert evidence.cd_prefix("") is None


# ---------------------------------------------------------------------------
# Manifest parsing and verdicts
# ---------------------------------------------------------------------------

MANIFEST = {"manifest": [
    {"id": "r1", "criterion": "1", "claim": "Plugin suite passes", "evidence": ["c2"], "predicate": "passed", "expected": {}},
    {"id": "r2", "criterion": "2", "claim": "12 passed, 0 failed", "evidence": ["c2"], "predicate": "count", "expected": {"passed": 12, "failed": 0}},
], "note": "2 result claims registered."}


def test_parse_manifest_reads_the_raw_and_the_bridge_wrapped_result():
    raw = evidence.parse_manifest(json.dumps(MANIFEST))
    assert [item["id"] for item in raw] == ["r1", "r2"] and raw[1]["expected"] == {"passed": 12, "failed": 0}
    wrapped = evidence.parse_manifest(json.dumps({"result": json.dumps(MANIFEST)}))
    assert wrapped == raw
    assert evidence.parse_manifest("not json") is None
    assert evidence.parse_manifest(json.dumps({"todos": []})) is None
    assert evidence.parse_manifest(json.dumps({"manifest": []})) == []
    many = evidence.parse_manifest(json.dumps({"manifest": [{"claim": f"c{i}", "evidence": ["c1"]} for i in range(40)]}))
    assert len(many) == evidence.MAX_MANIFEST_ITEMS
    assert many[0]["id"] == "r1" and many[0]["predicate"] == "passed"


def _rows(tmp_path, **overrides):
    ok, out_ok = evidence.make_row(2, "pytest -q", json.dumps({"output": PYTEST_OK, "exit_code": 0}), workspace="w1")
    ok["file"] = evidence.retain_output(tmp_path / "ev", "c2", out_ok)
    bad, out_bad = evidence.make_row(3, "pytest -q tests/x.py", json.dumps({"output": PYTEST_FAIL, "exit_code": 1}), workspace="w1")
    bad["file"] = evidence.retain_output(tmp_path / "ev", "c3", out_bad)
    unknown, _ = evidence.make_row(4, "python3 app.py", "hello world", workspace="w1")
    unknown["file"] = None
    rows = {"c2": ok, "c3": bad, "c4": unknown}
    for row in rows.values():
        row.update(overrides)
    return rows


def test_check_assertion_supported_contradicted_missing_and_insufficient(tmp_path):
    rows = _rows(tmp_path)
    check = lambda **item: evidence.check_assertion({"predicate": "passed", "evidence": ["c2"], **item}, rows, "w1")
    assert check()["verdict"] == "supported"
    assert check(evidence=["c3"])["verdict"] == "contradicted"
    assert check(evidence=["c2", "c3"])["verdict"] == "contradicted"
    assert check(evidence=["c4"])["verdict"] == "insufficient" and "not a check runner" in check(evidence=["c4"])["detail"]
    assert check(evidence=["c9"])["verdict"] == "missing" and "c9" in check(evidence=["c9"])["detail"]
    assert check()["basis"] == "agent", "every verdict says whose row it rests on"
    assert check(evidence=[])["verdict"] == "insufficient"
    assert check(predicate="bogus")["verdict"] == "insufficient"
    assert check(predicate="ran", evidence=["c4"])["verdict"] == "supported"


def test_check_assertion_count_exit_zero_and_contains(tmp_path):
    rows = _rows(tmp_path)
    count = lambda expected, ids=("c2",): evidence.check_assertion({"predicate": "count", "evidence": list(ids), "expected": expected}, rows, "w1")
    assert count({"passed": 12})["verdict"] == "supported"
    assert count({"passed": 12, "failed": 0})["verdict"] == "supported"
    assert count({"passed": 11})["verdict"] == "contradicted" and "claimed 11, ledger 12" in count({"passed": 11})["detail"]
    assert count({"passed": 15}, ids=("c2", "c3"))["verdict"] == "supported", "counts add across cited rows"
    assert count({"passed": 1}, ids=("c4",))["verdict"] == "insufficient", "no runner summary to compare"
    assert count({})["verdict"] == "insufficient"
    exit_zero = lambda ids: evidence.check_assertion({"predicate": "exit_zero", "evidence": list(ids)}, rows, "w1")
    assert exit_zero(["c2"])["verdict"] == "supported"
    assert exit_zero(["c3"])["verdict"] == "contradicted"
    assert exit_zero(["c4"])["verdict"] == "insufficient"
    contains = lambda text, ids=("c2",): evidence.check_assertion({"predicate": "contains", "evidence": list(ids), "expected": {"text": text}}, rows, "w1")
    assert contains("12 passed")["verdict"] == "supported"
    assert contains("deployed")["verdict"] == "contradicted"
    assert contains("x", ids=("c4",))["verdict"] == "insufficient", "no retained output"
    assert contains("")["verdict"] == "insufficient"


def test_a_row_that_ran_under_another_workspace_is_stale_by_code_except_for_ran(tmp_path):
    rows = _rows(tmp_path)
    stale = evidence.check_assertion({"predicate": "passed", "evidence": ["c2"]}, rows, "w2")
    assert stale["verdict"] == "stale" and "c2" in stale["detail"]
    assert evidence.check_assertion({"predicate": "ran", "evidence": ["c2"]}, rows, "w2")["verdict"] == "supported"
    assert evidence.check_assertion({"predicate": "passed", "evidence": ["c2"]}, rows, None)["verdict"] == "supported", "no final digest: freshness unknown, not stale"
    rows["c2"]["workspace"] = None
    assert evidence.check_assertion({"predicate": "passed", "evidence": ["c2"]}, rows, "w2")["verdict"] == "supported", "a row with no digest cannot be called stale"


def test_manifest_counts_and_machinery_paths():
    assert evidence.manifest_counts([{"verdict": "supported"}, {"verdict": "stale"}, {"verdict": "supported"}]) == {
        "supported": 2, "contradicted": 0, "stale": 1, "insufficient": 0, "missing": 0,
    }
    found = evidence.machinery_paths(
        ["/r/src/app.py", "/r/tests/test_app.py", "/r/conftest.py", "/r/package.json", "/r/src/a.test.ts", "/r/.github/workflows/ci.yml", "/r/docs/x.md", "/r/tsconfig.json"],
        "/r",
    )
    assert found == ["tests/test_app.py", "conftest.py", "package.json", "src/a.test.ts", ".github/workflows/ci.yml", "tsconfig.json"]
    assert evidence.machinery_paths(["/elsewhere/spec/thing.rb"]) == ["/elsewhere/spec/thing.rb"]


# ---------------------------------------------------------------------------
# Controller re-runs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("command", [
    "pytest -q",
    "python -m pytest tests/plugins -q -p no:cacheprovider",
    "/home/x/venv/bin/python -m pytest tests/ -q",
    "cd /app && npm test",
    "cd /app && npm run test:closure -- --grep verdict",
    "CLOSURE_STUB=1 node --experimental-strip-types --test test/a.test.mjs",
    "npx tsc --noEmit",
    "npx eslint src && npx tsc --noEmit",
    "ruff check . | tail -3",
    "pytest -q 2>&1 | tail -20",
    "timeout 120 pytest -q",
    "go test ./...",
    "cargo test",
    "make test",
])
def test_rerunnable_accepts_plain_check_runners_with_cd_and_filters(command):
    assert evidence.rerunnable(command) is True


@pytest.mark.parametrize("command", [
    "",
    "ls -la",
    "python3 app.py",
    "pytest -q && git push",
    "pytest -q; rm -rf build",
    "npm test && vercel deploy",
    "pytest -q > results.txt",
    "pytest -q $(cat args)",
    "pytest -q `cat args`",
    "curl -X POST https://x | tail -1",
    "echo '100 passed'",
    "npm run deploy",
    "cd /app && npm test && systemctl restart x",
    "pytest -q || echo failed",
])
def test_rerunnable_refuses_anything_that_is_not_a_check(command):
    assert evidence.rerunnable(command) is False


def test_run_check_captures_output_exit_and_timeouts(tmp_path):
    run = evidence.run_check(f"{sys.executable} -c \"print('7 passed in 0.1s'); import sys; sys.exit(0)\"", str(tmp_path), 10)
    assert run["exit_code"] == 0 and "7 passed" in run["output"] and run["timed_out"] is False and run["seconds"] >= 0
    run = evidence.run_check(f"{sys.executable} -c \"import sys; sys.stderr.write('boom'); sys.exit(3)\"", str(tmp_path), 10)
    assert run["exit_code"] == 3 and "boom" in run["output"]
    run = evidence.run_check(f"{sys.executable} -c \"import time; time.sleep(5)\"", str(tmp_path), 0.3)
    assert run["timed_out"] is True and run["exit_code"] is None
    assert os.environ.get("HERMES_CONTROLLER_RERUN") is None, "the marker is set for the child only"


def test_a_passed_claim_needs_a_runner_shaped_recognised_row(tmp_path):
    """Astra's risk, demonstrated on the first ledger: all four of these cleared a passed claim."""
    rows = {}
    for n, (cmd, out) in enumerate([
        ("echo '100 passed in 0.1s'", "100 passed in 0.1s"),
        ("bash run_tests.sh", "all good"),
        ("pytest -q >/dev/null 2>&1; echo '12 passed in 0.1s'", "12 passed in 0.1s"),
        ("pytest -q | grep -v failed", "..\n2 passed in 0.1s"),
        ("pytest -q", ""),
    ], 1):
        row, _ = evidence.make_row(n, cmd, json.dumps({"output": out, "exit_code": 0}), workspace="w")
        rows[row["id"]] = row
    verdict = lambda rid: evidence.check_assertion({"predicate": "passed", "evidence": [rid]}, rows, "w")
    assert verdict("c1")["verdict"] == "insufficient" and "not a check runner" in verdict("c1")["detail"], "an echo"
    assert verdict("c2")["verdict"] == "insufficient" and "not a check runner" in verdict("c2")["detail"], "a wrapper script"
    assert verdict("c5")["verdict"] == "insufficient" and "no runner summary recognised" in verdict("c5")["detail"], "a silent runner with exit 0"
    # The two runner-shaped forgeries still pass the row check; the gate's own re-run is what refuses them (plugin tests).
    assert verdict("c3")["verdict"] == "supported" and verdict("c3")["basis"] == "agent"
    assert verdict("c4")["verdict"] == "supported" and verdict("c4")["basis"] == "agent"
    assert evidence.rerunnable("pytest -q >/dev/null 2>&1; echo '12 passed in 0.1s'") is False
    gate, _ = evidence.make_row(9, "pytest -q", json.dumps({"output": PYTEST_OK, "exit_code": 0}), workspace="w", source="controller", prefix="k")
    rows["k9"] = gate
    assert evidence.check_assertion({"predicate": "passed", "evidence": ["k9"]}, rows, "w")["basis"] == "gate"
    assert evidence.check_assertion({"predicate": "passed", "evidence": ["k9", "c3"]}, rows, "w")["basis"] == "agent", "one agent row makes the basis agent"


def test_strip_filters_drops_trailing_filters_only():
    assert evidence.strip_filters("pytest -q | grep -v failed") == "pytest -q"
    assert evidence.strip_filters("pytest -q 2>&1 | tail -20") == "pytest -q 2>&1"
    assert evidence.strip_filters("cd /app && npm test | tail -5 | grep -c ok") == "cd /app && npm test"
    assert evidence.strip_filters("pytest -q || echo failed") == "pytest -q || echo failed", "a double pipe is not a filter pipe"
    assert evidence.strip_filters("pytest -q | tee out.txt") == "pytest -q | tee out.txt", "tee is not a filter; left alone (and not re-runnable)"
    assert evidence.strip_filters("pytest -q") == "pytest -q"
    assert evidence.strip_filters("") == ""


def test_tests_run_counts_only_test_runners():
    row, _ = evidence.make_row(1, "pytest -q", PYTEST_FAIL, workspace="w")
    assert evidence.tests_run(row) == 4
    row, _ = evidence.make_row(2, "npm test", NODE_OK, workspace="w")
    assert evidence.tests_run(row) == 4
    row, _ = evidence.make_row(3, "npx vitest run", VITEST_FAIL, workspace="w")
    assert evidence.tests_run(row) == 2
    row, _ = evidence.make_row(4, "npx tsc --noEmit", TSC_FAIL, workspace="w")
    assert evidence.tests_run(row) is None, "tsc counts errors, not tests"
    row, _ = evidence.make_row(5, "pytest -q", "", workspace="w")
    assert evidence.tests_run(row) is None


def test_weakening_signals_read_removed_assertions_and_added_skips_in_existing_test_files(repo):
    tests = repo / "tests"
    tests.mkdir()
    (tests / "test_app.py").write_text("import pytest\n\n\ndef test_a():\n    assert 1 == 1\n\n\ndef test_b():\n    assert 2 == 2\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "tests"], check=True, capture_output=True)
    (tests / "test_app.py").write_text("import pytest\n\n\n@pytest.mark.skip\ndef test_a():\n    pass\n")
    (tests / "test_new.py").write_text("def test_c():\n    pass\n")
    (repo / "app.py").write_text("print('changed')\n")
    found = evidence.weakening_signals(str(repo), [str(tests / "test_app.py"), str(tests / "test_new.py"), str(repo / "app.py")])
    assert found["files"] == [{"path": "tests/test_app.py", "removed": 3, "skips": 1}], "two asserts and one def test_ removed, one skip added; the new file and app.py do not count"
    assert found["removed"] == 3 and found["skips"] == 1
    assert evidence.weakening_signals(None, [str(tests / "test_app.py")]) == {"files": [], "removed": 0, "skips": 0}
    assert evidence.weakening_signals(str(repo), [str(repo / "app.py")]) == {"files": [], "removed": 0, "skips": 0}
