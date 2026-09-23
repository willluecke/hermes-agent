"""Red/green diffs for file-edit tool rows (2026-09-23)."""

from __future__ import annotations

from types import SimpleNamespace

from agent.claude_runtime import make_claude_code_event_bridge
from agent.tool_diff import MAX_DIFF_LINES, _finish, claude_tool_diff, file_change_diff


def _file(tmp_path, text):
    path = tmp_path / "src" / "app.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_edit_diff_uses_the_real_line_numbers_and_a_project_relative_path(tmp_path):
    path = _file(tmp_path, "".join(f"line {n}\n" for n in range(1, 21)).replace("line 12", "LINE 12"))

    rendered = claude_tool_diff(
        "Edit", {"file_path": str(path), "old_string": "line 12", "new_string": "LINE 12"}, cwd=str(tmp_path)
    )

    assert rendered["diff"].splitlines() == [
        "--- a/src/app.py", "+++ b/src/app.py", "@@ -12 +12 @@", "-line 12", "+LINE 12",
    ]
    assert (rendered["lines_added"], rendered["lines_removed"]) == (1, 1)


def test_multi_edit_keeps_one_file_header_and_one_hunk_per_edit(tmp_path):
    # the diff reads the file after the edit ran
    path = _file(tmp_path, "a = 1\nb = 2\nc = 3\nd = 4\ne = 5\n")

    rendered = claude_tool_diff(
        "MultiEdit",
        {"file_path": str(path), "edits": [
            {"old_string": "a = 0", "new_string": "a = 1"},
            {"old_string": "d = 0\n", "new_string": "d = 4\ne = 5\n"},
        ]},
        cwd=str(tmp_path),
    )

    lines = rendered["diff"].splitlines()
    assert lines.count("--- a/src/app.py") == 1
    assert [line for line in lines if line.startswith("@@")] == ["@@ -1 +1 @@", "@@ -4 +4,2 @@"]
    assert (rendered["lines_added"], rendered["lines_removed"]) == (3, 2)


def test_write_shows_every_line_as_added(tmp_path):
    rendered = claude_tool_diff("Write", {"file_path": str(tmp_path / "new.md"), "content": "# Title\n\nBody\n"},
                                cwd=str(tmp_path))

    assert rendered["diff"].splitlines()[2:] == ["@@ -0,0 +1,3 @@", "+# Title", "+", "+Body"]
    assert (rendered["lines_added"], rendered["lines_removed"]) == (3, 0)


def test_long_diffs_are_capped_with_a_marker_and_exact_counts():
    rendered = _finish(["--- a/x", "+++ b/x", "@@ -0,0 +1,900 @@"] + [f"+{n}" for n in range(900)])

    lines = rendered["diff"].splitlines()
    assert len(lines) == MAX_DIFF_LINES + 1
    assert lines[-1] == f"\\ Diff truncated: {903 - MAX_DIFF_LINES} more lines"
    assert rendered["lines_added"] == 900


def test_non_edit_tools_and_bad_arguments_have_no_diff(tmp_path):
    assert claude_tool_diff("Read", {"file_path": str(tmp_path / "x")}) is None
    assert claude_tool_diff("Edit", {"file_path": "x", "old_string": 1, "new_string": "y"}) is None
    assert claude_tool_diff("Edit", {"old_string": "a", "new_string": "b"}) is None


def test_file_change_diff_skips_unreadable_and_unchanged_files():
    rendered = file_change_diff([
        ("same.py", [b"x"], [b"x"]),
        ("binary.bin", None, [b"y"]),
        ("app.py", [b"one", b"two"], [b"one", b"2"]),
    ])

    assert rendered["diff"].splitlines()[:2] == ["--- a/app.py", "+++ b/app.py"]
    assert (rendered["lines_added"], rendered["lines_removed"]) == (1, 1)
    assert file_change_diff([("same.py", [b"x"], [b"x"])]) is None


def _bridge_agent():
    calls = []
    agent = SimpleNamespace(
        tool_progress_callback=lambda *args, **kwargs: calls.append((args, kwargs)),
        show_commentary=False,
    )
    return agent, calls


def _tool_turn(bridge, block, *, is_error=False, result="ok"):
    bridge({"type": "assistant", "message": {"content": [dict(block, type="tool_use")]}})
    bridge({"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": block["id"], "content": result, "is_error": is_error}]}})


def test_claude_bridge_labels_file_tools_by_path_and_reports_the_edit_diff(tmp_path):
    path = _file(tmp_path, "value = 2\n")
    agent, calls = _bridge_agent()
    bridge = make_claude_code_event_bridge(agent, cwd=str(tmp_path))

    _tool_turn(bridge, {"id": "t1", "name": "Edit", "input": {
        "file_path": str(path), "old_string": "value = 1", "new_string": "value = 2"}})

    (started_args, _), (completed_args, completed) = calls
    assert started_args[:3] == ("tool.started", "apply_patch", "src/app.py")
    assert completed_args[0] == "tool.completed"
    assert completed["diff"].splitlines()[-2:] == ["-value = 1", "+value = 2"]
    assert (completed["lines_added"], completed["lines_removed"]) == (1, 1)


def test_claude_bridge_sends_no_diff_for_a_failed_edit_and_labels_searches(tmp_path):
    agent, calls = _bridge_agent()
    bridge = make_claude_code_event_bridge(agent, cwd=str(tmp_path))

    _tool_turn(bridge, {"id": "t2", "name": "Edit", "input": {
        "file_path": str(tmp_path / "x.py"), "old_string": "a", "new_string": "b"}},
        is_error=True, result="String to replace not found")
    _tool_turn(bridge, {"id": "t3", "name": "Grep", "input": {"pattern": "TODO", "path": str(tmp_path / "src")}})
    _tool_turn(bridge, {"id": "t4", "name": "WebFetch", "input": {"url": "https://example.com", "prompt": "x"}})

    assert "diff" not in calls[1][1]
    assert calls[2][0][2] == "TODO in src"
    assert calls[4][0][2] == "https://example.com"


# --- The native CLIs' own diffs win (2026-09-23) ----------------------------
#
# Fixtures are real output. claude_code_edit_stream.jsonl is Claude Code
# 2.1.280 stream-json from an Edit, a Write over an existing file, a Write that
# creates a file and two more Edits (paths rewritten to /work/probe).
# codex_file_change_item.json holds real Codex FileChange records (an update,
# an add, a delete from ~/.codex/sessions) in the app-server v2 item shape;
# the app-server capture itself was blocked by the Codex usage limit.

import json
from pathlib import Path

from agent.codex_runtime import make_codex_app_server_event_bridge
from agent.tool_diff import claude_native_diff, codex_native_diff

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _claude_fixture_completions():
    agent, calls = _bridge_agent()
    bridge = make_claude_code_event_bridge(agent, cwd="/work/probe")
    for line in (FIXTURES / "claude_code_edit_stream.jsonl").read_text().splitlines():
        bridge(json.loads(line))
    return [kwargs for args, kwargs in calls if args[0] == "tool.completed"]


def test_claude_rows_use_claude_codes_own_hunks_with_real_line_numbers():
    edit, overwrite, create, first_edit, second_edit = _claude_fixture_completions()

    # the file is not on this machine, so these line numbers can only be Claude's
    assert edit["diff"].splitlines() == [
        "--- a/a.txt", "+++ b/a.txt", "@@ -3,6 +3,6 @@",
        " three", " four", " five", "-six", "+SIX", " seven", " eight",
    ]
    assert (edit["lines_added"], edit["lines_removed"]) == (1, 1)
    assert "@@ -5,4 +5,4 @@" in second_edit["diff"] and "+EIGHT" in second_edit["diff"]
    assert first_edit["diff"].splitlines()[3:5] == ["-one", "+ONE"]


def test_a_write_over_an_existing_file_shows_what_it_removed():
    overwrite = _claude_fixture_completions()[1]

    assert overwrite["diff"].splitlines()[2:] == [
        "@@ -1,2 +1,1 @@", "-old first", "-old second", "+new only", "\\ No newline at end of file",
    ]
    assert (overwrite["lines_added"], overwrite["lines_removed"]) == (1, 2)


def test_a_write_that_creates_a_file_falls_back_to_all_added():
    create = _claude_fixture_completions()[2]

    assert create["diff"].splitlines() == ["--- a/c.txt", "+++ b/c.txt", "@@ -0,0 +1 @@", "+fresh"]
    assert (create["lines_added"], create["lines_removed"]) == (1, 0)


def test_claude_native_diff_rejects_malformed_results():
    assert claude_native_diff(None, {"file_path": "x"}) is None
    assert claude_native_diff({"structuredPatch": []}, {"file_path": "x"}) is None
    assert claude_native_diff({"filePath": "x", "structuredPatch": [{"lines": ["+a"]}]}, {}) is None


def _codex_fixture():
    return json.loads((FIXTURES / "codex_file_change_item.json").read_text())


def test_codex_rows_use_codexs_own_diff_for_updates_adds_and_deletes():
    item = _codex_fixture()
    kinds = [change["kind"]["type"] for change in item["changes"]]
    assert kinds == ["update", "add", "delete"]

    rendered = codex_native_diff(item["changes"])

    lines = rendered["diff"].splitlines()
    update, add, delete = item["changes"]
    # Codex's update hunks pass through verbatim under file headers
    update_hunks = update["diff"].rstrip("\n").split("\n")
    assert lines[:2] == [f"--- a/{update['path']}", f"+++ b/{update['path']}"]
    assert lines[2:2 + len(update_hunks)] == update_hunks
    add_lines = add["diff"].rstrip("\n").split("\n")
    delete_lines = delete["diff"].rstrip("\n").split("\n")
    assert f"@@ -0,0 +1,{len(add_lines)} @@" in lines
    assert f"@@ -1,{len(delete_lines)} +0,0 @@" in lines
    update_added = sum(1 for l in update_hunks if l.startswith("+"))
    update_removed = sum(1 for l in update_hunks if l.startswith("-"))
    assert rendered["lines_added"] == update_added + len(add_lines)
    assert rendered["lines_removed"] == update_removed + len(delete_lines)


def test_codex_bridge_sends_codexs_diff_even_without_a_snapshot():
    item = _codex_fixture()
    calls = []
    agent = SimpleNamespace(
        tool_progress_callback=lambda *args, **kwargs: calls.append((args, kwargs)),
        session_cwd="/nonexistent",
    )
    bridge = make_codex_app_server_event_bridge(agent)

    bridge({"method": "item/completed", "params": {"item": item}})

    completed = [kwargs for args, kwargs in calls if args[0] == "tool.completed"][-1]
    expected = codex_native_diff(item["changes"], cwd="/nonexistent")
    assert completed["diff"] == expected["diff"]
    assert completed["lines_added"] == expected["lines_added"]


def test_a_codex_change_without_its_own_diff_keeps_the_snapshot_fallback():
    item = _codex_fixture()
    partial = [dict(item["changes"][0]), {k: v for k, v in item["changes"][1].items() if k != "diff"}]
    assert codex_native_diff(partial) is None
    assert codex_native_diff([{"path": "x", "kind": {"type": "update"}, "diff": "no hunks here"}]) is None
