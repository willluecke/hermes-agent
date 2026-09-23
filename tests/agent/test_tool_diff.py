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
