"""Regression tests for live Codex app-server events.

The history projector is completion-only. These tests protect the parallel
display bridge (make_codex_app_server_event_bridge) that makes deltas and
tool cards visible before resume: it fires both the tool_progress bubbles
AND the authoritative stable-ID tool_start/tool_complete callbacks the TUI
tool cards depend on.

Grafted from PR #65412 (@HaiderSultanArc) onto the merged bridge.
"""

from types import SimpleNamespace

from agent.codex_runtime import (
    _codex_item_completion_payload,
    make_codex_app_server_event_bridge,
)
from agent.transports.codex_event_projector import _deterministic_call_id


def _recording_agent():
    calls = {
        "stream": [],
        "reasoning": [],
        "tool_progress": [],
        "tool_start": [],
        "tool_complete": [],
    }
    agent = SimpleNamespace(
        _fire_stream_delta=lambda text: calls["stream"].append(text),
        _fire_reasoning_delta=lambda text: calls["reasoning"].append(text),
        tool_progress_callback=lambda *args, **kwargs: calls["tool_progress"].append((
            args,
            kwargs,
        )),
        tool_start_callback=lambda call_id, name, args: calls["tool_start"].append((
            call_id,
            name,
            args,
        )),
        tool_complete_callback=lambda call_id, name, args, result: calls[
            "tool_complete"
        ].append((call_id, name, args, result)),
        _emit_interim_assistant_message=None,
        show_commentary=True,
    )
    return agent, calls






def test_stable_ids_match_history_projector():
    """The bridge's stable call ids mirror CodexEventProjector so a live
    TUI tool card correlates with the projected history entry after
    resume."""
    agent, calls = _recording_agent()
    bridge = make_codex_app_server_event_bridge(agent)

    mcp = {
        "type": "mcpToolCall",
        "id": "m1",
        "server": "filesystem",
        "tool": "read",
        "arguments": {"path": "a.py"},
    }
    bridge({"method": "item/started", "params": {"item": mcp}})
    call_id, name, args = calls["tool_start"][0]
    assert call_id == _deterministic_call_id("mcp__filesystem__read", "m1")
    assert name == "mcp.filesystem.read"
    assert args == {"path": "a.py"}

    calls["tool_start"].clear()
    patch = {
        "type": "fileChange",
        "id": "p1",
        "changes": [{"kind": {"type": "add"}, "path": "a.py"}],
    }
    bridge({"method": "item/started", "params": {"item": patch}})
    call_id, name, args = calls["tool_start"][0]
    assert call_id == _deterministic_call_id("apply_patch", "p1")
    assert name == "apply_patch"
    assert args == {"changes": [{"kind": "add", "path": "a.py"}]}


def test_failed_command_result_and_error_flag_are_preserved():
    agent, calls = _recording_agent()
    bridge = make_codex_app_server_event_bridge(agent)
    item = {
        "type": "commandExecution",
        "id": "failed",
        "command": "false",
        "aggregatedOutput": "boom",
        "exitCode": 2,
    }

    bridge({"method": "item/completed", "params": {"item": item}})

    result, is_error = _codex_item_completion_payload(item)
    assert result == "[exit 2]\nboom"
    assert is_error is True
    assert calls["tool_progress"][0][1]["is_error"] is True
    assert calls["tool_complete"][0][3] == "[exit 2]\nboom"


def test_file_change_completion_reports_exact_line_counts(tmp_path):
    agent, calls = _recording_agent()
    agent.session_cwd = str(tmp_path)
    target = tmp_path / "example.py"
    target.write_text("one\ntwo\nthree\n", encoding="utf-8")
    bridge = make_codex_app_server_event_bridge(agent)
    started = {
        "type": "fileChange",
        "id": "patch-counts",
        "changes": [{"kind": {"type": "update"}, "path": str(target)}],
    }

    bridge({"method": "item/started", "params": {"item": started}})
    target.write_text("one\nsecond\nthree\nfour\n", encoding="utf-8")
    bridge({
        "method": "item/completed",
        "params": {"item": {**started, "status": "completed"}},
    })

    _args, kwargs = calls["tool_progress"][-1]
    assert kwargs["lines_added"] == 2
    assert kwargs["lines_removed"] == 1


def test_command_output_delta_keeps_stable_id_and_wire_order():
    agent, calls = _recording_agent()
    bridge = make_codex_app_server_event_bridge(agent)
    started = {
        "type": "commandExecution",
        "id": "cmd-live-1",
        "command": "printf hello",
    }
    completed = {
        **started,
        "aggregatedOutput": "hello",
        "exitCode": 0,
        "durationMs": 25,
    }

    bridge({"method": "item/started", "params": {"item": started}})
    bridge({
        "method": "item/commandExecution/outputDelta",
        "params": {"itemId": "cmd-live-1", "delta": "hel"},
    })
    bridge({
        "method": "item/commandExecution/outputDelta",
        "params": {"itemId": "cmd-live-1", "delta": "lo"},
    })
    bridge({"method": "item/completed", "params": {"item": completed}})

    event_types = [args[0] for args, _kwargs in calls["tool_progress"]]
    assert event_types == [
        "tool.started",
        "tool.output.delta",
        "tool.output.delta",
        "tool.completed",
    ]
    stable_id = _deterministic_call_id("exec", "cmd-live-1")
    assert [
        kwargs["tool_call_id"] for _args, kwargs in calls["tool_progress"]
    ] == [stable_id] * 4
    assert [
        kwargs["chunk"]
        for args, kwargs in calls["tool_progress"]
        if args[0] == "tool.output.delta"
    ] == ["hel", "lo"]
