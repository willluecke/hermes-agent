"""Exercise coordination items from the installed Codex 0.153.4 protocol."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.codex_runtime import make_codex_app_server_event_bridge
from agent.transports.codex_event_projector import CodexEventProjector


def notification(method, item):
    return {"method": method, "params": {"threadId": "parent", "turnId": "turn", "item": item}}


def test_delegation_wait_and_completion_stay_visible_without_finishing_parent(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr("agent.codex_runtime.time.monotonic", lambda: clock[0])
    agent = SimpleNamespace(tool_progress_callback=MagicMock())
    bridge = make_codex_app_server_event_bridge(agent)
    projector = CodexEventProjector()
    lifecycle = {"type": "subAgentActivity", "id": "spawn", "kind": "started",
                 "agentThreadId": "child", "agentPath": "/root/documenter"}
    bridge(notification("item/completed", lifecycle))
    wait = {"type": "collabAgentToolCall", "id": "wait-1", "tool": "wait",
            "status": "inProgress", "senderThreadId": "parent",
            "receiverThreadIds": [], "agentsStates": {}}
    bridge(notification("item/started", wait))
    start = agent.tool_progress_callback.call_args
    assert start.args[0:2] == ("tool.started", "wait_agent")
    assert "/root/documenter" in start.args[2]
    assert "Waiting for agent updates" in start.args[2]
    clock[0] += 60
    complete = notification("item/completed", {**wait, "status": "completed"})
    bridge(complete)
    end = agent.tool_progress_callback.call_args
    assert end.kwargs["duration"] == 60
    assert end.kwargs["tool_call_id"] == start.kwargs["tool_call_id"]
    assert end.kwargs["result"] == "Wait ended without an agent update."
    assert not end.kwargs["is_error"]
    projected = projector.project(complete)
    assert projected.final_text is None and not projected.is_final_answer
    assert projected.messages[-1]["tool_call_id"] == start.kwargs["tool_call_id"]
    assert projected.messages[-1]["content"] == end.kwargs["result"]
    bridge(notification("item/completed", {**lifecycle, "id": "child-end", "kind": "completed"}))
    bridge(notification("item/started", {**wait, "id": "wait-2"}))
    assert "documenter" not in agent.tool_progress_callback.call_args.args[2]


@pytest.mark.parametrize("tool,name", [
    ("spawnAgent", "spawn_agent"), ("sendMessage", "send_message"),
    ("followupTask", "followup_task"), ("interruptAgent", "interrupt_agent"),
])
def test_collaboration_result_preserves_identity_and_agent_result(tool, name):
    agent = SimpleNamespace(tool_progress_callback=MagicMock())
    bridge = make_codex_app_server_event_bridge(agent)
    item = {"type": "collabAgentToolCall", "id": "operation", "tool": tool,
            "status": "completed", "senderThreadId": "parent", "receiverThreadIds": ["child"],
            "agentsStates": {"child": {"status": "completed", "message": "Review passed."}}}
    bridge(notification("item/completed", item))
    start, end = agent.tool_progress_callback.call_args_list
    assert start.args[1] == end.args[1] == name
    assert "child" in start.args[2]
    assert "Review passed." in end.kwargs["result"]
    projected = CodexEventProjector().project(notification("item/completed", item))
    assert projected.messages[-1]["content"] == end.kwargs["result"]


def test_cancelled_wait_is_failed_activity_and_never_a_final_answer():
    agent = SimpleNamespace(tool_progress_callback=MagicMock())
    item = {"type": "collabAgentToolCall", "id": "wait", "tool": "wait", "status": "interrupted"}
    make_codex_app_server_event_bridge(agent)(notification("item/completed", item))
    assert agent.tool_progress_callback.call_args.kwargs["is_error"]
    assert not CodexEventProjector().project(notification("item/completed", item)).is_final_answer
