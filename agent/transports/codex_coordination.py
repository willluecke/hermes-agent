"""Shared live/history mapping for native Codex coordination items.

These are app-server items, not a reconstruction of hidden model reasoning.
The current protocol exposes agent lifecycle events separately from wait calls.
"""

import json


COORDINATION_ITEM_TYPES = frozenset({
    "collabAgentToolCall", "collabToolCall", "subAgentActivity", "sleep",
})


def coordination_item(item: dict) -> tuple[str, dict, str, bool, str]:
    """Return name, arguments, result, error flag, and display label."""
    if item["type"] == "subAgentActivity":
        path = item.get("agentPath") or item.get("agentThreadId") or "agent"
        kind = item.get("kind") or "updated"
        label = f"{path}: {kind}"
        return "agent_activity", {
            key: item[key] for key in ("agentPath", "agentThreadId", "kind")
            if key in item
        }, label, False, label
    if item["type"] == "sleep":
        duration_ms = item.get("durationMs") or 0
        label = f"Sleep for {duration_ms / 1000:g}s"
        return "sleep", {"duration_ms": duration_ms}, "Sleep ended", False, label

    tool = item.get("tool") or "agent_activity"
    name = {
        "spawnAgent": "spawn_agent", "sendInput": "send_input",
        "resumeAgent": "resume_agent", "wait": "wait_agent",
        "closeAgent": "close_agent", "sendMessage": "send_message",
        "followupTask": "followup_task", "interruptAgent": "interrupt_agent",
        "listAgents": "list_agents",
    }.get(tool, tool)
    args = {key: item[key] for key in (
        "receiverThreadIds", "receiverThreadId", "newThreadId", "prompt",
        "model", "reasoningEffort",
    ) if item.get(key) is not None}
    targets = item.get("receiverThreadIds") or [
        item.get("receiverThreadId") or item.get("newThreadId")
    ]
    targets = [str(target) for target in targets if target]
    label = "Waiting for agent updates" if name == "wait_agent" else name
    if targets:
        label += ": " + ", ".join(targets)
    if item.get("prompt"):
        label += "\n" + item["prompt"]
    states = item.get("agentsStates") or item.get("agentStatus") or {}
    status = item.get("status") or "unknown"
    result = json.dumps({"status": status, "agents": states}, ensure_ascii=False)
    if name == "wait_agent" and status == "completed" and not states:
        # This is what 0.153.4 reports for timed-out waits. Do not invent an
        # agent result or infer that the parent turn itself has completed.
        result = "Wait ended without an agent update."
    return name, args, result, status in {"failed", "interrupted"}, label
