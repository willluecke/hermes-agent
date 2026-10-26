"""Kanban updates for Hermes Chat conversations: batched, and visible.

The notifier used to answer every task event on an api_server subscription
by self-posting a turn into the gateway session. For a Hermes Chat
conversation that turn was invisible (the app only shows turns it started,
plus late-answer rows), one task event became one full agent turn, and the
flood of turns ran into the gateway's run cap. Reg-watch, 2026-10-25: seven
accepted fixes woke the conversation 41 times, the agent replied 177 times,
and none of it reached the thread.

Now each event that matters is held in a per-board digest
(``kanban_chat_digest``). When nothing the conversation follows on that board
is still in flight, the digest goes out as one turn queued into the
conversation through the sync store, the same path the app's own queued
messages take, so the prompt and the answer are real messages in the thread.
Anything that needs a person (a block that is not a dependency block, a task
that gave up, a block loop routed to triage) goes out on the next tick
instead of waiting for the batch.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger(__name__)

CHAT_SESSION_PREFIX = "hermes-chat-"
#: Events worth telling the conversation about. ``crashed`` and ``timed_out``
#: are retried by the dispatcher (``gave_up`` follows when retries run out).
HELD_KINDS = ("completed", "blocked", "gave_up", "block_loop_detected")
#: Blocks that do not need a person: waiting on another task, or a passing
#: failure the orchestrator reopens.
QUIET_BLOCK_KINDS = ("dependency", "transient")
#: How long the newest held event must sit before a settled batch is sent,
#: so the dispatcher can promote parents and spawn follow-ups first.
SETTLE_SECONDS = 90
RETRY_BACKOFF_SECONDS = 120
MAX_LISTED = 25


class DigestRetry(Exception):
    """The digest could not be delivered now; keep it and try later."""


def is_chat_session(chat_id: Optional[str]) -> bool:
    return bool(chat_id) and str(chat_id).startswith(CHAT_SESSION_PREFIX)


def _first_line(text: Any, limit: int = 200) -> str:
    lines = str(text or "").strip().splitlines()
    return (lines[0] if lines else "")[:limit]


def digest_entries(events: Iterable[Any], task: Any) -> list[dict[str, Any]]:
    """The held entries for one subscription's claimed events."""
    title = (getattr(task, "title", "") or "") if task is not None else ""
    entries: list[dict[str, Any]] = []
    for ev in events:
        kind = getattr(ev, "kind", "")
        if kind not in HELD_KINDS:
            continue
        payload = getattr(ev, "payload", None) or {}
        urgent = False
        if kind == "completed":
            detail = _first_line(payload.get("summary")) or _first_line(
                getattr(task, "result", "") if task is not None else ""
            )
        elif kind == "blocked":
            detail = _first_line(payload.get("reason"))
            urgent = payload.get("kind") not in QUIET_BLOCK_KINDS
        elif kind == "gave_up":
            detail = _first_line(payload.get("error"))
            urgent = True
        else:  # block_loop_detected: routed to triage for a person
            detail = _first_line(payload.get("reason"))
            urgent = True
        entries.append({"kind": kind, "title": title, "detail": detail, "urgent": urgent})
    return entries


def ready_for_flush(conn, *, now: Optional[float] = None) -> list[dict[str, Any]]:
    """Digests on this board that should go out now."""
    from hermes_cli import kanban_db as kb

    now = time.time() if now is None else now
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in kb.chat_digest_rows(conn):
        groups.setdefault((row["platform"], row["chat_id"]), []).append(row)
    ready: list[dict[str, Any]] = []
    for (platform, chat_id), rows in groups.items():
        urgent = any(r["urgent"] for r in rows)
        open_count = kb.open_followed_task_count(conn, platform=platform, chat_id=chat_id)
        settled = open_count == 0 and now - max(r["created_at"] for r in rows) >= SETTLE_SECONDS
        if urgent or settled:
            ready.append({
                "platform": platform, "chat_id": chat_id, "rows": rows,
                "upto_id": max(r["id"] for r in rows), "urgent": urgent,
                "open": open_count,
            })
    return ready


def _latest_per_task(rows: list[dict]) -> list[dict]:
    latest: dict[tuple[str, str], dict] = {}
    for row in rows:
        latest[(row["task_id"], row["kind"])] = row
    return sorted(latest.values(), key=lambda r: r["id"])


def _section(heading: str, rows: list[dict]) -> list[str]:
    if not rows:
        return []
    lines = ["", f"{heading}:"]
    for row in rows[:MAX_LISTED]:
        label = f"- {row['task_id']}"
        if row.get("title"):
            label += f" {row['title']}"
        if row.get("detail"):
            label += f": {row['detail']}"
        lines.append(label)
    if len(rows) > MAX_LISTED:
        lines.append(f"- and {len(rows) - MAX_LISTED} more")
    return lines


def digest_text(board: str, item: dict[str, Any], *, for_agent: bool = True) -> str:
    """The digest, as a prompt for the conversation's agent or, when the turn
    cannot be queued, as a note for the user."""
    rows = _latest_per_task(item["rows"])
    finished = [r for r in rows if r["kind"] == "completed"]
    needs = [r for r in rows if r["kind"] in ("blocked", "block_loop_detected")]
    gave_up = [r for r in rows if r["kind"] == "gave_up"]
    open_count = int(item.get("open") or 0)
    if open_count == 0:
        head = f"Everything this conversation is following on {board} has settled."
    else:
        head = (
            f"Something this conversation follows on {board} needs a decision; "
            f"{open_count} other task{'s' if open_count != 1 else ''} are still in flight."
        )
    lines = [f"[Kanban] {head}"]
    lines += _section("Finished", finished)
    lines += _section("Needs a decision", needs)
    lines += _section("Gave up after retries", gave_up)
    lines.append("")
    if for_agent:
        lines.append(
            "Give Will one short update: what got done, what needs his decision, "
            "and what failed. The board has the detail. Do not redo finished work "
            "or dispatch new tasks unless he asks."
        )
    else:
        lines.append(f"Details: `hermes kanban --board {board} show <id>`.")
    return "\n".join(lines)


def turn_id_for(board: str, item: dict[str, Any]) -> str:
    """Deterministic, so a retried flush replays the same queued turn."""
    digest = hashlib.sha1(f"{board}:{item['chat_id']}:{item['upto_id']}".encode()).hexdigest()
    return f"kanban_{digest[:24]}"


_REFUSED_RE = re.compile(r"refused \((\d{3})\)")


def _queue_turn(conversation_id: str, text: str, turn_id: str) -> None:
    from tools import hermes_conversation_tool as hct

    conversation = hct._conversation(conversation_id)
    if conversation.get("automation"):
        raise ValueError("hermes-sync refused (409): the conversation is an automation console")
    hct._call("/queued-turns", {
        "conversationId": conversation_id, "id": turn_id, "text": text,
        "runtime": hct._runtime_for(conversation), "explicitlyReleased": True,
    })


def _post_note(chat_id: str, text: str, board: str) -> Optional[int]:
    from hermes_cli.kanban_discovery import post_report_to_session

    return post_report_to_session(chat_id, text, board=board)


def deliver(
    board: str,
    item: dict[str, Any],
    *,
    queue: Optional[Callable[[str, str, str], None]] = None,
    post: Optional[Callable[[str, str, str], Optional[int]]] = None,
) -> str:
    """Send one digest. Returns "queued" (a visible turn waits in the
    conversation) or "posted" (the store refused the turn, so the list was
    posted as the orchestrator's own message). Raises ``DigestRetry`` when
    nothing could be delivered and the digest should be kept."""
    queue = queue or _queue_turn
    post = post or _post_note
    chat_id = item["chat_id"]
    conversation_id = chat_id[len(CHAT_SESSION_PREFIX):]
    try:
        queue(conversation_id, digest_text(board, item), turn_id_for(board, item))
        return "queued"
    except ValueError as exc:
        message = str(exc)
        refused = _REFUSED_RE.search(message)
        no_runtime = "no saved model" in message
        if not no_runtime and (not refused or int(refused.group(1)) >= 500):
            raise DigestRetry(message) from None
        logger.info("kanban digest: queued turn refused for %s (%s); posting the list", chat_id, message)
    row = post(chat_id, digest_text(board, item, for_agent=False), board)
    if row is None:
        raise DigestRetry("could not post the digest into the conversation")
    return "posted"
