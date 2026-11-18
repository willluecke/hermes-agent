"""Kanban updates for Hermes Chat conversations: held, batched, visible.

Reg-watch, 2026-10-25: seven accepted fixes woke the conversation 41 times
through hidden gateway turns and nothing reached the thread. Events are now
held in a per-board digest and go out as one queued, visible turn when the
batch the conversation follows settles, or at once when a person is needed.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import kanban_chat_digest as kcd
from hermes_cli import kanban_db as kb

CHAT = "hermes-chat-c_digest"


@pytest.fixture
def board(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
    try:
        yield conn
    finally:
        conn.close()


def _ev(event_kind, **payload):
    return SimpleNamespace(kind=event_kind, payload=payload)


def _followed(conn, title, status="running", block_kind=None):
    tid = kb.create_task(conn, title=title, assignee="worker")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status = ?, block_kind = ? WHERE id = ?", (status, block_kind, tid))
    kb.add_notify_sub(conn, task_id=tid, platform="api_server", chat_id=CHAT)
    return tid


def test_entries_hold_what_matters_and_flag_what_needs_a_person():
    task = SimpleNamespace(title="Trial cancel still charges", result="")
    entries = kcd.digest_entries([
        _ev("completed", summary="Cancelling in trial now ends it with no invoice\nmore"),
        _ev("review_requested", summary="Built, not shipped. Commit 88d79c24\nbranch fix/x"),
        _ev("crashed"),
        _ev("timed_out", limit_seconds=60),
        _ev("blocked", reason="waiting on t_parent", kind="dependency"),
        _ev("blocked", reason="needs the Stripe portal id", kind="needs_input"),
        _ev("gave_up", error="3 failures"),
        _ev("block_loop_detected", reason="same cause 3x"),
    ], task)
    assert [(e["kind"], e["urgent"]) for e in entries] == [
        ("completed", False), ("review_requested", True), ("blocked", False),
        ("blocked", True), ("gave_up", True), ("block_loop_detected", True),
    ]
    assert entries[0]["detail"] == "Cancelling in trial now ends it with no invoice"
    assert entries[0]["title"] == "Trial cancel still charges"
    # Talos, 2026-11-08: a worker finishing a card by requesting review is a
    # person's decision; it is held and goes out at once with its summary.
    assert entries[1]["detail"] == "Built, not shipped. Commit 88d79c24"


def test_a_pipeline_region_reaches_the_chat_only_when_it_needs_will():
    """Feature pipeline, 2026-11-18: regions finish through Astra's review and
    the pipeline posts its own notes, so a region's review request and
    completion stay quiet; its question or failure still goes out at once."""
    region = SimpleNamespace(title="Region 1/2: notes page", result="", skills=["feature-pipeline-region"])
    entries = kcd.digest_entries([
        _ev("review_requested", summary="built"),
        _ev("completed", summary="approved"),
        _ev("blocked", reason="which export format?", kind="needs_input"),
        _ev("gave_up", error="3 failures"),
    ], region)
    assert [(e["kind"], e["urgent"]) for e in entries] == [("blocked", True), ("gave_up", True)]
    root = SimpleNamespace(title="Feature pipeline (build): notes", result="", skills=["feature-pipeline"])
    assert [e["kind"] for e in kcd.digest_entries([_ev("completed", summary="done")], root)] == ["completed"]
    question = "Question for Will: " + "which export format, CSV or JSON? " * 16
    entry = kcd.digest_entries([_ev("blocked", reason=question, kind="needs_input")], root)[0]
    assert entry["detail"] == question.strip()[:600] and len(entry["detail"]) > 200, \
        "a question for Will is kept as far as the digest stores it, not cut at 200 characters"


def test_storage_and_the_in_flight_count(board):
    running = _followed(board, "running one")
    _followed(board, "done one", status="done")
    _followed(board, "waits on a person", status="blocked", block_kind="needs_input")
    _followed(board, "parked for later", status="scheduled")
    # A card in review waits on the reviewer (this conversation); it must not
    # hold back the other cards' notices (Talos, 2026-11-08).
    _followed(board, "awaits the reviewer", status="review")
    waiting = _followed(board, "waits on a task", status="blocked", block_kind="dependency")
    assert kb.open_followed_task_count(board, platform="api_server", chat_id=CHAT) == 2
    first = kb.chat_digest_add(board, platform="api_server", chat_id=CHAT, task_id=running,
                               kind="completed", title="running one")
    kb.chat_digest_add(board, platform="api_server", chat_id=CHAT, task_id=waiting, kind="completed")
    assert len(kb.chat_digest_rows(board, platform="api_server", chat_id=CHAT)) == 2
    assert kb.chat_digest_clear(board, platform="api_server", chat_id=CHAT, upto_id=first) == 1
    assert [r["task_id"] for r in kb.chat_digest_rows(board)] == [waiting]


def test_a_batch_goes_out_only_when_settled_unless_someone_is_needed(board):
    tid = _followed(board, "still running")
    kb.chat_digest_add(board, platform="api_server", chat_id=CHAT, task_id=tid, kind="completed")
    now = time.time() + kcd.SETTLE_SECONDS + 5
    assert kcd.ready_for_flush(board, now=now) == []  # a followed task is in flight
    with kb.write_txn(board):
        board.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (tid,))
    assert kcd.ready_for_flush(board, now=time.time()) == []  # not settled long enough
    ready = kcd.ready_for_flush(board, now=now)
    assert len(ready) == 1 and ready[0]["chat_id"] == CHAT and ready[0]["open"] == 0
    # a person is needed: it goes out at once, batch or not
    other = _followed(board, "another running")
    kb.chat_digest_add(board, platform="api_server", chat_id=CHAT, task_id=other,
                       kind="gave_up", detail="3 failures", urgent=True)
    urgent = kcd.ready_for_flush(board, now=time.time())
    assert len(urgent) == 1 and urgent[0]["urgent"] is True and urgent[0]["open"] == 1


def _item(rows, open_count=0):
    return {"platform": "api_server", "chat_id": CHAT, "rows": rows,
            "upto_id": max(r["id"] for r in rows), "urgent": False, "open": open_count}


ROWS = [
    {"id": 1, "task_id": "t_00000001", "kind": "completed", "title": "Trial cancel", "detail": "no invoice now"},
    {"id": 2, "task_id": "t_00000002", "kind": "blocked", "title": "Portal", "detail": "needs the portal id"},
    {"id": 3, "task_id": "t_00000003", "kind": "gave_up", "title": "Emails", "detail": "3 failures"},
    {"id": 4, "task_id": "t_00000004", "kind": "review_requested", "title": "Affiliate links", "detail": "Built, not shipped. Commit 88d79c24"},
]


def test_digest_text_lists_finished_needs_and_gave_up():
    text = kcd.digest_text("reg-watch", _item(ROWS))
    assert text.startswith("[Kanban] Everything this conversation is following on reg-watch has settled.")
    assert "Finished:\n- t_00000001 Trial cancel: no invoice now" in text
    assert "Ready for review:\n- t_00000004 Affiliate links: Built, not shipped. Commit 88d79c24" in text
    assert "Needs a decision:\n- t_00000002 Portal: needs the portal id" in text
    assert "Gave up after retries:\n- t_00000003 Emails: 3 failures" in text
    assert "Give Will one short update" in text
    note = kcd.digest_text("reg-watch", _item(ROWS, open_count=2), for_agent=False)
    assert "needs a decision; 2 other tasks are still in flight" in note
    assert "Give Will" not in note and "hermes kanban --board reg-watch show <id>" in note


def test_deliver_queues_a_visible_turn_with_a_stable_id():
    sent = []
    outcome = kcd.deliver("reg-watch", _item(ROWS), queue=lambda cid, text, tid: sent.append((cid, text, tid)),
                          post=lambda *a: pytest.fail("must not post"))
    assert outcome == "queued"
    cid, text, tid = sent[0]
    assert cid == "c_digest" and text.startswith("[Kanban]")
    assert tid == kcd.turn_id_for("reg-watch", _item(ROWS)) and tid.startswith("kanban_")


def test_a_refused_turn_posts_the_list_and_an_unreachable_store_retries():
    def refused(*a):
        raise ValueError("hermes-sync refused (409): A managed goal owns this repository")

    posted = []
    outcome = kcd.deliver("reg-watch", _item(ROWS), queue=refused,
                          post=lambda chat, text, b: posted.append((chat, text)) or 9)
    assert outcome == "posted" and posted[0][0] == CHAT and "Give Will" not in posted[0][1]

    def unreachable(*a):
        raise ValueError("hermes-sync is unreachable: connection refused")

    with pytest.raises(kcd.DigestRetry):
        kcd.deliver("reg-watch", _item(ROWS), queue=unreachable, post=lambda *a: pytest.fail("no post"))

    def busy(*a):
        raise ValueError("hermes-sync refused (503): try again")

    with pytest.raises(kcd.DigestRetry):
        kcd.deliver("reg-watch", _item(ROWS), queue=busy, post=lambda *a: pytest.fail("no post"))
    with pytest.raises(kcd.DigestRetry):
        kcd.deliver("reg-watch", _item(ROWS), queue=refused, post=lambda *a: None)
