"""Kanban workers orchestrated for a chat session.

``list_orchestrated_tasks_for_session`` feeds the chat's header indicator: a
task counts as working for a session when a notify subscription targets it on
the api_server platform with the session id as chat_id, and it is in a live
status. Parked, finished, and foreign-session tasks stay out.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    kb.init_db()
    return home


def _running_task(conn, title):
    tid = kb.create_task(conn, title=title, assignee="worker")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    assert kb.claim_task(conn, tid, claimer="worker") is not None
    return tid


def _subscribe(conn, tid, session):
    kb.add_notify_sub(conn, task_id=tid, platform="api_server", chat_id=session)


def test_running_subscribed_task_is_listed_for_its_session(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _running_task(conn, "wire the exporter")
        _subscribe(conn, tid, "hermes-chat-c1")
    rows = kb.list_orchestrated_tasks_for_session("hermes-chat-c1")
    assert [(r["task_id"], r["title"], r["status"], r["board"]) for r in rows] == [
        (tid, "wire the exporter", "running", "default"),
    ]


def test_blocked_and_foreign_session_tasks_are_excluded(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        live = _running_task(conn, "live")
        _subscribe(conn, live, "hermes-chat-c1")
        parked = _running_task(conn, "parked")
        _subscribe(conn, parked, "hermes-chat-c1")
        kb.block_task(conn, parked, reason="x", kind="capability")
        elsewhere = _running_task(conn, "elsewhere")
        _subscribe(conn, elsewhere, "hermes-chat-c2")
    rows = kb.list_orchestrated_tasks_for_session("hermes-chat-c1")
    assert [r["task_id"] for r in rows] == [live]
    assert kb.list_orchestrated_tasks_for_session("hermes-chat-c2") and \
        kb.list_orchestrated_tasks_for_session("hermes-chat-c2")[0]["task_id"] == elsewhere


def test_empty_session_and_missing_board_db_yield_nothing(kanban_home: Path) -> None:
    assert kb.list_orchestrated_tasks_for_session("") == []
    # The default board DB exists (init_db) but holds no subs; a board listed
    # without a database on disk is skipped rather than created.
    assert kb.list_orchestrated_tasks_for_session("hermes-chat-none") == []
