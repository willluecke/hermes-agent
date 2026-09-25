"""Late Claude answers: stored, adoptable, and invisible to continuity (2026-09-23)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import agent.claude_runtime as claude_runtime
from agent.claude_runtime import _claude_history_fingerprint, save_claude_late_answer
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_state import ADOPTABLE_DISPLAY_KINDS, LATE_ANSWER_DISPLAY_KIND, SessionDB


@pytest.fixture
def session_db(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        yield db
    finally:
        close = getattr(db, "close", None)
        if callable(close):
            close()


@pytest.fixture
def pings(monkeypatch):
    sent: list[str] = []
    monkeypatch.setattr(claude_runtime, "_sync_store_ping", sent.append)
    return sent


def _record(text="Deploy succeeded: build 42 is live.", **extra):
    return {
        "text": text,
        "complete": True,
        "is_error": False,
        "origin": "task-notification",
        "claude_session_id": "ab25112f-7b78-4d37-8e4f-6362a80b2010",
        "captured_at": 1790135633.2,
        **extra,
    }


def test_runtime_marker_matches_the_session_store_kind():
    assert claude_runtime._LATE_ANSWER_DISPLAY_KIND == LATE_ANSWER_DISPLAY_KIND
    assert LATE_ANSWER_DISPLAY_KIND in ADOPTABLE_DISPLAY_KINDS


def test_fingerprint_ignores_late_answers_wherever_they_land():
    user = {"role": "user", "content": "ship it"}
    answer = {"role": "assistant", "content": "The result will arrive from the poll."}
    late = {"role": "assistant", "content": "Deploy succeeded.", "display_kind": LATE_ANSWER_DISPLAY_KIND}
    base = _claude_history_fingerprint([user, answer])

    assert _claude_history_fingerprint([user, answer, late]) == base
    assert _claude_history_fingerprint([user, late, answer]) == base
    assert _claude_history_fingerprint([user, answer, {"role": "assistant", "content": "x"}]) != base
    # unchanged format for histories without late rows, so stored resume state stays valid
    assert base.startswith("v1:2:")


def test_fingerprint_ignores_the_verify_judges_synthetic_nudge():
    """The store never persists the nudge, so the next turn's prefix lacks it; the session must still match."""
    user = {"role": "user", "content": "Write the AGENTS.md"}
    interim = {"role": "assistant", "content": "Written, committed as ef5cc81."}
    nudge = {"role": "user", "content": "Preflight judge: prove or drop the claim.", "_pre_verify_synthetic": True}
    final = {"role": "assistant", "content": "The judge was right; recommitted as 4f1bde6."}
    persisted = _claude_history_fingerprint([user, interim, final])
    assert _claude_history_fingerprint([user, interim, nudge, final]) == persisted
    stop_nudge = {"role": "user", "content": "verify before finishing", "_verification_stop_synthetic": True}
    assert _claude_history_fingerprint([user, interim, stop_nudge, final]) == persisted
    assert _claude_history_fingerprint([user, interim, {"role": "user", "content": "a real question"}, final]) != persisted


def test_late_answer_is_stored_marked_and_adoptable(session_db, pings):
    session_id = session_db.create_session("hermes-chat-c_late1", "api_server")
    session_db.append_message(session_id, "user", "ship it")
    session_db.append_message(session_id, "assistant", "The result will arrive from the poll.")
    agent = SimpleNamespace(_session_db=session_db, session_id=session_id)

    row_id = save_claude_late_answer(agent, _record())

    assert row_id
    assert pings == [session_id]
    history = session_db.get_messages_as_conversation(session_id)
    assert history[-1]["content"] == "Deploy succeeded: build 42 is live."
    assert history[-1]["display_kind"] == LATE_ANSWER_DISPLAY_KIND
    assert history[-1]["display_metadata"]["origin"] == "task-notification"
    # loading the stored history back does not move the continuity fingerprint
    assert _claude_history_fingerprint(history) == _claude_history_fingerprint(history[:-1])

    rows, max_id = session_db.list_adoptable_messages(after_id=0)
    assert [row["id"] for row in rows] == [row_id]
    assert rows[0]["session_id"] == session_id and rows[0]["role"] == "assistant"
    assert rows[0]["display_metadata"]["complete"] is True
    assert max_id >= row_id
    assert session_db.list_adoptable_messages(after_id=row_id)[0] == []


def test_incomplete_late_answer_is_stored_as_incomplete(session_db, pings):
    session_id = session_db.create_session("hermes-chat-c_late2", "api_server")
    agent = SimpleNamespace(_session_db=session_db, session_id=session_id)

    save_claude_late_answer(agent, _record("Halfway through", complete=False))

    rows, _ = session_db.list_adoptable_messages()
    assert rows[0]["display_metadata"]["complete"] is False


def test_review_forks_and_empty_text_write_nothing(session_db, pings):
    session_id = session_db.create_session("hermes-chat-c_late3", "api_server")
    fork = SimpleNamespace(_session_db=session_db, session_id=session_id, _persist_disabled=True)

    assert save_claude_late_answer(fork, _record()) is None
    assert save_claude_late_answer(SimpleNamespace(_session_db=session_db, session_id=session_id), _record("  ")) is None
    assert session_db.list_adoptable_messages()[0] == []
    assert pings == []


def _app(adapter):
    app = web.Application()
    app.router.add_get("/api/adoptable-messages", adapter._handle_adoptable_messages)
    return app


@pytest.mark.asyncio
async def test_adoptable_messages_endpoint_pages_and_advances_the_cursor(session_db, pings):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-test"}))
    adapter._session_db = session_db
    session_id = session_db.create_session("hermes-chat-c_late4", "api_server")
    for index in range(3):
        session_db.append_message(session_id, "user", f"filler {index}")
    agent = SimpleNamespace(_session_db=session_db, session_id=session_id)
    first = save_claude_late_answer(agent, _record("one"))
    second = save_claude_late_answer(agent, _record("two"))
    session_db.append_message(session_id, "user", "later filler")
    headers = {"Authorization": "Bearer sk-test"}

    async with TestClient(TestServer(_app(adapter))) as cli:
        assert (await cli.get("/api/adoptable-messages")).status == 401
        assert (await cli.get("/api/adoptable-messages?after=-1", headers=headers)).status == 400

        page = await (await cli.get("/api/adoptable-messages?limit=1", headers=headers)).json()
        assert [row["id"] for row in page["data"]] == [first]
        assert page["next_after"] == first

        rest = await (
            await cli.get(f"/api/adoptable-messages?after={first}", headers=headers)
        ).json()
        assert [row["content"] for row in rest["data"]] == ["two"]
        # a short page advances past the trailing non-adoptable row too
        assert rest["next_after"] > second

        empty = await (
            await cli.get(f"/api/adoptable-messages?after={rest['next_after']}", headers=headers)
        ).json()
        assert empty["data"] == [] and empty["next_after"] == rest["next_after"]
