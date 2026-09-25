"""A chat that began before its first Hermes turn keeps its whole history.

2026-09-25: a Hermes Chat conversation whose earlier turns ran on a native
worker reached the gateway with 93 client messages. The gateway stored only
the rows of its own turns (3), and on the next turn replayed those 3 instead
of the client's 95, so its Claude session was rebuilt from 3 messages and
read as "edited or rolled back". The gateway now adopts the client's earlier
messages once and replays them ahead of the stored rows.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from gateway.platforms.api_server import _pre_session_prefix
from hermes_state import SessionDB
from tests.gateway.test_api_server_runs import _create_runs_app, _make_adapter


EARLIER = [
    {"role": "user", "content": "Continue"},
    {"role": "assistant", "content": "worker answer one"},
    {"role": "user", "content": "Proceed"},
    {"role": "assistant", "content": "[Previous run failed before a final answer.]"},
]


def _rc(history):
    """Role and content only: stored rows also carry a timestamp."""
    return [{"role": m["role"], "content": m["content"]} for m in history]


def _stored(db, session_id, rows):
    for row in rows:
        db.append_message(session_id, row["role"], row["content"])


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(tmp_path / "state.db")
    yield session_db
    session_db.close()


def test_prefix_is_everything_before_the_first_stored_user_message():
    # The stored rows repeat an earlier "Continue", and the stored user row
    # carries the gateway's re-attached-image note after the client's text.
    client = EARLIER + [
        {"role": "user", "content": "Continue"},
        {"role": "assistant", "content": "gateway answer"},
        {"role": "user", "content": "Circling  back\nto this"},
        {"role": "assistant", "content": "gateway answer two"},
    ]
    persisted = [
        {"role": "user", "content": "Continue"},
        {"role": "assistant", "content": "gateway answer"},
        {
            "role": "user",
            "content": "Circling back to this\n(Re-attached for reference — image the user shared earlier)",
        },
        {"role": "assistant", "content": "gateway answer two"},
    ]

    prefix, reason = _pre_session_prefix(client, persisted)

    assert reason == ""
    assert prefix == EARLIER


def test_prefix_is_refused_when_the_user_messages_do_not_line_up():
    client = EARLIER + [{"role": "user", "content": "something else"}]
    persisted = [{"role": "user", "content": "Circling back"}]

    prefix, reason = _pre_session_prefix(client, persisted)

    assert prefix is None
    assert "differs" in reason


def test_prefix_is_refused_when_the_client_has_fewer_user_messages():
    persisted = [
        {"role": "user", "content": "a"},
        {"role": "user", "content": "b"},
    ]

    prefix, reason = _pre_session_prefix([{"role": "user", "content": "b"}], persisted)

    assert prefix is None
    assert "1 user messages but 2 are stored" in reason


def test_goal_continuations_are_not_counted_as_user_messages():
    persisted = [
        {"role": "user", "content": "Circling back"},
        {"role": "user", "content": "Continue toward the goal", "display_kind": "auto_continue"},
    ]
    client = EARLIER + [{"role": "user", "content": "Circling back"}]

    prefix, _ = _pre_session_prefix(client, persisted)

    assert prefix == EARLIER


def test_first_turn_records_the_client_history_and_later_turns_replay_it(db):
    adapter = _make_adapter()
    session_id = "chat-first-gateway-turn"

    first, restored = adapter._canonical_run_history(db, session_id, list(EARLIER))
    assert first == EARLIER
    assert restored == 0
    assert db.get_session_imported_history(session_id) == EARLIER

    db.create_session(session_id, "api_server")
    turn = [
        {"role": "user", "content": "Circling back"},
        {"role": "assistant", "content": "gateway answer"},
    ]
    _stored(db, session_id, turn)
    # The client's window may slide or project rows differently; the
    # recorded prefix and the stored rows are what replays.
    client = EARLIER[1:] + turn

    second, restored = adapter._canonical_run_history(db, session_id, client)

    assert _rc(second) == EARLIER + turn
    assert restored == 0


def test_a_brand_new_chat_records_an_empty_prefix(db):
    adapter = _make_adapter()
    session_id = "chat-new"

    history, restored = adapter._canonical_run_history(db, session_id, [])

    assert (history, restored) == ([], 0)
    assert db.get_session_imported_history(session_id) == []


def test_a_session_that_started_mid_chat_gets_its_earlier_messages_back(db):
    adapter = _make_adapter()
    session_id = "chat-legacy"
    db.create_session(session_id, "api_server")
    turn = [
        {"role": "user", "content": "Circling back"},
        {"role": "assistant", "content": "gateway answer"},
    ]
    _stored(db, session_id, turn)

    history, restored = adapter._canonical_run_history(db, session_id, EARLIER + turn)

    assert _rc(history) == EARLIER + turn
    assert restored == len(EARLIER)
    assert db.get_session_imported_history(session_id) == EARLIER

    # Recorded once: the next turn replays it without counting it again.
    again, restored = adapter._canonical_run_history(db, session_id, EARLIER + turn)
    assert _rc(again) == EARLIER + turn
    assert restored == 0


def test_misaligned_history_falls_back_to_the_stored_rows(db, caplog):
    adapter = _make_adapter()
    session_id = "chat-misaligned"
    db.create_session(session_id, "api_server")
    turn = [
        {"role": "user", "content": "Circling back"},
        {"role": "assistant", "content": "gateway answer"},
    ]
    _stored(db, session_id, turn)

    with caplog.at_level("INFO"):
        history, restored = adapter._canonical_run_history(
            db, session_id, EARLIER + [{"role": "user", "content": "unrelated"}]
        )

    assert _rc(history) == turn
    assert restored == 0
    assert db.get_session_imported_history(session_id) is None
    assert "Not adopting the client's earlier messages for session chat-misaligned" in caplog.text


def test_compaction_stops_replaying_the_prefix(db):
    adapter = _make_adapter()
    session_id = "chat-compacted"
    adapter._canonical_run_history(db, session_id, list(EARLIER))
    db.create_session(session_id, "api_server")
    _stored(db, session_id, [{"role": "user", "content": "Circling back"}])

    summary = [{"role": "user", "content": "[Summary of the whole chat]"}]
    db.archive_and_compact(session_id, [dict(m) for m in summary])

    history, _ = adapter._canonical_run_history(db, session_id, [])
    assert _rc(history) == summary


def test_deleting_the_session_drops_its_prefix(db):
    adapter = _make_adapter()
    session_id = "chat-deleted"
    db.create_session(session_id, "api_server")
    adapter._canonical_run_history(db, session_id, list(EARLIER))

    db.delete_session(session_id)

    assert db.get_session_imported_history(session_id) is None


@pytest.mark.asyncio
async def test_run_replays_the_prefix_and_tells_the_agent_it_was_restored(db):
    adapter = _make_adapter()
    adapter._session_db = db
    session_id = "chat-run"
    db.create_session(session_id, "api_server")
    turn = [
        {"role": "user", "content": "Circling back"},
        {"role": "assistant", "content": "gateway answer"},
    ]
    _stored(db, session_id, turn)

    captured = {}
    mock_agent = MagicMock()

    def _capture_run(user_message=None, conversation_history=None, task_id=None):
        captured["history"] = conversation_history
        captured["restored"] = mock_agent._history_prefix_restored
        return {"final_response": "continued"}

    mock_agent.run_conversation.side_effect = _capture_run
    mock_agent.session_prompt_tokens = 0
    mock_agent.session_completion_tokens = 0
    mock_agent.session_total_tokens = 0

    app = _create_runs_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        with patch.object(adapter, "_create_agent", return_value=mock_agent):
            response = await cli.post(
                "/v1/runs",
                json={
                    "input": "One more question",
                    "session_id": session_id,
                    "conversation_history": EARLIER + turn,
                },
            )
            assert response.status == 202
            run_id = (await response.json())["run_id"]
            for _ in range(80):
                status = await (await cli.get(f"/v1/runs/{run_id}")).json()
                if status["status"] == "completed":
                    break
                await asyncio.sleep(0.025)

    assert _rc(captured["history"]) == EARLIER + turn
    assert captured["restored"] == len(EARLIER)
    assert mock_agent._history_prefix_restored == 0
