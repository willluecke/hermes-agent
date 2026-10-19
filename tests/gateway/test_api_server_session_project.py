"""The response store remembers which project workspace a session runs in.

A turn that names only the session (a kanban wake self-post, a resend from
another device) must land in that workspace, not the default project.
"""


def test_response_store_remembers_and_updates_a_session_project(tmp_path):
    from gateway.platforms.api_server import ResponseStore

    store = ResponseStore(db_path=str(tmp_path / "rs.db"))
    try:
        assert store.get_session_project("hermes-chat-c1") is None
        store.remember_session_project("hermes-chat-c1", "reg-watch")
        assert store.get_session_project("hermes-chat-c1") == "reg-watch"
        store.remember_session_project("hermes-chat-c1", "hermes-chat")
        assert store.get_session_project("hermes-chat-c1") == "hermes-chat"
        # Empty values are ignored rather than recorded.
        store.remember_session_project("", "x")
        store.remember_session_project("hermes-chat-c2", "")
        assert store.get_session_project("hermes-chat-c2") is None
        assert store.get_session_project("") is None
    finally:
        store.close()


def test_session_project_survives_reopening_the_store(tmp_path):
    from gateway.platforms.api_server import ResponseStore

    path = str(tmp_path / "rs.db")
    first = ResponseStore(db_path=path)
    first.remember_session_project("hermes-chat-c1", "reg-watch")
    first.close()
    second = ResponseStore(db_path=path)
    try:
        assert second.get_session_project("hermes-chat-c1") == "reg-watch"
    finally:
        second.close()
