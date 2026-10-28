"""The Codex steward lane carries the gateway-bound final review (and nothing
else) into the session it spawns for the turn; the steward contract still
rides the first turn of the thread."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent import steward_policy
from agent.codex_runtime import run_codex_app_server_turn

RUN = 'run_' + 'c' * 32
REVIEW = {'id': 'final_' + 'a' * 64, 'content_hash': 'b' * 64, 'queue_id': 'jim_final_' + 'a' * 64,
          'review_run_id': RUN, 'source': {'conversation_id': 'talos', 'message_id': RUN}}


class FakeSession:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.inputs = []
        self._thread_id = 'thread-review'
        self._resumed_existing_thread = False
        FakeSession.instances.append(self)

    def capture_workspace_snapshot(self, task_id):
        return None

    def ensure_started(self):
        return self._thread_id

    def run_turn(self, user_input, absolute_turn_timeout=None, **_):
        self.inputs.append(user_input)
        return SimpleNamespace(interrupted=False, error=None, thread_id=self._thread_id, turn_id='turn-1',
                               projected_messages=[{'role': 'assistant', 'content': 'Acknowledged.'}],
                               tool_iterations=1, final_text='Acknowledged.', should_retire=False, error_code=None)

    def close(self):
        pass


def _steward(tmp_path, monkeypatch, *, session_id, review):
    docs = tmp_path / '.hermes' / 'talos-steward' / 'docs'
    docs.mkdir(parents=True, exist_ok=True)
    (docs / 'steward-instructions.md').write_text('You are Jim. Review the assigned final.')
    monkeypatch.setattr(steward_policy.Path, 'home', classmethod(lambda cls: tmp_path))
    agent = MagicMock()
    agent._codex_session = None
    agent._codex_thread_state = {}
    agent._session_db = None
    agent._session_db_created = True
    agent.session_id = session_id
    agent.session_project = 'talos-steward'
    agent.session_cwd = str(tmp_path / '.hermes' / 'talos-steward')
    agent._steward_review = review
    agent.tool_progress_callback = None
    agent._iters_since_skill = 0
    agent._skill_nudge_interval = 0
    agent.valid_tool_names = set()
    agent.context_compressor = None
    agent._codex_compactions_announced = 0
    agent._history_prefix_restored = 0
    agent.read_only = False
    agent.reasoning_config = None
    agent.model = 'gpt-6.1-sol'
    agent.provider = 'openai-codex'
    agent.platform = 'api_server'
    return agent


@pytest.mark.parametrize('session_id,review', [('hermes-steward-review', REVIEW), ('hermes-chat-talos', None)])
def test_codex_steward_lane_spawns_the_session_with_exactly_the_bound_review(tmp_path, monkeypatch, session_id, review):
    import agent.transports.codex_app_server_session as transport
    FakeSession.instances.clear()
    monkeypatch.setattr(transport, 'CodexAppServerSession', FakeSession)
    agent = _steward(tmp_path, monkeypatch, session_id=session_id, review=review)
    result = run_codex_app_server_turn(agent, user_message='Internal final review', original_user_message='Internal final review',
                                       messages=[{'role': 'user', 'content': 'Internal final review'}],
                                       effective_task_id=session_id)
    assert result['completed'] is True and result['final_response'] == 'Acknowledged.'
    session, = FakeSession.instances
    assert session.kwargs['steward_review'] == review
    assert session.kwargs['project_key'] == 'talos-steward' and session.kwargs['hermes_session_id'] == session_id
    assert session.kwargs['read_only'] is False  # the session itself pins read-only for the steward
    # Jim's contract rides the first turn of the thread as before.
    assert len(session.inputs) == 1 and 'You are Jim.' in str(session.inputs[0])
    assert 'HERMES_STEWARD_REVIEW' not in str(session.inputs[0])


def test_codex_steward_lane_refuses_a_malformed_binding_instead_of_running_unbound(tmp_path, monkeypatch):
    import agent.transports.codex_app_server_session as transport
    from agent.transports.codex_app_server_session import CodexAppServerSession
    spawned = []

    def factory(**kwargs):
        spawned.append(kwargs)
        raise AssertionError('no Codex process may be spawned for a malformed binding')
    monkeypatch.setattr(transport, 'CodexAppServerSession',
                        lambda **kwargs: CodexAppServerSession(**{**kwargs, 'client_factory': factory}))
    agent = _steward(tmp_path, monkeypatch, session_id='hermes-steward-review',
                     review={**REVIEW, 'source': {'conversation_id': 'delegate', 'message_id': RUN}})
    # The turn fails before any Codex process exists; nothing runs unbound.
    with pytest.raises(ValueError):
        run_codex_app_server_turn(agent, user_message='Internal final review', original_user_message='Internal final review',
                                  messages=[{'role': 'user', 'content': 'Internal final review'}],
                                  effective_task_id='hermes-steward-review')
    assert spawned == [] and agent._codex_session is None


@pytest.mark.parametrize('session_id,review,sessions_expected', [('hermes-chat-talos', None, 1), ('hermes-steward-review', REVIEW, 2)])
def test_ordinary_steward_turns_keep_the_resident_session_but_reviews_respawn(tmp_path, monkeypatch, session_id, review, sessions_expected):
    # 2026-10-28: the per-turn rebuild that showed Will "Session resumed from
    # disk" on every message is gone; only a bound review spawns its own process.
    import agent.transports.codex_app_server_session as transport
    FakeSession.instances.clear()
    monkeypatch.setattr(transport, 'CodexAppServerSession', FakeSession)
    agent = _steward(tmp_path, monkeypatch, session_id=session_id, review=review)
    first = run_codex_app_server_turn(agent, user_message='First', original_user_message='First',
                                      messages=[{'role': 'user', 'content': 'First'}], effective_task_id=session_id)
    assert first['completed'] is True
    history = [{'role': 'user', 'content': 'First'}, {'role': 'assistant', 'content': 'Acknowledged.'}, {'role': 'user', 'content': 'Second'}]
    second = run_codex_app_server_turn(agent, user_message='Second', original_user_message='Second', messages=history, effective_task_id=session_id)
    assert second['completed'] is True
    assert len(FakeSession.instances) == sessions_expected
    if sessions_expected == 1:
        session, = FakeSession.instances
        assert len(session.inputs) == 2, 'the same resident process took both turns'
        assert 'You are Jim.' not in str(session.inputs[1]), 'the contract is not re-sent on an ordinary turn'
        assert agent._codex_session is session


def test_a_requested_reset_starts_a_fresh_thread_with_the_brief_and_no_transcript(tmp_path, monkeypatch):
    # Will, 2026-10-28: "reset your context" starts the next turn on a fresh
    # thread seeded with the instructions and the Talos state brief only.
    import agent.transports.codex_app_server_session as transport
    import tools.talos_tool as talos
    from agent import steward_policy
    FakeSession.instances.clear()
    monkeypatch.setattr(transport, 'CodexAppServerSession', FakeSession)
    monkeypatch.setattr(talos, '_call', lambda path, payload=None, timeout=20: {'text': 'Talos brief at now.\nOpen work: 2 waiting.', 'at': 1})
    agent = _steward(tmp_path, monkeypatch, session_id='hermes-chat-talos', review=None)
    rows = []
    agent.tool_progress_callback = lambda event, runtime, text, _none, **payload: rows.append((payload.get('mode'), text))
    run_codex_app_server_turn(agent, user_message='First', original_user_message='First',
                              messages=[{'role': 'user', 'content': 'First'}], effective_task_id='hermes-chat-talos')
    assert 'Talos brief at now.' in str(FakeSession.instances[0].inputs[0])  # a new thread gets the brief with the contract
    steward_policy.request_reset('Will asked')
    history = [{'role': 'user', 'content': 'First'}, {'role': 'assistant', 'content': 'Acknowledged.'}, {'role': 'user', 'content': 'Second'}]
    result = run_codex_app_server_turn(agent, user_message='Second', original_user_message='Second', messages=history, effective_task_id='hermes-chat-talos')
    assert result['completed'] is True
    assert len(FakeSession.instances) == 2 and FakeSession.instances[1].kwargs['resume_thread_id'] == ''
    fresh_input = str(FakeSession.instances[1].inputs[0])
    assert 'hermes_conversation_history' not in fresh_input and 'Acknowledged.' not in fresh_input, 'no transcript replay'
    assert 'You are Jim.' in fresh_input and 'Talos brief at now.' in fresh_input
    assert steward_policy.pending_reset() is None, 'the request is spent once the fresh thread is bound'
    assert any(mode == 'reset' and 'Will asked' in text and 'not replayed' in text for mode, text in rows)
    history += [{'role': 'assistant', 'content': 'Acknowledged.'}, {'role': 'user', 'content': 'Third'}]
    run_codex_app_server_turn(agent, user_message='Third', original_user_message='Third', messages=history, effective_task_id='hermes-chat-talos')
    assert len(FakeSession.instances) == 2 and len(FakeSession.instances[1].inputs) == 2, 'the fresh thread is resumed afterwards'
    assert 'Talos brief' not in str(FakeSession.instances[1].inputs[1]), 'the brief does not ride an ordinary turn'


def test_a_review_run_never_consumes_a_pending_reset(tmp_path, monkeypatch):
    import agent.transports.codex_app_server_session as transport
    from agent import steward_policy
    FakeSession.instances.clear()
    monkeypatch.setattr(transport, 'CodexAppServerSession', FakeSession)
    agent = _steward(tmp_path, monkeypatch, session_id='hermes-steward-review', review=REVIEW)
    steward_policy.request_reset('pending for the talos thread')
    run_codex_app_server_turn(agent, user_message='Internal final review', original_user_message='Internal final review',
                              messages=[{'role': 'user', 'content': 'Internal final review'}], effective_task_id='hermes-steward-review')
    assert steward_policy.pending_reset()['reason'] == 'pending for the talos thread'
