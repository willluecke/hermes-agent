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
