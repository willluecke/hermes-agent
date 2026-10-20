from types import SimpleNamespace
import pytest
from agent.steward_policy import StewardTools, require_steward_runtime
from agent.transports.hermes_tools_mcp_server import selected_tools


def test_steward_cannot_expand_tool_surface():
    assert selected_tools({'HERMES_STEWARD':'1','HERMES_TOOLS_MCP_ONLY':'terminal,delegate_task'})==('talos','hermes_conversation')


@pytest.mark.parametrize('name,args',[('terminal',{}),('delegate_task',{}),('hermes_conversation',{'action':'send','id':'delegate'}),('hermes_conversation',{'action':'goal','verb':'resume'}),('talos',{'action':'unknown'})])
def test_forbidden_tool_actions_are_refused(name,args):
    with pytest.raises(ValueError): StewardTools().check(name,args)


def test_dispatch_allows_one_acknowledgement_then_forces_report():
    policy=StewardTools()
    policy.check('talos',{'action':'dispatch'})
    policy.completed('talos',{'action':'dispatch'},'{}')
    policy.check('talos',{'action':'delegation','id':'one'})
    for args in [{'action':'delegation','id':'two'},{'action':'dispatch'},{'action':'live'}]:
        with pytest.raises(ValueError): policy.check('talos',args)


def test_status_reads_cannot_poll_by_varying_limits():
    policy=StewardTools()
    policy.check('hermes_conversation',{'action':'read','id':'c','limit':10})
    with pytest.raises(ValueError): policy.check('hermes_conversation',{'action':'read','id':'c','limit':20})
    StewardTools().check('hermes_conversation',{'action':'read','id':'c'}) # new user turn


def test_no_unrestricted_runtime_fallback_or_post_turn_review():
    from agent.codex_runtime import _codex_hook_parity
    from agent.claude_runtime import _claude_hook_parity
    agent=SimpleNamespace(session_project='talos-steward',api_mode='chat_completions')
    with pytest.raises(RuntimeError): require_steward_runtime(agent)
    for mode in ['codex_app_server','claude_code']:
        agent.api_mode=mode
        require_steward_runtime(agent)
    assert _codex_hook_parity(agent,None,[],None,'task')==0
    assert _claude_hook_parity(agent,None,None,[],[],None,'task')==0


def test_dispatch_blocks_runtime_change_before_any_remote_read(monkeypatch):
    from unittest.mock import Mock
    import tools.hermes_conversation_tool as conversation
    goal=Mock()
    monkeypatch.setattr(conversation,'_goal',goal)
    policy=StewardTools()
    policy.completed('talos',{'action':'dispatch'},'{}')
    with pytest.raises(ValueError): policy.check('hermes_conversation',{'action':'set_runtime','id':'delegate'})
    goal.assert_not_called()


def test_jev_preflight_never_runs_for_steward(monkeypatch):
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[2]/'plugins/system-one-preflight/__init__.py'
    spec=importlib.util.spec_from_file_location('steward_preflight_fixture',path)
    module=importlib.util.module_from_spec(spec)
    import sys
    monkeypatch.setitem(sys.modules,spec.name,module)
    spec.loader.exec_module(module)
    assert module.on_pre_llm_call(session_id='hermes-chat-talos',user_message='Fix delivery') is None


def test_uncertain_dispatch_error_still_closes_the_action_budget():
    policy=StewardTools()
    def unavailable(*args): raise OSError('lost response')
    with pytest.raises(OSError): policy.run('talos',{'action':'dispatch'},unavailable)
    with pytest.raises(ValueError): policy.run('talos',{'action':'dispatch'},unavailable)
    assert policy.run('talos',{'action':'delegation','id':'d'},lambda *_:'preparing')=='preparing'


def test_parallel_dispatches_cannot_pass_the_same_fence():
    from concurrent.futures import ThreadPoolExecutor
    import threading
    policy=StewardTools()
    start=threading.Barrier(2)
    calls=[]
    def invoke():
        start.wait(timeout=5)
        try: return policy.run('talos',{'action':'dispatch'},lambda *_:calls.append('one'))
        except ValueError: return 'refused'
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda _:invoke(),range(2)))
    assert calls==['one']
    assert results.count('refused')==1


WATCH_SAVE={'action':'watch_save','conversation_id':'regwatch','criteria':'Done means the claim is filed and the receipt number is quoted.',
            'source':{'conversation_id':'talos','message_id':'m1'},'active':True}


def test_watch_mode_actions_are_steward_actions_in_ordinary_turns():
    policy=StewardTools()
    policy.check('talos',{'action':'watches'})
    with pytest.raises(ValueError): policy.check('talos',{'action':'watches'})                      # a read: once per turn
    policy.check('talos',{'action':'watches','conversation_id':'regwatch'})                         # another query is another read
    with pytest.raises(ValueError): policy.check('talos',{'action':'watches','conversation_id':'regwatch'})
    policy.check('talos',WATCH_SAVE)
    policy.check('talos',{**WATCH_SAVE,'active':False})                                             # a write: start, then stop, in one turn
    policy.completed('talos',{'action':'dispatch'},'{}')
    with pytest.raises(ValueError): policy.check('talos',WATCH_SAVE)                                # after a dispatch only its acknowledgement remains


def test_watch_mode_actions_are_refused_in_review_mode():
    review={'id':'final_'+'a'*64,'content_hash':'b'*64,'queue_id':'jim_final_'+'a'*64,'review_run_id':'run_1',
            'source':{'conversation_id':'talos','message_id':'run_1'}}
    policy=StewardTools(review=review)
    for args in [{'action':'watches'},{'action':'watches','conversation_id':'regwatch'},WATCH_SAVE,{**WATCH_SAVE,'active':False}]:
        with pytest.raises(ValueError): policy.check('talos',args)
        with pytest.raises(ValueError): policy.run('talos',args,lambda *_: pytest.fail('a refused action never runs'))


def _steward_home(tmp_path, profile=None):
    docs = tmp_path/'.hermes'/'talos-steward'/'docs'
    docs.mkdir(parents=True)
    (docs/'steward-instructions.md').write_text('You are Jim, Will\'s project manager.')
    if profile is not None:
        (tmp_path/'.hermes'/'memories').mkdir(parents=True)
        (tmp_path/'.hermes'/'memories'/'USER.md').write_text(profile)
    return tmp_path


def test_instructions_append_wills_profile_fresh_each_turn(tmp_path, monkeypatch):
    from agent import steward_policy
    home = _steward_home(tmp_path, 'Will targets ~$10k/mo.\n§\nPriority: revenue this week.')
    monkeypatch.setattr(steward_policy.Path, 'home', classmethod(lambda cls: home))
    monkeypatch.delenv('HERMES_HOME', raising=False)
    text = steward_policy.instructions()
    assert text.startswith("You are Jim, Will's project manager.")
    assert '## About Will' in text
    assert 'Will targets ~$10k/mo.\n\nPriority: revenue this week.' in text
    assert '§' not in text
    (home/'.hermes'/'memories'/'USER.md').write_text('Will now targets $20k/mo.')
    assert 'Will now targets $20k/mo.' in steward_policy.instructions()


def test_instructions_survive_a_missing_profile(tmp_path, monkeypatch):
    from agent import steward_policy
    home = _steward_home(tmp_path)
    monkeypatch.setattr(steward_policy.Path, 'home', classmethod(lambda cls: home))
    monkeypatch.delenv('HERMES_HOME', raising=False)
    assert steward_policy.instructions() == "You are Jim, Will's project manager."


def test_steward_contract_rides_thread_start_and_compaction_only():
    from agent.codex_runtime import _steward_contract_due
    agent = SimpleNamespace(context_compressor=SimpleNamespace(compression_count=0))
    assert _steward_contract_due(agent, 'thread-a')          # first turn of a thread
    assert not _steward_contract_due(agent, 'thread-a')      # ordinary turns skip it
    assert not _steward_contract_due(agent, 'thread-a')
    agent.context_compressor.compression_count = 1           # compaction recorded
    assert _steward_contract_due(agent, 'thread-a')
    assert not _steward_contract_due(agent, 'thread-a')
    agent._codex_compactions_announced = 1                   # previous turn compacted live
    assert _steward_contract_due(agent, 'thread-a')
    agent._codex_compactions_announced = 0
    assert not _steward_contract_due(agent, 'thread-a')
    assert _steward_contract_due(agent, 'thread-b')          # new or rebuilt thread
    agent._steward_contract_key = None                       # a failed turn clears it
    assert _steward_contract_due(agent, 'thread-b')


def test_steward_contract_works_without_a_compressor():
    from agent.codex_runtime import _steward_contract_due
    agent = SimpleNamespace(context_compressor=None)
    assert _steward_contract_due(agent, 'thread-a')
    assert not _steward_contract_due(agent, 'thread-a')
    agent._codex_compactions_announced = 2
    assert _steward_contract_due(agent, 'thread-a')


def test_steward_contract_rides_again_when_its_text_changes():
    from agent.codex_runtime import _steward_contract_due
    agent = SimpleNamespace(context_compressor=SimpleNamespace(compression_count=0))
    assert _steward_contract_due(agent, 'thread-a', 'instructions v1')
    assert not _steward_contract_due(agent, 'thread-a', 'instructions v1')
    assert _steward_contract_due(agent, 'thread-a', 'instructions v2 with the project framework')
    assert not _steward_contract_due(agent, 'thread-a', 'instructions v2 with the project framework')


def test_turn_source_is_the_stewards_submitted_message_and_becomes_a_note():
    from agent.steward_policy import REVIEW_SESSION, source_note, turn_source
    source = {'conversation_id': 'talos', 'message_id': 'turn_mvfhs2vv_htduhdhk'}
    assert turn_source(source, session='hermes-chat-talos', project='talos-steward') == source
    note = source_note(source)
    assert note.startswith('Submitted message source: {"conversation_id":"talos","message_id":"turn_mvfhs2vv_htduhdhk"}')
    assert 'composer drafts are not submissions' in note
    for bad, kw in [({'conversation_id': 'talos'}, {}), ({**source, 'extra': 1}, {}), ({**source, 'message_id': 'a b'}, {}),
                    (source, {'session': 'hermes-chat-other', 'project': 'reg-watch'}), (source, {'session': REVIEW_SESSION, 'project': 'talos-steward'})]:
        with pytest.raises(ValueError):
            turn_source(bad, session=kw.get('session', 'hermes-chat-talos'), project=kw.get('project', 'talos-steward'))


def test_steward_files_reminders_for_will_and_no_other_routine():
    from agent.steward_policy import StewardTools
    reminder = {'action': 'routine_create', 'project_id': 'reg-watch', 'id': 'regwatch-email-warmup', 'owner': 'human',
                'body': {'title': 'Daily RegWatch email warmup', 'estimated_minutes': 5, 'missed_run_policy': 'retain',
                         'recurrence': {'kind': 'daily', 'at': '09:00', 'timezone': 'America/Los_Angeles'}}}
    assert StewardTools().run('talos', reminder, lambda name, args: 'ok') == 'ok'
    for broken in ({**reminder, 'owner': 'agent'},                                   # an agent routine runs work
                   {**reminder, 'body': {**reminder['body'], 'kind': 'steward'}},    # what Jim tried on 2026-10-19
                   {**reminder, 'body': {**reminder['body'], 'contract': {}}},       # a task contract
                   {**reminder, 'body': {**reminder['body'], 'missed_run_policy': 'coalesce'}},
                   {**reminder, 'body': {**reminder['body'], 'title': ' '}},
                   {k: v for k, v in reminder.items() if k != 'owner'}):
        with pytest.raises(ValueError):
            StewardTools().run('talos', broken, lambda name, args: 'ok')
    review = {'id': 'final_'+'a'*64, 'content_hash': 'b'*64, 'queue_id': 'jim_final_'+'a'*64, 'review_run_id': 'run_1',
              'source': {'conversation_id': 'talos', 'message_id': 'run_1'}}
    with pytest.raises(ValueError):  # review mode admits only the review actions
        StewardTools(review=review).run('talos', reminder, lambda name, args: 'ok')
