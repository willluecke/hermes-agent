from types import SimpleNamespace
import pytest
from agent.steward_policy import StewardTools, require_steward_runtime
from agent.transports.hermes_tools_mcp_server import selected_tools


def test_steward_sees_the_full_tool_surface():
    # Will, 2026-10-28: no guard rails except project deletion. HERMES_STEWARD
    # keeps the identity for that one refusal; it no longer narrows the tools.
    from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS
    assert selected_tools({'HERMES_STEWARD':'1'})==EXPOSED_TOOLS
    assert 'terminal' not in EXPOSED_TOOLS  # codex/claude native tools cover the shell and files
    assert selected_tools({'HERMES_STEWARD_REVIEW':'{}'})==('talos',)


@pytest.mark.parametrize('name,args',[('terminal',{}),('delegate_task',{}),('hermes_conversation',{'action':'send','id':'delegate'}),
                                      ('hermes_conversation',{'action':'goal','verb':'resume'}),('hermes_conversation',{'action':'create'}),
                                      ('talos',{'action':'admit'}),('talos',{'action':'project_save'}),('talos',{'action':'grant_revoke'}),
                                      ('talos',{'action':'routine_create','owner':'agent','body':{'kind':'steward'}}),('steward_jobs',{'action':'anything'})])
def test_every_former_steward_refusal_is_now_allowed(name,args):
    StewardTools().check(name,args)


def test_no_dispatch_or_read_fences_remain():
    policy=StewardTools()
    policy.check('talos',{'action':'dispatch'})
    policy.completed('talos',{'action':'dispatch'},'{}')
    for args in [{'action':'delegation','id':'one'},{'action':'delegation','id':'one'},{'action':'dispatch'},{'action':'live'},{'action':'live'}]:
        policy.check('talos',args)
    policy.check('hermes_conversation',{'action':'read','id':'c','limit':10})
    policy.check('hermes_conversation',{'action':'read','id':'c','limit':20})
    def unavailable(*args): raise OSError('lost response')
    with pytest.raises(OSError): policy.run('talos',{'action':'dispatch'},unavailable)
    assert policy.run('talos',{'action':'dispatch'},lambda *_:'dispatched')=='dispatched'  # an uncertain error closes nothing


def test_review_actions_stay_gateway_bound_in_ordinary_turns():
    for action in ('review','review_content','review_ack'):
        with pytest.raises(ValueError,match='review'): StewardTools().check('talos',{'action':action})


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


WATCH_SAVE={'action':'watch_save','conversation_id':'regwatch','criteria':'Done means the claim is filed and the receipt number is quoted.',
            'source':{'conversation_id':'talos','message_id':'m1'},'active':True}


def test_watch_mode_actions_are_steward_actions_in_ordinary_turns():
    policy=StewardTools()
    policy.check('talos',{'action':'watches'})
    policy.check('talos',{'action':'watches'})                                                      # reads repeat freely since 2026-10-28
    policy.check('talos',{'action':'watches','conversation_id':'regwatch'})
    policy.check('talos',WATCH_SAVE)
    policy.check('talos',{**WATCH_SAVE,'active':False})                                             # a write: start, then stop, in one turn
    policy.completed('talos',{'action':'dispatch'},'{}')
    policy.check('talos',WATCH_SAVE)                                                                # a dispatch no longer ends the turn's actions


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


def test_steward_files_any_routine_it_wants():
    from agent.steward_policy import StewardTools
    reminder = {'action': 'routine_create', 'project_id': 'reg-watch', 'id': 'regwatch-email-warmup', 'owner': 'human',
                'body': {'title': 'Daily RegWatch email warmup', 'estimated_minutes': 5, 'missed_run_policy': 'retain',
                         'recurrence': {'kind': 'daily', 'at': '09:00', 'timezone': 'America/Los_Angeles'}}}
    assert StewardTools().run('talos', reminder, lambda name, args: 'ok') == 'ok'
    for other in ({**reminder, 'owner': 'agent'}, {**reminder, 'body': {**reminder['body'], 'kind': 'steward'}},
                  {**reminder, 'body': {**reminder['body'], 'contract': {}}}):
        assert StewardTools().run('talos', other, lambda name, args: 'ok') == 'ok'  # Talos validates bodies itself
    review = {'id': 'final_'+'a'*64, 'content_hash': 'b'*64, 'queue_id': 'jim_final_'+'a'*64, 'review_run_id': 'run_1',
              'source': {'conversation_id': 'talos', 'message_id': 'run_1'}}
    with pytest.raises(ValueError):  # review mode admits only the review actions
        StewardTools(review=review).run('talos', reminder, lambda name, args: 'ok')


def test_steward_jobs_is_a_steward_tool_in_ordinary_turns_only():
    policy = StewardTools()
    assert policy.run('steward_jobs', {'action': 'jobs'}, lambda name, args: 'ok') == 'ok'
    policy.check('steward_jobs', {'action': 'jobs'})   # reads repeat freely
    assert policy.run('steward_jobs', {'action': 'job_save', 'id': 'j', 'prompt': 'p', 'reason': 'r'}, lambda name, args: 'ok') == 'ok'
    policy.check('steward_jobs', {'action': 'job_delete'})  # a job is not a project; the tool itself has no such action
    review = {'id': 'final_'+'a'*64, 'content_hash': 'b'*64, 'queue_id': 'jim_final_'+'a'*64, 'review_run_id': 'run_1',
              'source': {'conversation_id': 'talos', 'message_id': 'run_1'}}
    with pytest.raises(ValueError): StewardTools(review=review).check('steward_jobs', {'action': 'jobs'})
    assert selected_tools({'HERMES_STEWARD_REVIEW': '{}'}) == ('talos',)


# ---- Context reset and the Talos state brief (2026-10-28) -------------------

def test_reset_request_is_a_workspace_file_consumed_once(tmp_path, monkeypatch):
    from agent import steward_policy
    monkeypatch.setattr(steward_policy.Path, 'home', classmethod(lambda cls: tmp_path))
    assert steward_policy.pending_reset() is None
    record = steward_policy.request_reset('Will: reset your context', {'conversation_id': 'talos', 'message_id': 'turn_x'})
    assert record['reason'] == 'Will: reset your context' and record['by'] == 'jim' and record['requested_at'] > 0
    assert steward_policy.pending_reset() == record
    steward_policy.request_reset('again')                      # two asks are one reset
    assert steward_policy.pending_reset()['reason'] == 'again'
    steward_policy.clear_reset(); steward_policy.clear_reset()  # clearing twice is fine
    assert steward_policy.pending_reset() is None
    steward_policy.reset_flag_path().write_text('{not json')
    assert steward_policy.pending_reset()['reason'] == 'unreadable request'  # still a reset


def test_reset_applies_only_to_the_ordinary_talos_conversation():
    from agent.steward_policy import reset_applies
    ordinary = SimpleNamespace(session_id='hermes-chat-talos', session_project='talos-steward', session_cwd='', _steward_review=None)
    review = SimpleNamespace(session_id='hermes-steward-review', session_project='talos-steward', session_cwd='', _steward_review={'id': 'x'})
    other = SimpleNamespace(session_id='hermes-chat-other', session_project='reg-watch', session_cwd='', _steward_review=None)
    assert reset_applies(ordinary) and not reset_applies(review) and not reset_applies(other)


def test_brief_note_carries_the_talos_brief_or_a_notice_and_never_raises(monkeypatch):
    import tools.talos_tool as talos
    from agent.steward_policy import brief_note
    monkeypatch.setattr(talos, '_call', lambda path, payload=None, timeout=20: {'text': 'Talos brief at now.\nOpen work: 1 active.', 'at': 1})
    note = brief_note()
    assert note.startswith('Talos state brief (a snapshot of the ledger') and 'Open work: 1 active.' in note
    def down(path, payload=None, timeout=20): raise ValueError('Talos is unreachable: refused')
    monkeypatch.setattr(talos, '_call', down)
    assert brief_note().startswith('Talos state brief unavailable (ValueError: Talos is unreachable')
    monkeypatch.setattr(talos, '_call', lambda path, payload=None, timeout=20: {'text': ''})
    assert 'unavailable' in brief_note()


def test_the_brief_never_changes_the_contract_digest():
    from agent.codex_runtime import _steward_contract_due
    agent = SimpleNamespace(context_compressor=None)
    assert _steward_contract_due(agent, 'thread-a', 'instructions v1')
    assert not _steward_contract_due(agent, 'thread-a', 'instructions v1')  # a new brief rides as its own note; the key is the contract alone


def test_steward_tools_server_never_imports_from_its_working_directory():
    from agent.steward_policy import mcp_server

    assert mcp_server()["args"][:2] == ["-P", "-m"]

