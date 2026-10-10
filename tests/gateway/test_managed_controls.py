import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock, patch
import pytest
from gateway.managed_controls import authorized, start_allowed, steer_allowed


@pytest.fixture
def managed(tmp_path,monkeypatch):
    monkeypatch.setenv('HERMES_SYNC_DATA_DIR',str(tmp_path))
    monkeypatch.setenv('HERMES_MANAGED_CONTROL_FILE',str(tmp_path/'keys.json'))
    (tmp_path/'keys.json').write_text(json.dumps({'controller':'c'*40,'operator':'o'*40}))
    with sqlite3.connect(tmp_path/'sync.db') as db:
        db.execute('CREATE TABLE conversation_goals(conversation_id TEXT,data TEXT)')
        db.execute('CREATE TABLE goal_turns(conversation_id TEXT,run_id TEXT,state TEXT,data TEXT)')
        db.execute('INSERT INTO conversation_goals VALUES(?,?)',('delegate',json.dumps({'status':'active','delegation_id':'d'})))
        db.execute('INSERT INTO goal_turns VALUES(?,?,?,?)',('delegate','run_delegate','running',json.dumps({'delegation_id':'d'})))
    return tmp_path


def test_plain_api_key_cannot_steer_or_add_turns_to_managed_work(managed):
    headers={'Authorization':'Bearer ordinary-key'}
    assert not steer_allowed(headers,'run_delegate')
    assert not start_allowed(headers,{'session_id':'hermes-chat-delegate'})
    assert not start_allowed(headers,{'session_id':'hermes-chat-delegate','native_goal':False})
    assert start_allowed(headers,{'session_id':'hermes-chat-unmanaged'})
    assert steer_allowed(headers,'run_other')


def test_controller_is_launch_only_and_operator_can_steer(managed):
    controller={'X-Hermes-Managed-Controller':'c'*40}
    operator={'X-Hermes-Managed-Operator':'o'*40}
    assert start_allowed(controller,{'session_id':'hermes-chat-delegate','native_goal':False})
    assert not start_allowed(controller,{'session_id':'hermes-chat-delegate'})
    assert not steer_allowed(controller,'run_delegate')
    assert steer_allowed(operator,'run_delegate')
    assert start_allowed(operator,{'session_id':'hermes-chat-delegate'})
    assert not steer_allowed({},'not-registered-yet',managed=True)


def test_cleared_but_running_remains_owned_and_unreadable_capabilities_refuse(managed):
    with sqlite3.connect(managed/'sync.db') as db:
        db.execute("UPDATE conversation_goals SET data=?",(json.dumps({'status':'cleared','delegation_id':'d'}),))
    assert not start_allowed({}, {'session_id':'hermes-chat-delegate'})
    (managed/'keys.json').write_text('broken')
    assert not authorized({'X-Hermes-Managed-Operator':'o'*40},'operator')


@pytest.mark.asyncio
async def test_gateway_handlers_reject_before_creating_or_steering_agent(managed):
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.config import PlatformConfig
    adapter=APIServerAdapter(PlatformConfig(enabled=True,extra={}))
    request=SimpleNamespace(headers={},json=AsyncMock(return_value={'session_id':'hermes-chat-delegate','input':'extra'}),match_info={'run_id':'run_delegate'})
    with patch.object(adapter,'_persist_gateway_active_work'),patch.object(adapter,'_create_agent') as create:
        response=await adapter._handle_runs(request)
        assert response.status==403
        create.assert_not_called()
    agent=MagicMock()
    adapter._run_statuses['run_delegate']={'status':'running'}
    adapter._active_run_agents['run_delegate']=agent
    response=await adapter._handle_steer_run(request)
    assert response.status==403
    agent.steer.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('via',['path','previous_response'])
async def test_managed_session_identity_cannot_be_hidden_in_an_alias(managed,via):
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.config import PlatformConfig
    adapter=APIServerAdapter(PlatformConfig(enabled=True,extra={}))
    adapter._response_store={'previous':{'session_id':'hermes-chat-delegate'}}
    body={'input':'extra'}
    match={}
    if via=='path': match['session_id']='hermes-chat-delegate'
    else: body['previous_response_id']='previous'
    request=SimpleNamespace(headers={},json=AsyncMock(return_value=body),match_info=match)
    with patch.object(adapter,'_persist_gateway_active_work'),patch.object(adapter,'_create_agent') as create:
        response=await adapter._handle_runs(request)
        assert response.status==403
        create.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('existing',[None,{'run_id':'already-accepted'},'unavailable'])
async def test_rejected_controller_cannot_strand_or_release_an_uncertain_start(managed,existing):
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.config import PlatformConfig
    adapter=APIServerAdapter(PlatformConfig(enabled=True,extra={}))
    lookup=MagicMock(return_value=existing)
    if existing=='unavailable': lookup.side_effect=OSError('unavailable')
    adapter._response_store=SimpleNamespace(get_run_idempotency=lookup)
    request=SimpleNamespace(headers={'Idempotency-Key':'goal-turn'},json=AsyncMock(return_value={'session_id':'hermes-chat-delegate','native_goal':False}),match_info={})
    with patch.object(adapter,'_persist_gateway_active_work'):
        response=await adapter._handle_runs(request)
    assert response.status==403
    assert (response.headers.get('Hermes-Run-Admission')=='rejected') is (existing is None)
    lookup.assert_called_once_with('api_server.runs:default','goal-turn')
