"""Every-turn final review on the steward lanes (Talos docs/turn-final-reviews.md).

The gateway binds {id, content_hash, queue_id, review_run_id, source} to a run;
the scoped MCP server enforces review mode: only `talos review`,
`review_content` and `review_ack` for the assigned final, each at most once,
with provenance and digests injected from the binding. Talos and hermes-sync
are fake loopback servers here; no live store or provider is touched.
"""
from __future__ import annotations

import hashlib
import json
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agent import steward_policy
from agent.steward_policy import (
    REVIEW_ENV, StewardTools, bind_review, mcp_server, review_context, review_from_env, review_request,
)
from agent.transports.hermes_tools_mcp_server import selected_tools
from tools import talos_tool as talos_mod
from tools.talos_tool import REVIEW_CONTENT_LIMIT, REVIEW_SCHEMA, review_tool, talos_tool

ID = 'final_' + 'a' * 64
HASH = 'b' * 64
RUN = 'run_' + 'c' * 32
REQUEST = {'id': ID, 'content_hash': HASH, 'queue_id': 'jim_' + ID}
CONTEXT = {**REQUEST, 'review_run_id': RUN, 'source': {'conversation_id': 'talos', 'message_id': RUN}}
TALOS_TOKEN = 'talos-operator-token-' + 't' * 24
SYNC_KEY = 'sync-service-key-' + 's' * 24
COMPLETE_PATH = '/conversations/c1/delegation-finals/g1/1?run_id=run_worker&content=complete'
WATCH_COMPLETE_PATH = '/conversations/regwatch/finals/run_will?content=complete'
CRITERIA = 'Done means the claim is filed and the receipt number is quoted.'


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def record(text='The final answer: the header fits one row.', *, complete=True, availability='present',
           terminal='completed', reference=None, delivered=True, **changes):
    raw = text.encode()
    final = {'delegation_id': 'd1', 'conversation_id': 'c1', 'goal_id': 'g1', 'ordinal': 1, 'fence': 'turn_1',
             'run_id': 'run_worker', 'terminal': terminal, 'availability': availability, 'text': text,
             'complete': complete, 'sha256': sha(raw), 'bytes': len(raw), 'reference': reference,
             'reason': None if availability in {'present', 'blank'} else 'provider returned no final'}
    if reference is not None:
        final.update(sha256=reference['sha256'], bytes=reference['bytes'])
    row = {'id': ID, 'delegation_id': 'd1', 'goal_id': 'g1', 'ordinal': 1, 'run_id': 'run_worker',
           'state': 'delivered' if delivered else 'ready',
           'body': {'final': final, 'context': {
               'project_id': 'p', 'direction_id': 'dir', 'goal': 'Fit the header in one row',
               'scope': {'allowed_paths': ['src'], 'access': 'write'},
               'authorization': {'source': {'conversation_id': 'talos', 'message_id': 'm1'}, 'scope': 'header'},
               'direction_at_dispatch': None, 'runtime': {'model': 'claude-code::claude-opus-5-5'}, 'max_turns': 3}},
           'content_hash': HASH, 'delivery': REQUEST if delivered else None, 'acknowledgement': None,
           'attempts': 1, 'next_at': 0, 'last_error': None, 'created': 1.0, 'updated': 2.0}
    row.update(changes)
    return row


def bounded_record(complete_text: str, excerpt='The final answer:'):
    raw = complete_text.encode()
    return record(excerpt, complete=False, reference={'path': COMPLETE_PATH, 'sha256': sha(raw), 'bytes': len(raw)})


def watch_record(text='Filed the claim; receipt 4471 is quoted below.', *, complete=True, reference=None):
    """A watched final (watch mode): a turn Will ran himself in a thread he asked
    Jim to watch. No delegation, goal, ordinal or fence; the context is the
    watch and Will's criteria. Everything else is the same review record."""
    raw = text.encode()
    final = {'delegation_id': None, 'conversation_id': 'regwatch', 'goal_id': None, 'ordinal': None, 'fence': None,
             'run_id': 'run_will', 'terminal': 'completed', 'availability': 'present', 'text': text,
             'complete': complete, 'sha256': sha(raw), 'bytes': len(raw), 'reference': reference, 'reason': None}
    if reference is not None:
        final.update(sha256=reference['sha256'], bytes=reference['bytes'])
    return {'id': ID, 'delegation_id': None, 'goal_id': None, 'ordinal': None, 'watch_id': 'watch_regwatch_1',
            'run_id': 'run_will', 'state': 'delivered',
            'body': {'final': final, 'context': {'project_id': 'regwatch', 'criteria': CRITERIA, 'watch_id': 'watch_regwatch_1',
                                                 'conversation_id': 'regwatch', 'started_by': {'conversation_id': 'talos', 'message_id': 'm1'}}},
            'content_hash': HASH, 'delivery': REQUEST, 'acknowledgement': None,
            'attempts': 1, 'next_at': 0, 'last_error': None, 'created': 1.0, 'updated': 2.0}


class FakeService:
    """A scripted loopback HTTP service recording every request it serves."""

    def __init__(self):
        self.requests = []
        self.routes = {}      # (method, path) -> (status, body bytes|dict) or callable(handler) -> same
        service = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _serve(self):
                length = int(self.headers.get('Content-Length') or 0)
                body = self.rfile.read(length) if length else b''
                service.requests.append({'method': self.command, 'path': self.path,
                                         'authorization': self.headers.get('Authorization'),
                                         'body': json.loads(body) if body else None})
                route = service.routes.get((self.command, self.path))
                if route is None:
                    status, payload = 404, {'error': 'Not found'}
                else:
                    status, payload = route(self) if callable(route) else route
                if status == 'drop':
                    self.close_connection = True
                    return
                raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/octet-stream' if isinstance(payload, bytes) else 'application/json')
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            do_GET = _serve
            do_POST = _serve

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.url = f'http://127.0.0.1:{self.server.server_address[1]}'
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def services(tmp_path, monkeypatch):
    talos, sync = FakeService(), FakeService()
    token = tmp_path / 'dashboard.token'
    token.write_text(TALOS_TOKEN + '\n')
    monkeypatch.setenv('HERMES_TALOS_URL', talos.url)
    monkeypatch.setenv('HERMES_TALOS_TOKEN_FILE', str(token))
    monkeypatch.setenv('HERMES_SYNC_URL', sync.url)
    monkeypatch.setenv('HERMES_SYNC_KEY', SYNC_KEY)
    yield talos, sync
    talos.close()
    sync.close()


def serve_record(talos, row, ack_status=200):
    talos.routes[('GET', '/api/delegation-reviews/' + ID)] = (200, row)

    def ack(handler):
        payload = talos.requests[-1]['body']
        if ack_status != 200:
            return ack_status, {'error': 'scripted refusal'}
        return 200, {**row, 'state': 'reviewed', 'acknowledgement': payload}
    talos.routes[('POST', '/api/delegation-reviews/ack')] = ack


def run(policy, args):
    """Dispatch through the policy exactly as the MCP server does; the generic
    handler must never be reached in review mode."""
    def generic(name, args):
        raise AssertionError('review mode must not use the generic dispatcher')
    return json.loads(policy.run('talos', args, generic))


def assert_no_credentials(*texts):
    for text in texts:
        assert TALOS_TOKEN not in text and SYNC_KEY not in text


# ---- the binding ----------------------------------------------------------

@pytest.mark.parametrize('bad', [
    None, 'final_x', [], {}, {**REQUEST, 'extra': 1}, {'id': ID, 'content_hash': HASH},
    {**REQUEST, 'id': 'final_' + 'A' * 64}, {**REQUEST, 'id': 'review_' + 'a' * 64}, {**REQUEST, 'id': 'final_' + 'a' * 63},
    {**REQUEST, 'content_hash': 'B' * 64}, {**REQUEST, 'content_hash': HASH + 'b'}, {**REQUEST, 'content_hash': 7},
    {**REQUEST, 'queue_id': ID}, {**REQUEST, 'queue_id': 'jim_final_' + 'd' * 64}, {**REQUEST, 'queue_id': None},
])
def test_review_request_shape_is_strict(bad):
    with pytest.raises(ValueError):
        review_request(bad)


def test_review_request_normalizes_to_exactly_three_fields():
    assert review_request(dict(REQUEST)) == REQUEST


def test_bind_review_binds_the_gateway_run_id_as_run_and_message_provenance():
    assert bind_review(REQUEST, RUN) == CONTEXT
    for run_id in ('', None, 'run with space', 'r' * 129):
        with pytest.raises(ValueError):
            bind_review(REQUEST, run_id)


@pytest.mark.parametrize('bad', [
    {**CONTEXT, 'extra': True}, {k: v for k, v in CONTEXT.items() if k != 'source'},
    {**CONTEXT, 'source': {'conversation_id': 'delegate', 'message_id': RUN}},
    {**CONTEXT, 'source': {'conversation_id': 'talos', 'message_id': 'run_other'}},
    {**CONTEXT, 'source': {'conversation_id': 'talos', 'message_id': RUN, 'role': 'user'}},
    {**CONTEXT, 'review_run_id': 'run with space'}, {**CONTEXT, 'review_run_id': None}, 'text',
])
def test_review_context_is_strict(bad):
    with pytest.raises(ValueError):
        review_context(bad)


def test_mcp_server_carries_the_binding_only_for_reviews():
    ordinary = mcp_server()
    assert REVIEW_ENV not in ordinary['env']
    assert ordinary['env']['HERMES_TOOLS_MCP_ONLY'] == 'talos,hermes_conversation,steward_jobs'
    review = mcp_server(review=CONTEXT)
    assert review['command'] == ordinary['command'] and review['args'] == ordinary['args']
    assert review['env']['HERMES_STEWARD'] == '1'
    assert review['env']['HERMES_TOOLS_MCP_ONLY'] == 'talos'
    assert review['env'][REVIEW_ENV] == json.dumps(CONTEXT, sort_keys=True, separators=(',', ':'))
    assert review_from_env(review['env']) == CONTEXT
    assert review_from_env(ordinary['env']) is None
    assert selected_tools(review['env']) == ('talos',)
    assert selected_tools(ordinary['env']) == ('talos', 'hermes_conversation', 'steward_jobs')
    with pytest.raises(ValueError):
        mcp_server(review={**REQUEST, 'review_run_id': RUN})
    for raw in ('{', json.dumps({**CONTEXT, 'source': {'conversation_id': 'delegate', 'message_id': RUN}})):
        with pytest.raises(ValueError):
            review_from_env({REVIEW_ENV: raw})


# ---- the MCP server in review mode ----------------------------------------

def _fake_mcp_modules(monkeypatch, added, descriptions, handlers):
    class FakeMCPServer:
        def __init__(self, *args, **kwargs):
            pass

        def add_tool(self, fn, name=None, description=None, annotations=None, **kwargs):
            added.append(name)
            descriptions[name] = description
            handlers[name] = fn

    def never_dispatch(*args, **kwargs):
        raise AssertionError('review mode must not reach the generic dispatcher')

    defs = [{'type': 'function', 'function': {'name': name, 'description': 'operator ' + name,
                                              'parameters': {'type': 'object', 'properties': {}}}}
            for name in ('talos', 'hermes_conversation', 'web_search')]
    fake_mcp = types.ModuleType('mcp')
    fake_server = types.ModuleType('mcp.server')
    fake_server.MCPServer = FakeMCPServer
    fake_types = types.ModuleType('mcp.types')
    fake_types.ToolAnnotations = lambda **kwargs: None
    fake_mcp.server, fake_mcp.types = fake_server, fake_types
    fake_model_tools = types.ModuleType('model_tools')
    fake_model_tools.get_tool_definitions = lambda quiet_mode=True: defs
    fake_model_tools.handle_function_call = never_dispatch
    for name, module in (('mcp', fake_mcp), ('mcp.server', fake_server), ('mcp.types', fake_types), ('model_tools', fake_model_tools)):
        monkeypatch.setitem(sys.modules, name, module)


def test_build_server_in_review_mode_exposes_one_bound_tool(monkeypatch, services):
    import agent.transports.hermes_tools_mcp_server as m
    added, descriptions, handlers = [], {}, {}
    _fake_mcp_modules(monkeypatch, added, descriptions, handlers)
    for key, value in mcp_server(review=CONTEXT)['env'].items():
        monkeypatch.setenv(key, value)
    m._build_server()
    assert added == ['talos']
    assert descriptions['talos'] == REVIEW_SCHEMA['description']
    refused = json.loads(handlers['talos'](action='state'))
    assert refused['tool'] == 'talos' and 'review' in refused['error']
    refused = json.loads(handlers['talos'](action='dispatch', direction_id='dir'))
    assert 'review' in refused['error']


def test_build_server_fails_closed_on_a_malformed_binding(monkeypatch):
    import agent.transports.hermes_tools_mcp_server as m
    _fake_mcp_modules(monkeypatch, [], {}, {})
    monkeypatch.setenv('HERMES_STEWARD', '1')
    monkeypatch.setenv(REVIEW_ENV, json.dumps({**CONTEXT, 'source': {'conversation_id': 'delegate', 'message_id': RUN}}))
    with pytest.raises(ValueError):
        m._build_server()
    assert m.main([]) == 2


# ---- the review-mode fence -------------------------------------------------

DENIED = [
    ('hermes_conversation', {'action': 'list'}), ('hermes_conversation', {'action': 'read', 'id': 'c1'}),
    ('hermes_conversation', {'action': 'goal', 'id': 'c1'}), ('hermes_conversation', {'action': 'goal', 'id': 'c1', 'verb': 'pause'}),
    ('hermes_conversation', {'action': 'set_runtime', 'id': 'c1', 'model': 'x'}), ('hermes_conversation', {'action': 'send', 'id': 'c1', 'text': 'hi'}),
    ('talos', {'action': 'state'}), ('talos', {'action': 'live'}), ('talos', {'action': 'sessions'}),
    ('talos', {'action': 'delegation', 'id': 'd1'}), ('talos', {'action': 'dispatch', 'direction_id': 'dir'}),
    ('talos', {'action': 'delegation_control', 'id': 'd1', 'verb': 'pause'}), ('talos', {'action': 'hold', 'verb': 'pause'}),
    ('talos', {'action': 'delegation_control', 'id': 'd1', 'verb': 'resume', 'max_turns': 50}),   # cap change
    ('hermes_conversation', {'action': 'goal', 'id': 'c1', 'verb': 'steer', 'text': 'also fix the footer'}),  # steer
    ('hermes_conversation', {'action': 'set_runtime', 'id': 'c1', 'model': 'gpt-6.1-sol', 'effort': 'xhigh'}),  # model change
    ('talos', {'action': 'mandate_save', 'id': 'p'}), ('talos', {'action': 'direction_save', 'id': 'dir'}),
    ('talos', {'action': 'interview_save', 'project_id': 'p'}), ('talos', {'action': 'task', 'id': 't'}),
    ('talos', {'action': 'admit', 'project_id': 'p'}), ('talos', {'action': 'control', 'id': 't'}),
    ('terminal', {'command': 'ls'}), ('delegate_task', {'goal': 'x'}), ('read_file', {'path': '/etc/passwd'}),
    ('web_search', {'query': 'x'}), ('talos', {'action': 'review', 'id': 'final_' + 'f' * 64}),
    ('talos', {'action': 'review_content', 'id': ID}), ('talos', {'action': 'review_ack', 'id': ID, 'assessment': 'aligned'}),
    ('talos', {'action': 'watches'}), ('talos', {'action': 'watches', 'conversation_id': 'regwatch'}),   # watch mode: reads
    ('talos', {'action': 'watch_save', 'conversation_id': 'regwatch', 'criteria': CRITERIA,             # and writes stay outside
               'source': {'conversation_id': 'talos', 'message_id': 'm1'}, 'active': True}),
    ('talos', {'action': 'watch_save', 'conversation_id': 'regwatch', 'criteria': CRITERIA,
               'source': {'conversation_id': 'talos', 'message_id': 'm1'}, 'active': False}),
]


@pytest.mark.parametrize('name,args', DENIED)
def test_review_mode_denies_everything_but_the_assigned_review_actions(name, args):
    policy = StewardTools(review=CONTEXT)
    with pytest.raises(ValueError):
        policy.check(name, args)
    with pytest.raises(ValueError):
        policy.run(name, args, lambda *_: pytest.fail('a denied action must not run'))


def test_ordinary_steward_turns_keep_their_surface_and_cannot_review():
    policy = StewardTools()
    policy.check('talos', {'action': 'state'})
    policy.check('hermes_conversation', {'action': 'read', 'id': 'c1'})
    for action in ('review', 'review_content', 'review_ack'):
        with pytest.raises(ValueError):
            policy.check('talos', {'action': action, 'id': ID})
        assert json.loads(talos_tool(action, id=ID))['ok'] is False


# ---- review --------------------------------------------------------------

def test_review_reads_the_assigned_record_once_with_the_talos_bearer(services):
    talos, sync = services
    serve_record(talos, record())
    policy = StewardTools(review=CONTEXT)
    result = run(policy, {'action': 'review', 'id': ID})
    assert result['ok'] is True
    assert result['result']['body']['final']['text'] == 'The final answer: the header fits one row.'
    assert result['complete_content'] == 'in_record' and result['availability'] == 'present'
    assert policy.record['id'] == ID
    assert talos.requests == [{'method': 'GET', 'path': '/api/delegation-reviews/' + ID,
                               'authorization': 'Bearer ' + TALOS_TOKEN, 'body': None}]
    assert sync.requests == []
    assert_no_credentials(json.dumps(result))
    with pytest.raises(ValueError):
        policy.check('talos', {'action': 'review', 'id': ID})
    with pytest.raises(ValueError):
        policy.check('talos', {'action': 'review'})
    assert len(talos.requests) == 1


@pytest.mark.parametrize('row', [
    record(content_hash='0' * 64), record(delivered=False), record(delivery={**REQUEST, 'queue_id': 'jim_other'}),
    {**record(), 'id': 'final_' + 'e' * 64}, record(body=None),
])
def test_review_refuses_a_record_that_is_not_the_assigned_delivered_packet(services, row):
    talos, _ = services
    serve_record(talos, row)
    policy = StewardTools(review=CONTEXT)
    result = run(policy, {'action': 'review', 'id': ID})
    assert result['ok'] is False
    assert policy.record is None and 'review' in policy.attempted
    with pytest.raises(ValueError):  # no record, so nothing can be acknowledged or re-read
        policy.check('talos', {'action': 'review_ack', 'assessment': 'insufficient'})
    with pytest.raises(ValueError):
        policy.check('talos', {'action': 'review'})
    assert_no_credentials(json.dumps(result))


def test_review_is_uncertain_only_once(services):
    talos, _ = services
    talos.routes[('GET', '/api/delegation-reviews/' + ID)] = ('drop', None)
    policy = StewardTools(review=CONTEXT)
    assert run(policy, {'action': 'review', 'id': ID})['ok'] is False
    with pytest.raises(ValueError):
        policy.check('talos', {'action': 'review', 'id': ID})


# ---- watch mode: a final of a thread Will drives himself -------------------

def test_a_watched_final_reviews_like_a_delegated_one_and_shows_wills_criteria(services):
    talos, sync = services
    complete = 'Filed the claim. ' + 'Receipt 4471 is quoted on the confirmation page. ' * 30
    raw = complete.encode()
    serve_record(talos, watch_record('Filed the claim.', complete=False,
                                     reference={'path': WATCH_COMPLETE_PATH, 'sha256': sha(raw), 'bytes': len(raw)}))
    sync.routes[('GET', WATCH_COMPLETE_PATH)] = (200, raw)
    policy = StewardTools(review=CONTEXT)
    result = run(policy, {'action': 'review', 'id': ID})
    assert result['ok'] is True and result['complete_content'] == 'read_with_review_content'
    row = result['result']
    assert row['delegation_id'] is None and row['goal_id'] is None and row['ordinal'] is None
    assert row['watch_id'] == 'watch_regwatch_1' and row['body']['final']['fence'] is None
    assert row['body']['context']['criteria'] == CRITERIA
    assert row['body']['context']['conversation_id'] == 'regwatch'
    assert 'criteria' in REVIEW_SCHEMA['description']
    content = run(policy, {'action': 'review_content', 'id': ID})
    assert content['ok'] is True and content['result'] == {'id': ID, 'sha256': sha(raw), 'bytes': len(raw), 'text': complete}
    assert sync.requests == [{'method': 'GET', 'path': WATCH_COMPLETE_PATH, 'authorization': 'Bearer ' + SYNC_KEY, 'body': None}]
    ack = run(policy, {'action': 'review_ack', 'assessment': 'aligned', 'reason': 'Filed, with the receipt number quoted.',
                       'needed_action': 'none', 'coverage': 'complete'})
    assert ack['ok'] is True and ack['result']['state'] == 'reviewed'
    assert talos.requests[-1]['body'] == {
        'id': ID, 'content_hash': HASH, 'queue_id': 'jim_' + ID,
        'source': {'conversation_id': 'talos', 'message_id': RUN}, 'review_run_id': RUN,
        'assessment': 'aligned', 'reason': 'Filed, with the receipt number quoted.', 'needed_action': 'none',
        'action_detail': None, 'coverage': 'complete', 'final_sha256': sha(raw),
    }
    assert_no_credentials(json.dumps(result), json.dumps(content), json.dumps(ack))


def test_a_watched_final_in_the_record_is_acknowledged_against_the_criteria(services):
    talos, sync = services
    serve_record(talos, watch_record('Filed the claim; the receipt number was not shown.'))
    policy = StewardTools(review=CONTEXT)
    result = run(policy, {'action': 'review', 'id': ID})
    assert result['ok'] is True and result['complete_content'] == 'in_record'
    assert result['result']['body']['context']['criteria'] == CRITERIA
    with pytest.raises(ValueError):  # complete in the record: nothing to read
        policy.check('talos', {'action': 'review_content', 'id': ID})
    ack = run(policy, {'action': 'review_ack', 'assessment': 'needs_action', 'needed_action': 'clarification',
                       'action_detail': 'The receipt number is not quoted; ask for it before calling this done.',
                       'reason': 'Filed, but the criteria require the receipt number.', 'coverage': 'complete'})
    assert ack['ok'] is True and talos.requests[-1]['body']['assessment'] == 'needs_action'
    assert sync.requests == []


# ---- review_content ------------------------------------------------------

def test_review_content_fetches_exactly_the_recorded_reference_and_verifies_it(services):
    talos, sync = services
    complete = 'The final answer: ' + 'the header fits one row. ' * 40 + 'Done, ✓.'
    serve_record(talos, bounded_record(complete))
    sync.routes[('GET', COMPLETE_PATH)] = (200, complete.encode())
    policy = StewardTools(review=CONTEXT)
    assert run(policy, {'action': 'review', 'id': ID})['complete_content'] == 'read_with_review_content'
    with pytest.raises(ValueError):  # the packet alone is not complete coverage
        review_tool_ack(policy, coverage='complete')
    result = run(policy, {'action': 'review_content', 'id': ID})
    assert result['ok'] is True
    assert result['result'] == {'id': ID, 'sha256': sha(complete.encode()), 'bytes': len(complete.encode()), 'text': complete}
    assert policy.content_verified is True
    assert sync.requests == [{'method': 'GET', 'path': COMPLETE_PATH, 'authorization': 'Bearer ' + SYNC_KEY, 'body': None}]
    with pytest.raises(ValueError):
        policy.check('talos', {'action': 'review_content', 'id': ID})
    ack = run(policy, {'action': 'review_ack', 'id': ID, 'assessment': 'aligned', 'reason': 'Complete and on scope.',
                       'needed_action': 'none', 'coverage': 'complete'})
    assert ack['ok'] is True and ack['result']['state'] == 'reviewed'
    assert talos.requests[-1]['body'] == {
        'id': ID, 'content_hash': HASH, 'queue_id': 'jim_' + ID,
        'source': {'conversation_id': 'talos', 'message_id': RUN}, 'review_run_id': RUN,
        'assessment': 'aligned', 'reason': 'Complete and on scope.', 'needed_action': 'none',
        'action_detail': None, 'coverage': 'complete', 'final_sha256': sha(complete.encode()),
    }
    assert talos.requests[-1]['authorization'] == 'Bearer ' + TALOS_TOKEN
    assert_no_credentials(json.dumps(result), json.dumps(ack))


def review_tool_ack(policy, **fields):
    """Local review_ack validation through the policy, raising on refusal."""
    args = {'action': 'review_ack', 'assessment': 'aligned', 'reason': 'ok', 'needed_action': 'none', 'coverage': 'complete', **fields}
    result = run(policy, args)
    if not result['ok']:
        raise ValueError(result['error'])
    return result


@pytest.mark.parametrize('served', ['short', 'long', 'digest', 'utf8', 'missing'])
def test_review_content_refuses_unverified_bytes_and_coverage_stays_insufficient(services, served):
    talos, sync = services
    complete = 'The final answer: ' + 'the header fits one row. ' * 20
    serve_record(talos, bounded_record(complete))
    raw = complete.encode()
    body = {'short': raw[:-1], 'long': raw + b'!', 'digest': raw[:-1] + b'?', 'utf8': raw[:-1] + b'\xff', 'missing': None}[served]
    if body is not None:
        sync.routes[('GET', COMPLETE_PATH)] = (200, body)
    policy = StewardTools(review=CONTEXT)
    run(policy, {'action': 'review', 'id': ID})
    result = run(policy, {'action': 'review_content', 'id': ID})
    assert result['ok'] is False and policy.content_verified is False
    assert 'review_content' in policy.attempted
    with pytest.raises(ValueError):  # at most one read
        policy.check('talos', {'action': 'review_content', 'id': ID})
    assert run(policy, {'action': 'review_ack', 'assessment': 'aligned', 'reason': 'ok', 'needed_action': 'none',
                        'coverage': 'complete'})['ok'] is False
    assert 'review_ack' not in policy.attempted
    assert run(policy, {'action': 'review_ack', 'assessment': 'aligned', 'reason': 'ok', 'needed_action': 'none',
                        'coverage': 'insufficient'})['ok'] is False
    ack = run(policy, {'action': 'review_ack', 'assessment': 'insufficient', 'coverage': 'insufficient', 'needed_action': 'none',
                       'reason': 'The complete content could not be verified; only the excerpt was readable.'})
    assert ack['ok'] is True
    assert talos.requests[-1]['body']['coverage'] == 'insufficient'
    assert talos.requests[-1]['body']['final_sha256'] == sha(raw)
    assert_no_credentials(json.dumps(result), json.dumps(ack))


def test_review_content_bound_is_enforced_before_any_fetch(services):
    talos, sync = services
    oversized = bounded_record('x')
    oversized['body']['final']['reference'] = {'path': COMPLETE_PATH, 'sha256': 'f' * 64, 'bytes': REVIEW_CONTENT_LIMIT + 1}
    oversized['body']['final'].update(sha256='f' * 64, bytes=REVIEW_CONTENT_LIMIT + 1)
    serve_record(talos, oversized)
    policy = StewardTools(review=CONTEXT)
    run(policy, {'action': 'review', 'id': ID})
    result = run(policy, {'action': 'review_content', 'id': ID})
    assert result['ok'] is False and str(REVIEW_CONTENT_LIMIT) in result['error']
    assert sync.requests == [] and policy.content_verified is False
    assert run(policy, {'action': 'review_ack', 'assessment': 'aligned', 'reason': 'ok', 'needed_action': 'none',
                        'coverage': 'complete'})['ok'] is False


@pytest.mark.parametrize('reference', [
    {'path': 'https://evil.example/complete', 'sha256': 'f' * 64, 'bytes': 2},
    {'path': '/conversations/c1/messages?content=complete', 'sha256': 'f' * 64, 'bytes': 2, 'extra': 1},
    {'path': '/conversations/c1/delegation-finals/g1/1?run_id=run_worker', 'sha256': 'f' * 64, 'bytes': 2},
])
def test_review_content_never_follows_an_arbitrary_or_unrecorded_address(services, reference):
    talos, sync = services
    row = bounded_record('xy')
    row['body']['final']['reference'] = reference
    serve_record(talos, row)
    policy = StewardTools(review=CONTEXT)
    run(policy, {'action': 'review', 'id': ID})
    assert run(policy, {'action': 'review_content', 'id': ID})['ok'] is False
    assert sync.requests == []


def test_review_content_is_refused_for_a_complete_final(services):
    talos, sync = services
    serve_record(talos, record())
    policy = StewardTools(review=CONTEXT)
    run(policy, {'action': 'review', 'id': ID})
    with pytest.raises(ValueError):
        policy.check('talos', {'action': 'review_content', 'id': ID})
    assert sync.requests == []


# ---- review_ack ----------------------------------------------------------

@pytest.mark.parametrize('override', [
    {'content_hash': '0' * 64}, {'queue_id': 'jim_final_' + 'd' * 64}, {'review_run_id': 'run_other'},
    {'source': {'conversation_id': 'talos', 'message_id': 'run_other'}}, {'final_sha256': '0' * 64}, {'id': 'final_' + 'd' * 64},
])
def test_review_ack_provenance_cannot_be_changed_by_the_model(services, override):
    talos, _ = services
    serve_record(talos, record())
    policy = StewardTools(review=CONTEXT)
    run(policy, {'action': 'review', 'id': ID})
    with pytest.raises(ValueError):
        policy.check('talos', {'action': 'review_ack', 'assessment': 'aligned', 'reason': 'ok', 'needed_action': 'none',
                               'coverage': 'complete', **override})
    assert len(talos.requests) == 1


def test_review_ack_tolerates_restating_the_assigned_identity(services):
    talos, _ = services
    row = record()
    serve_record(talos, row)
    policy = StewardTools(review=CONTEXT)
    run(policy, {'action': 'review', 'id': ID})
    ack = run(policy, {'action': 'review_ack', 'id': ID, 'content_hash': HASH, 'final_sha256': row['body']['final']['sha256'],
                       'assessment': 'needs_action', 'reason': 'The final reports a passing suite but lists no evidence path.',
                       'needed_action': 'clarification', 'action_detail': 'Ask the worker for the suite output location.',
                       'coverage': 'complete'})
    assert ack['ok'] is True
    sent = talos.requests[-1]['body']
    assert sent['needed_action'] == 'clarification' and sent['action_detail'].startswith('Ask the worker')
    assert sent['review_run_id'] == RUN and sent['source'] == {'conversation_id': 'talos', 'message_id': RUN}


@pytest.mark.parametrize('fields', [
    {'assessment': 'aligned', 'needed_action': 'clarification', 'action_detail': 'x'},
    {'assessment': 'drift', 'needed_action': 'none'}, {'assessment': 'needs_action', 'needed_action': 'none'},
    {'assessment': 'excellent'}, {'needed_action': 'escalate'}, {'coverage': 'partial'},
    {'reason': ''}, {'reason': 'r' * 1501}, {'reason': None}, {'action_detail': 'detail with none'},
    {'assessment': 'needs_action', 'needed_action': 'approval'}, {'assessment': 'needs_action', 'needed_action': 'approval', 'action_detail': 'd' * 1501},
    {'resume': True}, {'max_turns': 9},
])
def test_review_ack_local_validation_refuses_without_spending_the_single_attempt(services, fields):
    talos, _ = services
    serve_record(talos, record())
    policy = StewardTools(review=CONTEXT)
    run(policy, {'action': 'review', 'id': ID})
    args = {'action': 'review_ack', 'assessment': 'aligned', 'reason': 'ok', 'needed_action': 'none', 'coverage': 'complete', **fields}
    assert run(policy, args)['ok'] is False
    assert 'review_ack' not in policy.attempted and len(talos.requests) == 1
    ack = run(policy, {'action': 'review_ack', 'assessment': 'aligned', 'reason': 'ok', 'needed_action': 'none', 'coverage': 'complete'})
    assert ack['ok'] is True and len(talos.requests) == 2


def test_a_refused_ack_tells_jim_nothing_was_recorded_and_how_to_correct_it(services):
    # 2026-10-14, the first live reviews: Jim judged a research final aligned
    # but needing Will's approval, read "An aligned review has no needed
    # action" as a failed acknowledgement and ended the review unrecorded.
    talos, _ = services
    serve_record(talos, record())
    policy = StewardTools(review=CONTEXT)
    run(policy, {'action': 'review', 'id': ID})
    refused = run(policy, {'action': 'review_ack', 'assessment': 'aligned', 'needed_action': 'approval',
                           'action_detail': 'Executing the plan needs Will.', 'reason': 'Complete and in scope.',
                           'coverage': 'complete'})
    assert refused['ok'] is False
    assert 'assess needs_action' in refused['error'] and 'Nothing was recorded' in refused['error']
    assert 'call review_ack again' in refused['error'] and 'review_ack' not in policy.attempted
    assert 'when aligned work still needs' in REVIEW_SCHEMA['description']
    ack = run(policy, {'action': 'review_ack', 'assessment': 'needs_action', 'needed_action': 'approval',
                       'action_detail': 'Executing the plan needs Will.', 'reason': 'Complete and in scope.',
                       'coverage': 'complete'})
    assert ack['ok'] is True and talos.requests[-1]['body']['assessment'] == 'needs_action'


@pytest.mark.parametrize('availability,terminal', [('blank', 'completed'), ('unavailable', 'failed'), ('provider_error', 'cancelled')])
def test_missing_finals_must_be_acknowledged_insufficient(services, availability, terminal):
    talos, _ = services
    serve_record(talos, record('' if availability == 'blank' else '', availability=availability, terminal=terminal))
    policy = StewardTools(review=CONTEXT)
    assert run(policy, {'action': 'review', 'id': ID})['availability'] == availability
    assert run(policy, {'action': 'review_ack', 'assessment': 'aligned', 'reason': 'ok', 'needed_action': 'none',
                        'coverage': 'complete'})['ok'] is False
    ack = run(policy, {'action': 'review_ack', 'assessment': 'insufficient', 'needed_action': 'none', 'coverage': 'insufficient',
                       'reason': f'No assistant final: {availability}; terminal status {terminal} preserved.'})
    assert ack['ok'] is True and talos.requests[-1]['body']['assessment'] == 'insufficient'


@pytest.mark.parametrize('outcome', [500, 409, 400, 'drop'])
def test_review_ack_is_attempted_at_most_once_whatever_talos_answers(services, outcome):
    talos, _ = services
    row = record()
    serve_record(talos, row, ack_status=outcome if outcome != 'drop' else 200)
    if outcome == 'drop':
        talos.routes[('POST', '/api/delegation-reviews/ack')] = ('drop', None)
    policy = StewardTools(review=CONTEXT)
    run(policy, {'action': 'review', 'id': ID})
    result = run(policy, {'action': 'review_ack', 'assessment': 'aligned', 'reason': 'ok', 'needed_action': 'none', 'coverage': 'complete'})
    assert result['ok'] is False and 'uncertain' in result['error']
    assert 'review_ack' in policy.attempted
    for args in ({'action': 'review_ack', 'assessment': 'aligned', 'reason': 'ok', 'needed_action': 'none', 'coverage': 'complete'},
                 {'action': 'review', 'id': ID}, {'action': 'state'}):
        with pytest.raises(ValueError):
            policy.check('talos', args)
    assert [r['method'] for r in talos.requests] == ['GET', 'POST']
    assert_no_credentials(json.dumps(result))


def test_review_tool_itself_refuses_anything_else_and_hides_credentials(services):
    talos, _ = services
    serve_record(talos, record())
    policy = StewardTools(review=CONTEXT)
    for action in ('state', 'dispatch', '', 'REVIEW_ACK'):
        result = json.loads(review_tool(policy, action=action))
        assert result['ok'] is False
    assert json.loads(review_tool(policy, action='review', id=ID, extra=1))['ok'] is False
    assert talos.requests == []
    output = review_tool(policy, action='review', id=ID)
    assert json.loads(output)['ok'] is True
    assert talos.requests[-1]['authorization'] == 'Bearer ' + TALOS_TOKEN  # sent to Talos only
    assert_no_credentials(output)                                          # never returned to the model


def test_review_actions_serialize_under_concurrent_calls(services):
    from concurrent.futures import ThreadPoolExecutor
    talos, _ = services
    serve_record(talos, record())
    policy = StewardTools(review=CONTEXT)
    run(policy, {'action': 'review', 'id': ID})
    start = threading.Barrier(2)

    def ack():
        start.wait(timeout=5)
        try:
            return run(policy, {'action': 'review_ack', 'assessment': 'aligned', 'reason': 'ok', 'needed_action': 'none', 'coverage': 'complete'})['ok']
        except ValueError:
            return 'refused'
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: ack(), range(2)))
    assert sorted(map(str, results)) == ['True', 'refused']
    assert [r['method'] for r in talos.requests] == ['GET', 'POST']
