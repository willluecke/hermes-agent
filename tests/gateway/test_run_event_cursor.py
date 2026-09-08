import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, _ReplayableRunEventStream


def events(body):
    return [json.loads(line[6:]) for line in body.splitlines() if line.startswith('data: ')]


@pytest.mark.asyncio
async def test_incremental_replay_and_independent_clients():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    stream = _ReplayableRunEventStream(1100)
    adapter._run_streams['run_cursor'] = stream
    for n in range(1000):
        stream.put_nowait({'event': 'tool.output.delta', 'sequence': n % 5 + 1,
                           'tool_call_id': f'tool-{n // 5}', 'delta': str(n)})
    stream.put_nowait(None)
    app = web.Application()
    app.router.add_get('/v1/runs/{run_id}/events', adapter._handle_run_events)
    async with TestClient(TestServer(app)) as client:
        full = await (await client.get('/v1/runs/run_cursor/events')).text()
        assert len(events(full)) == 1000
        assert [e['run_seq'] for e in events(full)] == list(range(1, 1001))
        for cursor in (997, 1000, 500, 999, 0):
            body = await (await client.get(f'/v1/runs/run_cursor/events?after={cursor}')).text()
            assert events(body) == events(full)[cursor:]
            if cursor == 997:
                assert len(body) < len(full) / 100
        body = await (await client.get('/v1/runs/run_cursor/events', headers={'Last-Event-ID': '998'})).text()
        assert events(body) == events(full)[998:]
        for cursor, expected in (('-1', 400), ('1.5', 400), ('9' * 1000, 400), ('1001', 409)):
            assert (await client.get(f'/v1/runs/run_cursor/events?after={cursor}')).status == expected
        ring = _ReplayableRunEventStream(2)
        adapter._run_streams['run_gap'] = ring
        for n in range(3):
            ring.put_nowait({'event': 'message.interim', 'message_id': f'm{n}', 'text': str(n)})
        ring.put_nowait(None)
        gap_replay = events(await (await client.get('/v1/runs/run_gap/events')).text())
        assert gap_replay[0] == {
            'event': 'run.replay.gap', 'run_id': 'run_gap',
            'event_id': 'run_gap:replay-gap:1:1', 'missing_events': 1,
        }
        assert [event['run_seq'] for event in gap_replay[1:]] == [2, 3]


def test_ring_eviction_retains_sequence_and_emission_timestamp():
    stream = _ReplayableRunEventStream(2)
    original = {'event': 'tool.started', 'tool_call_id': 'stable'}
    stream.put_nowait(original)
    stream.put_nowait({'event': 'tool.completed', 'tool_call_id': 'stable'})
    first, cursor, _ = stream.read_from(0)
    stream.put_nowait({'event': 'run.completed', 'output_kind': 'final', 'output': 'done'})
    replay, _, _ = stream.read_from(0)
    assert replay[0] == first[1]
    assert cursor == 2 and stream.first_index == 1
    assert replay[1]['run_seq'] == 3
    assert replay[0]['emitted_at_ms'] > 0
    assert 'run_seq' not in original
