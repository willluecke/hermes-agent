"""Tests for gateway/wake.py — background wake delivery.

Two strategies:
* push-capable adapters keep the synthetic MessageEvent / handle_message path;
* the stateless API server (supports_async_delivery=False) self-POSTs
  /v1/runs with the RAW session id (header and body), so the wake turn
  resumes the REAL session instead of a parallel invisible one keyed by
  build_session_key(). The runs API admits the turn and answers at once, so
  a slow wake turn cannot outlive the call and be re-posted as a duplicate;
  an Idempotency-Key makes any retry replay the same run.
"""

import asyncio

import pytest

from gateway.config import Platform
from gateway.session import SessionSource
from gateway.wake import deliver_wake, adapter_supports_push


class PushAdapter:
    """Default adapter shape — no supports_async_delivery attribute."""

    def __init__(self):
        self.handled = []

    async def handle_message(self, event):
        self.handled.append(event)


class ApiServerLikeAdapter:
    supports_async_delivery = False

    def __init__(self, host="0.0.0.0", port=0, key="test-key", model="hermes"):
        self._host = host
        self._port = port
        self._api_key = key
        self._model_name = model

    async def handle_message(self, event):  # pragma: no cover — must NOT be hit
        raise AssertionError("non-push adapter must not receive handle_message wakes")


def _source():
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        chat_type="group",
    )


def test_adapter_supports_push_default_true():
    assert adapter_supports_push(PushAdapter()) is True
    assert adapter_supports_push(ApiServerLikeAdapter()) is False


async def _serve(handler):
    """Spin an in-process aiohttp server on an ephemeral loopback port."""
    from aiohttp import web

    app = web.Application()
    app.router.add_post("/v1/runs", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, port


def test_deliver_wake_non_push_self_posts_raw_session_id(monkeypatch):
    """The self-post is a detached run on the RAW session: bearer auth, the
    session id in the header and the body, the wake text as ``input``, and
    the caller's Idempotency-Key. It returns on admission (HTTP 202)."""
    from aiohttp import web

    seen = {}

    async def handler(request):
        seen["session_id"] = request.headers.get("X-Hermes-Session-Id")
        seen["auth"] = request.headers.get("Authorization")
        seen["idempotency"] = request.headers.get("Idempotency-Key")
        seen["body"] = await request.json()
        return web.json_response({"run_id": "run_1", "status": "started"}, status=202)

    async def run():
        runner, port = await _serve(handler)
        try:
            adapter = ApiServerLikeAdapter(host="0.0.0.0", port=port, key="sekrit")
            await deliver_wake(
                adapter,
                text="task done — wake",
                session_id="raw-sid-42",
                idempotency_key="kanban-wake:b:t_1:raw-sid-42:7",
            )
        finally:
            await runner.cleanup()

    asyncio.run(run())
    assert seen["session_id"] == "raw-sid-42"
    assert seen["auth"] == "Bearer sekrit"
    assert seen["idempotency"] == "kanban-wake:b:t_1:raw-sid-42:7"
    assert seen["body"]["input"] == "task done — wake"
    assert seen["body"]["session_id"] == "raw-sid-42"
    assert "project" not in seen["body"]


def test_deliver_wake_omits_idempotency_header_when_no_key_given():
    from aiohttp import web

    seen = {}

    async def handler(request):
        seen["idempotency"] = request.headers.get("Idempotency-Key")
        return web.json_response({"run_id": "run_2", "status": "started"}, status=202)

    async def run():
        runner, port = await _serve(handler)
        try:
            await deliver_wake(ApiServerLikeAdapter(port=port), text="x", session_id="sid")
        finally:
            await runner.cleanup()

    asyncio.run(run())
    assert seen["idempotency"] is None


def test_deliver_wake_draining_raises_at_once_without_retry(monkeypatch):
    """A 503 gateway_draining answer is the gateway restarting, not a failed
    destination: no retries here, and a typed error the notifier can treat
    as "keep the event for the next gateway"."""
    from aiohttp import web

    import gateway.wake as wake_mod

    monkeypatch.setattr(wake_mod, "_RETRY_DELAYS_SECONDS", (0.01, 0.01, 0.01))
    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        return web.json_response(
            {"error": {"message": "Gateway is draining existing work; retry shortly.",
                       "code": "gateway_draining"}},
            status=503,
        )

    async def run():
        runner, port = await _serve(handler)
        try:
            with pytest.raises(wake_mod.WakeDrainingError):
                await deliver_wake(ApiServerLikeAdapter(port=port), text="x", session_id="sid")
        finally:
            await runner.cleanup()

    asyncio.run(run())
    assert calls["n"] == 1


def test_deliver_wake_retries_429_then_succeeds(monkeypatch):
    """HTTP 429 (max_concurrent_runs cap) is transient — retried with backoff."""
    from aiohttp import web

    import gateway.wake as wake_mod

    monkeypatch.setattr(wake_mod, "_RETRY_DELAYS_SECONDS", (0.01, 0.01, 0.01))
    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return web.json_response({"error": "busy"}, status=429)
        return web.json_response({"choices": []})

    async def run():
        runner, port = await _serve(handler)
        try:
            adapter = ApiServerLikeAdapter(port=port)
            await deliver_wake(adapter, text="x", session_id="sid")
        finally:
            await runner.cleanup()

    asyncio.run(run())
    assert calls["n"] == 2


