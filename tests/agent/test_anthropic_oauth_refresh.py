from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from agent.anthropic_adapter import refresh_anthropic_oauth_pure


class _Response:
    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def read(self) -> bytes:
        return self._body


def test_refresh_uses_claude_codes_json_wire(monkeypatch):
    captured = {}

    def fake_urlopen(request, *, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return _Response(
            {
                "access_token": "new-access",
                "refresh_token": "new-refresh",
                "expires_in": 3600,
            }
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    refreshed = refresh_anthropic_oauth_pure("old-refresh")

    request = captured["request"]
    assert request.full_url == "https://platform.claude.com/v1/oauth/token"
    assert request.get_header("Content-type") == "application/json"
    assert json.loads(request.data.decode()) == {
        "grant_type": "refresh_token",
        "refresh_token": "old-refresh",
        "client_id": "9d1c250a-e61b-44d9-88ed-5944d1962f5e",
    }
    assert captured["timeout"] == 10
    assert refreshed["access_token"] == "new-access"
    assert refreshed["refresh_token"] == "new-refresh"


def test_primary_oauth_error_is_not_masked_by_dead_fallback(monkeypatch):
    calls = []

    def fake_urlopen(request, *, timeout):
        calls.append(request.full_url)
        raise urllib.error.HTTPError(
            request.full_url,
            400,
            "invalid_grant",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(urllib.error.HTTPError) as raised:
        refresh_anthropic_oauth_pure("spent-refresh")

    assert raised.value.code == 400
    assert calls == ["https://platform.claude.com/v1/oauth/token"]
