"""The test suite must never touch a real Claude Code credential file."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agent.anthropic_adapter import (
    _write_claude_code_credentials,
    claude_code_credentials_path,
)


def test_tests_never_see_the_live_claude_config_dir():
    assert "CLAUDE_CONFIG_DIR" not in os.environ


def test_credential_write_lands_under_a_monkeypatched_home(monkeypatch, tmp_path):
    monkeypatch.setattr("agent.anthropic_adapter.Path.home", lambda: tmp_path)
    _write_claude_code_credentials("tok", "ref", 1)
    written = json.loads((tmp_path / ".claude" / ".credentials.json").read_text(encoding="utf-8"))
    assert written["claudeAiOauth"]["accessToken"] == "tok"
    assert claude_code_credentials_path() == tmp_path / ".claude" / ".credentials.json"


def test_credential_write_outside_tmp_is_refused_under_pytest(monkeypatch):
    live = Path.home() / ".claude-live-canary-should-not-exist"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(live))
    with pytest.raises(RuntimeError, match="refusing to write"):
        _write_claude_code_credentials("tok", "ref", 1)
    assert not live.exists()
