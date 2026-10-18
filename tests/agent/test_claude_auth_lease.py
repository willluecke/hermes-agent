from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
import threading
import time
from unittest.mock import patch

import pytest

from agent.claude_auth_lease import acquire_claude_auth_lease


def _write_credentials(config_dir, *, expires_at_ms: int) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / ".credentials.json").write_text(
        json.dumps(
            {
                "outerField": "preserved",
                "claudeAiOauth": {
                    "accessToken": "old-access",
                    "refreshToken": "old-refresh",
                    "expiresAt": expires_at_ms,
                    "refreshTokenExpiresAt": 9_999_999_999_999,
                    "subscriptionType": "max",
                    "rateLimitTier": "default_claude_max_20x",
                    "scopes": ["user:inference", "user:sessions:claude_code"],
                },
            }
        ),
        encoding="utf-8",
    )


def test_valid_credential_lease_does_not_rotate_token(tmp_path, monkeypatch):
    config_dir = tmp_path / "claude-config"
    _write_credentials(
        config_dir,
        expires_at_ms=int((time.time() + 3 * 60 * 60) * 1000),
    )
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))

    with patch(
        "agent.claude_auth_lease.refresh_anthropic_oauth_pure"
    ) as refresh:
        lease = acquire_claude_auth_lease(2 * 60 * 60)

    refresh.assert_not_called()
    assert lease.available is True
    assert lease.refreshed is False
    assert lease.credentials_found is True
    assert lease.generation


def test_expiring_credential_is_refreshed_and_fully_persisted(
    tmp_path, monkeypatch
):
    config_dir = tmp_path / "claude-config"
    _write_credentials(
        config_dir,
        expires_at_ms=int((time.time() + 5 * 60) * 1000),
    )
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
    refreshed_expiry = int((time.time() + 8 * 60 * 60) * 1000)

    with patch(
        "agent.claude_auth_lease.refresh_anthropic_oauth_pure",
        return_value={
            "access_token": "new-access",
            "refresh_token": "new-refresh",
            "expires_at_ms": refreshed_expiry,
        },
    ) as refresh:
        lease = acquire_claude_auth_lease(2 * 60 * 60)

    refresh.assert_called_once_with("old-refresh")
    assert lease.available is True
    assert lease.refreshed is True
    assert lease.expires_at_ms == refreshed_expiry
    persisted = json.loads(
        (config_dir / ".credentials.json").read_text(encoding="utf-8")
    )
    oauth = persisted["claudeAiOauth"]
    assert persisted["outerField"] == "preserved"
    assert oauth["accessToken"] == "new-access"
    assert oauth["refreshToken"] == "new-refresh"
    assert oauth["refreshTokenExpiresAt"] == 9_999_999_999_999
    assert oauth["subscriptionType"] == "max"
    assert oauth["rateLimitTier"] == "default_claude_max_20x"
    assert oauth["scopes"] == ["user:inference", "user:sessions:claude_code"]
    assert (config_dir / ".credentials.lock").exists()


def test_concurrent_leases_share_one_rotating_refresh(tmp_path, monkeypatch):
    config_dir = tmp_path / "claude-config"
    _write_credentials(
        config_dir,
        expires_at_ms=int((time.time() + 5 * 60) * 1000),
    )
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
    refreshed_expiry = int((time.time() + 8 * 60 * 60) * 1000)

    def refresh(_refresh_token):
        # Keep the first caller inside the machine-wide lock long enough for
        # the second caller to contend, then re-read the rotated credential.
        time.sleep(0.1)
        return {
            "access_token": "shared-access",
            "refresh_token": "shared-refresh",
            "expires_at_ms": refreshed_expiry,
        }

    with patch(
        "agent.claude_auth_lease.refresh_anthropic_oauth_pure",
        side_effect=refresh,
    ) as refresh_call, ThreadPoolExecutor(max_workers=2) as executor:
        leases = list(
            executor.map(lambda _: acquire_claude_auth_lease(2 * 60 * 60), range(2))
        )

    refresh_call.assert_called_once_with("old-refresh")
    assert all(lease.available for lease in leases)
    assert len({lease.generation for lease in leases}) == 1
    assert sum(lease.refreshed for lease in leases) == 1


def test_lease_and_anthropic_pool_share_one_rotating_refresh(tmp_path, monkeypatch):
    config_dir = tmp_path / "claude-config"
    hermes_home = tmp_path / "hermes"
    _write_credentials(
        config_dir,
        expires_at_ms=int((time.time() + 5 * 60) * 1000),
    )
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    refreshed_expiry = int((time.time() + 8 * 60 * 60) * 1000)
    refresh_calls = []

    def refresh(refresh_token):
        refresh_calls.append(refresh_token)
        time.sleep(0.1)
        return {
            "access_token": "shared-access",
            "refresh_token": "shared-refresh",
            "expires_at_ms": refreshed_expiry,
        }

    from agent.credential_pool import CredentialPool, PooledCredential

    entry = PooledCredential(
        provider="anthropic",
        id="claude-code",
        label="Claude Code",
        auth_type="oauth",
        priority=0,
        source="claude_code",
        access_token="old-access",
        refresh_token="old-refresh",
        expires_at_ms=int((time.time() + 5 * 60) * 1000),
    )
    pool = CredentialPool("anthropic", [entry])
    start = threading.Barrier(2)

    def acquire_lease():
        start.wait()
        return acquire_claude_auth_lease(2 * 60 * 60)

    def refresh_pool():
        start.wait()
        return pool._refresh_entry(entry, force=False)

    with patch(
        "agent.claude_auth_lease.refresh_anthropic_oauth_pure",
        side_effect=refresh,
    ), patch(
        "agent.anthropic_adapter.refresh_anthropic_oauth_pure",
        side_effect=refresh,
    ), ThreadPoolExecutor(max_workers=2) as executor:
        lease_future = executor.submit(acquire_lease)
        pool_future = executor.submit(refresh_pool)
        lease = lease_future.result()
        pooled = pool_future.result()

    assert refresh_calls == ["old-refresh"]
    assert lease.available is True
    assert pooled is not None
    assert pooled.access_token == "shared-access"
    assert pooled.refresh_token == "shared-refresh"


def test_setup_token_env_is_a_lease_without_the_credential_file(monkeypatch, tmp_path):
    """A `claude setup-token` value in the environment needs no file and no refresh."""
    from agent import claude_auth_lease as lease_module

    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))  # no credential file here
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-example-long-lived")
    monkeypatch.setattr(
        lease_module,
        "refresh_anthropic_oauth_pure",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("must not refresh")),
    )
    lease = lease_module.acquire_claude_auth_lease(10 * 60 * 60)
    assert lease.available is True
    assert lease.refreshed is False
    assert lease.generation == lease_module.claude_auth_generation("sk-ant-oat01-example-long-lived")
    assert not (tmp_path / ".credentials.json").exists()


def _write_token_file(lease_module, line: str):
    """Write ``line`` to the token file for this test's HERMES_HOME and return its path."""
    path = lease_module.claude_token_env_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(line, encoding="utf-8")
    return path


def test_token_file_is_a_lease_when_the_environment_was_stripped(monkeypatch, tmp_path):
    """The hermes-tools MCP server inherits neither the token nor the gateway's
    CLAUDE_CONFIG_DIR from Codex or Claude Code; the systemd token file must
    still sign it in without touching the credential file or refreshing."""
    from agent import claude_auth_lease as lease_module

    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude-code"))  # no credential file
    monkeypatch.setattr(
        lease_module,
        "refresh_anthropic_oauth_pure",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("must not refresh")),
    )
    _write_token_file(lease_module, "CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-from-token-file\n")

    lease = lease_module.acquire_claude_auth_lease(10 * 60 * 60)

    assert lease.available is True
    assert lease.credentials_found is True
    assert lease.refreshed is False
    assert lease.generation == lease_module.claude_auth_generation("sk-ant-oat01-from-token-file")
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in os.environ  # resolved, not injected


def test_environment_token_wins_over_the_token_file(monkeypatch, tmp_path):
    from agent import claude_auth_lease as lease_module

    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude-code"))
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-from-environment")
    _write_token_file(lease_module, "CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-from-token-file\n")

    assert lease_module.resolve_claude_oauth_token() == "sk-ant-oat01-from-environment"
    lease = lease_module.acquire_claude_auth_lease(0)
    assert lease.generation == lease_module.claude_auth_generation("sk-ant-oat01-from-environment")


def test_missing_token_file_leaves_the_lease_unavailable(monkeypatch, tmp_path):
    """Without the file the preflight keeps its old fallbacks and finds nothing."""
    from agent import claude_auth_lease as lease_module

    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude-code"))
    assert not lease_module.claude_token_env_path().exists()

    assert lease_module.resolve_claude_oauth_token() == ""
    lease = lease_module.acquire_claude_auth_lease(0)
    assert lease.available is False
    assert lease.credentials_found is False


@pytest.mark.parametrize(
    "content, expected",
    [
        ("CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-plain\n", "sk-ant-oat01-plain"),
        ('export CLAUDE_CODE_OAUTH_TOKEN="sk-ant-oat01-exported"\n', "sk-ant-oat01-exported"),
        ("# systemd EnvironmentFile\nOTHER_SETTING=1\nCLAUDE_CODE_OAUTH_TOKEN='sk-ant-oat01-quoted'\n", "sk-ant-oat01-quoted"),
        ("CLAUDE_CODE_OAUTH_TOKEN=\n", ""),
        ("OTHER_SETTING=1\n", ""),
        ("# CLAUDE_CODE_OAUTH_TOKEN=commented-out\n", ""),
    ],
)
def test_token_file_parser_accepts_systemd_and_shell_forms(tmp_path, content, expected):
    from agent import claude_auth_lease as lease_module

    path = tmp_path / "claude-token.env"
    path.write_text(content, encoding="utf-8")
    assert lease_module._read_claude_oauth_token_file(path) == expected
    assert lease_module._read_claude_oauth_token_file(tmp_path / "absent.env") == ""
