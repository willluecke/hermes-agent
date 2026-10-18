"""Cross-process OAuth lease preparation for managed Claude Code runtimes.

Claude Code Max access tokens currently live for roughly eight hours and use a
rotating, single-use refresh token.  Several resident ``claude -p`` processes
can otherwise reach the refresh window together: one rotates the credential
and a losing process can clear the shared credential file after its stale
refresh fails.  Hermes avoids that upstream race by refreshing early under a
machine-wide file lock, before a managed turn begins.

A long-lived ``claude setup-token`` value is the other credential source.
The gateway's systemd unit reads it from ``$HERMES_HOME/claude-token.env``,
but processes started without that unit's environment -- the hermes-tools
MCP server under Codex, which only receives a whitelist of variables, and
the one under Claude Code, which does not forward the token to MCP children
-- used to see no credential at all and report the subscription unavailable.
:func:`resolve_claude_oauth_token` reads the same file so every Hermes
process agrees on the token.

The module's CLI emits metadata only.  Tokens never leave the credential file.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Optional

from agent.anthropic_adapter import (
    _write_claude_code_credentials,
    claude_code_credentials_path,
    read_claude_code_credentials,
    refresh_anthropic_oauth_pure,
)


logger = logging.getLogger(__name__)

DEFAULT_MIN_VALIDITY_SECONDS = 2 * 60 * 60 + 10 * 60
_EXPIRY_SAFETY_SECONDS = 60
_LOCK_TIMEOUT_SECONDS = 30.0
CLAUDE_OAUTH_TOKEN_ENV = "CLAUDE_CODE_OAUTH_TOKEN"
#: systemd ``EnvironmentFile`` holding ``CLAUDE_CODE_OAUTH_TOKEN=...`` for the
#: gateway; the same file is the fallback for processes that did not inherit it.
CLAUDE_TOKEN_ENV_FILENAME = "claude-token.env"


def claude_token_env_path() -> Path:
    """Return ``$HERMES_HOME/claude-token.env`` for the active Hermes home."""
    from hermes_constants import get_hermes_home

    return get_hermes_home() / CLAUDE_TOKEN_ENV_FILENAME


def _read_claude_oauth_token_file(path: Path) -> str:
    """Parse the token out of a systemd-style env file; "" when absent.

    Accepts ``KEY=value``, ``export KEY=value``, single or double quotes and
    ``#`` comment lines.  The value is returned to the caller only; it is
    never logged, and a missing or unreadable file is not an error.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, separator, value = line.partition("=")
        if not separator or key.strip() != CLAUDE_OAUTH_TOKEN_ENV:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        return value.strip()
    return ""


def resolve_claude_oauth_token() -> str:
    """Return the long-lived Claude Code OAuth token this install signs in with.

    The environment wins.  When ``CLAUDE_CODE_OAUTH_TOKEN`` is absent -- the
    hermes-tools MCP server is spawned with a stripped environment by both
    Codex and Claude Code -- fall back to ``$HERMES_HOME/claude-token.env``,
    the file the gateway's systemd unit already loads.  Returns "" when
    neither source has a value, so callers keep their existing fallbacks.
    """
    explicit = os.environ.get(CLAUDE_OAUTH_TOKEN_ENV, "").strip()
    if explicit:
        return explicit
    return _read_claude_oauth_token_file(claude_token_env_path())


@dataclass(frozen=True)
class ClaudeAuthLease:
    available: bool
    generation: str = ""
    expires_at_ms: int = 0
    refreshed: bool = False
    credentials_found: bool = False


def _integer_expiry(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def claude_auth_generation(access_token: Any) -> str:
    token = str(access_token or "")
    if not token:
        return ""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:24]


def _lease_from_credentials(
    credentials: Optional[dict[str, Any]],
    *,
    refreshed: bool = False,
) -> ClaudeAuthLease:
    if not isinstance(credentials, dict):
        return ClaudeAuthLease(available=False)
    access_token = str(credentials.get("accessToken") or "")
    expires_at_ms = _integer_expiry(credentials.get("expiresAt"))
    now_ms = int(time.time() * 1000)
    available = bool(
        access_token
        and (not expires_at_ms or now_ms < expires_at_ms - _EXPIRY_SAFETY_SECONDS * 1000)
    )
    return ClaudeAuthLease(
        available=available,
        generation=claude_auth_generation(access_token),
        expires_at_ms=expires_at_ms,
        refreshed=refreshed,
        credentials_found=True,
    )


def _has_required_validity(
    credentials: Optional[dict[str, Any]], min_validity_seconds: float
) -> bool:
    if not isinstance(credentials, dict) or not credentials.get("accessToken"):
        return False
    expires_at_ms = _integer_expiry(credentials.get("expiresAt"))
    if not expires_at_ms:
        return True
    required_until_ms = int(
        (time.time() + max(_EXPIRY_SAFETY_SECONDS, min_validity_seconds)) * 1000
    )
    return expires_at_ms > required_until_ms


def acquire_claude_auth_lease(
    min_validity_seconds: float = DEFAULT_MIN_VALIDITY_SECONDS,
) -> ClaudeAuthLease:
    """Return a durable Claude credential lease, refreshing early when needed.

    Only the file-backed credential source can participate in Hermes's shared
    lock.  Platform keychains remain Claude-owned and are returned as-is; the
    caller can still use Claude's native auth-status fallback for stores Hermes
    cannot inspect.
    """
    minimum = max(0.0, float(min_validity_seconds))
    # A long-lived token from `claude setup-token` is the native headless
    # credential: the CLI reads it from the environment, nothing rotates, and
    # the shared credential file is out of the loop entirely.  The token file
    # covers processes whose parent stripped the variable.
    explicit = resolve_claude_oauth_token()
    if explicit:
        return ClaudeAuthLease(
            available=True,
            generation=claude_auth_generation(explicit),
            expires_at_ms=0,
            credentials_found=True,
        )
    try:
        initial = read_claude_code_credentials()
    except Exception:
        logger.warning("Could not read Claude Code credentials", exc_info=True)
        return ClaudeAuthLease(available=False)
    if not initial:
        return ClaudeAuthLease(available=False)
    if _has_required_validity(initial, minimum):
        return _lease_from_credentials(initial)
    if initial.get("source") != "claude_code_credentials_file":
        return _lease_from_credentials(initial)

    credential_path = claude_code_credentials_path()
    try:
        from hermes_cli.auth import _auth_store_lock

        with _auth_store_lock(
            timeout_seconds=_LOCK_TIMEOUT_SECONDS,
            target_path=credential_path,
        ):
            # A competing Hermes process may have rotated the pair while this
            # caller waited for the lock.  Always re-read inside the lock.
            current = read_claude_code_credentials()
            if _has_required_validity(current, minimum):
                return _lease_from_credentials(current)
            refresh_token = str((current or initial).get("refreshToken") or "")
            if not refresh_token:
                return _lease_from_credentials(current or initial)
            refreshed = refresh_anthropic_oauth_pure(refresh_token)
            _write_claude_code_credentials(
                refreshed["access_token"],
                refreshed["refresh_token"],
                refreshed["expires_at_ms"],
            )
            persisted = read_claude_code_credentials()
            if persisted and persisted.get("accessToken") == refreshed["access_token"]:
                return _lease_from_credentials(persisted, refreshed=True)
            # The network refresh succeeded but persistence did not.  Do not
            # pretend the next process can use an in-memory token it never saw.
            logger.warning("Claude OAuth refresh succeeded but was not persisted")
            return ClaudeAuthLease(available=False, credentials_found=True)
    except Exception:
        # Another native Claude process does not honor Hermes's lock.  It may
        # have won a concurrent refresh, so adopt a newly-written credential
        # before declaring the lease unavailable.
        logger.warning("Could not prepare Claude OAuth lease", exc_info=True)
        try:
            latest = read_claude_code_credentials()
        except Exception:
            latest = None
        if latest and latest.get("accessToken") != initial.get("accessToken"):
            return _lease_from_credentials(latest)
        return _lease_from_credentials(initial)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min-validity-seconds",
        type=float,
        default=DEFAULT_MIN_VALIDITY_SECONDS,
    )
    args = parser.parse_args(argv)
    lease = acquire_claude_auth_lease(args.min_validity_seconds)
    print(
        json.dumps(
            {
                "available": lease.available,
                "generation": lease.generation,
                "expiresAt": lease.expires_at_ms,
                "refreshed": lease.refreshed,
                "credentialsFound": lease.credentials_found,
            },
            separators=(",", ":"),
        )
    )
    return 0 if lease.available else 1


if __name__ == "__main__":
    raise SystemExit(main())
