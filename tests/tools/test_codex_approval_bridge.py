"""Behavior tests for native Codex approvals on Hermes gateway/API runs."""

from __future__ import annotations

import pytest

from tools import approval


@pytest.fixture
def approval_session():
    session_key = "run-codex-approval-test"
    token = approval.set_current_session_key(session_key)
    try:
        yield session_key
    finally:
        approval.unregister_gateway_notify(session_key)
        with approval._lock:
            approval._gateway_queues.pop(session_key, None)
        approval.reset_current_session_key(token)


def test_attached_channel_receives_redacted_request_and_approves_once(
    approval_session,
):
    captured = {}
    fake_secret = "sk-proj-" + "A" * 40

    def notify(data):
        captured.update(data)
        assert approval.resolve_gateway_approval(approval_session, "once")

    approval.register_gateway_notify(approval_session, notify)

    outcome = approval.request_codex_approval(
        f"git push https://user:{fake_secret}@github.com/example/repo.git",
        "publish reviewed commits",
    )

    assert outcome == "once"
    assert fake_secret not in captured["command"]
    assert captured["allow_permanent"] is False
    assert captured["allow_session"] is True


def test_genuine_human_denial_is_distinct(approval_session):
    def notify(_data):
        assert approval.resolve_gateway_approval(approval_session, "deny")

    approval.register_gateway_notify(approval_session, notify)

    assert approval.request_codex_approval("git push", "external write") == "deny"


def test_missing_channel_is_unavailable(approval_session):
    assert (
        approval.request_codex_approval("git push", "external write")
        == "unavailable"
    )


def test_notification_failure_is_unavailable(approval_session):
    def notify(_data):
        raise RuntimeError("client disconnected")

    approval.register_gateway_notify(approval_session, notify)

    assert (
        approval.request_codex_approval("git push", "external write")
        == "unavailable"
    )


def test_unanswered_request_times_out_without_becoming_denial(
    approval_session, monkeypatch
):
    approval.register_gateway_notify(approval_session, lambda _data: None)
    monkeypatch.setattr(approval, "_get_approval_timeout", lambda: 0)

    assert (
        approval.request_codex_approval("git push", "external write")
        == "timeout"
    )
