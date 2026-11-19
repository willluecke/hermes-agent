"""A kanban worker that hit a rate limit exits so its card waits (2026-11-18).

`hermes chat -q` used to return 0 after any turn, so a Claude-lane worker
that hit the plan's 5-hour limit looked like a worker that ended without
reporting, and the dispatcher gave up on the card after two "failures".
"""

from __future__ import annotations

import cli
from hermes_cli.kanban_db import KANBAN_RATE_LIMIT_EXIT_CODE


def test_a_rate_limited_or_billing_failure_in_a_worker_exits_with_tempfail(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_example")
    assert cli._kanban_rate_limit_exit({"failed": True, "failure_reason": "rate_limit"}) == KANBAN_RATE_LIMIT_EXIT_CODE
    assert cli._kanban_rate_limit_exit({"failed": True, "failure_reason": "billing"}) == KANBAN_RATE_LIMIT_EXIT_CODE


def test_other_turns_and_non_workers_keep_the_plain_exit(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_example")
    assert cli._kanban_rate_limit_exit({"failed": True, "failure_reason": "context_overflow"}) is None
    assert cli._kanban_rate_limit_exit({"failed": False, "failure_reason": "rate_limit"}) is None
    assert cli._kanban_rate_limit_exit(None) is None
    monkeypatch.delenv("HERMES_KANBAN_TASK")
    assert cli._kanban_rate_limit_exit({"failed": True, "failure_reason": "rate_limit"}) is None
