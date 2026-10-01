"""agent.turn_time_limit_hours: the wall-clock cap on one native CLI turn."""

from __future__ import annotations

import pytest

from agent.turn_time_limit import (
    DEFAULT_TURN_TIME_LIMIT_SECONDS,
    describe_duration,
    describe_limit,
    inactivity_stop_text,
    time_limit_stop_text,
    turn_time_limit_seconds,
)


def _cfg(value):
    return {"agent": {"turn_time_limit_hours": value}}


def test_default_cap_is_twelve_hours():
    assert DEFAULT_TURN_TIME_LIMIT_SECONDS == 12 * 60 * 60
    assert turn_time_limit_seconds({}) == 12 * 60 * 60
    assert turn_time_limit_seconds({"agent": {}}) == 12 * 60 * 60


def test_config_value_is_read_in_hours():
    assert turn_time_limit_seconds(_cfg(3)) == 3 * 60 * 60
    assert turn_time_limit_seconds(_cfg("1.5")) == 90 * 60


@pytest.mark.parametrize("value", [0, -1, "0", float("inf")])
def test_zero_negative_or_infinite_means_no_cap(value):
    assert turn_time_limit_seconds(_cfg(value)) == 0.0


@pytest.mark.parametrize("value", ["twelve", [], {}, True, float("nan")])
def test_a_malformed_value_keeps_the_default_not_no_cap(value):
    assert turn_time_limit_seconds(_cfg(value)) == DEFAULT_TURN_TIME_LIMIT_SECONDS


def test_reads_config_yaml_when_no_config_is_passed(monkeypatch):
    import hermes_cli.config

    monkeypatch.setattr(hermes_cli.config, "load_config", lambda: _cfg(5))
    assert turn_time_limit_seconds() == 5 * 60 * 60


def test_unreadable_config_keeps_the_default(monkeypatch):
    import hermes_cli.config

    def broken():
        raise OSError("config.yaml unreadable")

    monkeypatch.setattr(hermes_cli.config, "load_config", broken)
    assert turn_time_limit_seconds() == DEFAULT_TURN_TIME_LIMIT_SECONDS


def test_default_config_carries_the_key():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["agent"]["turn_time_limit_hours"] == 12


def test_limit_wording():
    assert describe_limit(12 * 3600) == "12-hour"
    assert describe_limit(5400) == "1.5-hour"
    assert describe_limit(600) == "10-minute"
    assert describe_limit(45) == "45-second"
    assert describe_duration(600) == "10 minutes"
    assert describe_duration(3600) == "1 hour"


def test_stop_texts_name_the_limit_and_how_to_continue():
    text = time_limit_stop_text("Codex", 12 * 3600)
    assert text == (
        "Codex stopped this turn at the 12-hour turn limit while it was still "
        "working. Send a message to have it continue."
    )
    quiet = inactivity_stop_text("Codex", 600)
    assert "after 10 minutes with no activity" in quiet
    assert quiet.endswith("Send a message to have it continue.")
