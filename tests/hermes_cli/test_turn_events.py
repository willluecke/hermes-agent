"""The per-session turn event bus lets a plugin put an event on the run stream."""

from __future__ import annotations

import pytest

from hermes_cli import turn_events


@pytest.fixture(autouse=True)
def _clean():
    turn_events._emitters.clear()
    yield
    turn_events._emitters.clear()


def test_emit_reaches_the_bound_emitter_with_the_progress_callback_shape():
    seen = []

    def emitter(event_type, tool_name=None, preview=None, args=None, **kwargs):
        seen.append((event_type, tool_name, preview, args, kwargs))

    turn_events.bind_turn_emitter("s1", emitter)
    sent = turn_events.emit_turn_event(
        "s1", "judge.verdict", text="Jev budget: k=3", source="jev", stage="budget", answers={"hard": 0.8}
    )
    assert sent is True
    assert seen == [("judge.verdict", "jev", "Jev budget: k=3", None, {"stage": "budget", "answers": {"hard": 0.8}})]


def test_emit_without_a_binding_or_through_a_failing_emitter_is_a_quiet_no_op():
    assert turn_events.emit_turn_event("nobody", "judge.verdict", text="x") is False

    def broken(*args, **kwargs):
        raise RuntimeError("stream closed")

    turn_events.bind_turn_emitter("s2", broken)
    assert turn_events.emit_turn_event("s2", "judge.verdict", text="x") is False


def test_rebinding_replaces_and_none_unbinds():
    first, second = [], []
    turn_events.bind_turn_emitter("s3", lambda *a, **k: first.append(a))
    turn_events.bind_turn_emitter("s3", lambda *a, **k: second.append(a))
    turn_events.emit_turn_event("s3", "judge.verdict", text="x")
    assert first == [] and len(second) == 1
    turn_events.bind_turn_emitter("s3", None)
    assert turn_events.turn_emitter("s3") is None
    turn_events.bind_turn_emitter("", lambda *a, **k: None)
    assert turn_events.turn_emitter("") is None
