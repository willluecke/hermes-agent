"""Tests for the system-one-preflight plugin (Jev verification preflight)."""

from __future__ import annotations

import importlib.util
import json
import threading
from pathlib import Path

import pytest

from hermes_cli.plugins import PluginContext, PluginManifest

_PLUGIN_FILE = Path(__file__).resolve().parents[2] / "plugins" / "system-one-preflight" / "__init__.py"
_spec = importlib.util.spec_from_file_location("system_one_preflight_under_test", _PLUGIN_FILE)
preflight = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(preflight)


class _FakeJev:
    def __init__(self, p=0.9, guard=None, raise_exc=None):
        self.p = p
        self.guard = guard or {}
        self.raise_exc = raise_exc
        self.calls = []

    def __call__(self, state, questions, timeout=None):
        self.calls.append({"state": state, "questions": questions, "timeout": timeout})
        if self.raise_exc:
            raise self.raise_exc
        answers = {}
        for key in questions:
            if key == "missing_verification":
                answers[key] = {"type": "noul", "noul": self.p}
            else:
                answers[key] = {"type": "noul", "noul": self.guard.get(key, 0.1)}
        return {"model": "jev-latest", "answers": answers, "usage": {"input_tokens": 100, "output_tokens": 0}}


@pytest.fixture
def harness(tmp_path, monkeypatch):
    settings = {"mode": "shadow", "threshold": 0.7, "log_path": str(tmp_path / "preflight.jsonl"), "tool_guard": "shadow"}
    monkeypatch.setattr(preflight, "_settings_reader", lambda key, default=None: settings.get(key, default))
    jev = _FakeJev()
    monkeypatch.setattr(preflight, "_ask", jev)
    preflight._turn_memo.clear()
    preflight._session_scope.clear()

    def records(event=None):
        path = tmp_path / "preflight.jsonl"
        if not path.exists():
            return []
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return [row for row in rows if event is None or row["event"] == event]

    return {"settings": settings, "jev": jev, "records": records}


HISTORY = [
    {"role": "user", "content": "Migrate the billing job to the new queue without losing events."},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "$ npm test\n3 passing"},
    {"role": "user", "content": "Keep the retry policy as is."},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
    {"role": "tool", "tool_call_id": "call_2", "content": "Error: ENOENT no such file queue.ts"},
    {"role": "user", "content": [{"type": "text", "text": "Is the migration done?"}, {"type": "image", "source": "x"}]},
]


def test_state_keeps_objective_instructions_evidence_and_failures_with_provenance():
    state = preflight.build_state(HISTORY[-1]["content"], HISTORY)
    assert state["objective"]["source"] == "user"
    assert state["objective"]["text"].startswith("Migrate the billing job")
    assert state["request"]["text"] == "Is the migration done?\n[image]"
    assert state["earlier_instructions"]["items"] == [
        "Migrate the billing job to the new queue without losing events.",
        "Keep the retry policy as is.",
    ]
    assert [item["tool"] for item in state["evidence"]["items"]] == ["terminal", "read_file"]
    assert state["evidence"]["source"] == "tool"
    assert state["unresolved_failures"]["items"] == ["read_file: Error: ENOENT no such file queue.ts"]
    assert "untrusted" in state["provenance"]


def test_shadow_mode_asks_jev_logs_and_injects_nothing(harness):
    result = preflight.on_pre_llm_call(session_id="s1", turn_id="t1", user_message="Is it done?", conversation_history=HISTORY, model="m", platform="api")
    assert result is None
    assert len(harness["jev"].calls) == 1
    call = harness["jev"].calls[0]
    assert list(call["questions"]) == ["missing_verification"]
    assert call["questions"]["missing_verification"]["type"] == "noul"
    assert call["timeout"] == preflight.DEFAULT_TIMEOUT_SECONDS
    [record] = harness["records"]("preflight")
    assert record["mode"] == "shadow" and record["arm"] == "shadow"
    assert record["p_missing"] == 0.9 and record["injected"] is False
    assert record["evidence_items"] == 2 and record["failures"] == 1


def test_jev_mode_injects_the_fixed_reminder_only_above_threshold(harness):
    harness["settings"]["mode"] = "jev"
    harness["jev"].p = 0.9
    assert preflight.on_pre_llm_call(session_id="s1", turn_id="t1", user_message="x", conversation_history=[]) == {"context": preflight.REMINDER}
    harness["jev"].p = 0.2
    assert preflight.on_pre_llm_call(session_id="s1", turn_id="t2", user_message="x", conversation_history=[]) is None
    harness["settings"]["threshold"] = 0.1
    assert preflight.on_pre_llm_call(session_id="s1", turn_id="t3", user_message="x", conversation_history=[]) == {"context": preflight.REMINDER}
    assert [row["injected"] for row in harness["records"]("preflight")] == [True, False, True]


def test_always_mode_reminds_even_when_jev_fails_and_logs_the_error(harness):
    harness["settings"]["mode"] = "always"
    harness["jev"].raise_exc = RuntimeError("timed out")
    assert preflight.on_pre_llm_call(session_id="s1", turn_id="t1", user_message="x", conversation_history=[]) == {"context": preflight.REMINDER}
    [record] = harness["records"]("preflight")
    assert record["p_missing"] is None and record["injected"] is True
    assert "timed out" in record["error"]


def test_jev_mode_never_injects_without_an_answer(harness):
    harness["settings"]["mode"] = "jev"
    harness["jev"].raise_exc = RuntimeError("boom")
    assert preflight.on_pre_llm_call(session_id="s1", turn_id="t1", user_message="x", conversation_history=[]) is None


def test_trial_arms_are_deterministic_per_session_and_cover_all_three(harness):
    harness["settings"]["mode"] = "trial"
    arms = {preflight.trial_arm(f"session-{n}") for n in range(60)}
    assert arms == set(preflight.TRIAL_ARMS)
    assert preflight.trial_arm("session-7") == preflight.trial_arm("session-7")
    harness["settings"]["trial_seed"] = "other"
    reseeded = {preflight.trial_arm(f"session-{n}") for n in range(60)}
    assert reseeded == set(preflight.TRIAL_ARMS)
    # Behaviour follows the arm.
    harness["jev"].p = 0.95
    for n in range(30):
        session = f"session-{n}"
        arm = preflight.trial_arm(session)
        result = preflight.on_pre_llm_call(session_id=session, turn_id="t1", user_message="x", conversation_history=[])
        assert (result is not None) == (arm in ("always", "jev")), (session, arm)
    logged = {row["session_id"]: row["arm"] for row in harness["records"]("preflight")}
    assert all(logged[f"session-{n}"] == preflight.trial_arm(f"session-{n}") for n in range(30))


def test_a_repeated_hook_within_one_turn_reuses_the_decision(harness):
    harness["settings"]["mode"] = "jev"
    first = preflight.on_pre_llm_call(session_id="s1", turn_id="t1", user_message="x", conversation_history=[])
    second = preflight.on_pre_llm_call(session_id="s1", turn_id="t1", user_message="x", conversation_history=[])
    assert first == second == {"context": preflight.REMINDER}
    assert len(harness["jev"].calls) == 1
    assert len(harness["records"]("preflight")) == 1


def test_off_mode_touches_nothing(harness):
    harness["settings"]["mode"] = "off"
    assert preflight.on_pre_llm_call(session_id="s1", turn_id="t1", user_message="x", conversation_history=[]) is None
    assert preflight.on_post_llm_call(session_id="s1", turn_id="t1", user_message="x", assistant_response="y", conversation_history=[]) is None
    assert preflight.on_pre_tool_call(tool_name="terminal", args={"command": "rm -rf /"}, session_id="s1") is None
    assert harness["jev"].calls == []
    assert harness["records"]() == []


def test_turn_end_records_what_the_model_did_after_the_prompt(harness):
    harness["settings"]["mode"] = "jev"
    preflight.on_pre_llm_call(session_id="s1", turn_id="t1", user_message="Is the migration done?", conversation_history=[])
    history = HISTORY + [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_3", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}, {"id": "call_4", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_3", "content": "ok"},
        {"role": "assistant", "content": "Yes, verified."},
    ]
    preflight.on_post_llm_call(session_id="s1", turn_id="t1", user_message="Is the migration done?\n[image]", assistant_response="Yes, verified.", conversation_history=history)
    [record] = harness["records"]("turn_end")
    assert record["tool_calls"] == 2 and record["tools"] == ["read_file", "terminal"]
    assert record["arm"] == "jev" and record["injected"] is True and record["p_missing"] == 0.9
    assert record["response_chars"] == len("Yes, verified.")


def _join_guard_threads():
    for thread in threading.enumerate():
        if thread.name == "system-one-preflight-guard":
            thread.join(timeout=5)


def test_tool_guard_classifies_off_the_critical_path_and_never_blocks(harness):
    harness["jev"].guard = {"deletes_data": 0.97, "within_scope": 0.12}
    preflight.on_pre_llm_call(session_id="s1", turn_id="t1", user_message="clean the build dir", conversation_history=[{"role": "user", "content": "we are tidying the repo"}])
    result = preflight.on_pre_tool_call(tool_name="terminal", args={"command": "rm -rf build"}, session_id="s1", turn_id="t1", tool_call_id="c9")
    assert result is None, "shadow guard must never block or modify"
    _join_guard_threads()
    [record] = harness["records"]("tool_guard")
    assert record["tool"] == "terminal" and record["tool_call_id"] == "c9"
    assert record["answers"]["deletes_data"] == 0.97 and record["answers"]["within_scope"] == 0.12
    assert set(record["answers"]) == set(preflight.GUARD_QUESTIONS)
    guard_call = harness["jev"].calls[-1]
    assert guard_call["state"]["action"]["value"]["command"] == "rm -rf build"
    assert guard_call["state"]["user_scope"]["items"] == ["we are tidying the repo", "clean the build dir"]
    assert record["scope_items"] == 2


def test_tool_guard_ignores_other_tools_and_can_be_switched_off(harness):
    assert preflight.on_pre_tool_call(tool_name="read_file", args={"path": "x"}, session_id="s1") is None
    harness["settings"]["tool_guard"] = "off"
    assert preflight.on_pre_tool_call(tool_name="terminal", args={"command": "ls"}, session_id="s1") is None
    _join_guard_threads()
    assert harness["jev"].calls == []


class _MinimalManager:
    _cli_ref = None
    _context_engine = None
    _tools: dict = {}

    def __init__(self):
        self._hooks: dict = {}
        self.tracked: list = []

    def _track_registration(self, manifest, kind, key, release):
        self.tracked.append((kind, key))
        return object()


def test_register_wires_the_three_hooks_and_the_settings_reader(monkeypatch):
    manifest = PluginManifest(name="system-one-preflight", version="0.1.0", description="test")
    manager = _MinimalManager()
    ctx = PluginContext(manifest=manifest, manager=manager)  # type: ignore[arg-type]
    monkeypatch.setattr(ctx, "get_config", lambda key, default=None: {"mode": "jev"}.get(key, default))
    preflight.register(ctx)
    assert set(manager._hooks) == {"pre_llm_call", "post_llm_call", "pre_tool_call"}
    assert manager._hooks["pre_llm_call"] == [preflight.on_pre_llm_call]
    assert ("hook", "pre_llm_call") in manager.tracked
    assert preflight.current_mode() == "jev"


class _FeedbackJev(_FakeJev):
    def __init__(self, p=0.1, ambiguous=0.1, guard=None, raise_exc=None):
        super().__init__(p=p, guard=guard, raise_exc=raise_exc)
        self.ambiguous = ambiguous

    def __call__(self, state, questions, timeout=None):
        body = super().__call__(state, questions, timeout=timeout)
        if "ambiguous" in questions:
            body["answers"]["ambiguous"] = {"type": "noul", "noul": self.ambiguous}
        return body


@pytest.fixture
def feedback(harness, monkeypatch):
    harness["settings"]["mode"] = "feedback"
    harness["settings"]["tool_guard"] = "feedback"
    jev = _FeedbackJev()
    monkeypatch.setattr(preflight, "_ask", jev)
    preflight._held_actions.clear()
    harness["jev"] = jev
    return harness


def test_feedback_mode_asks_both_questions_and_stays_quiet_when_jev_sees_no_issue(feedback):
    result = preflight.on_pre_llm_call(session_id="s1", turn_id="t1", user_message="What time is it in Lisbon?", conversation_history=[])
    assert result is None
    questions = feedback["jev"].calls[0]["questions"]
    assert set(questions) == {"missing_verification", "ambiguous"}
    [record] = feedback["records"]("preflight")
    assert record["arm"] == "feedback" and record["injected"] is False
    assert record["p_ambiguous"] == 0.1 and record["p_missing"] == 0.1


def test_feedback_mode_feeds_the_numbers_back_when_the_request_is_ambiguous(feedback):
    feedback["jev"].ambiguous = 0.82
    result = preflight.on_pre_llm_call(session_id="s1", turn_id="t1", user_message="fix it", conversation_history=[])
    assert result is not None
    note = result["context"]
    assert "advisory" in note and "0.82" in note and "0.10" in note
    assert "ask the user one focused clarifying question" in note
    assert preflight.REMINDER not in note
    [record] = feedback["records"]("preflight")
    assert record["injected"] is True and record["p_ambiguous"] == 0.82


def test_feedback_mode_feeds_back_when_jev_is_unsure_or_evidence_is_missing(feedback):
    feedback["jev"].ambiguous = 0.5  # unsure band
    assert preflight.on_pre_llm_call(session_id="s1", turn_id="t1", user_message="x", conversation_history=[]) is not None
    feedback["jev"].ambiguous = 0.05
    feedback["jev"].p = 0.9  # evidence missing
    assert preflight.on_pre_llm_call(session_id="s1", turn_id="t2", user_message="x", conversation_history=[]) is not None
    feedback["jev"].p = 0.05
    assert preflight.on_pre_llm_call(session_id="s1", turn_id="t3", user_message="x", conversation_history=[]) is None
    feedback["jev"].raise_exc = RuntimeError("down")
    assert preflight.on_pre_llm_call(session_id="s1", turn_id="t4", user_message="x", conversation_history=[]) is None, "no answer, no note"


def test_feedback_mode_repeats_the_same_note_within_a_turn(feedback):
    feedback["jev"].ambiguous = 0.9
    first = preflight.on_pre_llm_call(session_id="s1", turn_id="t1", user_message="x", conversation_history=[])
    second = preflight.on_pre_llm_call(session_id="s1", turn_id="t1", user_message="x", conversation_history=[])
    assert first == second and len(feedback["jev"].calls) == 1


def test_feedback_guard_holds_a_risky_out_of_scope_action_once_then_lets_the_retry_run(feedback):
    feedback["jev"].guard = {"deletes_data": 0.93, "within_scope": 0.15}
    preflight.on_pre_llm_call(session_id="s1", turn_id="t1", user_message="tidy the readme wording", conversation_history=[])
    held = preflight.on_pre_tool_call(tool_name="terminal", args={"command": "rm -rf ~/projects/old"}, session_id="s1", turn_id="t1", tool_call_id="c1")
    assert held is not None and held["action"] == "block"
    assert "deletes data" in held["message"] and "0.93" in held["message"] and "0.15" in held["message"]
    assert "confirm with the user" in held["message"]
    [hold] = feedback["records"]("tool_guard_hold")
    assert hold["tool"] == "terminal"
    # The model asked, the user agreed, the model retries the identical call.
    again = preflight.on_pre_tool_call(tool_name="terminal", args={"command": "rm -rf ~/projects/old"}, session_id="s1", turn_id="t1", tool_call_id="c2")
    assert again is None
    assert len(feedback["records"]("tool_guard_release")) == 1
    # A different command is judged afresh.
    feedback["jev"].guard = {"deletes_data": 0.02, "within_scope": 0.95}
    assert preflight.on_pre_tool_call(tool_name="terminal", args={"command": "ls"}, session_id="s1", turn_id="t1", tool_call_id="c3") is None


def test_feedback_guard_passes_in_scope_or_low_risk_actions_and_fails_open(feedback):
    feedback["jev"].guard = {"deletes_data": 0.95, "within_scope": 0.9}
    assert preflight.on_pre_tool_call(tool_name="terminal", args={"command": "rm -rf build"}, session_id="s1") is None, "in scope: runs"
    feedback["jev"].guard = {"external_disclosure": 0.3, "within_scope": 0.1}
    assert preflight.on_pre_tool_call(tool_name="write_file", args={"path": "x", "content": "y"}, session_id="s1") is None, "low risk: runs"
    feedback["jev"].raise_exc = RuntimeError("down")
    assert preflight.on_pre_tool_call(tool_name="terminal", args={"command": "rm -rf /"}, session_id="s1") is None, "Jev unavailable: fail open, log only"
    assert feedback["records"]("tool_guard_hold") == []
