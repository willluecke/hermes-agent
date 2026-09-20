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
            elif key.startswith("entails_") or key == "coverage":
                # The fidelity check: criteria are entailed and cover the request unless a test says otherwise.
                answers[key] = {"type": "noul", "noul": self.guard.get(key, 0.9)}
            else:
                answers[key] = {"type": "noul", "noul": self.guard.get(key, 0.1)}
        return {"model": "jev-latest", "answers": answers, "usage": {"input_tokens": 100, "output_tokens": 0}}


@pytest.fixture
def harness(tmp_path, monkeypatch):
    settings = {
        "mode": "shadow", "threshold": 0.7, "log_path": str(tmp_path / "preflight.jsonl"), "tool_guard": "shadow",
        # Self-tuning is off unless a test turns it on, and its state never comes from the real home.
        "tuning": "off", "tuning_state_path": str(tmp_path / "tuning.json"),
        # Retained command output never lands in the real home; the gate never re-runs anything unless a test says so.
        "ledger_dir": str(tmp_path / "evidence"), "controller_reruns": "on", "manifest_required": "off",
    }
    # The gate's runner answers from a table of bare command -> (output, exit); a test that wants the real runner sets it back.
    runner = {}
    monkeypatch.setattr(preflight, "_run_check", lambda command, cwd, timeout: {
        "output": runner.get(command, ("", 0))[0], "exit_code": runner.get(command, ("", 0))[1], "timed_out": False, "seconds": 0.01,
    })
    monkeypatch.setattr(preflight, "_settings_reader", lambda key, default=None: settings.get(key, default))
    jev = _FakeJev()
    monkeypatch.setattr(preflight, "_ask", jev)
    preflight._last_tune_check = 0.0
    preflight._tuned_cache.update(mtime=None, path=None, data={})
    preflight._turn_memo.clear()
    preflight._session_scope.clear()
    preflight._session_todos.clear()
    preflight._session_ledger.clear()
    preflight._session_manifest.clear()
    preflight._verify_memo.clear()
    preflight._held_actions.clear()
    preflight._session_excluded.clear()
    preflight._session_fidelity.clear()
    preflight._session_previous_answer.clear()
    preflight._pending_fidelity.clear()
    preflight._pending_fidelity_note.clear()
    preflight._session_budget.clear()

    def records(event=None):
        path = tmp_path / "preflight.jsonl"
        if not path.exists():
            return []
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return [row for row in rows if event is None or row["event"] == event]

    return {"settings": settings, "jev": jev, "records": records, "runner": runner}


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


def test_register_wires_the_hooks_and_the_settings_reader(monkeypatch):
    manifest = PluginManifest(name="system-one-preflight", version="0.1.0", description="test")
    manager = _MinimalManager()
    ctx = PluginContext(manifest=manifest, manager=manager)  # type: ignore[arg-type]
    monkeypatch.setattr(ctx, "get_config", lambda key, default=None: {"mode": "jev"}.get(key, default))
    preflight.register(ctx)
    assert set(manager._hooks) == {"pre_llm_call", "post_llm_call", "pre_tool_call", "post_tool_call", "transform_tool_result", "pre_verify"}
    assert manager._hooks["transform_tool_result"] == [preflight.on_transform_tool_result]
    assert manager._hooks["pre_llm_call"] == [preflight.on_pre_llm_call]
    assert ("hook", "pre_llm_call") in manager.tracked
    assert preflight.current_mode() == "jev"


class _FeedbackJev(_FakeJev):
    def __init__(self, p=0.1, ambiguous=0.1, hard=0.1, checkable=0.1, kind="answer", outcome=("worked", 0.9), guard=None, raise_exc=None):
        super().__init__(p=p, guard=guard, raise_exc=raise_exc)
        self.ambiguous = ambiguous
        self.hard = hard
        self.checkable = checkable
        self.kind = kind
        self.outcome = outcome

    def __call__(self, state, questions, timeout=None):
        body = super().__call__(state, questions, timeout=timeout)
        if "ambiguous" in questions:
            body["answers"]["ambiguous"] = {"type": "noul", "noul": self.ambiguous}
        if "previous_outcome" in questions:
            choice, p = self.outcome
            rest = round((1 - p) / 3, 4)
            probabilities = {key: (p if key == choice else rest) for key in questions["previous_outcome"]["criteria"]}
            body["answers"]["previous_outcome"] = {"type": "choice", "choice": choice, "confidence": 0.9, "probabilities": probabilities}
        if "difficulty" in questions:
            # Real score shape: per-level probabilities keyed by level number.
            easy = round(1 - self.hard, 4)
            body["answers"]["difficulty"] = {
                "type": "score", "score": 1 + 2 * self.hard, "confidence": 0.9,
                "probabilities": {"0": round(easy / 2, 4), "1": round(easy / 2, 4), "2": round(self.hard * 0.7, 4), "3": round(self.hard * 0.3, 4)},
                "legend": {str(i): level for i, level in enumerate(preflight.DIFFICULTY_LEVELS)},
            }
        if "checkable" in questions:
            body["answers"]["checkable"] = {"type": "noul", "noul": self.checkable}
        if "kind" in questions:
            body["answers"]["kind"] = {"type": "choice", "choice": self.kind, "confidence": 0.9, "probabilities": {self.kind: 0.9}}
        body["model"] = "jev-1.13.0"
        return body


@pytest.fixture
def emitted(monkeypatch):
    """Bind a run-stream emitter for session s1 and collect what the plugin sends."""
    from hermes_cli import turn_events

    events = []
    turn_events._emitters.clear()
    turn_events.bind_turn_emitter("s1", lambda event_type, tool_name=None, preview=None, args=None, **kwargs: events.append({"event": event_type, "judge": tool_name, "text": preview, **kwargs}))
    yield events
    turn_events._emitters.clear()


@pytest.fixture
def feedback(harness, monkeypatch):
    harness["settings"]["mode"] = "feedback"
    harness["settings"]["tool_guard"] = "feedback"
    jev = _FeedbackJev()
    monkeypatch.setattr(preflight, "_ask", jev)
    preflight._held_actions.clear()
    harness["jev"] = jev
    return harness


def test_feedback_mode_asks_the_budget_questions_and_stays_quiet_when_jev_sees_no_issue(feedback, emitted):
    result = preflight.on_pre_llm_call(session_id="s1", turn_id="t1", user_message="What time is it in Lisbon?", conversation_history=[])
    assert result is None
    questions = feedback["jev"].calls[0]["questions"]
    assert set(questions) == {"missing_verification", "ambiguous", "is_build", "difficulty", "checkable", "kind"}
    assert questions["difficulty"]["type"] == "score" and questions["difficulty"]["criteria"] == preflight.DIFFICULTY_LEVELS
    assert questions["checkable"]["type"] == "noul" and set(questions["checkable"]["criteria"]) == {"true", "false"}
    assert questions["kind"]["type"] == "choice" and "build" in questions["kind"]["criteria"]
    [record] = feedback["records"]("preflight")
    assert record["arm"] == "feedback" and record["injected"] is False
    assert record["p_ambiguous"] == 0.1 and record["p_missing"] == 0.1
    assert record["p_hard"] == pytest.approx(0.1) and record["p_checkable"] == 0.1 and record["kind"] == "answer" and record["k"] == 1
    # The budget is visible in the turn even when no note goes to the model.
    [event] = emitted
    assert event["event"] == "judge.verdict" and event["stage"] == "budget" and event["judge"] == "jev"
    assert event["decision"] == {"k": 1, "finish_loop": True, "plan": "direct", "injected": False, "note_reason": "", "baseline_n": 0}
    assert event["answers"]["hard"] == pytest.approx(0.1) and event["answers"]["kind"] == "answer"
    assert event["model"] == "jev-1.13.0" and "k=1" in event["text"]


def test_feedback_mode_feeds_the_numbers_back_when_the_request_is_ambiguous(feedback):
    feedback["jev"].ambiguous = 0.82
    result = preflight.on_pre_llm_call(session_id="s1", turn_id="t1", user_message="fix it", conversation_history=[])
    assert result is not None
    note = result["context"]
    assert "advisory" in note and "ambiguous enough to ask first) = 0.82" in note
    assert "ask the user one focused clarifying question" in note
    assert preflight.REMINDER not in note
    [record] = feedback["records"]("preflight")
    assert record["injected"] is True and record["p_ambiguous"] == 0.82


def test_feedback_mode_budget_asks_for_candidates_when_hard_and_checkable(feedback, emitted):
    feedback["jev"].hard = 0.8
    feedback["jev"].checkable = 0.9
    feedback["jev"].kind = "build"
    result = preflight.on_pre_llm_call(session_id="s1", turn_id="t1", user_message="make the importer idempotent", conversation_history=[])
    note = result["context"]
    assert "P(hard or very hard) = 0.80" in note and "P(objectively checkable) = 0.90" in note
    assert "3 independent candidate solutions" in note and "typesafe_decide" in note
    assert "never pick among your own candidates by reasoning alone" in note
    assert "clarifying question" not in note
    [record] = feedback["records"]("preflight")
    assert record["k"] == 3 and record["kind"] == "build" and record["p_hard"] == pytest.approx(0.8)
    [event] = emitted
    assert event["decision"] == {"k": 3, "finish_loop": True, "plan": "candidates", "injected": True, "note_reason": "candidates", "baseline_n": 0}
    # Hard but not checkable: criteria first, one candidate, no finish loop.
    feedback["jev"].checkable = 0.2
    result = preflight.on_pre_llm_call(session_id="s1", turn_id="t2", user_message="write a poem about the importer", conversation_history=[])
    assert "write the acceptance criteria first" in result["context"]
    assert feedback["records"]("preflight")[-1]["k"] == 1
    assert emitted[-1]["decision"]["plan"] == "criteria_only" and emitted[-1]["decision"]["finish_loop"] is False


def test_feedback_mode_no_longer_feeds_back_on_the_unsure_band_or_on_missing_evidence_alone(feedback):
    feedback["jev"].ambiguous = 0.5  # the unsure band used to trigger the note; it made the note a constant
    assert preflight.on_pre_llm_call(session_id="s1", turn_id="t1", user_message="x", conversation_history=[]) is None
    feedback["jev"].ambiguous = 0.7
    assert preflight.on_pre_llm_call(session_id="s1", turn_id="t1b", user_message="x", conversation_history=[]) is not None, "above the threshold, in the first turns"
    feedback["jev"].ambiguous = 0.05
    feedback["jev"].p = 0.9  # evidence missing: averaged 0.81 over real turns, so it no longer triggers on its own
    assert preflight.on_pre_llm_call(session_id="s1", turn_id="t2", user_message="x", conversation_history=[]) is None
    assert feedback["records"]("preflight")[-1]["p_missing"] == 0.9, "still asked and logged for the trial arms"
    feedback["jev"].p = 0.05
    assert preflight.on_pre_llm_call(session_id="s1", turn_id="t3", user_message="x", conversation_history=[]) is None
    feedback["jev"].raise_exc = RuntimeError("down")
    assert preflight.on_pre_llm_call(session_id="s1", turn_id="t4", user_message="x", conversation_history=[]) is None, "no answer, no note"


def test_budget_reads_level_mass_not_the_weighted_score():
    answers = {"difficulty": {"type": "score", "score": 1.9, "probabilities": {"0": 0.05, "1": 0.45, "2": 0.45, "3": 0.05}}}
    assert preflight._level_mass(answers, "difficulty", (2, 3)) == pytest.approx(0.5)
    assert preflight._level_mass({"difficulty": {"type": "noul", "noul": 0.9}}, "difficulty", (2, 3)) is None
    assert preflight.budget(0.5, 0.7) == {"k": 3, "finish_loop": True, "plan": "candidates"}
    assert preflight.budget(0.49, 0.99) == {"k": 1, "finish_loop": True, "plan": "direct"}
    assert preflight.budget(None, None) == {"k": 1, "finish_loop": True, "plan": "direct"}


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


import subprocess


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()

    def git(*args):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True)

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (root / "app.py").write_text("def greet():\n    return 'hi'\n")
    git("add", ".")
    git("commit", "-q", "-m", "init")
    (root / "app.py").write_text("def greet():\n    return 'hello world'\n")
    (root / "new_module.py").write_text("print('new')\n")
    (root / "demo.devproject").write_text(json.dumps({
        "features": [{"title": "Greeting", "description": "Says hello to the user", "status": "in-progress", "files_touched": ["app.py"]}]
    }))
    return root


TODOS_RESULT = json.dumps({"todos": [
    {"id": "1", "content": "Greeting returns hello world", "status": "completed"},
    {"id": "2", "content": "Errors are logged", "status": "in_progress"},
    {"id": "3", "content": "Docs updated", "status": "pending"},
]})


def test_post_tool_call_remembers_todos_and_ledger_rows(feedback):
    preflight.on_post_tool_call(tool_name="todo", args={}, result=TODOS_RESULT, session_id="s1")
    assert [item["content"] for item in preflight._session_todos["s1"]] == ["Greeting returns hello world", "Errors are logged", "Docs updated"]
    preflight.on_post_tool_call(tool_name="terminal", args={"command": "npm test"}, result="x" * 5000 + "\n1 failing", session_id="s1")
    preflight.on_post_tool_call(tool_name="terminal", args={"command": "ls -la"}, result="files", session_id="s1")
    rows = preflight.ledger_state("s1")["rows"]
    assert [(row["id"], row["command"], row["check"]) for row in rows] == [("c1", "npm test", True), ("c2", "ls -la", False)]
    assert rows[0]["exit"] is None and rows[0]["status"] == "unknown", "text with no exit code stays unknown"
    assert Path(rows[0]["file"]).read_text().endswith("1 failing") and rows[0]["chars"] == 5010


def test_verify_judge_keeps_the_model_going_on_unmet_criteria(feedback, repo, emitted):
    preflight.on_post_tool_call(tool_name="todo", args={}, result=TODOS_RESULT, session_id="s1")
    preflight.on_post_tool_call(tool_name="terminal", args={"command": "pytest -q"}, result="1 failed, 3 passed", session_id="s1")
    feedback["jev"].guard = {"criterion_1": 0.95, "criterion_2": 0.05, "checks_failing": 0.9, "claims_unverified": 0.1}
    result = preflight.on_pre_verify(
        session_id="s1", platform="api_server", model="m", coding=True, attempt=0,
        final_response="Done, everything passes.",
        changed_paths=[str(repo / "app.py"), str(repo / "new_module.py")],
    )
    assert result is not None and result["action"] == "continue"
    message = result["message"]
    assert "Errors are logged" in message and "0.05" in message
    assert "Greeting returns hello world" not in message, "a satisfied criterion is not listed"
    assert "still pending: 1" in message
    assert "Failure excerpts show a failure" in message
    call = feedback["jev"].calls[-1]
    state = call["state"]
    assert "+    return 'hello world'" in state["diff"]["text"]
    assert "new file: new_module.py" in state["diff"]["text"]
    assert state["features"]["items"][0]["title"] == "Greeting"
    assert [item["text"] for item in state["acceptance_criteria"]["items"]] == ["Greeting returns hello world", "Errors are logged"]
    assert state["still_pending_todos"]["items"] == ["Docs updated"]
    assert state["evidence_ledger"]["items"] == [{"id": "c1", "command": "pytest -q", "exit": "unknown", "status": "fail", "kind": "pytest", "source": "agent", "counts": {"failed": 1, "passed": 3}, "failure_text": True}]
    assert state["failure_excerpts"]["items"][0]["row"] == "c1" and "1 failed" in state["failure_excerpts"]["items"][0]["excerpt"]
    assert state["result_manifest"] == {"source": "agent, checked by code", "registered": False, "items": []}
    assert "commands_run" not in state and "check_outputs" not in state
    assert state["final_message"]["text"] == "Done, everything passes."
    assert set(call["questions"]) == {"criterion_1", "criterion_2", "checks_failing", "claims_unverified"}
    assert call["questions"]["criterion_2"]["criteria"] == preflight.CRITERION_CRITERIA, "nouls carry explicit boundaries"
    assert call["questions"]["claims_unverified"]["criteria"] == preflight.CLAIMS_CRITERIA
    assert call["timeout"] == preflight.VERIFY_TIMEOUT_SECONDS
    [record] = feedback["records"]("verify")
    assert record["criteria"] == 2 and record["pending"] == 1 and record["features"] == 1
    assert record["ledger"] == 1 and record["ledger_checks"] == 1 and record["manifest_registered"] is False and record["excerpts"] == 1
    [event] = emitted
    assert event["event"] == "judge.verdict" and event["stage"] == "verify" and event["attempt"] == 0
    assert event["decision"]["action"] == "nudge" and event["decision"]["criteria"] == 2 and event["decision"]["pending"] == 1
    assert event["answers"]["Errors are logged"] == 0.05 and event["answers"]["Greeting returns hello world"] == 0.95
    assert event["answers"]["claims_unverified"] == 0.1 and event["answers"]["checks_failing"] == 0.9
    assert event["text"].startswith("Jev verify (attempt 1): criteria met 1/2 · manifest not registered · claims beyond 0.10 · checks failing 0.90 · ledger 1 rows · nudge")
    assert "Errors are logged" in event["text"] and event["decision"]["ledger"] == 1 and event["decision"]["manifest_registered"] is False
    assert len(record["findings"]) == 3


def test_verify_judge_lets_a_satisfied_change_finish(feedback, repo):
    preflight.on_post_tool_call(tool_name="todo", args={}, result=json.dumps({"todos": [{"id": "1", "content": "Greeting returns hello world", "status": "completed"}]}), session_id="s1")
    feedback["jev"].guard = {"criterion_1": 0.97, "claims_unverified": 0.05}
    assert preflight.on_pre_verify(session_id="s1", attempt=0, final_response="Done.", changed_paths=[str(repo / "app.py")]) is None
    [record] = feedback["records"]("verify")
    assert record["findings"] == []


def test_verify_judge_uses_the_request_when_there_are_no_todos(feedback, repo):
    preflight.on_pre_llm_call(session_id="s2", turn_id="t1", user_message="Make greet return hello world", conversation_history=[])
    feedback["jev"].guard = {"criterion_1": 0.92, "claims_unverified": 0.05}
    assert preflight.on_pre_verify(session_id="s2", attempt=0, final_response="Done.", changed_paths=[str(repo / "app.py")]) is None
    call = feedback["jev"].calls[-1]
    assert "Make greet return hello world" in call["questions"]["criterion_1"]["instructions"]
    assert call["state"]["acceptance_criteria"]["items"] == [{"id": "criterion_1", "text": "Make greet return hello world"}]


def test_verify_judge_fails_open_and_respects_its_switches(feedback, repo):
    feedback["jev"].raise_exc = RuntimeError("down")
    assert preflight.on_pre_verify(session_id="s1", attempt=0, final_response="x", changed_paths=[str(repo / "app.py")]) is None
    [record] = feedback["records"]("verify")
    assert "down" in record["error"]
    feedback["jev"].raise_exc = None
    calls_before = len(feedback["jev"].calls)
    assert preflight.on_pre_verify(session_id="s1", attempt=0, final_response="x", changed_paths=[]) is None
    feedback["settings"]["verify_judge"] = "off"
    assert preflight.on_pre_verify(session_id="s1", attempt=0, final_response="x", changed_paths=[str(repo / "app.py")]) is None
    assert len(feedback["jev"].calls) == calls_before


def test_build_requests_get_the_criteria_nudge_until_todos_exist(feedback):
    feedback["jev"].guard = {"is_build": 0.9}
    result = preflight.on_pre_llm_call(session_id="s3", turn_id="t1", user_message="Add a --json flag to the CLI", conversation_history=[])
    assert result == {"context": preflight.CRITERIA_NUDGE}
    preflight.on_post_tool_call(tool_name="todo", args={}, result=TODOS_RESULT, session_id="s3")
    assert preflight.on_pre_llm_call(session_id="s3", turn_id="t2", user_message="now the tests", conversation_history=[]) is None
    feedback["jev"].guard = {"is_build": 0.1}
    assert preflight.on_pre_llm_call(session_id="s4", turn_id="t1", user_message="explain the flag", conversation_history=[]) is None


def test_every_command_is_a_ledger_row_and_only_checks_count_as_checks(feedback, repo):
    preflight.on_pre_llm_call(session_id="s1", turn_id="t1", user_message="make hello.py print hello world", conversation_history=[])
    preflight.on_post_tool_call(tool_name="terminal", args={"command": "python3 hello.py"}, result="hello world", session_id="s1")
    preflight.on_post_tool_call(tool_name="terminal", args={"command": "pytest -q"}, result="3 passed", session_id="s1")
    feedback["jev"].guard = {"criterion_1": 0.9, "checks_failing": 0.05, "claims_unverified": 0.05}
    assert preflight.on_pre_verify(session_id="s1", attempt=0, final_response="Ran it, prints hello world.", changed_paths=[str(repo / "app.py")]) is None
    call = feedback["jev"].calls[-1]
    state = call["state"]
    assert [(item["id"], item["command"], item["status"]) for item in state["evidence_ledger"]["items"]] == [("c1", "python3 hello.py", "unknown"), ("c2", "pytest -q", "pass")]
    assert state["evidence_ledger"]["dropped"] == 0 and state["failure_excerpts"]["items"] == []
    assert "checks_failing" not in call["questions"], "no failing row, nothing to ask about"
    record = feedback["records"]("verify")[-1]
    assert record["ledger"] == 2 and record["ledger_checks"] == 1


def test_an_unchanged_finding_is_not_nudged_twice(feedback, repo):
    feedback["jev"].guard = {"criterion_1": 0.9, "claims_unverified": 0.9}
    preflight.on_pre_llm_call(session_id="s1", turn_id="t1", user_message="do it", conversation_history=[])
    first = preflight.on_pre_verify(session_id="s1", attempt=0, final_response="Done.", changed_paths=[str(repo / "app.py")])
    assert first is not None and "claims results" in first["message"]
    # Same findings, same diff, no new commands: the model is not going to change its mind.
    second = preflight.on_pre_verify(session_id="s1", attempt=1, final_response="Done, really.", changed_paths=[str(repo / "app.py")])
    assert second is None
    records = feedback["records"]("verify")
    assert [row["repeated"] for row in records] == [False, True]
    # New evidence (a command ran) makes the judge look again.
    preflight.on_post_tool_call(tool_name="terminal", args={"command": "python3 app.py"}, result="hello world", session_id="s1")
    third = preflight.on_pre_verify(session_id="s1", attempt=2, final_response="Done.", changed_paths=[str(repo / "app.py")])
    assert third is not None


# ---------------------------------------------------------------------------
# Closing the loop: implicit outcomes from the follow-up, and self-tuning
# ---------------------------------------------------------------------------

PRIOR = [
    {"role": "user", "content": "Add the --json flag"},
    {"role": "assistant", "content": "Added --json; the tests pass."},
]


def test_implicit_outcome_is_read_from_the_follow_up_and_emitted(feedback, emitted):
    feedback["jev"].outcome = ("partly", 0.86)
    preflight.on_pre_llm_call(session_id="s1", turn_id="t1", user_message="It prints, but the output is not valid JSON.", conversation_history=PRIOR)
    call = feedback["jev"].calls[0]
    question = call["questions"]["previous_outcome"]
    assert question["type"] == "choice" and set(question["criteria"]) == {"worked", "partly", "failed", "unrelated"}
    assert call["state"]["previous_answer"] == {"source": "agent", "text": "Added --json; the tests pass."}
    record = feedback["records"]("preflight")[-1]
    assert record["implicit_outcome"] == "partly" and record["p_implicit"] == 0.86
    outcome_events = [event for event in emitted if event["event"] == "judge.outcome"]
    assert len(outcome_events) == 1
    event = outcome_events[0]
    assert event["stage"] == "implicit" and event["judge"] == "jev"
    assert event["decision"] == {"outcome": "partly", "about": "previous_run", "p": 0.86}
    assert event["answers"]["partly"] == 0.86 and "partly 0.86" in event["text"]

    # Unrelated, or below the bar: still asked and emitted with the numbers, but not recorded.
    feedback["jev"].outcome = ("unrelated", 0.9)
    preflight.on_pre_llm_call(session_id="s1", turn_id="t2", user_message="Different topic now.", conversation_history=PRIOR)
    assert feedback["records"]("preflight")[-1]["implicit_outcome"] is None
    assert emitted[-1]["event"] == "judge.outcome" and emitted[-1]["decision"]["outcome"] is None and "not recorded" in emitted[-1]["text"]
    feedback["jev"].outcome = ("failed", 0.5)
    preflight.on_pre_llm_call(session_id="s1", turn_id="t3", user_message="hmm", conversation_history=PRIOR)
    assert feedback["records"]("preflight")[-1]["implicit_outcome"] is None and feedback["records"]("preflight")[-1]["p_implicit"] == 0.5

    # No previous answer to judge: the question is not asked and nothing is emitted.
    before = len(emitted)
    preflight.on_pre_llm_call(session_id="s1", turn_id="t4", user_message="first message", conversation_history=[])
    assert "previous_outcome" not in feedback["jev"].calls[-1]["questions"]
    assert [event["event"] for event in emitted[before:]] == ["judge.verdict"]


@pytest.fixture
def tuner(feedback, monkeypatch, tmp_path):
    feedback["settings"]["tuning"] = "auto"
    feedback["settings"]["tuning_min_labels"] = 20
    calls = {"get": [], "post": []}
    calibration = {
        "labelled": 40,
        "recommendation": {
            "verify_fail_threshold": 0.35, "verify_fail_threshold_basis": 30,
            "claims_flag_threshold": 0.7, "claims_flag_threshold_basis": 12,
        },
    }
    monkeypatch.setattr(preflight, "_http_get_json", lambda url, headers, timeout: (calls["get"].append(url), calibration)[1])
    monkeypatch.setattr(preflight, "_http_post_json", lambda url, headers, payload, timeout: calls["post"].append((url, payload)))
    return {"calls": calls, "calibration": calibration, "state": tmp_path / "tuning.json"}


def test_tuning_applies_justified_moves_and_records_them_everywhere(tuner, feedback, emitted):
    assert preflight.verify_fail_threshold() == 0.2 and preflight.claims_flag_threshold() == 0.8
    changes = preflight.run_tune("s1")
    # The criterion threshold has 30 labels behind it (floor 20) and moves; the claims one has 12 and does not.
    assert changes == {"verify_fail_threshold": {"from": 0.2, "to": 0.35, "basis": 30}}
    assert preflight.verify_fail_threshold() == 0.35 and preflight.claims_flag_threshold() == 0.8
    state = json.loads(tuner["state"].read_text())
    assert state["verify_fail_threshold"] == 0.35 and len(state["history"]) == 1
    [record] = feedback["records"]("tune")
    assert record["changes"]["verify_fail_threshold"]["to"] == 0.35 and record["error"] == "" and record["labelled"] == 40
    event = emitted[-1]
    assert event["event"] == "judge.verdict" and event["stage"] == "tune"
    assert "verify_fail_threshold 0.2 → 0.35" in event["text"] and "30 labelled turns" in event["text"]
    assert tuner["calls"]["get"][0].endswith("/management/judge/calibration?days=90")
    url, payload = tuner["calls"]["post"][0]
    assert url.endswith("/management/judge/tuning") and payload["changes"] == changes and payload["basis"] == 30
    assert payload["thresholds"] == {"verify_fail_threshold": 0.35, "claims_flag_threshold": None}
    # A second pass with the same recommendation moves nothing and writes nothing.
    assert preflight.run_tune("s1") == {}
    assert len(json.loads(tuner["state"].read_text())["history"]) == 1


def test_tuning_clamps_to_bounds_and_respects_off_and_outages(tuner, feedback, monkeypatch):
    tuner["calibration"]["recommendation"] = {
        "verify_fail_threshold": 0.9, "verify_fail_threshold_basis": 100,
        "claims_flag_threshold": 0.3, "claims_flag_threshold_basis": 100,
    }
    changes = preflight.run_tune("s1")
    assert changes["verify_fail_threshold"]["to"] == 0.6 and changes["claims_flag_threshold"]["to"] == 0.5
    assert preflight.claims_flag_threshold() == 0.5
    feedback["settings"]["tuning"] = "off"
    assert preflight.verify_fail_threshold() == 0.2 and preflight.claims_flag_threshold() == 0.8, "off ignores the tuned state without deleting it"
    assert preflight.maybe_tune("s1") is False
    feedback["settings"]["tuning"] = "auto"
    assert preflight.verify_fail_threshold() == 0.6, "auto picks the tuned state up again"

    def down(url, headers, timeout):
        raise RuntimeError("sync down")

    monkeypatch.setattr(preflight, "_http_get_json", down)
    assert preflight.run_tune("s1") is None
    assert feedback["records"]("tune")[-1]["error"].startswith("RuntimeError")
    assert preflight.verify_fail_threshold() == 0.6, "an outage changes nothing"


def test_maybe_tune_starts_one_background_pass_per_interval(tuner, monkeypatch):
    started = []

    class _Thread:
        def __init__(self, target=None, args=(), name="", daemon=True):
            self.args = args
            self.name = name

        def start(self):
            started.append((self.name, self.args))

    monkeypatch.setattr(preflight.threading, "Thread", _Thread)
    assert preflight.maybe_tune("s1") is True
    assert preflight.maybe_tune("s1") is False, "within the interval nothing starts"
    assert started == [("system-one-preflight-tune", ("s1",))]


def test_the_verify_judge_uses_the_tuned_claims_threshold(tuner, feedback, repo):
    tuner["calibration"]["recommendation"] = {"claims_flag_threshold": 0.55, "claims_flag_threshold_basis": 60}
    preflight.run_tune("s1")
    assert preflight.claims_flag_threshold() == 0.55
    feedback["jev"].guard = {"criterion_1": 0.9, "claims_unverified": 0.6}
    result = preflight.on_pre_verify(session_id="s1", platform="api_server", model="m", coding=True, attempt=0, final_response="Done.", changed_paths=[str(repo / "app.py")])
    assert result is not None and "claims results beyond the manifest and the ledger (P=0.60)" in result["message"]


def test_the_verify_judge_takes_criteria_from_the_acceptance_criteria_tool(feedback, repo):
    """The Codex and Claude lanes cannot reach `todo`; the bridge's stateless stand-in answers in the same shape."""
    registered = json.dumps({"todos": [
        {"id": "1", "content": "Greeting returns hello world", "status": "in_progress"},
        {"id": "2", "content": "Errors are logged", "status": "in_progress"},
    ], "note": "2 acceptance criteria registered."})
    preflight.on_post_tool_call(tool_name="acceptance_criteria", args={"criteria": ["Greeting returns hello world", "Errors are logged"]}, result=registered, session_id="s1")
    assert [item["content"] for item in preflight._session_todos["s1"]] == ["Greeting returns hello world", "Errors are logged"]
    feedback["jev"].guard = {"criterion_1": 0.95, "criterion_2": 0.05, "claims_unverified": 0.1}
    result = preflight.on_pre_verify(session_id="s1", platform="api_server", model="m", coding=True, attempt=0, final_response="Done.", changed_paths=[str(repo / "app.py")])
    assert result is not None and "Errors are logged" in result["message"] and "still pending" not in result["message"]
    call = feedback["jev"].calls[-1]
    assert [item["text"] for item in call["state"]["acceptance_criteria"]["items"]] == ["Greeting returns hello world", "Errors are logged"]


# ---------------------------------------------------------------------------
# Drift check
# ---------------------------------------------------------------------------

class _DriftJev(_FeedbackJev):
    """Answers the drift check's choice question with a fixed pick and probability."""

    def __init__(self, serving=("none", 0.8), **kwargs):
        super().__init__(**kwargs)
        self.serving = serving

    def __call__(self, state, questions, timeout=None):
        body = super().__call__(state, questions, timeout=timeout)
        if "serving" in questions:
            choice, p = self.serving
            keys = list(questions["serving"]["criteria"])
            rest = round((1 - p) / max(1, len(keys) - 1), 4)
            body["answers"]["serving"] = {
                "type": "choice", "choice": choice, "confidence": 0.9,
                "probabilities": {key: (p if key == choice else rest) for key in keys},
            }
        return body


@pytest.fixture
def drift(feedback, monkeypatch):
    jev = _DriftJev(kind="build")
    monkeypatch.setattr(preflight, "_ask", jev)
    feedback["jev"] = jev
    feedback["settings"]["drift_every"] = 3
    preflight._session_drift.clear()
    preflight._pending_drift.clear()
    return feedback


CRITERIA_RESULT = json.dumps({"todos": [
    {"id": "1", "content": "The --json flag prints valid JSON", "status": "in_progress"},
    {"id": "2", "content": "Existing tests still pass", "status": "in_progress"},
]})


def _turn(session="s1", turn="t1", text="Add a --json flag to the exporter"):
    preflight.on_pre_llm_call(session_id=session, turn_id=turn, user_message=text, conversation_history=[])


def _register(session="s1"):
    preflight.on_post_tool_call(session_id=session, tool_name="acceptance_criteria", args={}, result=CRITERIA_RESULT)


def _reads(n, session="s1", start=0, **kwargs):
    return [
        preflight.on_post_tool_call(session_id=session, tool_name="read_file", args={"path": f"src/{i}.py"}, result="...", **kwargs)
        for i in range(start, start + n)
    ]


def _serving_calls(harness):
    return [call for call in harness["jev"].calls if "serving" in call["questions"]]


def test_drift_check_judges_every_window_and_holds_the_next_call_with_the_steer(drift, emitted):
    _register()
    _turn()
    assert _reads(2) == [None, None]
    assert _serving_calls(drift) == [], "nothing is asked before the window fills"
    assert _reads(1, start=2) == [None], "without a steerable caller the steer waits for the next tool call"
    [call] = _serving_calls(drift)
    question = call["questions"]["serving"]
    assert question["type"] == "choice"
    assert list(question["criteria"]) == ["c1", "c2", "none"]
    assert question["criteria"]["none"] == preflight.DRIFT_NONE_TEXT
    state = call["state"]
    assert state["request"]["text"] == "Add a --json flag to the exporter"
    assert [item["key"] for item in state["acceptance_criteria"]["items"]] == ["c1", "c2"]
    assert [item["summary"] for item in state["recent_tool_calls"]["items"]] == ["src/0.py", "src/1.py", "src/2.py"]
    assert state["calls_this_turn"] == 3
    held = preflight.on_pre_tool_call(tool_name="read_file", args={"path": "src/3.py"}, session_id="s1", turn_id="t1", tool_call_id="c9")
    assert held is not None and held["action"] == "block"
    assert "served none of your acceptance criteria" in held["message"] and "0.80" in held["message"]
    assert "The --json flag prints valid JSON" in held["message"], "the steer names the earliest open criterion"
    assert preflight.on_pre_tool_call(tool_name="read_file", args={"path": "src/3.py"}, session_id="s1") is None, "held once"
    [record] = drift["records"]("drift")
    assert record["rule"] == "jev" and record["calls"] == 3 and record["window"] == 3 and record["criteria"] == 2
    assert record["chosen"] == "none" and record["p_none"] == 0.8 and record["steer"] is True and record["target"] == "1"
    [hold] = drift["records"]("drift_hold")
    assert hold["tool"] == "read_file"
    verdicts = [event for event in emitted if event.get("stage") == "drift"]
    [event] = verdicts
    assert event["event"] == "judge.verdict" and event["attempt"] == 1
    assert event["text"] == "Jev drift (after 3 calls): serving none · none 0.80 · steer"
    assert event["decision"]["steer"] is True and event["decision"]["target"] == "1"
    assert event["answers"]["The --json flag prints valid JSON"] == 0.1


def test_drift_check_hands_the_steer_to_a_caller_that_can_deliver_it(drift):
    _register()
    _turn()
    results = _reads(3, steerable=True)
    assert results[:2] == [None, None]
    assert results[2] == {"message": preflight.DRIFT_TEMPLATE.format(n=3, p=0.8, target="The --json flag prints valid JSON")}
    assert preflight._pending_drift == {}, "delivered steers never wait for a hold"
    assert preflight.on_pre_tool_call(tool_name="read_file", args={"path": "x"}, session_id="s1") is None


def test_drift_check_stays_quiet_when_the_calls_serve_a_criterion(drift, emitted):
    drift["jev"].serving = ("c2", 0.9)
    _register()
    _turn()
    assert _reads(3) == [None, None, None]
    assert preflight._pending_drift == {}
    [record] = drift["records"]("drift")
    assert record["chosen"] == "c2" and record["steer"] is False and record["drifting"] is False
    [event] = [event for event in emitted if event.get("stage") == "drift"]
    assert event["text"] == "Jev drift (after 3 calls): serving criterion 2 · none 0.05 · on track"


def test_drift_check_steers_a_build_turn_to_register_criteria_by_rule(drift, emitted):
    _turn()
    assert _reads(3) == [None, None, None]
    assert _serving_calls(drift) == [], "no rubric, no Jev question"
    pending = preflight._pending_drift["s1"]
    assert "no acceptance criteria are registered" in pending["message"]
    [record] = drift["records"]("drift")
    assert record["rule"] == "no_criteria" and record["calls"] == 3 and record["steer"] is True
    [event] = [event for event in emitted if event.get("stage") == "drift"]
    assert event["text"] == "Jev drift (after 3 calls): no acceptance criteria on a build turn · steer"
    preflight._pending_drift.clear()
    assert _reads(3) == [None, None, None]
    assert preflight._pending_drift == {} and len(drift["records"]("drift")) == 1, "the rule fires once per turn"


def test_drift_check_leaves_a_non_build_turn_without_criteria_alone(drift):
    drift["jev"].kind = "answer"
    _turn(text="What does the exporter do?")
    assert _reads(3) == [None, None, None]
    assert preflight._pending_drift == {} and drift["records"]("drift") == []


def test_drift_check_fails_open_and_respects_its_switch_and_replay(drift):
    _register()
    _turn()
    drift["jev"].raise_exc = RuntimeError("jev down")
    assert _reads(3) == [None, None, None]
    [record] = drift["records"]("drift")
    assert record["error"].startswith("RuntimeError") and record["steer"] is False
    assert preflight._pending_drift == {}
    drift["jev"].raise_exc = None
    drift["settings"]["drift_check"] = "off"
    assert _reads(3) == [None, None, None]
    assert len(_serving_calls(drift)) == 1, "switched off: nothing asked"
    drift["settings"]["drift_check"] = "on"
    _turn(turn="t2")
    assert _reads(3, replay=True) == [None, None, None]
    assert len(_serving_calls(drift)) == 1, "a replayed call is counted, never judged"
    assert preflight.drift_state("s1")["total"] == 3


def test_drift_steers_are_bounded_per_turn_and_a_new_turn_resets_the_window(drift):
    drift["settings"]["drift_max_steers"] = 1
    _register()
    _turn()
    _reads(3)
    assert preflight._pending_drift["s1"]["message"]
    preflight._pending_drift.clear()
    _reads(3)
    assert preflight._pending_drift == {}, "the steer budget is spent"
    assert drift["records"]("drift")[-1]["drifting"] is True and drift["records"]("drift")[-1]["steer"] is False
    _reads(2)
    _turn(turn="t2")
    _reads(1)
    assert len(_serving_calls(drift)) == 2, "the new turn started a fresh window"
    assert preflight.drift_state("s1")["total"] == 1


def test_an_undelivered_drift_steer_becomes_a_verify_finding(drift, repo):
    _register()
    _turn()
    _reads(3)
    assert preflight._pending_drift["s1"]
    drift["jev"].guard = {"criterion_1": 0.95, "criterion_2": 0.9, "claims_unverified": 0.1}
    result = preflight.on_pre_verify(
        session_id="s1", platform="api_server", model="m", coding=True, attempt=0,
        final_response="Done.", changed_paths=[str(repo / "app.py")],
    )
    assert result is not None and result["action"] == "continue"
    assert "The drift check found the last 3 tool calls served none of the acceptance criteria (P(none)=0.80)." in result["message"]
    assert preflight._pending_drift == {}
    [record] = drift["records"]("verify")
    assert len(record["findings"]) == 1


# ---------------------------------------------------------------------------
# Criteria fidelity check
# ---------------------------------------------------------------------------

class _FidelityJev(_DriftJev):
    """Answers the fidelity check's entailment nouls from a map by criterion id, and coverage from one value."""

    def __init__(self, entails=None, coverage=0.9, **kwargs):
        super().__init__(**kwargs)
        self.entails = dict(entails or {})
        self.coverage = coverage

    def __call__(self, state, questions, timeout=None):
        body = super().__call__(state, questions, timeout=timeout)
        for key in questions:
            if key.startswith("entails_"):
                body["answers"][key] = {"type": "noul", "noul": self.entails.get(key[len("entails_"):], 0.9)}
        if "coverage" in questions:
            body["answers"]["coverage"] = {"type": "noul", "noul": self.coverage}
        return body


@pytest.fixture
def fidelity(drift, monkeypatch):
    jev = _FidelityJev(kind="build", serving=("c1", 0.9))
    monkeypatch.setattr(preflight, "_ask", jev)
    drift["jev"] = jev
    preflight._session_excluded.clear()
    preflight._session_fidelity.clear()
    preflight._pending_fidelity.clear()
    preflight._pending_fidelity_note.clear()
    return drift


C1, C2, C3 = "The --json flag prints valid JSON", "The README gains a section on exporters", "Existing tests still pass"


def _criteria_result(*contents, status="in_progress"):
    return json.dumps({"todos": [{"id": str(i + 1), "content": text, "status": status} for i, text in enumerate(contents)], "note": "registered"})


def _register_criteria(*contents, session="s1", tool="acceptance_criteria", **kwargs):
    return preflight.on_post_tool_call(session_id=session, tool_name=tool, args={"criteria": list(contents)}, result=_criteria_result(*contents), **kwargs)


def _fidelity_calls(harness):
    return [call for call in harness["jev"].calls if "coverage" in call["questions"]]


def test_fidelity_check_asks_one_noul_per_criterion_plus_coverage_at_registration(fidelity, emitted):
    _turn()
    assert _register_criteria(C1, C2, C3) is None, "all entailed, full coverage: nothing to tell the model"
    [call] = _fidelity_calls(fidelity)
    assert list(call["questions"]) == ["entails_1", "entails_2", "entails_3", "coverage"]
    assert {question["type"] for question in call["questions"].values()} == {"noul"}
    assert C2 in call["questions"]["entails_2"]["instructions"]
    assert call["questions"]["entails_1"]["criteria"] == preflight.FIDELITY_ENTAILMENT_CRITERIA
    assert call["questions"]["coverage"]["criteria"] == preflight.FIDELITY_COVERAGE_CRITERIA
    assert call["timeout"] == preflight.DEFAULT_FIDELITY_TIMEOUT_SECONDS
    state = call["state"]
    assert state["request"] == {"source": "user", "text": "Add a --json flag to the exporter"}
    assert state["earlier_instructions"] == {"source": "user", "items": []}
    assert state["previous_answer"] == {"source": "agent, previous turn", "text": ""}
    assert "previous_answer" in state["provenance"]
    assert [item["id"] for item in state["acceptance_criteria"]["items"]] == ["1", "2", "3"]
    assert state["acceptance_criteria"]["source"] == "agent"
    [record] = fidelity["records"]("fidelity")
    assert record["criteria"] == 3 and record["excluded"] == [] and record["p_coverage"] == 0.9
    assert record["entailment"] == {"1": 0.9, "2": 0.9, "3": 0.9}
    assert record["entailment_threshold"] == 0.4 and record["coverage_threshold"] == 0.6
    assert record["steer"] is False and record["coverage_low"] is False and record["replay"] is False
    [event] = [event for event in emitted if event.get("stage") == "fidelity"]
    assert event["event"] == "judge.verdict" and event["attempt"] == 1
    assert event["text"] == "Jev fidelity: 3 criteria · entailed 3/3 · coverage 0.90 · on track"
    assert event["answers"][C1] == 0.9 and event["answers"]["coverage"] == 0.9
    assert event["decision"]["excluded"] == [] and event["decision"]["steer"] is False
    assert preflight._session_excluded["s1"] == [] and preflight._pending_fidelity_note == {}
    assert preflight.on_transform_tool_result(tool_name="acceptance_criteria", session_id="s1", result=_criteria_result(C1, C2, C3), args={}) is None


def test_fidelity_excludes_an_unentailed_criterion_from_judging_and_names_it_in_the_tool_result(fidelity, repo, emitted):
    fidelity["jev"].entails = {"2": 0.1}
    _turn()
    assert _register_criteria(C1, C2, C3) is None, "the default loop puts the note in the result, not in the hook return"
    assert preflight._session_excluded["s1"] == ["2"]
    assert [item["id"] for item in preflight.active_criteria("s1")] == ["1", "3"]
    assert [item["id"] for item in preflight._session_todos["s1"]] == ["1", "2", "3"], "the registered list itself is untouched"
    transformed = preflight.on_transform_tool_result(tool_name="acceptance_criteria", session_id="s1", result=_criteria_result(C1, C2, C3), args={})
    payload = json.loads(transformed)
    assert [item["id"] for item in payload["todos"]] == ["1", "2", "3"]
    assert "do not follow from what the user asked and will not be judged" in payload["preflight"]
    assert f'"{C2}" (P(entailed)=0.10)' in payload["preflight"]
    assert C1 not in payload["preflight"]
    assert preflight.on_transform_tool_result(tool_name="acceptance_criteria", session_id="s1", result="{}", args={}) is None, "delivered once"
    [event] = [event for event in emitted if event.get("stage") == "fidelity"]
    assert event["text"] == "Jev fidelity: 3 criteria · entailed 2/3 · coverage 0.90 · 1 excluded"
    assert event["decision"]["excluded"] == ["2"]
    # The verify judge and the drift check never see the excluded criterion.
    fidelity["jev"].guard = {"criterion_1": 0.95, "criterion_2": 0.9, "claims_unverified": 0.1}
    result = preflight.on_pre_verify(session_id="s1", platform="api_server", model="m", coding=True, attempt=0, final_response="Done.", changed_paths=[str(repo / "app.py")])
    assert result is None
    verify_call = fidelity["jev"].calls[-1]
    assert [item["text"] for item in verify_call["state"]["acceptance_criteria"]["items"]] == [C1, C3]
    [record] = fidelity["records"]("verify")
    assert record["criteria"] == 2 and record["excluded"] == 1
    _reads(3)
    [serving] = _serving_calls(fidelity)
    assert list(serving["questions"]["serving"]["criteria"]) == ["c1", "c3", "none"]


def test_fidelity_steers_once_per_turn_when_the_criteria_do_not_cover_the_request(fidelity, emitted):
    fidelity["jev"].coverage = 0.2
    _turn()
    assert _register_criteria(C1, steerable=True) == {"message": preflight.FIDELITY_COVERAGE_TEMPLATE.format(p=0.2)}
    assert preflight._pending_fidelity_note == {} and preflight._pending_fidelity == {}, "delivered live: nothing waits"
    [record] = fidelity["records"]("fidelity")
    assert record["coverage_low"] is True and record["steer"] is True
    assert [event["text"] for event in emitted if event.get("stage") == "fidelity"] == ["Jev fidelity: 1 criteria · entailed 1/1 · coverage 0.20 · steer"]
    assert _register_criteria(C1, C3, steerable=True) is None, "a second list in the same turn is judged but not steered again"
    assert fidelity["records"]("fidelity")[-1]["steer"] is False and fidelity["records"]("fidelity")[-1]["coverage_low"] is True
    assert [event["attempt"] for event in emitted if event.get("stage") == "fidelity"] == [1, 2]
    assert emitted[-1]["text"].endswith("coverage 0.20 · coverage low, already steered")
    _turn(turn="t2")
    assert _register_criteria(C1, C2, steerable=True) == {"message": preflight.FIDELITY_COVERAGE_TEMPLATE.format(p=0.2)}, "a new turn may be steered again"


def test_fidelity_thresholds_are_settings_with_the_documented_starting_values(fidelity):
    assert preflight.fidelity_entailment_threshold() == 0.4 and preflight.fidelity_coverage_threshold() == 0.6
    fidelity["jev"].entails = {"2": 0.5}
    fidelity["jev"].coverage = 0.5
    _turn()
    _register_criteria(C1, C2)
    record = fidelity["records"]("fidelity")[-1]
    assert record["excluded"] == [] and record["coverage_low"] is True, "0.5 clears entailment at 0.4 and misses coverage at 0.6"
    fidelity["settings"]["fidelity_entailment_threshold"] = 0.7
    fidelity["settings"]["fidelity_coverage_threshold"] = 0.3
    _turn(turn="t2")
    _register_criteria(C1, C2, C3)
    record = fidelity["records"]("fidelity")[-1]
    assert record["excluded"] == ["2"] and record["coverage_low"] is False
    assert record["entailment_threshold"] == 0.7 and record["coverage_threshold"] == 0.3
    assert preflight._session_excluded["s1"] == ["2"]


def test_fidelity_fails_open_and_respects_its_switch(fidelity, emitted):
    fidelity["jev"].entails = {"2": 0.1}
    fidelity["jev"].coverage = 0.1
    fidelity["jev"].raise_exc = RuntimeError("jev down")
    _turn()
    assert _register_criteria(C1, C2, steerable=True) is None
    assert [item["id"] for item in preflight._session_todos["s1"]] == ["1", "2"], "the list is registered exactly as written"
    assert preflight._session_excluded.get("s1") in (None, []) and preflight._pending_fidelity_note == {} and preflight._pending_fidelity == {}
    assert [item["id"] for item in preflight.active_criteria("s1")] == ["1", "2"]
    [record] = fidelity["records"]("fidelity")
    assert record["error"].startswith("RuntimeError") and record["excluded"] == [] and record["steer"] is False
    [event] = [event for event in emitted if event.get("stage") == "fidelity"]
    assert event["text"] == "Jev fidelity: 2 criteria · entailed 2/2 · coverage unknown · unavailable"
    fidelity["jev"].raise_exc = None
    fidelity["settings"]["fidelity_check"] = "off"
    assert _register_criteria(C1, C2, C3, steerable=True) is None
    assert len(_fidelity_calls(fidelity)) == 1, "switched off: nothing asked"
    assert [item["id"] for item in preflight.active_criteria("s1")] == ["1", "2", "3"]


def test_fidelity_on_a_replayed_lane_applies_the_exclusions_and_makes_the_coverage_steer_a_verify_finding(fidelity, repo):
    fidelity["jev"].entails = {"2": 0.1}
    fidelity["jev"].coverage = 0.2
    _turn()
    assert _register_criteria(C1, C2, C3, replay=True) is None
    assert preflight._pending_fidelity_note == {}, "no result to write into after the turn"
    assert preflight._pending_fidelity["s1"]["finding"] == "The criteria check found the acceptance criteria may not cover the request (P(cover)=0.20)."
    assert preflight._session_excluded["s1"] == ["2"]
    assert fidelity["records"]("fidelity")[-1]["replay"] is True
    fidelity["jev"].guard = {"criterion_1": 0.95, "criterion_2": 0.9, "claims_unverified": 0.1}
    result = preflight.on_pre_verify(session_id="s1", platform="api_server", model="m", coding=True, attempt=0, final_response="Done.", changed_paths=[str(repo / "app.py")])
    assert result is not None and result["action"] == "continue"
    assert "may not cover the request (P(cover)=0.20)" in result["message"]
    assert [item["text"] for item in fidelity["jev"].calls[-1]["state"]["acceptance_criteria"]["items"]] == [C1, C3]
    assert preflight._pending_fidelity == {}
    [record] = fidelity["records"]("verify")
    assert record["findings"] == ["The criteria check found the acceptance criteria may not cover the request (P(cover)=0.20)."]


def test_fidelity_reuses_the_verdict_for_a_status_only_update_and_rejudges_a_changed_list(fidelity):
    fidelity["jev"].entails = {"2": 0.1}
    _turn()
    _register_criteria(C1, C2)
    assert len(_fidelity_calls(fidelity)) == 1 and preflight._session_excluded["s1"] == ["2"]
    preflight._pending_fidelity_note.clear()
    preflight.on_post_tool_call(session_id="s1", tool_name="acceptance_criteria", args={}, result=_criteria_result(C1, C2, status="completed"))
    assert len(_fidelity_calls(fidelity)) == 1, "the same statements are not asked about twice"
    assert preflight._session_excluded["s1"] == ["2"] and preflight._pending_fidelity_note == {}
    assert [item["status"] for item in preflight._session_todos["s1"]] == ["completed", "completed"]
    fidelity["jev"].entails = {}
    _register_criteria(C1, C3)
    assert len(_fidelity_calls(fidelity)) == 2 and preflight._session_excluded["s1"] == []


def test_fidelity_judges_the_todo_tool_on_the_default_loop_too(fidelity):
    fidelity["jev"].entails = {"1": 0.05}
    _turn()
    assert _register_criteria(C2, C1, tool="todo") is None
    [call] = _fidelity_calls(fidelity)
    assert [item["text"] for item in call["state"]["acceptance_criteria"]["items"]] == [C2, C1]
    assert preflight._session_excluded["s1"] == ["1"]
    transformed = preflight.on_transform_tool_result(tool_name="todo", session_id="s1", result="not json at all", args={})
    assert transformed.startswith("not json at all\n\n") and C2 in transformed
    assert preflight.on_transform_tool_result(tool_name="read_file", session_id="s1", result="x", args={}) is None


def test_fidelity_is_skipped_when_the_plugin_never_saw_the_request(fidelity):
    fidelity["jev"].entails = {"1": 0.05, "2": 0.05}
    assert _register_criteria(C1, C2) is None, "no turn yet: nothing to judge against"
    assert _fidelity_calls(fidelity) == [] and preflight._session_excluded.get("s1") is None
    assert [item["id"] for item in preflight.active_criteria("s1")] == ["1", "2"]
    [record] = fidelity["records"]("fidelity")
    assert record["skipped"] == "no_request" and record["criteria"] == 2


def test_fidelity_reads_the_request_with_the_previous_answer_it_refers_to(fidelity):
    """"Implement the changes" names nothing by itself: the answer it refers to is part of the state."""
    history = [
        {"role": "user", "content": "Is there anything you think should be updated?"},
        {"role": "assistant", "content": "Yes: the one change worth making is the criteria fidelity check."},
    ]
    preflight.on_pre_llm_call(session_id="s1", turn_id="t1", user_message="implement the changes fully", conversation_history=history)
    _register_criteria(C1)
    [call] = _fidelity_calls(fidelity)
    assert call["state"]["request"]["text"] == "implement the changes fully"
    assert call["state"]["earlier_instructions"]["items"] == ["Is there anything you think should be updated?"]
    assert call["state"]["previous_answer"] == {"source": "agent, previous turn", "text": "Yes: the one change worth making is the criteria fidelity check."}
    assert "previous answer the request refers to" in call["questions"]["entails_1"]["instructions"]
    preflight.on_pre_llm_call(session_id="s1", turn_id="t2", user_message="now the tests", conversation_history=[{"role": "user", "content": "x"}])
    _register_criteria(C2)
    assert _fidelity_calls(fidelity)[-1]["state"]["previous_answer"]["text"] == "", "a turn with no previous answer carries none"


def test_criteria_arrive_wrapped_by_the_mcp_bridge_on_the_claude_and_codex_lanes(fidelity):
    """The MCP SDK wraps a tool's string result as {"result": "<json>"}; that is what both bridge lanes replay."""
    fidelity["jev"].entails = {"2": 0.1}
    _turn()
    wrapped = json.dumps({"result": _criteria_result(C1, C2)})
    assert preflight.parse_todos(wrapped) == [{"id": "1", "content": C1, "status": "in_progress"}, {"id": "2", "content": C2, "status": "in_progress"}]
    assert preflight.parse_todos(json.dumps({"result": "not json"})) is None
    assert preflight.on_post_tool_call(session_id="s1", tool_name="acceptance_criteria", args={}, result=wrapped, steerable=True) is not None
    assert [item["content"] for item in preflight._session_todos["s1"]] == [C1, C2]
    assert preflight._session_excluded["s1"] == ["2"]


@pytest.mark.parametrize("structured", [False, True])
def test_codex_mcp_result_replaces_superseded_criteria(feedback, repo, structured):
    from agent.codex_runtime import _codex_item_completion_payload
    from mcp.types import CallToolResult, TextContent
    from tools.acceptance_criteria_tool import acceptance_criteria

    old = "Only Opus may implement the change"
    replacement = "The report attributes the implementation accurately"
    _register_criteria(old)
    raw = acceptance_criteria({"criteria": [{"content": replacement, "status": "completed"}]})
    envelope = CallToolResult(
        content=[TextContent(type="text", text=raw)],
        structured_content={"result": raw} if structured else None,
    ).model_dump(mode="json", by_alias=True, exclude_none=True)
    result, is_error = _codex_item_completion_payload({"type": "mcpToolCall", "result": envelope})
    assert not is_error
    preflight.on_post_tool_call(
        session_id="s1", tool_name="acceptance_criteria", args={}, result=result, replay=True,
    )
    assert [item["content"] for item in preflight.active_criteria("s1")] == [replacement]
    _verify(paths=[str(repo / "app.py")])
    state = feedback["jev"].calls[-1]["state"]
    assert [item["text"] for item in state["acceptance_criteria"]["items"]] == [replacement]


def test_failed_mcp_criteria_result_does_not_replace_registered_criteria(feedback):
    _register_criteria(C1)
    result = {
        "isError": True,
        "structuredContent": {"result": _criteria_result(C2)},
        "content": [{"type": "text", "text": _criteria_result(C2)}],
    }
    preflight.on_post_tool_call(
        session_id="s1", tool_name="acceptance_criteria", args={}, result=result, replay=True,
    )
    assert [item["content"] for item in preflight.active_criteria("s1")] == [C1]


# ---------------------------------------------------------------------------
# Evidence ledger and result manifest
# ---------------------------------------------------------------------------

import subprocess as _subprocess  # noqa: E402
import sys as _sys  # noqa: E402


def _cmd(command, result, session="s1", **kwargs):
    return preflight.on_post_tool_call(tool_name="terminal", args={"command": command}, result=result, session_id=session, **kwargs)


def _manifest(items, session="s1", wrapped=False):
    payload = json.dumps({"manifest": items, "note": "n"})
    if wrapped:
        payload = json.dumps({"result": payload})
    return preflight.on_post_tool_call(tool_name="report_results", args={}, result=payload, session_id=session)


def _item(claim, evidence, predicate="passed", expected=None, criterion="1", id="r1"):
    return {"id": id, "criterion": criterion, "claim": claim, "evidence": evidence, "predicate": predicate, "expected": expected or {}}


def _verify(session="s1", attempt=0, final="Done.", paths=None, coding=True, **kwargs):
    return preflight.on_pre_verify(session_id=session, platform="api_server", model="m", coding=coding, attempt=attempt, final_response=final, changed_paths=paths or [], **kwargs)


class _ManifestJev(_FeedbackJev):
    """Answers the verify judge's per-assertion choices with one fixed read."""

    def __init__(self, assertion=("supported", 0.9), **kwargs):
        super().__init__(**kwargs)
        self.assertion = assertion

    def __call__(self, state, questions, timeout=None):
        body = super().__call__(state, questions, timeout=timeout)
        for key, question in questions.items():
            if key.startswith("assertion"):
                choice, p = self.assertion
                keys = list(question["criteria"])
                rest = round((1 - p) / max(1, len(keys) - 1), 4)
                body["answers"][key] = {"type": "choice", "choice": choice, "confidence": 0.9, "probabilities": {k: (p if k == choice else rest) for k in keys}}
        return body


def test_every_terminal_call_is_a_ledger_row_with_retained_output_and_a_workspace_digest(feedback, repo):
    preflight.on_pre_llm_call(session_id="s1", turn_id="t1", user_message="fix greet", conversation_history=[])
    _cmd(f"cd {repo} && pytest -q", json.dumps({"output": "3 passed in 0.1s", "exit_code": 0}))
    _cmd("python3 app.py", "hello world", cwd=str(repo))
    rows = preflight.ledger_state("s1")["rows"]
    assert [row["id"] for row in rows] == ["c1", "c2"]
    assert rows[0]["exit"] == 0 and rows[0]["status"] == "pass" and rows[0]["counts"] == {"passed": 3} and rows[0]["check"] is True
    assert rows[1]["exit"] is None and rows[1]["status"] == "unknown" and rows[1]["check"] is False and rows[1]["cwd"] == str(repo)
    assert preflight.ledger_state("s1")["roots"] == [str(repo)], "the cd prefix and the hook's cwd both name the repository"
    assert rows[0]["workspace"] and rows[0]["workspace"] == rows[1]["workspace"]
    assert Path(rows[1]["file"]).read_text() == "hello world"
    assert str(Path(rows[1]["file"])).startswith(feedback["settings"]["ledger_dir"])
    # An edit moves the digest; the next command runs under the new one.
    (repo / "app.py").write_text("def greet():\n    return 'changed'\n")
    preflight.on_post_tool_call(tool_name="write_file", args={"path": str(repo / "app.py"), "content": "x"}, result="ok", session_id="s1")
    _cmd("pytest -q", "3 passed", cwd=str(repo))
    rows = preflight.ledger_state("s1")["rows"]
    assert rows[2]["workspace"] != rows[0]["workspace"]
    assert preflight.ledger_state("s1")["workspace"] == rows[2]["workspace"]


def test_a_new_turn_keeps_the_previous_rows_under_p_ids_and_clears_the_manifest(feedback, repo):
    preflight.on_pre_llm_call(session_id="s1", turn_id="t1", user_message="fix", conversation_history=[])
    _cmd("pytest -q", "3 passed", cwd=str(repo))
    _manifest([_item("tests pass", ["c1"])])
    assert preflight._session_manifest["s1"][0]["evidence"] == ["c1"]
    old_file = preflight.ledger_state("s1")["rows"][0]["file"]
    preflight.on_pre_llm_call(session_id="s1", turn_id="t2", user_message="now docs", conversation_history=[])
    state = preflight.ledger_state("s1")
    assert state["rows"] == [] and [row["id"] for row in state["previous"]] == ["pc1"]
    assert "s1" not in preflight._session_manifest
    assert Path(old_file).exists(), "the previous turn's output stays for one more turn"
    _cmd("ls", "x")
    assert preflight.ledger_state("s1")["rows"][0]["id"] == "c1", "ids restart per turn"
    preflight.on_pre_llm_call(session_id="s1", turn_id="t3", user_message="more", conversation_history=[])
    assert not Path(old_file).exists()


def test_report_results_is_captured_raw_and_through_the_bridge_wrapper_and_counted_by_the_drift_check(feedback):
    _manifest([_item("a", ["c1"])])
    assert [item["claim"] for item in preflight._session_manifest["s1"]] == ["a"]
    _manifest([_item("b", ["c2"], predicate="count", expected={"passed": 2})], wrapped=True)
    assert preflight._session_manifest["s1"][0]["claim"] == "b" and preflight._session_manifest["s1"][0]["expected"] == {"passed": 2}
    assert feedback["records"]("manifest")[-1]["items"] == 1
    assert preflight.drift_state("s1")["calls"][-1] == {"tool": "report_results", "summary": "results reported"}


def test_a_build_turn_that_ran_checks_without_a_manifest_is_sent_back_by_rule_with_no_jev_call(feedback, repo, emitted):
    feedback["settings"]["manifest_required"] = "on"
    preflight.on_pre_llm_call(session_id="s1", turn_id="t1", user_message="fix greet", conversation_history=[])
    _cmd("pytest -q", "3 passed", cwd=str(repo))
    calls = len(feedback["jev"].calls)
    result = _verify(final="Done, 3 passed.", paths=[str(repo / "app.py")])
    assert result is not None and "No result manifest was registered although 1 check commands ran" in result["message"] and "(ledger rows c1)" in result["message"]
    assert len(feedback["jev"].calls) == calls, "by rule: no Jev call"
    [record] = feedback["records"]("verify")
    assert record["rule"] == "no_manifest" and record["ledger"] == 1 and record["manifest_registered"] is False
    event = emitted[-1]
    assert event["stage"] == "verify" and "no result manifest on a build turn that ran 1 checks" in event["text"] and event["decision"]["rule"] == "no_manifest"
    # The same answer again: repeated, the turn finishes.
    assert _verify(attempt=1, paths=[str(repo / "app.py")]) is None
    assert feedback["records"]("verify")[-1]["repeated"] is True
    # A build turn that ran no checks needs no manifest.
    preflight.on_pre_llm_call(session_id="s1", turn_id="t2", user_message="tweak", conversation_history=[])
    feedback["jev"].guard = {"criterion_1": 0.9, "claims_unverified": 0.05}
    assert _verify(paths=[str(repo / "app.py")]) is None
    assert feedback["records"]("verify")[-1].get("rule") is None
    # Nor does a turn that is not a build.
    preflight.on_pre_llm_call(session_id="s1", turn_id="t3", user_message="explain", conversation_history=[])
    _cmd("pytest -q", "3 passed", cwd=str(repo))
    assert _verify(paths=[str(repo / "app.py")], coding=False) is None
    assert feedback["records"]("verify")[-1].get("rule") is None
    # An empty manifest satisfies the rule: the answer claims no check result.
    preflight.on_pre_llm_call(session_id="s1", turn_id="t4", user_message="fix", conversation_history=[])
    _cmd("pytest -q", "3 passed", cwd=str(repo))
    _manifest([])
    assert _verify(paths=[str(repo / "app.py")]) is None
    assert feedback["records"]("verify")[-1]["manifest_registered"] is True and feedback["records"]("verify")[-1].get("rule") is None


def test_code_checks_each_manifest_claim_and_contradiction_and_missing_rows_are_findings(feedback, repo, emitted):
    preflight.on_pre_llm_call(session_id="s1", turn_id="t1", user_message="fix greet", conversation_history=[])
    _cmd("pytest -q", json.dumps({"output": "12 passed in 0.3s", "exit_code": 0}), cwd=str(repo))
    _cmd("pytest -q tests/x.py", json.dumps({"output": "1 failed, 3 passed in 0.3s", "exit_code": 1}), cwd=str(repo))
    _manifest([
        _item("plugin suite passes", ["c1"], id="r1"),
        _item("12 passed", ["c1"], predicate="count", expected={"passed": 12}, id="r2"),
        _item("x suite passes", ["c2"], id="r3"),
        _item("lint clean", ["c7"], id="r4"),
    ])
    feedback["runner"]["pytest -q"] = ("12 passed in 0.3s", 0)
    feedback["runner"]["pytest -q tests/x.py"] = ("1 failed, 3 passed in 0.3s", 1)
    feedback["jev"].guard = {"criterion_1": 0.9, "claims_unverified": 0.05, "checks_failing": 0.1}
    result = _verify(paths=[str(repo / "app.py")])
    assert result is not None
    assert 'contradicted by the evidence ledger: "x suite passes" (reported failure: k4)' in result["message"], "the gate's own run is what contradicts"
    assert 'cite rows that are not in the ledger: "lint clean"' in result["message"] and "Rows this turn: c1, c2" in result["message"]
    call = feedback["jev"].calls[-1]
    state = call["state"]
    assert state["evidence_ledger"]["items"][0] == {"id": "c1", "command": "pytest -q", "exit": 0, "status": "pass", "kind": "pytest", "source": "agent", "counts": {"passed": 12}, "fresh": True}
    assert [(item["id"], item["code_verdict"], item["basis"]) for item in state["result_manifest"]["items"]] == [("r1", "supported", "gate"), ("r2", "supported", "gate"), ("r3", "contradicted", "gate"), ("r4", "missing", "agent")]
    assert [(item["id"], item["source"]) for item in state["evidence_ledger"]["items"]] == [("c1", "agent"), ("c2", "agent"), ("k3", "controller"), ("k4", "controller")]
    assert state["result_manifest"]["registered"] is True and state["evidence_ledger"]["workspace_known"] is True
    assert [item["row"] for item in state["failure_excerpts"]["items"]] == ["k4", "c2"], "the gate's own failing run comes first"
    assert all("1 failed" in item["excerpt"] for item in state["failure_excerpts"]["items"])
    assert "built by code" in state["provenance"] and "decided by code" in state["provenance"]
    # Jev gets one typed choice per claim code could not refute, plus the usual nouls.
    assert set(call["questions"]) == {"criterion_1", "checks_failing", "claims_unverified", "assertion_r1", "assertion_r2"}
    question = call["questions"]["assertion_r1"]
    assert question["type"] == "choice" and set(question["criteria"]) == {"supported", "contradicted", "insufficient"}
    assert "plugin suite passes" in question["instructions"] and "rows c1" in question["instructions"]
    assert call["questions"]["claims_unverified"]["instructions"] == preflight.CLAIMS_QUESTION
    assert call["questions"]["checks_failing"]["instructions"] == preflight.CHECKS_FAILING_QUESTION
    [record] = feedback["records"]("verify")
    assert record["manifest"] == {"supported": 2, "contradicted": 1, "stale": 0, "insufficient": 0, "missing": 1}
    assert record["ledger"] == 2 and record["ledger_checks"] == 2 and record["manifest_registered"] is True and record["workspace"]
    assert [(r["row"], r["controller"], r["status"]) for r in record["reruns"]] == [("c1", "k3", "pass"), ("c2", "k4", "fail")], "one gate run per cited row, two claims share c1's"
    assert record["regressions"] == [] and record["weakening"] == {"files": [], "removed": 0, "skips": 0}
    assert [item["code"] for item in record["assertions"]] == ["supported", "supported", "contradicted", "missing"]
    assert record["assertion_grouping"] == "none" and record["assertions_dropped"] == 0 and record["state_chars"] > 0
    event = emitted[-1]
    assert event["text"].startswith("Jev verify (attempt 1): criteria met 1/1 · manifest 2 supported, 1 contradicted, 1 missing · claims beyond 0.05 · checks failing 0.10 · ledger 2 rows · 2 re-run by the gate · nudge")
    assert event["decision"]["manifest"]["contradicted"] == 1 and event["decision"]["ledger"] == 2 and event["decision"]["reruns"] == 2
    assert set(event["answers"]) == {"fix greet", "claims_unverified", "checks_failing"}, "assertion reads stay out of the calibration answers"


def test_a_check_that_ran_before_a_later_edit_is_stale_and_the_gate_reruns_it_itself(feedback, repo, emitted, monkeypatch):
    monkeypatch.setattr(preflight, "_run_check", preflight.evidence.run_check)  # the real subprocess this once
    preflight.on_pre_llm_call(session_id="s1", turn_id="t1", user_message="fix greet", conversation_history=[])
    command = f"{_sys.executable} -m pytest --version"
    _cmd(command, "pytest 8.0.0", cwd=str(repo))
    (repo / "app.py").write_text("def greet():\n    return 'edited after the check'\n")
    _manifest([_item("pytest is available", ["c1"], predicate="exit_zero")])
    feedback["jev"].guard = {"criterion_1": 0.9, "claims_unverified": 0.05}
    assert _verify(paths=[str(repo / "app.py")]) is None, "the gate's own run settled the claim"
    [record] = feedback["records"]("verify")
    [rerun] = record["reruns"]
    assert rerun["row"] == "c1" and rerun["controller"] == "k2" and rerun["status"] == "pass" and rerun["timed_out"] is False
    rows = preflight.ledger_state("s1")["rows"]
    assert rows[-1]["id"] == "k2" and rows[-1]["source"] == "controller" and rows[-1]["exit"] == 0 and rows[-1]["for"] == "c1" and rows[-1]["fresh"] is True
    assert record["manifest"]["supported"] == 1 and record["assertions"][0]["detail"] == "exit 0"
    state = feedback["jev"].calls[-1]["state"]
    assert [(item["id"], item["source"]) for item in state["evidence_ledger"]["items"]] == [("c1", "agent"), ("k2", "controller")]
    assert "1 re-run by the gate" in emitted[-1]["text"] and emitted[-1]["decision"]["reruns"] == 1


def test_a_stale_or_unknown_claim_the_gate_cannot_rerun_is_reported_honestly(feedback, repo):
    feedback["settings"]["controller_reruns"] = "on"
    preflight.on_pre_llm_call(session_id="s1", turn_id="t1", user_message="fix greet", conversation_history=[])
    _cmd("python3 app.py", json.dumps({"output": "hello", "exit_code": 0}), cwd=str(repo))
    _cmd("curl -s http://x/health", "ok", cwd=str(repo))
    (repo / "app.py").write_text("def greet():\n    return 'edited after the check'\n")
    _manifest([
        _item("app prints hello", ["c1"], predicate="exit_zero", id="r1"),
        _item("service is healthy", ["c2"], predicate="ran", id="r2"),
    ])
    feedback["jev"].guard = {"criterion_1": 0.9, "claims_unverified": 0.05}
    result = _verify(paths=[str(repo / "app.py")])
    assert result is not None
    assert 'rest on checks that ran before later edits and the gate could not re-run: "app prints hello" (ran before later edits: c1 (not re-run by the gate: c1: not a plain check runner)). Run them again and cite the new rows.' in result["message"]
    record = feedback["records"]("verify")[-1]
    assert record["reruns"] == [], "python3 app.py is not a check runner, so the gate never runs it"
    assert [(item["code"], item["basis"]) for item in record["assertions"]] == [("stale", "agent"), ("supported", "agent")], "a ran claim is not about freshness and rests on the agent's row"
    # Insufficient is not a finding: a claim with unknown status is reported, not refuted.
    preflight.on_pre_llm_call(session_id="s1", turn_id="t2", user_message="fix greet", conversation_history=[])
    _cmd("python3 app.py", "hello", cwd=str(repo))
    _manifest([_item("app prints hello", ["c1"], predicate="exit_zero")])
    assert _verify(paths=[str(repo / "app.py")]) is None
    record = feedback["records"]("verify")[-1]
    assert record["manifest"]["insufficient"] == 1 and record["assertions"][0]["detail"].startswith("exit code unknown on this lane (not re-run by the gate: c1: not a plain check runner)")


def test_jev_can_still_refute_a_code_supported_claim_on_its_wording(feedback, repo, monkeypatch):
    jev = _ManifestJev(assertion=("contradicted", 0.9))
    monkeypatch.setattr(preflight, "_ask", jev)
    feedback["jev"] = jev
    preflight.on_pre_llm_call(session_id="s1", turn_id="t1", user_message="fix greet", conversation_history=[])
    _cmd("pytest -q tests/one.py", json.dumps({"output": "2 passed in 0.1s", "exit_code": 0}), cwd=str(repo))
    _manifest([_item("the whole suite passes", ["c1"])])
    feedback["runner"]["pytest -q tests/one.py"] = ("2 passed in 0.1s", 0)
    jev.guard = {"criterion_1": 0.9, "claims_unverified": 0.05}
    result = _verify(paths=[str(repo / "app.py")])
    assert result is not None
    assert 'Jev reads the cited rows as contradicting the claim as worded: "the whole suite passes" (P(contradicted)=0.90)' in result["message"]
    record = feedback["records"]("verify")[-1]
    assert record["assertions"][0]["code"] == "supported" and record["assertions"][0]["jev"]["contradicted"] == 0.9
    jev.assertion = ("supported", 0.9)
    assert _verify(attempt=1, paths=[str(repo / "app.py")]) is None


def test_assertion_questions_are_grouped_by_criterion_under_the_question_cap_and_never_dropped_silently(feedback, repo):
    def criteria(n):
        return json.dumps({"todos": [{"id": str(i), "content": f"criterion {i}", "status": "in_progress"} for i in range(1, n + 1)]})

    preflight.on_pre_llm_call(session_id="s1", turn_id="t1", user_message="fix", conversation_history=[])
    preflight.on_post_tool_call(tool_name="todo", args={}, result=criteria(20), session_id="s1")
    _cmd("pytest -q", json.dumps({"output": "1 failed, 2 passed in 0.1s", "exit_code": 1}), cwd=str(repo))
    _manifest([_item(f"claim {i}", ["c1"], predicate="ran", criterion=str(i % 3 + 1), id=f"r{i}") for i in range(16)])
    feedback["jev"].guard = {**{f"criterion_{i}": 0.9 for i in range(1, 21)}, "claims_unverified": 0.05, "checks_failing": 0.1}
    _verify(paths=[str(repo / "app.py")])
    call = feedback["jev"].calls[-1]
    assert len(call["questions"]) <= preflight.JEV_MAX_QUESTIONS
    grouped = sorted(key for key in call["questions"] if key.startswith("assertions_"))
    assert grouped == ["assertions_1", "assertions_2", "assertions_3"], "22 questions leave 10 slots: 16 claims become 3 groups"
    assert "r0: claim 0 (rows c1)" in call["questions"]["assertions_1"]["instructions"]
    record = feedback["records"]("verify")[-1]
    assert record["assertion_grouping"] == "criterion" and record["assertions_dropped"] == 0
    assert all(item["grouped"] for item in record["assertions"])
    # 30 criteria fill the cap: nothing is asked about the claims, and the log says how many were dropped.
    preflight.on_pre_llm_call(session_id="s1", turn_id="t2", user_message="fix", conversation_history=[])
    preflight.on_post_tool_call(tool_name="todo", args={}, result=criteria(30), session_id="s1")
    _cmd("pytest -q", json.dumps({"output": "1 failed, 2 passed in 0.1s", "exit_code": 1}), cwd=str(repo))
    _manifest([_item(f"claim {i}", ["c1"], predicate="ran", criterion=str(i % 3 + 1), id=f"r{i}") for i in range(16)])
    feedback["jev"].guard = {**{f"criterion_{i}": 0.9 for i in range(1, 31)}, "claims_unverified": 0.05, "checks_failing": 0.1}
    _verify(paths=[str(repo / "app.py")])
    call = feedback["jev"].calls[-1]
    assert len(call["questions"]) == preflight.JEV_MAX_QUESTIONS and not any(key.startswith("assertion") for key in call["questions"])
    record = feedback["records"]("verify")[-1]
    assert record["assertion_grouping"] == "truncated" and record["assertions_dropped"] == 16


def test_a_change_to_tests_or_runner_config_is_flagged_for_the_human_not_as_a_finding(feedback, repo, emitted):
    (repo / "tests").mkdir()
    (repo / "tests" / "test_app.py").write_text("def test_x():\n    pass\n")
    preflight.on_pre_llm_call(session_id="s1", turn_id="t1", user_message="fix greet", conversation_history=[])
    feedback["jev"].guard = {"criterion_1": 0.9, "claims_unverified": 0.05}
    assert _verify(paths=[str(repo / "app.py"), str(repo / "tests" / "test_app.py")]) is None
    record = feedback["records"]("verify")[-1]
    assert record["machinery"] == ["tests/test_app.py"] and record["findings"] == []
    assert "machinery changed (1)" in emitted[-1]["text"] and emitted[-1]["decision"]["machinery"] == ["tests/test_app.py"]


def test_a_decisive_claim_is_supported_only_by_the_gates_own_run(feedback, repo, emitted):
    preflight.on_pre_llm_call(session_id="s1", turn_id="t1", user_message="fix greet", conversation_history=[])
    _cmd("pytest -q", json.dumps({"output": "12 passed in 0.3s", "exit_code": 0}), cwd=str(repo))
    _cmd("pytest -q >/dev/null 2>&1; echo '12 passed in 0.1s'", json.dumps({"output": "12 passed in 0.1s", "exit_code": 0}), cwd=str(repo))
    _cmd("bash run_tests.sh", json.dumps({"output": "all good", "exit_code": 0}), cwd=str(repo))
    _manifest([
        _item("suite passes", ["c1"], id="r1"),
        _item("suite passes (forged)", ["c2"], id="r2"),
        _item("wrapper passes", ["c3"], predicate="exit_zero", id="r3"),
        _item("the run happened", ["c2"], predicate="ran", id="r4"),
    ])
    feedback["runner"]["pytest -q"] = ("12 passed in 0.3s", 0)
    feedback["jev"].guard = {"criterion_1": 0.9, "claims_unverified": 0.05}
    assert _verify(paths=[str(repo / "app.py")]) is None, "insufficient is reported, never a finding"
    record = feedback["records"]("verify")[-1]
    verdicts = {item["id"]: (item["code"], item["basis"], item["detail"]) for item in record["assertions"]}
    assert verdicts["r1"] == ("supported", "gate", "every cited row reports a pass"), "a fresh agent row is still re-run; only the gate's row clears it"
    assert verdicts["r2"][0] == "insufficient" and "could not re-run it (c2: not a plain check runner)" in verdicts["r2"][2], "the forged runner row clears nothing"
    assert verdicts["r3"][0] == "insufficient" and "c3: not a plain check runner" in verdicts["r3"][2], "a wrapper's exit 0 clears nothing"
    assert verdicts["r4"] == ("supported", "agent", "rows exist"), "ran keeps the agent's row and says so"
    assert [(r["row"], r["controller"]) for r in record["reruns"]] == [("c1", "k4")]
    assert record["manifest"] == {"supported": 2, "contradicted": 0, "stale": 0, "insufficient": 2, "missing": 0}
    # With the gate's runner off, no decisive claim can be supported at all.
    feedback["settings"]["controller_reruns"] = "off"
    preflight.on_pre_llm_call(session_id="s1", turn_id="t2", user_message="fix greet", conversation_history=[])
    _cmd("pytest -q", json.dumps({"output": "12 passed in 0.3s", "exit_code": 0}), cwd=str(repo))
    _manifest([_item("suite passes", ["c1"])])
    assert _verify(paths=[str(repo / "app.py")]) is None
    record = feedback["records"]("verify")[-1]
    assert record["assertions"][0]["code"] == "insufficient" and "controller re-runs are off" in record["assertions"][0]["detail"]


def test_the_gate_strips_output_filters_and_keeps_the_original_command(feedback, repo):
    preflight.on_pre_llm_call(session_id="s1", turn_id="t1", user_message="fix greet", conversation_history=[])
    _cmd("pytest -q | grep -v failed", json.dumps({"output": "..\n2 passed in 0.1s", "exit_code": 0}), cwd=str(repo))
    _manifest([_item("suite passes", ["c1"])])
    feedback["runner"]["pytest -q"] = ("F..\n1 failed, 2 passed in 0.1s", 1)
    feedback["jev"].guard = {"criterion_1": 0.9, "claims_unverified": 0.05}
    result = _verify(paths=[str(repo / "app.py")])
    assert result is not None and 'contradicted by the evidence ledger: "suite passes" (reported failure: k2)' in result["message"]
    row = preflight.ledger_state("s1")["rows"][-1]
    assert row["id"] == "k2" and row["command"] == "pytest -q" and row["filtered_from"] == "pytest -q | grep -v failed" and row["status"] == "fail"
    assert feedback["records"]("verify")[-1]["reruns"][0]["command"] == "pytest -q"


def test_fewer_tests_than_the_earliest_run_is_a_finding_once_and_the_memo_survives_re_runs(feedback, repo):
    preflight.on_pre_llm_call(session_id="s1", turn_id="t1", user_message="fix greet", conversation_history=[])
    _cmd("pytest -q", json.dumps({"output": "5 passed in 0.3s", "exit_code": 0}), cwd=str(repo))
    (repo / "app.py").write_text("def greet():\n    return 'edited'\n")
    _cmd("pytest -q | tail -1", json.dumps({"output": "3 passed in 0.2s", "exit_code": 0}), cwd=str(repo))
    _manifest([_item("suite passes", ["c2"])])
    feedback["runner"]["pytest -q"] = ("3 passed in 0.2s", 0)
    feedback["jev"].guard = {"criterion_1": 0.9, "claims_unverified": 0.05}
    result = _verify(paths=[str(repo / "app.py")])
    assert result is not None and "Fewer tests ran than before the change: pytest -q went from 5 to 3 (rows c1 then k3). Restore the tests or state why in the answer." in result["message"]
    record = feedback["records"]("verify")[-1]
    assert record["regressions"] == [{"command": "pytest -q", "before": 5, "after": 3, "baseline": "c1", "controller": "k3"}]
    # The same answer again: the gate runs again (a new k row), but the memo ignores controller rows and the turn finishes.
    assert _verify(attempt=1, paths=[str(repo / "app.py")]) is None
    assert feedback["records"]("verify")[-1]["repeated"] is True
    # A run in the previous turn is the baseline when this turn ran nothing before its edits.
    preflight.on_pre_llm_call(session_id="s1", turn_id="t2", user_message="more", conversation_history=[])
    _manifest([_item("suite passes", ["pc1"])])
    feedback["runner"]["pytest -q"] = ("2 passed in 0.2s", 0)
    result = _verify(paths=[str(repo / "app.py")])
    assert result is not None and "went from 5 to 2 (rows pc1 then k1)" in result["message"]


def test_removed_assertions_or_added_skips_in_existing_tests_are_a_finding_once(feedback, repo, emitted):
    tests = repo / "tests"
    tests.mkdir()
    (tests / "test_app.py").write_text("def test_a():\n    assert 1 == 1\n\n\ndef test_b():\n    assert 2 == 2\n")
    _subprocess.run(["git", "-C", str(repo), "add", "."], check=True, capture_output=True)
    _subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "tests"], check=True, capture_output=True)
    (repo / "app.py").write_text("def greet():\n    return 'changed'\n")
    (tests / "test_app.py").write_text("import pytest\n\n\n@pytest.mark.skip\ndef test_a():\n    pass\n")
    preflight.on_pre_llm_call(session_id="s1", turn_id="t1", user_message="fix greet", conversation_history=[])
    feedback["jev"].guard = {"criterion_1": 0.9, "claims_unverified": 0.05}
    paths = [str(repo / "app.py"), str(tests / "test_app.py")]
    result = _verify(paths=paths)
    assert result is not None
    assert "Verification machinery weakened: 3 assertion or test lines removed and 1 skip markers added in tests/test_app.py. Restore them or state why in the answer." in result["message"]
    record = feedback["records"]("verify")[-1]
    assert record["weakening"] == {"files": [{"path": "tests/test_app.py", "removed": 3, "skips": 1}], "removed": 3, "skips": 1}
    assert record["machinery"] == ["tests/test_app.py"]
    assert "· tests weakened ·" in emitted[-1]["text"] and emitted[-1]["decision"]["weakening"]["removed"] == 3
    assert _verify(attempt=1, paths=paths) is None, "once: the same finding on the same evidence lets the turn finish"
    assert feedback["records"]("verify")[-1]["repeated"] is True


# ---------------------------------------------------------------------------
# Noise rules: follow-up requests and the session baseline
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "continue", "Continue.", "proceed", "proceed as recommended", "okay, go ahead", "yes please", "Yes, do it.",
    "implement the changes fully", "implement the recommendation", "go for it", "sounds good, proceed", "ok continue then",
    "proceed with the implementation", "finish it", "lgtm",
])
def test_continuation_requests_are_matched_by_the_fixed_list(text):
    assert preflight.continuation_request(text) is True


@pytest.mark.parametrize("text", [
    "", "Fix the plugin", "continue with the tests only", "what happened?", "proceed to delete the database",
    "implement the changes to the sidebar", "do you recommend the closing design items?", "x" * 81,
])
def test_instructions_and_questions_are_not_continuations(text):
    assert preflight.continuation_request(text) is False


def test_fidelity_on_a_follow_up_judges_the_proposed_work_and_records_a_low_read_without_steering(fidelity, emitted):
    fidelity["jev"].coverage = 0.2
    preflight.on_pre_llm_call(session_id="s1", turn_id="t1", user_message="proceed as recommended", conversation_history=[
        {"role": "user", "content": "Should we add a --json flag?"},
        {"role": "assistant", "content": "Yes: add the flag, keep text output unchanged, and cover it with a test."},
    ])
    assert _register_criteria(C1, C2) is None, "no steer on a follow-up"
    call = _fidelity_calls(fidelity)[-1]
    assert call["questions"]["coverage"]["instructions"] == preflight.FIDELITY_COVERAGE_QUESTION_CONTINUATION
    assert "follow-up" in call["state"]["provenance"] and call["state"]["previous_answer"]["text"].startswith("Yes: add the flag")
    record = fidelity["records"]("fidelity")[-1]
    assert record["coverage_low"] is True and record["steer"] is False and record["continuation"] is True and record["suppressed"] == "continuation"
    event = [e for e in emitted if e.get("stage") == "fidelity"][-1]
    assert event["text"].endswith("coverage 0.20 · coverage low on a follow-up, not steered") and event["decision"]["suppressed"] == "continuation"
    assert preflight._pending_fidelity_note == {} and preflight._pending_fidelity == {}
    # A replayed lane leaves no verify finding behind either.
    preflight.on_pre_llm_call(session_id="s1", turn_id="t2", user_message="continue", conversation_history=[])
    assert _register_criteria(C1, C3, replay=True) is None
    assert preflight._pending_fidelity == {}
    # An exclusion is still delivered on a follow-up: it is about a criterion, not about the request's wording.
    fidelity["jev"].entails = {"2": 0.1}
    preflight.on_pre_llm_call(session_id="s1", turn_id="t3", user_message="go ahead", conversation_history=[])
    assert _register_criteria(C1, C2) is None, "the default loop puts the note in the tool result, not the hook's return"
    note = preflight._pending_fidelity_note.pop("s1")
    assert "will not be judged" in note and "may not cover" not in note
    assert preflight._session_excluded["s1"] == ["2"]
    # A real instruction still steers.
    fidelity["jev"].entails = {}
    preflight.on_pre_llm_call(session_id="s1", turn_id="t4", user_message="Add a --json flag to the exporter", conversation_history=[])
    assert _register_criteria(C1, C3) is None
    note = preflight._pending_fidelity_note.pop("s1")
    assert "may not cover everything the request asks for (P(cover) = 0.20)" in note
    assert fidelity["records"]("fidelity")[-1]["continuation"] is False


def test_the_budget_note_is_sent_for_the_first_turns_then_only_when_the_read_stands_out(feedback, emitted):
    feedback["jev"].ambiguous = 0.64
    for turn in ("t1", "t2", "t3"):
        assert preflight.on_pre_llm_call(session_id="s1", turn_id=turn, user_message="x", conversation_history=[]) is not None
        assert feedback["records"]("preflight")[-1]["note_reason"] == "first_turns"
    record = feedback["records"]("preflight")[-1]
    assert record["baseline_n"] == 2 and record["baseline_ambiguous"] == pytest.approx(0.64)
    # The fourth turn reads like the session's usual: withheld, and the row says so.
    assert preflight.on_pre_llm_call(session_id="s1", turn_id="t4", user_message="x", conversation_history=[]) is None
    record = feedback["records"]("preflight")[-1]
    assert record["injected"] is False and record["note_reason"] == "" and record["baseline_n"] == 3
    assert emitted[-1]["stage"] == "budget" and emitted[-1]["text"].endswith("· note withheld") and emitted[-1]["decision"]["note_reason"] == ""
    # A read that stands out is sent, with the reason.
    feedback["jev"].ambiguous = 0.85
    result = preflight.on_pre_llm_call(session_id="s1", turn_id="t5", user_message="x", conversation_history=[])
    assert result is not None and "clarifying question" in result["context"]
    assert feedback["records"]("preflight")[-1]["note_reason"] == "ambiguous_above_baseline"
    assert emitted[-1]["text"].endswith("· note sent")
    # Difficulty works the same way for the criteria-first plan.
    feedback["jev"].ambiguous = 0.1
    feedback["jev"].hard = 0.8
    feedback["jev"].checkable = 0.2
    reasons = []
    for turn in range(6, 14):
        preflight.on_pre_llm_call(session_id="s1", turn_id=f"t{turn}", user_message="x", conversation_history=[])
        reasons.append(feedback["records"]("preflight")[-1]["note_reason"])
    assert reasons[0] == "hard_above_baseline", "the first hard turn stands out from an easy session"
    assert reasons[-1] == "" and feedback["records"]("preflight")[-1]["injected"] is False, "hard on every recent turn is the session's usual"
    assert feedback["records"]("preflight")[-1]["baseline_hard"] == pytest.approx(0.8)
    feedback["jev"].hard = 0.99
    result = preflight.on_pre_llm_call(session_id="s1", turn_id="t20", user_message="x", conversation_history=[])
    assert result is not None and "write the acceptance criteria first" in result["context"]
    assert feedback["records"]("preflight")[-1]["note_reason"] == "hard_above_baseline"
    # Candidates are always worth saying.
    feedback["jev"].hard = 0.8
    feedback["jev"].checkable = 0.9
    result = preflight.on_pre_llm_call(session_id="s1", turn_id="t21", user_message="x", conversation_history=[])
    assert result is not None and "3 independent candidate solutions" in result["context"]
    assert feedback["records"]("preflight")[-1]["note_reason"] == "candidates"
    # A new session starts its own baseline.
    feedback["jev"].checkable = 0.2
    assert preflight.on_pre_llm_call(session_id="s2", turn_id="t1", user_message="x", conversation_history=[]) is not None
    assert feedback["records"]("preflight")[-1]["note_reason"] == "first_turns" and feedback["records"]("preflight")[-1]["baseline_n"] == 0
