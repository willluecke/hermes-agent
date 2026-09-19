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
    settings = {
        "mode": "shadow", "threshold": 0.7, "log_path": str(tmp_path / "preflight.jsonl"), "tool_guard": "shadow",
        # Self-tuning is off unless a test turns it on, and its state never comes from the real home.
        "tuning": "off", "tuning_state_path": str(tmp_path / "tuning.json"),
    }
    monkeypatch.setattr(preflight, "_settings_reader", lambda key, default=None: settings.get(key, default))
    jev = _FakeJev()
    monkeypatch.setattr(preflight, "_ask", jev)
    preflight._last_tune_check = 0.0
    preflight._tuned_cache.update(mtime=None, path=None, data={})
    preflight._turn_memo.clear()
    preflight._session_scope.clear()
    preflight._session_todos.clear()
    preflight._session_checks.clear()
    preflight._session_commands.clear()
    preflight._verify_memo.clear()
    preflight._held_actions.clear()

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
    assert set(manager._hooks) == {"pre_llm_call", "post_llm_call", "pre_tool_call", "post_tool_call", "pre_verify"}
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
    assert event["decision"] == {"k": 1, "finish_loop": True, "plan": "direct", "injected": False}
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
    assert event["decision"] == {"k": 3, "finish_loop": True, "plan": "candidates", "injected": True}
    # Hard but not checkable: criteria first, one candidate, no finish loop.
    feedback["jev"].checkable = 0.2
    result = preflight.on_pre_llm_call(session_id="s1", turn_id="t2", user_message="write a poem about the importer", conversation_history=[])
    assert "write the acceptance criteria first" in result["context"]
    assert feedback["records"]("preflight")[-1]["k"] == 1
    assert emitted[-1]["decision"]["plan"] == "criteria_only" and emitted[-1]["decision"]["finish_loop"] is False


def test_feedback_mode_feeds_back_when_jev_is_unsure_but_no_longer_on_missing_evidence_alone(feedback):
    feedback["jev"].ambiguous = 0.5  # unsure band
    assert preflight.on_pre_llm_call(session_id="s1", turn_id="t1", user_message="x", conversation_history=[]) is not None
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


def test_post_tool_call_remembers_todos_and_check_outputs(feedback):
    preflight.on_post_tool_call(tool_name="todo", args={}, result=TODOS_RESULT, session_id="s1")
    assert [item["content"] for item in preflight._session_todos["s1"]] == ["Greeting returns hello world", "Errors are logged", "Docs updated"]
    preflight.on_post_tool_call(tool_name="terminal", args={"command": "npm test"}, result="x" * 5000 + "\n1 failing", session_id="s1")
    preflight.on_post_tool_call(tool_name="terminal", args={"command": "ls -la"}, result="files", session_id="s1")
    checks = preflight._session_checks["s1"]
    assert len(checks) == 1 and checks[0]["command"] == "npm test"
    assert checks[0]["output"].endswith("1 failing") and len(checks[0]["output"]) <= preflight.MAX_CHECK_CHARS


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
    assert "Check outputs show a failure" in message
    call = feedback["jev"].calls[-1]
    state = call["state"]
    assert "+    return 'hello world'" in state["diff"]["text"]
    assert "new file: new_module.py" in state["diff"]["text"]
    assert state["features"]["items"][0]["title"] == "Greeting"
    assert [item["text"] for item in state["acceptance_criteria"]["items"]] == ["Greeting returns hello world", "Errors are logged"]
    assert state["still_pending_todos"]["items"] == ["Docs updated"]
    assert state["check_outputs"]["items"][0]["command"] == "pytest -q"
    assert state["final_message"]["text"] == "Done, everything passes."
    assert set(call["questions"]) == {"criterion_1", "criterion_2", "checks_failing", "claims_unverified"}
    assert call["questions"]["criterion_2"]["criteria"] == preflight.CRITERION_CRITERIA, "nouls carry explicit boundaries"
    assert call["questions"]["claims_unverified"]["criteria"] == preflight.CLAIMS_CRITERIA
    assert call["timeout"] == preflight.VERIFY_TIMEOUT_SECONDS
    [record] = feedback["records"]("verify")
    assert record["criteria"] == 2 and record["pending"] == 1 and record["features"] == 1
    [event] = emitted
    assert event["event"] == "judge.verdict" and event["stage"] == "verify" and event["attempt"] == 0
    assert event["decision"]["action"] == "nudge" and event["decision"]["criteria"] == 2 and event["decision"]["pending"] == 1
    assert event["answers"]["Errors are logged"] == 0.05 and event["answers"]["Greeting returns hello world"] == 0.95
    assert event["answers"]["claims_unverified"] == 0.1 and event["answers"]["checks_failing"] == 0.9
    assert event["text"].startswith("Jev verify (attempt 1): criteria met 1/2") and "Errors are logged" in event["text"]
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


def test_every_command_is_evidence_and_checks_are_the_test_subset(feedback, repo):
    preflight.on_post_tool_call(tool_name="terminal", args={"command": "python3 hello.py"}, result="hello world", session_id="s1")
    preflight.on_post_tool_call(tool_name="terminal", args={"command": "pytest -q"}, result="3 passed", session_id="s1")
    feedback["jev"].guard = {"criterion_1": 0.9, "checks_failing": 0.05, "claims_unverified": 0.05}
    preflight.on_pre_llm_call(session_id="s1", turn_id="t1", user_message="make hello.py print hello world", conversation_history=[])
    assert preflight.on_pre_verify(session_id="s1", attempt=0, final_response="Ran it, prints hello world.", changed_paths=[str(repo / "app.py")]) is None
    state = feedback["jev"].calls[-1]["state"]
    assert [item["command"] for item in state["commands_run"]["items"]] == ["python3 hello.py", "pytest -q"]
    assert [item["command"] for item in state["check_outputs"]["items"]] == ["pytest -q"]
    assert state["commands_run"]["items"][0]["output"] == "hello world"


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
    assert result is not None and "claims results the evidence does not show (P=0.60)" in result["message"]
