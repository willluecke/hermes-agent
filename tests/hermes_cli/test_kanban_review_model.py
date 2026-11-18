"""A kanban card can name its own review model (feature pipeline, 2026-11-18).

Opus builds a region; Astra reviews it. The build run keeps model_override;
a run claimed from review uses review_model_override instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def board(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
    try:
        yield conn
    finally:
        conn.close()


def test_review_model_is_stored_separately_and_cleared_together(board):
    tid = kb.create_task(board, title="Region 1", assignee="default")
    assert kb.set_model_override(board, tid, "claude-opus-5-5", provider="claude-code")
    assert kb.set_review_model_override(board, tid, "gpt-6.2-astra", provider="openai-codex")
    task = kb.get_task(board, tid)
    assert (task.model_override, task.provider_override) == ("claude-opus-5-5", "claude-code")
    assert (task.review_model_override, task.review_provider_override) == ("gpt-6.2-astra", "openai-codex")
    assert kb.set_review_model_override(board, tid, None)
    task = kb.get_task(board, tid)
    assert task.review_model_override is None and task.review_provider_override is None
    assert task.model_override == "claude-opus-5-5", "clearing the review model leaves the build model"
    with pytest.raises(ValueError):
        kb.set_review_model_override(board, tid, None, provider="openai-codex")


def test_a_review_run_is_spawned_with_the_review_model(board):
    tid = kb.create_task(board, title="Region 1", assignee="default")
    kb.set_model_override(board, tid, "claude-opus-5-5", provider="claude-code")
    kb.set_review_model_override(board, tid, "gpt-6.2-astra", provider="openai-codex")
    task = kb.get_task(board, tid)
    review = kb.review_spawn_task(task)
    assert (review.model_override, review.provider_override) == ("gpt-6.2-astra", "openai-codex")
    assert (task.model_override, task.provider_override) == ("claude-opus-5-5", "claude-code"), "the stored task is untouched"
    plain = kb.create_task(board, title="Plain", assignee="default")
    kb.set_model_override(board, plain, "claude-opus-5-5", provider="claude-code")
    same = kb.review_spawn_task(kb.get_task(board, plain))
    assert same.model_override == "claude-opus-5-5", "without a review model the review run uses the build model"


def test_a_card_can_be_created_with_its_review_model(board):
    """Feature pipeline, 2026-11-18: written with the row, so the card is never
    dispatchable without the reviewer it was made for."""
    tid = kb.create_task(board, title="Region 1", assignee="default", model_override="claude-opus-5-5",
                         provider_override="claude-code", review_model_override="gpt-6.2-astra",
                         review_provider_override="openai-codex")
    task = kb.get_task(board, tid)
    assert (task.review_model_override, task.review_provider_override) == ("gpt-6.2-astra", "openai-codex")
    with pytest.raises(ValueError):
        kb.create_task(board, title="Bad", assignee="default", review_provider_override="openai-codex")

