from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).parents[2]
    / "ops"
    / "command-center"
    / "install-command-center-crons.py"
)


def load_installer():
    spec = importlib.util.spec_from_file_location("install_command_center_crons", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def legacy_jobs() -> list[dict]:
    return [
        {
            "id": "morning",
            "name": "Daily Founder Revenue Dispatcher",
            "prompt": (
                "This job runs on the Pi every morning and renders the day-by-day "
                "founder calendar."
            ),
        },
        {
            "id": "evening",
            "name": "Daily Founder Evening Review",
            "prompt": "Runs on the Pi at 9pm PT.",
        },
        {
            "id": "monthly",
            "name": "Monthly Pipeline Drift Audit",
            "prompt": "\n".join(
                [
                    "You are GPT-5.5; delegate the actual review to Fable via delegate_task.",
                    (
                        "Pick ONE repo this month, alternating: check "
                        "/home/will/.hermes/pipeline-audit.json 'last_monthly_repo_audit' "
                        "— if it was hermes (or null), audit 3DCarParts this month; "
                        "otherwise audit hermes. For local repos not on the Pi, use the "
                        "consult-claude channel (~/.hermes/bin/consult-claude) to have "
                        "Claude Code on Will's Mac do read-only reads; for Pi-resident "
                        "code read directly."
                    ),
                    "Give Fable a DIRECTED packet, not 'review everything':",
                ]
            ),
        },
        {
            "id": "auth",
            "name": "Weekly Subagent Auth Health Check",
            "prompt": "legacy",
        },
    ]


def test_installer_rewrites_jobs_and_makes_auth_check_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installer = load_installer()
    hermes_home = tmp_path / ".hermes"
    jobs_path = hermes_home / "cron" / "jobs.json"
    jobs_path.parent.mkdir(parents=True)
    jobs_path.write_text(json.dumps({"jobs": legacy_jobs()}), encoding="utf-8")
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    auth_source = source_dir / "auth-health-check.sh"
    auth_source.write_text("#!/usr/bin/env bash\nprintf '[SILENT]\\n'\n", encoding="utf-8")

    calls: list[list[str]] = []

    def record_run(args, check):
        assert check is True
        calls.append(args)

    monkeypatch.setattr(installer, "HOME", tmp_path)
    monkeypatch.setattr(installer, "HERMES_HOME", hermes_home)
    monkeypatch.setattr(installer, "JOBS_PATH", jobs_path)
    monkeypatch.setattr(installer, "SOURCE_DIR", source_dir)
    monkeypatch.setattr(installer, "HERMES_BIN", "/test/hermes")
    monkeypatch.setattr(installer.subprocess, "run", record_run)

    installer.main()

    assert [call[3] for call in calls] == ["morning", "evening", "monthly", "auth"]
    for call in calls[:3]:
        assert call[-7:] == [
            "--provider",
            "openai-codex",
            "--model",
            "gpt-5.6-sol",
            "--reasoning-effort",
            "xhigh",
            "--agent",
        ]
    monthly_prompt = calls[2][calls[2].index("--prompt") + 1]
    assert "Claude Opus 5 subscription worker" in monthly_prompt
    assert "Both canonical checkouts are local on command-center" in monthly_prompt
    assert "Fable" not in monthly_prompt
    assert "consult-claude" not in monthly_prompt
    assert calls[3][-5:] == [
        "--no-agent",
        "--workdir",
        str(tmp_path),
        "--deliver",
        "local",
    ]

    installed_auth = hermes_home / "scripts" / "auth-health-check.sh"
    assert installed_auth.read_text(encoding="utf-8") == auth_source.read_text(
        encoding="utf-8"
    )
    assert installed_auth.stat().st_mode & 0o777 == 0o700


def test_replace_once_is_idempotent_and_rejects_ambiguous_legacy_text() -> None:
    installer = load_installer()

    assert installer.replace_once("new", "old", "new", "job") == "new"
    with pytest.raises(RuntimeError, match="found 2"):
        installer.replace_once("old old", "old", "new", "job")
