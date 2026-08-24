#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


HOME = Path(os.environ.get("HOME", "/home/will"))
HERMES_HOME = Path(os.environ.get("HERMES_HOME", HOME / ".hermes"))
SOURCE_DIR = Path(
    os.environ.get(
        "SOURCE_DIR",
        HOME / "src" / "hermes-agent" / "ops" / "command-center",
    )
)
HERMES_BIN = os.environ.get("HERMES_BIN", str(HOME / ".local" / "bin" / "hermes"))
JOBS_PATH = HERMES_HOME / "cron" / "jobs.json"
MODEL = "gpt-5.6-sol"
PROVIDER = "openai-codex"


def load_jobs() -> list[dict]:
    payload = json.loads(JOBS_PATH.read_text(encoding="utf-8"))
    jobs = payload.get("jobs", payload) if isinstance(payload, dict) else payload
    if not isinstance(jobs, list):
        raise RuntimeError("Hermes cron jobs file has an unsupported shape")
    return jobs


def named_job(jobs: list[dict], name: str) -> dict:
    matches = [job for job in jobs if job.get("name") == name]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one cron job named {name!r}, found {len(matches)}")
    return matches[0]


def replace_once(prompt: str, old: str, new: str, job_name: str) -> str:
    if new in prompt:
        return prompt
    if prompt.count(old) != 1:
        raise RuntimeError(
            f"Expected one legacy passage in {job_name!r}, found {prompt.count(old)}"
        )
    return prompt.replace(old, new, 1)


def edit_agent_job(job: dict, prompt: str) -> None:
    subprocess.run(
        [
            HERMES_BIN,
            "cron",
            "edit",
            job["id"],
            "--prompt",
            prompt,
            "--workdir",
            str(HOME),
            "--provider",
            PROVIDER,
            "--model",
            MODEL,
            "--reasoning-effort",
            "xhigh",
            "--agent",
        ],
        check=True,
    )


def main() -> None:
    jobs = load_jobs()

    morning = named_job(jobs, "Daily Founder Revenue Dispatcher")
    morning_prompt = replace_once(
        morning["prompt"],
        "This job runs on the Pi every morning and renders the day-by-day founder calendar.",
        "This job runs on command-center every morning and renders the day-by-day founder calendar.",
        morning["name"],
    )
    edit_agent_job(morning, morning_prompt)

    evening = named_job(jobs, "Daily Founder Evening Review")
    evening_prompt = replace_once(
        evening["prompt"],
        "Runs on the Pi at 9pm PT.",
        "Runs on command-center at 9pm PT.",
        evening["name"],
    )
    edit_agent_job(evening, evening_prompt)

    monthly = named_job(jobs, "Monthly Pipeline Drift Audit")
    monthly_prompt = replace_once(
        monthly["prompt"],
        "You are GPT-5.5; delegate the actual review to Fable via delegate_task.",
        "You are the command-center Codex decision authority; delegate one bounded independent review to the Claude Opus 5 subscription worker through opus_code_worker.",
        monthly["name"],
    )
    monthly_prompt = replace_once(
        monthly_prompt,
        "Pick ONE repo this month, alternating: check /home/will/.hermes/pipeline-audit.json 'last_monthly_repo_audit' — if it was hermes (or null), audit 3DCarParts this month; otherwise audit hermes. For local repos not on the Pi, use the consult-claude channel (~/.hermes/bin/consult-claude) to have Claude Code on Will's Mac do read-only reads; for Pi-resident code read directly.",
        "Pick ONE registered command-center repo this month, alternating: check /home/will/.hermes/pipeline-audit.json 'last_monthly_repo_audit' — if it was hermes-agent (or null), audit 3dcarparts this month; otherwise audit hermes-agent. Load that project's RecCli context before review. Both canonical checkouts are local on command-center; do not depend on the Pi or Mac.",
        monthly["name"],
    )
    monthly_prompt = replace_once(
        monthly_prompt,
        "Give Fable a DIRECTED packet, not 'review everything':",
        "Give the Opus worker a DIRECTED packet, not 'review everything':",
        monthly["name"],
    )
    monthly_prompt = replace_once(
        monthly_prompt,
        "rejections that turned outcome=refuted => Fable false-positive pattern there",
        "rejections that turned outcome=refuted => Opus false-positive pattern there",
        monthly["name"],
    )
    monthly_prompt = replace_once(
        monthly_prompt,
        "one Fable pass over a repo",
        "one Opus worker pass over a repo",
        monthly["name"],
    )
    edit_agent_job(monthly, monthly_prompt)

    auth_script = SOURCE_DIR / "auth-health-check.sh"
    installed_auth_script = HERMES_HOME / "scripts" / "auth-health-check.sh"
    installed_auth_script.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(auth_script, installed_auth_script)
    installed_auth_script.chmod(0o700)

    auth = named_job(jobs, "Weekly Subagent Auth Health Check")
    subprocess.run(
        [
            HERMES_BIN,
            "cron",
            "edit",
            auth["id"],
            "--prompt",
            "Run deterministic command-center subscription authentication checks.",
            "--script",
            "auth-health-check.sh",
            "--no-agent",
            "--workdir",
            str(HOME),
            "--deliver",
            "local",
        ],
        check=True,
    )

    print("Updated command-center founder, audit, and auth-health cron jobs.")


if __name__ == "__main__":
    main()
