"""Project deletion is the one thing the Talos steward never does (Will, 2026-10-28)."""
import json
import sqlite3

import pytest

from agent import project_deletion as guard
from agent.steward_policy import StewardTools


@pytest.fixture
def projects(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "coding-projects" / "reg-watch" / "src").mkdir(parents=True)
    (home / "coding-projects" / "talos").mkdir(parents=True)
    (home / ".hermes" / "talos-steward").mkdir(parents=True)
    registry = tmp_path / "subscription-projects.json"
    registry.write_text(json.dumps({"reg-watch": str(home / "coding-projects" / "reg-watch"),
                                    "talos-steward": str(home / ".hermes" / "talos-steward"),
                                    "command-center": str(home)}))
    ledger = tmp_path / "ledger.sqlite3"
    db = sqlite3.connect(ledger)
    db.execute("CREATE TABLE projects (id TEXT PRIMARY KEY, repo TEXT NOT NULL)")
    db.execute("INSERT INTO projects VALUES ('talos', ?)", (str(home / "coding-projects" / "talos"),))
    db.commit(); db.close()
    monkeypatch.setenv("HERMES_PROJECT_REGISTRY", str(registry))
    monkeypatch.setenv("HERMES_TALOS_LEDGER", str(ledger))
    return home


def test_registered_projects_come_from_the_registry_and_the_ledger(projects):
    registered = guard.registered_projects()
    assert set(registered) == {"reg-watch", "talos-steward", "command-center", "talos"}
    assert registered["talos"].endswith("coding-projects/talos")
    assert guard.project_roots() == tuple(sorted(set(registered.values())))


def test_shell_commands_that_remove_a_project_root_or_its_parent_are_refused(projects):
    talos = str(projects / "coding-projects" / "talos")
    for command in [f"rm -rf {talos}", f"rm -r {talos}/", f"cd /tmp && rm -rf {projects}/coding-projects", f"rmdir {talos}",
                    f"sudo rm -rf {projects}", f"find {talos} -delete", f"git worktree remove {talos}", f"git -C {talos} worktree remove {talos}",
                    f"python3 -c 'import shutil; shutil.rmtree(\"{talos}\")'", "rm -rf ~/coding-projects/talos"]:
        reason = guard.refuse_command(command.replace("~", str(projects)), cwd=str(projects))
        assert reason and reason.startswith(guard.REFUSAL), command
    assert guard.refuse_command("rm -rf /", cwd=str(projects))  # above every root


def test_ordinary_deletions_and_other_commands_are_allowed(projects):
    talos = str(projects / "coding-projects" / "talos")
    for command in [f"rm -rf {talos}/.pytest-cache", f"rm {talos}/build/out.txt", "rm -rf /tmp/scratch", f"git -C {talos} status",
                    f"python3 -m pytest -q {talos}", f"ls {talos}", "brenuc-preview restart talos", f"git -C {talos} clean -fdx", f"rm -rf {projects}/.hermes/talos-steward/notes.md",
                    "git worktree remove /tmp/talos-base", ""]:
        assert guard.refuse_command(command, cwd=str(projects)) is None, command


def test_tool_actions_that_delete_a_project_are_refused_and_nothing_else_is(projects):
    talos = str(projects / "coding-projects" / "talos")
    for name, args in [("talos", {"action": "project_delete", "id": "reg-watch"}), ("talos", {"action": "delete", "project_id": "talos"}),
                       ("hermes_conversation", {"action": "remove", "project": "reg-watch"}), ("terminal", {"action": "delete", "path": talos}),
                       ("management", {"action": "drop_project"})]:
        assert guard.refuse_tool(name, args) and guard.refuse_tool(name, args).startswith(guard.REFUSAL), (name, args)
        with pytest.raises(ValueError, match=guard.REFUSAL):
            StewardTools().check(name, args)
    policy = StewardTools()
    for name, args in [("talos", {"action": "dispatch"}), ("talos", {"action": "admit", "project_id": "reg-watch"}), ("talos", {"action": "hold", "verb": "resume"}),
                       ("talos", {"action": "grant_revoke", "id": "grant_1"}), ("talos", {"action": "routine_create", "owner": "agent"}),
                       ("hermes_conversation", {"action": "send", "id": "delegate"}), ("hermes_conversation", {"action": "goal", "verb": "set"}),
                       ("hermes_conversation", {"action": "set_runtime", "id": "delegate"}), ("steward_jobs", {"action": "job_save"}),
                       ("terminal", {"command": f"rm -rf {talos}/.pytest-cache"}), ("delegate_task", {}), ("web_search", {"query": "x"}),
                       ("talos", {"action": "delegation", "id": "one"}), ("talos", {"action": "delegation", "id": "one"})]:
        policy.check(name, args)                       # no allowlist, no per-turn fences
        policy.check(name, args)                       # repeated reads are fine
    policy.completed("talos", {"action": "dispatch"}, "{}")
    policy.check("talos", {"action": "dispatch"})      # a second dispatch in one turn is allowed
    with pytest.raises(ValueError, match="review"):
        policy.check("talos", {"action": "review"})    # review actions stay gateway-bound


def test_descendants_of_a_project_root_are_not_protected(projects):
    roots = guard.project_roots()
    talos = str(projects / "coding-projects" / "talos")
    assert guard.deletes_project_root(talos, roots)
    assert guard.deletes_project_root(str(projects / "coding-projects"), roots)
    assert not guard.deletes_project_root(talos + "/src/x.py", roots)
    assert not guard.deletes_project_root("/tmp/elsewhere", roots)
    assert not guard.deletes_project_root("", roots) and not guard.deletes_project_root(None, roots)


def test_missing_registry_and_ledger_mean_nothing_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_PROJECT_REGISTRY", str(tmp_path / "absent.json"))
    monkeypatch.setenv("HERMES_TALOS_LEDGER", str(tmp_path / "absent.sqlite3"))
    assert guard.registered_projects() == {}
    assert guard.refuse_command("rm -rf /home/will/coding-projects/talos") is None
    assert guard.refuse_tool("talos", {"action": "dispatch"}) is None


def test_the_delete_shim_refuses_project_roots_and_still_trashes_inside_them(projects, monkeypatch):
    from dataclasses import replace
    from tools.reversible_deletion import ReversibleDeletionPolicy, UnsafeDeleteTarget, validate_delete_target, validate_expanded_delete_target
    monkeypatch.setenv("HERMES_HOME", str(projects / ".hermes"))
    policy = replace(ReversibleDeletionPolicy(enabled=True), protected_roots=guard.project_roots())
    talos = str(projects / "coding-projects" / "talos")
    for validate in (validate_delete_target, validate_expanded_delete_target):
        for target in (talos, str(projects / "coding-projects"), str(projects)):
            with pytest.raises(UnsafeDeleteTarget, match="registered projects are never deleted"):
                validate(target, cwd=str(projects), workspace_root=str(projects), policy=policy)
        assert validate(talos + "/.pytest-cache", cwd=str(projects), workspace_root=str(projects), policy=policy) == talos + "/.pytest-cache"
    unprotected = ReversibleDeletionPolicy(enabled=True)
    assert validate_delete_target(talos, cwd=str(projects), workspace_root=str(projects), policy=unprotected) == talos
