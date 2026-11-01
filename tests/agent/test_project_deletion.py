"""Project deletion is the one thing the Talos steward never does (Will, 2026-10-28).

Since 2026-11-01 a kanban board is a project too: hard-deleting one is refused,
archiving stays allowed.
"""
import json
import os
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
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "no-kanban"))  # never the real ~/.hermes
    return home


@pytest.fixture
def boards(projects, monkeypatch):
    """A kanban home with the default board, two named boards and one archived board."""
    kanban = projects / ".hermes"
    (kanban / "kanban.db").write_bytes(b"")
    for slug in ("demo", "reg-watch-board"):
        (kanban / "kanban" / "boards" / slug / "workspaces" / "t_1").mkdir(parents=True)
        (kanban / "kanban" / "boards" / slug / "board.json").write_text(json.dumps({"slug": slug}))
        (kanban / "kanban" / "boards" / slug / "kanban.db").write_bytes(b"")
    (kanban / "kanban" / "boards" / "_archived" / "old-1700000000").mkdir(parents=True)
    (kanban / "kanban" / "boards" / "stray.txt").write_text("not a board")
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(kanban))
    return kanban


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
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "absent-kanban"))
    assert guard.registered_projects() == {}
    assert guard.project_roots() == ()
    assert guard.refuse_command("rm -rf /home/will/coding-projects/talos") is None
    assert guard.refuse_tool("talos", {"action": "dispatch"}) is None


def test_board_roots_are_every_board_directory_and_the_default_db(boards):
    roots = guard.board_roots()
    expected = {os.path.realpath(boards / "kanban.db"), os.path.realpath(boards / "kanban" / "boards" / "demo"),
                os.path.realpath(boards / "kanban" / "boards" / "reg-watch-board"),
                os.path.realpath(boards / "kanban" / "boards" / "_archived" / "old-1700000000")}
    assert set(roots) == expected                         # stray.txt and _archived itself are not boards
    assert set(guard.project_roots()) == expected | set(guard.registered_projects().values())
    assert guard.board_roots(kanban_home=boards) == roots


def test_hard_deleting_a_board_is_refused_in_every_form(boards, projects):
    demo = str(boards / "kanban" / "boards" / "demo")
    for command in ["hermes kanban boards rm demo --delete", "hermes kanban boards rm --delete demo",
                    "/kanban boards rm demo --delete", "hermes kanban --board demo boards rm demo --delete",
                    "hermes kanban boards remove demo --delete", "hermes kanban boards delete demo",
                    "cd /tmp && hermes kanban boards rm demo --delete", "sudo /usr/local/bin/hermes kanban boards rm demo --delete",
                    f"rm -rf {demo}", f"rm -r {demo}/", f"rmdir {demo}", f"rm -rf {boards}/kanban/boards", f"rm -rf {boards}/kanban",
                    f"rm {boards}/kanban.db", f"rm -rf {boards}", f"find {demo} -delete",
                    f"python3 -c 'import shutil; shutil.rmtree(\"{demo}\")'", f"rm -rf {boards}/kanban/boards/_archived/old-1700000000"]:
        reason = guard.refuse_command(command, cwd=str(projects))
        assert reason and reason.startswith(guard.REFUSAL), command
    assert "archive" in guard.refuse_command("hermes kanban boards rm demo --delete")
    assert guard.refuse_tool("terminal", {"action": "delete", "path": demo}).startswith(guard.REFUSAL)
    with pytest.raises(ValueError, match=guard.REFUSAL):
        StewardTools().check("terminal", {"action": "delete", "path": demo})


def test_archiving_a_board_and_deleting_inside_one_are_allowed(boards, projects):
    demo = str(boards / "kanban" / "boards" / "demo")
    for command in ["hermes kanban boards rm demo", "hermes kanban boards remove demo", "/kanban boards rm demo",
                    "hermes kanban --board demo boards rm demo", "hermes kanban boards list", "hermes kanban boards create demo2",
                    "hermes kanban rm t_123 --delete", "hermes kanban boards switch demo", "hermes boards rm demo --delete",
                    f"rm -rf {demo}/workspaces/t_1", f"rm {demo}/board.json", f"rm -rf {boards}/kanban/boards/_archived/old-1700000000/x",
                    f"rm -rf {boards}/kanban/workspaces", f"git -C {demo}/workspaces/t_1 clean -fdx", "rm -rf /tmp/demo"]:
        assert guard.refuse_command(command, cwd=str(projects)) is None, command
    assert not guard.hard_deletes_board([]) and not guard.hard_deletes_board(["hermes"]) and not guard.hard_deletes_board(["hermes", "kanban"])


def test_a_missing_kanban_home_protects_nothing_extra(projects, monkeypatch):
    absent = projects / "absent-kanban"
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(absent))
    assert guard.board_roots() == ()
    assert guard.refuse_command(f"rm -rf {absent}/kanban/boards/demo", cwd=str(projects)) is None
    assert guard.refuse_command(f"rm {absent}/kanban.db", cwd=str(projects)) is None
    assert guard.refuse_command("hermes kanban boards rm demo --delete")  # the CLI form never depends on what is on disk


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


def test_the_delete_shim_refuses_board_roots_through_project_roots(boards, projects, monkeypatch):
    from dataclasses import replace
    from tools.reversible_deletion import ReversibleDeletionPolicy, UnsafeDeleteTarget, validate_delete_target
    monkeypatch.setenv("HERMES_HOME", str(projects / ".hermes"))
    policy = replace(ReversibleDeletionPolicy(enabled=True), protected_roots=guard.project_roots())
    demo = str(boards / "kanban" / "boards" / "demo")
    for target in (demo, str(boards / "kanban" / "boards"), str(boards / "kanban.db")):
        with pytest.raises(UnsafeDeleteTarget, match="registered projects are never deleted"):
            validate_delete_target(target, cwd=str(projects), workspace_root=str(projects), policy=policy)
    assert validate_delete_target(demo + "/workspaces/t_1", cwd=str(projects), workspace_root=str(projects), policy=policy) == demo + "/workspaces/t_1"
