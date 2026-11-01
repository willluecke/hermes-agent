"""The one thing the Talos steward never does: delete a project.

Will, 2026-10-28: "The main talos agent shouldn't have any guard rails except
for project deletion, that's the only thing it can't do." Every other steward
fence was removed that day; this module is the single remaining refusal and
is consulted from three places:

* ``StewardTools.check`` (the scoped hermes-tools MCP server): a tool action
  that would delete a registered project (by id or by path) is refused.
* The reversible-delete shim (``tools.reversible_deletion``): ``rm``, ``rmdir``
  and ``unlink`` operands that name a registered project root, or a directory
  above one, are refused through ``ReversibleDeletionPolicy.protected_roots``.
* Codex exec approvals for the steward: a command that removes a project root
  with any tool the shim does not cover is cancelled.

A project is anything registered in ``~/.hermes/subscription-projects.json``
or in Talos's ledger ``projects`` table. The Talos ledger itself refuses
``DELETE FROM projects`` with a trigger, and no Talos route deletes one.
Descendants of a project root (its files, worktrees, caches) are ordinary
deletions; only the root and its ancestors are protected.

Will, 2026-11-01: deleting a kanban board is project deletion too. A board
root is ``<kanban home>/kanban/boards/<slug>`` (archived copies under
``boards/_archived`` included) or the default board's ``<kanban home>/
kanban.db``; ``board_roots`` lists them and ``project_roots`` carries them to
every consumer above. ``hermes kanban boards rm <slug> --delete`` (and the
``boards delete`` alias, which the CLI treats as a hard delete) is refused
in any argument order and in the ``/kanban`` slash form; ``hermes kanban
boards rm <slug>`` without ``--delete`` archives the board and stays allowed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import sqlite3

REFUSAL = "Project deletion is the one thing Talos never does"
REGISTRY = Path.home() / ".hermes" / "subscription-projects.json"
LEDGER = Path.home() / "coding-projects" / "talos" / ".talos" / "runtime" / "ledger.sqlite3"
REMOVERS = {"rm", "rmdir", "unlink", "shred", "srm", "wipe"}
DELETING_ACTION = re.compile(r"delete|remove|destroy|drop|purge|erase", re.I)
BOARD_REMOVE_ACTIONS = {"rm", "remove", "delete"}
SEGMENT_PREFIXES = {"sudo", "env", "command", "exec", "nice", "nohup", "time"}


def _registry_path():
    return Path(os.environ.get("HERMES_PROJECT_REGISTRY") or REGISTRY)


def _ledger_path():
    return Path(os.environ.get("HERMES_TALOS_LEDGER") or LEDGER)


def _kanban_home():
    """The kanban root, as hermes_cli.kanban_db resolves it (HERMES_KANBAN_HOME, then HERMES_HOME)."""
    try:
        from hermes_cli.kanban_db import kanban_home
        return Path(kanban_home())
    except Exception:
        override = os.environ.get("HERMES_KANBAN_HOME", "").strip()
        if override:
            return Path(override).expanduser()
        return Path(os.environ.get("HERMES_HOME", "").strip() or Path.home() / ".hermes").expanduser()


def board_roots(kanban_home=None):
    """Realpaths of every kanban board: ``boards/<slug>`` dirs (archived copies too) and the default ``kanban.db``."""
    home = Path(kanban_home) if kanban_home else _kanban_home()
    roots = set()
    default_db = home / "kanban.db"
    if default_db.is_file():
        roots.add(os.path.realpath(default_db))
    boards = home / "kanban" / "boards"
    try:
        entries = list(boards.iterdir())
    except OSError:
        entries = []
    for entry in entries:
        if not entry.is_dir():
            continue
        if entry.name == "_archived":
            try:
                roots.update(os.path.realpath(old) for old in entry.iterdir() if old.is_dir())
            except OSError:
                pass
            continue
        roots.add(os.path.realpath(entry))
    return tuple(sorted(roots))


def registered_projects(registry=None, ledger=None):
    """{project id: realpath of its root} from the registry and the Talos ledger."""
    projects = {}
    registry = Path(registry) if registry else _registry_path()
    try:
        raw = json.loads(registry.read_text())
    except (OSError, ValueError):
        raw = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            if isinstance(key, str) and isinstance(value, str) and value.strip():
                projects[key] = os.path.realpath(os.path.abspath(value.strip()))
    ledger = Path(ledger) if ledger else _ledger_path()
    if ledger.is_file():
        try:
            db = sqlite3.connect(ledger.resolve().as_uri() + "?mode=ro", uri=True, timeout=0.5)
            try:
                db.execute("PRAGMA query_only=ON")
                for id, repo in db.execute("SELECT id, repo FROM projects"):
                    if isinstance(id, str) and isinstance(repo, str) and repo.strip():
                        projects.setdefault(id, os.path.realpath(os.path.abspath(repo.strip())))
            finally:
                db.close()
        except sqlite3.Error:
            pass
    return projects


def project_roots(registry=None, ledger=None):
    """Every protected root: registered project roots plus kanban board roots."""
    return tuple(sorted(set(registered_projects(registry, ledger).values()) | set(board_roots())))


def _resolve(target, cwd):
    absolute = os.path.abspath(target if os.path.isabs(target) else os.path.join(cwd or os.getcwd(), target))
    parent = os.path.realpath(os.path.dirname(absolute))
    return os.path.join(parent, os.path.basename(absolute))


def deletes_project_root(target, roots, cwd=None):
    """True when `target` is a registered project root or a directory above one."""
    if not isinstance(target, str) or not target.strip():
        return False
    path = _resolve(os.path.expanduser(target.strip()), cwd)
    for root in roots:
        root = os.path.realpath(root)
        if path == root or root.startswith(path.rstrip(os.sep) + os.sep):
            return True
    return False


def refuse_tool(name, args, registry=None, ledger=None):
    """A refusal for a tool action that would delete a project, else None.

    Board roots count as project roots for the path check. No ``kanban_*``
    tool can remove a board (they show, list, create, complete, block, review,
    comment, attach and link tasks; board removal exists only in the CLI), so
    there is no board tool to refuse here.
    """
    if not isinstance(args, dict):
        return None
    action = str(args.get("action") or args.get("verb") or "")
    if not DELETING_ACTION.search(action) and not DELETING_ACTION.search(str(name or "")):
        return None
    projects = registered_projects(registry, ledger)
    roots = tuple(projects.values()) + board_roots()
    values = [v for v in args.values() if isinstance(v, str)]
    if any(v in projects for v in values if v not in {action}) or any(deletes_project_root(v, roots) for v in values if v.startswith(("/", "~"))):
        return f"{REFUSAL}: {name} {action} names a registered project"
    if "project" in action.lower():
        return f"{REFUSAL}: {name} {action}"
    return None


def hard_deletes_board(words):
    """True for ``hermes kanban [--board X] boards rm|remove|delete <slug> --delete`` or ``boards delete <slug>``.

    Also matches the ``/kanban …`` slash form. Any argument order after
    ``boards`` counts; without ``--delete`` the board is archived, which is
    allowed.
    """
    if not words:
        return False
    program = os.path.basename(words[0])
    if program == "hermes":
        rest = words[1:]
        if not rest or rest[0] != "kanban":
            return False
        rest = rest[1:]
    elif words[0] == "/kanban":
        rest = words[1:]
    else:
        return False
    if "boards" not in rest:
        return False
    after = rest[rest.index("boards") + 1:]
    action = next((w for w in after if not w.startswith("-")), None)
    if action not in BOARD_REMOVE_ACTIONS:
        return False
    return action == "delete" or "--delete" in after


def _segments(text):
    try:
        tokens = shlex.split(text, posix=True)
    except ValueError:
        tokens = text.split()
    segments, current = [], []
    for token in tokens:
        if token in {"&&", "||", ";", "|", "&"}:
            segments.append(current); current = []
        else:
            current.append(token)
    segments.append(current)
    return [[w for w in segment if w not in SEGMENT_PREFIXES] for segment in segments]


def refuse_command(command, cwd=None, registry=None, ledger=None):
    """A refusal for a shell command that removes a project root, else None.

    Scans each top-level segment for removal programs (rm, rmdir, unlink,
    shred, srm, wipe, `find … -delete`, `git worktree remove|prune`) and
    interpreter one-liners mentioning rmtree/rimraf/fs.rm/
    os.remove, and refuses when an operand resolves to a registered project
    root or an ancestor of one. A kanban board is a project: the hard-delete
    CLI form (``hermes kanban boards rm <slug> --delete``) is refused
    outright, and board directories are protected roots like any other.
    """
    text = command if isinstance(command, str) else " ".join(str(part) for part in (command or []))
    if not text.strip():
        return None
    segments = [segment for segment in _segments(text) if segment]
    for words in segments:
        if hard_deletes_board(words):
            return f"{REFUSAL}: `{text[:160]}` would delete a kanban board (archive it instead)"
    roots = project_roots(registry, ledger)
    if not roots:
        return None
    for words in segments:
        program = os.path.basename(words[0])
        if program == "git" and len(words) > 2 and words[1] == "-C":
            words = [words[0], *words[3:], words[2]]  # `git -C root worktree remove root`: the -C path is an operand too
        operands = [w for w in words[1:] if not w.startswith("-")]
        removal = program in REMOVERS
        if program == "find" and ("-delete" in words or "-exec" in words and any(os.path.basename(w) in REMOVERS for w in words)):
            removal = True
        if program == "git" and len(words) > 2 and words[1] == "worktree" and words[2] in {"remove", "prune"}:
            removal = True
        if program in {"python", "python3", "node", "perl", "ruby", "sh", "bash", "zsh"} and re.search(r"rmtree|rimraf|fs\.rm|os\.remove|os\.unlink|os\.rmdir|\brm -", " ".join(words[1:])):
            removal = True
            operands = re.findall(r"(/[^\s'\"),;]+|~/[^\s'\"),;]+)", " ".join(words[1:])) or operands
        if removal and any(deletes_project_root(operand, roots, cwd) for operand in operands):
            return f"{REFUSAL}: `{text[:160]}` would remove a registered project"
    return None
