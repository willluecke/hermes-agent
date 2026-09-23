"""Unified diffs for file-edit tool rows.

Hermes Chat renders a file edit as the diff Claude Code or Codex would print
in a terminal: removed lines red, added lines green, with real line numbers
when the file can be read. The runtimes only know what changed at the tool
boundary, so this module turns what they have (an Edit's old/new strings, a
Write's content, or a Codex change's before/after file lines) into one
bounded unified diff plus exact line counts.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Any, Iterable, Optional

MAX_DIFF_LINES = 400
MAX_DIFF_CHARS = 48_000
_MAX_FILE_BYTES = 4 * 1024 * 1024
_HUNK_RE = re.compile(r"^@@ -(\d+)(,\d+)? \+(\d+)(,\d+)? @@(.*)$")
_TRUNCATED_PREFIX = "\\ Diff truncated:"


def _lines(text: str) -> list[str]:
    if not text:
        return []
    lines = text.replace("\r\n", "\n").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def display_path(path: str, cwd: Optional[str] = None) -> str:
    """Show a path relative to the project when it lives inside it."""
    if not path:
        return ""
    if cwd:
        try:
            return str(Path(path).resolve(strict=False).relative_to(Path(cwd).resolve(strict=False)))
        except (ValueError, OSError, RuntimeError):
            pass
    return path


def _hunks(before: list[str], after: list[str], path: str, *, offset: int = 0) -> list[str]:
    """``---``/``+++`` headers and hunks, with hunk starts shifted by ``offset``."""
    diff = list(
        difflib.unified_diff(
            before, after, fromfile=f"a/{path}", tofile=f"b/{path}", n=3, lineterm=""
        )
    )
    if not offset:
        return diff
    shifted = []
    for line in diff:
        match = _HUNK_RE.match(line)
        if match:
            old_start = int(match.group(1)) + offset if int(match.group(1)) else 0
            new_start = int(match.group(3)) + offset if int(match.group(3)) else 0
            line = (
                f"@@ -{old_start}{match.group(2) or ''} +{new_start}{match.group(4) or ''} @@"
                f"{match.group(5)}"
            )
        shifted.append(line)
    return shifted


def _count(diff: Iterable[str]) -> tuple[int, int]:
    added = removed = 0
    for line in diff:
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
    return added, removed


def _finish(diff: list[str]) -> Optional[dict[str, Any]]:
    """Bound the diff and attach exact counts taken before truncation."""
    if not diff:
        return None
    added, removed = _count(diff)
    kept: list[str] = []
    size = 0
    for index, line in enumerate(diff):
        if len(kept) >= MAX_DIFF_LINES or size + len(line) + 1 > MAX_DIFF_CHARS:
            kept.append(f"{_TRUNCATED_PREFIX} {len(diff) - index} more lines")
            break
        kept.append(line)
        size += len(line) + 1
    return {"diff": "\n".join(kept), "lines_added": added, "lines_removed": removed}


def _read_text(path: str, cwd: Optional[str]) -> Optional[str]:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() and cwd:
        candidate = Path(cwd) / candidate
    try:
        if not candidate.is_file() or candidate.stat().st_size > _MAX_FILE_BYTES:
            return None
        data = candidate.read_bytes()
    except OSError:
        return None
    if b"\0" in data:
        return None
    return data.decode("utf-8", errors="replace")


def _line_offset(file_text: Optional[str], snippet: str) -> int:
    """Zero-based line of ``snippet`` in the edited file, or 0 when unknown."""
    if not file_text or not snippet:
        return 0
    index = file_text.replace("\r\n", "\n").find(snippet.replace("\r\n", "\n"))
    return file_text[:index].count("\n") if index > 0 else 0


def replacement_hunks(
    path: str, old: str, new: str, *, file_text: Optional[str] = None
) -> list[str]:
    return _hunks(_lines(old), _lines(new), path, offset=_line_offset(file_text, new))


def claude_tool_diff(
    raw_name: str, args: dict[str, Any], *, cwd: Optional[str] = None
) -> Optional[dict[str, Any]]:
    """Diff for a completed Claude Code Edit, MultiEdit or Write call."""
    if not isinstance(args, dict):
        return None
    file_path = str(args.get("file_path") or args.get("notebook_path") or "")
    if not file_path:
        return None
    shown = display_path(file_path, cwd)
    if raw_name == "Edit":
        old, new = args.get("old_string"), args.get("new_string")
        if not isinstance(old, str) or not isinstance(new, str):
            return None
        return _finish(
            replacement_hunks(shown, old, new, file_text=_read_text(file_path, cwd))
        )
    if raw_name == "MultiEdit":
        edits = [edit for edit in (args.get("edits") or []) if isinstance(edit, dict)]
        file_text = _read_text(file_path, cwd)
        diff: list[str] = []
        for edit in edits:
            old, new = edit.get("old_string"), edit.get("new_string")
            if not isinstance(old, str) or not isinstance(new, str):
                continue
            hunks = replacement_hunks(shown, old, new, file_text=file_text)
            diff.extend(hunks if not diff else hunks[2:])
        return _finish(diff)
    if raw_name == "Write":
        content = args.get("content")
        if not isinstance(content, str):
            return None
        return _finish(_hunks([], _lines(content), shown))
    return None


def file_change_diff(
    changes: Iterable[tuple[str, Optional[list[bytes]], Optional[list[bytes]]]],
) -> Optional[dict[str, Any]]:
    """Diff for whole-file before/after snapshots (a Codex file change)."""
    diff: list[str] = []
    for path, before, after in changes:
        if before is None or after is None or before == after:
            continue
        diff.extend(
            _hunks(
                [line.decode("utf-8", errors="replace") for line in before],
                [line.decode("utf-8", errors="replace") for line in after],
                path,
            )
        )
    return _finish(diff)
