"""Project instruction files: `LUCA.md`, `AGENTS.md`, `CLAUDE.md`.

Three tiers, concatenated least specific first so the file nearest the
workspace is read last and wins by recency:

1. one file from the luca config directory (personal, machine-wide)
2. one file per directory from the git root down to the workspace
3. whatever the config's `instructions` list names

Name precedence is applied PER DIRECTORY, not once for the whole tree, so a
repo root's `AGENTS.md` and a subpackage's `CLAUDE.md` both contribute. It also
means the common `CLAUDE.md` that contains nothing but `@AGENTS.md` is never
what gets read: `AGENTS.md` is checked first in that same directory.

DISCOVERED files are lenient — an unreadable or empty one is skipped and the
rest still load. A file the caller NAMED is not: a path in the config's
`instructions` that does not resolve to a readable file raises
`InstructionsError`, so a typo there fails loudly instead of silently
contributing nothing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

INSTRUCTION_FILE_NAMES = ("LUCA.md", "AGENTS.md", "CLAUDE.md")

# This text rides on every request of every conversation in the session,
# subagents included, so it gets a budget.
MAX_INSTRUCTION_BYTES = 32 * 1024


class InstructionsError(Exception):
    """A named instruction file that is missing, unreadable or not a file."""


@dataclass(frozen=True)
class InstructionFile:
    path: Path
    text: str


def get_config_directory() -> Path:
    """`$XDG_CONFIG_HOME/luca` or `~/.config/luca` — the same directory
    `luca.json` is read from, reimplemented here because contrib packages do
    not import from the TUI."""
    base = os.environ.get("XDG_CONFIG_HOME")
    return (Path(base) if base else Path.home() / ".config") / "luca"


def read_instruction_file(path: Path) -> InstructionFile | None:
    """The file's contents, or `None` when there is nothing usable there.
    Lenient by design — `read_named_instruction_file` is the strict door."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    return InstructionFile(path=path, text=text.strip()) if text.strip() else None


def read_named_instruction_file(path: Path) -> InstructionFile:
    """The same read for a path the caller NAMED, where every failure is the
    caller's mistake and has to surface. `is_file()`, so a directory fails
    rather than being read as nothing."""
    if not path.is_file():
        raise InstructionsError(f"{path}: not a readable instruction file")
    found = read_instruction_file(path)
    if found is None:
        raise InstructionsError(f"{path}: unreadable or empty")
    return found


def find_instruction_file(directory: Path) -> InstructionFile | None:
    """The one instruction file this directory contributes, by name
    precedence, or `None` when it holds none of them."""
    for name in INSTRUCTION_FILE_NAMES:
        found = read_instruction_file(directory / name)
        if found is not None:
            return found
    return None


def find_project_directories(workspace: str | os.PathLike[str]) -> list[Path]:
    """The git root down to the workspace, inclusive, outermost first.

    Without a repository the workspace alone: an unbounded upward walk reaches
    in from outside the project, and `.resolve()` matters because a comparison
    against an unresolved path silently never fires when a symlink is in play
    (`/tmp` vs `/private/tmp` on macOS)."""
    workspace = Path(workspace).resolve()
    chain: list[Path] = []
    for directory in (workspace, *workspace.parents):
        chain.append(directory)
        if (directory / ".git").exists():  # a worktree's `.git` is a file
            return list(reversed(chain))
    return [workspace]


def apply_budget(files: list[InstructionFile], max_bytes: int) -> list[InstructionFile]:
    """Trim to `max_bytes`, dropping the LEAST specific files first.

    Codex fills its 32 KiB budget in lookup order instead, so a bloated
    personal file starves the repo's own rules. The most specific file is
    always kept, however large, since dropping it silently is worse than
    spending the budget on it."""
    kept: list[InstructionFile] = []
    used = 0
    for file in reversed(files):
        size = len(file.text.encode("utf-8"))
        if kept and used + size > max_bytes:
            break
        kept.append(file)
        used += size
    return list(reversed(kept))


def find_instructions(
    workspace: str | os.PathLike[str] = ".",
    extra: list[str] | None = None,
    *,
    config_dir: Path | None = None,
) -> list[InstructionFile]:
    """Every instruction file that applies, least specific first. Relative
    `extra` entries resolve against the workspace, and each one must exist."""
    workspace = Path(workspace).resolve()
    directory = config_dir if config_dir is not None else get_config_directory()
    candidates = [find_instruction_file(directory)]
    candidates += [find_instruction_file(each) for each in find_project_directories(workspace)]
    for entry in extra or []:
        path = Path(entry).expanduser()
        candidates.append(read_named_instruction_file(path if path.is_absolute() else workspace / path))
    found: list[InstructionFile] = []
    seen: set[Path] = set()
    for file in candidates:
        # A config directory inside the workspace, or an `extra` naming a file
        # the walk already found, would otherwise contribute the same text twice.
        if file is not None and file.path.resolve() not in seen:
            seen.add(file.path.resolve())
            found.append(file)
    return apply_budget(found, MAX_INSTRUCTION_BYTES)
