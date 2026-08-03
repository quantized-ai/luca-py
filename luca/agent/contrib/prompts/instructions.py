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

Nothing here raises. An unreadable, undecodable or empty file is skipped.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

INSTRUCTION_FILES = ("LUCA.md", "AGENTS.md", "CLAUDE.md")

# This text rides on every request of every conversation in the session,
# subagents included, so it gets a budget.
MAX_INSTRUCTION_BYTES = 32 * 1024


@dataclass(frozen=True)
class InstructionFile:
    path: Path
    text: str


def config_directory() -> Path:
    """`$XDG_CONFIG_HOME/luca` or `~/.config/luca` — the same directory
    `luca.json` is read from, reimplemented here because contrib packages do
    not import from the TUI."""
    base = os.environ.get("XDG_CONFIG_HOME")
    return (Path(base) if base else Path.home() / ".config") / "luca"


def read_instruction_file(path: Path) -> InstructionFile | None:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    return InstructionFile(path=path, text=text.strip()) if text.strip() else None


def first_in_directory(directory: Path) -> InstructionFile | None:
    """The one instruction file this directory contributes, by name precedence."""
    for name in INSTRUCTION_FILES:
        found = read_instruction_file(directory / name)
        if found is not None:
            return found
    return None


def project_directories(workspace: str | os.PathLike[str]) -> list[Path]:
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
    max_bytes: int = MAX_INSTRUCTION_BYTES,
) -> list[InstructionFile]:
    """Every instruction file that applies, least specific first. Relative
    `extra` entries resolve against the workspace."""
    workspace = Path(workspace).resolve()
    directory = config_dir if config_dir is not None else config_directory()
    candidates = [first_in_directory(directory)]
    candidates += [first_in_directory(each) for each in project_directories(workspace)]
    for entry in extra or []:
        path = Path(entry).expanduser()
        candidates.append(read_instruction_file(path if path.is_absolute() else workspace / path))
    found: list[InstructionFile] = []
    seen: set[Path] = set()
    for file in candidates:
        # A config directory inside the workspace, or an `extra` naming a file
        # the walk already found, would otherwise contribute the same text twice.
        if file is not None and file.path.resolve() not in seen:
            seen.add(file.path.resolve())
            found.append(file)
    return apply_budget(found, max_bytes)
