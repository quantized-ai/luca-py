"""User-defined slash commands, read from markdown files on disk.

A custom command is ONE `.md` file: optional `---` fenced YAML frontmatter,
then a body. Typing `/name arg` sends that body to the model as an ordinary
user message, so a command is a saved prompt, not a new capability. This is
deliberately the same mechanism `/skill` uses (`post_message` with one
`TextContent`) and deliberately NOT a tool: nothing here reaches the model's
tool set, and no permission decision changes.

Frontmatter keys, all optional:

    description     one line, shown by `/help` and the palette
    argument-hint   the `[thing]` hint after the name in `/help`
    name            overrides the file stem

Placeholders in the body are filled from what was typed after the name:
`$ARGUMENTS` is the whole string, `$1`…`$9` the whitespace-split positions.
A body with NO placeholder still gets the argument, appended on its own line
— dropping what the user typed is worse than putting it somewhere obvious.

Only the `.md` files sitting DIRECTLY in a location are read. Claude Code
namespaces `commands/team/review.md` as `/team:review`; we do not, and since
`.claude/commands` is a directory it also owns, a nested file is silently not
a command here. Flat until someone asks for the colon.

Nothing here raises for a bad file. It is skipped and the rest still load,
the same contract as skill discovery.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from luca.agent.contrib.skills.discovery import parse_frontmatter

COMMAND_SUFFIX = ".md"
ARGUMENTS_TOKEN = "$ARGUMENTS"
_POSITIONAL = re.compile(r"\$([1-9])")
# A file stem has to survive being typed after a slash and split on whitespace.
_VALID_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
DEFAULT_DESCRIPTION = "user-defined command"


@dataclass(frozen=True)
class CustomCommand:
    """One discovered command. `path` is kept so `/help` can say where a name
    came from when two locations disagree."""

    name: str
    description: str
    argument_hint: str
    body: str
    path: Path


def _first_line(body: str) -> str:
    """The first non-empty, non-heading line — the fallback description when
    the file has no frontmatter, which most hand-written ones will not."""
    for line in body.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            return " ".join(stripped.split())
    return ""


def load_command(path: Path) -> CustomCommand | None:
    """The command in `path`, or None when it is unreadable, empty, or named
    something that cannot be typed after a slash."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    metadata, body = parse_frontmatter(text)
    body = body.strip()
    if not body:
        return None

    name = metadata.get("name")
    if not isinstance(name, str) or not name.strip():
        name = path.stem
    name = name.strip()
    if not _VALID_NAME.match(name):
        return None

    description = metadata.get("description")
    if not isinstance(description, str) or not description.strip():
        description = _first_line(body) or DEFAULT_DESCRIPTION
    hint = metadata.get("argument-hint")
    if not isinstance(hint, str):
        hint = ""

    return CustomCommand(
        name=name,
        description=" ".join(description.split()),
        argument_hint=hint.strip(),
        body=body,
        path=path,
    )


def default_locations(
    workspace: str | os.PathLike[str],
    home: Path | None = None,
) -> list[Path]:
    """The four standard locations, project before global. Mirrors skills so
    one repo holds both under the same two roots."""
    root = Path(workspace)
    house = home if home is not None else Path.home()
    return [
        root / ".claude" / "commands",
        root / ".agents" / "commands",
        house / ".claude" / "commands",
        house / ".agents" / "commands",
    ]


def resolve_locations(
    workspace: str | os.PathLike[str],
    extra: list[str] | None = None,
    home: Path | None = None,
) -> list[Path]:
    """Default locations plus the configured extras, in precedence order."""
    project, agents, home_claude, home_agents = default_locations(workspace, home)
    extras = [Path(entry).expanduser() for entry in extra or []]
    return [project, agents, *extras, home_claude, home_agents]


def discover_commands(
    locations: list[Path],
    reserved: frozenset[str] = frozenset(),
) -> dict[str, CustomCommand]:
    """Every readable command under `locations`, keyed by name.

    The first location wins a collision, and `sorted` keeps that independent of
    filesystem iteration order. A name in `reserved` is dropped outright: a
    built-in must never be shadowable, or a stray file in a cloned repo could
    redefine `/quit`."""
    found: dict[str, CustomCommand] = {}
    for location in locations:
        if not location.is_dir():
            continue
        for path in sorted(location.iterdir()):
            if path.suffix != COMMAND_SUFFIX or not path.is_file():
                continue
            command = load_command(path)
            if command is not None and command.name not in reserved:
                found.setdefault(command.name, command)
    return found


def expand(body: str, arg: str) -> str:
    """`body` with the typed argument substituted in.

    `$1`…`$9` take whitespace-split positions and `$ARGUMENTS` the whole
    string; both resolve to empty when nothing was typed. When the body used
    neither and there IS an argument, it is appended rather than dropped."""
    positions = arg.split()
    substituted = _POSITIONAL.sub(
        lambda m: positions[int(m.group(1)) - 1] if int(m.group(1)) <= len(positions) else "", body
    )
    used_positional = substituted != body
    if ARGUMENTS_TOKEN in substituted:
        return substituted.replace(ARGUMENTS_TOKEN, arg)
    if arg and not used_positional:
        return f"{substituted}\n\n{arg}"
    return substituted
