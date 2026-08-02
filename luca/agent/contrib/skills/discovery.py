"""Finding skills on disk and reading their frontmatter.

A skill is a directory holding a `SKILL.md`: YAML frontmatter fenced by `---`
carrying at least a `description`, then a body of instructions. The frontmatter
is what the model is shown up front; the body is loaded on demand.

Nothing here raises for a bad skill. A malformed `SKILL.md` sitting in a
directory the user forgot about must not stop the agent from starting, so a
directory that cannot be parsed is skipped and the rest still load.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised by uninstalling the group
    raise ImportError(
        "luca.agent.contrib.skills needs PyYAML: install the `skills` dependency group (uv sync)."
    ) from exc

FRONTMATTER_FENCE = "---"
SKILL_FILE = "SKILL.md"


@dataclass(frozen=True)
class Skill:
    """One discovered skill. `directory` is where bundled files live — a skill
    body routinely points at `references/*.md` next to it."""

    name: str
    description: str
    path: Path
    directory: Path
    body: str


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split a `---` fenced YAML header from the body.

    Returns `({}, text)` when there is no well-formed header, so a plain
    markdown file degrades to "no metadata" rather than an error."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != FRONTMATTER_FENCE:
        return {}, text
    for index in range(1, len(lines)):
        if lines[index].strip() == FRONTMATTER_FENCE:
            header = "\n".join(lines[1:index])
            body = "\n".join(lines[index + 1 :])
            try:
                loaded = yaml.safe_load(header)
            except yaml.YAMLError:
                return {}, text
            return (loaded if isinstance(loaded, dict) else {}), body
    return {}, text  # unterminated fence


def load_skill(directory: Path) -> Skill | None:
    """The skill in `directory`, or None if there isn't a usable one.

    `description` is required — it is the only thing the model sees before
    deciding whether to load the skill, so one without it cannot be offered.
    `name` falls back to the directory name, which is how skills are addressed
    anyway."""
    path = directory / SKILL_FILE
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    metadata, body = parse_frontmatter(text)
    description = metadata.get("description")
    if not isinstance(description, str) or not description.strip():
        return None
    name = metadata.get("name")
    if not isinstance(name, str) or not name.strip():
        name = directory.name
    return Skill(
        name=name.strip(),
        description=" ".join(description.split()),
        path=path,
        directory=directory,
        body=body.strip(),
    )


def default_locations(
    workspace: str | os.PathLike[str],
    home: Path | None = None,
) -> list[Path]:
    """The four locations from the spec, project before global.

    Project roots are anchored at the WORKSPACE rather than the process cwd:
    the shell tools already scope that way and the TUI already threads it, so
    one notion of "this project" serves both."""
    root = Path(workspace)
    house = home if home is not None else Path.home()
    return [
        root / ".claude" / "skills",
        root / ".agents" / "skills",
        house / ".claude" / "skills",
        house / ".agents" / "skills",
    ]


def resolve_locations(
    workspace: str | os.PathLike[str],
    extra: list[str] | None = None,
    home: Path | None = None,
) -> list[Path]:
    """Default locations plus the configured extras, in precedence order.

    Extras sit between the project and global defaults: more deliberate than a
    machine-wide default, less specific than a file committed in this repo."""
    project, agents, home_claude, home_agents = default_locations(workspace, home)
    extras = [Path(entry).expanduser() for entry in extra or []]
    return [project, agents, *extras, home_claude, home_agents]


def discover_skills(locations: list[Path]) -> dict[str, Skill]:
    """Every readable skill under `locations`, keyed by name.

    FIRST location wins a name collision, so a skill committed in the repo
    overrides a personal one of the same name. Within one location the order is
    sorted, so the winner does not depend on the filesystem's iteration order."""
    found: dict[str, Skill] = {}
    for location in locations:
        if not location.is_dir():
            continue
        for directory in sorted(location.iterdir()):
            if not directory.is_dir():
                continue
            skill = load_skill(directory)
            if skill is not None:
                found.setdefault(skill.name, skill)
    return found
