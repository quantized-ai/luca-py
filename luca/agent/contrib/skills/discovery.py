"""Finding skills on disk and reading their frontmatter.

A skill is a directory holding a `SKILL.md`: `---` fenced YAML with a
`description`, then a body. Nothing here raises for a bad skill — it is skipped
and the rest still load.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "luca.agent.contrib.skills needs PyYAML: install the `skills` dependency group (uv sync)."
    ) from exc

FRONTMATTER_FENCE = "---"
SKILL_FILE = "SKILL.md"


@dataclass(frozen=True)
class Skill:
    """One discovered skill. `directory` is where its bundled files live."""

    name: str
    description: str
    path: Path
    directory: Path
    body: str


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split a `---` fenced YAML header from the body. `({}, text)` when there
    is no well-formed one."""
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
    """The skill in `directory`, or None. `description` is required (it is all
    the model sees before choosing); `name` falls back to the directory name."""
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
    """The four standard locations, project before global. Project roots anchor
    at the workspace, not the process cwd."""
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
    """Default locations plus the configured extras, in precedence order."""
    project, agents, home_claude, home_agents = default_locations(workspace, home)
    extras = [Path(entry).expanduser() for entry in extra or []]
    return [project, agents, *extras, home_claude, home_agents]


def discover_skills(locations: list[Path]) -> dict[str, Skill]:
    """Every readable skill under `locations`, keyed by name. The first
    location wins a collision; `sorted` keeps that independent of filesystem
    iteration order."""
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
