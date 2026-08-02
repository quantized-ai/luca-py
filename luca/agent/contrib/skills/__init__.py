"""Agent Skills — `SKILL.md` instruction sets discovered from disk.

Reads the Claude- and Agent-compatible locations (`.claude/skills/<name>/`,
`.agents/skills/<name>/`, and the `~` equivalents) plus any extra roots the
config names. The model sees each skill's name and description in the system
prompt and loads a body on demand through the `skill` tool.

Needs the `skills` dependency group (PyYAML) — a default group, so `uv sync`
installs it.
"""

from .discovery import (
    Skill,
    default_locations,
    discover_skills,
    load_skill,
    parse_frontmatter,
    resolve_locations,
)
from .plugin import SkillsPlugin, SkillTool, format_skill_listing

__all__ = [
    "Skill",
    "SkillTool",
    "SkillsPlugin",
    "default_locations",
    "discover_skills",
    "format_skill_listing",
    "load_skill",
    "parse_frontmatter",
    "resolve_locations",
]
