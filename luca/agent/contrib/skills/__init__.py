"""Agent Skills — `SKILL.md` instruction sets discovered from disk.

Reads the Claude- and Agent-compatible locations plus any extra roots the config
names. Needs the `skills` dependency group (PyYAML).
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
