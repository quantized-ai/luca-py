"""The `skill` tool and the `SkillsPlugin` that bundles it.

The system prompt carries each skill's name and description; the body is loaded
by a tool call only when the model decides the skill is relevant.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from luca.agent.contrib.simple_tool_registry import (
    SimpleToolRegistry,
    YoloPermissionPolicy,
)
from luca.agent.contrib.tools import Tool
from luca.agent.core import (
    AgentSession,
    CancellationToken,
    SystemPromptPart,
    ToolKind,
)

from .discovery import Skill, discover_skills, resolve_locations

SKILLS_PROMPT_HEADER = """
### Skills
Skills are instruction sets for specific kinds of work. Load one with the `skill` tool when its description matches what you are doing, and follow it.
Available:
""".strip()


def format_skill_listing(skills: dict[str, Skill]) -> str:
    lines = [f"- {skill.name}: {skill.description}" for skill in sorted(skills.values(), key=lambda s: s.name)]
    return "\n".join([SKILLS_PROMPT_HEADER, *lines])


class SkillTool(Tool):
    namespace = "contrib.skills"
    name = "skill"
    description = "Load a skill's full instructions by name. Call this when a skill's description matches your task."
    tool_kind = ToolKind.READ

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")

        name: str = Field(description="The skill's name, as listed in the system prompt")

    def __init__(self, skills: dict[str, Skill]) -> None:
        self.skills = skills

    async def _execute(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
        *,
        cancellation_token: CancellationToken,
    ) -> str:
        skill = self.skills.get(args["name"])
        if skill is None:
            available = ", ".join(sorted(self.skills)) or "none"
            return f"No skill named {args['name']!r}. Available: {available}."
        # Bodies point at bundled files by relative path; the model needs the root.
        return f"Skill: {skill.name}\nBundled files are in: {skill.directory}\n\n{skill.body}"


class SkillsPlugin:
    """Bundles the `skill` tool with the prompt part advertising what is
    installed. Discovery runs once at construction."""

    def __init__(
        self,
        workspace: str | os.PathLike[str] = ".",
        extra_locations: list[str] | None = None,
        home: Path | None = None,
    ) -> None:
        self.locations = resolve_locations(workspace, extra_locations, home)
        self.skills = discover_skills(self.locations)

    @property
    def skill_directories(self) -> list[Path]:
        """The roots that actually held a skill — what the application grants
        read access over, so bundled files open without a prompt."""
        roots = []
        for location in self.locations:
            if any(skill.directory.parent == location for skill in self.skills.values()) and location not in roots:
                roots.append(location)
        return roots

    def get_tools(self) -> list[Tool]:
        return [SkillTool(self.skills)]

    def get_tool_registry(self, agent_session: AgentSession) -> SimpleToolRegistry:
        return SimpleToolRegistry(
            tools=self.get_tools(),
            permission_policy=YoloPermissionPolicy(),
        )

    def get_system_prompt_parts(self, agent_session: AgentSession) -> list:
        return [self._prompt_part]

    def _prompt_part(
        self,
        session: AgentSession,
        conversation_id: str,
    ) -> SystemPromptPart | None:
        if not self.skills:
            return None
        return SystemPromptPart(text=format_skill_listing(self.skills), source="skills")
