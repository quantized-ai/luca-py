"""The `skill` tool and the `SkillsPlugin` that bundles it.

PROGRESSIVE DISCLOSURE. The system prompt carries only each skill's name and
description; the body is loaded by a tool call when the model decides the skill
is relevant. Putting every body in the prompt would be simpler and is the
obvious first idea, but the cost is paid on every request of every conversation
— including each subagent's — and it grows with the number of skills installed,
most of which are irrelevant to any given turn.

The prompt part is a CALLABLE so it re-reads discovery per call and contributes
nothing when no skills exist, the same shape `SubagentsPlugin` uses to keep its
prompt and its tool list from disagreeing.
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
        # The directory is part of the answer: skill bodies routinely point at
        # bundled files ("see references/widgets.md") with no way for the model
        # to know where that is.
        return f"Skill: {skill.name}\nBundled files are in: {skill.directory}\n\n{skill.body}"


class SkillsPlugin:
    """Bundles the `skill` tool with the prompt part that advertises what is
    installed. A plain class implementing the plugin hooks it needs.

    Discovery runs ONCE at construction: the tool list and the prompt must
    agree, and re-globbing the filesystem before every LLM call would put disk
    I/O on the hot path for a set that effectively never changes mid-session."""

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
        """The roots that actually held a skill.

        The application grants read access over these so bundled files open
        without an approval prompt. Only roots that produced something, so a
        directory that does not exist is never turned into a permission rule."""
        roots = []
        for location in self.locations:
            if any(skill.directory.parent == location for skill in self.skills.values()) and location not in roots:
                roots.append(location)
        return roots

    def get_tools(self) -> list[Tool]:
        return [SkillTool(self.skills)]

    def get_tool_registry(self, agent_session: AgentSession) -> SimpleToolRegistry:
        # Auto-allowed, like memory and subagents: reading a skill the user
        # installed themselves is not a decision worth interrupting for, and
        # whatever the skill then tells the agent to DO is gated as usual.
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
