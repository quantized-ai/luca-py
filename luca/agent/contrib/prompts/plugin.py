"""The two prompt plugins.

`SystemPromptPlugin` contributes the agent's persona, the addendum for the
model's family, and the environment block. `InstructionsPlugin` contributes the
project's own `LUCA.md` / `AGENTS.md` / `CLAUDE.md`.

Two plugins rather than one with flags: an application that wants a project's
instructions but its own persona installs only the second.
"""

from __future__ import annotations

import os
import platform
from datetime import date
from pathlib import Path

from luca.agent.core import AgentSession, SystemPromptPart

from .environment import environment_text
from .instructions import (
    MAX_INSTRUCTION_BYTES,
    InstructionFile,
    find_instructions,
    project_directories,
)
from .selection import BASE, load_prompt, select_family

INSTRUCTIONS_PROMPT_HEADER = """
### Project instructions
The following was written by this project's maintainers. Follow it. Where it disagrees with anything above, it wins.
""".strip()

# The environment and the project's rules belong at the END of the assembled
# prompt, after every plugin's tool blurb. Everything else in the composition
# leaves `priority` at its -1 default, so these two are what pin the tail.
ENVIRONMENT_PRIORITY = 90
INSTRUCTIONS_PRIORITY = 100


def format_instructions(files: list[InstructionFile]) -> str:
    """The files under one header, each labelled with the path it came from so
    the model can go back and read the rest of it."""
    blocks = [f"--- {file.path} ---\n{file.text}" for file in files]
    return "\n\n".join([INSTRUCTIONS_PROMPT_HEADER, *blocks])


class SystemPromptPlugin:
    """The base persona, tuned to the model, plus the environment block.

    Every part is a CALLABLE rather than a static part: `/model` reassigns the
    session's `llm_config` mid-session, and both the family addendum and the
    environment block have to follow it."""

    def __init__(
        self,
        workspace: str | os.PathLike[str] = ".",
        *,
        environment: bool = True,
        today: date | None = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.environment = environment
        self._today = today
        # Once at construction, not once per LLM call: this is a filesystem
        # walk and the answer cannot change within a session.
        self.is_git_repo = (project_directories(self.workspace)[0] / ".git").exists()

    def get_system_prompt_parts(self, agent_session: AgentSession) -> list:
        parts = [self._base_part, self._family_part]
        return [*parts, self._environment_part] if self.environment else parts

    def _base_part(self, session: AgentSession, conversation_id: str) -> SystemPromptPart:
        return SystemPromptPart(text=load_prompt(BASE), source="prompt")

    def _family_part(self, session: AgentSession, conversation_id: str) -> SystemPromptPart:
        family = select_family(session.session_config.llm_config.model)
        return SystemPromptPart(text=load_prompt(family), source=f"prompt.{family}")

    def _environment_part(self, session: AgentSession, conversation_id: str) -> SystemPromptPart:
        config = session.session_config.llm_config
        return SystemPromptPart(
            text=environment_text(
                workspace=self.workspace,
                model=config.model,
                provider=config.provider,
                platform_name=platform.system(),
                today=self._today or date.today(),
                is_git_repo=self.is_git_repo,
            ),
            source="env",
            priority=ENVIRONMENT_PRIORITY,
        )


class InstructionsPlugin:
    """The project's own instruction files. Discovery runs once at
    construction, like `SkillsPlugin`."""

    def __init__(
        self,
        workspace: str | os.PathLike[str] = ".",
        extra: list[str] | None = None,
        *,
        config_dir: Path | None = None,
        max_bytes: int = MAX_INSTRUCTION_BYTES,
    ) -> None:
        self.files = find_instructions(
            workspace,
            extra,
            config_dir=config_dir,
            max_bytes=max_bytes,
        )

    def get_system_prompt_parts(self, agent_session: AgentSession) -> list:
        return [self._prompt_part]

    def _prompt_part(
        self,
        session: AgentSession,
        conversation_id: str,
    ) -> SystemPromptPart | None:
        if not self.files:
            return None
        return SystemPromptPart(
            text=format_instructions(self.files),
            source="agents.md",
            priority=INSTRUCTIONS_PRIORITY,
        )
