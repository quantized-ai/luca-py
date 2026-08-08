"""`ShellAccessPlugin` — the shell tools bundled behind one workspace.

Twelve tools: the eight generics, plus the four provider-native ones from
[`native/`](native/__init__.py). The registry always holds ALL of them — a
call born under one provider must still resolve, validate and execute after
the session switches to another — and `ShellNativeMiddleware`, registered
here through `get_middleware`, is what decides which of them each request
ADVERTISES and how they appear on the wire. With the native switch off, or on
a model whose provider has no natives, the four drop out of the tool list and
nothing else changes.

One plugin instance owns the wiring the tools require but cannot provide
individually: a single workspace directory every tool resolves against, one
shared `FileReadTracker` (the read-first contract only holds when read/edit/
write — and the native editors — see the same instance), and one
`PermissionStrategy` seeded with the directory grants. `delete_file`
deliberately stays outside that contract: `read` refuses binaries, so a
read-first guard would make a stray `.pyc` or archive undeletable — the
approval step is its containment.

Directories are stored ABSOLUTE at construction (cwd-anchored normpath — the
same no-symlink convention as `ShellTool._resolve`), so the rules written
from them keep meaning across resumed sessions regardless of the process's
later cwd.

The permission model is the two-step vocabulary the tools emit: every call
declares an `access_directory` step plus its own verb step. In ASK mode the
plugin seeds ALLOW rules for the read tier (`access_directory`, `read`,
`glob`, `grep`) over each permitted root and everything under it — reads
inside the workspace never prompt, while edit/write/apply_patch/bash prompt
for their verb, and any call reaching outside the permitted roots prompts
for the directory too — and so does every `delete_file` call, wherever it lands.
YOLO mode allows everything (full-disk: this is a permission gate, not a
sandbox — approval is the only containment).

The strategy is exposed as `permission_strategy` so the application can feed
`pending_requests()` / `apply_answer()` from its approval prompt, or share
the strategy with its own registries.
"""

from __future__ import annotations

import os
from pathlib import Path

from luca.agent.contrib.resource_permissions import (
    PermissionMode,
    PermissionStrategy,
    ResourcePermission,
    ToolRule,
)
from luca.agent.contrib.simple_tool_registry import SimpleToolRegistry
from luca.agent.contrib.tools import Tool
from luca.agent.core import AgentSession, ApprovalOption

from .native import (
    AnthropicBashTool,
    AnthropicTextEditorTool,
    OpenAIApplyPatchTool,
    OpenAIShellTool,
    ShellNativeMiddleware,
)
from .tools import (
    ApplyPatchTool,
    BashTool,
    DeleteFileTool,
    EditTool,
    FileReadTracker,
    GlobTool,
    GrepTool,
    ReadTool,
    WriteTool,
)

READ_TIER_PERMISSIONS = ("access_directory", "read", "glob", "grep")

SHELL_SYSTEM_PROMPT_TEMPLATE = """
### Shell access
You have filesystem and process tools ({tools}). Your workspace directory is {workspace}; relative paths resolve against it.{additional}
Paths outside these directories are NOT off-limits: calling a tool on one automatically asks the user to approve or deny that access. Never refuse a request or ask for permission in text because a path is outside the workspace — make the tool call and let the approval flow decide.
""".strip()


class ShellAccessPlugin:
    """Bundles the shell tools with a workspace-scoped permission strategy.
    A plain class implementing the plugin hooks (`get_tool_registry`,
    `get_system_prompt_parts`); pass it as `plugins=[...]` to
    `PluginAgentSessionRunner`."""

    def __init__(
        self,
        workspace: str | os.PathLike[str],
        additional_directories: list[str | os.PathLike[str]] | None = None,
        mode: PermissionMode | str = PermissionMode.ASK,
        extra_rules: list | None = None,
    ) -> None:
        self.workspace = _absolute(workspace)
        self.additional_directories = [_absolute(directory) for directory in additional_directories or []]
        self.mode = PermissionMode(mode)
        self.tracker = FileReadTracker()
        # Config-seeded rules follow the auto-seeded read-tier defaults, so a
        # later rule wins (the strategy is last-match-wins).
        self.permission_strategy = PermissionStrategy(
            mode=self.mode,
            rules=[*self._default_rules(), *(extra_rules or [])],
        )
        self.tools: list[Tool] = [
            ReadTool(workdir=self.workspace, tracker=self.tracker),
            GlobTool(workdir=self.workspace),
            GrepTool(workdir=self.workspace),
            EditTool(workdir=self.workspace, tracker=self.tracker),
            WriteTool(workdir=self.workspace, tracker=self.tracker),
            ApplyPatchTool(workdir=self.workspace),
            DeleteFileTool(workdir=self.workspace),
            BashTool(workdir=self.workspace),
            OpenAIApplyPatchTool(workdir=self.workspace, tracker=self.tracker),
            OpenAIShellTool(workdir=self.workspace),
            AnthropicTextEditorTool(workdir=self.workspace, tracker=self.tracker),
            AnthropicBashTool(workdir=self.workspace),
        ]

    def get_tool_registry(self, agent_session: AgentSession) -> SimpleToolRegistry:
        return SimpleToolRegistry(
            tools=list(self.tools),
            permission_policy=self.permission_strategy,
        )

    def get_middleware(self, agent_session: AgentSession) -> list:
        """The adaptation half of the plugin. Always installed: it reads the
        session's ACTIVE config per call, so with the native switch off (or on
        a model with no natives) it simply drops the four native specs from
        the tool list and touches nothing else."""
        return [ShellNativeMiddleware(agent_session)]

    def get_system_prompt_parts(self, agent_session: AgentSession) -> list:
        return [self.system_prompt_part]

    def system_prompt_part(self, session: AgentSession, conversation_id: str) -> str:
        """A CALLABLE part, resolved before every LLM call, because WHICH tools
        the model is about to be shown depends on the active provider and
        model. Naming `edit` and `bash` to a model that is being handed
        `str_replace_based_edit_tool` and a native `bash` sends it looking for
        tools it does not have."""
        visible = ShellNativeMiddleware.visible_names(session, [tool.name for tool in self.tools])
        additional = ""
        if self.additional_directories:
            listing = ", ".join(str(d) for d in self.additional_directories)
            additional = f"\nYou may also access: {listing}."
        return SHELL_SYSTEM_PROMPT_TEMPLATE.format(
            # the names the model will actually see: a native tool is declared
            # by its provider-facing name, not its internal identity.
            tools=", ".join(ShellNativeMiddleware.WIRE_NAMES.get(name, name) for name in visible),
            workspace=self.workspace,
            additional=additional,
        )

    def _default_rules(self) -> list[ToolRule]:
        """ALLOW rules for the read tier over each permitted root: the root
        itself plus `<root>/*` (fnmatch `*` crosses `/`, so the glob covers
        every depth)."""
        return [
            ToolRule(
                resource_permission=ResourcePermission(
                    permission=permission,
                    resource=resource,
                ),
                decision=ApprovalOption.ALLOW,
            )
            for directory in [self.workspace, *self.additional_directories]
            for permission in READ_TIER_PERMISSIONS
            for resource in (str(directory), f"{directory}/*")
        ]


def _absolute(path: str | os.PathLike[str]) -> Path:
    """Cwd-anchored normpath — absolute, no symlink resolution, matching
    `ShellTool._resolve`'s convention so rules and emitted pairs agree."""
    return Path(os.path.normpath(os.path.join(os.getcwd(), path)))
