"""`ShellAccessPlugin` — the seven shell tools bundled behind one workspace.

One plugin instance owns the wiring the tools require but cannot provide
individually: a single workspace directory every tool resolves against, one
shared `FileReadTracker` (the read-first contract only holds when read/edit/
write see the same instance), and one `PermissionStrategy` seeded with the
directory grants.

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
for the directory too. YOLO mode allows everything (full-disk: this is a
permission gate, not a sandbox — approval is the only containment).

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
    APPLY_PATCH_TYPE,
    SHELL_TYPE,
    NativeApplyPatchTool,
    NativeBashTool,
    NativeShellTool,
    NativeTextEditorTool,
    native_bash_type,
    native_editor_type,
    native_openai_tool_types,
)
from .tools import (
    ApplyPatchTool,
    BashTool,
    EditTool,
    FileReadTracker,
    GlobTool,
    GrepTool,
    ReadTool,
    WriteTool,
)

READ_TIER_PERMISSIONS = ("access_directory", "read", "glob", "grep")

# editor type, bash type, OpenAI types — luca's own tools everywhere.
_NO_NATIVE: tuple = (None, None, ())

SHELL_SYSTEM_PROMPT_TEMPLATE = """
### Shell access
You have filesystem and process tools (read, glob, grep, edit, write, apply_patch, bash). Your workspace directory is {workspace}; relative paths resolve against it.{additional}
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
        native_tools: bool = False,
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
        self.native_tools = native_tools
        self.tools: list[Tool] = []
        self._native_key: tuple | None = None
        self.install_tools(_NO_NATIVE)

    # ── which tools exist ────────────────────────────────────────────────────

    def native_key_for(self, session: AgentSession | None) -> tuple:
        """The native tool types this session's model routes to.

        AN OVERRIDE POINT. Subclass and return your own
        `(editor_type, bash_type, openai_types)` to change which provider
        tools a route gets — a host luca does not know, a type it has not
        learned yet, or a policy that turns one of them off.

        Resolved from the SESSION rather than fixed at construction. Which
        tools exist is a function of the route, and `/model` changes the route
        without rebuilding anything — leaving an Anthropic `text_editor` on a
        request bound for the Responses API, which refuses it before the HTTP
        call and keeps refusing it on every retry."""
        if not self.native_tools or session is None:
            return _NO_NATIVE
        llm = session.session_config.llm_config
        return (
            native_editor_type(llm.provider, llm.model),
            native_bash_type(llm.provider, llm.model),
            native_openai_tool_types(llm.provider, llm.model),
        )

    async def sync_tools(self, session: AgentSession) -> list[Tool]:
        """The tool list for this session's model, rebuilt if the route moved.

        Awaited because the outgoing set may be holding shells open, and a
        swap that leaks one is the same bug as a session swap that leaks
        one."""
        key = self.native_key_for(session)
        if key != self._native_key:
            await self.close()
            self.install_tools(key)
        return self.tools

    def install_tools(self, key: tuple) -> None:
        """Build `self.tools` for one set of native types.

        THE OTHER OVERRIDE POINT: `native_key_for` decides, this one composes.
        Subclass to add a tool, drop one, or swap in your own implementation
        of a tool luca ships."""
        editor_type, bash_type, openai_types = key
        self._native_key = key
        # The provider's editor REPLACES read/edit/write rather than joining
        # them: it covers the same ground, and offering both asks the model to
        # choose between two tools that do one job. `glob`, `grep`,
        # `apply_patch` and `bash` have no native equivalent and always stay.
        file_tools: list[Tool] = (
            [
                ReadTool(workdir=self.workspace, tracker=self.tracker),
                EditTool(workdir=self.workspace, tracker=self.tracker),
                WriteTool(workdir=self.workspace, tracker=self.tracker),
            ]
            if editor_type is None
            else [
                NativeTextEditorTool(
                    self.workspace,
                    self.tracker,
                    provider_type=editor_type,
                )
            ]
        )
        patch_tool: Tool = (
            NativeApplyPatchTool(workdir=self.workspace)
            if APPLY_PATCH_TYPE in openai_types
            else ApplyPatchTool(workdir=self.workspace)
        )
        # Anthropic's `bash` and OpenAI's `shell` both keep ONE shell alive
        # across calls; luca's `bash` spawns a fresh subprocess each time. Same
        # job, different contract, so each is a swap rather than a rename.
        if bash_type is not None:
            run_tool: Tool = NativeBashTool(workdir=self.workspace)
        elif SHELL_TYPE in openai_types:
            run_tool = NativeShellTool(workdir=self.workspace)
        else:
            run_tool = BashTool(workdir=self.workspace)
        self.tools = [
            *file_tools[:1],
            GlobTool(workdir=self.workspace),
            GrepTool(workdir=self.workspace),
            *file_tools[1:],
            patch_tool,
            run_tool,
        ]

    async def close(self) -> None:
        """Release anything the tools hold open.

        Only the native bash tool does: it keeps a shell alive per
        conversation. A shell dies on its own when the process exits (its stdin
        closes), but a long-lived TUI rebuilds this plugin on every session
        swap, so without this each `/clear` would leave another idle shell
        behind for the rest of the run."""
        for tool in self.tools:
            closer = getattr(tool, "close", None)
            if closer is not None:
                await closer()

    def get_tool_registry(self, agent_session: AgentSession) -> SimpleToolRegistry:
        return ModelAwareRegistry(self, self.permission_strategy)

    def get_system_prompt_parts(self, agent_session: AgentSession) -> list[str]:
        additional = ""
        if self.additional_directories:
            listing = ", ".join(str(d) for d in self.additional_directories)
            additional = f"\nYou may also access: {listing}."
        return [
            SHELL_SYSTEM_PROMPT_TEMPLATE.format(
                workspace=self.workspace,
                additional=additional,
            )
        ]

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


class ModelAwareRegistry(SimpleToolRegistry):
    """The shell tool set, re-resolved from the session on every call.

    `get_tool_registry` runs ONCE, at runner construction, so a registry that
    froze its list there would still be handing out Anthropic's editor after
    `/model` moved the session to a GPT. The runner asks a registry for its
    tools fresh on every LLM call, with the live session, which is the only
    place the current model is known."""

    def __init__(self, plugin: ShellAccessPlugin, permission_policy) -> None:
        super().__init__(tools=list(plugin.tools), permission_policy=permission_policy)
        self._plugin = plugin

    async def sync(self, session: AgentSession) -> None:
        """Point `self.tools` at whatever the plugin says the model needs now.
        Public because a subclass overriding `get_tools` still has to call
        it."""
        tools = await self._plugin.sync_tools(session)
        if tools is not self.tools:
            self.tools = list(tools)
            self.tools_by_name = {tool.name: tool for tool in self.tools}

    async def get_tools(self, session, conversation_id):
        await self.sync(session)
        return await super().get_tools(session, conversation_id)

    async def create_execution(self, session, conversation_id, call):
        await self.sync(session)
        return await super().create_execution(session, conversation_id, call)
