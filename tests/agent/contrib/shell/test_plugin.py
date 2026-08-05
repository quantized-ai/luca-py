"""Self-scoped tests for `ShellAccessPlugin`: construction wiring (absolute
roots, one shared tracker, one strategy), the seeded read-tier rules, and the
decide/pending flows they produce — no runner, no session. Executions are
built from each tool's real `build_permission_requests` output, serialized
exactly as `SimpleToolRegistry` would store it."""

import asyncio
from pathlib import Path

from luca.agent.contrib.resource_permissions import (
    PermissionMode,
    ResourcePermission,
    ToolRule,
)
from luca.agent.contrib.shell import ShellAccessPlugin
from luca.agent.contrib.simple_tool_registry import SimpleToolRegistry
from luca.agent.core import (
    AgentSession,
    ApprovalOption,
    ExecutionStatus,
    LLMConfig,
    SessionConfig,
    ToolCall,
    ToolExecution,
)
from luca.agent.core.runner import AgentSessionRunner
from tests.agent.scenarios import (
    conversation,
    main_conversation,
)


def make_plugin(tmp_path, **kwargs) -> ShellAccessPlugin:
    return ShellAccessPlugin(workspace=tmp_path, **kwargs)


def tool(plugin: ShellAccessPlugin, name: str):
    return next(t for t in plugin.tools if t.name == name)


def execution_for(plugin, name, args, session) -> ToolExecution:
    """A PENDING execution carrying the tool's real approval context, stored
    the way `SimpleToolRegistry` stores it."""
    target = tool(plugin, name)
    requests = target.build_permission_requests(
        target.Args.model_validate(args).model_dump(),
        session,
        main_conversation(session).id,
    )
    return ToolExecution(
        id="x_1",
        created_at=500,
        tool_call_id="c_1",
        raw_tool_call=ToolCall(id="c_1", name=name, arguments=args),
        tool_spec=target.get_tool_spec(),
        extras={
            "approval_context": {
                "requests": [request.model_dump() for request in requests],
            },
        },
        status=ExecutionStatus.PENDING,
    )


SESSION = AgentSession(
    id="s_plugin",
    conversations={"c1": conversation("c1", [], created_at=500, updated_at=500)},
    main_conversation_id="c1",
    session_config=SessionConfig(llm_config=LLMConfig(model="test-model", provider="faux")),
)


# ── construction wiring ───────────────────────────────────────────────────────


def test_workspace_and_additional_directories_are_absolutized(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    plugin = ShellAccessPlugin(
        workspace=Path("."),
        additional_directories=[Path("../sibling")],
    )

    assert plugin.workspace == tmp_path
    assert plugin.additional_directories == [tmp_path.parent / "sibling"]


def test_mode_accepts_the_string_form(tmp_path):
    assert make_plugin(tmp_path, mode="yolo").mode == PermissionMode.YOLO


def test_one_tracker_is_shared_across_read_edit_write(tmp_path):
    plugin = make_plugin(tmp_path)

    assert tool(plugin, "read").tracker is plugin.tracker
    assert tool(plugin, "edit").tracker is plugin.tracker
    assert tool(plugin, "write").tracker is plugin.tracker


def test_every_tool_resolves_against_the_workspace(tmp_path):
    plugin = make_plugin(tmp_path)

    assert [t.workdir for t in plugin.tools] == [tmp_path] * 7


async def test_get_tool_registry_bundles_the_tools_behind_the_strategy(tmp_path):
    plugin = make_plugin(tmp_path)

    registry = plugin.get_tool_registry(SESSION)

    assert isinstance(registry, SimpleToolRegistry)
    assert [spec.name for spec in await registry.get_tools(SESSION, main_conversation(SESSION).id)] == [
        "read",
        "glob",
        "grep",
        "edit",
        "write",
        "apply_patch",
        "bash",
    ]
    assert registry.permission_policy is plugin.permission_strategy


def test_system_prompt_part_names_the_permitted_directories(tmp_path):
    plugin = ShellAccessPlugin(
        workspace=tmp_path,
        additional_directories=[tmp_path.parent / "sibling"],
    )

    [part] = plugin.get_system_prompt_parts(SESSION)

    assert str(tmp_path) in part
    assert str(tmp_path.parent / "sibling") in part


# ── seeded rules ──────────────────────────────────────────────────────────────


def test_ask_mode_seeds_read_tier_allow_rules_over_the_workspace(tmp_path):
    plugin = make_plugin(tmp_path)

    assert plugin.permission_strategy.mode == PermissionMode.ASK
    assert plugin.permission_strategy.rules == [
        ToolRule(
            resource_permission=ResourcePermission(
                permission=permission,
                resource=resource,
            ),
            decision=ApprovalOption.ALLOW,
        )
        for permission in ("access_directory", "read", "glob", "grep")
        for resource in (str(tmp_path), f"{tmp_path}/*")
    ]


def test_additional_directories_seed_the_same_rules(tmp_path):
    extra = tmp_path.parent / "sibling"

    plugin = ShellAccessPlugin(workspace=tmp_path, additional_directories=[extra])

    assert [rule.resource_permission.resource for rule in plugin.permission_strategy.rules] == [
        resource
        for directory in (tmp_path, extra)
        for _ in ("access_directory", "read", "glob", "grep")
        for resource in (str(directory), f"{directory}/*")
    ]


# ── decide / pending flows ────────────────────────────────────────────────────


async def test_read_inside_the_workspace_is_allowed_silently(tmp_path, session):
    plugin = make_plugin(tmp_path)
    execution = execution_for(plugin, "read", {"file_path": "notes.txt"}, session)

    decision = await plugin.permission_strategy.decide(SESSION, execution)

    assert decision.decision == ApprovalOption.ALLOW
    assert plugin.permission_strategy.pending_requests(execution) == []


async def test_read_in_a_workspace_subdirectory_is_allowed(tmp_path, session):
    plugin = make_plugin(tmp_path)
    execution = execution_for(
        plugin,
        "read",
        {"file_path": "src/deep/notes.txt"},
        session,
    )

    decision = await plugin.permission_strategy.decide(SESSION, execution)

    assert decision.decision == ApprovalOption.ALLOW


async def test_read_outside_the_workspace_is_pending_with_both_steps(
    tmp_path,
    session,
):
    plugin = make_plugin(tmp_path / "workspace")
    outside = tmp_path / "elsewhere" / "secrets.txt"
    execution = execution_for(plugin, "read", {"file_path": str(outside)}, session)

    decision = await plugin.permission_strategy.decide(SESSION, execution)

    assert decision.decision == ApprovalOption.PENDING
    [access, verb] = plugin.permission_strategy.pending_requests(execution)
    assert access.resources == [
        ResourcePermission(
            permission="access_directory",
            resource=str(outside.parent),
        ),
    ]
    assert verb.resources == [
        ResourcePermission(permission="read", resource=str(outside)),
    ]


async def test_read_inside_an_additional_directory_is_allowed(tmp_path, session):
    extra = tmp_path / "elsewhere"
    plugin = ShellAccessPlugin(
        workspace=tmp_path / "workspace",
        additional_directories=[extra],
    )
    execution = execution_for(
        plugin,
        "read",
        {"file_path": str(extra / "notes.txt")},
        session,
    )

    decision = await plugin.permission_strategy.decide(SESSION, execution)

    assert decision.decision == ApprovalOption.ALLOW


async def test_edit_inside_the_workspace_prompts_only_for_the_verb(
    tmp_path,
    session,
):
    plugin = make_plugin(tmp_path)
    execution = execution_for(
        plugin,
        "edit",
        {"file_path": "notes.txt", "old_string": "a", "new_string": "b"},
        session,
    )

    decision = await plugin.permission_strategy.decide(SESSION, execution)

    assert decision.decision == ApprovalOption.PENDING
    [verb] = plugin.permission_strategy.pending_requests(execution)
    assert verb.resources == [
        ResourcePermission(permission="edit", resource=str(tmp_path / "notes.txt")),
    ]


async def test_bash_inside_the_workspace_prompts_only_for_the_command(
    tmp_path,
    session,
):
    plugin = make_plugin(tmp_path)
    execution = execution_for(plugin, "bash", {"command": "git status"}, session)

    decision = await plugin.permission_strategy.decide(SESSION, execution)

    assert decision.decision == ApprovalOption.PENDING
    [verb] = plugin.permission_strategy.pending_requests(execution)
    assert verb.resources == [
        ResourcePermission(permission="bash", resource="git status"),
    ]


async def test_yolo_mode_allows_everything(tmp_path, session):
    plugin = make_plugin(tmp_path, mode=PermissionMode.YOLO)
    execution = execution_for(
        plugin,
        "edit",
        {"file_path": "/etc/hosts", "old_string": "a", "new_string": "b"},
        session,
    )

    decision = await plugin.permission_strategy.decide(SESSION, execution)

    assert decision.decision == ApprovalOption.ALLOW


# ── the provider's editor replacing luca's ───────────────────────────────────


def _session(provider: str, model: str) -> AgentSession:
    return AgentSessionRunner.new_session(LLMConfig(model=model, provider=provider))


async def test_a_native_editor_replaces_read_edit_and_write(tmp_path):
    plugin = ShellAccessPlugin(tmp_path, native_tools=True)

    tools = await plugin.sync_tools(_session("anthropic", "claude-opus-5"))

    assert [tool.name for tool in tools] == [
        "str_replace_based_edit_tool",
        "glob",
        "grep",
        "apply_patch",
        "bash",
    ]
    await plugin.close()


async def test_luca_keeps_its_own_tools_when_native_is_off(tmp_path):
    plugin = ShellAccessPlugin(tmp_path, native_tools=False)

    tools = await plugin.sync_tools(_session("anthropic", "claude-opus-5"))

    assert [tool.name for tool in tools] == [
        "read",
        "glob",
        "grep",
        "edit",
        "write",
        "apply_patch",
        "bash",
    ]


async def test_the_native_editor_shares_the_plugins_one_tracker(tmp_path):
    # the read-first guard only holds while every file tool sees the same state
    plugin = ShellAccessPlugin(tmp_path, native_tools=True)

    tools = await plugin.sync_tools(_session("anthropic", "claude-opus-5"))

    assert tools[0].tracker is plugin.tracker
    await plugin.close()


async def test_switching_the_model_across_providers_rebuilds_the_tool_set(tmp_path):
    """`/model` changes the route, and the tools are a function of the route.
    Frozen at construction, an Anthropic editor rides along to a transport
    that refuses it before the HTTP call — on that turn and every retry."""
    plugin = ShellAccessPlugin(tmp_path, native_tools=True)
    registry = plugin.get_tool_registry(_session("anthropic", "claude-opus-5"))
    anthropic_session = _session("anthropic", "claude-opus-5")
    openai_session = _session("openai", "gpt-5.4")

    first = await registry.get_tools(anthropic_session, anthropic_session.main_conversation_id)
    second = await registry.get_tools(openai_session, openai_session.main_conversation_id)
    third = await registry.get_tools(anthropic_session, anthropic_session.main_conversation_id)

    assert [spec.provider_type for spec in first] == [
        "text_editor_20250728",
        None,
        None,
        None,
        "bash_20250124",
    ]
    assert [spec.provider_type for spec in second] == [None, None, None, None, None, "apply_patch", "shell"]
    assert [spec.provider_type for spec in third] == [
        "text_editor_20250728",
        None,
        None,
        None,
        "bash_20250124",
    ]
    await plugin.close()


async def test_a_model_switch_within_one_provider_keeps_the_tool_set(tmp_path):
    # nothing to rebuild, and rebuilding would drop the live shell for nothing
    plugin = ShellAccessPlugin(tmp_path, native_tools=True)
    session = _session("anthropic", "claude-opus-5")

    first = await plugin.sync_tools(session)
    session.session_config.llm_config = session.session_config.llm_config.model_copy(
        update={"model": "claude-sonnet-5"}
    )
    second = await plugin.sync_tools(session)

    assert first is second
    await plugin.close()


# ── the extension points ─────────────────────────────────────────────────────


class _EditorOnlyPlugin(ShellAccessPlugin):
    """A host luca does not know, serving a model it does not know, that the
    developer knows takes Anthropic's editor and nothing else."""

    def native_key_for(self, session):
        return ("text_editor_20250728", None, ())


class _NoPatchPlugin(ShellAccessPlugin):
    """Everything the base composes, minus `apply_patch`."""

    def install_tools(self, key):
        super().install_tools(key)
        self.tools = [tool for tool in self.tools if tool.name != "apply_patch"]


async def test_native_key_for_is_an_override_point(tmp_path):
    plugin = _EditorOnlyPlugin(tmp_path, native_tools=False)

    tools = await plugin.sync_tools(_session("openai", "gpt-4o"))

    # the subclass decides, not the route: native_tools=False and a non-Anthropic
    # model would both have said no
    assert [tool.name for tool in tools] == [
        "str_replace_based_edit_tool",
        "glob",
        "grep",
        "apply_patch",
        "bash",
    ]
    await plugin.close()


async def test_install_tools_is_an_override_point(tmp_path):
    plugin = _NoPatchPlugin(tmp_path, native_tools=True)

    tools = await plugin.sync_tools(_session("anthropic", "claude-opus-5"))

    assert [tool.name for tool in tools] == ["str_replace_based_edit_tool", "glob", "grep", "bash"]
    await plugin.close()


async def test_an_overridden_selection_survives_a_model_switch(tmp_path):
    # the registry re-asks the plugin, so the subclass stays in charge
    plugin = _EditorOnlyPlugin(tmp_path, native_tools=True)
    registry = plugin.get_tool_registry(_session("anthropic", "claude-opus-5"))
    openai_session = _session("openai", "gpt-5.4")

    specs = await registry.get_tools(openai_session, openai_session.main_conversation_id)

    assert [spec.provider_type for spec in specs] == ["text_editor_20250728", None, None, None, None]
    await plugin.close()


class _CountingPlugin(ShellAccessPlugin):
    installs = 0

    def install_tools(self, key):
        super().install_tools(key)
        type(self).installs += 1

    async def close(self):
        # A close with no shells open never reaches the event loop, so without
        # this the five callers below run start to finish one after another and
        # the test passes with or without the lock. In production the yield is
        # real: closing a live shell awaits the process.
        await asyncio.sleep(0)
        await super().close()


async def test_concurrent_syncs_swap_the_tool_set_once(tmp_path):
    """The runner births every tool call in a message concurrently, so several
    dispatches reach `sync_tools` at the same time. `close()` awaits, so an
    unlocked check-and-swap lets two callers both see the stale key, both
    release every shell, and both rebuild — leaving the registry holding a list
    whose shells nothing will ever close."""
    _CountingPlugin.installs = 0
    plugin = _CountingPlugin(tmp_path, native_tools=True)
    session = _session("anthropic", "claude-opus-5")

    results = await asyncio.gather(*(plugin.sync_tools(session) for _ in range(5)))

    assert _CountingPlugin.installs == 2  # the constructor's, plus one swap
    assert all(tools is plugin.tools for tools in results)
    await plugin.close()
