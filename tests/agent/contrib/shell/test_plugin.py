"""Self-scoped tests for `ShellAccessPlugin`: construction wiring (absolute
roots, one shared tracker, one strategy), the seeded read-tier rules, and the
decide/pending flows they produce — no runner, no session. Executions are
built from each tool's real `build_permission_requests` output, serialized
exactly as `SimpleToolRegistry` would store it."""

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
    Conversation,
    ExecutionStatus,
    LLMConfig,
    SessionConfig,
    ToolCall,
    ToolExecution,
)


def make_plugin(tmp_path, **kwargs) -> ShellAccessPlugin:
    return ShellAccessPlugin(workspace=tmp_path, **kwargs)


def tool(plugin: ShellAccessPlugin, name: str):
    return next(t for t in plugin.tools if t.name == name)


def execution_for(plugin, name, args, context) -> ToolExecution:
    """A PENDING execution carrying the tool's real approval context, stored
    the way `SimpleToolRegistry` stores it."""
    target = tool(plugin, name)
    requests = target.build_permission_requests(
        target.Args.model_validate(args).model_dump(), context,
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
    active_conversation=Conversation(id="c1", nodes=[], created_at=500, updated_at=500),
    session_config=SessionConfig(llm_config=LLMConfig(model="test-model", provider="faux")),
)


# ── construction wiring ───────────────────────────────────────────────────────


def test_workspace_and_additional_directories_are_absolutized(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    plugin = ShellAccessPlugin(
        workspace=Path("."), additional_directories=[Path("../sibling")],
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


def test_get_tool_registry_bundles_the_tools_behind_the_strategy(tmp_path):
    plugin = make_plugin(tmp_path)

    registry = plugin.get_tool_registry(SESSION)

    assert isinstance(registry, SimpleToolRegistry)
    assert [t.name for t in registry.get_tools(SESSION)] == [
        "read", "glob", "grep", "edit", "write", "apply_patch", "bash",
    ]
    assert registry.permission_policy is plugin.permission_strategy


def test_system_prompt_part_names_the_permitted_directories(tmp_path):
    plugin = ShellAccessPlugin(
        workspace=tmp_path, additional_directories=[tmp_path.parent / "sibling"],
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
                permission=permission, resource=resource,
            ),
            decision=ApprovalOption.ALLOW,
        )
        for permission in ("access_directory", "read", "glob", "grep")
        for resource in (str(tmp_path), f"{tmp_path}/*")
    ]


def test_additional_directories_seed_the_same_rules(tmp_path):
    extra = tmp_path.parent / "sibling"

    plugin = ShellAccessPlugin(workspace=tmp_path, additional_directories=[extra])

    assert [
        rule.resource_permission.resource for rule in plugin.permission_strategy.rules
    ] == [
        resource
        for directory in (tmp_path, extra)
        for _ in ("access_directory", "read", "glob", "grep")
        for resource in (str(directory), f"{directory}/*")
    ]


# ── decide / pending flows ────────────────────────────────────────────────────


async def test_read_inside_the_workspace_is_allowed_silently(tmp_path, context):
    plugin = make_plugin(tmp_path)
    execution = execution_for(plugin, "read", {"file_path": "notes.txt"}, context)

    decision = await plugin.permission_strategy.decide(execution)

    assert decision.decision == ApprovalOption.ALLOW
    assert plugin.permission_strategy.pending_requests(execution) == []


async def test_read_in_a_workspace_subdirectory_is_allowed(tmp_path, context):
    plugin = make_plugin(tmp_path)
    execution = execution_for(
        plugin, "read", {"file_path": "src/deep/notes.txt"}, context,
    )

    decision = await plugin.permission_strategy.decide(execution)

    assert decision.decision == ApprovalOption.ALLOW


async def test_read_outside_the_workspace_is_pending_with_both_steps(
    tmp_path, context,
):
    plugin = make_plugin(tmp_path / "workspace")
    outside = tmp_path / "elsewhere" / "secrets.txt"
    execution = execution_for(plugin, "read", {"file_path": str(outside)}, context)

    decision = await plugin.permission_strategy.decide(execution)

    assert decision.decision == ApprovalOption.PENDING
    [access, verb] = plugin.permission_strategy.pending_requests(execution)
    assert access.resources == [
        ResourcePermission(
            permission="access_directory", resource=str(outside.parent),
        ),
    ]
    assert verb.resources == [
        ResourcePermission(permission="read", resource=str(outside)),
    ]


async def test_read_inside_an_additional_directory_is_allowed(tmp_path, context):
    extra = tmp_path / "elsewhere"
    plugin = ShellAccessPlugin(
        workspace=tmp_path / "workspace", additional_directories=[extra],
    )
    execution = execution_for(
        plugin, "read", {"file_path": str(extra / "notes.txt")}, context,
    )

    decision = await plugin.permission_strategy.decide(execution)

    assert decision.decision == ApprovalOption.ALLOW


async def test_edit_inside_the_workspace_prompts_only_for_the_verb(
    tmp_path, context,
):
    plugin = make_plugin(tmp_path)
    execution = execution_for(
        plugin, "edit",
        {"file_path": "notes.txt", "old_string": "a", "new_string": "b"},
        context,
    )

    decision = await plugin.permission_strategy.decide(execution)

    assert decision.decision == ApprovalOption.PENDING
    [verb] = plugin.permission_strategy.pending_requests(execution)
    assert verb.resources == [
        ResourcePermission(permission="edit", resource=str(tmp_path / "notes.txt")),
    ]


async def test_bash_inside_the_workspace_prompts_only_for_the_command(
    tmp_path, context,
):
    plugin = make_plugin(tmp_path)
    execution = execution_for(plugin, "bash", {"command": "git status"}, context)

    decision = await plugin.permission_strategy.decide(execution)

    assert decision.decision == ApprovalOption.PENDING
    [verb] = plugin.permission_strategy.pending_requests(execution)
    assert verb.resources == [
        ResourcePermission(permission="bash", resource="git status"),
    ]


async def test_yolo_mode_allows_everything(tmp_path, context):
    plugin = make_plugin(tmp_path, mode=PermissionMode.YOLO)
    execution = execution_for(
        plugin, "edit",
        {"file_path": "/etc/hosts", "old_string": "a", "new_string": "b"},
        context,
    )

    decision = await plugin.permission_strategy.decide(execution)

    assert decision.decision == ApprovalOption.ALLOW
