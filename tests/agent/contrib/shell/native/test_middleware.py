"""Unit tests for `ShellNativeMiddleware` — the tables, in isolation: which
tools each mode advertises, how the surviving natives are declared, and which
inbound calls are adopted. The end-to-end projection (upgrade, replay,
provider switches) is the runner battery in `tests/agent/test_native_tools/`.
"""

import pytest

from luca.agent.contrib.shell import ShellNativeMiddleware, active_natives
from luca.agent.core import AgentSession, AgentSessionRunner, LLMConfig
from luca.client.providers.anthropic import BashTool, TextEditorTool
from luca.client.providers.openai import ApplyPatchTool, ApplyPatchToolCall, LocalShellTool, ShellToolCall
from luca.client.types import (
    AssistantMessage as ClientAssistantMessage,
    TextBlock,
    Tool as ClientTool,
    ToolCall as ClientToolCall,
)

ALL_TOOLS = [
    "read",
    "glob",
    "grep",
    "edit",
    "write",
    "apply_patch",
    "delete_file",
    "bash",
    "openai_apply_patch",
    "openai_shell",
    "anthropic_text_editor_20250728",
    "anthropic_bash_20250124",
]

GENERICS = ["read", "glob", "grep", "edit", "write", "apply_patch", "delete_file", "bash"]


def session(model: str = "gpt-5.6", provider: str = "openai", native: bool = True) -> AgentSession:
    built = AgentSessionRunner.new_session(
        LLMConfig(model=model, provider=provider),
        session_id="s_native_mw",
        conversation_id="c1",
    )
    built.session_config.use_native_tools = native
    built.use_native_tools = native
    return built


# ── which tools each mode advertises ──────────────────────────────────────────


@pytest.mark.parametrize(
    ("model", "provider", "native", "expected"),
    [
        (
            "gpt-5.6",
            "openai",
            True,
            ["read", "glob", "grep", "openai_apply_patch", "openai_shell"],
        ),
        (
            "claude-opus-5",
            "anthropic",
            True,
            [
                "read",
                "glob",
                "grep",
                "delete_file",
                "anthropic_text_editor_20250728",
                "anthropic_bash_20250124",
            ],
        ),
        # partial support: the native shell only, so every generic editor stays
        (
            "gpt-5.5-pro",
            "openai",
            True,
            ["read", "glob", "grep", "edit", "write", "apply_patch", "delete_file", "openai_shell"],
        ),
        # partial support the other way: the native patcher only, generic bash stays
        (
            "gpt-5.4-pro",
            "openai",
            True,
            ["read", "glob", "grep", "bash", "openai_apply_patch"],
        ),
        ("kimi-2.7", "openrouter", True, GENERICS),
        ("gpt-5.6", "openai", False, GENERICS),
    ],
)
def test_visible_names_per_mode(model, provider, native, expected):
    assert ShellNativeMiddleware.visible_names(session(model, provider, native), ALL_TOOLS) == expected


def test_active_natives_is_empty_when_the_switch_is_off():
    assert active_natives(session(native=False)) == frozenset()


def test_active_natives_follows_the_model_not_just_the_provider():
    assert active_natives(session("gpt-5.5-pro", "openai")) == frozenset({"openai_shell"})


# ── how the survivors are declared ────────────────────────────────────────────


def test_adapt_tool_declarations_swaps_the_openai_natives_for_native_items():
    middleware = ShellNativeMiddleware(session())
    tools = [
        ClientTool(name="read", description="Read a file.", parameters={}),
        ClientTool(name="openai_apply_patch", description="Apply a patch.", parameters={}),
        ClientTool(name="openai_shell", description="Run commands.", parameters={}),
    ]

    assert middleware.adapt_tool_declarations(middleware.session, "c1", tools) == [
        tools[0],
        ApplyPatchTool(),
        LocalShellTool(),
    ]


def test_adapt_tool_declarations_swaps_the_anthropic_natives_for_native_items():
    middleware = ShellNativeMiddleware(session("claude-opus-5", "anthropic"))
    tools = [
        ClientTool(name="anthropic_text_editor_20250728", description="Edit a file.", parameters={}),
        ClientTool(name="anthropic_bash_20250124", description="Run a command.", parameters={}),
    ]

    assert middleware.adapt_tool_declarations(middleware.session, "c1", tools) == [
        TextEditorTool(),
        BashTool(),
    ]


# ── adoption ──────────────────────────────────────────────────────────────────


def test_a_typed_openai_call_is_adopted_under_its_internal_name_with_its_extras():
    middleware = ShellNativeMiddleware(session())
    message = ClientAssistantMessage(
        content=[
            ApplyPatchToolCall(
                id="tc1", name="apply_patch", arguments={"type": "delete_file", "path": "a"}, item_id="apc_1"
            )
        ],
    )

    adopted = middleware.after_llm_response(middleware.session, "c1", message)

    assert adopted.content == [
        ClientToolCall(
            id="tc1",
            name="openai_apply_patch",
            arguments={"type": "delete_file", "path": "a"},
            extras={"custom_type": "apply_patch_call", "item_id": "apc_1", "status": "completed"},
        ),
    ]


def test_a_typed_shell_call_is_adopted_even_after_a_switch_to_another_provider():
    # adoption by `custom_type` needs no active table: a typed call can only
    # have come from the provider that mints it
    middleware = ShellNativeMiddleware(session("claude-opus-5", "anthropic"))
    message = ClientAssistantMessage(
        content=[ShellToolCall(id="tc2", name="shell", arguments={"commands": ["ls"]}, item_id="sh_1")],
    )

    adopted = middleware.after_llm_response(middleware.session, "c1", message)

    assert adopted.content[0].name == "openai_shell"


def test_a_plain_bash_call_is_adopted_while_the_anthropic_native_is_active():
    middleware = ShellNativeMiddleware(session("claude-opus-5", "anthropic"))
    message = ClientAssistantMessage(
        content=[ClientToolCall(id="tc3", name="bash", arguments={"command": "ls"})],
    )

    adopted = middleware.after_llm_response(middleware.session, "c1", message)

    assert adopted.content == [
        ClientToolCall(id="tc3", name="anthropic_bash_20250124", arguments={"command": "ls"}),
    ]


def test_a_plain_bash_call_stays_the_generic_tool_when_natives_are_off():
    middleware = ShellNativeMiddleware(session(native=False))
    message = ClientAssistantMessage(
        content=[ClientToolCall(id="tc4", name="bash", arguments={"command": "ls"})],
    )

    adopted = middleware.after_llm_response(middleware.session, "c1", message)

    assert adopted.content == [ClientToolCall(id="tc4", name="bash", arguments={"command": "ls"})]


def test_text_and_foreign_calls_ride_through_untouched():
    middleware = ShellNativeMiddleware(session())
    message = ClientAssistantMessage(
        content=[
            TextBlock(text="on it"),
            ClientToolCall(id="tc5", name="read", arguments={"file_path": "a.py"}),
        ],
    )

    assert middleware.after_llm_response(middleware.session, "c1", message).content == message.content
