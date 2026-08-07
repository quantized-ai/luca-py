"""Battery C — outbound projection (UC3, UC5, UC8, UC9, UC11-part2).

Each test builds a session by driving Battery-B-style turns, then posts one
more message and asserts the FULL `messages` list the mocked `acompletion`
received: the projector's provider-blind default, last-mile-upgraded by the
plugin's `before_llm_call` middleware only where the call's owner is active.
"""

from luca.client.providers.anthropic import BashTool, TextEditorTool
from luca.client.types import (
    AssistantMessage as ClientAssistantMessage,
    TextBlock,
    ToolCall as ClientToolCall,
    ToolMessage as ClientToolMessage,
    UserMessage as ClientUserMessage,
)

from .conftest import (
    assistant,
    make_bare_runner,
    make_runner,
    make_session,
    seed_anthropic_bash_turn,
    seed_openai_shell_turn,
    shell_call,
    switch_model,
    text,
    wire_tool,
)
from .plugin import ControllablePermissionPolicy
from .test_battery_a import GENERIC_WIRE_TOOLS
from .test_battery_b import SHELL_OUTPUT_EXTRAS
from .tools import DeleteFileTool, ReadTool

# The OpenAI-born pair (tc2 / sh_12, seeded by `seed_openai_shell_turn`) in
# its two projected forms. UPGRADED is what an openai-native request carries —
# byte-identical to the original emission; VERBATIM is the projector's
# provider-blind default that every other request carries.
USER_RUN_TESTS = ClientUserMessage(content=[TextBlock(text="run tests")])
UPGRADED_SHELL_CALL = ClientAssistantMessage(
    content=[
        ClientToolCall(
            id="tc2",
            name="shell",
            arguments={"commands": ["pytest -q"]},
            extras={"custom_type": "shell_call", "item_id": "sh_12", "status": "completed"},
        ),
    ],
    provider="openai",
    model="gpt-5.1",
)
UPGRADED_SHELL_RESULT = ClientToolMessage(
    tool_call_id="tc2",
    content=[TextBlock(text="shell executed")],
    extras=SHELL_OUTPUT_EXTRAS,
)
VERBATIM_SHELL_CALL = ClientAssistantMessage(
    content=[ClientToolCall(id="tc2", name="openai_shell", arguments={"commands": ["pytest -q"]})],
    provider="openai",
    model="gpt-5.1",
)
VERBATIM_SHELL_RESULT = ClientToolMessage(
    tool_call_id="tc2",
    content=[TextBlock(text="shell executed")],
)
DONE_OPENAI = ClientAssistantMessage(content=[TextBlock(text="done")], provider="openai", model="gpt-5.1")

# The Anthropic-born pair (tc6, seeded by `seed_anthropic_bash_turn`) — its
# default form IS its foreign-provider form (UC8's verbatim half).
USER_NOW_BASH = ClientUserMessage(content=[TextBlock(text="now bash")])
VERBATIM_BASH_CALL = ClientAssistantMessage(
    content=[ClientToolCall(id="tc6", name="anthropic_bash_20250124", arguments={"command": "ls"})],
    provider="anthropic",
    model="claude-sonnet-4-5",
)
VERBATIM_BASH_RESULT = ClientToolMessage(
    tool_call_id="tc6",
    content=[TextBlock(text="anthropic_bash_20250124 executed")],
)
DONE2_ANTHROPIC = ClientAssistantMessage(
    content=[TextBlock(text="done2")],
    provider="anthropic",
    model="claude-sonnet-4-5",
)


async def test_c1_same_provider_replay_is_byte_identical_to_the_original(llm):
    session = make_session("openai:gpt-5.1", native=True)
    runner = make_runner(session)
    await seed_openai_shell_turn(llm, runner)

    llm.queue(assistant(text("sure")))
    runner.post_message("again")
    await runner.run()

    assert llm.calls[-1]["messages"] == [
        USER_RUN_TESTS,
        UPGRADED_SHELL_CALL,
        UPGRADED_SHELL_RESULT,
        DONE_OPENAI,
        ClientUserMessage(content=[TextBlock(text="again")]),
    ]


async def test_c2_a_provider_switch_projects_the_default_form(llm):
    session = make_session("openai:gpt-5.1", native=True)
    runner = make_runner(session)
    await seed_openai_shell_turn(llm, runner)

    switch_model(session, "anthropic:claude-sonnet-4-5")
    llm.queue(assistant(text("sure")))
    runner.post_message("again")
    await runner.run()

    # internal name, no extras, flat-text result — the request shape validated
    # live by poc_tests/anthropic_undeclared_history.py
    assert llm.calls[-1]["messages"] == [
        USER_RUN_TESTS,
        VERBATIM_SHELL_CALL,
        VERBATIM_SHELL_RESULT,
        DONE_OPENAI,
        ClientUserMessage(content=[TextBlock(text="again")]),
    ]
    assert llm.calls[-1]["tools"] == [
        TextEditorTool(),
        BashTool(),
        wire_tool(ReadTool),
        wire_tool(DeleteFileTool),
    ]


async def test_c3_a_provider_with_no_natives_gets_everything_verbatim(llm):
    session = make_session("openai:gpt-5.1", native=True)
    runner = make_runner(session)
    await seed_openai_shell_turn(llm, runner)

    switch_model(session, "openrouter:kimi-2.7")
    llm.queue(assistant(text("sure")))
    runner.post_message("again")
    await runner.run()

    assert llm.calls[-1]["messages"] == [
        USER_RUN_TESTS,
        VERBATIM_SHELL_CALL,
        VERBATIM_SHELL_RESULT,
        DONE_OPENAI,
        ClientUserMessage(content=[TextBlock(text="again")]),
    ]
    assert llm.calls[-1]["tools"] == GENERIC_WIRE_TOOLS


async def test_c4_switching_back_rebuilds_the_native_form_from_storage(llm):
    session = make_session("openai:gpt-5.1", native=True)
    runner = make_runner(session)
    await seed_openai_shell_turn(llm, runner)
    switch_model(session, "anthropic:claude-sonnet-4-5")
    llm.queue(assistant(text("checked")))
    runner.post_message("check something")
    await runner.run()

    switch_model(session, "openai:gpt-5.1")
    llm.queue(assistant(text("sure")))
    runner.post_message("again")
    await runner.run()

    # the round trip: the tc2 pair re-emits EXACTLY as in C1, rebuilt from
    # storage after a whole turn on the other provider
    assert llm.calls[-1]["messages"] == [
        USER_RUN_TESTS,
        UPGRADED_SHELL_CALL,
        UPGRADED_SHELL_RESULT,
        DONE_OPENAI,
        ClientUserMessage(content=[TextBlock(text="check something")]),
        ClientAssistantMessage(
            content=[TextBlock(text="checked")],
            provider="anthropic",
            model="claude-sonnet-4-5",
        ),
        ClientUserMessage(content=[TextBlock(text="again")]),
    ]


async def test_c5_mixed_families_upgrade_mine_keep_foreign_verbatim(llm):
    session = make_session("openai:gpt-5.1", native=True)
    runner = make_runner(session)
    await seed_openai_shell_turn(llm, runner)
    switch_model(session, "anthropic:claude-sonnet-4-5")
    await seed_anthropic_bash_turn(llm, runner)

    switch_model(session, "openai:gpt-5.1")
    llm.queue(assistant(text("sure")))
    runner.post_message("replay")
    await runner.run()

    # UC8: the openai pair upgraded, the anthropic pair verbatim under its
    # internal name, in the SAME messages list
    assert llm.calls[-1]["messages"] == [
        USER_RUN_TESTS,
        UPGRADED_SHELL_CALL,
        UPGRADED_SHELL_RESULT,
        DONE_OPENAI,
        USER_NOW_BASH,
        VERBATIM_BASH_CALL,
        VERBATIM_BASH_RESULT,
        DONE2_ANTHROPIC,
        ClientUserMessage(content=[TextBlock(text="replay")]),
    ]


async def test_c6_a_rejected_native_shell_call_synthesizes_its_structured_result(llm):
    session = make_session("openai:gpt-5.1", native=True)
    runner = make_runner(session, ControllablePermissionPolicy("deny_all"))
    llm.queue(
        assistant(shell_call("tc8", "sh_44", {"commands": ["rm -rf build"]}), finish_reason="tool_use"),
        assistant(text("ok")),
    )
    runner.post_message("clean up")
    await runner.run()

    llm.queue(assistant(text("fine")))
    runner.post_message("again")
    await runner.run()

    # UC11 — the formerly-broken case: no ExecutionResult was ever stored, so
    # the owner synthesizes shell's structured output from the derived text
    assert llm.calls[-1]["messages"] == [
        ClientUserMessage(content=[TextBlock(text="clean up")]),
        ClientAssistantMessage(
            content=[
                ClientToolCall(
                    id="tc8",
                    name="shell",
                    arguments={"commands": ["rm -rf build"]},
                    extras={"custom_type": "shell_call", "item_id": "sh_44", "status": "completed"},
                ),
            ],
            provider="openai",
            model="gpt-5.1",
        ),
        ClientToolMessage(
            tool_call_id="tc8",
            content=[TextBlock(text="[tool execution rejected]")],
            is_error=True,
            extras={
                "custom_type": "shell_call_output",
                "results": [
                    {
                        "stdout": "",
                        "stderr": "[tool execution rejected]",
                        "outcome": {"type": "exit", "exit_code": 1},
                    },
                ],
            },
        ),
        ClientAssistantMessage(content=[TextBlock(text="ok")], provider="openai", model="gpt-5.1"),
        ClientUserMessage(content=[TextBlock(text="again")]),
    ]


async def test_c7_without_the_plugin_everything_projects_verbatim(llm):
    session = make_session("openai:gpt-5.1", native=True)
    runner = make_runner(session)
    await seed_openai_shell_turn(llm, runner)

    bare = make_bare_runner(session)
    llm.queue(assistant(text("still fine")))
    bare.post_message("again")
    await bare.run()

    # UC9 — fail-safe by construction: no plugin, no middleware, no upgrade,
    # no crash; the default the projector produced is the whole request
    assert llm.calls[-1]["messages"] == [
        USER_RUN_TESTS,
        VERBATIM_SHELL_CALL,
        VERBATIM_SHELL_RESULT,
        DONE_OPENAI,
        ClientUserMessage(content=[TextBlock(text="again")]),
    ]
    assert llm.calls[-1]["tools"] is None


async def test_c8_a_pruned_native_shell_result_does_not_put_its_output_back_on_the_wire(llm):
    session = make_session("openai:gpt-5.1", native=True)
    runner = make_runner(session)
    await seed_openai_shell_turn(llm, runner)
    execution = session.get_tool_execution("tc2")
    runner.ledger.prune(
        session.main_conversation_id,
        execution.id,
        lambda entry_id, parent_id, ts: runner.context_manager.prune_entry(session, execution).model_copy(
            update={"id": entry_id, "parent_id": parent_id, "created_at": ts},
        ),
    )

    llm.queue(assistant(text("sure")))
    runner.post_message("again")
    await runner.run()

    # The stored payload is still in `entries` (pruning never deletes it), so
    # the upgrade must NOT reach for it — but shell's wire result still has to
    # be structured, so the placeholder is wrapped in the original's outcome.
    assert llm.calls[-1]["messages"][2] == ClientToolMessage(
        tool_call_id="tc2",
        content=[TextBlock(text="[tool output has been pruned to reduce context]")],
        extras={
            "custom_type": "shell_call_output",
            "results": [
                {
                    "stdout": "[tool output has been pruned to reduce context]",
                    "stderr": "",
                    "outcome": {"type": "exit", "exit_code": 0},
                },
            ],
        },
    )
