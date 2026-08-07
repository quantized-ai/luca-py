"""Battery B — inbound + execution (UC1, UC2, UC4, UC7, UC10, UC11-part1).

What the runner STORES when a native (or ordinary) call arrives — adoption to
the internal name with the client's generic-form extras captured verbatim —
and what the tool's `execute()` then receives.
"""

from luca.agent.core.models import (
    ApprovalStatus,
    ExecutionResult,
    ExecutionStatus,
    TextContent,
    ToolCall,
)

from .conftest import (
    apply_patch_call,
    assistant,
    make_runner,
    make_session,
    plain_call,
    shell_call,
    stored_assistant_parts,
    text,
)
from .plugin import ControllablePermissionPolicy
from .tools import OpenAIApplyPatchTool

SHELL_OUTPUT_EXTRAS = {
    "custom_type": "shell_call_output",
    "results": [
        {"stdout": "shell executed", "stderr": "", "outcome": {"type": "exit", "exit_code": 0}},
    ],
}


async def test_b1_an_openai_apply_patch_call_is_adopted_gated_and_executed(llm, monkeypatch):
    session = make_session("openai:gpt-5.1", native=True)
    policy = ControllablePermissionPolicy("require_approval")
    runner = make_runner(session, policy)
    llm.queue(
        assistant(
            apply_patch_call(
                "call_Rjs",
                "apc_08f3",
                {"type": "update_file", "path": "lib/fib.py", "diff": "@@ fibonacci"},
            ),
            finish_reason="tool_use",
        ),
    )
    runner.post_message("patch it")

    await runner.run()

    part = ToolCall(
        id="call_Rjs",
        name="openai_apply_patch",
        arguments={"type": "update_file", "path": "lib/fib.py", "diff": "@@ fibonacci"},
        extras={"custom_type": "apply_patch_call", "item_id": "apc_08f3", "status": "completed"},
    )
    assert stored_assistant_parts(session) == [part]
    execution = session.get_tool_execution("call_Rjs")
    assert execution.status == ExecutionStatus.PENDING  # awaiting approval
    assert execution.approval_status == ApprovalStatus.PENDING
    assert execution.raw_tool_call == part
    # a deep copy, not aliased — pinned by mutation, since two independently
    # derived objects are `is not` by construction (restored right after: the
    # entry is durable history the rest of this test still replays)
    stored_part = stored_assistant_parts(session)[0]
    stored_part.arguments["path"] = "tampered.py"
    assert execution.raw_tool_call.arguments["path"] == "lib/fib.py"
    stored_part.arguments["path"] = "lib/fib.py"

    seen_args: list[dict] = []

    async def record_execute(self, args, session, conversation_id, *, cancellation_token):
        seen_args.append(args)
        return ExecutionResult(content=[TextContent(text="openai_apply_patch executed")])

    monkeypatch.setattr(OpenAIApplyPatchTool, "execute", record_execute)
    policy.mode = "allow_all"  # approve
    llm.queue(assistant(text("done")))

    await runner.run()

    assert seen_args == [{"type": "update_file", "path": "lib/fib.py", "diff": "@@ fibonacci"}]
    execution = session.get_tool_execution("call_Rjs")
    assert execution.status == ExecutionStatus.COMPLETED
    assert execution.result == ExecutionResult(content=[TextContent(text="openai_apply_patch executed")])


async def test_b2_an_openai_shell_call_stores_its_structured_result_extras_at_birth(llm):
    session = make_session("openai:gpt-5.1", native=True)
    runner = make_runner(session)
    llm.queue(
        assistant(shell_call("tc2", "sh_12", {"commands": ["pytest -q"]}), finish_reason="tool_use"),
        assistant(text("done")),
    )
    runner.post_message("run tests")

    await runner.run()

    assert stored_assistant_parts(session) == [
        ToolCall(
            id="tc2",
            name="openai_shell",
            arguments={"commands": ["pytest -q"]},
            extras={"custom_type": "shell_call", "item_id": "sh_12", "status": "completed"},
        ),
    ]
    execution = session.get_tool_execution("tc2")
    assert execution.status == ExecutionStatus.COMPLETED
    assert execution.result == ExecutionResult(
        content=[TextContent(text="shell executed")],
        extras=SHELL_OUTPUT_EXTRAS,
    )


async def test_b3_an_anthropic_native_call_is_adopted_by_wire_name(llm):
    session = make_session("anthropic:claude-sonnet-4-5", native=True)
    runner = make_runner(session)
    llm.queue(
        assistant(plain_call("tc6", "bash", {"command": "ls"}), finish_reason="tool_use"),
        assistant(text("done")),
    )
    runner.post_message("list files")

    await runner.run()

    assert stored_assistant_parts(session) == [
        ToolCall(id="tc6", name="anthropic_bash_20250124", arguments={"command": "ls"}, extras={}),
    ]
    execution = session.get_tool_execution("tc6")
    assert execution.status == ExecutionStatus.COMPLETED
    assert execution.result == ExecutionResult(
        content=[TextContent(text="anthropic_bash_20250124 executed")],
    )


async def test_b4_native_off_a_bash_call_is_the_generic_tool_not_an_adoption(llm):
    session = make_session("anthropic:claude-sonnet-4-5", native=False)
    runner = make_runner(session)
    llm.queue(
        assistant(plain_call("tc7", "bash", {"command": "ls"}), finish_reason="tool_use"),
        assistant(text("done")),
    )
    runner.post_message("list files")

    await runner.run()

    # the gating edge case: same wire name, no adoption, the GENERIC tool runs
    assert stored_assistant_parts(session) == [
        ToolCall(id="tc7", name="bash", arguments={"command": "ls"}, extras={}),
    ]
    execution = session.get_tool_execution("tc7")
    assert execution.status == ExecutionStatus.COMPLETED
    assert execution.result == ExecutionResult(content=[TextContent(text="bash executed")])


async def test_b5_an_ordinary_function_tool_is_byte_identical_to_today(llm):
    session = make_session("openai:gpt-5.1", native=True)
    runner = make_runner(session)
    llm.queue(
        assistant(plain_call("tc9", "read", {"file_path": "main.py"}), finish_reason="tool_use"),
        assistant(text("done")),
    )
    runner.post_message("read it")

    await runner.run()

    # no extras, no renames, executes (UC10 — the 99% path)
    assert stored_assistant_parts(session) == [
        ToolCall(id="tc9", name="read", arguments={"file_path": "main.py"}, extras={}),
    ]
    execution = session.get_tool_execution("tc9")
    assert execution.status == ExecutionStatus.COMPLETED
    assert execution.result == ExecutionResult(content=[TextContent(text="read executed")])


async def test_b6_a_denied_native_shell_call_is_rejected_with_no_result(llm):
    session = make_session("openai:gpt-5.1", native=True)
    runner = make_runner(session, ControllablePermissionPolicy("deny_all"))
    llm.queue(
        assistant(shell_call("tc8", "sh_44", {"commands": ["rm -rf build"]}), finish_reason="tool_use"),
        assistant(text("ok")),
    )
    runner.post_message("clean up")

    await runner.run()

    execution = session.get_tool_execution("tc8")
    assert execution.status == ExecutionStatus.REJECTED
    assert execution.result is None


async def test_b7_a_copied_inactive_native_name_resolves_and_executes(llm):
    session = make_session("anthropic:claude-sonnet-4-5", native=True)
    runner = make_runner(session)
    llm.queue(
        assistant(
            plain_call("tc7", "openai_apply_patch", {"type": "update_file", "path": "main.py", "diff": "@@x"}),
            finish_reason="tool_use",
        ),
        assistant(text("done")),
    )
    runner.post_message("patch it")

    await runner.run()

    # not adopted (already internal), resolves, EXECUTES (UC7): the registry's
    # list never shrank, and not being advertised is not an error
    assert stored_assistant_parts(session) == [
        ToolCall(
            id="tc7",
            name="openai_apply_patch",
            arguments={"type": "update_file", "path": "main.py", "diff": "@@x"},
            extras={},
        ),
    ]
    execution = session.get_tool_execution("tc7")
    assert execution.status == ExecutionStatus.COMPLETED
    assert execution.result == ExecutionResult(
        content=[TextContent(text="openai_apply_patch executed")],
    )
