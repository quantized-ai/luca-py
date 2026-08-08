"""Battery D — cross-cutting.

The seams the other batteries lean on, each pinned once: a pending approval
executing after a provider switch (UC6), doom-loop detection over adopted
calls (extras cannot defeat it), serialization round-trip re-projection, and
the streaming path applying adoption identically to the non-streaming one
(verification V1 of the phase-1 plan)."""

from luca.agent.core.models import (
    AgentSession,
    ExecutionResult,
    ExecutionStatus,
    TextContent,
    ToolCall,
)

from .conftest import (
    assistant,
    make_runner,
    make_session,
    seed_anthropic_bash_turn,
    seed_openai_shell_turn,
    shell_call,
    stored_assistant_parts,
    switch_model,
    text,
)
from .plugin import ControllablePermissionPolicy
from .tools import OpenAIShellTool


async def test_d1_a_pending_approval_executes_after_a_provider_switch(llm, monkeypatch):
    session = make_session("openai:gpt-5.1", native=True)
    policy = ControllablePermissionPolicy("require_approval")
    runner = make_runner(session, policy)
    llm.queue(assistant(shell_call("tc9", "sh_77", {"commands": ["make deploy"]}), finish_reason="tool_use"))
    runner.post_message("deploy")
    await runner.run()  # parks at the gate, born under openai

    switch_model(session, "anthropic:claude-sonnet-4-5")
    policy.mode = "allow_all"  # approve
    seen_args: list[dict] = []

    async def record_execute(self, args, session, conversation_id, *, cancellation_token):
        seen_args.append(args)
        return ExecutionResult(content=[TextContent(text="shell executed")])

    monkeypatch.setattr(OpenAIShellTool, "execute", record_execute)
    llm.queue(assistant(text("done")))
    await runner.run()

    # UC6: the registry's list never shrank — the switch is invisible to
    # dispatch
    assert seen_args == [{"commands": ["make deploy"], "timeout_ms": None}]
    execution = session.get_tool_execution("tc9")
    assert execution.status == ExecutionStatus.COMPLETED
    assert execution.result == ExecutionResult(content=[TextContent(text="shell executed")])


async def test_d2_extras_cannot_defeat_the_doom_loop(llm):
    session = make_session("openai:gpt-5.1", native=True, doom_loop_threshold=2)
    runner = make_runner(session)
    llm.queue(
        assistant(shell_call("tcA", "sh_1", {"commands": ["pytest -q"]}), finish_reason="tool_use"),
        assistant(shell_call("tcB", "sh_2", {"commands": ["pytest -q"]}), finish_reason="tool_use"),
        assistant(text("done")),
    )
    runner.post_message("loop")

    await runner.run()

    # same name + arguments, different item_id: the repeat is flagged —
    # `_is_doom_loop` compares name + arguments only, never extras
    assert session.get_tool_execution("tcA").is_doom_loop_flagged is False
    assert session.get_tool_execution("tcB").is_doom_loop_flagged is True


async def test_d3_a_reloaded_session_projects_the_identical_messages(llm):
    session = make_session("openai:gpt-5.1", native=True)
    runner = make_runner(session)
    await seed_openai_shell_turn(llm, runner)
    switch_model(session, "anthropic:claude-sonnet-4-5")
    await seed_anthropic_bash_turn(llm, runner)
    switch_model(session, "openai:gpt-5.1")
    reloaded = AgentSession.model_validate_json(session.model_dump_json())
    reloaded_runner = make_runner(reloaded)

    llm.queue(assistant(text("sure")), assistant(text("sure")))
    runner.post_message("replay")
    await runner.run()
    reloaded_runner.post_message("replay")
    await reloaded_runner.run()

    assert llm.calls[-1]["messages"] == llm.calls[-2]["messages"]
    assert llm.calls[-1]["tools"] == llm.calls[-2]["tools"]


async def test_d4_streaming_adoption_is_identical_to_non_streaming(llm):
    session = make_session("openai:gpt-5.1", native=True)
    runner = make_runner(session)
    llm.queue(
        assistant(shell_call("tc2", "sh_12", {"commands": ["pytest -q"]}), finish_reason="tool_use"),
        assistant(text("done")),
    )
    runner.post_message("run tests")

    await runner.run(streaming=True)

    # V1 pinned: `after_llm_response` adoption runs on the assembled stream
    # exactly as on the non-streaming path — B2's stored shape, verbatim
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
        extras={
            "custom_type": "shell_call_output",
            "results": [
                {"stdout": "shell executed", "stderr": "", "outcome": {"type": "exit", "exit_code": 0}},
            ],
        },
    )
