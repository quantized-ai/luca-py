"""Private tools: a `ToolSpec` the runtime resolves and dispatches but that is
never advertised to the model.

Nothing here is subagent-specific — `is_private` is a standalone capability
that subagents happen to be the first consumer of. Three rules, and each one
has a test below:

1. `get_tools()` STILL returns it. That is the point: the runtime has to see
   the tool, because that is how it resolves, prepares and dispatches it. What
   changes is one thing — the runner omits it from the WIRE list.
2. A model tool call naming a private spec is REFUSED (`NOT_FOUND`), not
   resolved. The tool was never offered, but a model can still emit a name it
   was never given; without this rule "private" would be advisory.
3. Its execution never projects as a `ToolMessage`. That one is FORCED, not
   policy: no `ToolCall` for it exists in any `AssistantMessage` on the path,
   and a tool result carrying a `tool_call_id` the provider never issued is a
   protocol violation.
"""

from luca.agent.core.events import FinishReason, TextBlock, ToolCallReceived, ToolExecuted
from luca.agent.core.models import (
    AgentSession,
    ApprovalStatus,
    ExecutionResult,
    ExecutionStatus,
    SessionConfig,
    TextContent,
    ToolCall,
    ToolExecution,
    ToolExecutionError,
    ToolSpec,
)
from luca.agent.core.projection import ConversationProjector
from luca.client.testing import (
    FauxProvider,
    faux_assistant_message,
    faux_text,
    faux_tool_call,
)
from tests.agent.scenarios import (
    ADD_SPEC,
    MODEL,
    SECRET_SPEC,
    AddTool,
    DeterministicRunner,
    FakeToolRegistry,
    PrivateTool,
    conversation,
    main_conversation,
    make_session,
)

PROJECTOR = ConversationProjector()


def empty_session(session_id: str) -> AgentSession:
    return make_session(
        id=session_id,
        conversations={"c1": conversation("c1", [], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )


# ── 1. the spec ────────────────────────────────────────────────────────────────


def test_is_private_participates_in_spec_id():
    # definition-scoped like every other field, so a tool that gains it is a
    # new stored row rather than a silent overwrite of the public one
    public = ToolSpec(name="t", description="d", input_schema={})
    private = ToolSpec(name="t", description="d", input_schema={}, is_private=True)

    assert public.spec_id() != private.spec_id()


def test_the_tool_base_stamps_is_private_onto_the_spec():
    assert PrivateTool().get_tool_spec() == ToolSpec(
        name="secret",
        description="Never advertised.",
        input_schema=SECRET_SPEC.input_schema,
        is_private=True,
    )


# ── 2. the wire ────────────────────────────────────────────────────────────────


async def test_the_runtime_sees_a_private_tool_and_the_model_does_not():
    session = empty_session("s_private_wire")
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool(), PrivateTool()]),
        now=1000,
    )

    specs = await runner.resolve_tool_specs("c1")
    visible = runner.build_tool_list("c1", specs)

    # the RUNTIME's view holds both — that is how `secret` ever dispatches
    assert [spec.name for spec in specs] == ["add", "secret"]
    # the MODEL's view holds only the public one
    assert [spec.name for spec in visible] == ["add"]


async def test_build_tool_list_middleware_never_sees_a_private_spec():
    # The private filter runs AHEAD of the hook, so a middleware cannot
    # observe, keep, or accidentally re-advertise a runtime-only tool. That
    # ordering is the whole reason `is_private` is a guarantee rather than a
    # convention.
    class SpecRecorder:
        def __init__(self) -> None:
            self.seen: list[str] = []

        def build_tool_list(self, session, conversation_id, tools):
            self.seen = [spec.name for spec in tools]
            return tools

    recorder = SpecRecorder()
    session = empty_session("s_private_middleware")
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool(), PrivateTool()]),
        now=1000,
        middleware=[recorder],
    )

    specs = await runner.resolve_tool_specs("c1")
    runner.build_tool_list("c1", specs)

    assert recorder.seen == ["add"]


async def test_a_private_tool_is_absent_from_the_request_the_model_receives():
    faux = FauxProvider()
    faux.set_responses([faux_assistant_message([faux_text("ok")], finish_reason="stop")])
    session = empty_session("s_private_request")
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool(), PrivateTool()]),
        provider=faux,
        ids=["u1", "ts", "a1", "tf"],
        now=1000,
    )

    runner.post_message("hi")
    await runner.run()

    assert [tool.name for tool in faux.requests[0].tools] == ["add"]


# ── 3. a model call naming one is refused ─────────────────────────────────────


async def test_a_model_call_naming_a_private_tool_records_not_found():
    # the model guessed a name it was never given. It is refused rather than
    # resolved — deliberately indistinguishable from a tool that does not
    # exist, because from the model's point of view it does not.
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("secret", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("oh well")], finish_reason="stop"),
        ]
    )
    session = empty_session("s_private_guess")
    registry = FakeToolRegistry([AddTool(), PrivateTool()])
    runner = DeterministicRunner(
        session,
        tool_registry=registry,
        provider=faux,
        ids=["u1", "ts", "a1", "te1", "a2", "tf"],
        now=1000,
    )

    runner.post_message("call the secret tool")
    await runner.run()

    assert runner.session.entries["te1"] == ToolExecution(
        id="te1",
        parent_id="a1",
        created_at=1000,
        conversation_id="c1",
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="secret", arguments={"a": 1, "b": 2}),
        status=ExecutionStatus.NOT_FOUND,
        error=ToolExecutionError(
            error_type="ToolNotFound",
            error_message="Unknown tool: 'secret'.",
            details={"phase": "create_execution"},
        ),
        ended_at=1000,
        updated_at=1000,
        context_tokens=5,
    )
    # it never reached the registry: no spec was resolved, nothing was prepared
    assert runner.session.entries["te1"].tool_spec is None
    assert registry.prepared == []


# ── 4. projection ─────────────────────────────────────────────────────────────


def test_a_private_execution_projects_nothing():
    # FORCED, not policy: no ToolCall for it exists on the path, so a
    # ToolMessage correlating to one would be a protocol violation.
    execution = ToolExecution(
        id="te1",
        created_at=500,
        conversation_id="c1",
        tool_call_id="tc_runtime",
        raw_tool_call=ToolCall(id="tc_runtime", name="secret"),
        tool_spec=SECRET_SPEC,
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(content=[TextContent(text="ran privately")]),
        approval_status=ApprovalStatus.ALLOWED,
        started_at=500,
        ended_at=500,
    )

    assert PROJECTOR.project(["te1"], {"te1": execution}) == []


def test_a_public_execution_beside_it_still_projects():
    entries = {
        "te1": ToolExecution(
            id="te1",
            created_at=500,
            conversation_id="c1",
            tool_call_id="tc_runtime",
            raw_tool_call=ToolCall(id="tc_runtime", name="secret"),
            tool_spec=SECRET_SPEC,
            status=ExecutionStatus.COMPLETED,
            result=ExecutionResult(content=[TextContent(text="ran privately")]),
            started_at=500,
            ended_at=500,
        ),
        "te2": ToolExecution(
            id="te2",
            created_at=500,
            conversation_id="c1",
            tool_call_id="tc1",
            raw_tool_call=ToolCall(id="tc1", name="add"),
            tool_spec=ADD_SPEC,
            status=ExecutionStatus.COMPLETED,
            result=ExecutionResult(content=[TextContent(text="3")]),
            started_at=500,
            ended_at=500,
        ),
    }

    assert [type(m).__name__ for m in PROJECTOR.project(["te1", "te2"], entries)] == ["ToolMessage"]


def test_a_subclass_can_surface_private_work_some_other_way():
    # the ToolMessage channel is closed, but the entry is not structurally
    # invisible — V0 just declines to render it
    class Surfacing(ConversationProjector):
        def project_private_execution(self, entry, entries):
            from luca.client.types import TextBlock as LucaTextBlock, UserMessage as LucaUserMessage

            return LucaUserMessage(content=[LucaTextBlock(text=f"[ran {entry.raw_tool_call.name}]")])

    execution = ToolExecution(
        id="te1",
        created_at=500,
        conversation_id="c1",
        tool_call_id="tc_runtime",
        raw_tool_call=ToolCall(id="tc_runtime", name="secret"),
        tool_spec=SECRET_SPEC,
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(content=[TextContent(text="ran privately")]),
        started_at=500,
        ended_at=500,
    )

    [message] = Surfacing().project(["te1"], {"te1": execution})

    assert message.content[0].text == "[ran secret]"


# ── 5. everything else about it is an ordinary tool call ─────────────────────


async def test_the_refusal_still_produces_exactly_one_tool_output():
    # the one-output-per-call invariant holds for a refused private name too
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("secret", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("done")], finish_reason="stop"),
        ]
    )
    session = empty_session("s_private_one_output")
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([PrivateTool()]),
        provider=faux,
        ids=["u1", "ts", "a1", "te1", "a2", "tf"],
        now=1000,
    )

    runner.post_message("go")
    async with runner.run() as run:
        events = [event async for event in run]

    assert [type(e).__name__ for e in events] == [
        FinishReason.__name__,
        ToolCallReceived.__name__,
        ToolExecuted.__name__,
        TextBlock.__name__,
        FinishReason.__name__,
    ]
    assert main_conversation(runner.session).nodes == ["u1", "ts", "a1", "te1", "a2", "tf"]
    assert faux.requests[1].messages[-1].content[0].text == "Unknown tool: 'secret'."
