"""Declarative runner scenarios driven by the FauxProvider.

Each test follows precondition → action → postcondition: a KNOWN starting
AgentSession (inline literal or a `scenarios.py` constant) + KNOWN scripted
faux responses + deterministic ids/clock → ONE action (drain a lazy run via
`async with runner.run(...) as run: [event async for event in run]`), then
assert FULL objects — the resulting AgentSession (status included) and the
complete event list. No helpers, no logic. The AgentRun handle's own
lifecycle (lazy/eager forms, suspend, RunResult) lives in
`test_runner_lifecycle.py`.

Approval-gate and resume scenarios live in `test_runner_approvals.py`; the
entry-derived queries in `test_ledger.py`. Tests here that aren't about
approval rely on `FakeToolRegistry`'s unscripted decide — a deterministic
allow-all whose ALLOW decisions carry the frozen clock — so every
ToolExecution literal shows `approval_status=ALLOWED` with
`approval_decisions=[ALLOW@now]`.

Event snapshots: `ToolCallReceived` carries the BIRTH state (persisted at
creation — `updated_at=None`), `ToolExecutionStarted` the persisted RUNNING
state, `ToolExecuted` the final terminal state. Each test spells the three
snapshots out as local literals and asserts the complete event list against
them. `ToolExecutionStarted` fires IFF the body was dispatched, so a call
that fails to resolve at dispatch time produces only the other two.

Tool-spec normalization: the ledger files every `ToolSpec` in
`session.tool_specs` under its content hash and stamps `tool_spec_id` on the
execution at birth, so every execution snapshot an event carries already holds
the id. Tests that spell those snapshots out therefore write
`tool_spec_id=ADD_SPEC_ID` on the literal and hand `make_session` the matching
`tool_specs` row; tests that only assert the session let `make_session` fill
both.

Determinism comes from `DeterministicRunner` (`scenarios.py`) — a test-side
subclass overriding the production `generate_id()` / `now_ms()` hooks. Its
`ids` script spans every call (post_message, run, resume) and is consumed in
this order per turn:
  TurnStart, (AssistantMessage, [ToolExecution per call])..., TurnFinish
"""

import pytest
from pydantic import ValidationError

from luca.agent.core.context import CancellationToken
from luca.agent.core.events import (
    FinishReason,
    ReasoningBlock,
    ReasoningDelta,
    ReasoningStart,
    TextBlock,
    TextDelta,
    TextStart,
    ToolCallReceived,
    ToolCallStart,
    ToolExecuted,
    ToolExecutionStarted,
)
from luca.agent.core.exceptions import AgentError, InvalidToolArguments, ToolNotFound
from luca.agent.core.models import (
    AgentSession,
    ApprovalDecision,
    ApprovalOption,
    ApprovalStatus,
    AssistantMessage,
    Conversation,
    ConversationStatus,
    ExecutionAttempt,
    ExecutionAttemptOutcome,
    ExecutionResult,
    ExecutionStatus,
    ImageContent,
    MediaBase64,
    SessionConfig,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolExecution,
    ToolExecutionError,
    TurnFinish,
    TurnOutcome,
    TurnStart,
    Usage,
    UserMessage,
)
from luca.agent.core.runner import AgentSessionRunner
from luca.agent.core.tool_registry import PreparedTool
from luca.client.testing import (
    FauxProvider,
    faux_assistant_message,
    faux_text,
    faux_thinking,
    faux_tool_call,
)
from luca.client.types import Tool as LucaTool, Usage as ClientUsage
from tests.agent.scenarios import (
    ADD_SPEC,
    BOOM_SPEC,
    MODEL,
    MULTIPLY_SPEC,
    REPORT_SPEC,
    AddTool,
    BinaryArgs,
    CapturingTool,
    DeterministicRunner,
    FakeTool,
    FakeToolRegistry,
    MultiplyTool,
    RaisingTool,
    RichErrorTool,
    conversation,
    main_conversation,
    make_session,
)

ALLOW_1000 = ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000)

# The content hash each spec is filed under — what the ledger stamps on the
# execution at birth and what an event snapshot therefore carries.
ADD_SPEC_ID = ADD_SPEC.spec_id()
MULTIPLY_SPEC_ID = MULTIPLY_SPEC.spec_id()
BOOM_SPEC_ID = BOOM_SPEC.spec_id()
REPORT_SPEC_ID = REPORT_SPEC.spec_id()

# Pydantic's own structured report for `add(a=1)`, carried verbatim under
# `details["errors"]`, and the failure text the projector derives from it.
MISSING_B = [
    {"type": "missing", "loc": ["b"], "msg": "Field required", "input": {"a": 1}},
]
INVALID_ADD_TEXT = (
    "Arguments for tool 'add' are invalid.\n"
    '[{"type": "missing", "loc": ["b"], "msg": "Field required", "input": {"a": 1}}]'
)


# ── local doubles ──────────────────────────────────────────────────────────────
#
# `scenarios.FakeToolRegistry` resolves the same tool at birth and at dispatch,
# which is exactly what a healthy registry does — so the dispatch-time failure
# modes need registries whose `prepare` disagrees with their `create_execution`.
# Each overrides one method and inherits the rest.


class BrokenBirthRegistry(FakeToolRegistry):
    """`create_execution` raises: the registry itself is broken, not the call.
    The runner synthesizes the FAILED draft — nothing resolved, so no
    `tool_spec`."""

    async def create_execution(
        self,
        session: AgentSession,
        conversation_id: str,
        call: ToolCall,
    ) -> ToolExecution:
        raise RuntimeError("registry unavailable")


class VanishingToolRegistry(FakeToolRegistry):
    """Resolves at birth and fails to resolve at dispatch — the tool left the
    catalog in between (a plugin unloaded, a remote server dropped it)."""

    async def prepare(
        self,
        session: AgentSession,
        conversation_id: str,
        tool_execution: ToolExecution,
    ) -> PreparedTool:
        raise ToolNotFound(f"Unknown tool: {tool_execution.raw_tool_call.name!r}.")


class RejectingArgumentsRegistry(FakeToolRegistry):
    """`prepare` rejects arguments the birth draft accepted — validation is
    the registry's, and it may happen at dispatch."""

    async def prepare(
        self,
        session: AgentSession,
        conversation_id: str,
        tool_execution: ToolExecution,
    ) -> PreparedTool:
        raise InvalidToolArguments(
            f"Arguments for tool {tool_execution.raw_tool_call.name!r} are invalid.",
            errors=MISSING_B,
        )


class NonCallablePrepareRegistry(FakeToolRegistry):
    """`prepare` returns None where the contract requires a callable."""

    async def prepare(
        self,
        session: AgentSession,
        conversation_id: str,
        tool_execution: ToolExecution,
    ) -> PreparedTool:
        return None


class LookupTool(FakeTool):
    """A BODY that raises `ToolNotFound` looking up a sub-resource — a tool
    failure, not a resolution failure."""

    name = "lookup"
    description = "Look a record up."
    Args = BinaryArgs

    async def _execute(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
        *,
        tool_name: str,
        tool_call_id: str,
        cancellation_token: CancellationToken,
    ) -> str:
        raise ToolNotFound("no such record: 7")


LOOKUP_SPEC = LookupTool().get_tool_spec()
LOOKUP_SPEC_ID = LOOKUP_SPEC.spec_id()


# ── plain turns (no approval gate) ─────────────────────────────────────────────


async def test_single_text_response_no_tools():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message([faux_text("Hello!")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s1",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")]),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        provider=faux,
        ids=["ts", "a1", "tf"],
        now=1000,
    )

    async with runner.run() as run:
        events = [event async for event in run]

    assert events == [
        TextBlock(conversation_id="c1", text="Hello!"),
        FinishReason(conversation_id="c1", finish_reason="stop"),
    ]
    assert runner.session == make_session(
        id="s1",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")]),
            "ts": TurnStart(id="ts", parent_id="u1", created_at=1000),
            "a1": AssistantMessage(
                id="a1",
                parent_id="ts",
                created_at=1000,
                parts=[TextContent(text="Hello!")],
                llm_config=MODEL,
                stop_reason="stop",
                context_tokens=1,  # len("Hello!") // 4
            ),
            "tf": TurnFinish(id="tf", parent_id="a1", created_at=1000),
        },
        tool_specs={},  # no execution resolved a tool — nothing to file
        usages={"c1": {"a1": Usage(conversation_id="c1", entry_id="a1")}},
        conversations={
            "c1": conversation(
                "c1",
                ["u1", "ts", "a1", "tf"],
                created_at=500,
                updated_at=1000,
            )
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )


async def test_run_passes_projected_tools_to_client():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message([faux_text("ok")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_tools",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")]),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()]),
        provider=faux,
        ids=["ts", "a1", "tf"],
        now=1000,
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    # the advertised spec's JSON Schema reaches the wire verbatim — the core
    # hands the transport plain data, never a Python class
    assert faux.requests[0].tools == [
        LucaTool(
            name="add",
            description="Add two numbers.",
            parameters=ADD_SPEC.input_schema,
        ),
    ]


async def test_resolve_tool_specs_then_build_tool_list_offers_every_spec():
    # the two halves of the per-call tool step: `resolve_tool_specs` is the
    # async one (`get_tools` may need I/O), `build_tool_list` is the pure,
    # middleware-bearing narrowing to what the model may see. Both speak
    # `ToolSpec`; the adapter converts to the wire type afterwards, inside
    # `_collect_tools`.
    session = make_session(
        id="s_build_tools",
        conversations={"c1": conversation("c1", [], created_at=900, updated_at=900)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool(), MultiplyTool()]),
        now=1000,
    )

    specs = await runner.resolve_tool_specs(runner.main_conversation_id)
    visible = runner.build_tool_list(runner.main_conversation_id, specs)

    assert specs == [ADD_SPEC, MULTIPLY_SPEC]
    assert visible == [ADD_SPEC, MULTIPLY_SPEC]


async def test_the_tool_step_of_a_toolless_runner_is_empty():
    session = make_session(
        id="s_toolless",
        conversations={"c1": conversation("c1", [], created_at=900, updated_at=900)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(session, now=1000)

    specs = await runner.resolve_tool_specs(runner.main_conversation_id)
    tools = runner.build_tool_list(runner.main_conversation_id, specs)

    assert specs == []

    assert tools == []


async def test_reasoning_plus_tool_call_then_text():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_thinking("Let me add."), faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("It's 3.")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s3",
        entries={
            "u1": UserMessage(
                id="u1",
                created_at=500,
                parts=[TextContent(text="Add 1 and 2")],
            ),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    registry = FakeToolRegistry([AddTool()])
    runner = DeterministicRunner(
        session,
        tool_registry=registry,
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
    )

    async with runner.run() as run:
        events = [event async for event in run]

    birth = ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        tool_spec=ADD_SPEC,
        tool_spec_id=ADD_SPEC_ID,
        status=ExecutionStatus.PENDING,
    )
    running = birth.model_copy(
        update={
            "status": ExecutionStatus.RUNNING,
            "approval_status": ApprovalStatus.ALLOWED,
            "approval_decisions": [ALLOW_1000],
            "attempts": [ExecutionAttempt(started_at=1000)],
            "updated_at": 1000,
        }
    )
    final = running.model_copy(
        update={
            "status": ExecutionStatus.COMPLETED,
            "result": ExecutionResult(content=[TextContent(text="3")], is_error=False),
            "attempts": [ExecutionAttempt(outcome=ExecutionAttemptOutcome.COMPLETED, started_at=1000, ended_at=1000)],
            "finished_at": 1000,
        }
    )
    assert events == [
        ReasoningBlock(conversation_id="c1", text="Let me add."),
        FinishReason(conversation_id="c1", finish_reason="tool_use"),
        ToolCallReceived(conversation_id="c1", tool_call_id="tc1", execution=birth),
        ToolExecutionStarted(conversation_id="c1", tool_call_id="tc1", execution=running),
        ToolExecuted(
            conversation_id="c1",
            tool_call_id="tc1",
            execution=final,
            result_text="3",
            is_error=False,
        ),
        TextBlock(conversation_id="c1", text="It's 3."),
        FinishReason(conversation_id="c1", finish_reason="stop"),
    ]
    assert registry.prepared == ["add"]  # resolved once, at dispatch
    assert runner.session == make_session(
        id="s3",
        entries={
            "u1": UserMessage(
                id="u1",
                created_at=500,
                parts=[TextContent(text="Add 1 and 2")],
            ),
            "ts": TurnStart(id="ts", parent_id="u1", created_at=1000),
            "a1": AssistantMessage(
                id="a1",
                parent_id="ts",
                created_at=1000,
                parts=[
                    ThinkingContent(thinking="Let me add."),
                    ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
                ],
                llm_config=MODEL,
                stop_reason="tool_use",
                context_tokens=7,  # (thinking + name + JSON args) // 4
            ),
            "te1": final,
            "a2": AssistantMessage(
                id="a2",
                parent_id="te1",
                created_at=1000,
                parts=[TextContent(text="It's 3.")],
                llm_config=MODEL,
                stop_reason="stop",
                context_tokens=1,  # len("It's 3.") // 4
            ),
            "tf": TurnFinish(id="tf", parent_id="a2", created_at=1000),
        },
        tool_specs={ADD_SPEC_ID: ADD_SPEC},
        usages={
            "c1": {
                "a1": Usage(conversation_id="c1", entry_id="a1"),
                "a2": Usage(conversation_id="c1", entry_id="a2"),
            }
        },
        conversations={
            "c1": conversation(
                "c1",
                ["u1", "ts", "a1", "te1", "a2", "tf"],
                created_at=500,
                updated_at=1000,
            )
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )


async def test_multi_turn_two_tool_rounds_then_text():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_thinking("First add."), faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message(
                [faux_thinking("Now multiply."), faux_tool_call("multiply", {"a": 3, "b": 4}, id="tc2")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("Done: 12")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s4",
        entries={
            "u1": UserMessage(id="u1", created_at=0, parts=[TextContent(text="Go")]),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=0, updated_at=0)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    registry = FakeToolRegistry([AddTool(), MultiplyTool()])
    runner = DeterministicRunner(
        session,
        tool_registry=registry,
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "te2", "a3", "tf"],
        now=1000,
    )

    async with runner.run() as run:
        events = [event async for event in run]

    add_birth = ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        tool_spec=ADD_SPEC,
        tool_spec_id=ADD_SPEC_ID,
        status=ExecutionStatus.PENDING,
    )
    add_running = add_birth.model_copy(
        update={
            "status": ExecutionStatus.RUNNING,
            "approval_status": ApprovalStatus.ALLOWED,
            "approval_decisions": [ALLOW_1000],
            "attempts": [ExecutionAttempt(started_at=1000)],
            "updated_at": 1000,
        }
    )
    add_final = add_running.model_copy(
        update={
            "status": ExecutionStatus.COMPLETED,
            "result": ExecutionResult(content=[TextContent(text="3")], is_error=False),
            "attempts": [ExecutionAttempt(outcome=ExecutionAttemptOutcome.COMPLETED, started_at=1000, ended_at=1000)],
            "finished_at": 1000,
        }
    )
    multiply_birth = ToolExecution(
        id="te2",
        conversation_id="c1",
        parent_id="a2",
        created_at=1000,
        tool_call_id="tc2",
        raw_tool_call=ToolCall(id="tc2", name="multiply", arguments={"a": 3, "b": 4}),
        tool_spec=MULTIPLY_SPEC,
        tool_spec_id=MULTIPLY_SPEC_ID,
        status=ExecutionStatus.PENDING,
    )
    multiply_running = multiply_birth.model_copy(
        update={
            "status": ExecutionStatus.RUNNING,
            "approval_status": ApprovalStatus.ALLOWED,
            "approval_decisions": [ALLOW_1000],
            "attempts": [ExecutionAttempt(started_at=1000)],
            "updated_at": 1000,
        }
    )
    multiply_final = multiply_running.model_copy(
        update={
            "status": ExecutionStatus.COMPLETED,
            "result": ExecutionResult(content=[TextContent(text="12")], is_error=False),
            "attempts": [ExecutionAttempt(outcome=ExecutionAttemptOutcome.COMPLETED, started_at=1000, ended_at=1000)],
            "finished_at": 1000,
        }
    )
    assert events == [
        ReasoningBlock(conversation_id="c1", text="First add."),
        FinishReason(conversation_id="c1", finish_reason="tool_use"),
        ToolCallReceived(conversation_id="c1", tool_call_id="tc1", execution=add_birth),
        ToolExecutionStarted(conversation_id="c1", tool_call_id="tc1", execution=add_running),
        ToolExecuted(
            conversation_id="c1",
            tool_call_id="tc1",
            execution=add_final,
            result_text="3",
            is_error=False,
        ),
        ReasoningBlock(conversation_id="c1", text="Now multiply."),
        FinishReason(conversation_id="c1", finish_reason="tool_use"),
        ToolCallReceived(conversation_id="c1", tool_call_id="tc2", execution=multiply_birth),
        ToolExecutionStarted(conversation_id="c1", tool_call_id="tc2", execution=multiply_running),
        ToolExecuted(
            conversation_id="c1",
            tool_call_id="tc2",
            execution=multiply_final,
            result_text="12",
            is_error=False,
        ),
        TextBlock(conversation_id="c1", text="Done: 12"),
        FinishReason(conversation_id="c1", finish_reason="stop"),
    ]
    assert registry.prepared == ["add", "multiply"]  # once per dispatch, in order
    assert runner.session == make_session(
        id="s4",
        entries={
            "u1": UserMessage(id="u1", created_at=0, parts=[TextContent(text="Go")]),
            "ts": TurnStart(id="ts", parent_id="u1", created_at=1000),
            "a1": AssistantMessage(
                id="a1",
                parent_id="ts",
                created_at=1000,
                parts=[
                    ThinkingContent(thinking="First add."),
                    ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
                ],
                llm_config=MODEL,
                stop_reason="tool_use",
                context_tokens=7,  # (thinking + name + JSON args) // 4
            ),
            "te1": add_final,
            "a2": AssistantMessage(
                id="a2",
                parent_id="te1",
                created_at=1000,
                parts=[
                    ThinkingContent(thinking="Now multiply."),
                    ToolCall(id="tc2", name="multiply", arguments={"a": 3, "b": 4}),
                ],
                llm_config=MODEL,
                stop_reason="tool_use",
                context_tokens=9,  # (thinking + name + JSON args) // 4
            ),
            "te2": multiply_final,
            "a3": AssistantMessage(
                id="a3",
                parent_id="te2",
                created_at=1000,
                parts=[TextContent(text="Done: 12")],
                llm_config=MODEL,
                stop_reason="stop",
                context_tokens=2,  # len("Done: 12") // 4
            ),
            "tf": TurnFinish(id="tf", parent_id="a3", created_at=1000),
        },
        tool_specs={ADD_SPEC_ID: ADD_SPEC, MULTIPLY_SPEC_ID: MULTIPLY_SPEC},
        usages={
            "c1": {
                "a1": Usage(conversation_id="c1", entry_id="a1"),
                "a2": Usage(conversation_id="c1", entry_id="a2"),
                "a3": Usage(conversation_id="c1", entry_id="a3"),
            }
        },
        conversations={
            "c1": conversation(
                "c1",
                ["u1", "ts", "a1", "te1", "a2", "te2", "a3", "tf"],
                created_at=0,
                updated_at=1000,
            )
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )


async def test_two_calls_to_one_tool_file_a_single_shared_spec_row():
    # Normalization end to end: a `ToolSpec` is a pure function of the tool
    # DEFINITION, so calling one tool twice writes ONE row in `tool_specs` and
    # both executions reference — and hold — that single spec.
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [
                    faux_tool_call("add", {"a": 1, "b": 2}, id="tc1"),
                    faux_tool_call("add", {"a": 3, "b": 4}, id="tc2"),
                ],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("3 and 7.")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_normalized",
        entries={
            "u1": UserMessage(id="u1", created_at=0, parts=[TextContent(text="Add twice")]),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=0, updated_at=0)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()]),
        provider=faux,
        ids=["ts", "a1", "te1", "te2", "a2", "tf"],
        now=1000,
    )

    await runner.run()

    first, second = runner.session.entries["te1"], runner.session.entries["te2"]
    assert runner.session.tool_specs == {ADD_SPEC_ID: ADD_SPEC}
    assert (first.tool_spec_id, second.tool_spec_id) == (ADD_SPEC_ID, ADD_SPEC_ID)
    # the stored spec is a value object held BY REFERENCE, not copied per call —
    # true in memory exactly as it is after a reload
    assert first.tool_spec is runner.session.tool_specs[ADD_SPEC_ID]
    assert second.tool_spec is runner.session.tool_specs[ADD_SPEC_ID]
    assert (first.status, second.status) == (
        ExecutionStatus.COMPLETED,
        ExecutionStatus.COMPLETED,
    )


async def test_provider_usage_is_recorded_per_assistant_entry():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
                usage=ClientUsage(
                    input_tokens=10,
                    output_tokens=5,
                    total_tokens=15,
                    cached_input_tokens=2,
                    cache_write_tokens=1,
                ),
            ),
            faux_assistant_message(
                [faux_text("It's 3.")],
                finish_reason="stop",
                usage=ClientUsage(input_tokens=20, output_tokens=7, total_tokens=27),
            ),
        ]
    )
    session = make_session(
        id="s_usage",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Add")]),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()]),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    # usage is accessory conversation-entry data: one record per assistant
    # entry in the store, nothing embedded on entries or rolled up on markers
    assert runner.session.usages == {
        "c1": {
            "a1": Usage(
                conversation_id="c1",
                entry_id="a1",
                input=10,
                output=5,
                cache_read=2,
                cache_write=1,
                total_tokens=15,
            ),
            "a2": Usage(
                conversation_id="c1",
                entry_id="a2",
                input=20,
                output=7,
                cache_read=0,
                cache_write=0,
                total_tokens=27,
            ),
        },
    }
    assert runner.session.entries["tf"] == TurnFinish(
        id="tf",
        parent_id="a2",
        created_at=1000,
    )


# ── streaming ──────────────────────────────────────────────────────────────────


async def test_streaming_produces_same_session_as_run():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_thinking("Let me add."), faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("It's 3.")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s5",
        entries={
            "u1": UserMessage(
                id="u1",
                created_at=500,
                parts=[TextContent(text="Add 1 and 2")],
            ),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()]),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
    )

    async with runner.run(streaming=True) as run:
        _ = [event async for event in run]

    assert runner.session == make_session(
        id="s5",
        entries={
            "u1": UserMessage(
                id="u1",
                created_at=500,
                parts=[TextContent(text="Add 1 and 2")],
            ),
            "ts": TurnStart(id="ts", parent_id="u1", created_at=1000),
            "a1": AssistantMessage(
                id="a1",
                parent_id="ts",
                created_at=1000,
                parts=[
                    ThinkingContent(thinking="Let me add."),
                    ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
                ],
                llm_config=MODEL,
                stop_reason="tool_use",
                context_tokens=7,  # (thinking + name + JSON args) // 4
            ),
            "te1": ToolExecution(
                id="te1",
                conversation_id="c1",
                parent_id="a1",
                created_at=1000,
                tool_call_id="tc1",
                raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
                tool_spec=ADD_SPEC,
                status=ExecutionStatus.COMPLETED,
                result=ExecutionResult(
                    content=[TextContent(text="3")],
                    is_error=False,
                ),
                approval_status=ApprovalStatus.ALLOWED,
                approval_decisions=[ALLOW_1000],
                attempts=[ExecutionAttempt(outcome=ExecutionAttemptOutcome.COMPLETED, started_at=1000, ended_at=1000)],
                finished_at=1000,
                updated_at=1000,
            ),
            "a2": AssistantMessage(
                id="a2",
                parent_id="te1",
                created_at=1000,
                parts=[TextContent(text="It's 3.")],
                llm_config=MODEL,
                stop_reason="stop",
                context_tokens=1,  # len("It's 3.") // 4
            ),
            "tf": TurnFinish(id="tf", parent_id="a2", created_at=1000),
        },
        usages={
            "c1": {
                "a1": Usage(conversation_id="c1", entry_id="a1"),
                "a2": Usage(conversation_id="c1", entry_id="a2"),
            }
        },
        conversations={
            "c1": conversation(
                "c1",
                ["u1", "ts", "a1", "te1", "a2", "tf"],
                created_at=500,
                updated_at=1000,
            )
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )


async def test_streaming_emits_delta_then_block_events_in_order():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_thinking("Let me add."), faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("It's 3.")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s6",
        entries={
            "u1": UserMessage(id="u1", created_at=0, parts=[TextContent(text="Add")]),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=0, updated_at=0)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()]),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
    )

    async with runner.run(streaming=True) as run:
        events = [event async for event in run]

    birth = ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        tool_spec=ADD_SPEC,
        tool_spec_id=ADD_SPEC_ID,
        status=ExecutionStatus.PENDING,
    )
    running = birth.model_copy(
        update={
            "status": ExecutionStatus.RUNNING,
            "approval_status": ApprovalStatus.ALLOWED,
            "approval_decisions": [ALLOW_1000],
            "attempts": [ExecutionAttempt(started_at=1000)],
            "updated_at": 1000,
        }
    )
    final = running.model_copy(
        update={
            "status": ExecutionStatus.COMPLETED,
            "result": ExecutionResult(content=[TextContent(text="3")], is_error=False),
            "attempts": [ExecutionAttempt(outcome=ExecutionAttemptOutcome.COMPLETED, started_at=1000, ended_at=1000)],
            "finished_at": 1000,
        }
    )
    assert events == [
        ReasoningStart(conversation_id="c1"),
        ReasoningDelta(conversation_id="c1", text="Let me add."),
        ToolCallStart(conversation_id="c1", tool_call_id="tc1", name="add"),
        ReasoningBlock(conversation_id="c1", text="Let me add."),
        FinishReason(conversation_id="c1", finish_reason="tool_use"),
        ToolCallReceived(conversation_id="c1", tool_call_id="tc1", execution=birth),
        ToolExecutionStarted(conversation_id="c1", tool_call_id="tc1", execution=running),
        ToolExecuted(
            conversation_id="c1",
            tool_call_id="tc1",
            execution=final,
            result_text="3",
            is_error=False,
        ),
        TextStart(conversation_id="c1"),
        TextDelta(conversation_id="c1", text="It's 3."),
        TextBlock(conversation_id="c1", text="It's 3."),
        FinishReason(conversation_id="c1", finish_reason="stop"),
    ]


# ── preflight failure modes (terminal at creation; the policy never sees them) ──


async def test_unknown_tool_records_not_found_and_skips_strategy():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("nope", {"x": 1}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("ok")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_unknown",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="go")]),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    registry = FakeToolRegistry([AddTool()], decisions=[])  # empty script:
    # any decide() would raise
    runner = DeterministicRunner(
        session,
        tool_registry=registry,
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
    )

    async with runner.run() as run:
        events = [event async for event in run]

    birth = ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="nope", arguments={"x": 1}),
        tool_spec=None,  # no partial specification is fabricated
        status=ExecutionStatus.NOT_FOUND,
        error=ToolExecutionError(
            error_type="ToolNotFound",
            error_message="Unknown tool: 'nope'.",
        ),  # the registry authored this one — no runner phase
        finished_at=1000,
        context_tokens=5,  # the structured error message // 4
    )
    final = birth.model_copy(update={"updated_at": 1000})
    assert events == [
        FinishReason(conversation_id="c1", finish_reason="tool_use"),
        ToolCallReceived(conversation_id="c1", tool_call_id="tc1", execution=birth),
        ToolExecuted(
            conversation_id="c1",
            tool_call_id="tc1",
            execution=final,
            result_text="Unknown tool: 'nope'.",
            is_error=True,
        ),
        TextBlock(conversation_id="c1", text="ok"),
        FinishReason(conversation_id="c1", finish_reason="stop"),
    ]
    assert runner.session.entries["te1"] == final
    assert registry.seen == []  # a terminal birth is never decided
    assert registry.prepared == []  # nor prepared
    assert runner.idle()


async def test_invalid_arguments_record_invalid_and_skip_strategy():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("ok")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_badargs",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="go")]),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    registry = FakeToolRegistry([AddTool()], decisions=[])  # empty script:
    # any decide() would raise
    runner = DeterministicRunner(
        session,
        tool_registry=registry,
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
    )

    async with runner.run() as run:
        events = [event async for event in run]

    birth = ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1}),
        tool_spec=ADD_SPEC,  # the tool RESOLVED; its arguments did not validate
        tool_spec_id=ADD_SPEC_ID,
        status=ExecutionStatus.INVALID,
        error=ToolExecutionError(
            error_type="InvalidToolArguments",
            error_message="Arguments for tool 'add' are invalid.",
            details={"errors": MISSING_B},  # the registry's own payload
        ),
        finished_at=1000,
        context_tokens=9,  # the structured error message // 4
    )
    final = birth.model_copy(update={"updated_at": 1000})
    assert events == [
        FinishReason(conversation_id="c1", finish_reason="tool_use"),
        ToolCallReceived(conversation_id="c1", tool_call_id="tc1", execution=birth),
        ToolExecuted(
            conversation_id="c1",
            tool_call_id="tc1",
            execution=final,
            result_text=INVALID_ADD_TEXT,
            is_error=True,
        ),
        TextBlock(conversation_id="c1", text="ok"),
        FinishReason(conversation_id="c1", finish_reason="stop"),
    ]
    assert runner.session.entries["te1"] == final
    assert registry.seen == []  # a terminal birth is never decided
    assert runner.idle()


async def test_create_execution_failure_records_failed_with_its_phase():
    # the registry raised at the birth door: nothing resolved, so no
    # tool_spec, and the phase says where the runner observed the raise
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("ok")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_birth_raise",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="go")]),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    registry = BrokenBirthRegistry([AddTool()], decisions=[])  # empty script:
    # any decide() would raise
    runner = DeterministicRunner(
        session,
        tool_registry=registry,
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
    )

    async with runner.run() as run:
        events = [event async for event in run]

    birth = ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        tool_spec=None,
        status=ExecutionStatus.FAILED,
        error=ToolExecutionError(
            error_type="RuntimeError",
            error_message="registry unavailable",
            details={"phase": "create_execution"},
        ),
        finished_at=1000,
        context_tokens=5,  # the structured error message // 4
    )
    final = birth.model_copy(update={"updated_at": 1000})
    assert events == [
        FinishReason(conversation_id="c1", finish_reason="tool_use"),
        ToolCallReceived(conversation_id="c1", tool_call_id="tc1", execution=birth),
        ToolExecuted(
            conversation_id="c1",
            tool_call_id="tc1",
            execution=final,
            result_text="Tool execution failed: RuntimeError: registry unavailable",
            is_error=True,
        ),
        TextBlock(conversation_id="c1", text="ok"),
        FinishReason(conversation_id="c1", finish_reason="stop"),
    ]
    assert runner.session.entries["te1"] == final
    assert registry.seen == []  # a terminal birth is never decided
    assert runner.session.tool_specs == {}  # nothing resolved, nothing filed
    assert runner.idle()


# ── dispatch-time failures (the body is never dispatched) ──────────────────────


async def test_prepare_resolution_failure_records_not_found_undispatched():
    # the call resolved at birth and vanished before dispatch: NOT_FOUND with
    # attempts=[] and dispatched=False, and NO ToolExecutionStarted — the
    # event fires if and only if the body was dispatched
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("ok")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_vanished",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="go")]),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        tool_registry=VanishingToolRegistry([AddTool()]),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
    )

    async with runner.run() as run:
        events = [event async for event in run]

    birth = ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        tool_spec=ADD_SPEC,
        tool_spec_id=ADD_SPEC_ID,
        status=ExecutionStatus.PENDING,
    )
    final = birth.model_copy(
        update={
            "status": ExecutionStatus.NOT_FOUND,
            "error": ToolExecutionError(
                error_type="ToolNotFound",
                error_message="Unknown tool: 'add'.",
                details={"phase": "prepare"},
            ),
            "approval_status": ApprovalStatus.ALLOWED,
            "approval_decisions": [ALLOW_1000],
            "finished_at": 1000,
            "updated_at": 1000,
            "context_tokens": 5,  # the structured error message // 4
        }
    )
    assert events == [
        FinishReason(conversation_id="c1", finish_reason="tool_use"),
        ToolCallReceived(conversation_id="c1", tool_call_id="tc1", execution=birth),
        ToolExecuted(
            conversation_id="c1",
            tool_call_id="tc1",
            execution=final,
            result_text="Unknown tool: 'add'.",
            is_error=True,
        ),
        TextBlock(conversation_id="c1", text="ok"),
        FinishReason(conversation_id="c1", finish_reason="stop"),
    ]
    assert runner.session.entries["te1"] == final
    assert runner.idle()


async def test_prepare_validation_failure_records_invalid_with_phase_and_errors():
    # a `prepare` that rejects the arguments: INVALID, undispatched, and the
    # structured list nests alongside the phase rather than replacing it
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("ok")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_prepare_invalid",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="go")]),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        tool_registry=RejectingArgumentsRegistry([AddTool()]),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
    )

    async with runner.run() as run:
        events = [event async for event in run]

    birth = ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        tool_spec=ADD_SPEC,
        tool_spec_id=ADD_SPEC_ID,
        status=ExecutionStatus.PENDING,
    )
    final = birth.model_copy(
        update={
            "status": ExecutionStatus.INVALID,
            "error": ToolExecutionError(
                error_type="InvalidToolArguments",
                error_message="Arguments for tool 'add' are invalid.",
                details={"phase": "prepare", "errors": MISSING_B},
            ),
            "approval_status": ApprovalStatus.ALLOWED,
            "approval_decisions": [ALLOW_1000],
            "finished_at": 1000,
            "updated_at": 1000,
            "context_tokens": 9,  # the structured error message // 4
        }
    )
    assert events == [
        FinishReason(conversation_id="c1", finish_reason="tool_use"),
        ToolCallReceived(conversation_id="c1", tool_call_id="tc1", execution=birth),
        ToolExecuted(
            conversation_id="c1",
            tool_call_id="tc1",
            execution=final,
            result_text=INVALID_ADD_TEXT,
            is_error=True,
        ),
        TextBlock(conversation_id="c1", text="ok"),
        FinishReason(conversation_id="c1", finish_reason="stop"),
    ]
    assert runner.session.entries["te1"] == final
    assert runner.idle()


async def test_prepare_returning_a_non_callable_records_failed():
    # the registry broke the contract; the runner records a preparation
    # failure instead of crashing the run at invocation time
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("ok")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_not_callable",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="go")]),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        tool_registry=NonCallablePrepareRegistry([AddTool()]),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
    )

    async with runner.run() as run:
        events = [event async for event in run]

    birth = ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        tool_spec=ADD_SPEC,
        tool_spec_id=ADD_SPEC_ID,
        status=ExecutionStatus.PENDING,
    )
    final = birth.model_copy(
        update={
            "status": ExecutionStatus.FAILED,
            "error": ToolExecutionError(
                error_type="AgentError",
                error_message=("prepare() for tool 'add' returned NoneType, which is not callable."),
                details={"phase": "prepare"},
            ),
            "approval_status": ApprovalStatus.ALLOWED,
            "approval_decisions": [ALLOW_1000],
            "finished_at": 1000,
            "updated_at": 1000,
            "context_tokens": 16,  # the structured error message // 4
        }
    )
    assert events == [
        FinishReason(conversation_id="c1", finish_reason="tool_use"),
        ToolCallReceived(conversation_id="c1", tool_call_id="tc1", execution=birth),
        ToolExecuted(
            conversation_id="c1",
            tool_call_id="tc1",
            execution=final,
            result_text=(
                "Tool execution failed: AgentError: prepare() for tool 'add' returned NoneType, which is not callable."
            ),
            is_error=True,
        ),
        TextBlock(conversation_id="c1", text="ok"),
        FinishReason(conversation_id="c1", finish_reason="stop"),
    ]
    assert runner.session.entries["te1"] == final
    assert runner.idle()


# ── post-dispatch failure modes (the body ran) ─────────────────────────────────


async def test_raising_tool_records_failed_with_structured_error():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("boom", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("ok")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_boom",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="go")]),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([RaisingTool()]),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
    )

    async with runner.run() as run:
        events = [event async for event in run]

    birth = ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="boom", arguments={"a": 1, "b": 2}),
        tool_spec=BOOM_SPEC,
        tool_spec_id=BOOM_SPEC_ID,
        status=ExecutionStatus.PENDING,
    )
    running = birth.model_copy(
        update={
            "status": ExecutionStatus.RUNNING,
            "approval_status": ApprovalStatus.ALLOWED,
            "approval_decisions": [ALLOW_1000],
            "attempts": [ExecutionAttempt(started_at=1000)],
            "updated_at": 1000,
        }
    )
    final = running.model_copy(
        update={
            "status": ExecutionStatus.FAILED,
            "error": ToolExecutionError(
                error_type="ValueError",
                error_message="kaboom",
                details={"phase": "execution"},
            ),
            "attempts": [ExecutionAttempt(outcome=ExecutionAttemptOutcome.FAILED, started_at=1000, ended_at=1000)],
            "finished_at": 1000,
            "context_tokens": 1,  # len("kaboom") // 4
        }
    )
    assert events == [
        FinishReason(conversation_id="c1", finish_reason="tool_use"),
        ToolCallReceived(conversation_id="c1", tool_call_id="tc1", execution=birth),
        ToolExecutionStarted(conversation_id="c1", tool_call_id="tc1", execution=running),
        ToolExecuted(
            conversation_id="c1",
            tool_call_id="tc1",
            execution=final,
            result_text="Tool execution failed: ValueError: kaboom",
            is_error=True,
        ),
        TextBlock(conversation_id="c1", text="ok"),
        FinishReason(conversation_id="c1", finish_reason="stop"),
    ]
    assert runner.session.entries["te1"] == final
    assert runner.idle()


async def test_tool_body_raising_tool_not_found_records_failed():
    # resolution and validation already happened in prepare(), so a body that
    # raises ToolNotFound looking up a sub-resource is a tool FAILURE — the
    # exception-type mapping does not apply once the body was dispatched
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("lookup", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("ok")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_lookup",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="go")]),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([LookupTool()]),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
    )

    async with runner.run() as run:
        events = [event async for event in run]

    birth = ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="lookup", arguments={"a": 1, "b": 2}),
        tool_spec=LOOKUP_SPEC,
        tool_spec_id=LOOKUP_SPEC_ID,
        status=ExecutionStatus.PENDING,
    )
    running = birth.model_copy(
        update={
            "status": ExecutionStatus.RUNNING,
            "approval_status": ApprovalStatus.ALLOWED,
            "approval_decisions": [ALLOW_1000],
            "attempts": [ExecutionAttempt(started_at=1000)],
            "updated_at": 1000,
        }
    )
    final = running.model_copy(
        update={
            "status": ExecutionStatus.FAILED,  # NOT_FOUND is a prepare-only verdict
            "error": ToolExecutionError(
                error_type="ToolNotFound",
                error_message="no such record: 7",
                details={"phase": "execution"},
            ),
            "attempts": [ExecutionAttempt(outcome=ExecutionAttemptOutcome.FAILED, started_at=1000, ended_at=1000)],
            "finished_at": 1000,
            "context_tokens": 4,  # the structured error message // 4
        }
    )
    assert events == [
        FinishReason(conversation_id="c1", finish_reason="tool_use"),
        ToolCallReceived(conversation_id="c1", tool_call_id="tc1", execution=birth),
        ToolExecutionStarted(conversation_id="c1", tool_call_id="tc1", execution=running),
        ToolExecuted(
            conversation_id="c1",
            tool_call_id="tc1",
            execution=final,
            result_text="Tool execution failed: ToolNotFound: no such record: 7",
            is_error=True,
        ),
        TextBlock(conversation_id="c1", text="ok"),
        FinishReason(conversation_id="c1", finish_reason="stop"),
    ]
    assert runner.session.entries["te1"] == final
    assert runner.idle()


async def test_rich_is_error_result_is_still_completed():
    # is_error is the TOOL's verdict about its result — the framework received
    # a result, so the execution is COMPLETED, not FAILED.
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("report", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("ok")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_rich",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="go")]),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([RichErrorTool()]),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
    )

    async with runner.run() as run:
        events = [event async for event in run]

    birth = ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="report", arguments={"a": 1, "b": 2}),
        tool_spec=REPORT_SPEC,
        tool_spec_id=REPORT_SPEC_ID,
        status=ExecutionStatus.PENDING,
    )
    running = birth.model_copy(
        update={
            "status": ExecutionStatus.RUNNING,
            "approval_status": ApprovalStatus.ALLOWED,
            "approval_decisions": [ALLOW_1000],
            "attempts": [ExecutionAttempt(started_at=1000)],
            "updated_at": 1000,
        }
    )
    final = running.model_copy(
        update={
            "status": ExecutionStatus.COMPLETED,
            "result": ExecutionResult(
                content=[TextContent(text="disk full")],
                metadata={"code": 28},
                is_error=True,
            ),
            "error": None,  # a result exists: nothing to record as a failure
            "attempts": [ExecutionAttempt(outcome=ExecutionAttemptOutcome.COMPLETED, started_at=1000, ended_at=1000)],
            "finished_at": 1000,
            "context_tokens": 2,  # len("disk full") // 4
        }
    )
    assert events == [
        FinishReason(conversation_id="c1", finish_reason="tool_use"),
        ToolCallReceived(conversation_id="c1", tool_call_id="tc1", execution=birth),
        ToolExecutionStarted(conversation_id="c1", tool_call_id="tc1", execution=running),
        ToolExecuted(
            conversation_id="c1",
            tool_call_id="tc1",
            execution=final,
            result_text="disk full",
            is_error=True,
        ),
        TextBlock(conversation_id="c1", text="ok"),
        FinishReason(conversation_id="c1", finish_reason="stop"),
    ]
    assert runner.session.entries["te1"] == final
    assert runner.idle()


# ── round keying (the tool calls, not finish_reason) ──────────────────────────


async def test_stop_finish_with_tool_calls_still_executes_the_round():
    # the round keys off the tool_calls themselves: a provider misclassifying
    # the finish as "stop" must not leave dangling tool_use blocks behind
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="stop",
            ),
            faux_assistant_message([faux_text("It's 3.")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_stop_with_calls",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Add")]),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()]),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
    )

    async with runner.run() as run:
        events = [event async for event in run]

    assert events[0] == FinishReason(conversation_id="c1", finish_reason="stop")  # recorded verbatim
    assert [event.type for event in events] == [
        "finish_reason",
        "tool_call_received",
        "tool_execution_started",
        "tool_executed",
        "text_block",
        "finish_reason",
    ]
    assert len(faux.requests) == 2
    assert runner.session.entries["te1"].status == ExecutionStatus.COMPLETED
    assert runner.session.entries["tf"].outcome == TurnOutcome.COMPLETED
    assert runner.idle()


async def test_tool_use_finish_with_no_calls_closes_the_turn():
    # the inverse misclassification must not loop the model forever
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message([faux_text("Hmm.")], finish_reason="tool_use"),
        ]
    )
    session = make_session(
        id="s_tool_use_no_calls",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")]),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()]),
        provider=faux,
        ids=["ts", "a1", "tf"],
        now=1000,
    )

    result = await runner.run()

    assert result.outcome == TurnOutcome.COMPLETED
    assert len(faux.requests) == 1  # exactly one call — no spin
    assert runner.session.entries["a1"].stop_reason == "tool_use"  # verbatim
    assert runner.session.entries["tf"].outcome == TurnOutcome.COMPLETED
    assert runner.idle()


# ── post_message / state guards / the live session ──────────────────────────────


async def test_post_message_sets_pending():
    session = make_session(
        id="s_pm",
        conversations={"c1": conversation("c1", [], created_at=900, updated_at=900)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(session, ids=["u1"], now=1000)

    assert runner.idle()
    msg_id = runner.post_message("Hello")

    assert msg_id == "u1"
    assert runner.busy()
    assert runner.session.entries["u1"] == UserMessage(
        id="u1",
        parent_id=None,
        created_at=1000,
        parts=[TextContent(text="Hello")],
        context_tokens=1,  # len("Hello") // 4
    )
    assert main_conversation(runner.session) == Conversation(
        id="c1",
        nodes=["u1"],
        created_at=900,
        updated_at=1000,
    )


async def test_post_message_accepts_a_part_list_and_keeps_its_order():
    session = make_session(
        id="s_pm_parts",
        conversations={"c1": conversation("c1", [], created_at=900, updated_at=900)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(session, ids=["u1"], now=1000)
    image = ImageContent(
        source=MediaBase64(data="aGk=", media_type="image/png"),
        metadata={"name": "a.png"},
    )

    runner.post_message([image, TextContent(text="Hello")])

    assert runner.session.entries["u1"] == UserMessage(
        id="u1",
        parent_id=None,
        created_at=1000,
        parts=[image, TextContent(text="Hello")],
        context_tokens=1_001,  # IMAGE_TOKENS + len("Hello") // 4
    )


async def test_post_message_rejects_empty_input():
    # a post with nothing in it is meaningless; "" used to persist an empty
    # text part, which some providers reject on the wire
    session = make_session(
        id="s_pm_empty",
        conversations={"c1": conversation("c1", [], created_at=900, updated_at=900)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(session, ids=["u1"], now=1000)

    for empty in ("", [], None):
        with pytest.raises(AgentError, match="non-empty"):
            runner.post_message(empty)


async def test_post_message_keeps_a_whitespace_only_string():
    # only *empty* is rejected — what counts as meaningful text is the
    # application's call, not the runner's
    session = make_session(
        id="s_pm_ws",
        conversations={"c1": conversation("c1", [], created_at=900, updated_at=900)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(session, ids=["u1"], now=1000)

    runner.post_message("   ")

    assert runner.session.entries["u1"].parts == [TextContent(text="   ")]


async def test_post_message_rejects_a_part_the_union_does_not_admit():
    session = make_session(
        id="s_pm_bad",
        conversations={"c1": conversation("c1", [], created_at=900, updated_at=900)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(session, ids=["u1"], now=1000)

    for rejected in (
        [ThinkingContent(thinking="not a user part")],
        [ToolCall(id="tc1", name="add")],
        ["a bare string"],
        [[TextContent(text="nested")]],
    ):
        with pytest.raises(ValidationError):
            runner.post_message(rejected)


async def test_post_message_validates_raw_dicts_into_parts():
    # shape is checked against ContentPart itself, so the wire form works too
    session = make_session(
        id="s_pm_dicts",
        conversations={"c1": conversation("c1", [], created_at=900, updated_at=900)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(session, ids=["u1"], now=1000)

    runner.post_message(
        [
            {
                "type": "image",
                "source": {
                    "kind": "base64",
                    "data": "aGk=",
                    "media_type": "image/png",
                },
            },
            {"type": "text", "text": "what is this?"},
        ]
    )

    assert runner.session.entries["u1"].parts == [
        ImageContent(source=MediaBase64(data="aGk=", media_type="image/png")),
        TextContent(text="what is this?"),
    ]


async def test_post_message_does_not_copy_the_parts_it_is_given():
    session = make_session(
        id="s_pm_ident",
        conversations={"c1": conversation("c1", [], created_at=900, updated_at=900)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(session, ids=["u1"], now=1000)
    part = TextContent(text="Hello")

    runner.post_message([part])

    assert runner.session.entries["u1"].parts[0] is part


async def test_run_when_idle_raises():
    session = make_session(
        id="s_idle",
        conversations={"c1": conversation("c1", [], created_at=900, updated_at=900)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(session, now=1000)

    with pytest.raises(AgentError):
        async with runner.run() as run:
            _ = [event async for event in run]


async def test_tool_body_receives_the_live_session_and_the_run_token():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("capture", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("done")], finish_reason="stop"),
        ]
    )
    tool = CapturingTool()
    session = make_session(
        id="s_ctx",
        entries={
            "u1": UserMessage(id="u1", created_at=0, parts=[TextContent(text="go")]),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=0, updated_at=0)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([tool]),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    # there is no ToolContext: the body is handed the session object the
    # runner and ledger write through, by reference
    assert tool.seen == [runner.session]
    assert tool.seen[0] is runner.session
    token = tool.tokens[0]
    assert isinstance(token, CancellationToken)
    assert token.cancelled is False


# ── event snapshots are immutable ───────────────────────────────────────────────


async def test_event_snapshots_do_not_track_later_ledger_updates():
    # ToolCallReceived shows the birth state and ToolExecutionStarted the
    # RUNNING state even after the ledger entry reached COMPLETED — events
    # carry deep snapshots, never live references.
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("It's 3.")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_snapshots",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Add")]),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()]),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
    )

    async with runner.run() as run:
        events = [event async for event in run]

    birth = ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        tool_spec=ADD_SPEC,
        tool_spec_id=ADD_SPEC_ID,
        status=ExecutionStatus.PENDING,
    )
    running = birth.model_copy(
        update={
            "status": ExecutionStatus.RUNNING,
            "approval_status": ApprovalStatus.ALLOWED,
            "approval_decisions": [ALLOW_1000],
            "attempts": [ExecutionAttempt(started_at=1000)],
            "updated_at": 1000,
        }
    )
    assert events[1].execution == birth
    assert events[2].execution == running
    assert runner.session.entries["te1"] == running.model_copy(
        update={
            "status": ExecutionStatus.COMPLETED,
            "result": ExecutionResult(content=[TextContent(text="3")], is_error=False),
            "attempts": [ExecutionAttempt(outcome=ExecutionAttemptOutcome.COMPLETED, started_at=1000, ended_at=1000)],
            "finished_at": 1000,
        }
    )


# ── serialization invariants ────────────────────────────────────────────────────


async def test_completed_session_round_trips_and_rederives_status():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("It's 3.")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_rt",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Add")]),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()]),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
    )
    async with runner.run() as run:
        _ = [event async for event in run]

    reloaded = AgentSession.model_validate_json(runner.session.model_dump_json())

    # the persisted payload round-trips losslessly — approval state, the
    # structured lifecycle timestamps, and the normalized tool spec (stripped
    # from the entry on the way out, restored from `tool_specs` on the way
    # back); nothing on the session is transient (the tool registry and
    # projector live on the runner)
    assert reloaded == runner.session
    # a fresh production runner over the reloaded data re-derives the status
    rebuilt = AgentSessionRunner(
        reloaded,
        tool_registry=FakeToolRegistry([AddTool()]),
        provider=faux,
    )
    assert rebuilt.status == ConversationStatus.IDLE


async def test_post_message_names_the_mistake_when_a_part_is_not_in_a_list():
    # an easy caller slip; pydantic would iterate the model field by field
    # and report something unreadable about tuples
    session = make_session(
        id="s_pm_bare",
        conversations={"c1": conversation("c1", [], created_at=900, updated_at=900)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(session, ids=["u1"], now=1000)

    with pytest.raises(AgentError, match="wrap the TextContent in a list"):
        runner.post_message(TextContent(text="hi"))


async def test_an_unknown_client_block_raises_before_anything_durable_is_written(monkeypatch):
    # The adapter's loud fall-through fires INSIDE `_record_assistant`, before
    # the assistant entry is appended: the raise reaches the caller and the
    # session shows an open turn with NO assistant entry — nothing durable was
    # written for the response. (The faux cannot script an unknown block, so
    # the LLM boundary is mocked at the runner's import site, the
    # test_native_tools pattern.)
    import luca.agent.core.runner as runner_module
    from luca.client.types import (
        AssistantMessage as ClientAssistantMessage,
        ChatCompletionResponse,
        ToolResultBlock,
    )

    async def mock_acompletion(*args, **kwargs):
        message = ClientAssistantMessage.model_construct(
            content=[ToolResultBlock(tool_call_id="tc1", content="output")],
            finish_reason="stop",
            usage=ClientUsage(),
            tool_calls=[],
        )
        return ChatCompletionResponse(messages=[message])

    monkeypatch.setattr(runner_module, "acompletion", mock_acompletion)
    session = make_session(
        id="s1",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")]),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(session, ids=["ts", "a1"], now=1000)

    with pytest.raises(AgentError, match="tool_result"):
        await runner.run()

    assert runner.session.entries == {
        "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")]),
        "ts": TurnStart(id="ts", parent_id="u1", created_at=1000),
    }


# ── hosted web content (phase-6 core wiring; the PLUGIN lives in contrib) ─────


async def test_web_deltas_interleave_before_their_block_events():
    # deltas narrate the operation DURING the call; the block events derive
    # from the RECORDED parts — after the record, before FinishReason
    from luca.agent.core.events import (
        WebOperationEnd,
        WebOperationStart,
        WebSearchBlock as WebSearchBlockEvent,
        WebSearchQueries,
        WebSearchResults,
    )
    from luca.agent.core.models import WebPageContent
    from luca.client.types import (
        PrivateProviderBlock as ClientPrivateProviderBlock,
        WebPagePart as ClientWebPagePart,
        WebSearchBlock as ClientWebSearchBlock,
    )

    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [
                    ClientPrivateProviderBlock(format="anthropic.messages", data={"type": "server_tool_use"}),
                    ClientWebSearchBlock(
                        queries=["apple"],
                        results=[ClientWebPagePart(url="https://apple.com", title="Apple")],
                        extras={"id": "srv_1"},
                    ),
                    faux_text("Apple is up."),
                ],
                finish_reason="stop",
            ),
        ]
    )
    session = make_session(
        id="s_web_stream",
        entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="find apple")])},
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(session, provider=faux, ids=["ts", "a1", "tf"], now=1000)

    async with runner.run(streaming=True) as run:
        events = [event async for event in run]

    results = [WebPageContent(url="https://apple.com", title="Apple")]
    assert events == [
        WebOperationStart(conversation_id="c1", id="srv_1"),
        WebSearchQueries(conversation_id="c1", id="srv_1", queries=["apple"]),
        WebSearchResults(conversation_id="c1", id="srv_1", results=results),
        WebOperationEnd(conversation_id="c1", id="srv_1"),
        TextStart(conversation_id="c1"),
        TextDelta(conversation_id="c1", text="Apple is up."),
        WebSearchBlockEvent(conversation_id="c1", id="srv_1", queries=["apple"], results=results),
        TextBlock(conversation_id="c1", text="Apple is up."),
        FinishReason(conversation_id="c1", finish_reason="stop"),
    ]


async def test_streaming_produces_same_session_as_run_with_web_parts():
    # same script, same ids, both modes: byte-identical sessions — the block
    # tier and the durable record never depend on streaming
    from luca.client.types import (
        PrivateProviderBlock as ClientPrivateProviderBlock,
        URLCitationAnnotation,
        WebFetchBlock as ClientWebFetchBlock,
        WebPagePart as ClientWebPagePart,
        WebSearchBlock as ClientWebSearchBlock,
    )

    def scripted_faux() -> FauxProvider:
        faux = FauxProvider()
        faux.set_responses(
            [
                faux_assistant_message(
                    [
                        ClientPrivateProviderBlock(format="anthropic.messages", data={"type": "server_tool_use"}),
                        ClientWebSearchBlock(queries=["apple"], extras={"id": "srv_1"}),
                        ClientWebFetchBlock(
                            web_page=ClientWebPagePart(url="https://apple.com"), extras={"id": "srv_2"}
                        ),
                        faux_text(
                            "Apple is up.",
                            annotations=[
                                URLCitationAnnotation(
                                    url="https://apple.com", title="Apple", start_index=0, end_index=12
                                )
                            ],
                        ),
                    ],
                    finish_reason="stop",
                ),
            ]
        )
        return faux

    def fresh() -> AgentSession:
        return make_session(
            id="s_web_parity",
            entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="find apple")])},
            conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
            main_conversation_id="c1",
            session_config=SessionConfig(llm_config=MODEL),
        )

    streamed = DeterministicRunner(fresh(), provider=scripted_faux(), ids=["ts", "a1", "tf"], now=1000)
    plain = DeterministicRunner(fresh(), provider=scripted_faux(), ids=["ts", "a1", "tf"], now=1000)

    async with streamed.run(streaming=True) as run:
        _ = [event async for event in run]
    await plain.run()

    assert streamed.session == plain.session


def test_to_delta_event_maps_all_seven_web_events():
    from luca.agent.core.events import (
        TextAnnotation,
        WebFetchUrls,
        WebFindInPage,
        WebOperationEnd,
        WebOperationStart,
        WebSearchQueries,
        WebSearchResults,
    )
    from luca.agent.core.models import URLCitation, WebPageContent
    from luca.agent.core.runner import _to_delta_event
    from luca.client.types import URLCitationAnnotation, WebPagePart as ClientWebPagePart
    from luca.client.types.streaming import (
        TextAnnotationEvent,
        WebEndEvent,
        WebFetchEvent,
        WebFindEvent,
        WebSearchEvent,
        WebSearchResultEvent,
        WebStartEvent,
    )

    assert _to_delta_event("c1", WebStartEvent(id="w1")) == WebOperationStart(conversation_id="c1", id="w1")
    assert _to_delta_event("c1", WebSearchEvent(id="w1", queries=["apple"])) == WebSearchQueries(
        conversation_id="c1", id="w1", queries=["apple"]
    )
    assert _to_delta_event(
        "c1", WebSearchResultEvent(id="w1", results=[ClientWebPagePart(url="https://a", title="A")])
    ) == WebSearchResults(conversation_id="c1", id="w1", results=[WebPageContent(url="https://a", title="A")])
    assert _to_delta_event("c1", WebFetchEvent(id="w2", urls=["https://a"])) == WebFetchUrls(
        conversation_id="c1", id="w2", urls=["https://a"]
    )
    # WebFindInPage is reachable ONLY here: find-in-page has no portable
    # block, so the faux cannot script it
    assert _to_delta_event("c1", WebFindEvent(id="w3", url="https://a", pattern="R&D")) == WebFindInPage(
        conversation_id="c1", id="w3", url="https://a", pattern="R&D"
    )
    assert _to_delta_event("c1", WebEndEvent(id="w1")) == WebOperationEnd(conversation_id="c1", id="w1")
    assert _to_delta_event(
        "c1",
        TextAnnotationEvent(
            index=0, annotation=URLCitationAnnotation(url="https://a", title="A", start_index=0, end_index=5)
        ),
    ) == TextAnnotation(
        conversation_id="c1", annotation=URLCitation(url="https://a", title="A", start_index=0, end_index=5)
    )


# ── pause-and-replay (D8) ─────────────────────────────────────────────────────


async def test_a_pause_continues_and_replays():
    from luca.agent.core.events import ResponsePaused, ResponseResumed

    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message([faux_text("Searching...")], finish_reason="pause"),
            faux_assistant_message([faux_text("Done.")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_pause",
        entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="find apple")])},
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(session, provider=faux, ids=["ts", "a1", "a2", "tf"], now=1000)

    async with runner.run() as run:
        events = [event async for event in run]

    assert events == [
        TextBlock(conversation_id="c1", text="Searching..."),
        FinishReason(conversation_id="c1", finish_reason="pause"),
        ResponsePaused(conversation_id="c1", finish_reason="pause", provider_finish_reason="pause"),
        ResponseResumed(conversation_id="c1"),
        TextBlock(conversation_id="c1", text="Done."),
        FinishReason(conversation_id="c1", finish_reason="stop"),
    ]
    # the replay really happened: two requests, the second carrying the
    # paused entry as the provider's continuation shape — TRAILING, verbatim
    assert len(faux.requests) == 2
    from luca.client.types import AssistantMessage as LucaAssistantMessage, TextBlock as LucaTextBlock

    assert faux.requests[1].messages[-1] == LucaAssistantMessage(
        content=[LucaTextBlock(text="Searching...")],
        provider="faux",
        model="test-model",
    )
    assert runner.session == make_session(
        id="s_pause",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="find apple")]),
            "ts": TurnStart(id="ts", parent_id="u1", created_at=1000),
            "a1": AssistantMessage(
                id="a1",
                parent_id="ts",
                created_at=1000,
                parts=[TextContent(text="Searching...")],
                llm_config=MODEL,
                stop_reason="pause",  # the durable marker
                context_tokens=3,  # len("Searching...") // 4
            ),
            "a2": AssistantMessage(
                id="a2",
                parent_id="a1",
                created_at=1000,
                parts=[TextContent(text="Done.")],
                llm_config=MODEL,
                stop_reason="stop",
                context_tokens=1,  # len("Done.") // 4
            ),
            "tf": TurnFinish(id="tf", parent_id="a2", created_at=1000),
        },
        usages={
            "c1": {
                "a1": Usage(conversation_id="c1", entry_id="a1"),
                "a2": Usage(conversation_id="c1", entry_id="a2"),
            }
        },
        conversations={"c1": conversation("c1", ["u1", "ts", "a1", "a2", "tf"], created_at=500, updated_at=1000)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )


async def test_a_streamed_pause_continues_and_replays():
    from luca.agent.core.events import ResponsePaused, ResponseResumed

    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message([faux_text("Searching...")], finish_reason="pause"),
            faux_assistant_message([faux_text("Done.")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_pause_stream",
        entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="find apple")])},
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(session, provider=faux, ids=["ts", "a1", "a2", "tf"], now=1000)

    async with runner.run(streaming=True) as run:
        events = [event async for event in run]

    assert events == [
        TextStart(conversation_id="c1"),
        TextDelta(conversation_id="c1", text="Searching..."),
        TextBlock(conversation_id="c1", text="Searching..."),
        FinishReason(conversation_id="c1", finish_reason="pause"),
        ResponsePaused(conversation_id="c1", finish_reason="pause", provider_finish_reason="pause"),
        ResponseResumed(conversation_id="c1"),
        TextStart(conversation_id="c1"),
        TextDelta(conversation_id="c1", text="Done."),
        TextBlock(conversation_id="c1", text="Done."),
        FinishReason(conversation_id="c1", finish_reason="stop"),
    ]
    assert len(faux.requests) == 2
    assert [entry.stop_reason for entry in (runner.session.entries["a1"], runner.session.entries["a2"])] == [
        "pause",
        "stop",
    ]


async def test_a_reloaded_paused_session_resumes_with_only_response_resumed():
    # after a crash in the pause window, the durable stop_reason is the whole
    # signal: the session derives BUSY, and the fresh drive replays with only
    # the Resumed half of the pair (Paused fired in the process that died)
    from luca.agent.core.events import ResponsePaused, ResponseResumed
    from luca.agent.core.models import ConversationStatus
    from tests.agent.scenarios import PAUSED_SESSION

    session = AgentSession.model_validate_json(PAUSED_SESSION.model_dump_json())
    assert session.get_conversation_status("c1").status == ConversationStatus.BUSY

    faux = FauxProvider()
    faux.set_responses([faux_assistant_message([faux_text("Done.")], finish_reason="stop")])
    runner = DeterministicRunner(session, provider=faux, ids=["a2", "tf"], now=1000)

    async with runner.run() as run:
        events = [event async for event in run]

    assert events == [
        ResponseResumed(conversation_id="c1"),
        TextBlock(conversation_id="c1", text="Done."),
        FinishReason(conversation_id="c1", finish_reason="stop"),
    ]
    assert not any(isinstance(event, ResponsePaused) for event in events)
    # the replay carried the paused entry as the continuation shape —
    # TRAILING, verbatim
    from luca.client.types import AssistantMessage as LucaAssistantMessage, TextBlock as LucaTextBlock

    assert len(faux.requests) == 1
    assert faux.requests[0].messages[-1] == LucaAssistantMessage(
        content=[LucaTextBlock(text="Searching...")],
        provider="faux",
        model="test-model",
    )
    assert main_conversation(runner.session).nodes == ["u1", "ts", "a1", "a2", "tf"]


async def test_a_post_landing_in_the_pause_window_still_resumes():
    # the Resumed detection tolerates a user post after the paused entry (the
    # documented in-flight shape): the continuation request ends with the
    # post, and the pair's second half still fires
    from luca.agent.core.events import ResponseResumed
    from tests.agent.scenarios import PAUSED_SESSION

    session = PAUSED_SESSION.model_copy(deep=True)
    faux = FauxProvider()
    faux.set_responses([faux_assistant_message([faux_text("Done.")], finish_reason="stop")])
    runner = DeterministicRunner(session, provider=faux, ids=["u2", "a2", "tf"], now=1000)
    runner.post_message("any luck?")

    async with runner.run() as run:
        events = [event async for event in run]

    from luca.client.types import TextBlock as LucaTextBlock, UserMessage as LucaUserMessage

    assert ResponseResumed(conversation_id="c1") in events
    assert faux.requests[0].messages[-1] == LucaUserMessage(content=[LucaTextBlock(text="any luck?")])
    assert faux.requests[0].messages[-2].content[0].text == "Searching..."
