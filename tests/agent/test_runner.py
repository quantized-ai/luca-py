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
them.

Determinism comes from `DeterministicRunner` (`scenarios.py`) — a test-side
subclass overriding the production `generate_id()` / `now_ms()` hooks. Its
`ids` script spans every call (post_message, run, resume) and is consumed in
this order per turn:
  TurnStart, (AssistantMessage, [ToolExecution per call])..., TurnFinish
"""

import pytest

from luca.agent.core.context import CancellationToken
from luca.agent.core.models import (
    AgentSession,
    ApprovalDecision,
    ApprovalOption,
    ApprovalStatus,
    AssistantMessage,
    Conversation,
    ConversationStatus,
    ExecutionResult,
    ExecutionStatus,
    ImageBase64,
    ImageContent,
    SessionConfig,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolExecution,
    ToolExecutionError,
    ToolSpec,
    TurnFinish,
    TurnOutcome,
    TurnStart,
    Usage,
    UserMessage,
)
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
from luca.agent.core.exceptions import AgentError
from luca.agent.core.runner import AgentSessionRunner
from luca.client.testing import (
    FauxProvider,
    faux_assistant_message,
    faux_text,
    faux_thinking,
    faux_tool_call,
)
from luca.client.types import Tool as LucaTool
from luca.client.types import Usage as ClientUsage

from tests.agent.scenarios import (
    MODEL,
    AddTool,
    BinaryArgs,
    CapturingTool,
    DeterministicRunner,
    FakeToolRegistry,
    MultiplyTool,
    RaisingTool,
    RichErrorTool,
)

ALLOW_1000 = ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000)

ADD_SPEC = ToolSpec(name="add", description="Add two numbers.")
MULTIPLY_SPEC = ToolSpec(name="multiply", description="Multiply two numbers.")


# ── plain turns (no approval gate) ─────────────────────────────────────────────


async def test_single_text_response_no_tools():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message([faux_text("Hello!")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s1",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, provider=faux, ids=["ts", "a1", "tf"], now=1000,
    )

    async with runner.run() as run:
        events = [event async for event in run]

    assert events == [
        TextBlock(text="Hello!"),
        FinishReason(finish_reason="stop"),
    ]
    assert runner.session == AgentSession(
        id="s1",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")]),
            "ts": TurnStart(id="ts", parent_id="u1", created_at=1000),
            "a1": AssistantMessage(
                id="a1", parent_id="ts", created_at=1000,
                parts=[TextContent(text="Hello!")],
                llm_config=MODEL, stop_reason="stop",
                context_tokens=1,  # len("Hello!") // 4
            ),
            "tf": TurnFinish(id="tf", parent_id="a1", created_at=1000),
        },
        tool_executions={},
        usages={"c1": {"a1": Usage(conversation_id="c1", entry_id="a1")}},
        active_conversation=Conversation(
            id="c1", nodes=["u1", "ts", "a1", "tf"], created_at=500, updated_at=1000,
            status=ConversationStatus.IDLE,
        ),
        session_config=SessionConfig(llm_config=MODEL),
    )


async def test_run_passes_projected_tools_to_client():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message([faux_text("ok")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_tools",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([AddTool()]), provider=faux,
        ids=["ts", "a1", "tf"], now=1000,
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    assert faux.requests[0].tools == [
        LucaTool(name="add", description="Add two numbers.", parameters=BinaryArgs),
    ]


async def test_reasoning_plus_tool_call_then_text():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_thinking("Let me add."),
             faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("It's 3.")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s3",
        entries={
            "u1": UserMessage(
                id="u1", created_at=500, parts=[TextContent(text="Add 1 and 2")],
            ),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([AddTool()]), provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"], now=1000,
    )

    async with runner.run() as run:
        events = [event async for event in run]

    birth = ToolExecution(
        id="te1", parent_id="a1", created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        tool_spec=ADD_SPEC,
        status=ExecutionStatus.PENDING,
    )
    running = birth.model_copy(update={
        "status": ExecutionStatus.RUNNING,
        "approval_status": ApprovalStatus.ALLOWED,
        "approval_decisions": [ALLOW_1000],
        "started_at": 1000,
        "updated_at": 1000,
    })
    final = running.model_copy(update={
        "status": ExecutionStatus.COMPLETED,
        "result": ExecutionResult(content=[TextContent(text="3")], is_error=False),
        "ended_at": 1000,
    })
    assert events == [
        ReasoningBlock(text="Let me add."),
        FinishReason(finish_reason="tool_use"),
        ToolCallReceived(tool_call_id="tc1", execution=birth),
        ToolExecutionStarted(tool_call_id="tc1", execution=running),
        ToolExecuted(
            tool_call_id="tc1", execution=final, result_text="3", is_error=False,
        ),
        TextBlock(text="It's 3."),
        FinishReason(finish_reason="stop"),
    ]
    assert runner.session == AgentSession(
        id="s3",
        entries={
            "u1": UserMessage(
                id="u1", created_at=500, parts=[TextContent(text="Add 1 and 2")],
            ),
            "ts": TurnStart(id="ts", parent_id="u1", created_at=1000),
            "a1": AssistantMessage(
                id="a1", parent_id="ts", created_at=1000,
                parts=[
                    ThinkingContent(thinking="Let me add."),
                    ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
                ],
                llm_config=MODEL, stop_reason="tool_use",
                context_tokens=7,  # (thinking + name + JSON args) // 4
            ),
            "te1": final,
            "a2": AssistantMessage(
                id="a2", parent_id="te1", created_at=1000,
                parts=[TextContent(text="It's 3.")],
                llm_config=MODEL, stop_reason="stop",
                context_tokens=1,  # len("It's 3.") // 4
            ),
            "tf": TurnFinish(id="tf", parent_id="a2", created_at=1000),
        },
        tool_executions={"tc1": ["te1"]},
        usages={"c1": {
            "a1": Usage(conversation_id="c1", entry_id="a1"),
            "a2": Usage(conversation_id="c1", entry_id="a2"),
        }},
        active_conversation=Conversation(
            id="c1", nodes=["u1", "ts", "a1", "te1", "a2", "tf"],
            created_at=500, updated_at=1000, status=ConversationStatus.IDLE,
        ),
        session_config=SessionConfig(llm_config=MODEL),
    )


async def test_multi_turn_two_tool_rounds_then_text():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_thinking("First add."),
             faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
        ),
        faux_assistant_message(
            [faux_thinking("Now multiply."),
             faux_tool_call("multiply", {"a": 3, "b": 4}, id="tc2")],
            finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("Done: 12")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s4",
        entries={
            "u1": UserMessage(id="u1", created_at=0, parts=[TextContent(text="Go")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=0, updated_at=0),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([AddTool(), MultiplyTool()]),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "te2", "a3", "tf"], now=1000,
    )

    async with runner.run() as run:
        events = [event async for event in run]

    add_birth = ToolExecution(
        id="te1", parent_id="a1", created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        tool_spec=ADD_SPEC,
        status=ExecutionStatus.PENDING,
    )
    add_running = add_birth.model_copy(update={
        "status": ExecutionStatus.RUNNING,
        "approval_status": ApprovalStatus.ALLOWED,
        "approval_decisions": [ALLOW_1000],
        "started_at": 1000,
        "updated_at": 1000,
    })
    add_final = add_running.model_copy(update={
        "status": ExecutionStatus.COMPLETED,
        "result": ExecutionResult(content=[TextContent(text="3")], is_error=False),
        "ended_at": 1000,
    })
    multiply_birth = ToolExecution(
        id="te2", parent_id="a2", created_at=1000,
        tool_call_id="tc2",
        raw_tool_call=ToolCall(id="tc2", name="multiply", arguments={"a": 3, "b": 4}),
        tool_spec=MULTIPLY_SPEC,
        status=ExecutionStatus.PENDING,
    )
    multiply_running = multiply_birth.model_copy(update={
        "status": ExecutionStatus.RUNNING,
        "approval_status": ApprovalStatus.ALLOWED,
        "approval_decisions": [ALLOW_1000],
        "started_at": 1000,
        "updated_at": 1000,
    })
    multiply_final = multiply_running.model_copy(update={
        "status": ExecutionStatus.COMPLETED,
        "result": ExecutionResult(content=[TextContent(text="12")], is_error=False),
        "ended_at": 1000,
    })
    assert events == [
        ReasoningBlock(text="First add."),
        FinishReason(finish_reason="tool_use"),
        ToolCallReceived(tool_call_id="tc1", execution=add_birth),
        ToolExecutionStarted(tool_call_id="tc1", execution=add_running),
        ToolExecuted(
            tool_call_id="tc1", execution=add_final,
            result_text="3", is_error=False,
        ),
        ReasoningBlock(text="Now multiply."),
        FinishReason(finish_reason="tool_use"),
        ToolCallReceived(tool_call_id="tc2", execution=multiply_birth),
        ToolExecutionStarted(tool_call_id="tc2", execution=multiply_running),
        ToolExecuted(
            tool_call_id="tc2", execution=multiply_final,
            result_text="12", is_error=False,
        ),
        TextBlock(text="Done: 12"),
        FinishReason(finish_reason="stop"),
    ]
    assert runner.session == AgentSession(
        id="s4",
        entries={
            "u1": UserMessage(id="u1", created_at=0, parts=[TextContent(text="Go")]),
            "ts": TurnStart(id="ts", parent_id="u1", created_at=1000),
            "a1": AssistantMessage(
                id="a1", parent_id="ts", created_at=1000,
                parts=[
                    ThinkingContent(thinking="First add."),
                    ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
                ],
                llm_config=MODEL, stop_reason="tool_use",
                context_tokens=7,  # (thinking + name + JSON args) // 4
            ),
            "te1": add_final,
            "a2": AssistantMessage(
                id="a2", parent_id="te1", created_at=1000,
                parts=[
                    ThinkingContent(thinking="Now multiply."),
                    ToolCall(id="tc2", name="multiply", arguments={"a": 3, "b": 4}),
                ],
                llm_config=MODEL, stop_reason="tool_use",
                context_tokens=9,  # (thinking + name + JSON args) // 4
            ),
            "te2": multiply_final,
            "a3": AssistantMessage(
                id="a3", parent_id="te2", created_at=1000,
                parts=[TextContent(text="Done: 12")],
                llm_config=MODEL, stop_reason="stop",
                context_tokens=2,  # len("Done: 12") // 4
            ),
            "tf": TurnFinish(id="tf", parent_id="a3", created_at=1000),
        },
        tool_executions={"tc1": ["te1"], "tc2": ["te2"]},
        usages={"c1": {
            "a1": Usage(conversation_id="c1", entry_id="a1"),
            "a2": Usage(conversation_id="c1", entry_id="a2"),
            "a3": Usage(conversation_id="c1", entry_id="a3"),
        }},
        active_conversation=Conversation(
            id="c1", nodes=["u1", "ts", "a1", "te1", "a2", "te2", "a3", "tf"],
            created_at=0, updated_at=1000, status=ConversationStatus.IDLE,
        ),
        session_config=SessionConfig(llm_config=MODEL),
    )


async def test_provider_usage_is_recorded_per_assistant_entry():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
            usage=ClientUsage(
                input_tokens=10, output_tokens=5, total_tokens=15,
                cached_input_tokens=2, cache_write_tokens=1,
            ),
        ),
        faux_assistant_message(
            [faux_text("It's 3.")],
            finish_reason="stop",
            usage=ClientUsage(input_tokens=20, output_tokens=7, total_tokens=27),
        ),
    ])
    session = AgentSession(
        id="s_usage",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Add")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([AddTool()]), provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"], now=1000,
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    # usage is accessory conversation-entry data: one record per assistant
    # entry in the store, nothing embedded on entries or rolled up on markers
    assert runner.session.usages == {
        "c1": {
            "a1": Usage(
                conversation_id="c1", entry_id="a1",
                input=10, output=5, cache_read=2, cache_write=1,
                total_tokens=15,
            ),
            "a2": Usage(
                conversation_id="c1", entry_id="a2",
                input=20, output=7, cache_read=0, cache_write=0,
                total_tokens=27,
            ),
        },
    }
    assert runner.session.entries["tf"] == TurnFinish(
        id="tf", parent_id="a2", created_at=1000,
    )


# ── streaming ──────────────────────────────────────────────────────────────────


async def test_streaming_produces_same_session_as_run():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_thinking("Let me add."),
             faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("It's 3.")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s5",
        entries={
            "u1": UserMessage(
                id="u1", created_at=500, parts=[TextContent(text="Add 1 and 2")],
            ),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([AddTool()]), provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"], now=1000,
    )

    async with runner.run(streaming=True) as run:
        _ = [event async for event in run]

    assert runner.session == AgentSession(
        id="s5",
        entries={
            "u1": UserMessage(
                id="u1", created_at=500, parts=[TextContent(text="Add 1 and 2")],
            ),
            "ts": TurnStart(id="ts", parent_id="u1", created_at=1000),
            "a1": AssistantMessage(
                id="a1", parent_id="ts", created_at=1000,
                parts=[
                    ThinkingContent(thinking="Let me add."),
                    ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
                ],
                llm_config=MODEL, stop_reason="tool_use",
                context_tokens=7,  # (thinking + name + JSON args) // 4
            ),
            "te1": ToolExecution(
                id="te1", parent_id="a1", created_at=1000,
                tool_call_id="tc1",
                raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
                tool_spec=ADD_SPEC,
                status=ExecutionStatus.COMPLETED,
                result=ExecutionResult(
                    content=[TextContent(text="3")], is_error=False,
                ),
                approval_status=ApprovalStatus.ALLOWED,
                approval_decisions=[ALLOW_1000],
                started_at=1000, ended_at=1000,
                updated_at=1000,
            ),
            "a2": AssistantMessage(
                id="a2", parent_id="te1", created_at=1000,
                parts=[TextContent(text="It's 3.")],
                llm_config=MODEL, stop_reason="stop",
                context_tokens=1,  # len("It's 3.") // 4
            ),
            "tf": TurnFinish(id="tf", parent_id="a2", created_at=1000),
        },
        tool_executions={"tc1": ["te1"]},
        usages={"c1": {
            "a1": Usage(conversation_id="c1", entry_id="a1"),
            "a2": Usage(conversation_id="c1", entry_id="a2"),
        }},
        active_conversation=Conversation(
            id="c1", nodes=["u1", "ts", "a1", "te1", "a2", "tf"],
            created_at=500, updated_at=1000, status=ConversationStatus.IDLE,
        ),
        session_config=SessionConfig(llm_config=MODEL),
    )


async def test_streaming_emits_delta_then_block_events_in_order():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_thinking("Let me add."),
             faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("It's 3.")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s6",
        entries={
            "u1": UserMessage(id="u1", created_at=0, parts=[TextContent(text="Add")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=0, updated_at=0),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([AddTool()]), provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"], now=1000,
    )

    async with runner.run(streaming=True) as run:
        events = [event async for event in run]

    birth = ToolExecution(
        id="te1", parent_id="a1", created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        tool_spec=ADD_SPEC,
        status=ExecutionStatus.PENDING,
    )
    running = birth.model_copy(update={
        "status": ExecutionStatus.RUNNING,
        "approval_status": ApprovalStatus.ALLOWED,
        "approval_decisions": [ALLOW_1000],
        "started_at": 1000,
        "updated_at": 1000,
    })
    final = running.model_copy(update={
        "status": ExecutionStatus.COMPLETED,
        "result": ExecutionResult(content=[TextContent(text="3")], is_error=False),
        "ended_at": 1000,
    })
    assert events == [
        ReasoningStart(),
        ReasoningDelta(text="Let me add."),
        ToolCallStart(tool_call_id="tc1", name="add"),
        ReasoningBlock(text="Let me add."),
        FinishReason(finish_reason="tool_use"),
        ToolCallReceived(tool_call_id="tc1", execution=birth),
        ToolExecutionStarted(tool_call_id="tc1", execution=running),
        ToolExecuted(
            tool_call_id="tc1", execution=final, result_text="3", is_error=False,
        ),
        TextStart(),
        TextDelta(text="It's 3."),
        TextBlock(text="It's 3."),
        FinishReason(finish_reason="stop"),
    ]


# ── preflight failure modes (terminal at creation; the policy never sees them) ──


async def test_unknown_tool_records_not_found_and_skips_strategy():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("nope", {"x": 1}, id="tc1")], finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("ok")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_unknown",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="go")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    registry = FakeToolRegistry([AddTool()], decisions=[])  # empty script:
    # any decide() would raise
    runner = DeterministicRunner(
        session, tool_registry=registry, provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"], now=1000,
    )

    async with runner.run() as run:
        events = [event async for event in run]

    birth = ToolExecution(
        id="te1", parent_id="a1", created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="nope", arguments={"x": 1}),
        tool_spec=None,  # no partial specification is fabricated
        status=ExecutionStatus.NOT_FOUND,
        error=ToolExecutionError(
            error_type="ToolNotFound",
            error_message="Unknown tool: 'nope'.",
        ),
        ended_at=1000,
        context_tokens=5,  # the structured error message // 4
    )
    final = birth.model_copy(update={"updated_at": 1000})
    assert events == [
        FinishReason(finish_reason="tool_use"),
        ToolCallReceived(tool_call_id="tc1", execution=birth),
        ToolExecuted(
            tool_call_id="tc1", execution=final,
            result_text="Unknown tool: 'nope'.", is_error=True,
        ),
        TextBlock(text="ok"),
        FinishReason(finish_reason="stop"),
    ]
    assert runner.session.entries["te1"] == final
    assert registry.seen == []  # a terminal birth is never decided
    assert runner.idle()


async def test_raising_tool_records_failed_with_structured_error():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("boom", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("ok")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_boom",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="go")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([RaisingTool()]), provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"], now=1000,
    )

    async with runner.run() as run:
        events = [event async for event in run]

    assert events[2] == ToolExecutionStarted(
        tool_call_id="tc1",
        execution=ToolExecution(
            id="te1", parent_id="a1", created_at=1000,
            tool_call_id="tc1",
            raw_tool_call=ToolCall(id="tc1", name="boom", arguments={"a": 1, "b": 2}),
            tool_spec=ToolSpec(name="boom", description="Always raises."),
            status=ExecutionStatus.RUNNING,
            approval_status=ApprovalStatus.ALLOWED,
            approval_decisions=[ALLOW_1000],
            started_at=1000,
            updated_at=1000,
        ),
    )
    assert events[3] == ToolExecuted(
        tool_call_id="tc1",
        execution=ToolExecution(
            id="te1", parent_id="a1", created_at=1000,
            tool_call_id="tc1",
            raw_tool_call=ToolCall(id="tc1", name="boom", arguments={"a": 1, "b": 2}),
            tool_spec=ToolSpec(name="boom", description="Always raises."),
            status=ExecutionStatus.FAILED,
            error=ToolExecutionError(
                error_type="ValueError",
                error_message="kaboom",
                details={"phase": "execution"},
            ),
            approval_status=ApprovalStatus.ALLOWED,
            approval_decisions=[ALLOW_1000],
            started_at=1000, ended_at=1000,
            updated_at=1000,
            context_tokens=1,  # len("kaboom") // 4
        ),
        result_text="Tool execution failed: ValueError: kaboom",
        is_error=True,
    )
    assert runner.session.entries["te1"].status == ExecutionStatus.FAILED
    assert runner.session.entries["te1"].result is None
    assert runner.idle()


async def test_invalid_arguments_record_invalid_and_skip_strategy():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("add", {"a": 1}, id="tc1")], finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("ok")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_badargs",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="go")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    registry = FakeToolRegistry([AddTool()], decisions=[])  # empty script:
    # any decide() would raise
    runner = DeterministicRunner(
        session, tool_registry=registry, provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"], now=1000,
    )

    async with runner.run() as run:
        events = [event async for event in run]

    assert events[0] == FinishReason(finish_reason="tool_use")
    assert events[1].type == "tool_call_received"
    assert events[1].execution.status == ExecutionStatus.INVALID
    assert events[2].type == "tool_executed"
    assert events[2].is_error is True
    assert events[2].result_text.startswith("Arguments for tool 'add' are invalid.")
    assert events[3:] == [TextBlock(text="ok"), FinishReason(finish_reason="stop")]
    execution = runner.session.entries["te1"]
    assert execution.status == ExecutionStatus.INVALID
    assert execution.result is None
    assert execution.error.error_type == "InvalidToolArguments"
    assert execution.error.error_message == "Arguments for tool 'add' are invalid."
    assert execution.error.details["errors"][0]["type"] == "missing"
    assert execution.error.details["errors"][0]["loc"] == ["b"]
    assert execution.approval_status is None
    assert execution.approval_decisions == []
    assert registry.seen == []
    assert runner.idle()


async def test_rich_is_error_result_is_still_completed():
    # is_error is the TOOL's verdict about its result — the framework received
    # a result, so the execution is COMPLETED, not FAILED.
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("report", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("ok")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_rich",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="go")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([RichErrorTool()]), provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"], now=1000,
    )

    async with runner.run() as run:
        events = [event async for event in run]

    assert events[3].type == "tool_executed"
    assert events[3].result_text == "disk full"
    assert events[3].is_error is True
    assert runner.session.entries["te1"].status == ExecutionStatus.COMPLETED
    assert runner.session.entries["te1"].error is None
    assert runner.session.entries["te1"].result == ExecutionResult(
        content=[TextContent(text="disk full")],
        metadata={"code": 28}, is_error=True,
    )
    assert runner.idle()


async def test_stop_finish_with_tool_calls_still_executes_the_round():
    # the round keys off the tool_calls themselves: a provider misclassifying
    # the finish as "stop" must not leave dangling tool_use blocks behind
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="stop",
        ),
        faux_assistant_message([faux_text("It's 3.")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_stop_with_calls",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Add")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([AddTool()]), provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"], now=1000,
    )

    async with runner.run() as run:
        events = [event async for event in run]

    assert events[0] == FinishReason(finish_reason="stop")  # recorded verbatim
    assert [event.type for event in events] == [
        "finish_reason", "tool_call_received", "tool_execution_started",
        "tool_executed", "text_block", "finish_reason",
    ]
    assert len(faux.requests) == 2
    assert runner.session.entries["te1"].status == ExecutionStatus.COMPLETED
    assert runner.session.entries["tf"].outcome == TurnOutcome.COMPLETED
    assert runner.idle()


async def test_tool_use_finish_with_no_calls_closes_the_turn():
    # the inverse misclassification must not loop the model forever
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message([faux_text("Hmm.")], finish_reason="tool_use"),
    ])
    session = AgentSession(
        id="s_tool_use_no_calls",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([AddTool()]), provider=faux,
        ids=["ts", "a1", "tf"], now=1000,
    )

    result = await runner.run()

    assert result.outcome == TurnOutcome.COMPLETED
    assert len(faux.requests) == 1  # exactly one call — no spin
    assert runner.session.entries["a1"].stop_reason == "tool_use"  # verbatim
    assert runner.session.entries["tf"].outcome == TurnOutcome.COMPLETED
    assert runner.idle()


# ── post_message / state guards / context ───────────────────────────────────────


async def test_post_message_sets_pending():
    session = AgentSession(
        id="s_pm",
        active_conversation=Conversation(id="c1", nodes=[], created_at=900, updated_at=900),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(session, ids=["u1"], now=1000)

    assert runner.idle()
    msg_id = runner.post_message("Hello")

    assert msg_id == "u1"
    assert runner.pending()
    assert runner.session.entries["u1"] == UserMessage(
        id="u1", parent_id=None, created_at=1000, parts=[TextContent(text="Hello")],
        context_tokens=1,  # len("Hello") // 4
    )
    assert runner.session.active_conversation == Conversation(
        id="c1", nodes=["u1"], created_at=900, updated_at=1000,
        status=ConversationStatus.PENDING,
    )


async def test_post_message_queues_behind_a_pending_message():
    # consecutive user messages are an established shape — a closed bracket
    # with PENDING status accepts more input (the open-turn rejections live
    # in test_runner_failures.py)
    session = AgentSession(
        id="s_pm2",
        active_conversation=Conversation(id="c1", nodes=[], created_at=900, updated_at=900),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(session, ids=["u1", "u2"], now=1000)
    runner.post_message("First")  # now PENDING

    runner.post_message("Second")

    assert runner.pending()
    assert runner.session.active_conversation.nodes == ["u1", "u2"]
    assert runner.session.entries["u2"] == UserMessage(
        id="u2", parent_id="u1", created_at=1000,
        parts=[TextContent(text="Second")],
        context_tokens=1,  # len("Second") // 4
    )


async def test_post_message_accepts_a_part_list_and_keeps_its_order():
    session = AgentSession(
        id="s_pm_parts",
        active_conversation=Conversation(id="c1", nodes=[], created_at=900, updated_at=900),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(session, ids=["u1"], now=1000)
    image = ImageContent(
        source=ImageBase64(data="aGk=", media_type="image/png"), name="a.png",
    )

    runner.post_message([image, TextContent(text="Hello")])

    assert runner.session.entries["u1"] == UserMessage(
        id="u1", parent_id=None, created_at=1000,
        parts=[image, TextContent(text="Hello")],
        context_tokens=1_001,  # IMAGE_TOKENS + len("Hello") // 4
    )


async def test_post_message_rejects_an_empty_part_list():
    session = AgentSession(
        id="s_pm_empty",
        active_conversation=Conversation(id="c1", nodes=[], created_at=900, updated_at=900),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(session, ids=["u1"], now=1000)

    with pytest.raises(AgentError, match="non-empty list"):
        runner.post_message([])


async def test_post_message_rejects_a_part_it_does_not_own():
    session = AgentSession(
        id="s_pm_bad",
        active_conversation=Conversation(id="c1", nodes=[], created_at=900, updated_at=900),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(session, ids=["u1"], now=1000)

    with pytest.raises(AgentError, match="ThinkingContent"):
        runner.post_message([ThinkingContent(thinking="not a user part")])


async def test_run_when_idle_raises():
    session = AgentSession(
        id="s_idle",
        active_conversation=Conversation(id="c1", nodes=[], created_at=900, updated_at=900),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(session, now=1000)

    with pytest.raises(AgentError):
        async with runner.run() as run:
            _ = [event async for event in run]


async def test_tool_receives_tool_context():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("capture", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("done")], finish_reason="stop"),
    ])
    tool = CapturingTool()
    session = AgentSession(
        id="s_ctx",
        entries={
            "u1": UserMessage(id="u1", created_at=0, parts=[TextContent(text="go")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=0, updated_at=0),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([tool]), provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"], now=1000,
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    assert len(tool.seen) == 1
    context = tool.seen[0]
    assert context.session_id == "s_ctx"
    assert context.model == MODEL
    assert len(tool.tokens) == 1
    assert isinstance(tool.tokens[0], CancellationToken)
    assert tool.tokens[0].cancelled is False


# ── event snapshots are immutable ───────────────────────────────────────────────


async def test_event_snapshots_do_not_track_later_ledger_updates():
    # ToolCallReceived shows the birth state and ToolExecutionStarted the
    # RUNNING state even after the ledger entry reached COMPLETED — events
    # carry deep snapshots, never live references.
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("It's 3.")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_snapshots",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Add")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([AddTool()]), provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"], now=1000,
    )

    async with runner.run() as run:
        events = [event async for event in run]

    received = events[1]
    started = events[2]
    assert received.execution.status == ExecutionStatus.PENDING
    assert received.execution.approval_status is None
    assert started.execution.status == ExecutionStatus.RUNNING
    assert started.execution.result is None
    assert runner.session.entries["te1"].status == ExecutionStatus.COMPLETED


# ── serialization invariants ────────────────────────────────────────────────────


async def test_completed_session_round_trips_and_rederives_status():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("It's 3.")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_rt",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Add")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([AddTool()]), provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"], now=1000,
    )
    async with runner.run() as run:
        _ = [event async for event in run]

    reloaded = AgentSession.model_validate_json(runner.session.model_dump_json())

    # the persisted payload round-trips losslessly — approval state, the
    # structured lifecycle timestamps, and all; nothing on the session is
    # transient (the tool registry and projector live on the runner)
    assert reloaded == runner.session
    # a fresh production runner over the reloaded data re-derives the status
    rebuilt = AgentSessionRunner(
        reloaded, tool_registry=FakeToolRegistry([AddTool()]), provider=faux,
    )
    assert rebuilt.status == ConversationStatus.IDLE
