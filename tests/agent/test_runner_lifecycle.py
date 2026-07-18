"""AgentRun lifecycle scenarios: the lazy/eager handles, their three
consumption forms, and the suspend semantics.

Same declarative shape as `test_runner.py` — precondition → one action →
postcondition — but here the subject is the HANDLE: lazy laziness, the
iteration-requires-CM rule, await idempotency, suspend finalization (break →
derived status → cold resume), eager buffering/joining, background-exception
surfacing, `on_event` delivery, and the `RunResult` shapes at both stopping
points. What the engine writes into the session is covered by
`test_runner.py` / `test_runner_approvals.py`.
"""

import pytest

from luca.agent.core.models import (
    AgentSession,
    ApprovalDecision,
    ApprovalOption,
    ApprovalStatus,
    Conversation,
    ConversationStatus,
    ExecutionResult,
    ExecutionStatus,
    SessionConfig,
    TextContent,
    ToolCall,
    ToolExecution,
    ToolSpec,
    TurnOutcome,
    Usage,
    UserMessage,
)
from luca.agent.core.events import (
    FinishReason,
    TextBlock,
    ToolCallReceived,
    ToolExecuted,
    ToolExecutionStarted,
)
from luca.agent.core.exceptions import AgentError
from luca.agent.core.runner import RunResult
from luca.client.exceptions import ProviderAPIError
from luca.client.testing import (
    FauxProvider,
    faux_assistant_message,
    faux_error,
    faux_text,
    faux_tool_call,
)

from tests.agent.scenarios import (
    MODEL,
    AddTool,
    DeterministicRunner,
    FakeToolRegistry,
)

PENDING_1000 = ApprovalDecision(decision=ApprovalOption.PENDING, created_at=1000)
ALLOW_1000 = ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000)

ADD_SPEC = ToolSpec(name="add", description="Add two numbers.")

# the three lifecycle snapshots of the standard one-tool-round turn
ADD_BIRTH = ToolExecution(
    id="te1", parent_id="a1", created_at=1000,
    tool_call_id="tc1",
    raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
    tool_spec=ADD_SPEC,
    status=ExecutionStatus.PENDING,
)
ADD_RUNNING = ADD_BIRTH.model_copy(update={
    "status": ExecutionStatus.RUNNING,
    "approval_status": ApprovalStatus.ALLOWED,
    "approval_decisions": [ALLOW_1000],
    "started_at": 1000,
    "updated_at": 1000,
})
ADD_FINAL = ADD_RUNNING.model_copy(update={
    "status": ExecutionStatus.COMPLETED,
    "result": ExecutionResult(content=[TextContent(text="3")], is_error=False),
    "ended_at": 1000,
})

# the event list of the standard one-tool-round turn used throughout
TOOL_TURN_EVENTS = [
    FinishReason(finish_reason="tool_use"),
    ToolCallReceived(tool_call_id="tc1", execution=ADD_BIRTH),
    ToolExecutionStarted(tool_call_id="tc1", execution=ADD_RUNNING),
    ToolExecuted(
        tool_call_id="tc1", execution=ADD_FINAL, result_text="3", is_error=False,
    ),
    TextBlock(text="It's 3."),
    FinishReason(finish_reason="stop"),
]


# ── lazy: creation is inert ──────────────────────────────────────────────────


async def test_lazy_handle_creation_is_a_noop():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message([faux_text("Hello!")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_inert",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, provider=faux, ids=["ts", "a1", "tf"], now=1000,
    )

    discarded = runner.run()  # never driven: no work, no validation, no guard

    assert runner.pending()
    assert runner.session.active_conversation.nodes == ["u1"]
    assert faux.requests == []
    # a second handle drives normally — the discarded one held nothing
    result = await runner.run()
    assert result.outcome == TurnOutcome.COMPLETED
    assert discarded.result is None


async def test_run_on_idle_session_raises_at_first_drive():
    session = AgentSession(
        id="s_idle",
        active_conversation=Conversation(id="c1", nodes=[], created_at=900, updated_at=900),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(session, now=1000)

    run = runner.run()  # creation never validates (lazy laziness)

    with pytest.raises(AgentError):
        await run


async def test_start_on_idle_session_raises_at_call_time():
    session = AgentSession(
        id="s_idle_eager",
        active_conversation=Conversation(id="c1", nodes=[], created_at=900, updated_at=900),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(session, now=1000)

    with pytest.raises(AgentError):
        runner.start()


def test_start_outside_a_running_loop_raises_and_leaves_the_runner_usable():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message([faux_text("Hello!")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_no_loop",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, provider=faux, ids=["ts", "a1", "tf"], now=1000,
    )

    with pytest.raises(RuntimeError):
        runner.start()  # no running loop — fails BEFORE taking the run guard

    # nothing was taken or written: a retry fails the same way (RuntimeError
    # again — not the one-engine AgentError a leaked guard would produce)
    with pytest.raises(RuntimeError):
        runner.start()
    assert runner.pending()
    assert runner.session.active_conversation.nodes == ["u1"]


async def test_second_drive_while_a_run_is_live_raises():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("It's 3.")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_guard",
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
        async for _ in run:
            break  # the engine is live (suspended mid-turn, not finalized)
        with pytest.raises(AgentError):
            await runner.run()  # second first-drive while the guard is held

    # the suspend finalized the first run — a fresh handle resumes freely
    result = await runner.run()
    assert result.outcome == TurnOutcome.COMPLETED


# ── consumption rules ────────────────────────────────────────────────────────


async def test_iteration_outside_the_context_manager_raises():
    session = AgentSession(
        id="s_cm",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(session, provider=FauxProvider(), now=1000)
    run = runner.run()

    with pytest.raises(AgentError):
        async for _ in run:
            pass


async def test_eager_iteration_outside_the_context_manager_raises():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message([faux_text("Hello!")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_cm_eager",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, provider=faux, ids=["ts", "a1", "tf"], now=1000,
    )
    run = runner.start()

    with pytest.raises(AgentError):
        async for _ in run:
            pass

    await run  # join so the background task never outlives the test


async def test_await_twice_returns_the_cached_result():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message([faux_text("Hello!")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_idem",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, provider=faux, ids=["ts", "a1", "tf"], now=1000,
    )
    run = runner.run()

    first = await run
    second = await run

    assert second == first
    assert run.result == first
    assert len(faux.requests) == 1  # the second await drove nothing


async def test_await_inside_the_block_after_partial_iteration_drives_to_the_stop():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("It's 3.")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_mix",
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
        async for _ in run:
            break  # consumed one event
        result = await run  # drives the rest, discarding events

    assert result.outcome == TurnOutcome.COMPLETED
    assert runner.idle()


# ── RunResult at the two stopping points ─────────────────────────────────────


async def test_await_returns_completed_result_at_idle_stop():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message([faux_text("Hello!")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_rr_idle",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, provider=faux, ids=["ts", "a1", "tf"], now=1000,
    )

    result = await runner.run()

    assert result == RunResult(
        status=ConversationStatus.IDLE,
        outcome=TurnOutcome.COMPLETED,
        pending_approvals=[],
    )
    assert runner.idle()


async def test_await_returns_pause_result_at_approval_stop():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
        ),
    ])
    session = AgentSession(
        id="s_rr_gate",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Add")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()], decisions=[PENDING_1000]),
        provider=faux, ids=["ts", "a1", "te1"], now=1000,
    )

    result = await runner.run()

    assert result == RunResult(
        status=ConversationStatus.AWAITING_APPROVAL,
        outcome=None,
        pending_approvals=[
            ToolExecution(
                id="te1", parent_id="a1", created_at=1000,
                tool_call_id="tc1",
                raw_tool_call=ToolCall(
                    id="tc1", name="add", arguments={"a": 1, "b": 2},
                ),
                tool_spec=ADD_SPEC,
                status=ExecutionStatus.PENDING,
                result=None,
                approval_status=ApprovalStatus.PENDING,
                approval_decisions=[PENDING_1000],
                updated_at=1000,
            ),
        ],
    )
    assert runner.awaiting_approval()


# ── suspend: break / exit the lazy block ─────────────────────────────────────


async def test_break_suspends_with_derived_status_and_a_fresh_run_resumes():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("It's 3.")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_suspend",
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
        async for _ in run:
            break  # suspend after the first event (FinishReason)

    # nothing new recorded by the exit; the status re-derived (no stale
    # RUNNING) and record+create were atomic: the execution exists undecided
    assert runner.pending()
    assert runner.session.active_conversation.nodes == ["u1", "ts", "a1", "te1"]
    assert runner.session.entries["te1"].status == ExecutionStatus.PENDING
    assert runner.session.entries["te1"].approval_status is None
    assert runner.session.entries["te1"].approval_decisions == []

    async with runner.run() as resumed:
        events = [event async for event in resumed]

    assert [event.type for event in events] == [
        "tool_execution_started", "tool_executed", "text_block", "finish_reason",
    ]
    assert events[1].result_text == "3"
    assert runner.idle()


async def test_suspended_session_cold_resumes_after_reload():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("It's 3.")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_cold",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Add")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([AddTool()]), provider=faux,
        ids=["ts", "a1", "te1"], now=1000,
    )
    async with runner.run() as run:
        async for _ in run:
            break
    payload = runner.session.model_dump_json()  # "process exits" here

    reloaded = AgentSession.model_validate_json(payload)
    resumed = DeterministicRunner(
        reloaded, tool_registry=FakeToolRegistry([AddTool()]), provider=faux,
        ids=["a2", "tf"], now=1000,
    )

    assert resumed.pending()
    async with resumed.run() as run:
        events = [event async for event in run]

    assert [event.type for event in events] == [
        "tool_execution_started", "tool_executed", "text_block", "finish_reason",
    ]
    assert events[1].result_text == "3"
    assert resumed.idle()


async def test_finalized_suspended_handle_rejects_await_and_reentry():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("It's 3.")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_final",
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
        async for _ in run:
            break

    with pytest.raises(AgentError):
        await run
    with pytest.raises(AgentError):
        async with run:
            pass


async def test_completed_lazy_run_still_answers_await_after_exit():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message([faux_text("Hello!")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_done",
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

    assert events == [TextBlock(text="Hello!"), FinishReason(finish_reason="stop")]
    assert (await run).outcome == TurnOutcome.COMPLETED  # cached, not re-driven
    assert len(faux.requests) == 1


# ── eager mechanics ──────────────────────────────────────────────────────────


async def test_eager_run_completes_without_observation():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("It's 3.")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_eager",
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

    result = await runner.start()  # join only — no iteration anywhere

    assert result.outcome == TurnOutcome.COMPLETED
    assert runner.idle()
    assert runner.session.active_conversation.nodes == [
        "u1", "ts", "a1", "te1", "a2", "tf",
    ]


async def test_eager_empty_block_runs_to_completion_and_exit_joins():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message([faux_text("Hello!")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_eager_cm",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, provider=faux, ids=["ts", "a1", "tf"], now=1000,
    )

    async with runner.start():
        pass  # never iterated; __aexit__ joins

    assert runner.idle()


async def test_eager_late_consumer_sees_the_full_history_from_event_zero():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("It's 3.")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_late",
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
    run = runner.start()
    await run  # the run is long finished before anyone iterates

    async with run:
        events = [event async for event in run]

    assert events == TOOL_TURN_EVENTS


async def test_eager_break_does_not_stop_the_agent_and_the_cursor_continues():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("It's 3.")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_break",
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

    async with runner.start() as run:
        consumed = []
        async for event in run:
            consumed.append(event)
            break  # stop observing — the agent must proceed regardless
        result = await run  # join inside the block
        consumed.extend([event async for event in run])  # cursor continues

    assert result.outcome == TurnOutcome.COMPLETED
    assert runner.idle()
    assert consumed == TOOL_TURN_EVENTS


async def test_eager_background_exception_surfaces_on_join_and_iteration():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [], finish_reason="stop",
            error=faux_error("provider down", error_class=ProviderAPIError),
        ),
    ])
    session = AgentSession(
        id="s_bg_exc",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, provider=faux, ids=["ts", "tf"], now=1000,
    )
    run = runner.start()

    with pytest.raises(ProviderAPIError, match="provider down"):
        await run

    assert run.result is None
    with pytest.raises(ProviderAPIError, match="provider down"):
        async with run:
            _ = [event async for event in run]  # buffer drains, then raises


# ── on_event delivery ────────────────────────────────────────────────────────


async def test_on_event_sync_callback_sees_every_event_of_an_awaited_run():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("It's 3.")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_hook",
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
    seen = []

    await runner.run(on_event=seen.append)  # await-only: events still delivered

    assert seen == TOOL_TURN_EVENTS


async def test_on_event_async_callback_is_awaited_inline():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message([faux_text("Hello!")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_hook_async",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, provider=faux, ids=["ts", "a1", "tf"], now=1000,
    )
    seen = []

    async def on_event(event):
        seen.append(event)

    result = await runner.start(on_event=on_event)

    assert result.outcome == TurnOutcome.COMPLETED
    assert seen == [TextBlock(text="Hello!"), FinishReason(finish_reason="stop")]


async def test_on_event_exception_after_the_final_answer_leaves_the_turn_complete():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message([faux_text("Hello!")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_hook_boom",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, provider=faux, ids=["ts", "a1", "tf"], now=1000,
    )

    def boom(event):
        raise RuntimeError("app hook failed")

    with pytest.raises(RuntimeError, match="app hook failed"):
        await runner.run(on_event=boom)

    # every session write for the round — the answer AND the close — landed
    # before the first event was delivered: the crash cost only the event
    # delivery, never a duplicate LLM call
    assert runner.idle()
    assert runner.session.active_conversation.nodes == ["u1", "ts", "a1", "tf"]
    assert runner.session.entries["tf"].outcome == TurnOutcome.COMPLETED
    assert len(faux.requests) == 1


async def test_on_event_exception_mid_tool_round_crashes_resumably():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("It's 3.")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_hook_boom_mid",
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

    def boom(event):
        raise RuntimeError("app hook failed")

    with pytest.raises(RuntimeError, match="app hook failed"):
        await runner.run(on_event=boom)

    # crash on the first event of an UNFINISHED round: open bracket, no
    # TurnFinish, derived status — resumable
    assert runner.pending()
    assert runner.session.active_conversation.nodes == ["u1", "ts", "a1", "te1"]

    async with runner.run() as resumed:
        events = [event async for event in resumed]
    assert [event.type for event in events] == [
        "tool_execution_started", "tool_executed", "text_block", "finish_reason",
    ]
    assert events[1].result_text == "3"
    assert runner.idle()
