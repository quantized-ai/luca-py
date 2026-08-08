"""RuntimeConfig step limits and doom-loop detection.

Each test is declarative: KNOWN session → one action (drain run()) → assert
the invariant. The FauxProvider scripts exactly the LLM responses the turn
needs and `FakeToolRegistry` allows every call, so the `RuntimeConfig` is the
only thing that varies between these turns.

Invariants tested:
- hard_max_steps: the turn closes ERRORED once step_count reaches the limit,
  before the next LLM call; 0 / Inf disable it
- soft_max_steps + limit_tool_choice_on_soft_max_steps_reached: tool_choice
  "none" from the request that reaches the limit onward (the tools stay
  advertised — only the CHOICE is restricted), or never when the flag is off
- doom_loop_threshold: `ToolExecution.is_doom_loop_flagged` on the Nth
  consecutive identical call; name AND arguments must match; 0 / Inf disable
  detection
- limit_tool_choice_on_doom_loop_flagged: tool_choice "none" once a flagged
  execution exists in the open turn, or never when the flag is off
- the soft == hard misconfiguration warning

`ConversationRuntimeStatus` derivation (what step_count counts) belongs to the
ledger and is covered in `test_ledger.py`; here it is only an input.
"""

import warnings

import pytest

from luca.agent.core.models import (
    AgentSession,
    ApprovalDecision,
    ApprovalOption,
    ApprovalStatus,
    ConversationStatus,
    ExecutionAttempt,
    ExecutionAttemptOutcome,
    ExecutionResult,
    ExecutionStatus,
    RuntimeConfig,
    SessionConfig,
    TextContent,
    ToolCall,
    ToolExecution,
    TurnFinish,
    TurnOutcome,
    UserMessage,
)
from luca.agent.core.runner import RunResult
from luca.client.testing import (
    FauxProvider,
    faux_assistant_message,
    faux_text,
    faux_tool_call,
)
from luca.client.types import Tool as LucaTool
from tests.agent.scenarios import (
    ADD_SPEC,
    MODEL,
    MULTIPLY_SPEC,
    AddTool,
    DeterministicRunner,
    FakeToolRegistry,
    MultiplyTool,
    conversation,
    make_session,
)

ALLOW_1000 = ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000)
# the spec each doubled tool produces, keyed by the name the model calls
SPECS = {"add": ADD_SPEC, "multiply": MULTIPLY_SPEC}


def session_with(runtime_config: RuntimeConfig) -> AgentSession:
    """One queued question and nothing else — the shared precondition, so the
    `RuntimeConfig` is the only variable in every scenario below."""
    return make_session(
        id="s1",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")]),
        },
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL, runtime_config=runtime_config),
    )


def completed(
    entry_id: str,
    parent_id: str,
    call: ToolCall,
    output: str,
    *,
    flagged: bool = False,
) -> ToolExecution:
    """The durable record of one allowed-and-completed tool round under the
    frozen clock. `flagged` is the doom-loop detector's verdict — the one
    field these turns differ in."""
    return ToolExecution(
        id=entry_id,
        parent_id=parent_id,
        created_at=1000,
        tool_call_id=call.id,
        conversation_id="c1",
        raw_tool_call=call,
        tool_spec=SPECS[call.name],
        tool_spec_id=SPECS[call.name].spec_id(),
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(content=[TextContent(text=output)]),
        approval_status=ApprovalStatus.ALLOWED,
        approval_decisions=[ALLOW_1000],
        attempts=[ExecutionAttempt(outcome=ExecutionAttemptOutcome.COMPLETED, started_at=1000, ended_at=1000)],
        finished_at=1000,
        updated_at=1000,
        is_doom_loop_flagged=flagged,
    )


# ── hard_max_steps ────────────────────────────────────────────────────────────


async def test_hard_max_steps_closes_the_turn_with_errored():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            # a second response is scripted nowhere: reaching the model again
            # would exhaust the transport and fail the run.
        ]
    )
    session = session_with(RuntimeConfig(hard_max_steps=1))
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()]),
        provider=faux,
        ids=["ts", "a1", "te1", "tf"],
        now=1000,
    )

    result = await runner.run()

    # the turn is closed but the conversation stays retry-ready PENDING
    assert result == RunResult(
        status=ConversationStatus.IDLE,
        outcome=TurnOutcome.ERRORED,
        pending_approvals=[],
    )
    assert runner.session.entries["tf"] == TurnFinish(
        id="tf",
        parent_id="te1",
        created_at=1000,
        outcome=TurnOutcome.ERRORED,
        error="Hard max steps limit reached: 1",
    )
    assert len(faux.requests) == 1


async def test_hard_max_steps_allows_exactly_n_steps_before_closing():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message(
                [faux_tool_call("add", {"a": 3, "b": 4}, id="tc2")],
                finish_reason="tool_use",
            ),
        ]
    )
    session = session_with(RuntimeConfig(hard_max_steps=2))
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()]),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "te2", "tf"],
        now=1000,
    )

    result = await runner.run()

    assert result == RunResult(
        status=ConversationStatus.IDLE,
        outcome=TurnOutcome.ERRORED,
        pending_approvals=[],
    )
    assert runner.session.entries["tf"] == TurnFinish(
        id="tf",
        parent_id="te2",
        created_at=1000,
        outcome=TurnOutcome.ERRORED,
        error="Hard max steps limit reached: 2",
    )
    assert len(faux.requests) == 2


async def test_hard_max_steps_of_zero_disables_the_limit():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("Done.")], finish_reason="stop"),
        ]
    )
    session = session_with(RuntimeConfig(hard_max_steps=0))
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()]),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
    )

    result = await runner.run()

    assert result == RunResult(
        status=ConversationStatus.IDLE,
        outcome=TurnOutcome.COMPLETED,
        pending_approvals=[],
    )
    assert runner.session.entries["tf"] == TurnFinish(
        id="tf",
        parent_id="a2",
        created_at=1000,
        outcome=TurnOutcome.COMPLETED,
    )
    assert len(faux.requests) == 2


# ── soft_max_steps ────────────────────────────────────────────────────────────


async def test_soft_max_steps_with_limit_sets_tool_choice_none():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("Done.")], finish_reason="stop"),
        ]
    )
    session = session_with(
        RuntimeConfig(
            soft_max_steps=1,
            limit_tool_choice_on_soft_max_steps_reached=True,
        )
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()]),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
    )

    await runner.run()

    # the wire request carries messages/model/system this test does not own,
    # so only the two fields the limit governs are asserted.
    # first request: step_count=0, limit not reached → no restriction
    assert faux.requests[0].tool_choice is None
    # second request: step_count=1 >= soft_max_steps=1 → tool_choice="none"
    assert faux.requests[1].tool_choice == "none"
    # the tools stay advertised — the limit restricts the CHOICE, not the list
    assert faux.requests[1].tools == [
        LucaTool(
            name="add",
            description="Add two numbers.",
            parameters=ADD_SPEC.input_schema,
        ),
    ]


async def test_soft_max_steps_without_limit_does_not_restrict_tool_choice():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("Done.")], finish_reason="stop"),
        ]
    )
    session = session_with(
        RuntimeConfig(
            soft_max_steps=1,
            limit_tool_choice_on_soft_max_steps_reached=False,
        )
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()]),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
    )

    await runner.run()

    assert faux.requests[0].tool_choice is None
    assert faux.requests[1].tool_choice is None


# ── doom loop: flagging ───────────────────────────────────────────────────────


async def test_doom_loop_not_flagged_before_threshold():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc2")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("Done.")], finish_reason="stop"),
        ]
    )
    session = session_with(
        RuntimeConfig(
            doom_loop_threshold=3,
            limit_tool_choice_on_doom_loop_flagged=False,  # isolate flagging only
        )
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()]),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "te2", "a3", "tf"],
        now=1000,
    )

    await runner.run()

    assert runner.session.entries["te1"] == completed(
        "te1",
        "a1",
        ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        "3",
    )
    assert runner.session.entries["te2"] == completed(
        "te2",
        "a2",
        ToolCall(id="tc2", name="add", arguments={"a": 1, "b": 2}),
        "3",
    )


async def test_doom_loop_flagged_at_threshold():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc2")],
                finish_reason="tool_use",
            ),
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc3")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("Done.")], finish_reason="stop"),
        ]
    )
    session = session_with(
        RuntimeConfig(
            doom_loop_threshold=3,
            limit_tool_choice_on_doom_loop_flagged=False,  # isolate flagging only
        )
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()]),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "te2", "a3", "te3", "a4", "tf"],
        now=1000,
    )

    await runner.run()

    assert runner.session.entries["te1"] == completed(
        "te1",
        "a1",
        ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        "3",
    )
    assert runner.session.entries["te2"] == completed(
        "te2",
        "a2",
        ToolCall(id="tc2", name="add", arguments={"a": 1, "b": 2}),
        "3",
    )
    # the third consecutive identical call: flagged, and otherwise ordinary —
    # the flag is a marker, not an outcome
    assert runner.session.entries["te3"] == completed(
        "te3",
        "a3",
        ToolCall(id="tc3", name="add", arguments={"a": 1, "b": 2}),
        "3",
        flagged=True,
    )


async def test_doom_loop_not_flagged_when_arguments_differ():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc2")],
                finish_reason="tool_use",
            ),
            faux_assistant_message(
                [faux_tool_call("add", {"a": 9, "b": 9}, id="tc3")],  # different args
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("Done.")], finish_reason="stop"),
        ]
    )
    session = session_with(
        RuntimeConfig(
            doom_loop_threshold=3,
            limit_tool_choice_on_doom_loop_flagged=False,
        )
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()]),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "te2", "a3", "te3", "a4", "tf"],
        now=1000,
    )

    await runner.run()

    assert runner.session.entries["te3"] == completed(
        "te3",
        "a3",
        ToolCall(id="tc3", name="add", arguments={"a": 9, "b": 9}),
        "18",
    )


async def test_doom_loop_not_flagged_when_the_tool_differs():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc2")],
                finish_reason="tool_use",
            ),
            faux_assistant_message(
                # same arguments, another tool — the streak is broken
                [faux_tool_call("multiply", {"a": 1, "b": 2}, id="tc3")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("Done.")], finish_reason="stop"),
        ]
    )
    session = session_with(
        RuntimeConfig(
            doom_loop_threshold=3,
            limit_tool_choice_on_doom_loop_flagged=False,
        )
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool(), MultiplyTool()]),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "te2", "a3", "te3", "a4", "tf"],
        now=1000,
    )

    await runner.run()

    assert runner.session.entries["te3"] == completed(
        "te3",
        "a3",
        ToolCall(id="tc3", name="multiply", arguments={"a": 1, "b": 2}),
        "2",
    )


async def test_doom_loop_disabled_when_threshold_is_inf():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc2")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("Done.")], finish_reason="stop"),
        ]
    )
    session = session_with(RuntimeConfig(doom_loop_threshold=-1))  # Inf = disabled
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()]),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "te2", "a3", "tf"],
        now=1000,
    )

    await runner.run()

    assert runner.session.entries["te1"] == completed(
        "te1",
        "a1",
        ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        "3",
    )
    assert runner.session.entries["te2"] == completed(
        "te2",
        "a2",
        ToolCall(id="tc2", name="add", arguments={"a": 1, "b": 2}),
        "3",
    )


async def test_doom_loop_disabled_when_threshold_is_zero():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc2")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("Done.")], finish_reason="stop"),
        ]
    )
    session = session_with(RuntimeConfig(doom_loop_threshold=0))  # 0 = disabled
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()]),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "te2", "a3", "tf"],
        now=1000,
    )

    await runner.run()

    assert runner.session.entries["te1"] == completed(
        "te1",
        "a1",
        ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        "3",
    )
    assert runner.session.entries["te2"] == completed(
        "te2",
        "a2",
        ToolCall(id="tc2", name="add", arguments={"a": 1, "b": 2}),
        "3",
    )


# ── doom loop: tool_choice ────────────────────────────────────────────────────


async def test_doom_loop_with_limit_sets_tool_choice_none():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc2")],
                finish_reason="tool_use",
            ),
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc3")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("Done.")], finish_reason="stop"),
        ]
    )
    session = session_with(
        RuntimeConfig(
            doom_loop_threshold=3,
            limit_tool_choice_on_doom_loop_flagged=True,
        )
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()]),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "te2", "a3", "te3", "a4", "tf"],
        now=1000,
    )

    await runner.run()

    # requests 0-2: nothing is flagged yet → no restriction
    assert faux.requests[0].tool_choice is None
    assert faux.requests[1].tool_choice is None
    assert faux.requests[2].tool_choice is None
    # request 3: te3 is flagged → tool_choice="none"
    assert faux.requests[3].tool_choice == "none"


async def test_doom_loop_without_limit_does_not_restrict_tool_choice():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc2")],
                finish_reason="tool_use",
            ),
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc3")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("Done.")], finish_reason="stop"),
        ]
    )
    session = session_with(
        RuntimeConfig(
            doom_loop_threshold=3,
            limit_tool_choice_on_doom_loop_flagged=False,
        )
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()]),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "te2", "a3", "te3", "a4", "tf"],
        now=1000,
    )

    await runner.run()

    # the flag is still raised — only the restriction is off
    assert runner.session.entries["te3"] == completed(
        "te3",
        "a3",
        ToolCall(id="tc3", name="add", arguments={"a": 1, "b": 2}),
        "3",
        flagged=True,
    )
    assert faux.requests[0].tool_choice is None
    assert faux.requests[1].tool_choice is None
    assert faux.requests[2].tool_choice is None
    assert faux.requests[3].tool_choice is None


# ── warning: soft == hard ─────────────────────────────────────────────────────


def test_warns_when_soft_and_hard_max_steps_are_equal():
    faux = FauxProvider()
    session = session_with(RuntimeConfig(soft_max_steps=3, hard_max_steps=3))

    with pytest.warns(UserWarning, match="hard_max_steps prevails"):
        DeterministicRunner(session, provider=faux, ids=[], now=1000)


def test_no_warning_when_only_one_limit_set():
    faux = FauxProvider()
    session = session_with(RuntimeConfig(soft_max_steps=3))

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        DeterministicRunner(session, provider=faux, ids=[], now=1000)

    assert recorded == []


def test_no_warning_when_both_limits_are_disabled():
    faux = FauxProvider()
    session = session_with(RuntimeConfig())  # both default to Inf — equal, but off

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        DeterministicRunner(session, provider=faux, ids=[], now=1000)

    assert recorded == []
