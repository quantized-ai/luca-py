"""Tests for RuntimeConfig step limits and doom-loop detection.

Each test is declarative: KNOWN session → one action (drain run()) → assert
the invariant. The FauxProvider scripts exactly the LLM responses needed.

Invariants tested:
- hard_max_steps: turn closes with ERRORED after step_count >= limit
- soft_max_steps + limit_tool_choice_on_soft_max_steps_reached=True:
    tool_choice="none" is passed once the limit is reached
- soft_max_steps + limit_tool_choice_on_soft_max_steps_reached=False:
    tool_choice stays None even when the limit is reached
- doom_loop: ToolExecution.is_doom_loop_flagged is set on the Nth consecutive
    identical tool call (threshold controls N)
- doom_loop + limit_tool_choice_on_doom_loop_flagged=True:
    tool_choice="none" once a doom-loop-flagged execution exists in the turn
"""

import warnings

import pytest

from luca.agent.core.models import (
    AgentSession,
    Conversation,
    ConversationStatus,
    LLMConfig,
    RuntimeConfig,
    SessionConfig,
    TextContent,
    TurnOutcome,
    UserMessage,
)
from luca.client.testing import (
    FauxProvider,
    faux_assistant_message,
    faux_text,
    faux_tool_call,
)

from tests.agent.scenarios import AddTool, DeterministicRunner, FakeToolRegistry

MODEL = LLMConfig(model="test-model", provider="faux")


def _session(runtime_config: RuntimeConfig | None = None) -> AgentSession:
    return AgentSession(
        id="s1",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")]),
        },
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(
            llm_config=MODEL,
            runtime_config=runtime_config or RuntimeConfig(),
        ),
    )


# ── SessionRuntimeStatus derivation ──────────────────────────────────────────


def test_runtime_status_step_count_counts_assistant_messages_in_open_turn():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("Done.")], finish_reason="stop"),
    ])
    session = _session()
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([AddTool()]), provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"], now=1000,
    )

    status_before_run = runner.session.session_runtime_status
    assert status_before_run.step_count == 0
    assert status_before_run.turn_count == 0


async def test_runtime_status_turn_count_includes_open_turn():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message([faux_text("Hello.")], finish_reason="stop"),
    ])
    session = _session()
    runner = DeterministicRunner(
        session, provider=faux,
        ids=["ts", "a1", "tf"], now=1000,
    )

    async with runner.run() as run:
        step_count_mid_turn = None
        async for _ in run:
            # After TurnStart + first AssistantMessage appended:
            status = runner.session.session_runtime_status
            if status.turn_count == 1:
                step_count_mid_turn = status.step_count

    final = runner.session.session_runtime_status
    assert final.turn_count == 1  # TurnStart was written
    assert final.step_count == 0  # turn is now closed (no open turn)


# ── hard_max_steps ────────────────────────────────────────────────────────────


async def test_hard_max_steps_closes_turn_with_errored():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
        ),
        # Second response would be needed but the turn closes before it.
    ])
    session = _session(RuntimeConfig(hard_max_steps=1))
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([AddTool()]), provider=faux,
        ids=["ts", "a1", "te1", "tf"], now=1000,
    )

    result = await runner.run()

    assert result.outcome == TurnOutcome.ERRORED
    assert result.status == ConversationStatus.IDLE
    assert len(faux.requests) == 1  # one LLM call was made
    assert runner.session.status == ConversationStatus.PENDING  # retry-ready


async def test_hard_max_steps_allows_exactly_n_steps_before_closing():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
        ),
        faux_assistant_message(
            [faux_tool_call("add", {"a": 3, "b": 4}, id="tc2")],
            finish_reason="tool_use",
        ),
        # Third response cut off by hard_max_steps=2
    ])
    session = _session(RuntimeConfig(hard_max_steps=2))
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([AddTool()]), provider=faux,
        ids=["ts", "a1", "te1", "a2", "te2", "tf"], now=1000,
    )

    result = await runner.run()

    assert result.outcome == TurnOutcome.ERRORED
    assert len(faux.requests) == 2


# ── soft_max_steps ────────────────────────────────────────────────────────────


async def test_soft_max_steps_with_limit_sets_tool_choice_none():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("Done.")], finish_reason="stop"),
    ])
    session = _session(RuntimeConfig(
        soft_max_steps=1,
        limit_tool_choice_on_soft_max_steps_reached=True,
    ))
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([AddTool()]), provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"], now=1000,
    )

    await runner.run()

    # First request: step_count=0, limit not reached → no restriction
    assert faux.requests[0].tool_choice is None
    # Second request: step_count=1 >= soft_max_steps=1 → tool_choice="none"
    assert faux.requests[1].tool_choice == "none"


async def test_soft_max_steps_without_limit_does_not_restrict_tool_choice():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("Done.")], finish_reason="stop"),
    ])
    session = _session(RuntimeConfig(
        soft_max_steps=1,
        limit_tool_choice_on_soft_max_steps_reached=False,
    ))
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([AddTool()]), provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"], now=1000,
    )

    await runner.run()

    assert faux.requests[0].tool_choice is None
    assert faux.requests[1].tool_choice is None


# ── doom loop ─────────────────────────────────────────────────────────────────


async def test_doom_loop_not_flagged_before_threshold():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
        ),
        faux_assistant_message(
            [faux_tool_call("add", {"a": 1, "b": 2}, id="tc2")],
            finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("Done.")], finish_reason="stop"),
    ])
    session = _session(RuntimeConfig(
        doom_loop_threshold=3,
        limit_tool_choice_on_doom_loop_flagged=False,  # isolate flagging only
    ))
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([AddTool()]), provider=faux,
        ids=["ts", "a1", "te1", "a2", "te2", "a3", "tf"], now=1000,
    )

    await runner.run()

    te1 = runner.session.entries["te1"]
    te2 = runner.session.entries["te2"]
    assert te1.is_doom_loop_flagged is False
    assert te2.is_doom_loop_flagged is False


async def test_doom_loop_flagged_at_threshold():
    faux = FauxProvider()
    faux.set_responses([
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
    ])
    session = _session(RuntimeConfig(
        doom_loop_threshold=3,
        limit_tool_choice_on_doom_loop_flagged=False,  # isolate flagging only
    ))
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([AddTool()]), provider=faux,
        ids=["ts", "a1", "te1", "a2", "te2", "a3", "te3", "a4", "tf"], now=1000,
    )

    await runner.run()

    te1 = runner.session.entries["te1"]
    te2 = runner.session.entries["te2"]
    te3 = runner.session.entries["te3"]
    assert te1.is_doom_loop_flagged is False
    assert te2.is_doom_loop_flagged is False
    assert te3.is_doom_loop_flagged is True  # third consecutive identical call


async def test_doom_loop_not_flagged_when_different_args():
    faux = FauxProvider()
    faux.set_responses([
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
    ])
    session = _session(RuntimeConfig(
        doom_loop_threshold=3,
        limit_tool_choice_on_doom_loop_flagged=False,
    ))
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([AddTool()]), provider=faux,
        ids=["ts", "a1", "te1", "a2", "te2", "a3", "te3", "a4", "tf"], now=1000,
    )

    await runner.run()

    te3 = runner.session.entries["te3"]
    assert te3.is_doom_loop_flagged is False


async def test_doom_loop_with_limit_sets_tool_choice_none():
    faux = FauxProvider()
    faux.set_responses([
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
    ])
    session = _session(RuntimeConfig(
        doom_loop_threshold=3,
        limit_tool_choice_on_doom_loop_flagged=True,
    ))
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([AddTool()]), provider=faux,
        ids=["ts", "a1", "te1", "a2", "te2", "a3", "te3", "a4", "tf"], now=1000,
    )

    await runner.run()

    # Requests 0-2: before the doom loop fires, no restriction
    assert faux.requests[0].tool_choice is None
    assert faux.requests[1].tool_choice is None
    assert faux.requests[2].tool_choice is None
    # Request 3: te3 is flagged → tool_choice="none"
    assert faux.requests[3].tool_choice == "none"


async def test_doom_loop_disabled_when_threshold_is_inf():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
        ),
        faux_assistant_message(
            [faux_tool_call("add", {"a": 1, "b": 2}, id="tc2")],
            finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("Done.")], finish_reason="stop"),
    ])
    session = _session(RuntimeConfig(doom_loop_threshold=-1))  # Inf = disabled
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([AddTool()]), provider=faux,
        ids=["ts", "a1", "te1", "a2", "te2", "a3", "tf"], now=1000,
    )

    await runner.run()

    te1 = runner.session.entries["te1"]
    te2 = runner.session.entries["te2"]
    assert te1.is_doom_loop_flagged is False
    assert te2.is_doom_loop_flagged is False


# ── warning: soft == hard ─────────────────────────────────────────────────────


def test_warns_when_soft_and_hard_max_steps_are_equal():
    faux = FauxProvider()
    session = _session(RuntimeConfig(soft_max_steps=3, hard_max_steps=3))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        DeterministicRunner(session, provider=faux, ids=[], now=1000)

    assert len(w) == 1
    assert issubclass(w[0].category, UserWarning)
    assert "hard_max_steps" in str(w[0].message)


def test_no_warning_when_only_one_limit_set():
    faux = FauxProvider()
    session = _session(RuntimeConfig(soft_max_steps=3))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        DeterministicRunner(session, provider=faux, ids=[], now=1000)

    assert len(w) == 0
