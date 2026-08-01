"""The run TREE surface: `children` / `child()`, `approvals`, `notify()`, and
per-conversation run state.

No subagent exists yet at this layer — every run here has an empty subtree —
so what these tests pin is the DEGENERATE case: with no children the tree
surface must behave exactly as a single-conversation run always did. That is
the whole point of exercising it before subagents can populate it; a
regression here would otherwise only surface once several conversations are
running and be indistinguishable from a scheduling bug.
"""

import asyncio

import pytest

from luca.agent.core.events import ApprovalRequired
from luca.agent.core.exceptions import AgentError
from luca.agent.core.models import (
    ApprovalDecision,
    ApprovalOption,
    ApprovalStatus,
    ConversationStatus,
    SessionConfig,
    ToolExecution,
    TurnOutcome,
)
from luca.client.testing import (
    FauxProvider,
    faux_assistant_message,
    faux_text,
    faux_tool_call,
)
from tests.agent.scenarios import (
    CLEARED_SESSION,
    GATED_SESSION,
    MODEL,
    AddTool,
    DeterministicRunner,
    FakeToolRegistry,
    conversation,
    make_session,
)


def answering(text: str = "done") -> FauxProvider:
    faux = FauxProvider()
    faux.set_responses([faux_assistant_message([faux_text(text)], finish_reason="stop")])
    return faux


# ── the subtree, empty ────────────────────────────────────────────────────────


async def test_a_run_names_the_conversation_it_drives():
    session = CLEARED_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(session, tool_registry=FakeToolRegistry([AddTool()]), now=1000)

    run = runner.run()

    assert run.conversation_id == session.main_conversation_id


async def test_a_run_with_no_subagents_has_an_empty_subtree():
    session = CLEARED_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(session, tool_registry=FakeToolRegistry([AddTool()]), now=1000)

    run = runner.run()

    assert run.children == {}
    assert run.child("nope") is None


async def test_children_is_a_copy():
    session = CLEARED_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(session, tool_registry=FakeToolRegistry([AddTool()]), now=1000)
    run = runner.run()

    run.children["forged"] = run

    assert run.children == {}


async def test_autostart_subagents_defaults_to_true_and_start_implies_it():
    session = CLEARED_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()]),
        provider=answering(),
        ids=["a2", "tf"],
        now=1000,
    )

    assert runner.run().autostart_subagents is True
    assert runner.run(autostart_subagents=False).autostart_subagents is False

    async with runner.start() as run:
        assert run.autostart_subagents is True
        _ = [event async for event in run]


# ── the one-engine guard is per conversation ─────────────────────────────────


async def test_two_runs_on_the_same_conversation_still_collide():
    session = CLEARED_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()]),
        provider=answering(),
        ids=["a2", "tf"],
        now=1000,
    )

    first = runner.run()
    async with first:
        await first.__anext__()
        second = runner.run()
        with pytest.raises(AgentError, match="already active on conversation"):
            async with second:
                await second.__anext__()


# ── the approvals stream ──────────────────────────────────────────────────────


async def test_a_gate_reaches_the_approvals_stream_as_it_is_raised():
    # the durable list and the stream expose the same fact; the stream's job is
    # delivering it WITHOUT the caller polling
    session = GATED_SESSION.model_copy(deep=True)
    session.entries["te1"] = session.entries["te1"].model_copy(
        update={"approval_status": None, "approval_decisions": []},
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry(
            [AddTool()],
            decisions=[ApprovalDecision(decision=ApprovalOption.PENDING, created_at=1000)],
        ),
        now=1000,
    )
    seen: list[ToolExecution] = []

    run = runner.run()

    watcher = asyncio.create_task(_collect(run, seen))
    async with run:
        _ = [event async for event in run]
    await watcher

    assert [e.id for e in seen] == ["te1"]
    assert [e.conversation_id for e in seen] == ["c1"]
    # the same fact, the other door
    assert [e.id for e in runner.pending_approvals()] == ["te1"]


async def test_the_approvals_stream_closes_with_the_run():
    # the watcher's lifetime is the run's: it ends by itself rather than
    # needing to be cancelled
    session = CLEARED_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()]),
        provider=answering(),
        ids=["a2", "tf"],
        now=1000,
    )
    run = runner.run()

    async def watch():
        return [execution async for execution in run.approvals]

    watcher = asyncio.create_task(watch())
    async with run:
        _ = [event async for event in run]

    assert await asyncio.wait_for(watcher, timeout=1) == []


async def test_the_stream_carries_the_same_objects_the_durable_list_does():
    session = GATED_SESSION.model_copy(deep=True)
    session.entries["te1"] = session.entries["te1"].model_copy(
        update={"approval_status": None, "approval_decisions": []},
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry(
            [AddTool()],
            decisions=[ApprovalDecision(decision=ApprovalOption.PENDING, created_at=1000)],
        ),
        now=1000,
    )
    run = runner.run()
    streamed: list[ToolExecution] = []

    watcher = asyncio.create_task(_collect(run, streamed))
    async with run:
        _ = [event async for event in run]
    await watcher

    [durable] = runner.pending_approvals()
    assert streamed[0].id == durable.id
    assert streamed[0].raw_tool_call == durable.raw_tool_call
    # a deep snapshot, not the live ledger entry
    assert streamed[0] is not durable


async def _collect(run, sink):
    async for execution in run.approvals:
        sink.append(execution)


# ── notify() ──────────────────────────────────────────────────────────────────


async def test_notify_needs_an_execution_that_names_its_conversation():
    session = CLEARED_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(session, tool_registry=FakeToolRegistry([AddTool()]), now=1000)
    run = runner.run()
    orphan = session.entries["te1"].model_copy(update={"conversation_id": None})

    with pytest.raises(AgentError, match="needs an execution that carries a conversation_id"):
        run.notify(orphan)


async def test_notify_marks_the_conversation_for_a_recheck():
    # no decision travels through it — it says only "look again"
    session = GATED_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(session, tool_registry=FakeToolRegistry([AddTool()]), now=1000)
    run = runner.run()

    run.notify(session.entries["te1"])

    assert "c1" in runner._recheck


async def test_notify_is_sync_and_legal_before_the_run_has_driven():
    session = GATED_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(session, tool_registry=FakeToolRegistry([AddTool()]), now=1000)
    run = runner.run()

    assert run.notify(session.entries["te1"]) is None


async def test_a_drive_consumes_the_recheck_flag_before_asking_decide():
    # clear first, then read: a notify() landing WHILE decide() is in flight
    # has to survive it, and consuming the flag afterwards would swallow it
    session = GATED_SESSION.model_copy(deep=True)
    session.entries["te1"] = session.entries["te1"].model_copy(
        update={"approval_status": None, "approval_decisions": []},
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry(
            [AddTool()],
            decisions=[ApprovalDecision(decision=ApprovalOption.PENDING, created_at=1000)],
        ),
        now=1000,
    )
    run = runner.run()
    run.notify(session.entries["te1"])

    async with run:
        _ = [event async for event in run]

    assert "c1" not in runner._recheck


# ── the degenerate park is unchanged ──────────────────────────────────────────


async def test_a_gate_with_no_subtree_still_ends_the_run():
    # the rule is "a drive returns when nothing in its SUBTREE can advance" —
    # with no children that is the moment the gate appears, exactly as before
    session = GATED_SESSION.model_copy(deep=True)
    session.entries["te1"] = session.entries["te1"].model_copy(
        update={"approval_status": None, "approval_decisions": []},
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry(
            [AddTool()],
            decisions=[ApprovalDecision(decision=ApprovalOption.PENDING, created_at=1000)],
        ),
        now=1000,
    )

    async with runner.run() as run:
        events = [event async for event in run]

    assert [type(e).__name__ for e in events] == [ApprovalRequired.__name__]
    assert runner.blocked()
    assert run.result.status is ConversationStatus.BLOCKED
    assert run.result.outcome is None
    assert [e.id for e in run.result.pending_approvals] == ["te1"]


async def test_approval_required_is_announced_once_per_park():
    session = GATED_SESSION.model_copy(deep=True)
    session.entries["te1"] = session.entries["te1"].model_copy(
        update={"approval_status": None, "approval_decisions": []},
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry(
            [AddTool()],
            decisions=[ApprovalDecision(decision=ApprovalOption.PENDING, created_at=1000)],
        ),
        now=1000,
    )

    async with runner.run() as run:
        events = [event async for event in run]

    assert sum(isinstance(e, ApprovalRequired) for e in events) == 1


# ── cancel is conversation-scoped ─────────────────────────────────────────────


async def test_run_cancel_targets_its_own_conversation():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
        ]
    )
    session = make_session(
        id="s_cancel_scope",
        conversations={"c1": conversation("c1", [], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()]),
        provider=faux,
        ids=["u1", "ts", "cr", "tf"],
        now=1000,
    )
    runner.post_message("add")

    # cancel BEFORE any drive: the bracket is not open yet, so the first drive
    # opens it and immediately winds down — no LLM call is made at all
    run = runner.start()
    run.cancel()

    async with run:
        _ = [event async for event in run]

    assert runner.session.entries["cr"].outcome is TurnOutcome.CANCELLED
    assert run.result.outcome is TurnOutcome.CANCELLED
    assert runner.idle()


async def test_cancelling_a_conversation_with_no_open_turn_is_a_no_op():
    session = make_session(
        id="s_cancel_noop",
        conversations={"c1": conversation("c1", [], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(session, now=1000)

    runner.cancel(conversation_id="c1")

    assert runner.session.entries == {}


# ── run state is per conversation ─────────────────────────────────────────────


async def test_the_live_run_is_tracked_under_its_conversation():
    session = CLEARED_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()]),
        provider=answering(),
        ids=["a2", "tf"],
        now=1000,
    )
    run = runner.run()

    async with run:
        await run.__anext__()
        assert runner._runs == {"c1": run}
        _ = [event async for event in run]

    assert runner._runs == {}


async def test_an_awaiting_execution_still_reaches_pending_approvals():
    session = GATED_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(session, tool_registry=FakeToolRegistry([AddTool()]), now=1000)

    [pending] = runner.pending_approvals()

    assert pending.id == "te1"
    assert pending.approval_status is ApprovalStatus.PENDING
    assert pending.conversation_id == "c1"
