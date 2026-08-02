"""`subagents_max_workers` — the session-wide worker pool, and the subagent
lifecycle events that make it observable.

The rule under test: a slot is held only while a subagent does its own
productive work. A conversation waiting on its children, on a human, or
winding a cancelled turn down holds none — which is why a nested tree cannot
deadlock at any cap, and why `subagents_max_workers=1` is legal and correct.

Determinism notes: the faux transport serves responses FIFO per REQUEST, so
capped tests either run at cap 1 (fully serialized) or script identical
responses for every sibling at the same stage, and assert structure rather
than which child said what.
"""

import asyncio

import pytest
from pydantic import BaseModel, ConfigDict

from luca.agent.core import AgentMiddlewareMixin, AgentSession
from luca.agent.core.events import (
    SubagentFinished,
    SubagentPaused,
    SubagentsSpawned,
    SubagentStarted,
)
from luca.agent.core.exceptions import AgentError
from luca.agent.core.models import (
    ApprovalDecision,
    ApprovalOption,
    ConversationStatus,
    TurnOutcome,
    TurnStart,
)
from luca.client.testing import faux_assistant_message, faux_text, faux_tool_call
from tests.agent.scenarios import AddTool, DeterministicRunner, FakeTool
from tests.agent.subagents.conftest import SubagentRegistry, spawn_call, subagent_session

IDS = [f"x{n}" for n in range(120)]

LIFECYCLE = (SubagentStarted, SubagentPaused, SubagentFinished)


class SlowTool(FakeTool):
    """A body slow enough that admission order and queue state are observable
    while it runs."""

    name = "slow"
    description = "Sleeps."

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")

    async def _execute(self, args, session, conversation_id, *, cancellation_token) -> str:
        await asyncio.sleep(0.05)
        return "slow done"


def lifecycle(events) -> list:
    return [e for e in events if isinstance(e, LIFECYCLE)]


def peak_concurrency(events) -> int:
    """Replay the lifecycle stream: how many subagents were ever running at
    once, per the events themselves."""
    running: set[str] = set()
    peak = 0
    for event in events:
        if isinstance(event, SubagentStarted):
            running.add(event.conversation_id)
            peak = max(peak, len(running))
        elif isinstance(event, (SubagentPaused, SubagentFinished)):
            running.discard(event.conversation_id)
    return peak


def children_of(session: AgentSession) -> list[str]:
    return [cid for cid, c in session.conversations.items() if c.depth == 1]


def links(session: AgentSession, conversation_id: str = "c1"):
    from luca.agent.core import ChildConversation

    return [
        entry
        for node in session.conversations[conversation_id].nodes
        if isinstance(entry := session.entries[node], ChildConversation)
    ]


def spawns(count: int):
    return faux_assistant_message(
        [spawn_call(f"work {n}", f"task {n}", call_id=f"tc{n}", task_id=f"t{n}") for n in range(count)],
        finish_reason="tool_use",
    )


def texts(*bodies: str):
    return [faux_assistant_message([faux_text(body)], finish_reason="stop") for body in bodies]


# ── T1 / T21 / T25: the default pool is invisible, the events are not ────────


async def test_the_default_pool_starts_every_child_immediately(faux):
    faux.set_responses([spawns(2), *texts("done", "done", "final")])
    session = subagent_session()
    runner = DeterministicRunner(session, tool_registry=SubagentRegistry(), provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")

    async with runner.run() as run:
        events = [event async for event in run]

    [spawned] = [e for e in events if isinstance(e, SubagentsSpawned)]
    children = spawned.conversation_ids
    # both starts are announced immediately after the batch, on the same queue
    at = events.index(spawned)
    assert [type(e) for e in events[at : at + 3]] == [SubagentsSpawned, SubagentStarted, SubagentStarted]
    assert [e.conversation_id for e in events[at + 1 : at + 3]] == children
    # exactly one Started and one Finished per child, attributed to the child
    started = [e for e in events if isinstance(e, SubagentStarted)]
    finished = [e for e in events if isinstance(e, SubagentFinished)]
    assert sorted(e.conversation_id for e in started) == sorted(children)
    assert sorted(e.conversation_id for e in finished) == sorted(children)
    assert all(e.outcome is TurnOutcome.COMPLETED for e in finished)
    assert not any(isinstance(e, SubagentPaused) for e in events)
    assert all(e.conversation_id in children for e in lifecycle(events))
    assert runner.idle()


# ── T2 / T20: a flat cap bounds concurrency, and the stream shows it ─────────


async def test_a_cap_bounds_how_many_children_work_at_once(faux):
    faux.set_responses([spawns(4), *texts("done", "done", "done", "done", "final")])
    session = subagent_session(subagents_max_workers=2)
    runner = DeterministicRunner(session, tool_registry=SubagentRegistry(), provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")

    async with runner.run() as run:
        events = [event async for event in run]

    [spawned] = [e for e in events if isinstance(e, SubagentsSpawned)]
    # exactly the first two start with the announcement…
    at = events.index(spawned)
    assert [type(e) for e in events[at + 1 : at + 3]] == [SubagentStarted, SubagentStarted]
    assert [e.conversation_id for e in events[at + 1 : at + 3]] == spawned.conversation_ids[:2]
    # …and the stream never shows more than two running at once
    started = [e for e in events if isinstance(e, SubagentStarted)]
    finished = [e for e in events if isinstance(e, SubagentFinished)]
    assert len(started) == len(finished) == 4
    assert peak_concurrency(events) == 2
    third_start = events.index(started[2])
    first_finish = events.index(finished[0])
    assert first_finish < third_start  # the third start waited for a slot
    # every child resolved and the parent answered with all four results
    assert all(link.execution_result is not None for link in links(session))
    assert runner.idle()


# ── T3: a queued child is BUSY, holds only its seed, and is never resolved ───


async def test_a_queued_child_is_busy_and_untouched_until_admitted(faux):
    faux.set_responses(
        [
            faux_assistant_message(
                [
                    spawn_call("work slowly", "slow one", call_id="tc1", task_id="t1"),
                    spawn_call("work later", "queued one", call_id="tc2", task_id="t2"),
                ],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_tool_call("slow", {}, id="tcS")], finish_reason="tool_use"),
            *texts("first done", "second done", "final"),
        ]
    )
    session = subagent_session(subagents_max_workers=1)
    runner = DeterministicRunner(
        session,
        tool_registry=SubagentRegistry([SlowTool()]),
        provider=faux,
        ids=list(IDS),
        now=1000,
    )
    runner.post_message("go")
    queued_snapshot: tuple = ()

    async with runner.run() as run:
        async for event in run:
            if isinstance(event, SubagentsSpawned) and event.conversation_id == "c1":
                queued = event.conversation_ids[1]
                queued_snapshot = (
                    len(session.conversations[queued].nodes),
                    session.get_conversation_status(queued).status,
                )

    # at announcement time the queued child held exactly its seed message and
    # derived BUSY — "there is work here, and a run would do it"
    assert queued_snapshot == (1, ConversationStatus.BUSY)
    assert all(link.execution_result is not None for link in links(session))
    assert runner.idle()


# ── T4: admission is FIFO, in model-request order ────────────────────────────


async def test_admission_order_is_fifo(faux):
    faux.set_responses([spawns(3), *texts("done", "done", "done", "final")])
    session = subagent_session(subagents_max_workers=1)
    runner = DeterministicRunner(session, tool_registry=SubagentRegistry(), provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")

    async with runner.run() as run:
        events = [event async for event in run]

    [spawned] = [e for e in events if isinstance(e, SubagentsSpawned)]
    started = [e.conversation_id for e in events if isinstance(e, SubagentStarted)]
    assert started == spawned.conversation_ids
    assert peak_concurrency(events) == 1
    assert runner.idle()


# ── T5 / T6: nesting never deadlocks — the release-on-park rule ──────────────


async def test_nesting_does_not_deadlock_at_cap_one(faux):
    faux.set_responses(
        [
            faux_assistant_message([spawn_call("child work", "child", call_id="tc1")], finish_reason="tool_use"),
            faux_assistant_message([spawn_call("leaf work", "leaf", call_id="tc2")], finish_reason="tool_use"),
            *texts("leaf done", "child done", "all done"),
        ]
    )
    session = subagent_session(max_depth=2, subagents_max_workers=1)
    runner = DeterministicRunner(session, tool_registry=SubagentRegistry(), provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")

    async with runner.run() as run:
        _ = [event async for event in run]

    # the parent released its slot when it parked on the leaf — the whole rule
    assert sorted(c.depth for c in session.conversations.values()) == [0, 1, 2]
    assert all(session.get_conversation_status(cid).status is ConversationStatus.IDLE for cid in session.conversations)
    assert faux.requests[-2].messages[-1].content[0].text == (
        'Subagent task update:\n<task id=t1 status=completed completed_at="1970-01-01T00:00:01Z">\nleaf done\n</task>'
    )
    assert runner.idle()


async def test_nesting_under_a_wider_cap(faux):
    leaf_spawn = faux_assistant_message(
        [spawn_call("leaf work", "leaf", call_id="tcL", task_id="tl")], finish_reason="tool_use"
    )
    faux.set_responses(
        [
            spawns(2),
            leaf_spawn,  # both children run concurrently and each pops one
            leaf_spawn,
            *texts("leaf done", "leaf done", "child done", "child done", "all done"),
        ]
    )
    session = subagent_session(max_depth=2, subagents_max_workers=2)
    runner = DeterministicRunner(session, tool_registry=SubagentRegistry(), provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")

    async with runner.run() as run:
        events = [event async for event in run]

    assert sorted(c.depth for c in session.conversations.values()) == [0, 1, 1, 2, 2]
    assert all(session.get_conversation_status(cid).status is ConversationStatus.IDLE for cid in session.conversations)
    # NO event-level concurrency assertion here: a nested parent parked on its
    # child emits nothing (deliberately — its subtree is plainly working), so
    # the stream's running-set legitimately exceeds the cap in a nested tree.
    subagents = [cid for cid, c in session.conversations.items() if c.depth > 0]
    started = [e for e in events if isinstance(e, SubagentStarted)]
    finished = [e for e in events if isinstance(e, SubagentFinished)]
    assert sorted(e.conversation_id for e in started) == sorted(subagents)
    assert sorted(e.conversation_id for e in finished) == sorted(subagents)
    assert runner.idle()


# ── T7 / T23: a gate releases its slot, and Paused precedes the next start ───


def gated_first_child(extra_decisions: list[ApprovalDecision] = ()):
    registry = SubagentRegistry([AddTool()])
    registry.other.decisions = [
        ApprovalDecision(decision=ApprovalOption.PENDING, created_at=1000),
        *extra_decisions,
    ]
    return registry


def gated_first_script(faux):
    faux.set_responses(
        [
            spawns(2),
            faux_assistant_message([faux_tool_call("add", {"a": 1, "b": 2}, id="tcA")], finish_reason="tool_use"),
            # the sibling's resolution wakes the parent for one update round
            # before it parks on the still-gated first child
            *texts("second done", "waiting on the first task"),
        ]
    )


async def test_a_gate_releases_its_slot_and_paused_precedes_the_next_start(faux):
    gated_first_script(faux)
    session = subagent_session(subagents_max_workers=1)
    runner = DeterministicRunner(session, tool_registry=gated_first_child(), provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")

    async with runner.run() as run:
        events = [event async for event in run]

    [spawned] = [e for e in events if isinstance(e, SubagentsSpawned)]
    first, second = spawned.conversation_ids
    # the gated child paused, and only then did the freed slot start its sibling
    [paused] = [e for e in events if isinstance(e, SubagentPaused)]
    assert paused.conversation_id == first
    second_started = next(e for e in events if isinstance(e, SubagentStarted) and e.conversation_id == second)
    assert events.index(paused) < events.index(second_started)
    # the sibling finished; the gated child holds the tree open
    assert [link.execution_result is not None for link in links(session)] == [False, True]
    assert [ex.conversation_id for ex in runner.pending_approvals()] == [first]
    assert runner.blocked()


async def test_an_answered_gate_re_acquires_and_announces(faux):
    # T8 / T22 — the recovery contract under a cap: answer, run again; the
    # restart is a start a consumer has to see
    gated_first_script(faux)
    session = subagent_session(subagents_max_workers=1)
    registry = gated_first_child([ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000)])
    runner = DeterministicRunner(session, tool_registry=registry, provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")

    async with runner.run() as run:
        _ = [event async for event in run]
    assert runner.blocked()

    [gated_id] = [ex.conversation_id for ex in runner.pending_approvals()]
    faux.set_responses(texts("first done", "final"))
    async with runner.run() as run:  # the registry now allows
        second_events = [event async for event in run]

    assert [e.conversation_id for e in second_events if isinstance(e, SubagentStarted)] == [gated_id]
    [finished] = [e for e in second_events if isinstance(e, SubagentFinished)]
    assert (finished.conversation_id, finished.outcome) == (gated_id, TurnOutcome.COMPLETED)
    assert runner.idle()


# ── T9 / T10 / T11 / T24 / T27: cancellation against the pool ────────────────


async def test_cancelling_a_parked_subtree_never_grants_its_queued_child(faux):
    # One scenario, four claims: a parked parent whose re-acquire is pending
    # still consumes its cancel (T11); its queued child is never granted a
    # slot and closes seeded-cancelled-settled (T10); the wind-down needs no
    # slot while an unrelated subagent keeps working (T9); and BOTH endings
    # are announced — the cancelled parent's Finished(CANCELLED), and the
    # never-started child's, whose Finished with no Started reads "cancelled
    # before admission".
    faux.set_responses(
        [
            faux_assistant_message(
                [
                    spawn_call("spawn a leaf", "nested", call_id="tc1", task_id="t1"),
                    spawn_call("work slowly", "unrelated", call_id="tc2", task_id="t2"),
                ],
                finish_reason="tool_use",
            ),
            faux_assistant_message([spawn_call("leaf work", "leaf", call_id="tcL")], finish_reason="tool_use"),
            faux_assistant_message([faux_tool_call("slow", {}, id="tcS")], finish_reason="tool_use"),
            # the cancelled subtree resolves while the slow sibling still
            # works, so the parent takes a wake round with that first update
            *texts("noted the cancellation", "slow child done", "only the slow one came back"),
        ]
    )
    session = subagent_session(max_depth=2, subagents_max_workers=1)
    runner = DeterministicRunner(
        session,
        tool_registry=SubagentRegistry([SlowTool()]),
        provider=faux,
        ids=list(IDS),
        now=1000,
    )
    runner.post_message("go")
    nested_id = ""
    events = []

    async with runner.run() as run:
        async for event in run:
            events.append(event)
            if isinstance(event, SubagentsSpawned) and event.conversation_id == "c1":
                nested_id = event.conversation_ids[0]
            if isinstance(event, SubagentsSpawned) and event.conversation_id == nested_id:
                # the nested child spawned its leaf and parked; cancel it while
                # its re-acquire is queued behind the slow sibling
                runner.cancel(conversation_id=nested_id)

    assert runner.idle()
    [leaf_id] = [cid for cid, c in session.conversations.items() if c.depth == 2]
    # the queued leaf was never granted a slot: seeded, cancelled, settled
    assert [session.entries[n].type for n in session.conversations[leaf_id].nodes] == [
        "user",
        "turn_start",
        "cancel_requested",
        "turn_finish",
    ]
    # …and its flush announced the ending: Finished(CANCELLED) with no Started
    assert SubagentFinished(conversation_id=leaf_id, outcome=TurnOutcome.CANCELLED) in events
    assert not any(isinstance(e, SubagentStarted) and e.conversation_id == leaf_id for e in events)
    # the cancelled subtree resolved with an error result; the sibling's is real
    assert all(link.execution_result is not None for link in links(session))
    nested_link = next(link for link in links(session) if link.conversation_id == nested_id)
    other_link = next(link for link in links(session) if link.conversation_id != nested_id)
    assert nested_link.execution_result.is_error is True
    assert other_link.execution_result.is_error is False


async def test_a_cancelled_running_child_finishes_with_outcome_cancelled(faux):
    faux.set_responses(
        [
            faux_assistant_message([spawn_call("work slowly", "slow", call_id="tc1")], finish_reason="tool_use"),
            faux_assistant_message([faux_tool_call("slow", {}, id="tcS")], finish_reason="tool_use"),
            *texts("nothing came back"),
        ]
    )
    session = subagent_session()
    runner = DeterministicRunner(
        session,
        tool_registry=SubagentRegistry([SlowTool()]),
        provider=faux,
        ids=list(IDS),
        now=1000,
    )
    runner.post_message("go")

    async with runner.run() as run:
        events = []
        async for event in run:
            events.append(event)
            if type(event).__name__ == "ToolExecutionStarted" and event.conversation_id != "c1":
                runner.cancel(conversation_id=event.conversation_id)

    [child_id] = children_of(session)
    [finished] = [e for e in events if isinstance(e, SubagentFinished)]
    assert (finished.conversation_id, finished.outcome) == (child_id, TurnOutcome.CANCELLED)
    assert not any(isinstance(e, SubagentPaused) for e in events)
    assert runner.idle()


async def test_an_errored_child_finishes_with_outcome_errored(faux):
    faux.set_responses(
        [
            faux_assistant_message([spawn_call("A", "a", call_id="tc1")], finish_reason="tool_use"),
            faux_assistant_message([faux_tool_call("add", {"a": 1, "b": 2}, id="tcA")], finish_reason="tool_use"),
            *texts("parent carries on"),
        ]
    )
    session = subagent_session(subagent_hard_max_steps=1)
    runner = DeterministicRunner(
        session,
        tool_registry=SubagentRegistry([AddTool()]),
        provider=faux,
        ids=list(IDS),
        now=1000,
    )
    runner.post_message("go")

    async with runner.run() as run:
        events = [event async for event in run]

    [child_id] = children_of(session)
    [finished] = [e for e in events if isinstance(e, SubagentFinished)]
    assert (finished.conversation_id, finished.outcome) == (child_id, TurnOutcome.ERRORED)
    assert runner.idle()


# ── cancellation of QUEUED children: the paths with no drive to consume ──────


async def test_cancelling_a_queued_grandchild_wakes_the_flushing_parent(faux):
    # The deadlock regression: a queued child has no drive and no token, so
    # cancel() must wake the PARENT — the only conversation that will flush
    # it. Without that, the parent parks forever on a CANCELLING child while
    # the pool sits empty. The slow sibling is what keeps the leaf QUEUED long
    # enough for the cancel to land before any admission.
    faux.set_responses(
        [
            faux_assistant_message(
                [
                    spawn_call("spawn a leaf", "nested", call_id="tc1", task_id="t1"),
                    spawn_call("work slowly", "unrelated", call_id="tc2", task_id="t2"),
                ],
                finish_reason="tool_use",
            ),
            faux_assistant_message([spawn_call("leaf work", "leaf", call_id="tcL")], finish_reason="tool_use"),
            faux_assistant_message([faux_tool_call("slow", {}, id="tcS")], finish_reason="tool_use"),
            # the slow sibling resolves first → the main conversation's wake
            # round; the nested parent then re-acquires the freed slot,
            # flushes its cancelled leaf and takes its own wake round
            *texts("slow child done", "waiting on the nested task", "the leaf was cancelled", "all done"),
        ]
    )
    session = subagent_session(max_depth=2, subagents_max_workers=1)
    runner = DeterministicRunner(
        session,
        tool_registry=SubagentRegistry([SlowTool()]),
        provider=faux,
        ids=list(IDS),
        now=1000,
    )
    runner.post_message("go")
    nested_id = ""

    async with asyncio.timeout(10):
        async with runner.run() as run:
            async for event in run:
                if isinstance(event, SubagentsSpawned) and event.conversation_id == "c1":
                    nested_id = event.conversation_ids[0]
                if isinstance(event, SubagentsSpawned) and event.conversation_id == nested_id:
                    # cancel the QUEUED leaf only — its parent must be woken
                    runner.cancel(conversation_id=event.conversation_ids[0])

    assert runner.idle()
    [leaf_id] = [cid for cid, c in session.conversations.items() if c.depth == 2]
    assert [session.entries[n].type for n in session.conversations[leaf_id].nodes] == [
        "user",
        "turn_start",
        "cancel_requested",
        "turn_finish",
    ]
    assert all(session.get_conversation_status(cid).status is ConversationStatus.IDLE for cid in session.conversations)


async def test_awaiting_a_queued_handle_survives_its_cancellation(faux):
    # The joiner regression: a queued child cancelled before admission is
    # ABANDONED — `await run.child(cid)` returns the session-derived result
    # instead of blocking on a wake that would never come.
    faux.set_responses(
        [
            faux_assistant_message(
                [
                    spawn_call("work slowly", "slow one", call_id="tc1", task_id="t1"),
                    spawn_call("never starts", "doomed", call_id="tc2", task_id="t2"),
                ],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_tool_call("slow", {}, id="tcS")], finish_reason="tool_use"),
            # the abandoned child is flushed and resolved while the slow one
            # still works — the parent takes a wake round with that update
            *texts("doomed is gone", "first done", "only the first came back"),
        ]
    )
    session = subagent_session(subagents_max_workers=1)
    runner = DeterministicRunner(
        session,
        tool_registry=SubagentRegistry([SlowTool()]),
        provider=faux,
        ids=list(IDS),
        now=1000,
    )
    runner.post_message("go")
    joins: list[asyncio.Task] = []

    async def join(handle):
        return await handle

    async with asyncio.timeout(10):
        async with runner.run() as run:
            async for event in run:
                if isinstance(event, SubagentsSpawned):
                    queued = run.child(event.conversation_ids[1])
                    joins.append(asyncio.ensure_future(join(queued)))
                    runner.cancel(conversation_id=event.conversation_ids[1])
        [joined] = joins
        result = await joined

    assert result is not None
    assert runner.idle()


# ── ordering: Started always precedes the child's own events ─────────────────


async def test_started_precedes_the_childs_own_events_for_a_slow_consumer(faux):
    # The overtaking regression: with fast children and a consumer that
    # awaits between pulls, an engine-yielded start would lose to the child's
    # inbox-forwarded events (Finished before Started). The announcement and
    # every start ride the same inbox the child events do, in order.
    faux.set_responses([spawns(4), *texts("done", "done", "done", "done", "final")])
    session = subagent_session()
    runner = DeterministicRunner(session, tool_registry=SubagentRegistry(), provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")
    events = []

    async with runner.run() as run:
        async for event in run:
            events.append(event)
            await asyncio.sleep(0.001)  # a real consumer renders between pulls

    [spawned] = [e for e in events if isinstance(e, SubagentsSpawned)]
    spawned_at = events.index(spawned)
    for cid in spawned.conversation_ids:
        child_events = [i for i, e in enumerate(events) if e.conversation_id == cid]
        started_at = next(
            i for i, e in enumerate(events) if isinstance(e, SubagentStarted) and e.conversation_id == cid
        )
        assert started_at == min(child_events)  # nothing of the child's precedes its start
        assert spawned_at < started_at
    assert runner.idle()


# ── T12 / T13: the pool is rebuilt from scratch on resume ────────────────────


async def test_reload_mid_tree_resumes_under_the_cap(faux):
    gated_first_script(faux)
    session = subagent_session(subagents_max_workers=1)
    runner = DeterministicRunner(session, tool_registry=gated_first_child(), provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")
    async with runner.run() as run:
        _ = [event async for event in run]
    payload = runner.session.model_dump_json()  # the process exits here

    reloaded = AgentSession.model_validate_json(payload)
    faux.set_responses(texts("first done", "final"))
    allowing = SubagentRegistry([AddTool()])
    allowing.other.decisions = [ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000)]
    resumed = DeterministicRunner(
        reloaded,
        tool_registry=allowing,
        provider=faux,
        ids=[f"y{n}" for n in range(40)],
        now=1000,
    )

    async with resumed.run() as run:
        events = [event async for event in run]

    assert any(isinstance(e, SubagentStarted) for e in events)  # the restart announced
    assert resumed.idle()


async def test_lowering_the_cap_mid_session_serializes_the_restarts(faux):
    faux.set_responses(
        [
            spawns(2),
            faux_assistant_message([faux_tool_call("add", {"a": 1, "b": 2}, id="tcA")], finish_reason="tool_use"),
            faux_assistant_message([faux_tool_call("add", {"a": 3, "b": 4}, id="tcB")], finish_reason="tool_use"),
        ]
    )
    session = subagent_session()  # no cap yet — both children gate in parallel
    registry = SubagentRegistry([AddTool()])
    registry.other.decisions = [
        ApprovalDecision(decision=ApprovalOption.PENDING, created_at=1000),
        ApprovalDecision(decision=ApprovalOption.PENDING, created_at=1000),
        ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000),
        ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000),
    ]
    runner = DeterministicRunner(session, tool_registry=registry, provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")
    async with runner.run() as run:
        _ = [event async for event in run]
    assert runner.blocked()

    # the knob is persisted config, read live — lowering it between runs is
    # an ordinary path, and the pool re-applies from scratch
    session.session_config.runtime_config.subagents_max_workers = 1
    faux.set_responses(texts("done", "done", "final"))
    async with runner.run() as run:
        events = [event async for event in run]

    assert peak_concurrency(events) == 1
    assert len([e for e in events if isinstance(e, SubagentFinished)]) == 2
    assert runner.idle()


# ── T14: the cap is framework-owned ──────────────────────────────────────────


async def test_the_cap_refuses_application_driven_children(faux):
    session = subagent_session(subagents_max_workers=2)
    runner = DeterministicRunner(session, tool_registry=SubagentRegistry(), provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")

    with pytest.raises(AgentError, match="framework-owned"):
        runner.run(autostart_subagents=False)

    # the default is unaffected, and so is start(), which implies True
    faux.set_responses(texts("hello"))
    async with runner.run() as run:
        _ = [event async for event in run]
    assert runner.idle()


async def test_start_is_never_affected_by_the_cap(faux):
    faux.set_responses(texts("hello"))
    session = subagent_session(subagents_max_workers=1)
    runner = DeterministicRunner(session, tool_registry=SubagentRegistry(), provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")

    result = await runner.start()

    assert result is not None
    assert runner.idle()


# ── T16: a queued handle is still a handle ───────────────────────────────────


async def test_a_queued_handle_is_consumable(faux):
    faux.set_responses([spawns(2), *texts("done", "done", "final")])
    session = subagent_session(subagents_max_workers=1)
    runner = DeterministicRunner(session, tool_registry=SubagentRegistry(), provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")
    joined = None

    async with runner.run() as run:
        async for event in run:
            if isinstance(event, SubagentsSpawned):
                queued = run.child(event.conversation_ids[1])
                assert queued is not None
                joined = await queued  # blocks until admitted, then finished

    assert joined is not None
    assert runner.idle()


# ── T18: a pool-granted start that fails surfaces on the parent ──────────────


class ThirdBracketBomb(AgentMiddlewareMixin):
    """Raises on the third `TurnStart` append — main's, the first child's,
    then the QUEUED child's, whose start has no synchronous caller."""

    def __init__(self) -> None:
        self.turn_starts = 0

    def before_entry_written(self, entry):
        if isinstance(entry, TurnStart):
            self.turn_starts += 1
            if self.turn_starts == 3:
                raise RuntimeError("third bracket refused")
        return entry


async def test_admission_failure_surfaces_on_the_parents_run(faux):
    faux.set_responses([spawns(2), *texts("done")])
    session = subagent_session(subagents_max_workers=1)
    runner = DeterministicRunner(
        session,
        tool_registry=SubagentRegistry(),
        provider=faux,
        ids=list(IDS),
        now=1000,
        middleware=[ThirdBracketBomb()],
    )
    runner.post_message("go")

    with pytest.raises(RuntimeError, match="third bracket refused"):
        async with runner.run() as run:
            _ = [event async for event in run]


# ── T19: the model is never told about the cap ───────────────────────────────


async def test_the_model_is_not_told_about_the_cap(faux):
    def scripted():
        return [
            faux_assistant_message([spawn_call("A", "a", call_id="tc1")], finish_reason="tool_use"),
            *texts("A done", "final"),
        ]

    async def drive(max_workers_kwargs: dict, provider) -> AgentSession:
        provider.set_responses(scripted())
        session = subagent_session(**max_workers_kwargs)
        runner = DeterministicRunner(
            session, tool_registry=SubagentRegistry(), provider=provider, ids=list(IDS), now=1000
        )
        runner.post_message("go")
        async with runner.run() as run:
            _ = [event async for event in run]
        return session

    from luca.client.testing import FauxProvider

    uncapped_faux, capped_faux = FauxProvider(), FauxProvider()
    uncapped = await drive({}, uncapped_faux)
    capped = await drive({"subagents_max_workers": 1}, capped_faux)

    # request for request: same messages, same tool list — the cap is a
    # scheduling detail with no transcript footprint
    assert [r.messages for r in uncapped_faux.requests] == [r.messages for r in capped_faux.requests]
    assert [r.tools for r in uncapped_faux.requests] == [r.tools for r in capped_faux.requests]
    # and the durable record has the identical shape, conversation for
    # conversation (full-session equality is off the table only because the
    # Yolo policy stamps decisions with wall-clock time)
    assert set(uncapped.conversations) == set(capped.conversations)
    for cid in uncapped.conversations:
        assert [uncapped.entries[n].type for n in uncapped.conversations[cid].nodes] == [
            capped.entries[n].type for n in capped.conversations[cid].nodes
        ]


# ── T26: no lifecycle events for application-driven children ─────────────────


async def test_no_lifecycle_events_under_application_driven_children(faux):
    faux.set_responses([spawns(2), *texts("A done", "B done", "final")])
    session = subagent_session()  # Inf — the capped combination raises (T14)
    runner = DeterministicRunner(session, tool_registry=SubagentRegistry(), provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")
    child_events: list = []

    async def drive(child):
        async with child:
            child_events.extend([event async for event in child])

    run = runner.run(autostart_subagents=False)
    parent_events: list = []
    async with run:
        async for event in run:
            parent_events.append(event)
            if isinstance(event, SubagentsSpawned):
                await asyncio.gather(*(drive(run.child(cid)) for cid in event.conversation_ids))

    assert runner.idle()
    assert lifecycle(parent_events) == []
    assert lifecycle(child_events) == []
