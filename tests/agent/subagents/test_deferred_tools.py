"""Deferred tool calling inside a subagent tree (spec 0007 × the subagents).

`tests/agent/test_deferred_tools.py` pins the deferral itself on one
conversation. This file pins the four places a TREE makes it a different
question, and nothing else:

- DISCOVERY. A parked subagent is found by asking the PARENT:
  `pending_deferred_tool_executions()` is subtree-scoped exactly like
  `pending_approvals()`, and each execution is attributed through its own
  `conversation_id`. It is a plain read over the session, so it answers while
  the parent's run is still live — which is the only moment that exists when a
  sibling is still working.
- THE THIRD STATUS TERM. `_derive_status`'s subagent branch has to name a
  parked call beside a gated one. Left to the unseen-material term below it,
  ANY completed sibling satisfies that term, the parent derives BUSY, and a
  `while not idle(): run()` driver spins on a drive that only re-parks. The
  same "unless an unseen post" guard applies, because both project a
  placeholder.
- RE-DISPATCH FROM OUTSIDE. A subagent's drive ENDS when it parks — there is
  nothing left to signal — so waking it is a RESTART, and the two doors are the
  same ones a gated subagent uses: `run.notify(execution)` inside a live run,
  and `_restart_unresolved_children` on the next `run()`. Neither needed a code
  change for deferrals, which is the claim under test.
- THE WORKER POOL. A parked subagent holds no slot. The park goes through
  `_await_subtree`, which releases before waiting, so at
  `subagents_max_workers=1` a deferral admits the queued sibling.

Ordering is deterministic the same way it is in the sibling files: the faux
transport serves responses FIFO per REQUEST, and children issue their first
request in spawn order, so the FIRST child is always the one scripted to defer.
"""

import asyncio

from pydantic import BaseModel, ConfigDict

from luca.agent.core import AgentSession, ChildConversation
from luca.agent.core.events import SubagentPaused, SubagentsSpawned, SubagentStarted
from luca.agent.core.models import (
    ConversationStatus,
    ExecutionAttempt,
    ExecutionAttemptOutcome,
    ExecutionStatus,
)
from luca.client.testing import faux_assistant_message, faux_text, faux_tool_call
from tests.agent.scenarios import DeferringTool, DeterministicRunner, FakeTool
from tests.agent.subagents.conftest import SubagentRegistry, spawn_call, subagent_session

IDS = [f"x{n}" for n in range(120)]

DEFERRED_ATTEMPT = ExecutionAttempt(
    outcome=ExecutionAttemptOutcome.DEFERRED,
    started_at=1000,
    ended_at=1000,
)


class SlowTool(FakeTool):
    """A body slow enough that a sibling parks, is discovered and is restarted
    while this one is still working."""

    name = "slow"
    description = "Sleeps."

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")

    async def _execute(self, args, session, conversation_id, *, tool_name, tool_call_id, cancellation_token) -> str:
        await asyncio.sleep(0.05)
        return "slow done"


def spawns(count: int):
    return faux_assistant_message(
        [spawn_call(f"work {n}", f"task {n}", call_id=f"tc{n}", task_id=f"t{n}") for n in range(count)],
        finish_reason="tool_use",
    )


def ask_round(call_id: str):
    return faux_assistant_message([faux_tool_call("ask", {"a": 1, "b": 2}, id=call_id)], finish_reason="tool_use")


def slow_round(call_id: str):
    return faux_assistant_message([faux_tool_call("slow", {}, id=call_id)], finish_reason="tool_use")


def texts(*bodies: str):
    return [faux_assistant_message([faux_text(body)], finish_reason="stop") for body in bodies]


def children_of(session: AgentSession) -> list[str]:
    return [cid for cid, c in session.conversations.items() if c.depth == 1]


def links(session: AgentSession, conversation_id: str = "c1") -> list[ChildConversation]:
    return [
        entry
        for node in session.conversations[conversation_id].nodes
        if isinstance(entry := session.entries[node], ChildConversation)
    ]


def parent_asks_and_spawns(count: int):
    """One assistant message that BOTH calls the deferring tool and spawns
    `count` children — the shape where a parked PARENT keeps looping."""
    return faux_assistant_message(
        [
            faux_tool_call("ask", {"a": 1, "b": 2}, id="tcP"),
            *[spawn_call(f"work {n}", f"task {n}", call_id=f"tc{n}", task_id=f"t{n}") for n in range(count)],
        ],
        finish_reason="tool_use",
    )


# ── one poll per park ──────────────────────────────────────────────────────────


async def test_a_parked_parent_polls_once_however_often_its_children_wake_it(faux):
    # ── precondition ─────────────────────────────────────────────────────────
    # A PARENT holding a parked call and four children whose bodies finish at
    # staggered times. The parent's drive MUST loop on every one of those wakes
    # — that is how it reaches the resolve step — but a wake is a SIBLING
    # finishing, not anything that could have answered the parked call.
    faux.set_responses(
        [
            parent_asks_and_spawns(4),
            *[slow_round(f"tcS{n}") for n in range(4)],
            *texts(*[f"child {n} done" for n in range(4)]),
        ]
    )
    tool = DeferringTool()
    runner = DeterministicRunner(
        subagent_session(subagents_max_per_turn=4),
        tool_registry=SubagentRegistry([tool, SlowTool()]),
        provider=faux,
        ids=list(IDS),
        now=1000,
    )
    runner.post_message("go")

    # ── action ───────────────────────────────────────────────────────────────
    await runner.run()

    # ── postcondition ────────────────────────────────────────────────────────
    # ONE poll, one attempt — not one per wake. Without the drive-local
    # already-polled set the tool body is invoked once per child completion,
    # which for a remote tool is a redundant out-of-process round trip per
    # unrelated sibling event.
    assert tool.dispatches == ["tcP"]
    parked = runner.pending_deferred_tool_executions()
    assert [(execution.tool_call_id, execution.attempts) for execution in parked] == [("tcP", [DEFERRED_ATTEMPT])]


async def test_notify_re_arms_the_poll_inside_the_same_drive(faux):
    # The counterpart: `notify()` is the ONE thing that says a parked call may
    # have moved, so it clears the already-polled set and the next pass really
    # does re-dispatch. Without that, holding the poll back would make
    # `run.notify()` inert for a parked PARENT.
    faux.set_responses([parent_asks_and_spawns(1), slow_round("tcS"), *texts("child done", "answer noted", "all set")])
    tool = DeferringTool()
    runner = DeterministicRunner(
        subagent_session(),
        tool_registry=SubagentRegistry([tool, SlowTool()]),
        provider=faux,
        ids=list(IDS),
        now=1000,
    )
    runner.post_message("go")

    parked = None
    async with runner.run() as run:
        async for event in run:
            # `SubagentsSpawned` is the first event AFTER the parent's own
            # dispatch batch, so the deferral has already happened by here —
            # notifying at `ToolExecutionStarted` would answer the call before
            # its body ever ran and prove nothing.
            if isinstance(event, SubagentsSpawned):
                parked = runner.pending_deferred_tool_executions()[0]
                tool.answers["tcP"] = "the answer arrived out of band"
                run.notify(parked)

    assert tool.dispatches == ["tcP", "tcP"]
    assert (
        runner.session.entries[
            next(
                node
                for node in runner.session.conversations["c1"].nodes
                if getattr(runner.session.entries[node], "tool_call_id", None) == "tcP"
            )
        ].status
        is ExecutionStatus.COMPLETED
    )


# ── discovery ──────────────────────────────────────────────────────────────────


async def test_a_parked_subagent_is_discovered_from_the_parent_mid_run(faux):
    # ── precondition ─────────────────────────────────────────────────────────
    # Two children: the first asks and parks, the second works long enough that
    # the parent's run cannot return — so the ONLY moment this parked call is
    # observable is from inside the live run.
    faux.set_responses([spawns(2), ask_round("tcA"), slow_round("tcS"), *texts("B done", "still waiting on A")])
    session = subagent_session()
    runner = DeterministicRunner(
        session,
        tool_registry=SubagentRegistry([DeferringTool(), SlowTool()]),
        provider=faux,
        ids=list(IDS),
        now=1000,
    )
    runner.post_message("go")
    snapshot: tuple = ()

    # ── action ───────────────────────────────────────────────────────────────
    async with runner.run() as run:
        async for event in run:
            if isinstance(event, SubagentPaused) and not snapshot:
                snapshot = (
                    session.get_conversation_status("c1").status,
                    event.conversation_id,
                    [
                        (ex.conversation_id, ex.tool_call_id, ex.status, ex.attempts)
                        for ex in runner.pending_deferred_tool_executions()
                    ],
                )

    # ── postcondition ────────────────────────────────────────────────────────
    # The parent is BUSY (a sibling is working) and answers for its whole
    # subtree: the parked execution is the CHILD's, attributed through its own
    # `conversation_id`. Only that projection is asserted — the executions carry
    # runner-generated ids this test does not own.
    first_child, second_child = children_of(session)
    assert snapshot == (
        ConversationStatus.BUSY,
        first_child,
        [(first_child, "tcA", ExecutionStatus.AWAITING_RESULT, [DEFERRED_ATTEMPT])],
    )
    # …and once the sibling has resolved, the same poll answers the same thing
    # with the tree at rest.
    assert [(ex.conversation_id, ex.tool_call_id) for ex in runner.pending_deferred_tool_executions()] == [
        (first_child, "tcA")
    ]
    assert [link.execution_result is not None for link in links(session)] == [False, True]
    assert session.get_conversation_status(second_child).status is ConversationStatus.IDLE
    assert runner.blocked()


# ── the third status term ──────────────────────────────────────────────────────


async def test_a_parked_parent_stays_blocked_when_a_child_completes(faux):
    # ── precondition ─────────────────────────────────────────────────────────
    # The parent spawns two children AND asks — so it owns a parked call of its
    # own while children are unresolved, which is the only shape that reaches
    # `_derive_status`'s subagent branch with something parked. One child parks
    # too, the other completes: its result is unseen material sitting in the
    # open turn.
    faux.set_responses(
        [
            faux_assistant_message(
                [
                    spawn_call("ask and park", "the parked one", call_id="tc0", task_id="t0"),
                    spawn_call("answer fast", "the quick one", call_id="tc1", task_id="t1"),
                    faux_tool_call("ask", {"a": 1, "b": 2}, id="tcP"),
                ],
                finish_reason="tool_use",
            ),
            ask_round("tcA"),
            *texts("B done"),
        ]
    )
    session = subagent_session()
    runner = DeterministicRunner(
        session,
        tool_registry=SubagentRegistry([DeferringTool()]),
        provider=faux,
        ids=list(IDS),
        now=1000,
    )
    runner.post_message("go")

    # ── action ───────────────────────────────────────────────────────────────
    async with runner.run() as run:
        _ = [event async for event in run]

    # ── postcondition ────────────────────────────────────────────────────────
    # THREE requests: the parent's own round and one per child. The completed
    # sibling did NOT wake the model — the park branch sits ahead of the
    # unresolved-children branch, so "a post is the only thing that projects a
    # parked call" holds inside a tree too.
    first_child, second_child = children_of(session)
    assert len(faux.requests) == 3
    # BLOCKED comes from the parked term. The completed sibling's result is
    # unseen material sitting in the parent's open turn, so the term below it
    # answers True here — without the parked term this derives BUSY and a
    # `while not idle(): run()` driver spins on drives that only re-park.
    assert {cid: session.get_conversation_status(cid).status for cid in session.conversations} == {
        "c1": ConversationStatus.BLOCKED,
        first_child: ConversationStatus.BLOCKED,
        second_child: ConversationStatus.IDLE,
    }
    assert runner.blocked()
    # subtree scope, root first: the parent's own parked call, then the child's
    assert [(ex.conversation_id, ex.tool_call_id) for ex in runner.pending_deferred_tool_executions()] == [
        ("c1", "tcP"),
        (first_child, "tcA"),
    ]


# ── re-dispatch from outside ───────────────────────────────────────────────────


async def test_notify_redispatches_a_parked_subagent_inside_the_same_run(faux):
    # ── precondition ─────────────────────────────────────────────────────────
    # The shape polling between drives cannot reach: one child parks while its
    # sibling works, so the run does not return. The parked child's drive is
    # already gone — `notify()` has to find that handle through the parent's
    # subtree and RESTART it, which re-runs the whole dispatch path from
    # `prepare()` down.
    faux.set_responses(
        [
            spawns(2),
            ask_round("tcA"),
            slow_round("tcS"),
            *texts("A done", "still waiting on B", "B done", "both done"),
        ]
    )
    tool = DeferringTool()
    session = subagent_session()
    runner = DeterministicRunner(
        session,
        tool_registry=SubagentRegistry([tool, SlowTool()]),
        provider=faux,
        ids=list(IDS),
        now=1000,
    )
    runner.post_message("go")
    resolved: list = []

    # ── action ───────────────────────────────────────────────────────────────
    async with asyncio.timeout(10):
        async with runner.run() as run:
            async for event in run:
                if isinstance(event, SubagentPaused) and not resolved:
                    [execution] = runner.pending_deferred_tool_executions()
                    resolved.append(execution)
                    tool.answers[execution.tool_call_id] = "at last"
                    run.notify(execution)

    # ── postcondition ────────────────────────────────────────────────────────
    # ONE run: the tree never came back blocked. The restarted child dispatched
    # the SAME call a second time — one execution, two attempts — and both
    # children resolved into the parent.
    assert [ex.conversation_id for ex in resolved] == [children_of(session)[0]]
    assert tool.dispatches == ["tcA", "tcA"]
    assert all(link.execution_result is not None for link in links(session))
    assert runner.idle()


async def test_the_next_run_restarts_a_deferral_parked_subagent(faux):
    # ── precondition ─────────────────────────────────────────────────────────
    # The whole tree at rest with one parked call in it: the child's drive ended
    # when it parked, and the parent's drive ended because nothing in its
    # subtree could advance. `_restart_unresolved_children` is the only thing
    # that can hand that child a drive again.
    faux.set_responses([spawns(1), ask_round("tcA")])
    tool = DeferringTool()
    session = subagent_session()
    runner = DeterministicRunner(
        session,
        tool_registry=SubagentRegistry([tool]),
        provider=faux,
        ids=list(IDS),
        now=1000,
    )
    runner.post_message("go")
    async with runner.run() as run:
        _ = [event async for event in run]
    assert runner.blocked()
    [parked] = runner.pending_deferred_tool_executions()

    # ── action ───────────────────────────────────────────────────────────────
    tool.answers[parked.tool_call_id] = "at last"
    faux.set_responses(texts("A done", "final"))
    async with runner.run() as run:
        events = [event async for event in run]

    # ── postcondition ────────────────────────────────────────────────────────
    # The restart is a start a consumer has to see, exactly as it is for an
    # answered gate.
    assert [e.conversation_id for e in events if isinstance(e, SubagentStarted)] == [parked.conversation_id]
    assert tool.dispatches == ["tcA", "tcA"]
    assert [link.execution_result.content[0].text for link in links(session)] == ["A done"]
    assert runner.idle()


# ── the worker pool ────────────────────────────────────────────────────────────


async def test_a_parked_subagent_releases_its_slot_to_a_queued_sibling(faux):
    # ── precondition ─────────────────────────────────────────────────────────
    # One worker. The first child takes it, parks on its deferral, and the
    # second child can only ever run if that park RELEASED the slot — which is
    # `_await_subtree`'s job, unchanged from the approval gate.
    faux.set_responses([spawns(2), ask_round("tcA"), *texts("B done", "still waiting on A")])
    session = subagent_session(subagents_max_workers=1)
    runner = DeterministicRunner(
        session,
        tool_registry=SubagentRegistry([DeferringTool()]),
        provider=faux,
        ids=list(IDS),
        now=1000,
    )
    runner.post_message("go")

    # ── action ───────────────────────────────────────────────────────────────
    async with asyncio.timeout(10):
        async with runner.run() as run:
            events = [event async for event in run]

    # ── postcondition ────────────────────────────────────────────────────────
    # The pause precedes the sibling's start — the freed slot is what admitted
    # it — and the tree ends blocked on the one call nobody answered.
    first_child, second_child = children_of(session)
    [paused] = [e for e in events if isinstance(e, SubagentPaused)]
    started = [e for e in events if isinstance(e, SubagentStarted)]
    assert [e.conversation_id for e in started] == [first_child, second_child]
    assert paused.conversation_id == first_child
    assert events.index(paused) < events.index(started[1])
    assert [(ex.conversation_id, ex.tool_call_id) for ex in runner.pending_deferred_tool_executions()] == [
        (first_child, "tcA")
    ]
    assert [link.execution_result is not None for link in links(session)] == [False, True]
    assert runner.blocked()
