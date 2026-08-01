"""`subagents_max_per_turn` — the two-layer spawn budget.

Layer 1 withholds the spawn tool (and, in contrib, the prompt part) once the
open turn's budget is spent. Layer 2 catches the one case layer 1 structurally
cannot: several spawn calls in ONE assistant message, after the tool list was
already sent — the overflow call is born REFUSED and its body never runs.
`spawns_committed` is the number both layers count, and it is a pure read over
durable entries, so a reload mid-turn changes nothing.
"""

from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from luca.agent.contrib.subagents import SPAWN_TOOL_NAME, spawn_gate_open, spawning_prompt_part
from luca.agent.core import (
    AgentSession,
    ExecutionResult,
    TextContent,
    ToolSpec,
    declares_spawn,
    spawns_committed,
)
from luca.agent.core.models import ApprovalDecision, ApprovalOption, ExecutionStatus, ToolExecution
from luca.client.testing import faux_assistant_message, faux_text, faux_tool_call
from tests.agent.scenarios import AddTool, DeterministicRunner, FakeTool
from tests.agent.subagents.conftest import SubagentRegistry, spawn_call, subagent_session

IDS = [f"x{n}" for n in range(80)]


class _NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DecliningSpawnTool(FakeTool):
    """Declares the spawn schema, declines at runtime — a refusal is not a
    subagent, so it must not consume the budget."""

    name = "maybe_spawn"
    description = "Declares spawn, declines at runtime."
    Args = _NoArgs
    output_schema: ClassVar[dict] = {"type": "object", "properties": {"is_subagent_spawn": {"type": "boolean"}}}

    async def execute(self, args, session, conversation_id, *, cancellation_token):
        return ExecutionResult(
            content=[TextContent(text="declined")],
            structured_content={"is_subagent_spawn": False},
        )


class FailingSpawnTool(FakeTool):
    """Declares the spawn schema, raises — settled without a payload, so the
    reservation is released and the budget is untouched."""

    name = "failing_spawn"
    description = "Declares spawn, raises."
    Args = _NoArgs
    output_schema: ClassVar[dict] = {"type": "object", "properties": {"is_subagent_spawn": {"type": "boolean"}}}

    async def execute(self, args, session, conversation_id, *, cancellation_token):
        raise RuntimeError("boom")


class BudgetRegistry(SubagentRegistry):
    """`SubagentRegistry` whose gate covers BOTH halves — required whenever a
    spawn-declaring test double lives in `other`, or the budget closing the
    gate would make that half leak and trip `_verify_gate`."""

    async def get_tools(self, session, conversation_id) -> list[ToolSpec]:
        specs = [
            *await self.subagent_tools.get_tools(session, conversation_id),
            *await self.other.get_tools(session, conversation_id),
        ]
        if spawn_gate_open(session, conversation_id):
            return specs
        return [spec for spec in specs if not declares_spawn(spec)]


def executions(session: AgentSession, conversation_id: str = "c1") -> list[ToolExecution]:
    return [
        entry
        for node in session.conversations[conversation_id].nodes
        if isinstance(entry := session.entries[node], ToolExecution)
    ]


def children_of(session: AgentSession) -> list[str]:
    return [cid for cid, c in session.conversations.items() if c.depth == 1]


# ── S3/S4: the case layer 1 cannot catch ─────────────────────────────────────


async def test_the_overflow_spawn_in_one_message_is_born_refused(faux):
    # limit 1, one message with two spawn calls: the tool list was built when
    # the budget had room, so both calls arrive — the first spawns, the second
    # is refused before its body runs
    faux.set_responses(
        [
            faux_assistant_message(
                [
                    spawn_call("Research A", "research A", call_id="tc1", task_id="t1"),
                    spawn_call("Research B", "research B", call_id="tc2", task_id="t2"),
                ],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("A done")], finish_reason="stop"),
            faux_assistant_message([faux_text("done")], finish_reason="stop"),
        ]
    )
    session = subagent_session(subagents_max_per_turn=1)
    runner = DeterministicRunner(session, tool_registry=SubagentRegistry(), provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")

    async with runner.run() as run:
        _ = [event async for event in run]

    assert len(children_of(session)) == 1
    refused = next(e for e in executions(session) if e.tool_call_id == "tc2")
    assert refused.status is ExecutionStatus.REFUSED
    assert refused.error.error_type == "SpawnLimitReached"
    assert refused.error.details == {"limit": 1, "committed": 1}
    assert refused.approval_status is None  # born terminal — no decision ran
    assert refused.started_at is None  # the body never dispatched
    # the model was told twice: the refusal text as tc2's tool result…
    final = faux.requests[-1].messages
    refusal_text = next(m.content[0].text for m in final if getattr(m, "tool_call_id", None) == "tc2")
    assert refusal_text == (
        "Spawn limit reached (1/1 subagents this turn). Complete the remaining work yourself; do not retry."
    )
    # …and by the capability disappearing from the next step's tool list
    assert SPAWN_TOOL_NAME not in [tool.name for tool in (faux.requests[-1].tools or [])]
    assert runner.idle()


async def test_the_boundary_spawn_is_allowed_and_the_next_step_is_withheld(faux):
    # limit 2, exactly two spawns in one message: both fit, nothing is refused,
    # and the capability disappears only from the step AFTER
    faux.set_responses(
        [
            faux_assistant_message(
                [
                    spawn_call("Research A", "research A", call_id="tc1", task_id="t1"),
                    spawn_call("Research B", "research B", call_id="tc2", task_id="t2"),
                ],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("A done")], finish_reason="stop"),
            faux_assistant_message([faux_text("B done")], finish_reason="stop"),
            faux_assistant_message([faux_text("done")], finish_reason="stop"),
        ]
    )
    session = subagent_session(subagents_max_per_turn=2)
    runner = DeterministicRunner(session, tool_registry=SubagentRegistry(), provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")

    async with runner.run() as run:
        _ = [event async for event in run]

    assert len(children_of(session)) == 2
    assert not any(e.status is ExecutionStatus.REFUSED for e in executions(session))
    assert SPAWN_TOOL_NAME in [tool.name for tool in faux.requests[0].tools]
    assert SPAWN_TOOL_NAME not in [tool.name for tool in (faux.requests[-1].tools or [])]
    assert runner.idle()


# ── S10: the self-counting trap ──────────────────────────────────────────────


async def test_the_last_legitimate_spawn_converts_without_raising(faux):
    # the Nth spawn is settled and already counted when `_spawn_one` re-checks
    # the gate — without `exclude` it would read as the (N+1)th and the runner
    # would kill the run on a spawn both layers correctly allowed
    faux.set_responses(
        [
            faux_assistant_message([spawn_call("A", "a", call_id="tc1")], finish_reason="tool_use"),
            faux_assistant_message([faux_text("A done")], finish_reason="stop"),
            faux_assistant_message([faux_text("done")], finish_reason="stop"),
        ]
    )
    session = subagent_session(subagents_max_per_turn=1)
    runner = DeterministicRunner(session, tool_registry=SubagentRegistry(), provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")

    async with runner.run() as run:
        _ = [event async for event in run]

    assert len(children_of(session)) == 1
    assert runner.idle()


# ── S5/S6/S7: what does NOT consume the budget ───────────────────────────────


async def test_a_spawn_tool_that_declines_does_not_consume_the_budget(faux):
    faux.set_responses(
        [
            faux_assistant_message([faux_tool_call("maybe_spawn", {}, id="tcM")], finish_reason="tool_use"),
            faux_assistant_message([spawn_call("A", "a", call_id="tc1")], finish_reason="tool_use"),
            faux_assistant_message([faux_text("A done")], finish_reason="stop"),
            faux_assistant_message([faux_text("done")], finish_reason="stop"),
        ]
    )
    session = subagent_session(subagents_max_per_turn=1)
    runner = DeterministicRunner(
        session,
        tool_registry=BudgetRegistry([DecliningSpawnTool()]),
        provider=faux,
        ids=list(IDS),
        now=1000,
    )
    runner.post_message("go")

    async with runner.run() as run:
        _ = [event async for event in run]

    # the declined call settled COMPLETED but spawned nothing; the real spawn
    # still fit in the budget
    declined = next(e for e in executions(session) if e.tool_call_id == "tcM")
    assert declined.status is ExecutionStatus.COMPLETED
    assert declined.result.structured_content == {"is_subagent_spawn": False}
    assert len(children_of(session)) == 1
    assert runner.idle()


async def test_a_denied_spawn_call_does_not_consume_the_budget(faux):
    faux.set_responses(
        [
            faux_assistant_message([faux_tool_call("maybe_spawn", {}, id="tcM")], finish_reason="tool_use"),
            faux_assistant_message([spawn_call("A", "a", call_id="tc1")], finish_reason="tool_use"),
            faux_assistant_message([faux_text("A done")], finish_reason="stop"),
            faux_assistant_message([faux_text("done")], finish_reason="stop"),
        ]
    )
    session = subagent_session(subagents_max_per_turn=1)
    registry = BudgetRegistry([DecliningSpawnTool()])
    registry.other.decisions = [ApprovalDecision(decision=ApprovalOption.DENY, created_at=1000)]
    runner = DeterministicRunner(session, tool_registry=registry, provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")

    async with runner.run() as run:
        _ = [event async for event in run]

    denied = next(e for e in executions(session) if e.tool_call_id == "tcM")
    assert denied.status is ExecutionStatus.REJECTED
    assert len(children_of(session)) == 1
    assert runner.idle()


async def test_a_failed_spawn_call_does_not_consume_the_budget(faux):
    faux.set_responses(
        [
            faux_assistant_message([faux_tool_call("failing_spawn", {}, id="tcF")], finish_reason="tool_use"),
            faux_assistant_message([spawn_call("A", "a", call_id="tc1")], finish_reason="tool_use"),
            faux_assistant_message([faux_text("A done")], finish_reason="stop"),
            faux_assistant_message([faux_text("done")], finish_reason="stop"),
        ]
    )
    session = subagent_session(subagents_max_per_turn=1)
    runner = DeterministicRunner(
        session,
        tool_registry=BudgetRegistry([FailingSpawnTool()]),
        provider=faux,
        ids=list(IDS),
        now=1000,
    )
    runner.post_message("go")

    async with runner.run() as run:
        _ = [event async for event in run]

    failed = next(e for e in executions(session) if e.tool_call_id == "tcF")
    assert failed.status is ExecutionStatus.FAILED
    assert len(children_of(session)) == 1
    assert runner.idle()


# ── S2/S8: durable mid-turn state — both layers read the same number ─────────


def gated_child_registry(decisions: list[ApprovalDecision]):
    registry = SubagentRegistry([AddTool()])
    registry.other.decisions = decisions
    return registry


async def test_a_spent_budget_withholds_the_tool_and_the_prompt_mid_turn(faux):
    # park the tree with the budget spent (the child gated on an approval):
    # the open turn durably holds one committed spawn, and both the gate and
    # the prompt part read it
    faux.set_responses(
        [
            faux_assistant_message([spawn_call("A", "a", call_id="tc1")], finish_reason="tool_use"),
            faux_assistant_message([faux_tool_call("add", {"a": 1, "b": 2}, id="tcA")], finish_reason="tool_use"),
        ]
    )
    session = subagent_session(subagents_max_per_turn=1)
    registry = gated_child_registry([ApprovalDecision(decision=ApprovalOption.PENDING, created_at=1000)])
    runner = DeterministicRunner(session, tool_registry=registry, provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")

    async with runner.run() as run:
        _ = [event async for event in run]

    assert runner.blocked()
    assert spawns_committed(session, "c1") == 1
    assert spawn_gate_open(session, "c1") is False
    assert spawning_prompt_part(session, "c1") is None


async def test_a_reload_mid_turn_rebuilds_the_count_from_entries(faux):
    faux.set_responses(
        [
            faux_assistant_message([spawn_call("A", "a", call_id="tc1")], finish_reason="tool_use"),
            faux_assistant_message([faux_tool_call("add", {"a": 1, "b": 2}, id="tcA")], finish_reason="tool_use"),
        ]
    )
    session = subagent_session(subagents_max_per_turn=1)
    registry = gated_child_registry([ApprovalDecision(decision=ApprovalOption.PENDING, created_at=1000)])
    runner = DeterministicRunner(session, tool_registry=registry, provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")

    async with runner.run() as run:
        _ = [event async for event in run]
    payload = runner.session.model_dump_json()  # the process exits here

    reloaded = AgentSession.model_validate_json(payload)
    assert spawns_committed(reloaded, "c1") == 1
    assert spawn_gate_open(reloaded, "c1") is False

    faux.set_responses(
        [
            faux_assistant_message([faux_text("A done")], finish_reason="stop"),
            faux_assistant_message([faux_text("done")], finish_reason="stop"),
        ]
    )
    resumed = DeterministicRunner(
        reloaded,
        tool_registry=gated_child_registry([ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000)]),
        provider=faux,
        ids=[f"y{n}" for n in range(40)],
        now=1000,
    )
    async with resumed.run() as run:
        _ = [event async for event in run]

    assert resumed.idle()


# ── S9: the budget is per turn, not per session ──────────────────────────────


async def test_the_count_resets_when_the_turn_closes(faux):
    faux.set_responses(
        [
            faux_assistant_message([spawn_call("A", "a", call_id="tc1")], finish_reason="tool_use"),
            faux_assistant_message([faux_text("A done")], finish_reason="stop"),
            faux_assistant_message([faux_text("done")], finish_reason="stop"),
            faux_assistant_message([spawn_call("B", "b", call_id="tc2")], finish_reason="tool_use"),
            faux_assistant_message([faux_text("B done")], finish_reason="stop"),
            faux_assistant_message([faux_text("done again")], finish_reason="stop"),
        ]
    )
    session = subagent_session(subagents_max_per_turn=1)
    runner = DeterministicRunner(session, tool_registry=SubagentRegistry(), provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")
    async with runner.run() as run:
        _ = [event async for event in run]

    runner.post_message("again")
    async with runner.run() as run:
        _ = [event async for event in run]

    assert len(children_of(session)) == 2
    # the second turn's first request offered the spawn tool again
    second_turn_first = faux.requests[-3]
    assert SPAWN_TOOL_NAME in [tool.name for tool in second_turn_first.tools]
    assert runner.idle()


# ── S11: every conversation gets its own budget ──────────────────────────────


async def test_a_subagent_has_its_own_budget(faux):
    faux.set_responses(
        [
            faux_assistant_message([spawn_call("child work", "child", call_id="tc1")], finish_reason="tool_use"),
            faux_assistant_message([spawn_call("leaf work", "leaf", call_id="tc2")], finish_reason="tool_use"),
            faux_assistant_message([faux_text("leaf done")], finish_reason="stop"),
            faux_assistant_message([faux_text("child done")], finish_reason="stop"),
            faux_assistant_message([faux_text("all done")], finish_reason="stop"),
        ]
    )
    session = subagent_session(max_depth=2, subagents_max_per_turn=1)
    runner = DeterministicRunner(session, tool_registry=SubagentRegistry(), provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")

    async with runner.run() as run:
        _ = [event async for event in run]

    # main spent its budget on the child; the child still spawned the leaf on
    # its OWN budget — the caps never aggregate across the tree
    assert sorted(c.depth for c in session.conversations.values()) == [0, 1, 2]
    assert runner.idle()


# ── S12: the default changes nothing ─────────────────────────────────────────


async def test_the_default_budget_never_refuses_and_never_withholds(faux):
    faux.set_responses(
        [
            faux_assistant_message(
                [
                    spawn_call("Research A", "research A", call_id="tc1", task_id="t1"),
                    spawn_call("Research B", "research B", call_id="tc2", task_id="t2"),
                ],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("A done")], finish_reason="stop"),
            faux_assistant_message([faux_text("B done")], finish_reason="stop"),
            faux_assistant_message([faux_text("done")], finish_reason="stop"),
        ]
    )
    session = subagent_session()  # subagents_max_per_turn stays Inf
    runner = DeterministicRunner(session, tool_registry=SubagentRegistry(), provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")

    async with runner.run() as run:
        _ = [event async for event in run]

    assert len(children_of(session)) == 2
    assert not any(e.status is ExecutionStatus.REFUSED for e in executions(session))
    # the tool is still offered on the final step — nothing was withheld
    assert SPAWN_TOOL_NAME in [tool.name for tool in faux.requests[-1].tools]
    assert runner.idle()
