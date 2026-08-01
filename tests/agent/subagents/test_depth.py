"""Nesting past depth 1: a subagent that spawns its own subagents.

Every mechanism on the spawn path is depth-generic by construction — the gate
compares `depth < max_depth`, the event fan-in chains parent by parent, and
restarts recurse. These tests pin that down end to end at `max_depth=2`:
grandchild spawn, the gate at the leaf, cancellation, approvals and reload.
"""

import pytest

from luca.agent.contrib.subagents import SPAWN_TOOL_NAME
from luca.agent.core import AgentSession
from luca.agent.core.events import SubagentsSpawned
from luca.agent.core.exceptions import AgentError
from luca.agent.core.models import ApprovalDecision, ApprovalOption, ConversationStatus
from luca.client.testing import faux_assistant_message, faux_text, faux_tool_call
from tests.agent.scenarios import AddTool, DeterministicRunner
from tests.agent.subagents.conftest import SubagentRegistry, spawn_call, subagent_session

IDS = [f"x{n}" for n in range(80)]


def at_depth(session: AgentSession, depth: int) -> list[str]:
    return [cid for cid, c in session.conversations.items() if c.depth == depth]


def chain_script(faux):
    """main spawns a child, the child spawns a grandchild, answers bubble up.
    One conversation works at a time along the chain, so the FIFO faux queue
    is deterministic."""
    faux.set_responses(
        [
            faux_assistant_message([spawn_call("child work", "child", call_id="tc1")], finish_reason="tool_use"),
            faux_assistant_message([spawn_call("leaf work", "leaf", call_id="tc2")], finish_reason="tool_use"),
            faux_assistant_message([faux_text("leaf done")], finish_reason="stop"),
            faux_assistant_message([faux_text("child done")], finish_reason="stop"),
            faux_assistant_message([faux_text("all done")], finish_reason="stop"),
        ]
    )


async def test_a_grandchild_spawns_and_results_flow_leaf_to_root(faux):
    chain_script(faux)
    session = subagent_session(max_depth=2)
    runner = DeterministicRunner(session, tool_registry=SubagentRegistry(), provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")

    async with runner.run() as run:
        _ = [event async for event in run]

    assert sorted(c.depth for c in session.conversations.values()) == [0, 1, 2]
    # the leaf's answer reached the CHILD's last model call, and the child's
    # answer reached the MAIN conversation's — one hop per level
    child_final = faux.requests[-2].messages[-1].content[0].text
    main_final = faux.requests[-1].messages[-1].content[0].text
    assert child_final == "<task id=tc2>\nleaf done\n</task>"
    assert main_final == "<task id=tc1>\nchild done\n</task>"
    assert all(session.get_conversation_status(cid).status is ConversationStatus.IDLE for cid in session.conversations)
    assert runner.idle()


async def test_the_spawn_tool_is_withheld_at_the_leaf(faux):
    # `max_depth=2`: depth 1 may still spawn, depth 2 may not — the gate is a
    # comparison, not a "subagents cannot spawn" special case
    chain_script(faux)
    session = subagent_session(max_depth=2)
    runner = DeterministicRunner(session, tool_registry=SubagentRegistry(), provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")

    async with runner.run() as run:
        _ = [event async for event in run]

    child_request = next(r for r in faux.requests if r.messages[0].content[0].text == "child work")
    leaf_request = next(r for r in faux.requests if r.messages[0].content[0].text == "leaf work")
    assert SPAWN_TOOL_NAME in [tool.name for tool in child_request.tools]
    assert SPAWN_TOOL_NAME not in [tool.name for tool in (leaf_request.tools or [])]


async def test_inf_depth_means_unlimited_nesting(faux):
    # -1 is the config-wide "no limit" sentinel; without its guard the depth
    # clause would read `depth >= -1` and silently disable spawning everywhere
    from luca.agent.contrib.subagents import spawn_gate_open
    from luca.agent.core import Inf

    session = subagent_session(max_depth=Inf)
    session.conversations["deep"] = session.conversations["c1"].model_copy(update={"id": "deep", "depth": 7})
    runner = DeterministicRunner(session, tool_registry=SubagentRegistry(), provider=faux, ids=list(IDS), now=1000)

    assert spawn_gate_open(session, "deep") is True
    assert runner._spawn_gate_open("deep") is True


async def test_a_registry_that_leaks_the_spawn_tool_to_the_leaf_raises(faux):
    # same violation as depth 1, one level down: the runner's mirror of the
    # gate holds at every depth, not only at the old cap
    from luca.agent.contrib.simple_tool_registry import SimpleToolRegistry, YoloPermissionPolicy
    from luca.agent.contrib.subagents import CreateConversationResult, SpawnSubagent

    session = subagent_session(max_depth=2)
    session.conversations["c1"].depth = 2  # a leaf, wired to a gateless registry
    runner = DeterministicRunner(
        session,
        tool_registry=SimpleToolRegistry(
            tools=[SpawnSubagent(), CreateConversationResult()],
            permission_policy=YoloPermissionPolicy(),
        ),
        provider=faux,
        ids=list(IDS),
        now=1000,
    )
    runner.post_message("go")

    with pytest.raises(AgentError, match="offered spawn-declaring tool"):
        await runner.run()


async def test_cancel_cascades_through_three_levels(faux):
    faux.set_responses(
        [
            faux_assistant_message([spawn_call("child work", "child", call_id="tc1")], finish_reason="tool_use"),
            faux_assistant_message([spawn_call("leaf work", "leaf", call_id="tc2")], finish_reason="tool_use"),
        ]
    )
    session = subagent_session(max_depth=2)
    runner = DeterministicRunner(session, tool_registry=SubagentRegistry(), provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")

    run = runner.run(autostart_subagents=False)
    async with run:
        async for event in run:
            if isinstance(event, SubagentsSpawned):
                [child_id] = event.conversation_ids
                child = run.child(child_id)
                async with child:
                    async for child_event in child:
                        if isinstance(child_event, SubagentsSpawned):
                            run.cancel()  # main → child → grandchild

    assert runner.idle()
    for cid in session.conversations:
        assert session.get_conversation_status(cid).status is ConversationStatus.IDLE
    # the never-driven grandchild was seeded, cancelled and settled
    [leaf_id] = at_depth(session, 2)
    assert [session.entries[n].type for n in session.conversations[leaf_id].nodes] == [
        "user",
        "turn_start",
        "cancel_requested",
        "turn_finish",
    ]


def gated_leaf_registry(decisions: list[ApprovalDecision]):
    registry = SubagentRegistry([AddTool()])
    registry.other.decisions = decisions
    return registry


def gated_leaf_script(faux):
    """The grandchild's own tool call is what gates, two levels below main."""
    faux.set_responses(
        [
            faux_assistant_message([spawn_call("child work", "child", call_id="tc1")], finish_reason="tool_use"),
            faux_assistant_message([spawn_call("leaf work", "leaf", call_id="tc2")], finish_reason="tool_use"),
            faux_assistant_message([faux_tool_call("add", {"a": 1, "b": 2}, id="tc3")], finish_reason="tool_use"),
        ]
    )


async def test_an_approval_gate_on_a_grandchild_blocks_the_whole_chain(faux):
    gated_leaf_script(faux)
    session = subagent_session(max_depth=2)
    registry = gated_leaf_registry(
        [
            ApprovalDecision(decision=ApprovalOption.PENDING, created_at=1000),
            ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000),
        ]
    )
    runner = DeterministicRunner(session, tool_registry=registry, provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")

    async with runner.run() as run:
        _ = [event async for event in run]

    [leaf_id] = at_depth(session, 2)
    assert runner.blocked()
    assert [ex.conversation_id for ex in runner.pending_approvals()] == [leaf_id]

    faux.set_responses(
        [
            faux_assistant_message([faux_text("leaf done")], finish_reason="stop"),
            faux_assistant_message([faux_text("child done")], finish_reason="stop"),
            faux_assistant_message([faux_text("all done")], finish_reason="stop"),
        ]
    )
    async with runner.run() as run:  # the registry now allows — nothing else changed
        _ = [event async for event in run]

    assert runner.idle()


async def test_reload_mid_tree_with_an_unresolved_grandchild_resumes(faux):
    gated_leaf_script(faux)
    session = subagent_session(max_depth=2)
    registry = gated_leaf_registry([ApprovalDecision(decision=ApprovalOption.PENDING, created_at=1000)])
    runner = DeterministicRunner(session, tool_registry=registry, provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")

    async with runner.run() as run:
        _ = [event async for event in run]
    payload = runner.session.model_dump_json()  # the process exits here

    reloaded = AgentSession.model_validate_json(payload)
    faux.set_responses(
        [
            faux_assistant_message([faux_text("leaf done")], finish_reason="stop"),
            faux_assistant_message([faux_text("child done")], finish_reason="stop"),
            faux_assistant_message([faux_text("all done")], finish_reason="stop"),
        ]
    )
    resumed = DeterministicRunner(
        reloaded,
        tool_registry=gated_leaf_registry([ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000)]),
        provider=faux,
        ids=[f"y{n}" for n in range(40)],
        now=1000,
    )
    assert resumed.blocked()

    async with resumed.run() as run:
        _ = [event async for event in run]

    assert resumed.idle()
    assert all(
        reloaded.get_conversation_status(cid).status is ConversationStatus.IDLE for cid in reloaded.conversations
    )
