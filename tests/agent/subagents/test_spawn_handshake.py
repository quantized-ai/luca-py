"""The spawn handshake: how a tool call becomes a second conversation.

The §4.1 timeline, one stage per test, plus the three violations the runner
refuses loudly. Everything here drives the real `contrib.subagents` tools
through the real runner — the handshake is a contract between the two, so
testing either side alone would prove nothing about it.
"""

import pytest

from luca.agent.contrib.subagents import SPAWN_TOOL_NAME
from luca.agent.core import AgentSession, ChildConversation
from luca.agent.core.events import SubagentsSpawned
from luca.agent.core.exceptions import AgentError
from luca.agent.core.models import ConversationStatus, ExecutionStatus, ToolExecution
from luca.client.testing import faux_assistant_message, faux_text
from luca.client.types import TextBlock as LucaTextBlock
from tests.agent.scenarios import DeterministicRunner
from tests.agent.subagents.conftest import SubagentRegistry, spawn_call, subagent_session

IDS = [f"x{n}" for n in range(60)]


def scripted(faux, *responses):
    faux.set_responses(list(responses))
    return faux


# ── the timeline ──────────────────────────────────────────────────────────────


async def test_one_spawn_creates_one_child_conversation(faux):
    scripted(
        faux,
        faux_assistant_message([spawn_call("Research A", "research A", call_id="tc1")], finish_reason="tool_use"),
        faux_assistant_message([faux_text("A is fine.")], finish_reason="stop"),
        faux_assistant_message([faux_text("A is fine, then.")], finish_reason="stop"),
    )
    session = subagent_session()
    runner = DeterministicRunner(session, tool_registry=SubagentRegistry(), provider=faux, ids=list(IDS), now=1000)
    runner.post_message("Research A")

    async with runner.run() as run:
        _ = [event async for event in run]

    [child_id] = [cid for cid, c in session.conversations.items() if c.depth == 1]
    # the parent's path: the spawn execution, the link, the private result
    # execution, then the answer
    assert [session.entries[n].type for n in session.conversations["c1"].nodes] == [
        "user",
        "turn_start",
        "assistant",
        "tool_execution",
        "child_conversation",
        "tool_execution",
        "assistant",
        "turn_finish",
    ]
    # the child's own conversation: seeded with the prompt, then an ordinary turn
    assert [session.entries[n].type for n in session.conversations[child_id].nodes] == [
        "user",
        "turn_start",
        "assistant",
        "turn_finish",
    ]
    assert session.get_conversation_status("c1").status is ConversationStatus.IDLE
    assert session.get_conversation_status(child_id).status is ConversationStatus.IDLE


async def test_the_child_is_seeded_with_the_prompt_and_nothing_else(faux):
    scripted(
        faux,
        faux_assistant_message([spawn_call("Read main.py", "read it", call_id="tc1")], finish_reason="tool_use"),
        faux_assistant_message([faux_text("read it")], finish_reason="stop"),
        faux_assistant_message([faux_text("ok")], finish_reason="stop"),
    )
    session = subagent_session()
    runner = DeterministicRunner(session, tool_registry=SubagentRegistry(), provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")

    async with runner.run() as run:
        _ = [event async for event in run]

    [child_id] = [cid for cid, c in session.conversations.items() if c.depth == 1]
    seed = session.entries[session.conversations[child_id].nodes[0]]
    assert seed.parts[0].text == "Read main.py"
    # the child sees ONLY its prompt — the parent's history is not carried over
    child_request = next(r for r in faux.requests if r.messages[0].content[0].text == "Read main.py")
    assert len(child_request.messages) == 1


async def test_the_spawn_call_completes_immediately(faux):
    # its scope is to SPAWN, and it did. Keeping it open until the child
    # finished would make background subagents unrepresentable later, since a
    # non-terminal execution blocks the parent's projection by design.
    scripted(
        faux,
        faux_assistant_message([spawn_call("A", "a", call_id="tc1")], finish_reason="tool_use"),
        faux_assistant_message([faux_text("A")], finish_reason="stop"),
        faux_assistant_message([faux_text("ok")], finish_reason="stop"),
    )
    session = subagent_session()
    runner = DeterministicRunner(session, tool_registry=SubagentRegistry(), provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")

    async with runner.run() as run:
        _ = [event async for event in run]

    spawn = _spawn_execution(session)
    assert spawn.status is ExecutionStatus.COMPLETED
    assert spawn.result.content[0].text == "Spawned subagent: a"
    # the handshake rides on structured_content, which never reaches the model
    assert spawn.result.structured_content["is_subagent_spawn"] is True
    assert spawn.result.structured_content["process_subagent_result_tool_name"] == "create_conversation_result"


async def test_two_spawns_in_one_response_create_two_children(faux):
    scripted(
        faux,
        faux_assistant_message(
            [
                spawn_call("Research A", "research A", call_id="tc1", task_id="t1"),
                spawn_call("Research B", "research B", call_id="tc2", task_id="t2"),
            ],
            finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("A is fine.")], finish_reason="stop"),
        faux_assistant_message([faux_text("B is broken.")], finish_reason="stop"),
        faux_assistant_message([faux_text("A is fine, B is broken.")], finish_reason="stop"),
    )
    session = subagent_session()
    runner = DeterministicRunner(session, tool_registry=SubagentRegistry(), provider=faux, ids=list(IDS), now=1000)
    runner.post_message("Research A and B in parallel")

    async with runner.run() as run:
        events = [event async for event in run]

    children = [cid for cid, c in session.conversations.items() if c.depth == 1]
    assert len(children) == 2
    # ONE batch signal, not one per child — otherwise an application driving
    # them itself would be forced to drive them sequentially
    spawned = [e for e in events if isinstance(e, SubagentsSpawned)]
    assert len(spawned) == 1
    assert sorted(spawned[0].conversation_ids) == sorted(children)
    # both links land after both spawn executions, in call order
    links = [e for e in _entries(session, "c1") if isinstance(e, ChildConversation)]
    assert [link.conversation_id for link in links] == spawned[0].conversation_ids


async def test_both_child_results_reach_the_parents_wire_history(faux):
    scripted(
        faux,
        faux_assistant_message(
            [
                spawn_call("Research A", "research A", call_id="tc1", task_id="t1"),
                spawn_call("Research B", "research B", call_id="tc2", task_id="t2"),
            ],
            finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("A is fine.")], finish_reason="stop"),
        faux_assistant_message([faux_text("B is broken.")], finish_reason="stop"),
        faux_assistant_message([faux_text("done")], finish_reason="stop"),
    )
    session = subagent_session()
    runner = DeterministicRunner(session, tool_registry=SubagentRegistry(), provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")

    async with runner.run() as run:
        _ = [event async for event in run]

    final = faux.requests[-1].messages
    tagged = [
        block.text
        for message in final
        for block in (message.content if isinstance(message.content, list) else [])
        if isinstance(block, LucaTextBlock) and block.text.startswith("<task")
    ]
    assert tagged == ["<task id=tc1>\nA is fine.\n</task>", "<task id=tc2>\nB is broken.\n</task>"]


async def test_the_private_result_tool_never_reaches_the_wire(faux):
    scripted(
        faux,
        faux_assistant_message([spawn_call("A", "a", call_id="tc1")], finish_reason="tool_use"),
        faux_assistant_message([faux_text("A")], finish_reason="stop"),
        faux_assistant_message([faux_text("ok")], finish_reason="stop"),
    )
    session = subagent_session()
    runner = DeterministicRunner(session, tool_registry=SubagentRegistry(), provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")

    async with runner.run() as run:
        _ = [event async for event in run]

    # never advertised
    for request in faux.requests:
        assert "create_conversation_result" not in [tool.name for tool in (request.tools or [])]
    # and its execution projects nothing: the parent's final request carries a
    # ToolMessage for the SPAWN call and none for the result tool
    final = faux.requests[-1].messages
    tool_call_ids = [m.tool_call_id for m in final if hasattr(m, "tool_call_id")]
    assert tool_call_ids == ["tc1"]


# ── the three violations ──────────────────────────────────────────────────────


async def test_a_registry_that_leaks_the_spawn_tool_past_the_cap_raises(faux):
    # VIOLATION 1: caught at the tool list, BEFORE the model call, naming the
    # spec. The alternative failure is invisible — the spec is offered, the
    # model calls it, and a grandchild appears with nothing in the log.
    from luca.agent.contrib.simple_tool_registry import SimpleToolRegistry, YoloPermissionPolicy
    from luca.agent.contrib.subagents import CreateConversationResult, SpawnSubagent

    session = subagent_session()
    # a conversation at the cap, wired to a registry with NO gate at all
    session.conversations["c1"].depth = 1
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


async def test_a_spawn_payload_from_a_spec_that_never_declared_one_raises(faux):
    # VIOLATION 2: this payload went around the gate entirely — the child would
    # be spawned at any depth and the cap would silently not exist
    from tests.agent.subagents.conftest import UndeclaredSpawnTool

    scripted(faux, faux_assistant_message([_undeclared_call()], finish_reason="tool_use"))
    session = subagent_session()
    runner = DeterministicRunner(
        session,
        tool_registry=SubagentRegistry([UndeclaredSpawnTool()]),
        provider=faux,
        ids=list(IDS),
        now=1000,
    )
    runner.post_message("go")

    with pytest.raises(AgentError, match="never declared one"):
        await runner.run()


async def test_a_spawn_from_a_conversation_at_the_cap_raises(faux):
    # VIOLATION 3: the gate lives in `get_tools`, but a registry resolves by
    # NAME at dispatch, so anything reaching dispatch with that name runs. This
    # check puts the cap where it means something: child creation.
    scripted(
        faux,
        faux_assistant_message([spawn_call("A", "a", call_id="tc1")], finish_reason="tool_use"),
    )
    session = subagent_session(max_depth=0)
    runner = DeterministicRunner(session, tool_registry=SubagentRegistry(), provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")

    with pytest.raises(AgentError, match="may not spawn subagents"):
        await runner.run()


async def test_a_spawn_payload_missing_a_required_field_raises(faux):
    from tests.agent.subagents.conftest import IncompleteSpawnTool

    scripted(
        faux,
        faux_assistant_message([_incomplete_call()], finish_reason="tool_use"),
    )
    session = subagent_session()
    runner = DeterministicRunner(
        session,
        tool_registry=SubagentRegistry([IncompleteSpawnTool()]),
        provider=faux,
        ids=list(IDS),
        now=1000,
    )
    runner.post_message("go")

    with pytest.raises(AgentError, match="missing required field"):
        await runner.run()


# ── the gate withholds rather than failing ───────────────────────────────────


async def test_a_subagent_is_not_offered_the_spawn_tool(faux):
    scripted(
        faux,
        faux_assistant_message([spawn_call("A", "a", call_id="tc1")], finish_reason="tool_use"),
        faux_assistant_message([faux_text("A")], finish_reason="stop"),
        faux_assistant_message([faux_text("ok")], finish_reason="stop"),
    )
    session = subagent_session()
    runner = DeterministicRunner(session, tool_registry=SubagentRegistry(), provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")

    async with runner.run() as run:
        _ = [event async for event in run]

    child_request = next(r for r in faux.requests if r.messages[0].content[0].text == "A")
    assert SPAWN_TOOL_NAME not in [tool.name for tool in (child_request.tools or [])]
    # the parent WAS offered it
    assert SPAWN_TOOL_NAME in [tool.name for tool in faux.requests[0].tools]


async def test_subagents_disabled_withholds_the_tool_entirely(faux):
    scripted(faux, faux_assistant_message([faux_text("no tools for me")], finish_reason="stop"))
    session = subagent_session(enabled=False)
    runner = DeterministicRunner(session, tool_registry=SubagentRegistry(), provider=faux, ids=list(IDS), now=1000)
    runner.post_message("go")

    await runner.run()

    # a toolless request carries no tool list at all
    assert faux.requests[0].tools is None


# ── helpers ───────────────────────────────────────────────────────────────────


def _entries(session: AgentSession, conversation_id: str):
    return [session.entries[n] for n in session.conversations[conversation_id].nodes]


def _spawn_execution(session: AgentSession) -> ToolExecution:
    return next(
        e for e in _entries(session, "c1") if isinstance(e, ToolExecution) and e.raw_tool_call.name == SPAWN_TOOL_NAME
    )


def _undeclared_call():
    from luca.client.testing import faux_tool_call

    return faux_tool_call("undeclared_spawn", {}, id="tc1")


def _incomplete_call():
    from luca.client.testing import faux_tool_call

    return faux_tool_call("incomplete_spawn", {}, id="tc1")
