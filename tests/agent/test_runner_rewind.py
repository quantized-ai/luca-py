"""`AgentSessionRunner.rewind_to`: archiving the main conversation and
installing a successor over a truncated prefix.

Same declarative shape as the rest of the runner suite — precondition → one
action → postcondition. The subject is the TRANSITION: which conversation is
named afterwards, what the successor's path is, that the predecessor and every
entry survive untouched, and the four guards that refuse a rewind before
anything is written.

`RICH_IDLE_SESSION` is the precondition for almost all of it: a closed,
multi-bracket conversation (`ts1..tf1`, `ts2..tf2`, `ts3..tf3`) with a
compaction summary and an archived `c0` already behind it, so a rewind has real
history on both sides to preserve.
"""

import pytest

from luca.agent.core.exceptions import AgentError
from luca.agent.core.models import (
    AssistantMessage,
    ChildConversation,
    Conversation,
    ConversationStatus,
    ExecutionAttempt,
    ExecutionAttemptOutcome,
    ExecutionResult,
    ExecutionStatus,
    SessionConfig,
    TextContent,
    ToolCall,
    ToolExecution,
    TurnFinish,
    TurnStart,
    UserMessage,
)
from luca.client.testing import FauxProvider, faux_assistant_message, faux_text
from tests.agent.scenarios import (
    GATED_SESSION,
    MODEL,
    RICH_IDLE_SESSION,
    RICH_SESSION,
    DeterministicRunner,
    conversation,
    main_conversation,
    make_session,
    spec,
)

# The full closed path `RICH_IDLE_SESSION` presents, for slicing in expectations.
RICH_IDLE_NODES = list(main_conversation(RICH_IDLE_SESSION).nodes)

# Everything up to and including the second bracket's `TurnFinish` — the prefix
# a rewind past the third turn leaves behind.
THROUGH_TF2 = RICH_IDLE_NODES[: RICH_IDLE_NODES.index("tf2") + 1]


# ── the transition ────────────────────────────────────────────────────────────


async def test_rewind_to_a_turn_boundary_installs_a_successor():
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(session, ids=["c2"], now=1000)

    installed = runner.rewind_to("tf2")

    assert installed == Conversation(
        id="c2",
        previous_conversation_id="c1",
        nodes=THROUGH_TF2,
        created_at=1000,
        updated_at=1000,
    )
    assert main_conversation(session) == installed
    assert session.main_conversation_id == "c2"


async def test_the_rewound_conversation_is_archived_with_its_whole_path():
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(session, ids=["c2"], now=1000)

    runner.rewind_to("tf2")

    assert session.conversations["c1"] == conversation(
        "c1",
        RICH_IDLE_NODES,
        created_at=500,
        updated_at=1000,
        previous_conversation_id="c0",
    )


async def test_no_entry_is_mutated_by_a_rewind():
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    before = {key: entry.model_copy(deep=True) for key, entry in session.entries.items()}
    runner = DeterministicRunner(session, ids=["c2"], now=1000)

    runner.rewind_to("tf2")

    assert session.entries == before


async def test_rewind_to_none_empties_the_conversation():
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(session, ids=["c2"], now=1000)

    installed = runner.rewind_to(None)

    assert installed == Conversation(
        id="c2",
        previous_conversation_id="c1",
        nodes=[],
        created_at=1000,
        updated_at=1000,
    )
    assert session.get_conversation_status("c2").status is ConversationStatus.IDLE


async def test_rewind_drops_a_trailing_queued_user_message():
    """`u4` sits outside every bracket, so the cut is legal — the boundary rule
    is derived from `open_turn_index`, not from "the next node is a
    TurnStart"."""
    session = RICH_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(session, ids=["c2"], now=1000)

    installed = runner.rewind_to("tf3")

    assert installed.nodes == RICH_IDLE_NODES
    assert session.get_conversation_status("c2").status is ConversationStatus.IDLE


async def test_rewind_to_the_current_leaf_is_a_noop():
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(session, ids=[], now=1000)

    installed = runner.rewind_to(RICH_IDLE_NODES[-1])

    assert installed is session.conversations["c1"]
    assert session.main_conversation_id == "c1"
    assert list(session.conversations) == ["c1", "c0"]


# ── what the transition preserves ─────────────────────────────────────────────


async def test_usage_records_survive_a_rewind():
    """Undoing a turn does not refund it: usages key on the conversation and
    the archived row keeps them, so a catalog-wide total is unchanged."""
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    before = {
        cid: {eid: usage.model_copy(deep=True) for eid, usage in rows.items()} for cid, rows in session.usages.items()
    }
    runner = DeterministicRunner(session, ids=["c2"], now=1000)

    runner.rewind_to("tf2")

    assert session.usages == before


async def test_an_earlier_archived_conversation_is_untouched():
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(session, ids=["c2"], now=1000)

    runner.rewind_to("tf2")

    assert session.conversations["c0"] == RICH_IDLE_SESSION.conversations["c0"]


def _session_with_a_subagent():
    """A closed turn that spawned and resolved one subagent, plus the child's
    own conversation row."""
    spawn = ToolExecution(
        id="te1",
        parent_id="ts",
        created_at=500,
        conversation_id="c1",
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="spawn_subagent", arguments={"task_id": "t1"}),
        tool_spec=spec(
            "spawn_subagent",
            output_schema={"type": "object", "properties": {"is_subagent_spawn": {"type": "boolean"}}},
        ),
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(
            content=[TextContent(text="Spawned t1")],
            structured_content={
                "is_subagent_spawn": True,
                "task_id": "t1",
                "prompt": "Research A",
                "description": "research A",
                "process_subagent_result_tool_name": "create_conversation_result",
            },
        ),
        attempts=[ExecutionAttempt(outcome=ExecutionAttemptOutcome.COMPLETED, started_at=500, ended_at=500)],
        finished_at=500,
    )
    return make_session(
        id="s_rewind_subagent",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Research A")]),
            "ts": TurnStart(id="ts", parent_id="u1", created_at=500),
            "te1": spawn,
            "cc1": ChildConversation(
                id="cc1",
                parent_id="te1",
                created_at=500,
                conversation_id="c2",
                tool_execution_id="te1",
                execution_result=ExecutionResult(content=[TextContent(text="A is fine.")]),
            ),
            "a1": AssistantMessage(
                id="a1",
                parent_id="cc1",
                created_at=600,
                parts=[TextContent(text="A is fine.")],
                llm_config=MODEL,
                stop_reason="stop",
            ),
            "tf": TurnFinish(id="tf", parent_id="a1", created_at=600),
            "cu1": UserMessage(id="cu1", created_at=500, parts=[TextContent(text="Research A")]),
        },
        conversations={
            "c1": conversation("c1", ["u1", "ts", "te1", "cc1", "a1", "tf"], created_at=500, updated_at=600),
            "c2": conversation("c2", ["cu1"], created_at=500, updated_at=500, depth=1),
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )


async def test_a_dropped_subagent_link_leaves_its_conversation_inert():
    """The child's row survives (nothing is ever deleted) but no link to it is
    on the current path, so nothing will drive it again."""
    session = _session_with_a_subagent()
    runner = DeterministicRunner(session, ids=["c3"], now=1000)

    installed = runner.rewind_to("u1")

    assert installed.nodes == ["u1"]
    assert session.conversations["c2"] == conversation("c2", ["cu1"], created_at=500, updated_at=500, depth=1)
    assert [node for node in installed.nodes if isinstance(session.entries[node], ChildConversation)] == []


# ── the guards ────────────────────────────────────────────────────────────────


async def test_rewind_into_a_turn_bracket_raises():
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(session, ids=["c2"], now=1000)

    with pytest.raises(AgentError, match="inside a turn bracket"):
        runner.rewind_to("a1")


async def test_a_refused_rewind_writes_nothing():
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(session, ids=["c2"], now=1000)

    with pytest.raises(AgentError):
        runner.rewind_to("a1")

    assert session.main_conversation_id == "c1"
    assert list(session.conversations) == ["c1", "c0"]
    assert main_conversation(session).nodes == RICH_IDLE_NODES


async def test_rewind_to_an_entry_off_the_path_raises():
    """`u0` is a real entry, but it lives on the archived `c0` only."""
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(session, ids=["c2"], now=1000)

    with pytest.raises(AgentError, match="not on conversation"):
        runner.rewind_to("u0")


async def test_rewind_to_an_unknown_entry_raises():
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(session, ids=["c2"], now=1000)

    with pytest.raises(AgentError, match="not on conversation"):
        runner.rewind_to("nope")


async def test_rewind_with_an_open_turn_raises():
    """`GATED_SESSION` is paused at an approval gate: the bracket is open, so
    the rewind is refused until the turn is cancelled and flushed."""
    session = GATED_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(session, ids=["c2"], now=1000)

    with pytest.raises(AgentError, match="requires a closed turn"):
        runner.rewind_to(None)


async def test_rewind_during_a_live_run_raises():
    faux = FauxProvider()
    faux.set_responses([faux_assistant_message([faux_text("Hello!")], finish_reason="stop")])
    session = make_session(
        id="s_rewind_live",
        entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")])},
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(session, provider=faux, ids=["ts", "a1", "tf"], now=1000)
    run = runner.start()  # eager: `_begin_run` takes the guard synchronously

    with pytest.raises(AgentError, match="while a run is active"):
        runner.rewind_to(None)

    await run  # join so the background task never outlives the test
