"""`ChildConversation` as a DATA type: projection, context accounting,
`pretty_print`, mutability and serialization.

Nothing here drives a subagent — no runner, no spawn handshake. This is the
entry on its own: how its result reaches its PARENT's wire history (a task
update rendered at the RESULT EXECUTION's path position, so the projected
history stays append-only while the link mutates in place), what it costs the
parent's window, and how it renders. The runtime that produces one is covered
under `tests/agent/subagents/`.
"""

import pytest

from luca.agent.core.context_manager import ContextManager
from luca.agent.core.exceptions import ProjectionError
from luca.agent.core.models import (
    AgentSession,
    AssistantMessage,
    ChildConversation,
    ExecutionAttempt,
    ExecutionAttemptOutcome,
    ExecutionResult,
    ExecutionStatus,
    ImageContent,
    MediaBase64,
    SessionConfig,
    TextContent,
    ToolCall,
    ToolExecution,
    TurnFinish,
    TurnStart,
    UserMessage,
)
from luca.agent.core.projection import ConversationProjector
from luca.agent.core.utils import pretty_print
from luca.client.types import TextBlock as LucaTextBlock, UserMessage as LucaUserMessage
from tests.agent.scenarios import MODEL, conversation, make_session, spec

PROJECTOR = ConversationProjector()
CM = ContextManager()

SPAWN_SPEC = spec(
    "spawn_subagent",
    output_schema={"type": "object", "properties": {"is_subagent_spawn": {"type": "boolean"}}},
)
RESULT_SPEC = spec("create_conversation_result", is_private=True)

# The spawn execution the child hangs off. Its PAYLOAD task id is what the
# projection tags the result with — the same identifier the spawn
# confirmation, `list_subagents` and `stop_subagent` speak, so the model can
# correlate all four.
SPAWN = ToolExecution(
    id="te1",
    created_at=500,
    conversation_id="c1",
    tool_call_id="tc1",
    raw_tool_call=ToolCall(
        id="tc1",
        name="spawn_subagent",
        arguments={"prompt": "Research A", "description": "research A", "task_id": "t1"},
    ),
    tool_spec=SPAWN_SPEC,
    status=ExecutionStatus.COMPLETED,
    result=ExecutionResult(
        content=[TextContent(text="Spawned subagent with id t1: research A")],
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

# The runtime-minted result execution whose path position carries the update.
RESULT_EXEC = ToolExecution(
    id="ter1",
    created_at=700,
    conversation_id="c1",
    tool_call_id="rtc1",
    raw_tool_call=ToolCall(id="rtc1", name="create_conversation_result"),
    tool_spec=RESULT_SPEC,
    status=ExecutionStatus.COMPLETED,
    result=ExecutionResult(content=[TextContent(text="A is fine.")]),
    attempts=[ExecutionAttempt(outcome=ExecutionAttemptOutcome.COMPLETED, started_at=700, ended_at=700)],
    finished_at=700,
)

TS = TurnStart(id="ts", created_at=500)
TF = TurnFinish(id="tf", created_at=800)


def child(**over) -> ChildConversation:
    fields: dict = {
        "id": "cc1",
        "parent_id": "te1",
        "created_at": 500,
        "conversation_id": "c2",
        "tool_execution_id": "te1",
    }
    fields.update(over)
    return ChildConversation(**fields)


RESOLVED = child(
    execution_result=ExecutionResult(content=[TextContent(text="A is fine.")]),
    result_execution_id="ter1",
)

ENTRIES = {"ts": TS, "te1": SPAWN, "cc1": RESOLVED, "ter1": RESULT_EXEC, "tf": TF}


# ── projection ────────────────────────────────────────────────────────────────


def test_the_update_renders_at_the_result_executions_position():
    # the spawn call projects its own status line as a ToolMessage where it
    # sits; the child's ANSWER renders where its RESOLUTION happened — at the
    # private result execution's position, after everything that predates it.
    # The link itself projects nothing: rendering at ITS position would
    # rewrite mid-history on every resolution.
    messages = PROJECTOR.project(["ts", "te1", "cc1", "ter1"], ENTRIES)

    assert [type(m).__name__ for m in messages] == ["ToolMessage", "UserMessage"]
    assert messages[0].content[0].text == "Spawned subagent with id t1: research A"
    assert messages[1] == LucaUserMessage(
        content=[
            LucaTextBlock(
                text=(
                    "Subagent task update:\n"
                    '<task id=t1 status=completed completed_at="1970-01-01T00:00:00Z">\nA is fine.\n</task>'
                ),
            ),
        ],
    )


def test_a_failed_result_renders_status_failed():
    resolved = child(
        execution_result=ExecutionResult(content=[TextContent(text="[subagent cancelled]")], is_error=True),
        result_execution_id="ter1",
    )
    entries = {**ENTRIES, "cc1": resolved}

    messages = PROJECTOR.project(["ts", "te1", "cc1", "ter1"], entries)

    assert messages[1].content[0].text == (
        "Subagent task update:\n"
        '<task id=t1 status=failed completed_at="1970-01-01T00:00:00Z">\n[subagent cancelled]\n</task>'
    )


def test_a_link_resolved_without_a_result_execution_renders_in_place():
    # the cancel wind-down and the hard-limit settle resolve links directly,
    # appending nothing — only then does the link itself render, without a
    # timestamp, inside what is by construction a failing bracket
    resolved = child(
        execution_result=ExecutionResult(content=[TextContent(text="[subagent cancelled]")], is_error=True),
    )
    entries = {"ts": TS, "te1": SPAWN, "cc1": resolved, "tf": TF}

    messages = PROJECTOR.project(["ts", "te1", "cc1", "tf"], entries)

    assert [m.content[0].text for m in messages][-1] == (
        "Subagent task update:\n<task id=t1 status=failed>\n[subagent cancelled]\n</task>"
    )


def test_an_unresolved_child_in_the_open_turn_renders_nothing():
    # mid-orchestration the model tracks its tasks through the spawn
    # confirmations, the updates and list_subagents — the link is silent
    messages = PROJECTOR.project(["ts", "te1", "cc1"], {"ts": TS, "te1": SPAWN, "cc1": child()})

    assert [type(m).__name__ for m in messages] == ["ToolMessage"]


def test_an_unresolved_child_outside_the_open_turn_fails_loud():
    # no close may leave an unresolved child behind: this state is a framework
    # bug or hand-authored corruption, and silence would project a task
    # nothing will ever finish — the same fail-loud stance a nonterminal
    # execution gets
    with pytest.raises(ProjectionError, match="unresolved outside the open turn"):
        PROJECTOR.project(["ts", "te1", "cc1", "tf"], {"ts": TS, "te1": SPAWN, "cc1": child(), "tf": TF})


def test_a_child_whose_spawn_execution_is_missing_fails_loud():
    with pytest.raises(ProjectionError, match="missing from the entry store"):
        PROJECTOR.project(["ts", "cc1", "ter1"], {"ts": TS, "cc1": RESOLVED, "ter1": RESULT_EXEC})


def test_a_spawn_execution_without_a_payload_fails_loud():
    # a child cannot exist without a validated spawn payload; state that says
    # otherwise is corruption, not a task with defaults
    stripped = SPAWN.model_copy(update={"result": ExecutionResult(content=[TextContent(text="spawned")])})
    entries = {**ENTRIES, "te1": stripped}

    with pytest.raises(ProjectionError, match="no spawn payload"):
        PROJECTOR.project(["ts", "te1", "cc1", "ter1"], entries)


def test_a_non_text_part_of_a_child_result_survives_beside_the_tag():
    image = ImageContent(source=MediaBase64(data="aGk=", media_type="image/png"))
    resolved = child(
        execution_result=ExecutionResult(content=[TextContent(text="see this"), image]),
        result_execution_id="ter1",
    )
    entries = {**ENTRIES, "cc1": resolved}

    messages = PROJECTOR.project(["ts", "te1", "cc1", "ter1"], entries)

    update = messages[-1]
    assert [type(block).__name__ for block in update.content] == ["TextBlock", "ImageBlock"]
    assert update.content[0].text == (
        'Subagent task update:\n<task id=t1 status=completed completed_at="1970-01-01T00:00:00Z">\nsee this\n</task>'
    )


def test_a_subclass_can_replace_the_update_rendering_wholesale():
    class Terse(ConversationProjector):
        CHILD_UPDATE_PREAMBLE = ""
        CHILD_UPDATE_TEMPLATE = "[subagent {task_id}] {content}"

    messages = Terse().project(["ts", "te1", "cc1", "ter1"], ENTRIES)

    assert messages[-1].content[0].text == "[subagent t1] A is fine."


# ── context accounting ────────────────────────────────────────────────────────


def _session() -> AgentSession:
    return make_session(
        id="s",
        conversations={"c1": conversation("c1", [], created_at=0, updated_at=0)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )


def test_an_unresolved_child_costs_its_parent_nothing():
    # a child's own conversation does not count against its parent's window —
    # that separation is the main reason subagents are useful
    assert CM.calculate_context(_session(), child()) == 0


def test_a_link_whose_result_execution_carries_the_content_costs_nothing():
    # the result execution's own context_tokens already cover the content;
    # counting the link too would double every subagent result
    assert CM.calculate_context(_session(), RESOLVED) == 0


def test_a_link_that_renders_its_own_result_costs_exactly_that_result():
    resolved = child(
        execution_result=ExecutionResult(content=[TextContent(text="x" * 40)]),
    )

    assert CM.calculate_context(_session(), resolved) == 10  # 40 chars // 4


def test_an_image_in_a_link_rendered_result_is_counted_as_media():
    resolved = child(
        execution_result=ExecutionResult(
            content=[ImageContent(source=MediaBase64(data="aGk=", media_type="image/png"))],
        ),
    )

    assert CM.calculate_context(_session(), resolved) == CM.IMAGE_TOKENS


# ── the durable entry ─────────────────────────────────────────────────────────


def test_a_child_conversation_round_trips_through_json():
    session = make_session(
        id="s_child_roundtrip",
        # deep copies: make_session stamps `tool_spec_id` on what it is given,
        # and the module constants serve every test in this file
        entries={
            "te1": SPAWN.model_copy(deep=True),
            "cc1": RESOLVED.model_copy(deep=True),
            "ter1": RESULT_EXEC.model_copy(deep=True),
        },
        conversations={"c1": conversation("c1", ["te1", "cc1", "ter1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )

    reloaded = AgentSession.model_validate_json(session.model_dump_json())

    assert reloaded == session
    assert reloaded.entries["cc1"].result_execution_id == "ter1"
    assert reloaded.entries["cc1"].execution_result == RESOLVED.execution_result


def test_resolving_a_child_is_an_in_place_mutation():
    # the THIRD mutable entry type: `execution_result` and the
    # `result_execution_id` naming where it came from land together, after the
    # entry is already durable — which is why `before_entry_written` fires
    # twice for it
    unresolved = child()
    resolved = unresolved.model_copy(
        update={
            "execution_result": ExecutionResult(content=[TextContent(text="done")]),
            "result_execution_id": "ter1",
        },
    )

    assert unresolved.execution_result is None
    assert unresolved.result_execution_id is None
    assert resolved.id == unresolved.id
    assert resolved.execution_result.content == [TextContent(text="done")]
    assert resolved.result_execution_id == "ter1"


# ── pretty_print ──────────────────────────────────────────────────────────────


def test_pretty_print_renders_a_resolved_child():
    session = make_session(
        id="s_pp_child",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Research A")]),
            "ts": TurnStart(id="ts", parent_id="u1", created_at=500),
            "a1": AssistantMessage(
                id="a1",
                parent_id="ts",
                created_at=500,
                parts=[TextContent(text="on it")],
                llm_config=MODEL,
                stop_reason="stop",
            ),
            "cc1": RESOLVED.model_copy(deep=True, update={"parent_id": "a1"}),
            "ter1": RESULT_EXEC.model_copy(deep=True, update={"parent_id": "cc1"}),
            "tf": TurnFinish(id="tf", parent_id="ter1", created_at=500),
        },
        conversations={
            "c1": conversation("c1", ["u1", "ts", "a1", "cc1", "ter1", "tf"], created_at=500, updated_at=500),
            "c2": conversation("c2", [], created_at=500, updated_at=500, depth=1),
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )

    transcript = pretty_print(session)

    assert "Subagent · c2" in transcript
    assert "A is fine." in transcript


def test_pretty_print_marks_an_unresolved_child():
    session = make_session(
        id="s_pp_child_open",
        entries={
            "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Research A")]),
            "ts": TurnStart(id="ts", parent_id="u1", created_at=500),
            "cc1": child(parent_id="ts"),
        },
        conversations={
            "c1": conversation("c1", ["u1", "ts", "cc1"], created_at=500, updated_at=500),
            "c2": conversation("c2", [], created_at=500, updated_at=500, depth=1),
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )

    assert "Subagent · c2 · unresolved" in pretty_print(session)


def test_pretty_print_of_a_subagent_conversation_shows_its_depth():
    session = make_session(
        id="s_pp_depth",
        entries={
            "u2": UserMessage(id="u2", created_at=500, parts=[TextContent(text="You are a subagent")]),
        },
        conversations={
            "c1": conversation("c1", [], created_at=500, updated_at=500),
            "c2": conversation("c2", ["u2"], created_at=500, updated_at=500, depth=1),
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )

    assert "· depth 1" in pretty_print(session, "c2")
    # the main conversation is depth 0 and its header says nothing about it
    assert "· depth" not in pretty_print(session, "c1")
