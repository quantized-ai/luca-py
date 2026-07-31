"""`ChildConversation` as a DATA type: projection, context accounting,
`pretty_print`, mutability and serialization.

Nothing here drives a subagent — no runner, no spawn handshake. This is the
entry on its own: what it contributes to its PARENT's wire history, what it
costs its parent's window, and how it renders. The runtime that produces one
is covered under `tests/agent/subagents/`.
"""

import pytest

from luca.agent.core.context_manager import ContextManager
from luca.agent.core.exceptions import ProjectionError
from luca.agent.core.models import (
    AgentSession,
    AssistantMessage,
    ChildConversation,
    ExecutionResult,
    ExecutionStatus,
    ImageBase64,
    ImageContent,
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
from tests.agent.scenarios import ADD_SPEC, MODEL, conversation, make_session

PROJECTOR = ConversationProjector()
CM = ContextManager()

# The spawn execution the child hangs off. Its `tool_call_id` is what the
# projection tags the result with — the model correlates the answer with the
# request IT made, which is the tool call, not an internal entry id.
SPAWN = ToolExecution(
    id="te1",
    created_at=500,
    conversation_id="c1",
    tool_call_id="tc1",
    raw_tool_call=ToolCall(id="tc1", name="spawn_subagent", arguments={"prompt": "Research A"}),
    tool_spec=ADD_SPEC,
    status=ExecutionStatus.COMPLETED,
    result=ExecutionResult(content=[TextContent(text="Spawned subagent: research A")]),
    started_at=500,
    ended_at=500,
)


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
)


# ── projection ────────────────────────────────────────────────────────────────


def test_the_spawn_output_and_the_child_result_are_two_separate_messages():
    # the spawn call completes immediately and projects its own status line as
    # a ToolMessage; the child's ANSWER arrives later, separately, as a
    # synthetic user message. Two channels, deliberately.
    messages = PROJECTOR.project(["te1", "cc1"], {"te1": SPAWN, "cc1": RESOLVED})

    assert [type(m).__name__ for m in messages] == ["ToolMessage", "UserMessage"]
    assert messages[0].content[0].text == "Spawned subagent: research A"
    assert messages[1].content[0].text == "<task id=tc1>\nA is fine.\n</task>"


def test_the_child_result_is_tagged_with_the_spawning_tool_call():
    [message] = PROJECTOR.project(["cc1"], {"te1": SPAWN, "cc1": RESOLVED})

    assert message == LucaUserMessage(
        content=[LucaTextBlock(text="<task id=tc1>\nA is fine.\n</task>")],
    )


def test_two_children_project_as_two_messages_in_path_order():
    second = child(
        id="cc2",
        conversation_id="c3",
        tool_execution_id="te2",
        execution_result=ExecutionResult(content=[TextContent(text="B is broken.")]),
    )
    spawn_b = SPAWN.model_copy(
        update={
            "id": "te2",
            "tool_call_id": "tc2",
            "raw_tool_call": ToolCall(id="tc2", name="spawn_subagent"),
        },
    )
    entries = {"te1": SPAWN, "te2": spawn_b, "cc1": RESOLVED, "cc2": second}

    messages = PROJECTOR.project(["cc1", "cc2"], entries)

    assert [m.content[0].text for m in messages] == [
        "<task id=tc1>\nA is fine.\n</task>",
        "<task id=tc2>\nB is broken.\n</task>",
    ]


def test_an_unresolved_child_fails_loud():
    # the same rule a PENDING ToolExecution gets: the runtime must never call
    # the model while a subagent is still working, and a placeholder would tell
    # the model a subagent answered when it has not
    with pytest.raises(ProjectionError, match="unresolved subagent is not projectable"):
        PROJECTOR.project(["cc1"], {"te1": SPAWN, "cc1": child()})


def test_a_child_whose_spawn_execution_is_missing_fails_loud():
    with pytest.raises(ProjectionError, match="missing from the entry store"):
        PROJECTOR.project(["cc1"], {"cc1": RESOLVED})


def test_a_non_text_part_of_a_child_result_survives_beside_the_tag():
    image = ImageContent(source=ImageBase64(data="aGk=", media_type="image/png"))
    resolved = child(
        execution_result=ExecutionResult(content=[TextContent(text="see this"), image]),
    )

    [message] = PROJECTOR.project(["cc1"], {"te1": SPAWN, "cc1": resolved})

    assert [type(block).__name__ for block in message.content] == ["TextBlock", "ImageBlock"]
    assert message.content[0].text == "<task id=tc1>\nsee this\n</task>"


def test_a_subclass_can_replace_the_child_rendering_wholesale():
    class Terse(ConversationProjector):
        CHILD_TASK_TEMPLATE = "[subagent {task_id}] {content}"

    [message] = Terse().project(["cc1"], {"te1": SPAWN, "cc1": RESOLVED})

    assert message.content[0].text == "[subagent tc1] A is fine."


# ── context accounting ────────────────────────────────────────────────────────


def test_an_unresolved_child_costs_its_parent_nothing():
    # a child's own conversation does not count against its parent's window —
    # that separation is the main reason subagents are useful
    session = make_session(
        id="s",
        conversations={"c1": conversation("c1", [], created_at=0, updated_at=0)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )

    assert CM.calculate_context(session, child()) == 0


def test_a_resolved_child_costs_exactly_its_result():
    session = make_session(
        id="s",
        conversations={"c1": conversation("c1", [], created_at=0, updated_at=0)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    resolved = child(
        execution_result=ExecutionResult(content=[TextContent(text="x" * 40)]),
    )

    assert CM.calculate_context(session, resolved) == 10  # 40 chars // 4


def test_an_image_in_a_child_result_is_counted_as_media():
    session = make_session(
        id="s",
        conversations={"c1": conversation("c1", [], created_at=0, updated_at=0)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    resolved = child(
        execution_result=ExecutionResult(
            content=[ImageContent(source=ImageBase64(data="aGk=", media_type="image/png"))],
        ),
    )

    assert CM.calculate_context(session, resolved) == CM.IMAGE_TOKENS


# ── the durable entry ─────────────────────────────────────────────────────────


def test_a_child_conversation_round_trips_through_json():
    session = make_session(
        id="s_child_roundtrip",
        entries={"te1": SPAWN, "cc1": RESOLVED},
        conversations={"c1": conversation("c1", ["te1", "cc1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )

    reloaded = AgentSession.model_validate_json(session.model_dump_json())

    assert reloaded == session
    assert reloaded.entries["cc1"] == RESOLVED


def test_resolving_a_child_is_an_in_place_mutation():
    # the THIRD mutable entry type: `execution_result` lands after the entry is
    # already durable, which is why `before_entry_written` fires twice for it
    unresolved = child()
    resolved = unresolved.model_copy(
        update={"execution_result": ExecutionResult(content=[TextContent(text="done")])},
    )

    assert unresolved.execution_result is None
    assert resolved.id == unresolved.id
    assert resolved.execution_result.content == [TextContent(text="done")]


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
            "cc1": RESOLVED.model_copy(update={"parent_id": "a1"}),
            "tf": TurnFinish(id="tf", parent_id="cc1", created_at=500),
        },
        conversations={
            "c1": conversation("c1", ["u1", "ts", "a1", "cc1", "tf"], created_at=500, updated_at=500),
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
