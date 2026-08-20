"""Declarative tests for the ConversationProjector: a KNOWN conversation
projects to a KNOWN list of canonical client messages, and a KNOWN
ToolExecution projects to a KNOWN ToolMessage. Hardcoded state in, full
expected object out; no loop, no runner, no provider.

Unit boundary: deterministic derivation from durable entries. Ground truths:
entry constructors produce correct objects; client DTOs compare by value.
Projection errors fail loudly (ProjectionError) — never silent omissions or
synthetic fallbacks.

`tool_spec` is carried by every real `ToolExecution` but the projector never
reads it — the wire form comes from `raw_tool_call` and `result`/`error` — so
the literals here use `scenarios.spec()` rather than spelling out a required
`description` and `input_schema` this file does not own.
"""

from typing import ClassVar

import pytest

from luca.agent.core.exceptions import ProjectionError
from luca.agent.core.models import (
    ApprovalDecision,
    ApprovalOption,
    ApprovalStatus,
    AssistantMessage,
    AudioContent,
    CancelRequested,
    CompactionEntry,
    CompactionSource,
    Conversation,
    ExecutionAttempt,
    ExecutionAttemptOutcome,
    ExecutionResult,
    ExecutionStatus,
    FileContent,
    ImageContent,
    LLMConfig,
    MediaBase64,
    MediaFileId,
    MediaURL,
    PrunedEntry,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolExecution,
    ToolExecutionError,
    TurnFinish,
    TurnOutcome,
    TurnStart,
    UserMessage,
)
from luca.agent.core.projection import (
    CANCELLED_TURN_MARKER,
    ConversationProjector,
    tool_message_text,
)
from luca.client.types import (
    AssistantMessage as LucaAssistantMessage,
    AudioBlock as LucaAudioBlock,
    FileBlock as LucaFileBlock,
    ImageBlock as LucaImageBlock,
    MediaBase64 as LucaMediaBase64,
    MediaFileId as LucaMediaFileId,
    MediaURL as LucaMediaURL,
    TextBlock,
    ThinkingBlock,
    ToolCall as LucaToolCall,
    ToolMessage,
    UserMessage as LucaUserMessage,
)
from tests.agent.scenarios import spec

PROJECTOR = ConversationProjector()

MODEL = LLMConfig(model="m", provider="p")


# ── top-level traversal ────────────────────────────────────────────────────────


def test_single_user_message():
    entries = {
        "u1": UserMessage(id="u1", created_at=1000, parts=[TextContent(text="Hello")]),
    }
    conversation = Conversation(id="c1", nodes=["u1"], created_at=1000, updated_at=1000)

    assert PROJECTOR.project(conversation.nodes, entries) == [
        LucaUserMessage(content=[TextBlock(text="Hello")]),
    ]


def test_turn_markers_are_dropped():
    entries = {
        "u1": UserMessage(id="u1", created_at=1000, parts=[TextContent(text="Hi")]),
        "ts": TurnStart(id="ts", created_at=1001),
        "a1": AssistantMessage(
            id="a1",
            created_at=1002,
            parts=[TextContent(text="Hey there")],
            llm_config=MODEL,
            stop_reason="stop",
        ),
        "tf": TurnFinish(id="tf", created_at=1003),
    }
    conversation = Conversation(
        id="c1",
        nodes=["u1", "ts", "a1", "tf"],
        created_at=1000,
        updated_at=1003,
    )

    assert PROJECTOR.project(conversation.nodes, entries) == [
        LucaUserMessage(content=[TextBlock(text="Hi")]),
        LucaAssistantMessage(content=[TextBlock(text="Hey there")], provider="p", model="m"),
    ]


def test_full_tool_call_turn():
    entries = {
        "u1": UserMessage(
            id="u1",
            created_at=1000,
            parts=[TextContent(text="Add 1 and 2")],
        ),
        "ts": TurnStart(id="ts", created_at=1001),
        "a1": AssistantMessage(
            id="a1",
            created_at=1002,
            parts=[
                ThinkingContent(thinking="Use the add tool."),
                ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
            ],
            llm_config=MODEL,
            stop_reason="tool_use",
        ),
        "te1": ToolExecution(
            id="te1",
            conversation_id="c1",
            created_at=1003,
            tool_call_id="tc1",
            raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
            tool_spec=spec("add"),
            status=ExecutionStatus.COMPLETED,
            result=ExecutionResult(content=[TextContent(text="3")]),
            approval_status=ApprovalStatus.ALLOWED,
            approval_decisions=[
                ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1003),
            ],
            attempts=[ExecutionAttempt(outcome=ExecutionAttemptOutcome.COMPLETED, started_at=1003, ended_at=1003)],
            finished_at=1003,
        ),
        "a2": AssistantMessage(
            id="a2",
            created_at=1004,
            parts=[TextContent(text="The answer is 3.")],
            llm_config=MODEL,
            stop_reason="stop",
        ),
        "tf": TurnFinish(id="tf", created_at=1005),
    }
    conversation = Conversation(
        id="c1",
        nodes=["u1", "ts", "a1", "te1", "a2", "tf"],
        created_at=1000,
        updated_at=1005,
    )

    assert PROJECTOR.project(conversation.nodes, entries) == [
        LucaUserMessage(content=[TextBlock(text="Add 1 and 2")]),
        LucaAssistantMessage(
            content=[
                ThinkingBlock(text="Use the add tool."),
                LucaToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
            ],
            provider="p",
            model="m",
        ),
        ToolMessage(tool_call_id="tc1", content=[TextBlock(text="3")]),
        LucaAssistantMessage(content=[TextBlock(text="The answer is 3.")], provider="p", model="m"),
    ]


def test_compaction_renders_as_synthetic_user_message():
    entries = {
        "cmp": CompactionEntry(
            id="cmp",
            created_at=2000,
            source=CompactionSource.POLICY,
            parts=[TextContent(text="## Goal\nFix the failing test suite.")],
            compacted_nodes=["u1", "a1"],
        ),
        "u2": UserMessage(
            id="u2",
            created_at=2001,
            parts=[TextContent(text="What next?")],
        ),
    }
    conversation = Conversation(id="c1", nodes=["cmp", "u2"], created_at=2000, updated_at=2001)

    assert PROJECTOR.project(conversation.nodes, entries) == [
        LucaUserMessage(content=[TextBlock(text="## Goal\nFix the failing test suite.")]),
        LucaUserMessage(content=[TextBlock(text="What next?")]),
    ]


def test_a_summary_carrying_an_image_projects_both_blocks_in_order():
    entries = {
        "cmp": CompactionEntry(
            id="cmp",
            created_at=2000,
            source=CompactionSource.POLICY,
            parts=[
                TextContent(text="Earlier: a screenshot of the failure."),
                ImageContent(
                    source=MediaBase64(data="aGk=", media_type="image/png"),
                ),
            ],
            compacted_nodes=["u1", "a1"],
        ),
    }
    conversation = Conversation(id="c1", nodes=["cmp"], created_at=2000, updated_at=2000)

    assert PROJECTOR.project(conversation.nodes, entries) == [
        LucaUserMessage(
            content=[
                TextBlock(text="Earlier: a screenshot of the failure."),
                LucaImageBlock(source=LucaMediaBase64(data="aGk=", media_type="image/png")),
            ]
        ),
    ]


def test_a_compaction_with_no_parts_projects_nothing():
    entries = {
        "cmp": CompactionEntry(
            id="cmp",
            created_at=2000,
            source=CompactionSource.USER,
        ),
        "u2": UserMessage(
            id="u2",
            created_at=2001,
            parts=[TextContent(text="What next?")],
        ),
    }
    conversation = Conversation(id="c1", nodes=["cmp", "u2"], created_at=2000, updated_at=2001)

    assert PROJECTOR.project(conversation.nodes, entries) == [
        LucaUserMessage(content=[TextBlock(text="What next?")]),
    ]


def test_a_compaction_with_empty_parts_projects_nothing():
    entry = CompactionEntry(
        id="cmp",
        created_at=2000,
        source=CompactionSource.USER,
        parts=[],
    )

    assert PROJECTOR.project_compaction(entry, {"cmp": entry}) is None


def test_a_subclass_can_replace_project_compaction():
    class Prefixed(ConversationProjector):
        def project_compaction(self, entry, entries):
            return LucaUserMessage(
                content=[TextBlock(text=f"SUMMARY: {entry.parts[0].text}")],
            )

    entries = {
        "cmp": CompactionEntry(
            id="cmp",
            created_at=2000,
            source=CompactionSource.POLICY,
            parts=[TextContent(text="everything so far")],
            compacted_nodes=["u1"],
        ),
    }
    conversation = Conversation(id="c1", nodes=["cmp"], created_at=2000, updated_at=2000)

    assert Prefixed().project(conversation.nodes, entries) == [
        LucaUserMessage(content=[TextBlock(text="SUMMARY: everything so far")]),
    ]


# ── the compaction bracket: a positional rule on project() ────────────────────


def test_a_cancelled_compaction_bracket_projects_nothing_at_all():
    # the user cancelled a COMPACTION, not their question — without the
    # positional rule tf_c would put "[Request interrupted by user]" on the
    # wire, durably, about a question the model was never shown
    entries = {
        "u4": UserMessage(id="u4", created_at=2000, parts=[TextContent(text="what is X?")]),
        "ts_c": TurnStart(id="ts_c", parent_id="u4", created_at=2001),
        "cmp": CompactionEntry(
            id="cmp",
            parent_id="ts_c",
            created_at=2002,
            source=CompactionSource.POLICY,
            started_at=2002,
            ended_at=2003,
        ),
        "cr": CancelRequested(id="cr", parent_id="cmp", created_at=2003),
        "tf_c": TurnFinish(
            id="tf_c",
            parent_id="cr",
            created_at=2003,
            outcome=TurnOutcome.CANCELLED,
        ),
    }
    conversation = Conversation(
        id="c1",
        nodes=["u4", "ts_c", "cmp", "cr", "tf_c"],
        created_at=2000,
        updated_at=2003,
    )

    assert PROJECTOR.project(conversation.nodes, entries) == [
        LucaUserMessage(content=[TextBlock(text="what is X?")]),
    ]


def test_an_archived_conversation_projects_its_originals_and_no_summary():
    # c1 still holds every original entry, so the summary of them must NOT
    # also project here — the entry is bookkeeping inside its own bracket
    entries = {
        "u1": UserMessage(id="u1", created_at=2000, parts=[TextContent(text="hello")]),
        "ts_c": TurnStart(id="ts_c", parent_id="u1", created_at=2001),
        "cmp": CompactionEntry(
            id="cmp",
            parent_id="ts_c",
            created_at=2002,
            source=CompactionSource.POLICY,
            parts=[TextContent(text="the user said hello")],
            compacted_nodes=["u1"],
            started_at=2002,
            ended_at=2002,
        ),
        "tf_c": TurnFinish(id="tf_c", parent_id="cmp", created_at=2002),
    }
    conversation = Conversation(
        id="c1",
        nodes=["u1", "ts_c", "cmp", "tf_c"],
        created_at=2000,
        updated_at=2002,
    )

    assert PROJECTOR.project(conversation.nodes, entries) == [
        LucaUserMessage(content=[TextBlock(text="hello")]),
    ]


def test_the_new_conversation_projects_the_summary_as_a_user_message():
    # the same entry, one node later: on the path where the originals are gone
    entries = {
        "cmp": CompactionEntry(
            id="cmp",
            parent_id="ts_c",
            created_at=2002,
            source=CompactionSource.POLICY,
            parts=[TextContent(text="the user said hello")],
            compacted_nodes=["u1"],
            started_at=2002,
            ended_at=2002,
        ),
        "u4": UserMessage(id="u4", created_at=2003, parts=[TextContent(text="and now?")]),
    }
    conversation = Conversation(
        id="c2",
        nodes=["cmp", "u4"],
        created_at=2002,
        updated_at=2003,
    )

    assert PROJECTOR.project(conversation.nodes, entries) == [
        LucaUserMessage(content=[TextBlock(text="the user said hello")]),
        LucaUserMessage(content=[TextBlock(text="and now?")]),
    ]


def test_two_stacked_closed_compaction_brackets_are_both_skipped():
    entries = {
        "u1": UserMessage(id="u1", created_at=2000, parts=[TextContent(text="hello")]),
        "ts_a": TurnStart(id="ts_a", parent_id="u1", created_at=2001),
        "cmp_a": CompactionEntry(
            id="cmp_a",
            parent_id="ts_a",
            created_at=2002,
            source=CompactionSource.POLICY,
            started_at=2002,
            ended_at=2002,
        ),
        "tf_a": TurnFinish(id="tf_a", parent_id="cmp_a", created_at=2002),
        "ts_b": TurnStart(id="ts_b", parent_id="tf_a", created_at=2003),
        "cmp_b": CompactionEntry(
            id="cmp_b",
            parent_id="ts_b",
            created_at=2004,
            source=CompactionSource.USER,
            started_at=2004,
            ended_at=2004,
        ),
        "tf_b": TurnFinish(
            id="tf_b",
            parent_id="cmp_b",
            created_at=2004,
            outcome=TurnOutcome.ERRORED,
            error="kaboom",
        ),
    }
    conversation = Conversation(
        id="c1",
        nodes=["u1", "ts_a", "cmp_a", "tf_a", "ts_b", "cmp_b", "tf_b"],
        created_at=2000,
        updated_at=2004,
    )

    assert PROJECTOR.project(conversation.nodes, entries) == [
        LucaUserMessage(content=[TextBlock(text="hello")]),
    ]


def test_an_open_compaction_bracket_is_skipped_to_the_end_of_the_path():
    entries = {
        "u1": UserMessage(id="u1", created_at=2000, parts=[TextContent(text="hello")]),
        "ts_c": TurnStart(id="ts_c", parent_id="u1", created_at=2001),
        "cmp": CompactionEntry(
            id="cmp",
            parent_id="ts_c",
            created_at=2002,
            source=CompactionSource.USER,
        ),
    }
    conversation = Conversation(
        id="c1",
        nodes=["u1", "ts_c", "cmp"],
        created_at=2000,
        updated_at=2002,
    )

    assert PROJECTOR.project(conversation.nodes, entries) == [
        LucaUserMessage(content=[TextBlock(text="hello")]),
    ]


def test_a_conversational_cancelled_bracket_still_projects_the_marker():
    # the regression guard: ONE predicate separates a compaction bracket from
    # an ordinary turn, and an ordinary cancelled turn is unchanged
    entries = {
        "u1": UserMessage(id="u1", created_at=2000, parts=[TextContent(text="hello")]),
        "ts": TurnStart(id="ts", parent_id="u1", created_at=2001),
        "cr": CancelRequested(id="cr", parent_id="ts", created_at=2002),
        "tf": TurnFinish(
            id="tf",
            parent_id="cr",
            created_at=2002,
            outcome=TurnOutcome.CANCELLED,
        ),
    }
    conversation = Conversation(
        id="c1",
        nodes=["u1", "ts", "cr", "tf"],
        created_at=2000,
        updated_at=2002,
    )

    assert PROJECTOR.project(conversation.nodes, entries) == [
        LucaUserMessage(content=[TextBlock(text="hello")]),
        LucaUserMessage(content=[TextBlock(text=CANCELLED_TURN_MARKER)]),
    ]


def test_cancelled_turn_finish_projects_as_interrupted_marker():
    # end-to-end cancel → post → projection order: the cancelled bracket
    # renders [assistant, synthetic user marker], CancelRequested is dropped,
    # and the follow-up user message lands after the marker.
    entries = {
        "u1": UserMessage(id="u1", created_at=1000, parts=[TextContent(text="Go")]),
        "ts": TurnStart(id="ts", created_at=1001),
        "a1": AssistantMessage(
            id="a1",
            created_at=1002,
            parts=[TextContent(text="Working on it…")],
            llm_config=MODEL,
            stop_reason="stop",
        ),
        "cr": CancelRequested(id="cr", created_at=1003),
        "tf": TurnFinish(
            id="tf",
            created_at=1004,
            outcome=TurnOutcome.CANCELLED,
        ),
        "u2": UserMessage(
            id="u2",
            created_at=1005,
            parts=[TextContent(text="Try again")],
        ),
    }
    conversation = Conversation(
        id="c1",
        nodes=["u1", "ts", "a1", "cr", "tf", "u2"],
        created_at=1000,
        updated_at=1005,
    )

    assert PROJECTOR.project(conversation.nodes, entries) == [
        LucaUserMessage(content=[TextBlock(text="Go")]),
        LucaAssistantMessage(content=[TextBlock(text="Working on it…")], provider="p", model="m"),
        LucaUserMessage(content=[TextBlock(text=CANCELLED_TURN_MARKER)]),
        LucaUserMessage(content=[TextBlock(text="Try again")]),
    ]


def test_failed_turn_finish_is_dropped_but_its_content_projects():
    # TIMED_OUT / ERRORED closes are not the model's business — but real work
    # recorded inside the failed bracket (a completed tool round-trip) is.
    entries = {
        "u1": UserMessage(id="u1", created_at=1000, parts=[TextContent(text="Add")]),
        "ts": TurnStart(id="ts", created_at=1001),
        "a1": AssistantMessage(
            id="a1",
            created_at=1002,
            parts=[ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2})],
            llm_config=MODEL,
            stop_reason="tool_use",
        ),
        "te1": ToolExecution(
            id="te1",
            conversation_id="c1",
            created_at=1003,
            tool_call_id="tc1",
            raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
            tool_spec=spec("add"),
            status=ExecutionStatus.COMPLETED,
            result=ExecutionResult(content=[TextContent(text="3")]),
            approval_status=ApprovalStatus.ALLOWED,
            attempts=[ExecutionAttempt(outcome=ExecutionAttemptOutcome.COMPLETED, started_at=1003, ended_at=1003)],
            finished_at=1003,
        ),
        "tf": TurnFinish(
            id="tf",
            created_at=1004,
            outcome=TurnOutcome.TIMED_OUT,
            error="client timeout",
        ),
    }
    conversation = Conversation(
        id="c1",
        nodes=["u1", "ts", "a1", "te1", "tf"],
        created_at=1000,
        updated_at=1004,
    )

    assert PROJECTOR.project(conversation.nodes, entries) == [
        LucaUserMessage(content=[TextBlock(text="Add")]),
        LucaAssistantMessage(
            content=[
                LucaToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
            ],
            provider="p",
            model="m",
        ),
        ToolMessage(tool_call_id="tc1", content=[TextBlock(text="3")]),
    ]


def test_errored_turn_finish_is_dropped():
    entries = {
        "u1": UserMessage(id="u1", created_at=1000, parts=[TextContent(text="Hi")]),
        "ts": TurnStart(id="ts", created_at=1001),
        "tf": TurnFinish(
            id="tf",
            created_at=1002,
            outcome=TurnOutcome.ERRORED,
            error="provider 500",
        ),
    }
    conversation = Conversation(
        id="c1",
        nodes=["u1", "ts", "tf"],
        created_at=1000,
        updated_at=1002,
    )

    assert PROJECTOR.project(conversation.nodes, entries) == [
        LucaUserMessage(content=[TextBlock(text="Hi")]),
    ]


def test_missing_entry_id_fails_loudly():
    conversation = Conversation(id="c1", nodes=["ghost"], created_at=1000, updated_at=1000)

    with pytest.raises(ProjectionError, match="ghost"):
        PROJECTOR.project(conversation.nodes, {})


# ── tool-execution projection: one method, every status ───────────────────────


def _execution(**overrides) -> ToolExecution:
    base = {
        "id": "te1",
        "created_at": 1000,
        "tool_call_id": "tc1",
        "raw_tool_call": ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        "tool_spec": spec("add"),
    }
    return ToolExecution(**{**base, **overrides})


def test_completed_projects_result_content_and_preserves_is_error():
    execution = _execution(
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(
            content=[TextContent(text="The requested file does not exist.")],
            metadata={"path": "/tmp/missing.txt"},
            is_error=True,
        ),
        attempts=[ExecutionAttempt(outcome=ExecutionAttemptOutcome.COMPLETED, started_at=1000, ended_at=1001)],
        finished_at=1001,
    )

    # is_error=True is the TOOL's verdict; the execution is still COMPLETED
    # and must NOT be reinterpreted as a framework failure.
    assert PROJECTOR.project_tool_execution(execution, {}) == ToolMessage(
        tool_call_id="tc1",
        content=[TextBlock(text="The requested file does not exist.")],
        is_error=True,
    )


def test_structured_content_never_reaches_the_tool_message():
    # `content` is the sole model-facing channel: a payload the app reads must
    # not change one byte of what the model is told
    execution = _execution(
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(
            content=[TextContent(text="25°C, wind from the south.")],
            structured_content={"degrees_in_celsius": 25, "wind_direction": "south"},
        ),
        attempts=[ExecutionAttempt(outcome=ExecutionAttemptOutcome.COMPLETED, started_at=1000, ended_at=1001)],
        finished_at=1001,
    )

    assert PROJECTOR.project_tool_execution(execution, {}) == ToolMessage(
        tool_call_id="tc1",
        content=[TextBlock(text="25°C, wind from the south.")],
        is_error=False,
    )


def test_not_found_projects_the_structured_error_message():
    execution = _execution(
        tool_spec=None,
        raw_tool_call=ToolCall(id="tc1", name="read_database", arguments={}),
        status=ExecutionStatus.NOT_FOUND,
        error=ToolExecutionError(
            error_type="ToolNotFound",
            error_message="Unknown tool: 'read_database'.",
        ),
        finished_at=1001,
    )

    assert PROJECTOR.project_tool_execution(execution, {}) == ToolMessage(
        tool_call_id="tc1",
        content=[TextBlock(text="Unknown tool: 'read_database'.")],
        is_error=True,
    )


def test_invalid_projects_message_plus_validation_errors():
    execution = _execution(
        status=ExecutionStatus.INVALID,
        error=ToolExecutionError(
            error_type="InvalidToolArguments",
            error_message="Arguments for tool 'add' are invalid.",
            # exactly what the runner writes: the phase alongside the
            # structured errors, which the projector must pick out and render
            details={
                "phase": "prepare",
                "errors": [
                    {"type": "missing", "loc": ["b"], "msg": "Field required", "input": {"a": 1}},
                ],
            },
        ),
        finished_at=1001,
    )

    assert PROJECTOR.project_tool_execution(execution, {}) == ToolMessage(
        tool_call_id="tc1",
        content=[
            TextBlock(
                text=(
                    "Arguments for tool 'add' are invalid.\n"
                    '[{"type": "missing", "loc": ["b"], "msg": "Field required", '
                    '"input": {"a": 1}}]'
                ),
            )
        ],
        is_error=True,
    )


def test_failed_projects_error_type_and_message():
    execution = _execution(
        status=ExecutionStatus.FAILED,
        error=ToolExecutionError(
            error_type="ConnectionError",
            error_message="Connection to api.example.com was closed.",
            details={"phase": "execution"},
        ),
        attempts=[ExecutionAttempt(outcome=ExecutionAttemptOutcome.FAILED, started_at=1000, ended_at=1001)],
        finished_at=1001,
    )

    assert PROJECTOR.project_tool_execution(execution, {}) == ToolMessage(
        tool_call_id="tc1",
        content=[
            TextBlock(
                text=("Tool execution failed: ConnectionError: Connection to api.example.com was closed."),
            )
        ],
        is_error=True,
    )


def test_status_only_terminals_project_their_placeholders():
    # REJECTED and CANCELLED never reached a body, so they carry no attempt;
    # INTERRUPTED and TIMED_OUT came out of one, so they carry the closed
    # attempt that produced them.
    rejected = _execution(
        status=ExecutionStatus.REJECTED,
        approval_status=ApprovalStatus.REJECTED,
        finished_at=1001,
    )
    cancelled = _execution(
        status=ExecutionStatus.CANCELLED,
        cancel_signalled_at=1001,
        finished_at=1001,
    )
    interrupted = _execution(
        status=ExecutionStatus.INTERRUPTED,
        attempts=[ExecutionAttempt(outcome=ExecutionAttemptOutcome.INTERRUPTED, started_at=1000, ended_at=1001)],
        finished_at=1001,
    )
    timed_out = _execution(
        status=ExecutionStatus.TIMED_OUT,
        attempts=[ExecutionAttempt(outcome=ExecutionAttemptOutcome.TIMED_OUT, started_at=1000, ended_at=1001)],
        finished_at=1001,
    )

    assert PROJECTOR.project_tool_execution(rejected, {}) == ToolMessage(
        tool_call_id="tc1",
        content=[TextBlock(text="[tool execution rejected]")],
        is_error=True,
    )
    assert PROJECTOR.project_tool_execution(cancelled, {}) == ToolMessage(
        tool_call_id="tc1",
        content=[TextBlock(text="[tool execution cancelled]")],
        is_error=True,
    )
    assert PROJECTOR.project_tool_execution(interrupted, {}) == ToolMessage(
        tool_call_id="tc1",
        content=[TextBlock(text="[tool execution interrupted]")],
        is_error=True,
    )
    assert PROJECTOR.project_tool_execution(timed_out, {}) == ToolMessage(
        tool_call_id="tc1",
        content=[TextBlock(text="[tool execution timed_out]")],
        is_error=True,
    )


def test_refused_projects_the_limits_own_wording():
    # REFUSED carries the refusing limit's message — the model must read the
    # real reason, not a placeholder; without an error it rides the dict
    refused = _execution(
        status=ExecutionStatus.REFUSED,
        error=ToolExecutionError(
            error_type="SpawnLimitReached",
            error_message="Spawn limit reached (3/3 subagents this turn). Do not retry.",
            details={"limit": 3, "committed": 3},
        ),
        finished_at=1001,
    )
    bare = _execution(status=ExecutionStatus.REFUSED, finished_at=1001)

    assert PROJECTOR.project_tool_execution(refused, {}) == ToolMessage(
        tool_call_id="tc1",
        content=[TextBlock(text="Spawn limit reached (3/3 subagents this turn). Do not retry.")],
        is_error=True,
    )
    assert PROJECTOR.project_tool_execution(bare, {}) == ToolMessage(
        tool_call_id="tc1",
        content=[TextBlock(text="[tool execution refused]")],
        is_error=True,
    )


def test_gated_execution_projects_the_awaiting_approval_placeholder():
    # THE FIRST PROJECTABLE NONTERMINAL STATE (0008): PENDING with an approval
    # the policy explicitly deferred. `is_error` stays False — the call has
    # not failed, it has not run, and an error result is exactly what makes a
    # model retry.
    execution = _execution(
        status=ExecutionStatus.PENDING,
        approval_status=ApprovalStatus.PENDING,
        approval_decisions=[
            ApprovalDecision(decision=ApprovalOption.PENDING, created_at=1000),
        ],
    )

    assert PROJECTOR.project_tool_execution(execution, {}) == ToolMessage(
        tool_call_id="tc1",
        content=[TextBlock(text="[tool execution is awaiting approval — it has not run, and this is not its result]")],
        is_error=False,
    )


def test_parked_execution_projects_the_awaiting_result_placeholder():
    # THE SECOND (0007): the tool was dispatched and returned ExecutionDeferred,
    # so the attempt is closed as DEFERRED while the execution itself stays
    # nonterminal — `finished_at` is still None. `is_error` stays False for the
    # gate's reason, and the wording actively discourages a re-call, which would
    # mint a second ToolExecution for one job.
    execution = _execution(
        status=ExecutionStatus.AWAITING_RESULT,
        approval_status=ApprovalStatus.ALLOWED,
        attempts=[ExecutionAttempt(outcome=ExecutionAttemptOutcome.DEFERRED, started_at=1000, ended_at=1000)],
    )

    assert PROJECTOR.project_tool_execution(execution, {}) == ToolMessage(
        tool_call_id="tc1",
        content=[
            TextBlock(
                text=(
                    "[tool execution has not finished — this is not its result; "
                    "do not call it again, the result will follow when it completes]"
                )
            )
        ],
        is_error=False,
    )


def test_pending_execution_is_not_projectable():
    # approval_status None: the policy never processed it — a runtime state
    # the drive advances, not the durable gate the carve-out is for
    execution = _execution(status=ExecutionStatus.PENDING)

    with pytest.raises(ProjectionError, match="pending"):
        PROJECTOR.project_tool_execution(execution, {})


def test_pending_allowed_execution_is_not_projectable():
    # ALLOWED but not yet dispatched: the drive's to move, still mid-execution
    execution = _execution(
        status=ExecutionStatus.PENDING,
        approval_status=ApprovalStatus.ALLOWED,
        approval_decisions=[
            ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000),
        ],
    )

    with pytest.raises(ProjectionError, match="pending"):
        PROJECTOR.project_tool_execution(execution, {})


def test_received_execution_is_not_projectable():
    execution = _execution(status=ExecutionStatus.RECEIVED)

    with pytest.raises(ProjectionError, match="received"):
        PROJECTOR.project_tool_execution(execution, {})


def test_running_execution_is_not_projectable():
    # in flight: one OPEN attempt, no outcome and no ended_at on it
    execution = _execution(
        status=ExecutionStatus.RUNNING,
        attempts=[ExecutionAttempt(started_at=1000)],
    )

    with pytest.raises(ProjectionError, match="running"):
        PROJECTOR.project_tool_execution(execution, {})


def test_completed_without_result_fails_loudly():
    execution = _execution(
        status=ExecutionStatus.COMPLETED,
        attempts=[ExecutionAttempt(outcome=ExecutionAttemptOutcome.COMPLETED, started_at=1000, ended_at=1001)],
        finished_at=1001,
    )

    with pytest.raises(ProjectionError, match="no ExecutionResult"):
        PROJECTOR.project_tool_execution(execution, {})


def test_projection_preserves_tool_call_id_correlation():
    execution = _execution(
        tool_call_id="toolu_0abc",
        raw_tool_call=ToolCall(id="toolu_0abc", name="add", arguments={}),
        status=ExecutionStatus.TIMED_OUT,
        attempts=[ExecutionAttempt(outcome=ExecutionAttemptOutcome.TIMED_OUT, started_at=1000, ended_at=1001)],
        finished_at=1001,
    )

    assert PROJECTOR.project_tool_execution(execution, {}).tool_call_id == "toolu_0abc"


def test_projection_is_deterministic_for_the_same_execution():
    execution = _execution(
        status=ExecutionStatus.FAILED,
        error=ToolExecutionError(error_type="ValueError", error_message="kaboom"),
        attempts=[ExecutionAttempt(outcome=ExecutionAttemptOutcome.FAILED, started_at=1000, ended_at=1001)],
        finished_at=1001,
    )

    assert PROJECTOR.project_tool_execution(execution, {}) == PROJECTOR.project_tool_execution(execution, {})


def test_projection_does_not_mutate_the_execution():
    execution = _execution(
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(content=[TextContent(text="3")]),
        attempts=[ExecutionAttempt(outcome=ExecutionAttemptOutcome.COMPLETED, started_at=1000, ended_at=1001)],
        finished_at=1001,
    )
    snapshot = execution.model_copy(deep=True)

    PROJECTOR.project_tool_execution(execution, {})

    assert execution == snapshot


# ── customization: subclass override points ────────────────────────────────────


def test_subclass_can_change_status_only_wording_via_class_attribute():
    class TersePlacebo(ConversationProjector):
        STATUS_ONLY_OUTPUTS: ClassVar[dict[ExecutionStatus, str]] = {
            **ConversationProjector.STATUS_ONLY_OUTPUTS,
            ExecutionStatus.REJECTED: "The user declined this tool call.",
        }

    execution = _execution(
        status=ExecutionStatus.REJECTED,
        approval_status=ApprovalStatus.REJECTED,
        finished_at=1001,
    )

    assert TersePlacebo().project_tool_execution(execution, {}) == ToolMessage(
        tool_call_id="tc1",
        content=[TextBlock(text="The user declined this tool call.")],
        is_error=True,
    )


def test_subclass_can_change_the_awaiting_approval_wording():
    class Terse(ConversationProjector):
        AWAITING_APPROVAL_OUTPUT: ClassVar[str] = "[waiting on you]"

    execution = _execution(
        status=ExecutionStatus.PENDING,
        approval_status=ApprovalStatus.PENDING,
        approval_decisions=[
            ApprovalDecision(decision=ApprovalOption.PENDING, created_at=1000),
        ],
    )

    assert Terse().project_tool_execution(execution, {}) == ToolMessage(
        tool_call_id="tc1",
        content=[TextBlock(text="[waiting on you]")],
        is_error=False,
    )


def test_subclass_can_change_the_awaiting_result_wording():
    class Terse(ConversationProjector):
        AWAITING_RESULT_OUTPUT: ClassVar[str] = "[still out]"

    execution = _execution(
        status=ExecutionStatus.AWAITING_RESULT,
        approval_status=ApprovalStatus.ALLOWED,
        attempts=[ExecutionAttempt(outcome=ExecutionAttemptOutcome.DEFERRED, started_at=1000, ended_at=1000)],
    )

    assert Terse().project_tool_execution(execution, {}) == ToolMessage(
        tool_call_id="tc1",
        content=[TextBlock(text="[still out]")],
        is_error=False,
    )


def test_the_placeholder_is_replaced_in_place_once_the_gate_resolves():
    # The 0008 headline pair: the SAME path projects the placeholder while
    # gated and the real result once the execution completed — the projection
    # is a pure function of the path, and the one projected tool message that
    # is not final is replaced at the same position.
    call = ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2})
    gated = {
        "u1": UserMessage(id="u1", created_at=1000, parts=[TextContent(text="Add 1 and 2")]),
        "ts": TurnStart(id="ts", created_at=1000),
        "a1": AssistantMessage(
            id="a1",
            created_at=1000,
            parts=[call],
            llm_config=MODEL,
            stop_reason="tool_use",
        ),
        "te1": _execution(
            status=ExecutionStatus.PENDING,
            approval_status=ApprovalStatus.PENDING,
            approval_decisions=[
                ApprovalDecision(decision=ApprovalOption.PENDING, created_at=1000),
            ],
        ),
        "um": UserMessage(id="um", created_at=1001, parts=[TextContent(text="wait, what are you adding?")]),
    }
    nodes = ["u1", "ts", "a1", "te1", "um"]

    assert PROJECTOR.project(nodes, gated) == [
        LucaUserMessage(content=[TextBlock(text="Add 1 and 2")]),
        LucaAssistantMessage(
            content=[LucaToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2})],
            provider="p",
            model="m",
        ),
        ToolMessage(
            tool_call_id="tc1",
            content=[
                TextBlock(text="[tool execution is awaiting approval — it has not run, and this is not its result]")
            ],
            is_error=False,
        ),
        LucaUserMessage(content=[TextBlock(text="wait, what are you adding?")]),
    ]

    resolved = {
        **gated,
        "te1": _execution(
            status=ExecutionStatus.COMPLETED,
            approval_status=ApprovalStatus.ALLOWED,
            approval_decisions=[
                ApprovalDecision(decision=ApprovalOption.PENDING, created_at=1000),
                ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1002),
            ],
            result=ExecutionResult(content=[TextContent(text="3")]),
            attempts=[ExecutionAttempt(outcome=ExecutionAttemptOutcome.COMPLETED, started_at=1002, ended_at=1002)],
            finished_at=1002,
        ),
    }

    assert PROJECTOR.project(nodes, resolved) == [
        LucaUserMessage(content=[TextBlock(text="Add 1 and 2")]),
        LucaAssistantMessage(
            content=[LucaToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2})],
            provider="p",
            model="m",
        ),
        ToolMessage(
            tool_call_id="tc1",
            content=[TextBlock(text="3")],
            is_error=False,
        ),
        LucaUserMessage(content=[TextBlock(text="wait, what are you adding?")]),
    ]


def test_subclass_can_replace_project_tool_execution_wholesale():
    class Redacting(ConversationProjector):
        def project_tool_execution(self, entry: ToolExecution, entries) -> ToolMessage:
            return ToolMessage(
                tool_call_id=entry.tool_call_id,
                content=[TextBlock(text="[redacted]")],
                is_error=False,
            )

    execution = _execution(
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(content=[TextContent(text="secret")]),
        attempts=[ExecutionAttempt(outcome=ExecutionAttemptOutcome.COMPLETED, started_at=1000, ended_at=1001)],
        finished_at=1001,
    )
    conversation = Conversation(id="c1", nodes=["te1"], created_at=1000, updated_at=1000)

    assert Redacting().project(conversation.nodes, {"te1": execution}) == [
        ToolMessage(
            tool_call_id="tc1",
            content=[TextBlock(text="[redacted]")],
            is_error=False,
        ),
    ]


def test_subclass_can_change_the_cancelled_turn_marker():
    class Localized(ConversationProjector):
        CANCELLED_TURN_MARKER = "[Solicitud interrumpida por el usuario]"

    finish = TurnFinish(
        id="tf",
        created_at=1000,
        outcome=TurnOutcome.CANCELLED,
    )

    assert Localized().project_turn_finish(finish, {}) == LucaUserMessage(
        content=[TextBlock(text="[Solicitud interrumpida por el usuario]")],
    )


# ── pruned entries: replacement content, original role and correlation ────────


def test_pruned_tool_execution_projects_replacement_with_original_correlation():
    # the original execution stays in the store; the PATH holds the pruned
    # node — projection resolves the referent for its tool_call_id only, and
    # the replacement marker is neutral content (is_error False)
    entries = {
        "u1": UserMessage(id="u1", created_at=1000, parts=[TextContent(text="Add")]),
        "ts": TurnStart(id="ts", created_at=1001),
        "a1": AssistantMessage(
            id="a1",
            created_at=1002,
            parts=[ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2})],
            llm_config=MODEL,
            stop_reason="tool_use",
        ),
        "te1": ToolExecution(
            id="te1",
            conversation_id="c1",
            created_at=1003,
            tool_call_id="tc1",
            raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
            tool_spec=spec("add"),
            status=ExecutionStatus.COMPLETED,
            result=ExecutionResult(content=[TextContent(text="3")]),
            approval_status=ApprovalStatus.ALLOWED,
            attempts=[ExecutionAttempt(outcome=ExecutionAttemptOutcome.COMPLETED, started_at=1003, ended_at=1003)],
            finished_at=1003,
        ),
        "p1": PrunedEntry(
            id="p1",
            created_at=2000,
            pruned_entry_type="tool_execution",
            pruned_entry_id="te1",
            content=[TextContent(text="[tool output has been pruned to reduce context]")],
        ),
        "a2": AssistantMessage(
            id="a2",
            created_at=1004,
            parts=[TextContent(text="The answer is 3.")],
            llm_config=MODEL,
            stop_reason="stop",
        ),
        "tf": TurnFinish(id="tf", created_at=1005),
    }
    conversation = Conversation(
        id="c1",
        nodes=["u1", "ts", "a1", "p1", "a2", "tf"],  # p1 replaced te1
        created_at=1000,
        updated_at=2000,
    )

    assert PROJECTOR.project(conversation.nodes, entries) == [
        LucaUserMessage(content=[TextBlock(text="Add")]),
        LucaAssistantMessage(
            content=[
                LucaToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
            ],
            provider="p",
            model="m",
        ),
        ToolMessage(
            tool_call_id="tc1",
            content=[TextBlock(text="[tool output has been pruned to reduce context]")],
            is_error=False,
        ),
        LucaAssistantMessage(content=[TextBlock(text="The answer is 3.")], provider="p", model="m"),
    ]


def test_pruned_user_message_projects_with_the_user_role():
    entries = {
        "u1": UserMessage(
            id="u1",
            created_at=1000,
            parts=[TextContent(text="a very long prompt")],
        ),
    }
    pruned = PrunedEntry(
        id="p1",
        created_at=2000,
        pruned_entry_type="user",
        pruned_entry_id="u1",
        content=[TextContent(text="[pruned]")],
    )

    assert PROJECTOR.project_pruned(pruned, entries) == LucaUserMessage(
        content=[TextBlock(text="[pruned]")],
    )


def test_pruned_assistant_message_projects_with_the_assistant_role():
    entries = {
        "a1": AssistantMessage(
            id="a1",
            created_at=1000,
            parts=[TextContent(text="a very long answer")],
            llm_config=MODEL,
            stop_reason="stop",
        ),
    }
    pruned = PrunedEntry(
        id="p1",
        created_at=2000,
        pruned_entry_type="assistant",
        pruned_entry_id="a1",
        content=[TextContent(text="[pruned]")],
    )

    assert PROJECTOR.project_pruned(pruned, entries) == LucaAssistantMessage(
        content=[TextBlock(text="[pruned]")],
        provider="p",
        model="m",
    )


def test_pruned_entry_with_a_missing_referent_fails_loudly():
    pruned = PrunedEntry(
        id="p1",
        created_at=2000,
        pruned_entry_type="tool_execution",
        pruned_entry_id="ghost",
        content=[TextContent(text="[pruned]")],
    )

    with pytest.raises(ProjectionError, match="ghost"):
        PROJECTOR.project_pruned(pruned, {})


def test_pruned_entry_with_a_mismatched_type_fails_loudly():
    entries = {
        "u1": UserMessage(id="u1", created_at=1000, parts=[TextContent(text="hi")]),
    }
    pruned = PrunedEntry(
        id="p1",
        created_at=2000,
        pruned_entry_type="tool_execution",  # the referent is 'user'
        pruned_entry_id="u1",
        content=[TextContent(text="[pruned]")],
    )

    with pytest.raises(ProjectionError, match="pruned_entry_type"):
        PROJECTOR.project_pruned(pruned, entries)


def test_pruned_entry_with_an_unprojectable_source_fails_loudly():
    entries = {"ts": TurnStart(id="ts", created_at=1000)}
    pruned = PrunedEntry(
        id="p1",
        created_at=2000,
        pruned_entry_type="turn_start",
        pruned_entry_id="ts",
        content=[TextContent(text="[pruned]")],
    )

    with pytest.raises(ProjectionError, match="no pruned projection"):
        PROJECTOR.project_pruned(pruned, entries)


# ── event presentation helper ─────────────────────────────────────────────────


def test_tool_message_text_uses_string_content_directly():
    assert (
        tool_message_text(
            ToolMessage(tool_call_id="tc1", content="plain output"),
        )
        == "plain output"
    )


def test_tool_message_text_concatenates_text_blocks_in_order():
    assert (
        tool_message_text(
            ToolMessage(
                tool_call_id="tc1",
                content=[TextBlock(text="first "), TextBlock(text="second")],
            ),
        )
        == "first second"
    )


# ── image parts ────────────────────────────────────────────────────────────────


def test_user_message_projects_image_and_text_parts_in_order():
    entries = {
        "u1": UserMessage(
            id="u1",
            created_at=1,
            parts=[
                ImageContent(
                    source=MediaBase64(data="aGk=", media_type="image/png"),
                    metadata={"name": "receipt.jpg"},
                ),
                TextContent(text="how much did I tip?"),
            ],
        ),
    }
    conversation = Conversation(id="c1", nodes=["u1"], created_at=1, updated_at=1)

    assert PROJECTOR.project(conversation.nodes, entries) == [
        LucaUserMessage(
            content=[
                LucaImageBlock(
                    source=LucaMediaBase64(data="aGk=", media_type="image/png"),
                ),
                TextBlock(text="how much did I tip?"),
            ],
        ),
    ]


def test_image_url_source_projects_to_a_media_url():
    part = ImageContent(
        source=MediaURL(url="https://example.com/a.png", media_type="image/png"),
    )

    assert PROJECTOR._image_block(part) == LucaImageBlock(
        source=LucaMediaURL(url="https://example.com/a.png", media_type="image/png"),
    )


def test_image_file_id_source_projects_to_a_media_file_id():
    part = ImageContent(source=MediaFileId(file_id="file_123"))

    assert PROJECTOR._image_block(part) == LucaImageBlock(
        source=LucaMediaFileId(file_id="file_123", media_type=None),
    )


def test_image_metadata_is_not_projected():
    source = MediaBase64(data="aGk=", media_type="image/png")

    assert PROJECTOR._image_block(
        ImageContent(source=source, metadata={"name": "receipt.jpg"}),
    ) == PROJECTOR._image_block(ImageContent(source=source))


def test_unknown_content_type_still_fails_loudly():
    with pytest.raises(ProjectionError, match="ThinkingContent"):
        PROJECTOR._content_block(ThinkingContent(thinking="hmm"))


def test_subclass_can_rewrite_image_media_only():
    class Uploading(ConversationProjector):
        def _image_block(self, part):
            return LucaImageBlock(source=LucaMediaFileId(file_id="uploaded_1"))

    entries = {
        "u1": UserMessage(
            id="u1",
            created_at=1,
            parts=[
                ImageContent(
                    source=MediaBase64(data="aGk=", media_type="image/png"),
                ),
                TextContent(text="what is this?"),
            ],
        ),
    }
    conversation = Conversation(id="c1", nodes=["u1"], created_at=1, updated_at=1)

    assert Uploading().project(conversation.nodes, entries) == [
        LucaUserMessage(
            content=[
                LucaImageBlock(source=LucaMediaFileId(file_id="uploaded_1")),
                TextBlock(text="what is this?"),
            ],
        ),
    ]


# ── images in tool results ─────────────────────────────────────────────────────


def test_completed_execution_projects_image_result_content():
    entries = {
        "te1": ToolExecution(
            id="te1",
            conversation_id="c1",
            created_at=1,
            tool_call_id="tc1",
            raw_tool_call=ToolCall(id="tc1", name="read", arguments={}),
            status=ExecutionStatus.COMPLETED,
            result=ExecutionResult(
                content=[
                    ImageContent(
                        source=MediaBase64(data="aGk=", media_type="image/png"),
                        metadata={"name": "shot.png"},
                    ),
                    TextContent(text="shot.png"),
                ]
            ),
            attempts=[ExecutionAttempt(outcome=ExecutionAttemptOutcome.COMPLETED, started_at=1, ended_at=1)],
            finished_at=1,
        ),
    }

    assert PROJECTOR.project_tool_execution(entries["te1"], entries) == ToolMessage(
        tool_call_id="tc1",
        content=[
            LucaImageBlock(source=LucaMediaBase64(data="aGk=", media_type="image/png")),
            TextBlock(text="shot.png"),
        ],
    )


def test_tool_message_text_marks_an_image_rather_than_dropping_it():
    message = ToolMessage(
        tool_call_id="tc1",
        content=[
            LucaImageBlock(source=LucaMediaBase64(data="aGk=", media_type="image/png")),
            TextBlock(text=" shot.png"),
        ],
    )

    assert tool_message_text(message) == "[image] shot.png"


def test_pruned_entry_can_carry_an_image_replacement():
    entries = {
        "te1": ToolExecution(
            id="te1",
            conversation_id="c1",
            created_at=1,
            tool_call_id="tc1",
            raw_tool_call=ToolCall(id="tc1", name="read", arguments={}),
            status=ExecutionStatus.COMPLETED,
            result=ExecutionResult(content=[TextContent(text="original")]),
            attempts=[ExecutionAttempt(outcome=ExecutionAttemptOutcome.COMPLETED, started_at=1, ended_at=1)],
            finished_at=1,
        ),
        "p1": PrunedEntry(
            id="p1",
            created_at=2,
            pruned_entry_type="tool_execution",
            pruned_entry_id="te1",
            content=[
                ImageContent(source=MediaBase64(data="aGk=", media_type="image/png")),
            ],
        ),
    }

    assert PROJECTOR.project_pruned(entries["p1"], entries) == ToolMessage(
        tool_call_id="tc1",
        content=[
            LucaImageBlock(source=LucaMediaBase64(data="aGk=", media_type="image/png")),
        ],
    )


# ── thinking signatures ────────────────────────────────────────────────────────


def test_assistant_thinking_projects_with_its_signature():
    # the signature is what lets a provider accept a replayed thinking block
    entries = {
        "a1": AssistantMessage(
            id="a1",
            created_at=1,
            parts=[
                ThinkingContent(thinking="reasoning", signature="sig-abc"),
                TextContent(text="the answer"),
            ],
            llm_config=MODEL,
            stop_reason="stop",
        ),
    }
    conversation = Conversation(id="c1", nodes=["a1"], created_at=1, updated_at=1)

    assert PROJECTOR.project(conversation.nodes, entries) == [
        LucaAssistantMessage(
            content=[
                ThinkingBlock(text="reasoning", signature="sig-abc"),
                TextBlock(text="the answer"),
            ],
            provider="p",
            model="m",
        ),
    ]


def test_a_redacted_thinking_part_projects_as_redacted():
    entries = {
        "a1": AssistantMessage(
            id="a1",
            created_at=1,
            parts=[ThinkingContent(thinking="", signature="enc", redacted=True)],
            llm_config=MODEL,
            stop_reason="stop",
        ),
    }
    conversation = Conversation(id="c1", nodes=["a1"], created_at=1, updated_at=1)

    assert PROJECTOR.project(conversation.nodes, entries) == [
        LucaAssistantMessage(
            content=[
                ThinkingBlock(text="", signature="enc", redacted=True),
            ],
            provider="p",
            model="m",
        ),
    ]


def test_assistant_thinking_projects_with_its_reasoning_item_id():
    # `id` is the other half of an OpenAI reasoning item's identity; the
    # transport needs both to replay it.
    entries = {
        "a1": AssistantMessage(
            id="a1",
            created_at=1,
            parts=[ThinkingContent(thinking="reasoning", id="rs_1", signature="enc-1")],
            llm_config=MODEL,
            stop_reason="stop",
        ),
    }
    conversation = Conversation(id="c1", nodes=["a1"], created_at=1, updated_at=1)

    assert PROJECTOR.project(conversation.nodes, entries) == [
        LucaAssistantMessage(
            content=[ThinkingBlock(text="reasoning", id="rs_1", signature="enc-1")],
            provider="p",
            model="m",
        ),
    ]


def test_the_producing_model_is_projected_as_provenance():
    # Load-bearing, not decorative: the transport compares this against the
    # model being called to decide whether an opaque attestation may be
    # replayed. Two entries produced by different models keep their own.
    entries = {
        "a1": AssistantMessage(
            id="a1",
            created_at=1,
            parts=[ThinkingContent(thinking="first", signature="sig-1")],
            llm_config=LLMConfig(model="gpt-5.4", provider="openai"),
            stop_reason="stop",
        ),
        "a2": AssistantMessage(
            id="a2",
            created_at=2,
            parts=[TextContent(text="second")],
            llm_config=LLMConfig(model="claude-sonnet-5", provider="anthropic"),
            stop_reason="stop",
        ),
    }
    conversation = Conversation(id="c1", nodes=["a1", "a2"], created_at=1, updated_at=2)

    assert PROJECTOR.project(conversation.nodes, entries) == [
        LucaAssistantMessage(
            content=[ThinkingBlock(text="first", signature="sig-1")],
            provider="openai",
            model="gpt-5.4",
        ),
        LucaAssistantMessage(
            content=[TextBlock(text="second")],
            provider="anthropic",
            model="claude-sonnet-5",
        ),
    ]


def test_a_pruned_private_execution_projects_nothing():
    # the ToolMessage channel is closed to a private execution by protocol —
    # no ToolCall for it exists in any assistant message, and a runner-minted
    # correlation id on the wire would be rejected by every provider. Pruning
    # one (a subagent-result execution is the natural target) simply drops its
    # contribution.
    entries = {
        "te1": ToolExecution(
            id="te1",
            conversation_id="c1",
            created_at=1000,
            tool_call_id="rtc1",
            raw_tool_call=ToolCall(id="rtc1", name="create_conversation_result"),
            tool_spec=spec("create_conversation_result", is_private=True),
            status=ExecutionStatus.COMPLETED,
            result=ExecutionResult(content=[TextContent(text="A done")]),
            attempts=[ExecutionAttempt(outcome=ExecutionAttemptOutcome.COMPLETED, started_at=1000, ended_at=1000)],
            finished_at=1000,
        ),
        "p1": PrunedEntry(
            id="p1",
            created_at=2000,
            pruned_entry_type="tool_execution",
            pruned_entry_id="te1",
            content=[TextContent(text="[tool output has been pruned to reduce context]")],
        ),
    }

    assert PROJECTOR.project_pruned(entries["p1"], entries) is None
    assert PROJECTOR.project(["p1"], entries) == []


# ── file parts ────────────────────────────────────────────────────────────────


def test_a_file_part_projects_to_a_client_file_block():
    part = FileContent(
        source=MediaBase64(data="JVBERi0=", media_type="application/pdf"),
        name="report.pdf",
        metadata={"mention": {"path": "/tmp/report.pdf"}},
    )

    assert PROJECTOR._file_block(part) == LucaFileBlock(
        source=LucaMediaBase64(data="JVBERi0=", media_type="application/pdf"),
        name="report.pdf",
    )


def test_a_file_name_travels_but_metadata_does_not():
    # `name` is the one thing the model is told about the original file, and
    # two providers require it; `metadata` follows ImageContent's rule
    source = MediaBase64(data="JVBERi0=", media_type="application/pdf")

    assert PROJECTOR._file_block(
        FileContent(source=source, name="report.pdf", metadata={"path": "/tmp/report.pdf"}),
    ) == PROJECTOR._file_block(FileContent(source=source, name="report.pdf"))


def test_every_file_source_projects():
    assert (
        PROJECTOR._file_block(FileContent(source=MediaURL(url="https://example.com/a.pdf"))),
        PROJECTOR._file_block(FileContent(source=MediaFileId(file_id="file_123"))),
    ) == (
        LucaFileBlock(source=LucaMediaURL(url="https://example.com/a.pdf", media_type=None), name=None),
        LucaFileBlock(source=LucaMediaFileId(file_id="file_123", media_type=None), name=None),
    )


def test_a_user_message_projects_file_and_text_parts_in_order():
    entries = {
        "u1": UserMessage(
            id="u1",
            created_at=1,
            parts=[
                FileContent(
                    source=MediaBase64(data="JVBERi0=", media_type="application/pdf"),
                    name="report.pdf",
                ),
                TextContent(text="what does it conclude?"),
            ],
        ),
    }
    conversation = Conversation(id="c1", nodes=["u1"], created_at=1, updated_at=1)

    assert PROJECTOR.project(conversation.nodes, entries) == [
        LucaUserMessage(
            content=[
                LucaFileBlock(
                    source=LucaMediaBase64(data="JVBERi0=", media_type="application/pdf"),
                    name="report.pdf",
                ),
                TextBlock(text="what does it conclude?"),
            ],
        ),
    ]


def test_subclass_can_rewrite_file_media_only():
    # `_file_block` is documented as an override point, so it has to actually
    # be one: the same upload-and-swap policy `_image_block` supports
    class Uploading(ConversationProjector):
        def _file_block(self, part):
            return LucaFileBlock(source=LucaMediaFileId(file_id="uploaded_1"), name=part.name)

    entries = {
        "u1": UserMessage(
            id="u1",
            created_at=1,
            parts=[
                FileContent(
                    source=MediaBase64(data="JVBERi0=", media_type="application/pdf"),
                    name="report.pdf",
                ),
                TextContent(text="what does it say?"),
            ],
        ),
    }
    conversation = Conversation(id="c1", nodes=["u1"], created_at=1, updated_at=1)

    assert Uploading().project(conversation.nodes, entries) == [
        LucaUserMessage(
            content=[
                LucaFileBlock(source=LucaMediaFileId(file_id="uploaded_1"), name="report.pdf"),
                TextBlock(text="what does it say?"),
            ],
        ),
    ]


def test_the_shared_media_mapping_is_one_override_for_every_part_type():
    # `_media_source` exists so a rewrite policy is written once rather than
    # once per content part — proving it means one override moving BOTH
    class Proxying(ConversationProjector):
        def _media_source(self, source):
            return LucaMediaURL(url="https://cdn.example.com/proxied", media_type=None)

    projector = Proxying()

    assert (
        projector._image_block(ImageContent(source=MediaBase64(data="aGk=", media_type="image/png"))),
        projector._audio_block(AudioContent(source=MediaBase64(data="SUQz", media_type="audio/mpeg"))),
        projector._file_block(FileContent(source=MediaBase64(data="JVBERi0=", media_type="application/pdf"))),
    ) == (
        LucaImageBlock(source=LucaMediaURL(url="https://cdn.example.com/proxied", media_type=None)),
        LucaAudioBlock(source=LucaMediaURL(url="https://cdn.example.com/proxied", media_type=None)),
        LucaFileBlock(source=LucaMediaURL(url="https://cdn.example.com/proxied", media_type=None), name=None),
    )


# ── audio parts ───────────────────────────────────────────────────────────────


def test_an_audio_part_projects_to_a_client_audio_block():
    part = AudioContent(
        source=MediaBase64(data="SUQz", media_type="audio/mpeg"),
        metadata={"mention": {"path": "/tmp/clip.mp3"}},
    )

    assert PROJECTOR._audio_block(part) == LucaAudioBlock(
        source=LucaMediaBase64(data="SUQz", media_type="audio/mpeg"),
    )


def test_audio_metadata_does_not_travel():
    # same rule as ImageContent: `metadata` is the application's, not the wire's
    source = MediaBase64(data="SUQz", media_type="audio/mpeg")

    assert PROJECTOR._audio_block(
        AudioContent(source=source, metadata={"name": "clip.mp3"}),
    ) == PROJECTOR._audio_block(AudioContent(source=source))


def test_every_audio_source_projects():
    # the core carries all three; whether the WIRE takes them is the
    # transport's call, and chat completions refuses everything but base64
    assert (
        PROJECTOR._audio_block(AudioContent(source=MediaURL(url="https://example.com/a.mp3"))),
        PROJECTOR._audio_block(AudioContent(source=MediaFileId(file_id="file_123"))),
    ) == (
        LucaAudioBlock(source=LucaMediaURL(url="https://example.com/a.mp3", media_type=None)),
        LucaAudioBlock(source=LucaMediaFileId(file_id="file_123", media_type=None)),
    )


def test_a_user_message_projects_audio_and_text_parts_in_order():
    entries = {
        "u1": UserMessage(
            id="u1",
            created_at=1,
            parts=[
                AudioContent(source=MediaBase64(data="SUQz", media_type="audio/mpeg")),
                TextContent(text="what is said here?"),
            ],
        ),
    }
    conversation = Conversation(id="c1", nodes=["u1"], created_at=1, updated_at=1)

    assert PROJECTOR.project(conversation.nodes, entries) == [
        LucaUserMessage(
            content=[
                LucaAudioBlock(source=LucaMediaBase64(data="SUQz", media_type="audio/mpeg")),
                TextBlock(text="what is said here?"),
            ],
        ),
    ]
