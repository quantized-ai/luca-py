"""Declarative tests for the ConversationProjector: a KNOWN conversation
projects to a KNOWN list of canonical client messages, and a KNOWN
ToolExecution projects to a KNOWN ToolMessage. Hardcoded state in, full
expected object out; no loop, no runner, no provider.

Unit boundary: deterministic derivation from durable entries. Ground truths:
entry constructors produce correct objects; client DTOs compare by value.
Projection errors fail loudly (ProjectionError) — never silent omissions or
synthetic fallbacks.
"""

import pytest

from luca.agent.core.exceptions import ProjectionError
from luca.agent.core.models import (
    ApprovalDecision,
    ApprovalOption,
    ApprovalStatus,
    AssistantMessage,
    CancelRequested,
    CompactionEntry,
    Conversation,
    ExecutionResult,
    ExecutionStatus,
    ImageBase64,
    ImageContent,
    ImageFileId,
    ImageURL,
    LLMConfig,
    PrunedEntry,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolExecution,
    ToolExecutionError,
    ToolSpec,
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
    MediaBase64,
    MediaFileId,
    MediaURL,
    TextBlock,
    ThinkingBlock,
    ToolMessage,
)
from luca.client.types import AssistantMessage as LucaAssistantMessage
from luca.client.types import ImageBlock as LucaImageBlock
from luca.client.types import ToolCall as LucaToolCall
from luca.client.types import UserMessage as LucaUserMessage

PROJECTOR = ConversationProjector()

MODEL = LLMConfig(model="m", provider="p")


# ── top-level traversal ────────────────────────────────────────────────────────


def test_single_user_message():
    entries = {
        "u1": UserMessage(id="u1", created_at=1000, parts=[TextContent(text="Hello")]),
    }
    conversation = Conversation(id="c1", nodes=["u1"], created_at=1000, updated_at=1000)

    assert PROJECTOR.project(conversation, entries) == [
        LucaUserMessage(content=[TextBlock(text="Hello")]),
    ]


def test_turn_markers_are_dropped():
    entries = {
        "u1": UserMessage(id="u1", created_at=1000, parts=[TextContent(text="Hi")]),
        "ts": TurnStart(id="ts", created_at=1001),
        "a1": AssistantMessage(
            id="a1", created_at=1002,
            parts=[TextContent(text="Hey there")],
            llm_config=MODEL,
            stop_reason="stop",
        ),
        "tf": TurnFinish(id="tf", created_at=1003),
    }
    conversation = Conversation(
        id="c1", nodes=["u1", "ts", "a1", "tf"], created_at=1000, updated_at=1003,
    )

    assert PROJECTOR.project(conversation, entries) == [
        LucaUserMessage(content=[TextBlock(text="Hi")]),
        LucaAssistantMessage(content=[TextBlock(text="Hey there")]),
    ]


def test_full_tool_call_turn():
    entries = {
        "u1": UserMessage(
            id="u1", created_at=1000, parts=[TextContent(text="Add 1 and 2")],
        ),
        "ts": TurnStart(id="ts", created_at=1001),
        "a1": AssistantMessage(
            id="a1", created_at=1002,
            parts=[
                ThinkingContent(thinking="Use the add tool."),
                ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
            ],
            llm_config=MODEL,
            stop_reason="tool_use",
        ),
        "te1": ToolExecution(
            id="te1", created_at=1003,
            tool_call_id="tc1",
            raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
            tool_spec=ToolSpec(name="add"),
            status=ExecutionStatus.COMPLETED,
            result=ExecutionResult(content=[TextContent(text="3")]),
            approval_status=ApprovalStatus.ALLOWED,
            approval_decisions=[
                ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1003),
            ],
            started_at=1003, ended_at=1003,
        ),
        "a2": AssistantMessage(
            id="a2", created_at=1004,
            parts=[TextContent(text="The answer is 3.")],
            llm_config=MODEL,
            stop_reason="stop",
        ),
        "tf": TurnFinish(id="tf", created_at=1005),
    }
    conversation = Conversation(
        id="c1", nodes=["u1", "ts", "a1", "te1", "a2", "tf"], created_at=1000, updated_at=1005,
    )

    assert PROJECTOR.project(conversation, entries) == [
        LucaUserMessage(content=[TextBlock(text="Add 1 and 2")]),
        LucaAssistantMessage(content=[
            ThinkingBlock(text="Use the add tool."),
            LucaToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        ]),
        ToolMessage(tool_call_id="tc1", content=[TextBlock(text="3")]),
        LucaAssistantMessage(content=[TextBlock(text="The answer is 3.")]),
    ]


def test_compaction_renders_as_synthetic_user_message():
    entries = {
        "c1": CompactionEntry(
            id="c1", created_at=2000,
            summary="## Goal\nFix the failing test suite.",
            summarized=["u1", "a1"],
        ),
        "u2": UserMessage(
            id="u2", created_at=2001, parts=[TextContent(text="What next?")],
        ),
    }
    conversation = Conversation(id="c1", nodes=["c1", "u2"], created_at=2000, updated_at=2001)

    assert PROJECTOR.project(conversation, entries) == [
        LucaUserMessage(content=[TextBlock(text="## Goal\nFix the failing test suite.")]),
        LucaUserMessage(content=[TextBlock(text="What next?")]),
    ]


def test_cancelled_turn_finish_projects_as_interrupted_marker():
    # end-to-end cancel → post → projection order: the cancelled bracket
    # renders [assistant, synthetic user marker], CancelRequested is dropped,
    # and the follow-up user message lands after the marker.
    entries = {
        "u1": UserMessage(id="u1", created_at=1000, parts=[TextContent(text="Go")]),
        "ts": TurnStart(id="ts", created_at=1001),
        "a1": AssistantMessage(
            id="a1", created_at=1002,
            parts=[TextContent(text="Working on it…")],
            llm_config=MODEL,
            stop_reason="stop",
        ),
        "cr": CancelRequested(id="cr", created_at=1003),
        "tf": TurnFinish(
            id="tf", created_at=1004,
            outcome=TurnOutcome.CANCELLED,
        ),
        "u2": UserMessage(
            id="u2", created_at=1005, parts=[TextContent(text="Try again")],
        ),
    }
    conversation = Conversation(
        id="c1", nodes=["u1", "ts", "a1", "cr", "tf", "u2"], created_at=1000, updated_at=1005,
    )

    assert PROJECTOR.project(conversation, entries) == [
        LucaUserMessage(content=[TextBlock(text="Go")]),
        LucaAssistantMessage(content=[TextBlock(text="Working on it…")]),
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
            id="a1", created_at=1002,
            parts=[ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2})],
            llm_config=MODEL,
            stop_reason="tool_use",
        ),
        "te1": ToolExecution(
            id="te1", created_at=1003,
            tool_call_id="tc1",
            raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
            tool_spec=ToolSpec(name="add"),
            status=ExecutionStatus.COMPLETED,
            result=ExecutionResult(content=[TextContent(text="3")]),
            approval_status=ApprovalStatus.ALLOWED,
            started_at=1003, ended_at=1003,
        ),
        "tf": TurnFinish(
            id="tf", created_at=1004,
            outcome=TurnOutcome.TIMED_OUT, error="client timeout",
        ),
    }
    conversation = Conversation(
        id="c1", nodes=["u1", "ts", "a1", "te1", "tf"], created_at=1000, updated_at=1004,
    )

    assert PROJECTOR.project(conversation, entries) == [
        LucaUserMessage(content=[TextBlock(text="Add")]),
        LucaAssistantMessage(content=[
            LucaToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        ]),
        ToolMessage(tool_call_id="tc1", content=[TextBlock(text="3")]),
    ]


def test_errored_turn_finish_is_dropped():
    entries = {
        "u1": UserMessage(id="u1", created_at=1000, parts=[TextContent(text="Hi")]),
        "ts": TurnStart(id="ts", created_at=1001),
        "tf": TurnFinish(
            id="tf", created_at=1002,
            outcome=TurnOutcome.ERRORED, error="provider 500",
        ),
    }
    conversation = Conversation(
        id="c1", nodes=["u1", "ts", "tf"], created_at=1000, updated_at=1002,
    )

    assert PROJECTOR.project(conversation, entries) == [
        LucaUserMessage(content=[TextBlock(text="Hi")]),
    ]


def test_missing_entry_id_fails_loudly():
    conversation = Conversation(id="c1", nodes=["ghost"], created_at=1000, updated_at=1000)

    with pytest.raises(ProjectionError, match="ghost"):
        PROJECTOR.project(conversation, {})


# ── tool-execution projection: one method, every status ───────────────────────


def _execution(**overrides) -> ToolExecution:
    base = dict(
        id="te1", created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        tool_spec=ToolSpec(name="add"),
    )
    return ToolExecution(**{**base, **overrides})


def test_completed_projects_result_content_and_preserves_is_error():
    execution = _execution(
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(
            content=[TextContent(text="The requested file does not exist.")],
            metadata={"path": "/tmp/missing.txt"},
            is_error=True,
        ),
        started_at=1000, ended_at=1001,
    )

    # is_error=True is the TOOL's verdict; the execution is still COMPLETED
    # and must NOT be reinterpreted as a framework failure.
    assert PROJECTOR.project_tool_execution(execution, {}) == ToolMessage(
        tool_call_id="tc1",
        content=[TextBlock(text="The requested file does not exist.")],
        is_error=True,
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
        ended_at=1001,
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
            details={
                "errors": [
                    {"type": "missing", "loc": ["b"], "msg": "Field required", "input": {"a": 1}},
                ],
            },
        ),
        ended_at=1001,
    )

    assert PROJECTOR.project_tool_execution(execution, {}) == ToolMessage(
        tool_call_id="tc1",
        content=[TextBlock(
            text=(
                "Arguments for tool 'add' are invalid.\n"
                '[{"type": "missing", "loc": ["b"], "msg": "Field required", '
                '"input": {"a": 1}}]'
            ),
        )],
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
        started_at=1000, ended_at=1001,
    )

    assert PROJECTOR.project_tool_execution(execution, {}) == ToolMessage(
        tool_call_id="tc1",
        content=[TextBlock(
            text=(
                "Tool execution failed: ConnectionError: "
                "Connection to api.example.com was closed."
            ),
        )],
        is_error=True,
    )


def test_status_only_terminals_project_their_placeholders():
    rejected = _execution(
        status=ExecutionStatus.REJECTED,
        approval_status=ApprovalStatus.REJECTED, ended_at=1001,
    )
    cancelled = _execution(
        status=ExecutionStatus.CANCELLED,
        cancel_signalled_at=1001, ended_at=1001,
    )
    interrupted = _execution(
        status=ExecutionStatus.INTERRUPTED,
        started_at=1000, ended_at=1001,
    )
    timed_out = _execution(
        status=ExecutionStatus.TIMED_OUT,
        started_at=1000, ended_at=1001,
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


def test_pending_execution_is_not_projectable():
    execution = _execution(status=ExecutionStatus.PENDING)

    with pytest.raises(ProjectionError, match="pending"):
        PROJECTOR.project_tool_execution(execution, {})


def test_running_execution_is_not_projectable():
    execution = _execution(status=ExecutionStatus.RUNNING, started_at=1000)

    with pytest.raises(ProjectionError, match="running"):
        PROJECTOR.project_tool_execution(execution, {})


def test_completed_without_result_fails_loudly():
    execution = _execution(status=ExecutionStatus.COMPLETED, ended_at=1001)

    with pytest.raises(ProjectionError, match="no ExecutionResult"):
        PROJECTOR.project_tool_execution(execution, {})


def test_projection_preserves_tool_call_id_correlation():
    execution = _execution(
        tool_call_id="toolu_0abc",
        raw_tool_call=ToolCall(id="toolu_0abc", name="add", arguments={}),
        status=ExecutionStatus.TIMED_OUT,
        started_at=1000, ended_at=1001,
    )

    assert PROJECTOR.project_tool_execution(execution, {}).tool_call_id == "toolu_0abc"


def test_projection_is_deterministic_for_the_same_execution():
    execution = _execution(
        status=ExecutionStatus.FAILED,
        error=ToolExecutionError(error_type="ValueError", error_message="kaboom"),
        started_at=1000, ended_at=1001,
    )

    assert PROJECTOR.project_tool_execution(execution, {}) == \
        PROJECTOR.project_tool_execution(execution, {})


def test_projection_does_not_mutate_the_execution():
    execution = _execution(
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(content=[TextContent(text="3")]),
        started_at=1000, ended_at=1001,
    )
    snapshot = execution.model_copy(deep=True)

    PROJECTOR.project_tool_execution(execution, {})

    assert execution == snapshot


# ── customization: subclass override points ────────────────────────────────────


def test_subclass_can_change_status_only_wording_via_class_attribute():
    class TersePlacebo(ConversationProjector):
        STATUS_ONLY_OUTPUTS = {
            **ConversationProjector.STATUS_ONLY_OUTPUTS,
            ExecutionStatus.REJECTED: "The user declined this tool call.",
        }

    execution = _execution(
        status=ExecutionStatus.REJECTED,
        approval_status=ApprovalStatus.REJECTED, ended_at=1001,
    )

    assert TersePlacebo().project_tool_execution(execution, {}) == ToolMessage(
        tool_call_id="tc1",
        content=[TextBlock(text="The user declined this tool call.")],
        is_error=True,
    )


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
        started_at=1000, ended_at=1001,
    )
    conversation = Conversation(id="c1", nodes=["te1"], created_at=1000, updated_at=1000)

    assert Redacting().project(conversation, {"te1": execution}) == [
        ToolMessage(
            tool_call_id="tc1", content=[TextBlock(text="[redacted]")],
            is_error=False,
        ),
    ]


def test_subclass_can_change_the_cancelled_turn_marker():
    class Localized(ConversationProjector):
        CANCELLED_TURN_MARKER = "[Solicitud interrumpida por el usuario]"

    finish = TurnFinish(
        id="tf", created_at=1000, outcome=TurnOutcome.CANCELLED,
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
            id="a1", created_at=1002,
            parts=[ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2})],
            llm_config=MODEL,
            stop_reason="tool_use",
        ),
        "te1": ToolExecution(
            id="te1", created_at=1003,
            tool_call_id="tc1",
            raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
            tool_spec=ToolSpec(name="add"),
            status=ExecutionStatus.COMPLETED,
            result=ExecutionResult(content=[TextContent(text="3")]),
            approval_status=ApprovalStatus.ALLOWED,
            started_at=1003, ended_at=1003,
        ),
        "p1": PrunedEntry(
            id="p1", created_at=2000,
            pruned_entry_type="tool_execution",
            pruned_entry_id="te1",
            content=[TextContent(text="[tool output has been pruned to reduce context]")],
        ),
        "a2": AssistantMessage(
            id="a2", created_at=1004,
            parts=[TextContent(text="The answer is 3.")],
            llm_config=MODEL,
            stop_reason="stop",
        ),
        "tf": TurnFinish(id="tf", created_at=1005),
    }
    conversation = Conversation(
        id="c1", nodes=["u1", "ts", "a1", "p1", "a2", "tf"],  # p1 replaced te1
        created_at=1000, updated_at=2000,
    )

    assert PROJECTOR.project(conversation, entries) == [
        LucaUserMessage(content=[TextBlock(text="Add")]),
        LucaAssistantMessage(content=[
            LucaToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        ]),
        ToolMessage(
            tool_call_id="tc1",
            content=[TextBlock(text="[tool output has been pruned to reduce context]")],
            is_error=False,
        ),
        LucaAssistantMessage(content=[TextBlock(text="The answer is 3.")]),
    ]


def test_pruned_user_message_projects_with_the_user_role():
    entries = {
        "u1": UserMessage(
            id="u1", created_at=1000, parts=[TextContent(text="a very long prompt")],
        ),
    }
    pruned = PrunedEntry(
        id="p1", created_at=2000,
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
            id="a1", created_at=1000,
            parts=[TextContent(text="a very long answer")],
            llm_config=MODEL, stop_reason="stop",
        ),
    }
    pruned = PrunedEntry(
        id="p1", created_at=2000,
        pruned_entry_type="assistant",
        pruned_entry_id="a1",
        content=[TextContent(text="[pruned]")],
    )

    assert PROJECTOR.project_pruned(pruned, entries) == LucaAssistantMessage(
        content=[TextBlock(text="[pruned]")],
    )


def test_pruned_entry_with_a_missing_referent_fails_loudly():
    pruned = PrunedEntry(
        id="p1", created_at=2000,
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
        id="p1", created_at=2000,
        pruned_entry_type="tool_execution",  # the referent is 'user'
        pruned_entry_id="u1",
        content=[TextContent(text="[pruned]")],
    )

    with pytest.raises(ProjectionError, match="pruned_entry_type"):
        PROJECTOR.project_pruned(pruned, entries)


def test_pruned_entry_with_an_unprojectable_source_fails_loudly():
    entries = {"ts": TurnStart(id="ts", created_at=1000)}
    pruned = PrunedEntry(
        id="p1", created_at=2000,
        pruned_entry_type="turn_start",
        pruned_entry_id="ts",
        content=[TextContent(text="[pruned]")],
    )

    with pytest.raises(ProjectionError, match="no pruned projection"):
        PROJECTOR.project_pruned(pruned, entries)


# ── event presentation helper ─────────────────────────────────────────────────


def test_tool_message_text_uses_string_content_directly():
    assert tool_message_text(
        ToolMessage(tool_call_id="tc1", content="plain output"),
    ) == "plain output"


def test_tool_message_text_concatenates_text_blocks_in_order():
    assert tool_message_text(
        ToolMessage(
            tool_call_id="tc1",
            content=[TextBlock(text="first "), TextBlock(text="second")],
        ),
    ) == "first second"


# ── image parts ────────────────────────────────────────────────────────────────


def test_user_message_projects_image_and_text_parts_in_order():
    entries = {
        "u1": UserMessage(
            id="u1", created_at=1,
            parts=[
                ImageContent(
                    source=ImageBase64(data="aGk=", media_type="image/png"),
                    name="receipt.jpg",
                ),
                TextContent(text="how much did I tip?"),
            ],
        ),
    }
    conversation = Conversation(id="c1", nodes=["u1"], created_at=1, updated_at=1)

    assert PROJECTOR.project(conversation, entries) == [
        LucaUserMessage(
            content=[
                LucaImageBlock(
                    source=MediaBase64(data="aGk=", media_type="image/png"),
                ),
                TextBlock(text="how much did I tip?"),
            ],
        ),
    ]


def test_image_url_source_projects_to_a_media_url():
    part = ImageContent(
        source=ImageURL(url="https://example.com/a.png", media_type="image/png"),
    )

    assert PROJECTOR._image_block(part) == LucaImageBlock(
        source=MediaURL(url="https://example.com/a.png", media_type="image/png"),
    )


def test_image_file_id_source_projects_to_a_media_file_id():
    part = ImageContent(source=ImageFileId(file_id="file_123"))

    assert PROJECTOR._image_block(part) == LucaImageBlock(
        source=MediaFileId(file_id="file_123", media_type=None),
    )


def test_image_name_is_not_projected():
    source = ImageBase64(data="aGk=", media_type="image/png")

    assert PROJECTOR._image_block(
        ImageContent(source=source, name="receipt.jpg"),
    ) == PROJECTOR._image_block(ImageContent(source=source))


def test_unknown_content_type_still_fails_loudly():
    with pytest.raises(ProjectionError, match="ThinkingContent"):
        PROJECTOR._content_block(ThinkingContent(thinking="hmm"))


def test_subclass_can_rewrite_image_media_only():
    class Uploading(ConversationProjector):
        def _image_block(self, part):
            return LucaImageBlock(source=MediaFileId(file_id="uploaded_1"))

    entries = {
        "u1": UserMessage(
            id="u1", created_at=1,
            parts=[
                ImageContent(
                    source=ImageBase64(data="aGk=", media_type="image/png"),
                ),
                TextContent(text="what is this?"),
            ],
        ),
    }
    conversation = Conversation(id="c1", nodes=["u1"], created_at=1, updated_at=1)

    assert Uploading().project(conversation, entries) == [
        LucaUserMessage(
            content=[
                LucaImageBlock(source=MediaFileId(file_id="uploaded_1")),
                TextBlock(text="what is this?"),
            ],
        ),
    ]
