"""Declarative tests for the default ContextManager: a KNOWN entry in, a
KNOWN context-token count (or PrunedEntry template) out. Pure strategy checks
— no runner, no session, no provider.

The default policy under test: one token per CHARS_PER_TOKEN (4) characters
of the entry's OWN model-facing content — a user message owns its content; an
assistant message its text + thinking + tool-call requests (name + JSON
arguments, counted here and never again on the execution); a tool execution
only its outcome (result content, else the structured error message; 0 while
nonterminal); a compaction its summary; a pruned entry its replacement
content; markers own nothing. Pruning supports terminal tool executions only
and returns an identity-less TEMPLATE — stamping ids/clocks belongs to the
persisting door, never to a strategy.
"""

import pytest

from luca.agent.core.context_manager import (
    PRUNED_TOOL_OUTPUT_MARKER,
    ContextManager,
)
from luca.agent.core.exceptions import AgentError
from luca.agent.core.models import (
    AssistantMessage,
    CancelRequested,
    CompactionEntry,
    ExecutionResult,
    ExecutionStatus,
    ImageBase64,
    ImageContent,
    LLMConfig,
    PrunedEntry,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolExecution,
    ToolExecutionError,
    TurnFinish,
    TurnStart,
    UserMessage,
)

CM = ContextManager()

MODEL = LLMConfig(model="m", provider="p")


# ── calculate_context: per-type ownership ─────────────────────────────────────


def test_user_message_counts_its_content():
    entry = UserMessage(
        id="u1", created_at=1000,
        parts=[TextContent(text="Add 1 and 2")],  # 11 chars
    )

    assert CM.calculate_context(entry) == 2


def test_assistant_message_counts_text_thinking_and_tool_call_requests():
    entry = AssistantMessage(
        id="a1", created_at=1000,
        parts=[
            ThinkingContent(thinking="Let me add."),  # 11 chars
            TextContent(text="Adding now."),  # 11 chars
            # "add" (3) + '{"a": 1, "b": 2}' (16) = 19 chars
            ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        ],
        llm_config=MODEL, stop_reason="tool_use",
    )

    assert CM.calculate_context(entry) == 10  # (11 + 11 + 19) // 4


def test_completed_execution_counts_only_its_result_content():
    # the tool-call REQUEST was counted on the assistant message; the
    # execution owns only the model-facing outcome
    entry = ToolExecution(
        id="te1", created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(
            content=[TextContent(text="the answer is 3.")],  # 16 chars
        ),
        started_at=1000, ended_at=1000,
    )

    assert CM.calculate_context(entry) == 4


def test_failed_execution_counts_its_structured_error_message():
    entry = ToolExecution(
        id="te1", created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={}),
        status=ExecutionStatus.FAILED,
        error=ToolExecutionError(
            error_type="ValueError",
            error_message="kaboom kaboom",  # 13 chars
        ),
        started_at=1000, ended_at=1000,
    )

    assert CM.calculate_context(entry) == 3


def test_nonterminal_execution_counts_zero():
    entry = ToolExecution(
        id="te1", created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        status=ExecutionStatus.PENDING,
    )

    assert CM.calculate_context(entry) == 0


def test_resultless_errorless_terminal_execution_counts_zero():
    # CANCELLED / INTERRUPTED / TIMED_OUT are complete lifecycle facts with no
    # stored outcome content of their own
    entry = ToolExecution(
        id="te1", created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={}),
        status=ExecutionStatus.CANCELLED,
        ended_at=1000, cancel_signalled_at=1000,
    )

    assert CM.calculate_context(entry) == 0


def test_compaction_counts_its_summary():
    entry = CompactionEntry(
        id="c1", created_at=1000,
        summary="## Goal\nFix the failing test suite.",  # 35 chars
        summarized=["u1", "a1"],
    )

    assert CM.calculate_context(entry) == 8


def test_pruned_entry_counts_its_replacement_content():
    entry = PrunedEntry(
        id="p1", created_at=1000,
        pruned_entry_type="tool_execution",
        pruned_entry_id="te1",
        content=[TextContent(text=PRUNED_TOOL_OUTPUT_MARKER)],  # 46 chars
    )

    assert CM.calculate_context(entry) == 11


def test_markers_count_zero():
    assert CM.calculate_context(TurnStart(id="ts", created_at=1000)) == 0
    assert CM.calculate_context(TurnFinish(id="tf", created_at=1000)) == 0
    assert CM.calculate_context(CancelRequested(id="cr", created_at=1000)) == 0


# ── prune_entry ────────────────────────────────────────────────────────────────


def test_prune_entry_builds_a_template_for_a_terminal_execution():
    entry = ToolExecution(
        id="te1", created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(content=[TextContent(text="3")]),
        started_at=1000, ended_at=1000,
    )

    assert CM.prune_entry(entry) == PrunedEntry(
        id="",  # placeholder identity — the persisting door stamps the real one
        parent_id=None,
        created_at=0,
        pruned_entry_type="tool_execution",
        pruned_entry_id="te1",
        content=[TextContent(text=PRUNED_TOOL_OUTPUT_MARKER)],
    )


def test_prune_entry_rejects_a_non_execution_entry():
    entry = UserMessage(
        id="u1", created_at=1000, parts=[TextContent(text="hi")],
    )

    with pytest.raises(AgentError, match="only tool executions"):
        CM.prune_entry(entry)


def test_prune_entry_rejects_a_nonterminal_execution():
    entry = ToolExecution(
        id="te1", created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={}),
        status=ExecutionStatus.RUNNING,
        started_at=1000,
    )

    with pytest.raises(AgentError, match="nonterminal"):
        CM.prune_entry(entry)


# ── process_tool_output ────────────────────────────────────────────────────────


def test_process_tool_output_is_an_identity_pass_through():
    result = ExecutionResult(
        content=[TextContent(text="raw output")],
        metadata={"exit_code": 0},
        is_error=False,
    )

    assert CM.process_tool_output(result) is result


# ── subclass override points ───────────────────────────────────────────────────


def test_subclass_can_change_the_chars_per_token_ratio():
    class Coarse(ContextManager):
        CHARS_PER_TOKEN = 2

    entry = UserMessage(
        id="u1", created_at=1000,
        parts=[TextContent(text="Add 1 and 2")],  # 11 chars
    )

    assert Coarse().calculate_context(entry) == 5


def test_subclass_can_change_the_pruned_output_marker():
    class Terse(ContextManager):
        PRUNED_TOOL_OUTPUT_MARKER = "[gone]"

    entry = ToolExecution(
        id="te1", created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={}),
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(content=[TextContent(text="3")]),
        started_at=1000, ended_at=1000,
    )

    assert Terse().prune_entry(entry).content == [TextContent(text="[gone]")]


# ── image content ──────────────────────────────────────────────────────────────


def test_image_only_message_counts_the_flat_image_constant():
    entry = UserMessage(
        id="u1", created_at=1000,
        parts=[
            ImageContent(source=ImageBase64(data="aGk=", media_type="image/png")),
        ],
    )

    assert CM.calculate_context(entry) == 1_000


def test_images_add_to_the_text_estimate():
    entry = UserMessage(
        id="u1", created_at=1000,
        parts=[
            ImageContent(source=ImageBase64(data="aGk=", media_type="image/png")),
            TextContent(text="Add 1 and 2"),  # 11 chars
            ImageContent(source=ImageBase64(data="aGk=", media_type="image/jpeg")),
        ],
    )

    assert CM.calculate_context(entry) == 2_002  # 2 * 1000 + 11 // 4


def test_subclass_can_change_the_per_image_cost():
    class Free(ContextManager):
        IMAGE_TOKENS = 0

    entry = UserMessage(
        id="u1", created_at=1000,
        parts=[
            ImageContent(source=ImageBase64(data="aGk=", media_type="image/png")),
            TextContent(text="Add 1 and 2"),  # 11 chars
        ],
    )

    assert Free().calculate_context(entry) == 2


def test_non_user_entries_have_no_media_contribution():
    entry = AssistantMessage(
        id="a1", created_at=1000,
        parts=[TextContent(text="Add 1 and 2")],  # 11 chars
        llm_config=MODEL,
        stop_reason="stop",
    )

    assert CM._media_tokens(entry) == 0
    assert CM.calculate_context(entry) == 2
