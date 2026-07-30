"""Declarative tests for the default ContextManager: a KNOWN entry in, a
KNOWN context-token count (or PrunedEntry template) out. Pure strategy checks
— no provider, and no runner except where the append ORDERING is the subject.

Every method takes the live `AgentSession` first. The shipped policy never
reads it — it is there so an application's tokenizer can see the active model —
so every default-policy assertion below passes the same inert `SESSION` and
the count depends on the entry alone. `PerModelTokenizer` is the counterpart:
one entry, two sessions, two counts.

The default policy under test: one token per CHARS_PER_TOKEN (4) characters
of the entry's OWN model-facing content — a user message owns its content; an
assistant message its text + thinking + tool-call requests (name + JSON
arguments, counted here and never again on the execution); a tool execution
only its outcome (result content, else the structured error message; 0 while
nonterminal); a compaction its summary `parts` (0 until they land, images
included); a pruned entry its replacement
content; markers own nothing. Pruning supports terminal tool executions only
and returns an identity-less TEMPLATE — stamping ids/clocks belongs to the
persisting door, never to a strategy. `process_tool_output` is an identity
pass-through that hands a subclass the IN-TRANSITION execution (still RUNNING,
no result attached) to select a per-tool policy on. The compaction pair is the
last section: the default declines and raises, and a subclass owns the plan.
"""

from typing import ClassVar

import pytest

from luca.agent.core.compaction import CompactionPlan, UsageCounters
from luca.agent.core.context_manager import (
    PRUNED_TOOL_OUTPUT_MARKER,
    ContextManager,
)
from luca.agent.core.exceptions import AgentError
from luca.agent.core.models import (
    AgentSession,
    ApprovalStatus,
    AssistantMessage,
    CancelRequested,
    CompactionEntry,
    CompactionSource,
    Conversation,
    ConversationStatus,
    Entry,
    ExecutionResult,
    ExecutionStatus,
    ImageBase64,
    ImageContent,
    PrunedEntry,
    SessionConfig,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolExecution,
    ToolExecutionError,
    TurnFinish,
    TurnStart,
    UserMessage,
)
from tests.agent.scenarios import (
    ADD_SPEC,
    CHEAP,
    MODEL,
    MULTIPLY_SPEC,
    DeterministicRunner,
    make_session,
)

CM = ContextManager()

# The session handed to every default-policy call: an empty one, because the
# shipped character estimate reads nothing off it.
SESSION = make_session(
    id="s_ctx",
    active_conversation=Conversation(
        id="c1",
        nodes=[],
        created_at=500,
        updated_at=500,
        status=ConversationStatus.IDLE,
    ),
    session_config=SessionConfig(llm_config=MODEL),
)

# The same session on a different model — the second half of the pair a
# model-aware manager counts differently.
CHEAP_SESSION = SESSION.model_copy(
    deep=True,
    update={
        "id": "s_ctx_cheap",
        "session_config": SessionConfig(llm_config=CHEAP),
    },
)

# A tool call mid-flight, exactly as `process_tool_output` receives it: RUNNING,
# dispatched, no result attached yet.
RUNNING_ADD = ToolExecution(
    id="te1",
    created_at=500,
    tool_call_id="tc1",
    raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
    tool_spec=ADD_SPEC,
    status=ExecutionStatus.RUNNING,
    approval_status=ApprovalStatus.ALLOWED,
    started_at=500,
    updated_at=500,
)

SUMMARY = [TextContent(text="the story so far")]

# A compaction exactly as `compact()` receives it: the deep copy the runner
# hands over, `parts` still None.
SCHEDULED_COMPACTION = CompactionEntry(
    id="c1",
    created_at=500,
    source=CompactionSource.USER,
    started_at=500,
)

RUNNING_MULTIPLY = ToolExecution(
    id="te2",
    created_at=500,
    tool_call_id="tc2",
    raw_tool_call=ToolCall(id="tc2", name="multiply", arguments={"a": 3, "b": 4}),
    tool_spec=MULTIPLY_SPEC,
    status=ExecutionStatus.RUNNING,
    approval_status=ApprovalStatus.ALLOWED,
    started_at=500,
    updated_at=500,
)


# ── calculate_context: per-type ownership ─────────────────────────────────────


def test_user_message_counts_its_content():
    entry = UserMessage(
        id="u1",
        created_at=1000,
        parts=[TextContent(text="Add 1 and 2")],  # 11 chars
    )

    assert CM.calculate_context(SESSION, entry) == 2


def test_assistant_message_counts_text_thinking_and_tool_call_requests():
    entry = AssistantMessage(
        id="a1",
        created_at=1000,
        parts=[
            ThinkingContent(thinking="Let me add."),  # 11 chars
            TextContent(text="Adding now."),  # 11 chars
            # "add" (3) + '{"a": 1, "b": 2}' (16) = 19 chars
            ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        ],
        llm_config=MODEL,
        stop_reason="tool_use",
    )

    assert CM.calculate_context(SESSION, entry) == 10  # (11 + 11 + 19) // 4


def test_completed_execution_counts_only_its_result_content():
    # the tool-call REQUEST was counted on the assistant message; the
    # execution owns only the model-facing outcome
    entry = ToolExecution(
        id="te1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(
            content=[TextContent(text="the answer is 3.")],  # 16 chars
        ),
        started_at=1000,
        ended_at=1000,
    )

    assert CM.calculate_context(SESSION, entry) == 4


def test_failed_execution_counts_its_structured_error_message():
    entry = ToolExecution(
        id="te1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={}),
        status=ExecutionStatus.FAILED,
        error=ToolExecutionError(
            error_type="ValueError",
            error_message="kaboom kaboom",  # 13 chars
            details={"phase": "execution"},
        ),
        started_at=1000,
        ended_at=1000,
    )

    assert CM.calculate_context(SESSION, entry) == 3


def test_nonterminal_execution_counts_zero():
    entry = ToolExecution(
        id="te1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        status=ExecutionStatus.PENDING,
    )

    assert CM.calculate_context(SESSION, entry) == 0


def test_resultless_errorless_terminal_execution_counts_zero():
    # CANCELLED / INTERRUPTED / TIMED_OUT are complete lifecycle facts with no
    # stored outcome content of their own
    entry = ToolExecution(
        id="te1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={}),
        status=ExecutionStatus.CANCELLED,
        ended_at=1000,
        cancel_signalled_at=1000,
    )

    assert CM.calculate_context(SESSION, entry) == 0


def test_compaction_counts_its_summary_parts():
    entry = CompactionEntry(
        id="c1",
        created_at=1000,
        source=CompactionSource.POLICY,
        # 35 chars
        parts=[TextContent(text="## Goal\nFix the failing test suite.")],
        compacted_nodes=["u1", "a1"],
    )

    assert CM.calculate_context(SESSION, entry) == 8


def test_a_compaction_with_no_parts_yet_counts_zero():
    # scheduled or running: the runner recalculates when `parts` land
    entry = CompactionEntry(
        id="c1",
        created_at=1000,
        source=CompactionSource.USER,
    )

    assert CM.calculate_context(SESSION, entry) == 0


def test_a_compaction_with_empty_parts_counts_zero():
    entry = CompactionEntry(
        id="c1",
        created_at=1000,
        source=CompactionSource.USER,
        parts=[],
    )

    assert CM.calculate_context(SESSION, entry) == 0


def test_an_image_carrying_summary_counts_text_plus_the_image_constant():
    entry = CompactionEntry(
        id="c1",
        created_at=1000,
        source=CompactionSource.POLICY,
        parts=[
            TextContent(text="## Goal\nFix the failing test suite."),  # 35
            ImageContent(source=ImageBase64(data="aGk=", media_type="image/png")),
        ],
        compacted_nodes=["u1", "a1"],
    )

    assert CM.calculate_context(SESSION, entry) == 1_008


def test_pruned_entry_counts_its_replacement_content():
    entry = PrunedEntry(
        id="p1",
        created_at=1000,
        pruned_entry_type="tool_execution",
        pruned_entry_id="te1",
        content=[TextContent(text=PRUNED_TOOL_OUTPUT_MARKER)],  # 46 chars
    )

    assert CM.calculate_context(SESSION, entry) == 11


def test_markers_count_zero():
    assert CM.calculate_context(SESSION, TurnStart(id="ts", created_at=1000)) == 0
    assert CM.calculate_context(SESSION, TurnFinish(id="tf", created_at=1000)) == 0
    assert CM.calculate_context(SESSION, CancelRequested(id="cr", created_at=1000)) == 0


# ── calculate_context: what the session argument is for ───────────────────────


class PerModelTokenizer(ContextManager):
    """A model-aware manager — the reason `calculate_context` takes the
    session. It charges by `session_config.llm_config`, so one entry has as
    many counts as the session has models."""

    CHARS_PER_TOKEN_BY_MODEL: ClassVar[dict[str, int]] = {
        "test-model": 4,
        "cheap-model": 1,
    }

    def calculate_context(self, session: AgentSession, entry: Entry) -> int:
        model = session.session_config.llm_config.model
        return len(self._model_facing_text(entry)) // self.CHARS_PER_TOKEN_BY_MODEL[model]


def test_a_model_aware_manager_counts_one_entry_differently_per_session_model():
    entry = UserMessage(
        id="u1",
        created_at=1000,
        parts=[TextContent(text="Add 1 and 2")],  # 11 chars
    )

    counts = (
        PerModelTokenizer().calculate_context(SESSION, entry),
        PerModelTokenizer().calculate_context(CHEAP_SESSION, entry),
    )

    assert counts == (2, 11)  # 11 // 4 on test-model, 11 // 1 on cheap-model


class MembershipProbe(ContextManager):
    """Records `(entry id, already a member of session.entries)` per call."""

    def __init__(self) -> None:
        self.observed: list[tuple[str | None, bool]] = []

    def calculate_context(self, session: AgentSession, entry: Entry) -> int:
        self.observed.append((entry.id, entry.id in session.entries))
        return super().calculate_context(session, entry)


def test_calculate_context_runs_before_the_appended_entry_joins_the_store():
    # on an append it runs inside the ledger's build callback: the entry
    # already has its id, but the store does not have the entry yet — a
    # manager that looks itself up there raises KeyError on every append
    probe = MembershipProbe()
    runner = DeterministicRunner(
        make_session(
            id="s_append",
            active_conversation=Conversation(
                id="c1",
                nodes=[],
                created_at=500,
                updated_at=500,
                status=ConversationStatus.IDLE,
            ),
            session_config=SessionConfig(llm_config=MODEL),
        ),
        context_manager=probe,
        ids=["u1"],
        now=1000,
    )

    runner.post_message("Add 1 and 2")

    assert probe.observed == [("u1", False)]


# ── prune_entry ────────────────────────────────────────────────────────────────


def test_prune_entry_builds_a_template_for_a_terminal_execution():
    entry = ToolExecution(
        id="te1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(content=[TextContent(text="3")]),
        started_at=1000,
        ended_at=1000,
    )

    assert CM.prune_entry(SESSION, entry) == PrunedEntry(
        id=None,  # uncommitted — the persisting door stamps identity
        parent_id=None,
        created_at=None,
        pruned_entry_type="tool_execution",
        pruned_entry_id="te1",
        content=[TextContent(text=PRUNED_TOOL_OUTPUT_MARKER)],
    )


def test_prune_entry_rejects_a_non_execution_entry():
    entry = UserMessage(
        id="u1",
        created_at=1000,
        parts=[TextContent(text="hi")],
    )

    with pytest.raises(AgentError, match="only tool executions"):
        CM.prune_entry(SESSION, entry)


def test_prune_entry_rejects_a_nonterminal_execution():
    entry = ToolExecution(
        id="te1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={}),
        status=ExecutionStatus.RUNNING,
        started_at=1000,
    )

    with pytest.raises(AgentError, match="nonterminal"):
        CM.prune_entry(SESSION, entry)


# ── process_tool_output ────────────────────────────────────────────────────────


def test_process_tool_output_is_an_identity_pass_through():
    result = ExecutionResult(
        content=[TextContent(text="raw output")],
        metadata={"exit_code": 0},
        is_error=False,
    )

    assert CM.process_tool_output(SESSION, RUNNING_ADD, result) is result


# ── subclass override points ───────────────────────────────────────────────────


def test_subclass_can_change_the_chars_per_token_ratio():
    class Coarse(ContextManager):
        CHARS_PER_TOKEN = 2

    entry = UserMessage(
        id="u1",
        created_at=1000,
        parts=[TextContent(text="Add 1 and 2")],  # 11 chars
    )

    assert Coarse().calculate_context(SESSION, entry) == 5


def test_subclass_can_change_the_pruned_output_marker():
    class Terse(ContextManager):
        PRUNED_TOOL_OUTPUT_MARKER = "[gone]"

    entry = ToolExecution(
        id="te1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={}),
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(content=[TextContent(text="3")]),
        started_at=1000,
        ended_at=1000,
    )

    assert Terse().prune_entry(SESSION, entry).content == [TextContent(text="[gone]")]


class TruncatingAdd(ContextManager):
    """Truncates `add`'s output and passes every other tool through — the
    per-tool policy the in-transition `execution` argument exists to make
    expressible."""

    def process_tool_output(
        self,
        session: AgentSession,
        execution: ToolExecution,
        result: ExecutionResult,
    ) -> ExecutionResult:
        if execution.tool_spec.name != "add":
            return result
        return ExecutionResult(
            content=[TextContent(text="[truncated]")],
            metadata={"truncated": True},
            is_error=result.is_error,
        )


def test_subclass_can_select_the_output_policy_on_the_in_transition_execution():
    result = ExecutionResult(content=[TextContent(text="the answer is 3.")])

    assert TruncatingAdd().process_tool_output(SESSION, RUNNING_ADD, result) == ExecutionResult(
        content=[TextContent(text="[truncated]")],
        metadata={"truncated": True},
        is_error=False,
    )


def test_a_tool_selecting_subclass_leaves_the_tools_it_does_not_match_alone():
    result = ExecutionResult(content=[TextContent(text="12")])

    assert TruncatingAdd().process_tool_output(SESSION, RUNNING_MULTIPLY, result) is result


# ── image content ──────────────────────────────────────────────────────────────


def test_image_only_message_counts_the_flat_image_constant():
    entry = UserMessage(
        id="u1",
        created_at=1000,
        parts=[
            ImageContent(source=ImageBase64(data="aGk=", media_type="image/png")),
        ],
    )

    assert CM.calculate_context(SESSION, entry) == 1_000


def test_images_add_to_the_text_estimate():
    entry = UserMessage(
        id="u1",
        created_at=1000,
        parts=[
            ImageContent(source=ImageBase64(data="aGk=", media_type="image/png")),
            TextContent(text="Add 1 and 2"),  # 11 chars
            ImageContent(source=ImageBase64(data="aGk=", media_type="image/jpeg")),
        ],
    )

    assert CM.calculate_context(SESSION, entry) == 2_002  # 2 * 1000 + 11 // 4


def test_subclass_can_change_the_per_image_cost():
    class Free(ContextManager):
        IMAGE_TOKENS = 0

    entry = UserMessage(
        id="u1",
        created_at=1000,
        parts=[
            ImageContent(source=ImageBase64(data="aGk=", media_type="image/png")),
            TextContent(text="Add 1 and 2"),  # 11 chars
        ],
    )

    assert Free().calculate_context(SESSION, entry) == 2


def test_non_user_entries_have_no_media_contribution():
    entry = AssistantMessage(
        id="a1",
        created_at=1000,
        parts=[TextContent(text="Add 1 and 2")],  # 11 chars
        llm_config=MODEL,
        stop_reason="stop",
    )

    assert CM._media_tokens(entry) == 0
    assert CM.calculate_context(SESSION, entry) == 2


def test_tool_result_images_are_counted():
    entry = ToolExecution(
        id="te1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="read", arguments={}),
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(
            content=[
                ImageContent(source=ImageBase64(data="aGk=", media_type="image/png")),
                TextContent(text="the answer is 3."),  # 16 chars
            ]
        ),
        started_at=1000,
        ended_at=1000,
    )

    assert CM.calculate_context(SESSION, entry) == 1_004  # IMAGE_TOKENS + 16 // 4


def test_pruned_entry_images_are_counted():
    entry = PrunedEntry(
        id="p1",
        created_at=1000,
        pruned_entry_type="tool_execution",
        pruned_entry_id="te1",
        content=[
            ImageContent(source=ImageBase64(data="aGk=", media_type="image/png")),
        ],
    )

    assert CM.calculate_context(SESSION, entry) == 1_000


# ── the compaction pair ───────────────────────────────────────────────────────


def test_the_default_manager_never_compacts():
    assert CM.should_compact(SESSION) is False


async def test_the_default_manager_has_no_compact_implementation():
    with pytest.raises(NotImplementedError):
        await CM.compact(SESSION, ("u1", "c1"), SCHEDULED_COMPACTION)


async def test_a_subclass_implements_compact_over_session_nodes_and_entry():
    class Folding(ContextManager):
        def should_compact(self, session):
            return True

        async def compact(self, session, nodes, entry):
            return CompactionPlan(
                entry=entry.model_copy(update={"parts": SUMMARY, "llm_config": CHEAP}),
                nodes=[entry.id],
                usage=UsageCounters(input=100, output=20, total_tokens=120),
            )

    manager = Folding()

    plan = await manager.compact(
        SESSION,
        ("u1", "c1"),
        SCHEDULED_COMPACTION.model_copy(deep=True),
    )

    assert manager.should_compact(SESSION) is True
    assert plan == CompactionPlan(
        entry=SCHEDULED_COMPACTION.model_copy(update={"parts": SUMMARY, "llm_config": CHEAP}),
        nodes=["c1"],
        usage=UsageCounters(input=100, output=20, total_tokens=120),
    )
