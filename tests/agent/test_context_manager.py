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
    AudioContent,
    CancelRequested,
    CompactionEntry,
    CompactionSource,
    Entry,
    ExecutionAttempt,
    ExecutionAttemptOutcome,
    ExecutionResult,
    ExecutionStatus,
    FileContent,
    ImageContent,
    LLMConfig,
    MediaBase64,
    PrivateProviderContent,
    PrunedEntry,
    SessionConfig,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolExecution,
    ToolExecutionError,
    TurnFinish,
    TurnStart,
    URLCitation,
    UserMessage,
    WebFetchContent,
    WebPageContent,
    WebSearchContent,
)
from tests.agent.scenarios import (
    ADD_SPEC,
    CHEAP,
    MODEL,
    MULTIPLY_SPEC,
    DeterministicRunner,
    conversation,
    make_session,
)

CM = ContextManager()

# The session handed to every default-policy call: an empty one, because the
# shipped character estimate reads nothing off it.
SESSION = make_session(
    id="s_ctx",
    conversations={
        "c1": conversation(
            "c1",
            [],
            created_at=500,
            updated_at=500,
        )
    },
    main_conversation_id="c1",
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
    conversation_id="c1",
    created_at=500,
    tool_call_id="tc1",
    raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
    tool_spec=ADD_SPEC,
    status=ExecutionStatus.RUNNING,
    approval_status=ApprovalStatus.ALLOWED,
    attempts=[ExecutionAttempt(started_at=500)],
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
    conversation_id="c1",
    created_at=500,
    tool_call_id="tc2",
    raw_tool_call=ToolCall(id="tc2", name="multiply", arguments={"a": 3, "b": 4}),
    tool_spec=MULTIPLY_SPEC,
    status=ExecutionStatus.RUNNING,
    approval_status=ApprovalStatus.ALLOWED,
    attempts=[ExecutionAttempt(started_at=500)],
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
        conversation_id="c1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(
            content=[TextContent(text="the answer is 3.")],  # 16 chars
        ),
        attempts=[ExecutionAttempt(outcome=ExecutionAttemptOutcome.COMPLETED, started_at=1000, ended_at=1000)],
        finished_at=1000,
    )

    assert CM.calculate_context(SESSION, entry) == 4


def test_structured_content_is_not_counted_toward_context():
    # same 16 chars of model-facing content as the test above, so the same 4
    # tokens: the payload never goes on the wire and must never inflate the
    # estimate
    entry = ToolExecution(
        id="te1",
        conversation_id="c1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(
            content=[TextContent(text="the answer is 3.")],  # 16 chars
            structured_content={"answer": 3, "operands": [1, 2], "note": "a" * 500},
        ),
        attempts=[ExecutionAttempt(outcome=ExecutionAttemptOutcome.COMPLETED, started_at=1000, ended_at=1000)],
        finished_at=1000,
    )

    assert CM.calculate_context(SESSION, entry) == 4


def test_failed_execution_counts_its_structured_error_message():
    entry = ToolExecution(
        id="te1",
        conversation_id="c1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={}),
        status=ExecutionStatus.FAILED,
        error=ToolExecutionError(
            error_type="ValueError",
            error_message="kaboom kaboom",  # 13 chars
            details={"phase": "execution"},
        ),
        attempts=[ExecutionAttempt(outcome=ExecutionAttemptOutcome.FAILED, started_at=1000, ended_at=1000)],
        finished_at=1000,
    )

    assert CM.calculate_context(SESSION, entry) == 3


def test_nonterminal_execution_counts_zero():
    entry = ToolExecution(
        id="te1",
        conversation_id="c1",
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
        conversation_id="c1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={}),
        status=ExecutionStatus.CANCELLED,
        finished_at=1000,
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
            ImageContent(source=MediaBase64(data="aGk=", media_type="image/png")),
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
            conversations={
                "c1": conversation(
                    "c1",
                    [],
                    created_at=500,
                    updated_at=500,
                )
            },
            main_conversation_id="c1",
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
        conversation_id="c1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(content=[TextContent(text="3")]),
        attempts=[ExecutionAttempt(outcome=ExecutionAttemptOutcome.COMPLETED, started_at=1000, ended_at=1000)],
        finished_at=1000,
    )

    assert CM.prune_entry(SESSION, entry) == PrunedEntry(
        id=None,  # uncommitted — the persisting door stamps identity
        parent_id=None,
        created_at=None,
        pruned_entry_type="tool_execution",
        pruned_entry_id="te1",
        content=[TextContent(text=PRUNED_TOOL_OUTPUT_MARKER)],
    )


def test_a_user_message_is_still_refused():
    entry = UserMessage(
        id="u1",
        created_at=1000,
        parts=[TextContent(text="hi")],
    )

    with pytest.raises(AgentError, match="only tool executions and assistant messages"):
        CM.prune_entry(SESSION, entry)


def test_prune_entry_rejects_a_nonterminal_execution():
    entry = ToolExecution(
        id="te1",
        conversation_id="c1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={}),
        status=ExecutionStatus.RUNNING,
        attempts=[ExecutionAttempt(started_at=1000)],
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
        conversation_id="c1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={}),
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(content=[TextContent(text="3")]),
        attempts=[ExecutionAttempt(outcome=ExecutionAttemptOutcome.COMPLETED, started_at=1000, ended_at=1000)],
        finished_at=1000,
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
            ImageContent(source=MediaBase64(data="aGk=", media_type="image/png")),
        ],
    )

    assert CM.calculate_context(SESSION, entry) == 1_000


def test_images_add_to_the_text_estimate():
    entry = UserMessage(
        id="u1",
        created_at=1000,
        parts=[
            ImageContent(source=MediaBase64(data="aGk=", media_type="image/png")),
            TextContent(text="Add 1 and 2"),  # 11 chars
            ImageContent(source=MediaBase64(data="aGk=", media_type="image/jpeg")),
        ],
    )

    assert CM.calculate_context(SESSION, entry) == 2_002  # 2 * 1000 + 11 // 4


def test_a_file_counts_the_flat_file_constant_on_top_of_images_and_text():
    entry = UserMessage(
        id="u1",
        created_at=1000,
        parts=[
            FileContent(source=MediaBase64(data="JVBERi0=", media_type="application/pdf"), name="a.pdf"),
            ImageContent(source=MediaBase64(data="aGk=", media_type="image/png")),
            TextContent(text="Add 1 and 2"),  # 11 chars
        ],
    )

    assert CM.calculate_context(SESSION, entry) == 6_002  # 5000 + 1000 + 11 // 4


def test_subclass_can_change_the_per_file_cost():
    class Free(ContextManager):
        FILE_TOKENS = 0

    entry = UserMessage(
        id="u1",
        created_at=1000,
        parts=[FileContent(source=MediaBase64(data="JVBERi0=", media_type="application/pdf"))],
    )

    assert Free().calculate_context(SESSION, entry) == 0


def test_audio_counts_the_flat_audio_constant_on_top_of_the_rest():
    entry = UserMessage(
        id="u1",
        created_at=1000,
        parts=[
            AudioContent(source=MediaBase64(data="SUQz", media_type="audio/mpeg")),
            FileContent(source=MediaBase64(data="JVBERi0=", media_type="application/pdf"), name="a.pdf"),
            ImageContent(source=MediaBase64(data="aGk=", media_type="image/png")),
            TextContent(text="Add 1 and 2"),  # 11 chars
        ],
    )

    assert CM.calculate_context(SESSION, entry) == 16_002  # 10000 + 5000 + 1000 + 11 // 4


def test_subclass_can_change_the_per_audio_cost():
    class Free(ContextManager):
        AUDIO_TOKENS = 0

    entry = UserMessage(
        id="u1",
        created_at=1000,
        parts=[AudioContent(source=MediaBase64(data="SUQz", media_type="audio/mpeg"))],
    )

    assert Free().calculate_context(SESSION, entry) == 0


def test_subclass_can_change_the_per_image_cost():
    class Free(ContextManager):
        IMAGE_TOKENS = 0

    entry = UserMessage(
        id="u1",
        created_at=1000,
        parts=[
            ImageContent(source=MediaBase64(data="aGk=", media_type="image/png")),
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
        conversation_id="c1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="read", arguments={}),
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(
            content=[
                ImageContent(source=MediaBase64(data="aGk=", media_type="image/png")),
                TextContent(text="the answer is 3."),  # 16 chars
            ]
        ),
        attempts=[ExecutionAttempt(outcome=ExecutionAttemptOutcome.COMPLETED, started_at=1000, ended_at=1000)],
        finished_at=1000,
    )

    assert CM.calculate_context(SESSION, entry) == 1_004  # IMAGE_TOKENS + 16 // 4


def test_pruned_entry_images_are_counted():
    entry = PrunedEntry(
        id="p1",
        created_at=1000,
        pruned_entry_type="tool_execution",
        pruned_entry_id="te1",
        content=[
            ImageContent(source=MediaBase64(data="aGk=", media_type="image/png")),
        ],
    )

    assert CM.calculate_context(SESSION, entry) == 1_000


# ── the compaction pair ───────────────────────────────────────────────────────


def test_the_default_manager_never_compacts():
    assert CM.should_compact(SESSION, "c1") is False


async def test_the_default_manager_has_no_compact_implementation():
    with pytest.raises(NotImplementedError):
        await CM.compact(SESSION, "c1", ("u1", "c1"), SCHEDULED_COMPACTION)


async def test_a_subclass_implements_compact_over_session_nodes_and_entry():
    class Folding(ContextManager):
        def should_compact(self, session, conversation_id):
            return True

        async def compact(self, session, conversation_id, nodes, entry):
            return CompactionPlan(
                entry=entry.model_copy(update={"parts": SUMMARY, "llm_config": CHEAP}),
                nodes=[entry.id],
                usage=UsageCounters(input=100, output=20, total_tokens=120),
            )

    manager = Folding()

    plan = await manager.compact(
        SESSION,
        "c1",
        ("u1", "c1"),
        SCHEDULED_COMPACTION.model_copy(deep=True),
    )

    assert manager.should_compact(SESSION, "c1") is True
    assert plan == CompactionPlan(
        entry=SCHEDULED_COMPACTION.model_copy(update={"parts": SUMMARY, "llm_config": CHEAP}),
        nodes=["c1"],
        usage=UsageCounters(input=100, output=20, total_tokens=120),
    )


# ── the effective wire view (assistant entries; D3) ──────────────────────────

# An ACTIVE Anthropic target — a REGISTERED provider, so the assistant branch
# measures through the client's effective view instead of falling back.
ANTHROPIC_SESSION = make_session(
    id="s_ctx_anthropic",
    conversations={"c1": conversation("c1", [], created_at=500, updated_at=500)},
    main_conversation_id="c1",
    session_config=SessionConfig(llm_config=LLMConfig(model="claude-sonnet-5", provider="anthropic")),
)


def test_an_anthropic_private_blob_counts_on_anthropic_and_drops_after_a_switch():
    entry = AssistantMessage(
        id="a1",
        created_at=1,
        parts=[
            PrivateProviderContent(format="anthropic.messages", data={"blob": "x" * 26}),
            TextContent(text="Answer."),
        ],
        llm_config=LLMConfig(model="claude-sonnet-5", provider="anthropic"),
        stop_reason="stop",
    )
    session = ANTHROPIC_SESSION.model_copy(deep=True)

    # on the producing target the private replays verbatim, so its JSON bytes
    # are real context: (38 dumps chars + 7 text chars) // 4
    assert CM.calculate_context(session, entry) == 11

    session.update_llm_config("openai:gpt-5.1", False)

    # a foreign target drops the private; only the text is left: 7 // 4
    assert CM.calculate_context(session, entry) == 1


def test_web_blocks_are_never_counted():
    # no transport sends the portable web blocks, so their (potentially huge)
    # payloads never reach the estimate
    entry = AssistantMessage(
        id="a1",
        created_at=1,
        parts=[
            WebSearchContent(
                queries=["apple latest quarterly results"],
                results=[WebPageContent(url="https://apple.com", title="Apple", content="snippet " * 100)],
            ),
            TextContent(text="Answer."),
        ],
        llm_config=LLMConfig(model="claude-sonnet-5", provider="anthropic"),
        stop_reason="stop",
    )

    # only the text survives the wire: 7 // 4
    assert CM.calculate_context(ANTHROPIC_SESSION, entry) == 1


def test_annotated_cited_text_measures_by_the_merged_text_rule():
    # On Anthropic the merged cited text replays through its split privates —
    # the text drops and the privates' JSON counts. On a foreign target the
    # privates drop and the merged text counts.
    entry = AssistantMessage(
        id="a1",
        created_at=1,
        parts=[
            TextContent(
                text="Hello world.",
                annotations=[URLCitation(url="https://a", title="A", start_index=0, end_index=12)],
            ),
            PrivateProviderContent(format="anthropic.messages", data={"type": "text", "text": "Hello "}),
            PrivateProviderContent(format="anthropic.messages", data={"type": "text", "text": "world."}),
        ],
        llm_config=LLMConfig(model="claude-sonnet-5", provider="anthropic"),
        stop_reason="stop",
    )
    session = ANTHROPIC_SESSION.model_copy(deep=True)

    # the two split privates' dumps (34 + 34 chars) // 4; the merged text is 0
    assert CM.calculate_context(session, entry) == 17

    session.update_llm_config("openai:gpt-5.1", False)

    # foreign target: privates gone, merged text counts: 12 // 4
    assert CM.calculate_context(session, entry) == 3


def test_plain_assistant_entries_measure_identically_to_the_naive_path():
    # count parity for non-web content: a plain text + tool-call entry
    # measures the same through the effective view (registered provider) and
    # through the naive fallback (SESSION's provider "p" is unregistered)
    entry = AssistantMessage(
        id="a1",
        created_at=1,
        parts=[
            TextContent(text="Sure — adding now."),
            ToolCall(id="tc1", name="add", arguments={"a": 1}),
        ],
        llm_config=LLMConfig(model="claude-sonnet-5", provider="anthropic"),
        stop_reason="stop",
    )

    # (18 text chars + 3 name chars + 8 argument-JSON chars) // 4, both paths
    assert (
        CM.calculate_context(ANTHROPIC_SESSION, entry),
        CM.calculate_context(SESSION, entry),
    ) == (7, 7)


def test_an_unresolvable_provider_falls_back_to_the_naive_estimate():
    # SESSION's provider "p" resolves to nothing offline — the estimate stays
    # the naive one (private blobs contribute 0 there), and nothing raises;
    # this is also what keeps the faux-driven runner suite alive
    entry = AssistantMessage(
        id="a1",
        created_at=1,
        parts=[
            PrivateProviderContent(format="anthropic.messages", data={"blob": "x" * 26}),
            TextContent(text="Answer."),
        ],
        llm_config=LLMConfig(model="claude-sonnet-5", provider="anthropic"),
        stop_reason="stop",
    )

    # the naive path ignores private payloads: 7 // 4
    assert CM.calculate_context(SESSION, entry) == 1


# ── whole-assistant-message pruning (D6) ─────────────────────────────────────

WEB_HEAVY_ENTRY = AssistantMessage(
    id="am1",
    created_at=1000,
    parts=[
        ThinkingContent(thinking="Let me check the markets.", signature="sig-1"),
        PrivateProviderContent(format="anthropic.messages", data={"type": "server_tool_use", "id": "s_1"}),
        PrivateProviderContent(
            format="anthropic.messages",
            data={"type": "web_search_tool_result", "tool_use_id": "s_1", "content": [{"encrypted": "blob" * 500}]},
        ),
        WebSearchContent(
            queries=["Apple stock price today"],
            results=[
                WebPageContent(url="https://cnn.com/a"),
                WebPageContent(url="https://www.tradingview.com/b"),
            ],
            extras={"id": "s_1"},
        ),
        WebSearchContent(queries=["NVIDIA stock price today"], results=None, extras={"id": "s_2"}),
        WebFetchContent(
            web_page=WebPageContent(url="https://apple.com/report", title="Apple report"),
            extras={"id": "f_1"},
        ),
        TextContent(
            text="Apple rose 2.8% today.",
            annotations=[URLCitation(url="https://cnn.com/a", title="A", start_index=0, end_index=22)],
        ),
    ],
    llm_config=LLMConfig(model="claude-sonnet-5", provider="anthropic"),
    stop_reason="stop",
)


def test_a_web_assistant_entry_prunes_to_summary_plus_verbatim_answer():
    template = CM.prune_entry(SESSION, WEB_HEAVY_ENTRY)

    assert template == PrunedEntry(
        id=None,  # uncommitted — the persisting door stamps identity
        parent_id=None,
        created_at=None,
        pruned_entry_type="assistant",
        pruned_entry_id="am1",
        content=[
            TextContent(
                text=(
                    'Searched the web: "Apple stock price today" (2 results: cnn.com, tradingview.com), '
                    '"NVIDIA stock price today" (result metadata not returned). '
                    'Fetched: https://apple.com/report ("Apple report"). '
                    "Answered: Apple rose 2.8% today."
                )
            )
        ],
    )


def test_a_tool_call_carrying_message_is_refused():
    entry = AssistantMessage(
        id="am2",
        created_at=1000,
        parts=[
            TextContent(text="Adding."),
            ToolCall(id="tc1", name="add", arguments={"a": 1}),
        ],
        llm_config=LLMConfig(model="claude-sonnet-5", provider="anthropic"),
        stop_reason="tool_use",
    )

    with pytest.raises(AgentError, match="client-executed tool calls"):
        CM.prune_entry(SESSION, entry)


def test_the_assistant_wording_overrides_via_subclass():
    class Condensed(ContextManager):
        PRUNED_SEARCH_PREFIX = "Web: "
        PRUNED_ANSWER_PREFIX = "A: "

    entry = AssistantMessage(
        id="am3",
        created_at=1000,
        parts=[
            WebSearchContent(queries=["apple"], results=[], extras={"id": "s_1"}),
            TextContent(text="Nothing found."),
        ],
        llm_config=LLMConfig(model="claude-sonnet-5", provider="anthropic"),
        stop_reason="stop",
    )

    template = Condensed().prune_entry(SESSION, entry)

    assert template.content == [TextContent(text='Web: "apple" (no results). A: Nothing found.')]


def test_a_failed_fetch_prunes_to_the_failed_marker_and_hosts_cap_at_five():
    # the fetch error stamp renders as failed (never as a completed fetch),
    # and a long result list caps at five hosts with the remainder counted
    entry = AssistantMessage(
        id="am4",
        created_at=1000,
        parts=[
            WebSearchContent(
                queries=["Apple stock price today"],
                results=[
                    WebPageContent(url="https://cnn.com/a"),
                    WebPageContent(url="https://www.tradingview.com/b"),
                    WebPageContent(url="https://coinbase.com/c"),
                    WebPageContent(url="https://cnbc.com/d"),
                    WebPageContent(url="https://finance.yahoo.com/e"),
                    WebPageContent(url="https://f.com"),
                    WebPageContent(url="https://g.com"),
                    WebPageContent(url="https://h.com"),
                    WebPageContent(url="https://i.com"),
                ],
                extras={"id": "s_1"},
            ),
            WebFetchContent(
                web_page=WebPageContent(url="https://apple.com/x"),
                extras={"id": "f_1", "error": {"type": "web_fetch_tool_result_error", "error_code": "unavailable"}},
            ),
            TextContent(text="Partial answer."),
        ],
        llm_config=LLMConfig(model="claude-sonnet-5", provider="anthropic"),
        stop_reason="stop",
    )

    template = CM.prune_entry(SESSION, entry)

    assert template.content == [
        TextContent(
            text=(
                'Searched the web: "Apple stock price today" (9 results: cnn.com, tradingview.com, '
                "coinbase.com, cnbc.com, finance.yahoo.com, +4 more). "
                "Fetched: https://apple.com/x (failed: unavailable). "
                "Answered: Partial answer."
            )
        )
    ]
