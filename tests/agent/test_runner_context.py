"""The runner ↔ ContextManager seam.

Declarative scenarios locking the contact points:

- every NEW entry gets `context_tokens` from `calculate_context()` before
  `before_entry_written` middleware runs (and middleware has the final say —
  the framework never recalculates after it);
- a terminal `ToolExecution` is recalculated from its final model-facing
  outcome before `after_tool_execution` runs;
- a returned `ExecutionResult` passes through `process_tool_output()` before
  the terminal execution is constructed, so the durable session, the
  `ToolExecuted` event, and the next wire request all show the PROCESSED
  output — they can never disagree; the execution handed to it is the
  IN-TRANSITION one (still RUNNING, no result attached), which is what makes
  a per-tool policy expressible;
- pruning is machinery, not a runner method: `ContextManager.prune_entry()`
  composes with the ledger's prune door and the projector resolves the
  replacement on the next LLM call;
- `recalculate_context_tokens()` is the only way back from a stale basis: it
  re-derives every stored count, and nothing else in the framework — the
  constructor included — ever touches one. It runs NO middleware: it spans
  every conversation at once, so no `conversation_id` would honestly scope a
  `before_entry_written` call (that case lives in
  `test_runner_middleware.py`).
"""

from typing import ClassVar

from luca.agent.core.context_manager import (
    PRUNED_TOOL_OUTPUT_MARKER,
    ContextManager,
)
from luca.agent.core.events import ToolExecuted
from luca.agent.core.models import (
    AgentSession,
    ApprovalDecision,
    ApprovalOption,
    ApprovalStatus,
    Entry,
    ExecutionAttempt,
    ExecutionAttemptOutcome,
    ExecutionResult,
    ExecutionStatus,
    SessionConfig,
    TextContent,
    ToolCall,
    ToolExecution,
    UserMessage,
)
from luca.client.testing import (
    FauxProvider,
    faux_assistant_message,
    faux_text,
    faux_tool_call,
)
from luca.client.types import (
    AssistantMessage as LucaAssistantMessage,
    TextBlock,
    ToolCall as LucaToolCall,
    ToolMessage,
    UserMessage as LucaUserMessage,
)
from tests.agent.scenarios import (
    ADD_SPEC,
    CHEAP,
    MODEL,
    RICH_SESSION,
    AddTool,
    DeterministicRunner,
    FakeToolRegistry,
    conversation,
    main_conversation,
)


class FixedContextManager(ContextManager):
    """Counts every entry as a constant — proves the runner asks the
    configured strategy, not the default estimator."""

    def calculate_context(self, session: AgentSession, entry: Entry) -> int:
        return 42


class ModelAwareContextManager(ContextManager):
    """Counts against the ACTIVE MODEL — the strategy shape
    `recalculate_context_tokens()` exists for: a count written under one model
    is on the wrong basis the moment `session_config.llm_config` changes."""

    PER_MODEL: ClassVar[dict[str, int]] = {"test-model": 3, "cheap-model": 11}

    def calculate_context(self, session: AgentSession, entry: Entry) -> int:
        return self.PER_MODEL[session.session_config.llm_config.model]


class TruncatingContextManager(ContextManager):
    """Replaces every returned tool output with a fixed marker, stashing the
    original in metadata — the documented truncation policy shape. Records
    every execution it was handed in `seen`."""

    def __init__(self) -> None:
        self.seen: list[ToolExecution] = []

    def process_tool_output(
        self,
        session: AgentSession,
        execution: ToolExecution,
        result: ExecutionResult,
    ) -> ExecutionResult:
        self.seen.append(execution)
        original = "".join(part.text for part in result.content)
        return ExecutionResult(
            content=[TextContent(text="[output truncated]")],
            metadata={**result.metadata, "original": original},
            is_error=result.is_error,
        )


class ContextOverridingMiddleware:
    """`before_entry_written` middleware has the final say on context —
    whatever it returns is persisted, never recalculated."""

    def before_entry_written(self, session, conversation_id, entry):
        entry.context_tokens = 7
        return entry


def _session(session_id: str) -> AgentSession:
    return AgentSession(
        id=session_id,
        entries={
            "u1": UserMessage(
                id="u1",
                created_at=500,
                parts=[TextContent(text="Add 1 and 2")],
            ),
        },
        conversations={
            "c1": conversation(
                "c1",
                ["u1"],
                created_at=500,
                updated_at=500,
            )
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )


async def test_the_configured_context_manager_counts_every_new_entry():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("It's 3.")], finish_reason="stop"),
        ]
    )
    session = _session("s_fixed")
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()]),
        provider=faux,
        context_manager=FixedContextManager(),
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
    )

    await runner.run()

    assert {entry_id: entry.context_tokens for entry_id, entry in runner.session.entries.items()} == {
        "u1": 0,  # pre-existing entries are untouched
        "ts": 42,
        "a1": 42,
        "te1": 42,  # recalculated at the terminal transition
        "a2": 42,
        "tf": 42,
    }


async def test_middleware_has_the_final_say_on_context_tokens():
    session = AgentSession(
        id="s_mw",
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
    runner = DeterministicRunner(
        session,
        middleware=[ContextOverridingMiddleware()],
        ids=["u1"],
        now=1000,
    )

    runner.post_message("A long message the estimator would count differently")

    # calculated before the hook, overridden by it, never repaired after
    assert runner.session.entries["u1"].context_tokens == 7


async def test_processed_tool_output_reaches_session_event_and_wire_identically():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("It's 3.")], finish_reason="stop"),
        ]
    )
    session = _session("s_trunc")
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()]),
        provider=faux,
        context_manager=TruncatingContextManager(),
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
    )

    async with runner.run() as run:
        events = [event async for event in run]

    # durable: the persisted result IS the processed one (original preserved
    # by the manager's own policy, in metadata), and its context was
    # calculated from the processed content
    final = ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        tool_spec=ADD_SPEC,
        tool_spec_id=ADD_SPEC.spec_id(),
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(
            content=[TextContent(text="[output truncated]")],
            metadata={"original": "3"},
            is_error=False,
        ),
        approval_status=ApprovalStatus.ALLOWED,
        approval_decisions=[
            ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000),
        ],
        attempts=[ExecutionAttempt(outcome=ExecutionAttemptOutcome.COMPLETED, started_at=1000, ended_at=1000)],
        finished_at=1000,
        updated_at=1000,
        context_tokens=4,  # len("[output truncated]") // 4
    )
    assert runner.session.entries["te1"] == final
    # event: derived from the same persisted execution
    executed = next(event for event in events if event.type == "tool_executed")
    assert executed == ToolExecuted(
        conversation_id="c1",
        tool_call_id="tc1",
        execution=final,
        result_text="[output truncated]",
        is_error=False,
    )
    # wire: the second LLM request projects the same processed output
    assert faux.requests[1].messages[-1] == ToolMessage(
        tool_call_id="tc1",
        content=[TextBlock(text="[output truncated]")],
        is_error=False,
    )


async def test_process_tool_output_is_handed_the_in_transition_execution():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("It's 3.")], finish_reason="stop"),
        ]
    )
    session = _session("s_transition")
    manager = TruncatingContextManager()
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()]),
        provider=faux,
        context_manager=manager,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
    )

    await runner.run()

    # identity is fully readable (raw_tool_call + tool_spec), the outcome is
    # not: still RUNNING, no result, no ended_at
    assert manager.seen == [
        ToolExecution(
            id="te1",
            conversation_id="c1",
            parent_id="a1",
            created_at=1000,
            tool_call_id="tc1",
            raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
            tool_spec=ADD_SPEC,
            tool_spec_id=ADD_SPEC.spec_id(),
            status=ExecutionStatus.RUNNING,
            approval_status=ApprovalStatus.ALLOWED,
            approval_decisions=[
                ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000),
            ],
            attempts=[ExecutionAttempt(started_at=1000)],
            updated_at=1000,
        )
    ]


async def test_pruning_machinery_composes_and_reaches_the_next_wire_request():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("It's 3.")], finish_reason="stop"),
            faux_assistant_message([faux_text("Still 3.")], finish_reason="stop"),
        ]
    )
    session = _session("s_prune")
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()]),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf", "p1", "u2", "ts2", "a3", "tf2"],
        now=1000,
    )
    await runner.run()  # turn 1: the tool round completes

    # prune the completed execution: strategy builds the replacement TEMPLATE,
    # the composer follows the new-entry ordering (identity → context →
    # persistence door); the runner deliberately exposes no prune method yet
    manager = runner.context_manager
    template = manager.prune_entry(runner.session, runner.session.entries["te1"])

    def build(entry_id, parent_id, ts):
        pruned = template.model_copy(
            update={"id": entry_id, "parent_id": parent_id, "created_at": ts},
        )
        pruned.context_tokens = manager.calculate_context(runner.session, pruned)
        return pruned

    pruned = runner.ledger.prune(session.main_conversation_id, "te1", build)

    # the path visits the replacement; the original entry is untouched
    assert main_conversation(runner.session).nodes == [
        "u1",
        "ts",
        "a1",
        "p1",
        "a2",
        "tf",
    ]
    assert pruned.parent_id == "a1"  # the original's position, not the leaf
    assert pruned.context_tokens == 11  # len(marker) // 4
    assert runner.session.entries["te1"].status == ExecutionStatus.COMPLETED

    runner.post_message("And now?")
    await runner.run()  # turn 2

    assert faux.requests[2].messages == [
        LucaUserMessage(content=[TextBlock(text="Add 1 and 2")]),
        LucaAssistantMessage(
            content=[
                LucaToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
            ],
            provider="faux",
            model="test-model",
        ),
        ToolMessage(
            tool_call_id="tc1",
            content=[TextBlock(text=PRUNED_TOOL_OUTPUT_MARKER)],
            is_error=False,
        ),
        LucaAssistantMessage(
            content=[TextBlock(text="It's 3.")],
            provider="faux",
            model="test-model",
        ),
        LucaUserMessage(content=[TextBlock(text="And now?")]),
    ]


async def test_recalculating_puts_every_stored_entry_on_the_new_model_basis():
    """A model swap leaves every stored count on the old basis;
    `recalculate_context_tokens()` re-derives them ALL — the archived
    conversation's entries (`u0`, `a0` on `c0`) and the pruned referent that
    is on no path at all (`te0`) included, because the count is intrinsic to
    an entry and shared by every conversation referencing it."""
    session = RICH_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=ModelAwareContextManager(),
        now=1000,
    )
    session.session_config.llm_config = CHEAP

    runner.recalculate_context_tokens()

    assert {entry_id: entry.context_tokens for entry_id, entry in runner.session.entries.items()} == {
        "u0": 11,  # archived conversation c0
        "a0": 11,  # archived conversation c0
        "te0": 11,  # in the store, on no path
        "cmp0": 11,  # was 9 on the old basis
        "u1": 11,
        "ts1": 11,
        "a1": 11,
        "te1": 11,
        "te2": 11,
        "pr1": 11,  # was 7 on the old basis
        "a2": 11,
        "tf1": 11,
        "u2": 11,
        "ts2": 11,
        "tf2": 11,
        "u3": 11,
        "ts3": 11,
        "a3": 11,
        "te3": 11,
        "cr1": 11,
        "tf3": 11,
        "u4": 11,
    }


async def test_constructing_a_runner_recalculates_no_context_tokens():
    """Taking ownership of a loaded session re-derives the conversation
    STATUS and nothing else: a manager that would count every entry as 3
    changes no stored count until the application calls
    `recalculate_context_tokens()` itself."""
    session = RICH_SESSION.model_copy(deep=True)

    runner = DeterministicRunner(
        session,
        context_manager=ModelAwareContextManager(),
        now=1000,
    )

    assert {entry_id: entry.context_tokens for entry_id, entry in runner.session.entries.items()} == {
        "u0": 0,
        "a0": 0,
        "te0": 0,
        "cmp0": 9,
        "u1": 0,
        "ts1": 0,
        "a1": 0,
        "te1": 0,
        "te2": 0,
        "pr1": 7,
        "a2": 0,
        "tf1": 0,
        "u2": 0,
        "ts2": 0,
        "tf2": 0,
        "u3": 0,
        "ts3": 0,
        "a3": 0,
        "te3": 0,
        "cr1": 0,
        "tf3": 0,
        "u4": 0,
    }
