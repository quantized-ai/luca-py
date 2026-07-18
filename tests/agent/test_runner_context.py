"""The runner ↔ ContextManager seam.

Declarative scenarios locking the four contact points:

- every NEW entry gets `context_tokens` from `calculate_context()` before
  `before_entry_written` middleware runs (and middleware has the final say —
  the framework never recalculates after it);
- a terminal `ToolExecution` is recalculated from its final model-facing
  outcome before `after_tool_execution` runs;
- a returned `ExecutionResult` passes through `process_tool_output()` before
  the terminal execution is constructed, so the durable session, the
  `ToolExecuted` event, and the next wire request all show the PROCESSED
  output — they can never disagree;
- pruning is machinery, not a runner method: `ContextManager.prune_entry()`
  composes with the ledger's prune door and the projector resolves the
  replacement on the next LLM call.
"""

from luca.agent.core.context_manager import (
    PRUNED_TOOL_OUTPUT_MARKER,
    ContextManager,
)
from luca.agent.core.models import (
    AgentSession,
    Conversation,
    ExecutionResult,
    ExecutionStatus,
    SessionConfig,
    TextContent,
    UserMessage,
)
from luca.client.testing import (
    FauxProvider,
    faux_assistant_message,
    faux_text,
    faux_tool_call,
)
from luca.client.types import TextBlock, ToolMessage
from luca.client.types import AssistantMessage as LucaAssistantMessage
from luca.client.types import ToolCall as LucaToolCall
from luca.client.types import UserMessage as LucaUserMessage

from tests.agent.scenarios import (
    MODEL,
    AddTool,
    DeterministicRunner,
    FakeToolRegistry,
)


class FixedContextManager(ContextManager):
    """Counts every entry as a constant — proves the runner asks the
    configured strategy, not the default estimator."""

    def calculate_context(self, entry) -> int:
        return 42


class TruncatingContextManager(ContextManager):
    """Replaces every returned tool output with a fixed marker, stashing the
    original in metadata — the documented truncation policy shape."""

    def process_tool_output(self, execution_result: ExecutionResult) -> ExecutionResult:
        original = "".join(part.text for part in execution_result.content)
        return ExecutionResult(
            content=[TextContent(text="[output truncated]")],
            metadata={**execution_result.metadata, "original": original},
            is_error=execution_result.is_error,
        )


class ContextOverridingMiddleware:
    """`before_entry_written` middleware has the final say on context —
    whatever it returns is persisted, never recalculated."""

    def before_entry_written(self, entry):
        entry.context_tokens = 7
        return entry


def _session(session_id: str) -> AgentSession:
    return AgentSession(
        id=session_id,
        entries={
            "u1": UserMessage(
                id="u1", created_at=500, parts=[TextContent(text="Add 1 and 2")],
            ),
        },
        active_conversation=Conversation(
            id="c1", nodes=["u1"], created_at=500, updated_at=500,
        ),
        session_config=SessionConfig(llm_config=MODEL),
    )


async def test_the_configured_context_manager_counts_every_new_entry():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("It's 3.")], finish_reason="stop"),
    ])
    session = _session("s_fixed")
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([AddTool()]), provider=faux,
        context_manager=FixedContextManager(),
        ids=["ts", "a1", "te1", "a2", "tf"], now=1000,
    )

    await runner.run()

    entries = runner.session.entries
    assert entries["ts"].context_tokens == 42
    assert entries["a1"].context_tokens == 42
    assert entries["te1"].context_tokens == 42  # recalculated at terminal
    assert entries["a2"].context_tokens == 42
    assert entries["tf"].context_tokens == 42
    assert entries["u1"].context_tokens == 0  # pre-existing entries untouched


async def test_middleware_has_the_final_say_on_context_tokens():
    session = AgentSession(
        id="s_mw",
        active_conversation=Conversation(
            id="c1", nodes=[], created_at=500, updated_at=500,
        ),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, middleware=[ContextOverridingMiddleware()],
        ids=["u1"], now=1000,
    )

    runner.post_message("A long message the estimator would count differently")

    # calculated before the hook, overridden by it, never repaired after
    assert runner.session.entries["u1"].context_tokens == 7


async def test_processed_tool_output_reaches_session_event_and_wire_identically():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("It's 3.")], finish_reason="stop"),
    ])
    session = _session("s_trunc")
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([AddTool()]), provider=faux,
        context_manager=TruncatingContextManager(),
        ids=["ts", "a1", "te1", "a2", "tf"], now=1000,
    )

    async with runner.run() as run:
        events = [event async for event in run]

    # durable: the persisted result IS the processed one (original preserved
    # by the manager's own policy, in metadata)
    assert runner.session.entries["te1"].status == ExecutionStatus.COMPLETED
    assert runner.session.entries["te1"].result == ExecutionResult(
        content=[TextContent(text="[output truncated]")],
        metadata={"original": "3"},
        is_error=False,
    )
    # context was calculated from the processed content
    assert runner.session.entries["te1"].context_tokens == 4  # 18 // 4
    # event: derived from the same persisted execution
    executed = [event for event in events if event.type == "tool_executed"][0]
    assert executed.result_text == "[output truncated]"
    assert executed.is_error is False
    # wire: the second LLM request projects the same processed output
    assert faux.requests[1].messages[-1] == ToolMessage(
        tool_call_id="tc1",
        content=[TextBlock(text="[output truncated]")],
        is_error=False,
    )


async def test_pruning_machinery_composes_and_reaches_the_next_wire_request():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
            finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("It's 3.")], finish_reason="stop"),
        faux_assistant_message([faux_text("Still 3.")], finish_reason="stop"),
    ])
    session = _session("s_prune")
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([AddTool()]), provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf", "p1", "u2", "ts2", "a3", "tf2"],
        now=1000,
    )
    await runner.run()  # turn 1: the tool round completes

    # prune the completed execution: strategy builds the replacement TEMPLATE,
    # the composer follows the new-entry ordering (identity → context →
    # persistence door); the runner deliberately exposes no prune method yet
    manager = runner.context_manager
    template = manager.prune_entry(runner.session.entries["te1"])

    def build(entry_id, parent_id, ts):
        pruned = template.model_copy(
            update={"id": entry_id, "parent_id": parent_id, "created_at": ts},
        )
        pruned.context_tokens = manager.calculate_context(pruned)
        return pruned

    pruned = runner.ledger.prune("te1", build)

    # the path visits the replacement; the original entry is untouched
    assert runner.session.active_conversation.nodes == [
        "u1", "ts", "a1", "p1", "a2", "tf",
    ]
    assert pruned.parent_id == "a1"  # the original's position, not the leaf
    assert pruned.context_tokens == 11  # len(marker) // 4
    assert runner.session.entries["te1"].status == ExecutionStatus.COMPLETED

    runner.post_message("And now?")
    await runner.run()  # turn 2

    assert faux.requests[2].messages == [
        LucaUserMessage(content=[TextBlock(text="Add 1 and 2")]),
        LucaAssistantMessage(content=[
            LucaToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        ]),
        ToolMessage(
            tool_call_id="tc1",
            content=[TextBlock(text=PRUNED_TOOL_OUTPUT_MARKER)],
            is_error=False,
        ),
        LucaAssistantMessage(content=[TextBlock(text="It's 3.")]),
        LucaUserMessage(content=[TextBlock(text="And now?")]),
    ]
