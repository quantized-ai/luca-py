"""Middleware scenarios: each hook is invoked in order and its return value
is used downstream.

Two kinds of tests:
1. One middleware, one method — verify the method was called AND the return
   value was actually used for the "downstream object" (the model call,
   the stored entry, the persisted execution, …).
2. Two middlewares for one method — verify the second middleware receives
   the first middleware's return value (ordering).

The tool-execution pair is exercised across outcomes: `before_tool_execution`
sees an allowed call pre-dispatch (and its returned `raw_tool_call` is the
effective call) AND terminal-at-birth / rejected calls with their status
already set; `after_tool_execution` observes EVERY outcome — with the live
exception on a dispatch failure (registry-authored terminal births carry no
live exception) — and its return value is what gets persisted.

Each test follows the declarative shape: known session + scripted faux
responses + middleware doubles → one action → assert the downstream effect.
"""

from luca.agent.core import AgentMiddlewareMixin
from luca.agent.core.models import (
    AgentSession,
    ApprovalDecision,
    ApprovalOption,
    ApprovalStatus,
    Conversation,
    ExecutionStatus,
    ImageBase64,
    ImageContent,
    SessionConfig,
    TextContent,
    ToolExecution,
    TurnFinish,
    UserMessage,
)
from luca.client.testing import (
    FauxProvider,
    faux_assistant_message,
    faux_text,
    faux_tool_call,
)

from tests.agent.scenarios import (
    MODEL,
    AddTool,
    DeterministicRunner,
    FakeToolRegistry,
    MultiplyTool,
    RaisingTool,
)

ALLOW_1000 = ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000)


# ── build_model_string ─────────────────────────────────────────────────────────


async def test_middleware_build_model_string_return_used_for_llm_call():
    class ModelStringMiddleware:
        def build_model_string(self, model_string: str, model_cfg) -> str:
            return "faux:override-model"  # client strips the provider prefix

    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message([faux_text("Hi!")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_mw_ms",
        entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")])},
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, provider=faux, ids=["ts", "a1", "tf"], now=1000,
        middleware=[ModelStringMiddleware()],
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    # The client parses "provider:model" and stores just the model part
    assert faux.requests[0].model == "override-model"


# ── build_tool_list ────────────────────────────────────────────────────────────


async def test_middleware_build_tool_list_return_used_for_llm_call():
    class FilterToolsMiddleware:
        def build_tool_list(self, tools: list) -> list:
            return tools[:1]  # keep only the first tool

    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message([faux_text("done")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_mw_tl",
        entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Go")])},
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([AddTool(), MultiplyTool()]),
        provider=faux, ids=["ts", "a1", "tf"], now=1000,
        middleware=[FilterToolsMiddleware()],
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    assert faux.requests[0].tools is not None
    assert len(faux.requests[0].tools) == 1
    assert faux.requests[0].tools[0].name == "add"


# ── before_llm_call ────────────────────────────────────────────────────────────


async def test_middleware_before_llm_call_return_used_for_llm_call():
    class SystemOverrideMiddleware:
        def before_llm_call(
            self, messages: list, system_message,
        ) -> tuple[list, str | None]:
            return messages, "OVERRIDE SYSTEM"

    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message([faux_text("ok")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_mw_llm",
        entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")])},
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, provider=faux, ids=["ts", "a1", "tf"], now=1000,
        middleware=[SystemOverrideMiddleware()],
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    assert faux.requests[0].system_message == "OVERRIDE SYSTEM"


# ── after_llm_response ────────────────────────────────────────────────────────


async def test_middleware_after_llm_response_return_stored_in_session():
    from luca.client.types.content import TextBlock

    class ResponseMiddleware:
        def after_llm_response(self, message):
            new_content = [TextBlock(text="MODIFIED")]
            return message.model_copy(update={"content": new_content})

    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message([faux_text("original text")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_mw_llm_resp",
        entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")])},
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, provider=faux, ids=["ts", "a1", "tf"], now=1000,
        middleware=[ResponseMiddleware()],
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    stored = runner.session.entries["a1"]
    assert stored.parts == [TextContent(text="MODIFIED")]


# ── before_post_message ───────────────────────────────────────────────────────


async def test_middleware_before_post_message_return_stored_in_entry():
    class UpperCaseMiddleware:
        def before_post_message(self, parts: list) -> list:
            return [TextContent(text=p.text.upper()) for p in parts]

    session = AgentSession(
        id="s_mw_pm",
        active_conversation=Conversation(id="c1", nodes=[], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, ids=["u1"], now=1000,
        middleware=[UpperCaseMiddleware()],
    )

    runner.post_message("hello world")

    assert runner.session.entries["u1"].parts == [TextContent(text="HELLO WORLD")]


async def test_before_post_message_sees_every_part_including_images():
    class RecordingMiddleware:
        def __init__(self) -> None:
            self.seen: list[str] = []

        def before_post_message(self, parts: list) -> list:
            self.seen = [part.type for part in parts]
            return parts

    middleware = RecordingMiddleware()
    session = AgentSession(
        id="s_mw_pm_seen",
        active_conversation=Conversation(id="c1", nodes=[], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, ids=["u1"], now=1000, middleware=[middleware],
    )
    image = ImageContent(source=ImageBase64(data="aGk=", media_type="image/png"))

    runner.post_message([image, TextContent(text="hi")])

    assert middleware.seen == ["image", "text"]


async def test_before_post_message_can_drop_a_part():
    class TextOnlyMiddleware:
        def before_post_message(self, parts: list) -> list:
            return [p for p in parts if isinstance(p, TextContent)]

    session = AgentSession(
        id="s_mw_pm_drop",
        active_conversation=Conversation(id="c1", nodes=[], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, ids=["u1"], now=1000, middleware=[TextOnlyMiddleware()],
    )
    image = ImageContent(source=ImageBase64(data="aGk=", media_type="image/png"))

    runner.post_message([image, TextContent(text="hi")])

    assert runner.session.entries["u1"].parts == [TextContent(text="hi")]


async def test_before_post_message_can_add_a_part():
    class ReminderMiddleware:
        def before_post_message(self, parts: list) -> list:
            return [*parts, TextContent(text="be concise")]

    session = AgentSession(
        id="s_mw_pm_add",
        active_conversation=Conversation(id="c1", nodes=[], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, ids=["u1"], now=1000, middleware=[ReminderMiddleware()],
    )
    image = ImageContent(source=ImageBase64(data="aGk=", media_type="image/png"))

    runner.post_message([image])

    assert runner.session.entries["u1"].parts == [
        image, TextContent(text="be concise"),
    ]


# ── before_entry_written ──────────────────────────────────────────────────────


async def test_middleware_before_entry_written_return_stored_in_session():
    class MarkTurnFinishMiddleware:
        def before_entry_written(self, entry):
            if isinstance(entry, TurnFinish):
                return entry.model_copy(update={"error": "mw_mark"})
            return entry

    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message([faux_text("Hi!")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_mw_bew",
        entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")])},
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, provider=faux, ids=["ts", "a1", "tf"], now=1000,
        middleware=[MarkTurnFinishMiddleware()],
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    # The TurnFinish entry was modified by the middleware before storage
    assert runner.session.entries["tf"].error == "mw_mark"
    # Other entries pass through unchanged (TurnStart has no error field)
    assert runner.session.entries["ts"].type == "turn_start"


async def test_middleware_before_entry_written_sees_every_execution_persistence():
    # Every ToolExecution persistence passes through the hook: creation, the
    # approval update, the RUNNING transition, and the terminal outcome.
    class ExecutionStatusRecorder:
        def __init__(self) -> None:
            self.seen: list[ExecutionStatus] = []

        def before_entry_written(self, entry):
            if isinstance(entry, ToolExecution):
                self.seen.append(entry.status)
            return entry

    recorder = ExecutionStatusRecorder()
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")], finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("3")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_mw_bew_exec",
        entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="add")])},
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([AddTool()]), provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"], now=1000,
        middleware=[recorder],
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    assert recorder.seen == [
        ExecutionStatus.PENDING,  # creation
        ExecutionStatus.PENDING,  # the ALLOW approval update
        ExecutionStatus.RUNNING,  # the dispatch transition
        ExecutionStatus.COMPLETED,  # the terminal outcome
    ]


# ── before_permission_check ───────────────────────────────────────────────────


async def test_middleware_before_permission_check_modified_execution_is_seen_and_persisted():
    class EnrichContextMiddleware:
        def before_permission_check(self, execution: ToolExecution) -> ToolExecution:
            return execution.model_copy(update={
                "extras": {**execution.extras, "mw_enriched": True},
            })

    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")], finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("3")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_mw_bpc",
        entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="add")])},
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    registry = FakeToolRegistry(
        [AddTool()],
        decisions=[ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000)],
    )
    runner = DeterministicRunner(
        session, tool_registry=registry, provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"], now=1000,
        middleware=[EnrichContextMiddleware()],
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    # The registry received the middleware-modified execution...
    assert registry.seen[0].extras.get("mw_enriched") is True
    # ...and the decision was applied to (and persisted from) that SAME
    # modified execution, not the original — its changes stick.
    assert runner.session.entries["te1"].extras.get("mw_enriched") is True


# ── after_permission_decision ─────────────────────────────────────────────────


async def test_middleware_after_permission_decision_return_recorded_and_used():
    # Strategy says DENY; middleware overrides to ALLOW → tool runs
    class OverrideDecisionMiddleware:
        def after_permission_decision(
            self, decision: ApprovalDecision, execution: ToolExecution,
        ) -> ApprovalDecision:
            return decision.model_copy(update={"decision": ApprovalOption.ALLOW})

    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("add", {"a": 2, "b": 3}, id="tc1")], finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("5")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_mw_apd",
        entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="add")])},
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    registry = FakeToolRegistry(
        [AddTool()],
        decisions=[ApprovalDecision(decision=ApprovalOption.DENY, created_at=1000)],
    )
    runner = DeterministicRunner(
        session, tool_registry=registry, provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"], now=1000,
        middleware=[OverrideDecisionMiddleware()],
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    # The middleware overrode DENY → ALLOW, so the tool ran (not REJECTED)
    execution = runner.session.entries["te1"]
    assert execution.status == ExecutionStatus.COMPLETED
    assert execution.approval_status == ApprovalStatus.ALLOWED
    assert execution.approval_decisions[-1].decision == ApprovalOption.ALLOW


# ── before_tool_execution ─────────────────────────────────────────────────────


async def test_middleware_before_tool_execution_effective_call_is_dispatched():
    # The returned execution's raw_tool_call IS the effective call: the runner
    # re-validates and runs it, and the persisted record shows it.
    class Args10xMiddleware:
        def before_tool_execution(self, execution: ToolExecution) -> ToolExecution:
            arguments = execution.raw_tool_call.arguments
            return execution.model_copy(update={
                "raw_tool_call": execution.raw_tool_call.model_copy(update={
                    "arguments": {"a": arguments["a"] * 10, "b": arguments["b"] * 10},
                }),
            })

    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")], finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("30")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_mw_bte",
        entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="add")])},
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([AddTool()]), provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"], now=1000,
        middleware=[Args10xMiddleware()],
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    # add(10, 20) = 30, not add(1, 2) = 3
    execution = runner.session.entries["te1"]
    assert execution.result.content[0].text == "30"
    assert execution.raw_tool_call.arguments == {"a": 10, "b": 20}
    # the original request block in the assistant message is untouched
    assert runner.session.entries["a1"].parts[0].arguments == {"a": 1, "b": 2}


async def test_middleware_before_tool_execution_sees_terminal_and_rejected_calls():
    # A preflight-terminal call (unknown tool) and a denied call both pass
    # through the hook with their status already set; a dispatched call
    # arrives PENDING and is NOT re-visited at its terminal transition.
    class StatusRecorder:
        def __init__(self) -> None:
            self.seen: list[tuple[str, ExecutionStatus]] = []

        def before_tool_execution(self, execution: ToolExecution) -> ToolExecution:
            self.seen.append((execution.tool_call_id, execution.status))
            return execution

    recorder = StatusRecorder()
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("nope", {"x": 1}, id="tc1"),
             faux_tool_call("add", {"a": 1, "b": 2}, id="tc2"),
             faux_tool_call("multiply", {"a": 3, "b": 4}, id="tc3")],
            finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("done")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_mw_bte_all",
        entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="go")])},
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    registry = FakeToolRegistry([AddTool(), MultiplyTool()], decisions=[
        ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000),  # tc2
        ApprovalDecision(decision=ApprovalOption.DENY, created_at=1000),  # tc3
    ])
    runner = DeterministicRunner(
        session, tool_registry=registry,
        provider=faux, ids=["ts", "a1", "te1", "te2", "te3", "a2", "tf"], now=1000,
        middleware=[recorder],
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    assert recorder.seen == [
        ("tc1", ExecutionStatus.NOT_FOUND),  # terminal at birth
        ("tc3", ExecutionStatus.REJECTED),  # denied at decision time
        ("tc2", ExecutionStatus.PENDING),  # allowed, about to dispatch
    ]


# ── after_tool_execution ──────────────────────────────────────────────────────


async def test_middleware_after_tool_execution_return_persisted():
    class ResultTransformMiddleware:
        def after_tool_execution(
            self, execution: ToolExecution, exception: Exception | None = None,
        ) -> ToolExecution:
            return execution.model_copy(update={
                "result": execution.result.model_copy(update={
                    "content": [TextContent(text="RESULT_MODIFIED")],
                }),
            })

    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")], finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("ok")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_mw_ate",
        entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="add")])},
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([AddTool()]), provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"], now=1000,
        middleware=[ResultTransformMiddleware()],
    )

    async with runner.run() as run:
        events = [event async for event in run]

    assert runner.session.entries["te1"].result.content[0].text == "RESULT_MODIFIED"
    # the ToolExecuted event projects the transformed execution
    assert events[3].type == "tool_executed"
    assert events[3].result_text == "RESULT_MODIFIED"


async def test_middleware_after_tool_execution_observes_every_outcome():
    # One middleware, four outcomes in one turn: NOT_FOUND (a terminal birth —
    # registry-authored, so no live exception reaches the hook), REJECTED
    # (denied), FAILED (body raised — with the live exception), and COMPLETED.
    # The hook sees each terminal state exactly once.
    class OutcomeRecorder:
        def __init__(self) -> None:
            self.seen: list[tuple[str, ExecutionStatus, type | None]] = []

        def after_tool_execution(
            self, execution: ToolExecution, exception: Exception | None = None,
        ) -> ToolExecution:
            self.seen.append((
                execution.tool_call_id,
                execution.status,
                type(exception) if exception is not None else None,
            ))
            return execution

    recorder = OutcomeRecorder()
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("nope", {}, id="tc1"),
             faux_tool_call("multiply", {"a": 3, "b": 4}, id="tc2"),
             faux_tool_call("boom", {"a": 1, "b": 2}, id="tc3"),
             faux_tool_call("add", {"a": 1, "b": 2}, id="tc4")],
            finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("done")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_mw_ate_all",
        entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="go")])},
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    registry = FakeToolRegistry(
        [AddTool(), MultiplyTool(), RaisingTool()],
        decisions=[
            ApprovalDecision(decision=ApprovalOption.DENY, created_at=1000),  # tc2
            ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000),  # tc3
            ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000),  # tc4
        ],
    )
    runner = DeterministicRunner(
        session, tool_registry=registry, provider=faux,
        ids=["ts", "a1", "te1", "te2", "te3", "te4", "a2", "tf"], now=1000,
        middleware=[recorder],
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    assert recorder.seen == [
        ("tc1", ExecutionStatus.NOT_FOUND, None),
        ("tc2", ExecutionStatus.REJECTED, None),
        ("tc3", ExecutionStatus.FAILED, ValueError),
        ("tc4", ExecutionStatus.COMPLETED, None),
    ]


# ── ordering: 2 middlewares, each method ──────────────────────────────────────


async def test_middlewares_applied_in_order_second_receives_first_output():
    """The second middleware receives the first middleware's return value, not
    the original. This proves chaining — not that the result is used downstream
    (covered by the single-middleware tests above)."""

    class AppendV1Middleware:
        def build_model_string(self, model_string: str, model_cfg) -> str:
            return model_string + "-v1"

    class AppendV2Middleware:
        def build_model_string(self, model_string: str, model_cfg) -> str:
            return model_string + "-v2"

    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message([faux_text("ok")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_mw_order",
        entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")])},
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, provider=faux, ids=["ts", "a1", "tf"], now=1000,
        middleware=[AppendV1Middleware(), AppendV2Middleware()],
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    # "faux:test-model" → "-v1" → "faux:test-model-v1" → "-v2" → "faux:test-model-v1-v2"
    # The client strips the "faux:" prefix, leaving just the model name part
    assert faux.requests[0].model == "test-model-v1-v2"


async def test_middlewares_applied_in_order_for_before_llm_call():
    """Two before_llm_call middlewares: second receives the first's output."""

    class AddSuffixMiddleware:
        def __init__(self, suffix: str) -> None:
            self.suffix = suffix

        def before_llm_call(self, messages, system_message):
            return messages, (system_message or "") + self.suffix

    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message([faux_text("ok")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_mw_order2",
        entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")])},
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, provider=faux, ids=["ts", "a1", "tf"], now=1000,
        middleware=[AddSuffixMiddleware("-A"), AddSuffixMiddleware("-B")],
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    # "" + "-A" → "-A", then "-A" + "-B" → "-A-B"
    assert faux.requests[0].system_message == "-A-B"


# ── AgentMiddlewareMixin: identity hooks, safe partial override ───────────────


def test_mixin_every_hook_returns_its_input():
    mixin = AgentMiddlewareMixin()
    entry = UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")])
    message = object()
    execution = object()
    decision = object()

    assert mixin.build_model_string("openrouter:openai/gpt-4o-mini", MODEL) == "openrouter:openai/gpt-4o-mini"
    assert mixin.build_tool_list(["t1", "t2"]) == ["t1", "t2"]
    assert mixin.before_post_message([TextContent(text="hello")]) == [
        TextContent(text="hello"),
    ]
    assert mixin.before_entry_written(entry) is entry
    assert mixin.before_llm_call(["m1"], "sys") == (["m1"], "sys")
    assert mixin.before_llm_call(["m1"], None) == (["m1"], None)
    assert mixin.after_llm_response(message) is message
    assert mixin.before_permission_check(execution) is execution
    assert mixin.after_permission_decision(decision, execution) is decision
    assert mixin.before_tool_execution(execution) is execution
    assert mixin.after_tool_execution(execution) is execution
    assert mixin.after_tool_execution(execution, ValueError("x")) is execution


def test_mixin_has_no_build_messages_hook():
    # conversation projection is a runner collaborator (ConversationProjector),
    # not a middleware stage
    assert not hasattr(AgentMiddlewareMixin, "build_messages")


async def test_mixin_subclass_partial_override_does_not_clobber_post_message():
    class OnlyResponse(AgentMiddlewareMixin):  # subclass, override one hook
        def after_llm_response(self, message):
            return message

    session = AgentSession(
        id="s_mw_sub",
        active_conversation=Conversation(id="c1", nodes=[], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, ids=["u1"], now=1000,
        middleware=[OnlyResponse()],
    )

    # The inherited before_post_message must NOT blank the text
    runner.post_message("hello world")

    assert runner.session.entries["u1"].parts == [TextContent(text="hello world")]


async def test_mixin_subclass_override_applies_and_inherited_hooks_pass_full_turn_through():
    class OnlyModelSuffix(AgentMiddlewareMixin):
        def build_model_string(self, model_string: str, llm_cfg) -> str:
            return model_string + "-routed"

    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")], finish_reason="tool_use",
        ),
        faux_assistant_message([faux_text("3")], finish_reason="stop"),
    ])
    session = AgentSession(
        id="s_mw_mixin_run",
        entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="add")])},
        active_conversation=Conversation(id="c1", nodes=["u1"], created_at=500, updated_at=500),
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session, tool_registry=FakeToolRegistry([AddTool()]), provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"], now=1000,
        middleware=[OnlyModelSuffix()],
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    # The overridden hook applied…
    assert faux.requests[0].model == "test-model-routed"
    # …and every inherited hook passed its stage through untouched: the tool
    # ran with the original args, the result and final answer were stored.
    assert runner.session.entries["te1"].status == ExecutionStatus.COMPLETED
    assert runner.session.entries["te1"].result.content[0].text == "3"
    assert runner.session.entries["a2"].parts == [TextContent(text="3")]
