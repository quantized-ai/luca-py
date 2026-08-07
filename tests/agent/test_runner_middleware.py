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
effective call the registry's `prepare()` then resolves from — rewrite it to
an unknown name and the call records NOT_FOUND, undispatched) AND
terminal-at-birth / rejected calls with their status already set;
`after_tool_execution` observes EVERY outcome — with the live exception on a
dispatch failure (registry-authored terminal births carry no live exception) —
and its return value is what gets persisted.

Two hooks are pinned to the shape of the value they receive, because in both
cases a plausible-looking alternative exists one layer away:
`build_tool_list` runs on the registry's `ToolSpec`s, never on the converted
wire list (`luca.client.types.Tool`), and never sees a private spec;
`before_entry_written` may replace a `ToolExecution`'s `tool_spec`, and the id
stamped on the way to the store is re-derived from what the hook returned.
`recalculate_context_tokens()` is the one write-shaped door that runs NO
middleware at all — it rewrites every entry across every conversation, so no
single conversation id would be honest.

Every hook receives `(session, conversation_id, …)`. The session is the live
runner-held object and the id is the conversation whose operation invoked the
hook, which is what makes one instance safe across a subagent tree.

Each test follows the declarative shape: known session + scripted faux
responses + middleware doubles → one action → assert the downstream effect.
"""

import pytest

from luca.agent.core import AgentMiddlewareMixin
from luca.agent.core.compaction import CompactionPlan
from luca.agent.core.context_manager import ContextManager
from luca.agent.core.events import ToolExecuted
from luca.agent.core.models import (
    ApprovalDecision,
    ApprovalOption,
    ApprovalStatus,
    AssistantMessage,
    CompactionEntry,
    ExecutionResult,
    ExecutionStatus,
    ImageBase64,
    ImageContent,
    LLMConfig,
    SessionConfig,
    TextContent,
    ToolCall,
    ToolExecution,
    ToolExecutionError,
    TurnFinish,
    TurnOutcome,
    TurnStart,
    UserMessage,
)
from luca.agent.core.projection import ConversationProjector
from luca.client.exceptions import ClientError
from luca.client.testing import (
    FauxProvider,
    faux_assistant_message,
    faux_error,
    faux_text,
    faux_tool_call,
)
from luca.client.types import TextBlock, Tool as LucaTool, UserMessage as LucaUserMessage
from tests.agent.scenarios import (
    ADD_SPEC,
    MODEL,
    MULTIPLY_SPEC,
    RICH_IDLE_SESSION,
    AddTool,
    DeterministicRunner,
    FakeContextManager,
    FakeToolRegistry,
    MultiplyTool,
    PrivateTool,
    RaisingTool,
    conversation,
    main_conversation,
    make_session,
)

ALLOW_1000 = ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000)
DENY_1000 = ApprovalDecision(decision=ApprovalOption.DENY, created_at=1000)


class ConstantContext(ContextManager):
    """Every entry counts 7, so a whole-store refresh asserts as a literal."""

    def calculate_context(self, session, entry) -> int:
        return 7


# The same tool, re-specced by a middleware on its way to the store: a new
# `spec_id()`, and therefore a new `tool_spec_id` on the execution.
NAMESPACED_ADD_SPEC = ADD_SPEC.model_copy(update={"namespace": "billing"})


# ── build_model_string ─────────────────────────────────────────────────────────


async def test_middleware_build_model_string_return_used_for_llm_call():
    class ModelStringMiddleware:
        def build_model_string(self, session, conversation_id, model_string: str, model_cfg) -> str:
            return "faux:override-model"  # client strips the provider prefix

    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message([faux_text("Hi!")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_mw_ms",
        entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")])},
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        provider=faux,
        ids=["ts", "a1", "tf"],
        now=1000,
        middleware=[ModelStringMiddleware()],
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    # The client parses "provider:model" and stores just the model part
    assert faux.requests[0].model == "override-model"


# ── build_tool_list ────────────────────────────────────────────────────────────


async def test_middleware_build_tool_list_sees_tool_specs_and_its_return_is_sent():
    # The hook runs BEFORE the adapter, on the registry's own `ToolSpec`s —
    # the core's tool type, carrying `tool_kind` / `namespace` /
    # `output_schema` / `metadata` that the wire tool drops — and the list it
    # returns is what the adapter converts into the request's tools.
    class FirstToolOnlyMiddleware:
        def __init__(self) -> None:
            self.seen: list = []

        def build_tool_list(self, session, conversation_id, tools: list) -> list:
            self.seen = tools
            return tools[:1]

    middleware = FirstToolOnlyMiddleware()
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message([faux_text("done")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_mw_tl",
        entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Go")])},
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool(), MultiplyTool()]),
        provider=faux,
        ids=["ts", "a1", "tf"],
        now=1000,
        middleware=[middleware],
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    assert middleware.seen == [ADD_SPEC, MULTIPLY_SPEC]
    assert faux.requests[0].tools == [
        LucaTool(
            name="add",
            description="Add two numbers.",
            parameters=ADD_SPEC.input_schema,
        ),
    ]


# ── before_llm_call ────────────────────────────────────────────────────────────


async def test_middleware_before_llm_call_return_used_for_llm_call():
    class SystemOverrideMiddleware:
        def before_llm_call(
            self,
            session,
            conversation_id,
            messages: list,
            system_message,
        ) -> tuple[list, str | None]:
            return messages, "OVERRIDE SYSTEM"

    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message([faux_text("ok")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_mw_llm",
        entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")])},
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        provider=faux,
        ids=["ts", "a1", "tf"],
        now=1000,
        middleware=[SystemOverrideMiddleware()],
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    assert faux.requests[0].system_message == "OVERRIDE SYSTEM"


# ── after_llm_response ────────────────────────────────────────────────────────


async def test_middleware_after_llm_response_return_stored_in_session():
    class ResponseMiddleware:
        def after_llm_response(self, session, conversation_id, message):
            return message.model_copy(update={"content": [TextBlock(text="MODIFIED")]})

    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message([faux_text("original text")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_mw_llm_resp",
        entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")])},
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        provider=faux,
        ids=["ts", "a1", "tf"],
        now=1000,
        middleware=[ResponseMiddleware()],
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    # The recorded entry is built from the RETURNED message, so its parts —
    # and the context estimate derived from them — are the middleware's.
    assert runner.session.entries["a1"] == AssistantMessage(
        id="a1",
        parent_id="ts",
        created_at=1000,
        context_tokens=2,
        parts=[TextContent(text="MODIFIED")],
        llm_config=MODEL,
        stop_reason="stop",
    )


# ── before_post_message ───────────────────────────────────────────────────────


async def test_middleware_before_post_message_return_stored_in_entry():
    class UpperCaseMiddleware:
        def before_post_message(self, session, conversation_id, parts: list) -> list:
            return [TextContent(text=p.text.upper()) for p in parts]

    session = make_session(
        id="s_mw_pm",
        conversations={"c1": conversation("c1", [], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        ids=["u1"],
        now=1000,
        middleware=[UpperCaseMiddleware()],
    )

    runner.post_message("hello world")

    assert runner.session.entries["u1"].parts == [TextContent(text="HELLO WORLD")]


async def test_before_post_message_sees_every_part_including_images():
    class RecordingMiddleware:
        def __init__(self) -> None:
            self.seen: list[str] = []

        def before_post_message(self, session, conversation_id, parts: list) -> list:
            self.seen = [part.type for part in parts]
            return parts

    middleware = RecordingMiddleware()
    session = make_session(
        id="s_mw_pm_seen",
        conversations={"c1": conversation("c1", [], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        ids=["u1"],
        now=1000,
        middleware=[middleware],
    )
    image = ImageContent(source=ImageBase64(data="aGk=", media_type="image/png"))

    runner.post_message([image, TextContent(text="hi")])

    assert middleware.seen == ["image", "text"]


async def test_before_post_message_can_drop_a_part():
    class TextOnlyMiddleware:
        def before_post_message(self, session, conversation_id, parts: list) -> list:
            return [p for p in parts if isinstance(p, TextContent)]

    session = make_session(
        id="s_mw_pm_drop",
        conversations={"c1": conversation("c1", [], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        ids=["u1"],
        now=1000,
        middleware=[TextOnlyMiddleware()],
    )
    image = ImageContent(source=ImageBase64(data="aGk=", media_type="image/png"))

    runner.post_message([image, TextContent(text="hi")])

    assert runner.session.entries["u1"].parts == [TextContent(text="hi")]


async def test_before_post_message_can_add_a_part():
    class ReminderMiddleware:
        def before_post_message(self, session, conversation_id, parts: list) -> list:
            return [*parts, TextContent(text="be concise")]

    session = make_session(
        id="s_mw_pm_add",
        conversations={"c1": conversation("c1", [], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        ids=["u1"],
        now=1000,
        middleware=[ReminderMiddleware()],
    )
    image = ImageContent(source=ImageBase64(data="aGk=", media_type="image/png"))

    runner.post_message([image])

    assert runner.session.entries["u1"].parts == [
        image,
        TextContent(text="be concise"),
    ]


# ── before_entry_written ──────────────────────────────────────────────────────


async def test_middleware_before_entry_written_return_stored_in_session():
    class MarkTurnFinishMiddleware:
        def before_entry_written(self, session, conversation_id, entry):
            if isinstance(entry, TurnFinish):
                return entry.model_copy(update={"error": "mw_mark"})
            return entry

    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message([faux_text("Hi!")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_mw_bew",
        entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")])},
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        provider=faux,
        ids=["ts", "a1", "tf"],
        now=1000,
        middleware=[MarkTurnFinishMiddleware()],
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    # The TurnFinish was modified by the middleware before storage…
    assert runner.session.entries["tf"] == TurnFinish(
        id="tf",
        parent_id="a1",
        created_at=1000,
        outcome=TurnOutcome.COMPLETED,
        error="mw_mark",
    )
    # …and the entries the hook returned untouched are stored untouched.
    assert runner.session.entries["ts"] == TurnStart(
        id="ts",
        parent_id="u1",
        created_at=1000,
    )


async def test_middleware_before_entry_written_sees_every_execution_persistence():
    # Every ToolExecution persistence passes through the hook: creation, the
    # approval update, the RUNNING transition, and the terminal outcome.
    class ExecutionStatusRecorder:
        def __init__(self) -> None:
            self.seen: list[ExecutionStatus] = []

        def before_entry_written(self, session, conversation_id, entry):
            if isinstance(entry, ToolExecution):
                self.seen.append(entry.status)
            return entry

    recorder = ExecutionStatusRecorder()
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("3")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_mw_bew_exec",
        entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="add")])},
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()]),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
        middleware=[recorder],
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    assert recorder.seen == [
        ExecutionStatus.RECEIVED,  # appended with the assistant message
        ExecutionStatus.PENDING,  # the registry's birth, folded in
        ExecutionStatus.PENDING,  # the ALLOW approval update
        ExecutionStatus.RUNNING,  # the dispatch transition
        ExecutionStatus.COMPLETED,  # the terminal outcome
    ]


async def test_middleware_before_entry_written_replacing_the_spec_restamps_tool_spec_id():
    # `tool_spec_id` is the durable reference and `tool_spec` only a cache, so
    # a hook that replaces the spec on one write has the id re-derived from
    # what it returned — and the version filed at birth stays in the store,
    # still resolvable by anything that points at it.
    class NamespaceStamper:
        def before_entry_written(self, session, conversation_id, entry):
            if isinstance(entry, ToolExecution) and entry.status == ExecutionStatus.COMPLETED:
                return entry.model_copy(update={"tool_spec": NAMESPACED_ADD_SPEC})
            return entry

    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("3")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_mw_bew_spec",
        entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="add")])},
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()]),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
        middleware=[NamespaceStamper()],
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    assert runner.session.entries["te1"] == ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        tool_spec=NAMESPACED_ADD_SPEC,
        tool_spec_id=NAMESPACED_ADD_SPEC.spec_id(),
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(content=[TextContent(text="3")]),
        approval_status=ApprovalStatus.ALLOWED,
        approval_decisions=[ALLOW_1000],
        started_at=1000,
        ended_at=1000,
        updated_at=1000,
    )
    assert runner.session.tool_specs == {
        ADD_SPEC.spec_id(): ADD_SPEC,
        NAMESPACED_ADD_SPEC.spec_id(): NAMESPACED_ADD_SPEC,
    }


async def test_recalculate_context_tokens_re_derives_every_entry_and_runs_no_middleware():
    # The application-called refresh rewrites EVERY entry across EVERY
    # conversation at once — the archived conversation's entries and the
    # pruned referent that sits on no path included — so there is no single
    # conversation id that honestly scopes it. It is an operational refresh of
    # a derived estimate, not a write with a scope, so `before_entry_written`
    # does not fire: a middleware that raises on every entry cannot break it.
    class Bomb:
        def before_entry_written(self, session, conversation_id, entry):
            raise AssertionError("recalculate_context_tokens must run no middleware")

    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=ConstantContext(),
        middleware=[Bomb()],
        now=1000,
    )

    runner.recalculate_context_tokens()

    assert {entry_id: entry.context_tokens for entry_id, entry in runner.session.entries.items()} == {
        "u0": 7,  # archived conversation c0
        "a0": 7,  # archived conversation c0
        "te0": 7,  # in the store, on no path
        "cmp0": 7,
        "u1": 7,
        "ts1": 7,
        "a1": 7,
        "te1": 7,
        "te2": 7,
        "pr1": 7,
        "a2": 7,
        "tf1": 7,
        "u2": 7,
        "ts2": 7,
        "tf2": 7,
        "u3": 7,
        "ts3": 7,
        "a3": 7,
        "te3": 7,
        "cr1": 7,
        "tf3": 7,
    }


# ── before_permission_check ───────────────────────────────────────────────────


async def test_middleware_before_permission_check_modified_execution_is_seen_and_persisted():
    class EnrichContextMiddleware:
        def before_permission_check(self, session, conversation_id, execution: ToolExecution) -> ToolExecution:
            return execution.model_copy(
                update={
                    "extras": {**execution.extras, "mw_enriched": True},
                }
            )

    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("3")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_mw_bpc",
        entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="add")])},
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    registry = FakeToolRegistry([AddTool()], decisions=[ALLOW_1000])
    runner = DeterministicRunner(
        session,
        tool_registry=registry,
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
        middleware=[EnrichContextMiddleware()],
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    # The registry was asked about the middleware-modified execution...
    assert registry.seen == [
        ToolExecution(
            id="te1",
            conversation_id="c1",
            parent_id="a1",
            created_at=1000,
            tool_call_id="tc1",
            raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
            tool_spec=ADD_SPEC,
            tool_spec_id=ADD_SPEC.spec_id(),
            extras={"mw_enriched": True},
            status=ExecutionStatus.PENDING,
        ),
    ]
    # ...and the decision was applied to (and persisted from) that SAME
    # modified execution, not the original — its changes stick.
    assert runner.session.entries["te1"] == ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        tool_spec=ADD_SPEC,
        tool_spec_id=ADD_SPEC.spec_id(),
        extras={"mw_enriched": True},
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(content=[TextContent(text="3")]),
        approval_status=ApprovalStatus.ALLOWED,
        approval_decisions=[ALLOW_1000],
        started_at=1000,
        ended_at=1000,
        updated_at=1000,
    )


# ── after_permission_decision ─────────────────────────────────────────────────


async def test_middleware_after_permission_decision_return_recorded_and_used():
    # Strategy says DENY; middleware overrides to ALLOW → tool runs
    class OverrideDecisionMiddleware:
        def after_permission_decision(
            self,
            session,
            conversation_id,
            decision: ApprovalDecision,
            execution: ToolExecution,
        ) -> ApprovalDecision:
            return decision.model_copy(update={"decision": ApprovalOption.ALLOW})

    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 2, "b": 3}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("5")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_mw_apd",
        entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="add")])},
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    registry = FakeToolRegistry([AddTool()], decisions=[DENY_1000])
    runner = DeterministicRunner(
        session,
        tool_registry=registry,
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
        middleware=[OverrideDecisionMiddleware()],
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    # The middleware's decision is the one recorded in the audit log and the
    # one the runner acted on: ALLOWED and dispatched, never REJECTED.
    assert runner.session.entries["te1"] == ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 2, "b": 3}),
        tool_spec=ADD_SPEC,
        tool_spec_id=ADD_SPEC.spec_id(),
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(content=[TextContent(text="5")]),
        approval_status=ApprovalStatus.ALLOWED,
        approval_decisions=[ALLOW_1000],
        started_at=1000,
        ended_at=1000,
        updated_at=1000,
    )


# ── before_tool_execution ─────────────────────────────────────────────────────


async def test_middleware_before_tool_execution_effective_call_is_dispatched():
    # The returned execution's raw_tool_call IS the effective call: the runner
    # re-validates and runs it, and the persisted record shows it.
    class Args10xMiddleware:
        def before_tool_execution(self, session, conversation_id, execution: ToolExecution) -> ToolExecution:
            arguments = execution.raw_tool_call.arguments
            return execution.model_copy(
                update={
                    "raw_tool_call": execution.raw_tool_call.model_copy(
                        update={
                            "arguments": {"a": arguments["a"] * 10, "b": arguments["b"] * 10},
                        }
                    ),
                }
            )

    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("30")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_mw_bte",
        entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="add")])},
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()]),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
        middleware=[Args10xMiddleware()],
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    # add(10, 20) = 30, not add(1, 2) = 3 — the rewritten call is what ran and
    # what the durable record shows.
    assert runner.session.entries["te1"] == ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 10, "b": 20}),
        tool_spec=ADD_SPEC,
        tool_spec_id=ADD_SPEC.spec_id(),
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(content=[TextContent(text="30")]),
        approval_status=ApprovalStatus.ALLOWED,
        approval_decisions=[ALLOW_1000],
        started_at=1000,
        ended_at=1000,
        updated_at=1000,
    )
    # the original request block in the assistant message is untouched
    assert runner.session.entries["a1"].parts == [
        ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
    ]


async def test_middleware_before_tool_execution_effective_call_is_what_prepare_resolves():
    # The hook runs AHEAD of the registry's `prepare()`, so an effective call
    # naming a tool that does not exist fails RESOLUTION: NOT_FOUND with
    # `details["phase"] == "prepare"`, `started_at` unset (`dispatched` False)
    # and nothing ever resolved.
    class RerouteMiddleware:
        def before_tool_execution(self, session, conversation_id, execution: ToolExecution) -> ToolExecution:
            return execution.model_copy(
                update={
                    "raw_tool_call": execution.raw_tool_call.model_copy(
                        update={"name": "nope"},
                    ),
                }
            )

    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("done")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_mw_bte_prepare",
        entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="add")])},
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    registry = FakeToolRegistry([AddTool()])
    runner = DeterministicRunner(
        session,
        tool_registry=registry,
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
        middleware=[RerouteMiddleware()],
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    assert runner.session.entries["te1"] == ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        context_tokens=5,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="nope", arguments={"a": 1, "b": 2}),
        tool_spec=ADD_SPEC,  # the birth spec stands; there is no re-snapshot
        tool_spec_id=ADD_SPEC.spec_id(),
        status=ExecutionStatus.NOT_FOUND,
        error=ToolExecutionError(
            error_type="ToolNotFound",
            error_message="Unknown tool: 'nope'.",
            details={"phase": "prepare"},
        ),
        approval_status=ApprovalStatus.ALLOWED,
        approval_decisions=[ALLOW_1000],
        ended_at=1000,
        updated_at=1000,
    )
    assert registry.prepared == []


async def test_before_tool_execution_is_dispatch_only_while_after_sees_every_outcome():
    # `before_tool_execution` means "a dispatch attempt is starting" and
    # nothing else: a call terminal at birth (unknown tool) and a call denied
    # at decision time never dispatch, so neither reaches it — only the
    # allowed one does, PENDING, and it is not re-visited at its terminal
    # transition. `after_tool_execution` is the universal terminal tail and
    # sees all three.
    class StatusRecorder:
        def __init__(self) -> None:
            self.seen: list[tuple[str, ExecutionStatus]] = []
            self.outcomes: list[tuple[str, ExecutionStatus]] = []

        def before_tool_execution(self, session, conversation_id, execution: ToolExecution) -> ToolExecution:
            self.seen.append((execution.tool_call_id, execution.status))
            return execution

        def after_tool_execution(
            self,
            session,
            conversation_id,
            execution: ToolExecution,
            exception: Exception | None = None,
        ) -> ToolExecution:
            self.outcomes.append((execution.tool_call_id, execution.status))
            return execution

    recorder = StatusRecorder()
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [
                    faux_tool_call("nope", {"x": 1}, id="tc1"),
                    faux_tool_call("add", {"a": 1, "b": 2}, id="tc2"),
                    faux_tool_call("multiply", {"a": 3, "b": 4}, id="tc3"),
                ],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("done")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_mw_bte_all",
        entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="go")])},
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    registry = FakeToolRegistry(
        [AddTool(), MultiplyTool()],
        decisions=[ALLOW_1000, DENY_1000],  # tc2, then tc3
    )
    runner = DeterministicRunner(
        session,
        tool_registry=registry,
        provider=faux,
        ids=["ts", "a1", "te1", "te2", "te3", "a2", "tf"],
        now=1000,
        middleware=[recorder],
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    assert recorder.seen == [
        ("tc2", ExecutionStatus.PENDING),  # allowed, about to dispatch — the only one
    ]
    assert recorder.outcomes == [
        ("tc1", ExecutionStatus.NOT_FOUND),  # terminal at birth
        ("tc3", ExecutionStatus.REJECTED),  # denied at decision time
        ("tc2", ExecutionStatus.COMPLETED),  # dispatched and returned
    ]


# ── after_tool_execution ──────────────────────────────────────────────────────


async def test_middleware_after_tool_execution_return_persisted():
    class ResultTransformMiddleware:
        def after_tool_execution(
            self,
            session,
            conversation_id,
            execution: ToolExecution,
            exception: Exception | None = None,
        ) -> ToolExecution:
            return execution.model_copy(
                update={
                    "result": execution.result.model_copy(
                        update={
                            "content": [TextContent(text="RESULT_MODIFIED")],
                        }
                    ),
                }
            )

    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("ok")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_mw_ate",
        entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="add")])},
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()]),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
        middleware=[ResultTransformMiddleware()],
    )

    async with runner.run() as run:
        events = [event async for event in run]

    # `context_tokens` stays the 0 derived from the tool's own "3": context
    # settles BEFORE the hook and is never recalculated behind it.
    persisted = ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        tool_spec=ADD_SPEC,
        tool_spec_id=ADD_SPEC.spec_id(),
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(content=[TextContent(text="RESULT_MODIFIED")]),
        approval_status=ApprovalStatus.ALLOWED,
        approval_decisions=[ALLOW_1000],
        started_at=1000,
        ended_at=1000,
        updated_at=1000,
    )
    assert runner.session.entries["te1"] == persisted
    # the ToolExecuted event projects that same transformed execution
    assert events[3] == ToolExecuted(
        conversation_id="c1",
        tool_call_id="tc1",
        execution=persisted,
        result_text="RESULT_MODIFIED",
        is_error=False,
    )


async def test_middleware_after_tool_execution_observes_every_outcome():
    # One middleware, four outcomes in one turn: NOT_FOUND (a terminal birth —
    # registry-authored, so no live exception reaches the hook), REJECTED
    # (denied), FAILED (body raised — with the live exception), and COMPLETED.
    # The hook sees each terminal state exactly once.
    class OutcomeRecorder:
        def __init__(self) -> None:
            self.seen: list[tuple[str, ExecutionStatus, type | None]] = []

        def after_tool_execution(
            self,
            session,
            conversation_id,
            execution: ToolExecution,
            exception: Exception | None = None,
        ) -> ToolExecution:
            self.seen.append(
                (
                    execution.tool_call_id,
                    execution.status,
                    type(exception) if exception is not None else None,
                )
            )
            return execution

    recorder = OutcomeRecorder()
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [
                    faux_tool_call("nope", {}, id="tc1"),
                    faux_tool_call("multiply", {"a": 3, "b": 4}, id="tc2"),
                    faux_tool_call("boom", {"a": 1, "b": 2}, id="tc3"),
                    faux_tool_call("add", {"a": 1, "b": 2}, id="tc4"),
                ],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("done")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_mw_ate_all",
        entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="go")])},
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    registry = FakeToolRegistry(
        [AddTool(), MultiplyTool(), RaisingTool()],
        decisions=[DENY_1000, ALLOW_1000, ALLOW_1000],  # tc2, tc3, tc4
    )
    runner = DeterministicRunner(
        session,
        tool_registry=registry,
        provider=faux,
        ids=["ts", "a1", "te1", "te2", "te3", "te4", "a2", "tf"],
        now=1000,
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
        def build_model_string(self, session, conversation_id, model_string: str, model_cfg) -> str:
            return model_string + "-v1"

    class AppendV2Middleware:
        def build_model_string(self, session, conversation_id, model_string: str, model_cfg) -> str:
            return model_string + "-v2"

    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message([faux_text("ok")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_mw_order",
        entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")])},
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        provider=faux,
        ids=["ts", "a1", "tf"],
        now=1000,
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

        def before_llm_call(self, session, conversation_id, messages, system_message):
            return messages, (system_message or "") + self.suffix

    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message([faux_text("ok")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_mw_order2",
        entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")])},
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        provider=faux,
        ids=["ts", "a1", "tf"],
        now=1000,
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
    session = object()
    cid = "c1"
    message = object()
    call = object()
    execution = object()
    decision = object()

    assert mixin.build_model_string(session, cid, "openrouter:openai/gpt-4o-mini", MODEL) == (
        "openrouter:openai/gpt-4o-mini"
    )
    assert mixin.build_tool_list(session, cid, [ADD_SPEC, MULTIPLY_SPEC]) == [ADD_SPEC, MULTIPLY_SPEC]
    assert mixin.before_post_message(session, cid, [TextContent(text="hello")]) == [
        TextContent(text="hello"),
    ]
    assert mixin.before_entry_written(session, cid, entry) is entry
    assert mixin.before_llm_call(session, cid, ["m1"], "sys") == (["m1"], "sys")
    assert mixin.before_llm_call(session, cid, ["m1"], None) == (["m1"], None)
    assert mixin.after_llm_response(session, cid, message) is message
    assert mixin.before_tool_creation(session, cid, call) is call
    assert mixin.after_tool_creation(session, cid, execution) is execution
    assert mixin.after_tool_creation(session, cid, execution, ValueError("x")) is execution
    assert mixin.before_permission_check(session, cid, execution) is execution
    assert mixin.after_permission_decision(session, cid, decision, execution) is decision
    assert mixin.before_tool_execution(session, cid, execution) is execution
    assert mixin.after_tool_execution(session, cid, execution) is execution
    assert mixin.after_tool_execution(session, cid, execution, ValueError("x")) is execution


def test_mixin_exposes_exactly_the_thirteen_hooks_with_the_scope_prefix():
    import inspect

    hooks = {
        name: list(inspect.signature(fn).parameters)[:3]
        for name, fn in inspect.getmembers(AgentMiddlewareMixin, inspect.isfunction)
        if not name.startswith("_")
    }

    assert hooks == {
        "adapt_tool_declarations": ["self", "session", "conversation_id"],
        "after_llm_response": ["self", "session", "conversation_id"],
        "after_permission_decision": ["self", "session", "conversation_id"],
        "after_tool_creation": ["self", "session", "conversation_id"],
        "after_tool_execution": ["self", "session", "conversation_id"],
        "before_entry_written": ["self", "session", "conversation_id"],
        "before_llm_call": ["self", "session", "conversation_id"],
        "before_permission_check": ["self", "session", "conversation_id"],
        "before_post_message": ["self", "session", "conversation_id"],
        "before_tool_creation": ["self", "session", "conversation_id"],
        "before_tool_execution": ["self", "session", "conversation_id"],
        "build_model_string": ["self", "session", "conversation_id"],
        "build_tool_list": ["self", "session", "conversation_id"],
    }


def test_mixin_has_no_build_messages_hook():
    # conversation projection is a runner collaborator (ConversationProjector),
    # not a middleware stage
    assert not hasattr(AgentMiddlewareMixin, "build_messages")


async def test_mixin_subclass_partial_override_does_not_clobber_post_message():
    class OnlyResponse(AgentMiddlewareMixin):  # subclass, override one hook
        def after_llm_response(self, session, conversation_id, message):
            return message

    session = make_session(
        id="s_mw_sub",
        conversations={"c1": conversation("c1", [], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        ids=["u1"],
        now=1000,
        middleware=[OnlyResponse()],
    )

    # The inherited before_post_message must NOT blank the text
    runner.post_message("hello world")

    assert runner.session.entries["u1"].parts == [TextContent(text="hello world")]


async def test_mixin_subclass_override_applies_and_inherited_hooks_pass_full_turn_through():
    class OnlyModelSuffix(AgentMiddlewareMixin):
        def build_model_string(self, session, conversation_id, model_string: str, llm_cfg) -> str:
            return model_string + "-routed"

    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("3")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_mw_mixin_run",
        entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="add")])},
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool()]),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
        middleware=[OnlyModelSuffix()],
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    # The overridden hook applied…
    assert faux.requests[0].model == "test-model-routed"
    # …and every inherited hook passed its stage through untouched: the tool
    # ran with the original args, the result and final answer were stored.
    assert runner.session.entries["te1"] == ToolExecution(
        id="te1",
        conversation_id="c1",
        parent_id="a1",
        created_at=1000,
        tool_call_id="tc1",
        raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
        tool_spec=ADD_SPEC,
        tool_spec_id=ADD_SPEC.spec_id(),
        status=ExecutionStatus.COMPLETED,
        result=ExecutionResult(content=[TextContent(text="3")]),
        approval_status=ApprovalStatus.ALLOWED,
        approval_decisions=[ALLOW_1000],
        started_at=1000,
        ended_at=1000,
        updated_at=1000,
    )
    assert runner.session.entries["a2"] == AssistantMessage(
        id="a2",
        parent_id="te1",
        created_at=1000,
        parts=[TextContent(text="3")],
        # The ROUTED model, not the session's: provenance has to name the
        # model that produced the turn, because the transports compare it
        # against the model being called to decide whether a thinking
        # signature may be replayed.
        llm_config=MODEL.model_copy(update={"model": "test-model-routed"}),
        stop_reason="stop",
    )


# ── compaction ────────────────────────────────────────────────────────────────


async def test_before_entry_written_sees_every_entry_a_compaction_writes():
    # The bracket markers, the entry on append AND on both mutations that
    # change it, and every entry the plan creates.
    class EntryRecorder:
        def __init__(self) -> None:
            self.seen: list[tuple] = []

        def before_entry_written(self, session, conversation_id, entry):
            self.seen.append((entry.type, entry.id))
            return entry

    recorder = EntryRecorder()
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(plan=_frame_and_fold),
        middleware=[recorder],
        ids=["ts_c", "cmp", "new1", "tf_c", "c2"],
        now=1000,
    )

    runner.schedule_compaction()
    await runner.run()

    assert recorder.seen == [
        ("turn_start", "ts_c"),  # the bracket opens
        ("compaction", "cmp"),  # the entry is appended
        ("compaction", "cmp"),  # the started_at stamp
        ("compaction", "cmp"),  # the content mutation, at the commit point
        ("user", "new1"),  # the entry the plan created
        ("turn_finish", "tf_c"),  # the bracket closes
    ]


async def test_the_turn_hooks_are_not_invoked_for_the_summarization_call():
    # The policy owns its LLM call end to end. A middleware that appends a
    # trailing reminder, or routes the model by turn count, must not silently
    # start corrupting summarization requests — it has no argument telling it
    # which call it is in.
    class TurnHookCounter:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def before_llm_call(self, session, conversation_id, messages, system_message):
            self.calls.append("before_llm_call")
            return messages, system_message

        def after_llm_response(self, session, conversation_id, message):
            self.calls.append("after_llm_response")
            return message

        def build_model_string(self, session, conversation_id, model_string, llm_cfg):
            self.calls.append("build_model_string")
            return model_string

        def build_tool_list(self, session, conversation_id, tools):
            self.calls.append("build_tool_list")
            return tools

    counter = TurnHookCounter()
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message([faux_text("X is 42.")], finish_reason="stop"),
        ]
    )
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        provider=faux,
        context_manager=FakeContextManager(plan=_fold_everything),
        middleware=[counter],
        ids=["ts_c", "cmp", "tf_c", "c2", "u5", "ts4", "a4", "tf4"],
        now=1000,
    )

    runner.schedule_compaction()
    await runner.run()

    assert counter.calls == []  # a compaction-only drive touched none of them

    runner.post_message("what is X?")
    await runner.run()

    assert counter.calls == [
        "build_model_string",
        "before_llm_call",
        "build_tool_list",
        "after_llm_response",
    ]


async def test_before_entry_written_may_redact_the_summary_before_it_persists():
    class Redactor:
        def before_entry_written(self, session, conversation_id, entry):
            if isinstance(entry, CompactionEntry) and entry.parts:
                return entry.model_copy(
                    update={"parts": [TextContent(text="[redacted]")]},
                )
            return entry

    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(
        session,
        context_manager=FakeContextManager(plan=_fold_everything),
        middleware=[Redactor()],
        ids=["ts_c", "cmp", "tf_c", "c2"],
        now=1000,
    )

    runner.schedule_compaction()
    await runner.run()

    assert runner.session.entries["cmp"].parts == [TextContent(text="[redacted]")]
    assert ConversationProjector().project(
        main_conversation(runner.session).nodes,
        runner.session.entries,
    ) == [LucaUserMessage(content=[TextBlock(text="[redacted]")])]


def _fold_everything(session, nodes, entry):
    return CompactionPlan(
        entry=entry.model_copy(update={"parts": [TextContent(text="the story")]}),
        nodes=[entry.id],
    )


def _frame_and_fold(session, nodes, entry):
    return CompactionPlan(
        entry=entry.model_copy(update={"parts": [TextContent(text="the story")]}),
        nodes=[UserMessage(parts=[TextContent(text="[compacted]")]), entry.id],
    )


async def test_a_routed_turn_records_the_model_it_actually_ran_on():
    # Provenance is load-bearing: the transports compare it against the model
    # being called to decide whether a thinking signature may be replayed, so
    # recording the session config for a routed turn breaks reasoning replay
    # in both directions.
    class RouteElsewhere(AgentMiddlewareMixin):
        def build_model_string(self, session, conversation_id, model_string: str, llm_cfg) -> str:
            return "openai:gpt-5.4-codex"

    faux = FauxProvider()
    faux.set_responses([faux_assistant_message([faux_text("done")], finish_reason="stop")])
    session = make_session(
        id="s_mw_routed",
        entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="hi")])},
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        provider=faux,
        ids=["ts", "a1", "tf"],
        now=1000,
        middleware=[RouteElsewhere()],
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    assert runner.session.entries["a1"].llm_config == LLMConfig(provider="openai", model="gpt-5.4-codex")


# ── the scope prefix ──────────────────────────────────────────────────────────


async def test_every_hook_receives_the_live_session_and_the_driven_conversation():
    # The session is the runner's own object, not a copy — a hook may read
    # state written moments earlier in the same drive. The id is the
    # conversation being driven; the single-conversation case is where a
    # regression would hide, because "the main one" and "the right one" agree.
    class ScopeRecorder(AgentMiddlewareMixin):
        def __init__(self) -> None:
            self.scopes: list[tuple[str, int, str]] = []

        def build_model_string(self, session, conversation_id, model_string, llm_cfg):
            self.scopes.append(("build_model_string", id(session), conversation_id))
            return model_string

        def before_entry_written(self, session, conversation_id, entry):
            self.scopes.append(("before_entry_written", id(session), conversation_id))
            return entry

        def before_post_message(self, session, conversation_id, parts):
            self.scopes.append(("before_post_message", id(session), conversation_id))
            return parts

    recorder = ScopeRecorder()
    faux = FauxProvider()
    faux.set_responses([faux_assistant_message([faux_text("ok")], finish_reason="stop")])
    session = make_session(
        id="s_mw_scope",
        conversations={"c1": conversation("c1", [], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        provider=faux,
        ids=["u1", "ts", "a1", "tf"],
        now=1000,
        middleware=[recorder],
    )

    runner.post_message("hi")
    async with runner.run() as run:
        _ = [event async for event in run]

    assert recorder.scopes == [
        ("before_post_message", id(runner.session), "c1"),
        ("before_entry_written", id(runner.session), "c1"),  # the user message
        ("before_entry_written", id(runner.session), "c1"),  # TurnStart
        ("build_model_string", id(runner.session), "c1"),
        ("before_entry_written", id(runner.session), "c1"),  # the assistant message
        ("before_entry_written", id(runner.session), "c1"),  # TurnFinish
    ]


async def test_chained_after_hooks_all_receive_the_same_live_exception():
    # `exception` is CONTEXT, not a transformed value: it passes unchanged to
    # every middleware in the chain, and it is the identical object the
    # runner's error converter saw.
    seen: list[tuple[str, int]] = []

    class First:
        def after_tool_execution(self, session, conversation_id, execution, exception=None):
            seen.append(("first", id(exception)))
            return execution.model_copy(update={"extras": {"chain": ["first"]}})

    class Second:
        def after_tool_execution(self, session, conversation_id, execution, exception=None):
            seen.append(("second", id(exception)))
            return execution.model_copy(
                update={"extras": {"chain": [*execution.extras["chain"], "second"]}},
            )

    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("boom", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("done")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_mw_exc_chain",
        entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="go")])},
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([RaisingTool()]),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
        middleware=[First(), Second()],
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    # one exception object, seen by both, in declared order
    assert [name for name, _ in seen] == ["first", "second"]
    assert len({exc_id for _, exc_id in seen}) == 1
    # and the value chained through them both
    assert runner.session.entries["te1"].extras == {"chain": ["first", "second"]}
    assert runner.session.entries["te1"].status == ExecutionStatus.FAILED


# ── before_tool_creation ──────────────────────────────────────────────────────


async def test_before_tool_creation_rewrites_the_call_the_registry_sees():
    class RerouteToMultiply:
        def before_tool_creation(self, session, conversation_id, call: ToolCall) -> ToolCall:
            return call.model_copy(update={"name": "multiply"})

    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 3, "b": 4}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("done")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_mw_btc",
        entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="go")])},
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    registry = FakeToolRegistry([AddTool(), MultiplyTool()])
    runner = DeterministicRunner(
        session,
        tool_registry=registry,
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
        middleware=[RerouteToMultiply()],
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    execution = runner.session.entries["te1"]
    # the birth resolved the REWRITTEN tool, and the effective call is durable
    assert execution.raw_tool_call == ToolCall(id="tc1", name="multiply", arguments={"a": 3, "b": 4})
    assert execution.tool_spec == MULTIPLY_SPEC
    assert registry.prepared == ["multiply"]
    assert execution.result.content == [TextContent(text="12")]


async def test_before_tool_creation_cannot_reach_a_private_tool_by_renaming():
    # The private check reads the EFFECTIVE name, so rewriting into a private
    # tool records NOT_FOUND exactly as a model guessing the name would —
    # otherwise "private" would be bypassable from middleware by accident.
    class RenameToSecret:
        def before_tool_creation(self, session, conversation_id, call: ToolCall) -> ToolCall:
            return call.model_copy(update={"name": "secret"})

    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("done")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_mw_btc_private",
        entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="go")])},
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool(), PrivateTool()]),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
        middleware=[RenameToSecret()],
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    execution = runner.session.entries["te1"]
    assert execution.status == ExecutionStatus.NOT_FOUND
    assert execution.error.error_message == "Unknown tool: 'secret'."


# ── after_tool_creation ───────────────────────────────────────────────────────


async def test_after_tool_creation_sees_the_effective_birth_state():
    class BirthRecorder:
        def __init__(self) -> None:
            self.seen: list[tuple[str, ExecutionStatus, str | None]] = []

        def after_tool_creation(self, session, conversation_id, execution, exception=None):
            self.seen.append(
                (
                    execution.tool_call_id,
                    execution.status,
                    type(exception).__name__ if exception is not None else None,
                )
            )
            return execution

    recorder = BirthRecorder()
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [
                    faux_tool_call("add", {"a": 1, "b": 2}, id="tc1"),
                    faux_tool_call("nope", {}, id="tc2"),
                    faux_tool_call("secret", {"a": 1, "b": 2}, id="tc3"),
                ],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("done")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_mw_atc",
        entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="go")])},
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([AddTool(), PrivateTool()]),
        provider=faux,
        ids=["ts", "a1", "te1", "te2", "te3", "a2", "tf"],
        now=1000,
        middleware=[recorder],
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    # A healthy birth arrives PENDING. A terminal birth the REGISTRY authored
    # carries no live exception — the registry wrote the error itself. Only a
    # FRAMEWORK-synthesized draft (here: a model naming a private tool) hands
    # the hook the exception that produced it.
    assert recorder.seen == [
        ("tc1", ExecutionStatus.PENDING, None),
        ("tc2", ExecutionStatus.NOT_FOUND, None),
        ("tc3", ExecutionStatus.NOT_FOUND, "ToolNotFound"),
    ]


async def test_after_tool_creation_can_terminalize_a_birth_before_it_reaches_decide():
    # The returned execution decides the next lifecycle step: a middleware
    # that refuses a call at birth sends it straight to the outcome tail, so
    # the registry is never asked to decide and the body never runs.
    class RefuseAdd:
        def after_tool_creation(self, session, conversation_id, execution, exception=None):
            if execution.raw_tool_call.name != "add":
                return execution
            return execution.model_copy(
                update={
                    "status": ExecutionStatus.REFUSED,
                    "ended_at": 1000,
                    "error": ToolExecutionError(
                        error_type="PolicyRefusal",
                        error_message="add is disabled this turn.",
                    ),
                }
            )

    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("done")], finish_reason="stop"),
        ]
    )
    session = make_session(
        id="s_mw_atc_refuse",
        entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="go")])},
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    registry = FakeToolRegistry([AddTool()])
    runner = DeterministicRunner(
        session,
        tool_registry=registry,
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=1000,
        middleware=[RefuseAdd()],
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    execution = runner.session.entries["te1"]
    assert execution.status == ExecutionStatus.REFUSED
    assert execution.approval_status is None  # decide() was never reached
    assert registry.seen == []  # …and the registry was never asked
    assert registry.prepared == []  # …and the body never ran


# ── after_llm_response invocation count ───────────────────────────────────────


@pytest.mark.parametrize("streaming", [False, True])
async def test_after_llm_response_fires_exactly_once_per_complete_response(streaming):
    # Streaming and non-streaming converge on ONE call site: the stream is
    # assembled to its terminal message first, and the hook runs on that.
    class Counter:
        def __init__(self) -> None:
            self.calls = 0

        def after_llm_response(self, session, conversation_id, message):
            self.calls += 1
            return message

    counter = Counter()
    faux = FauxProvider()
    faux.set_responses([faux_assistant_message([faux_text("ok")], finish_reason="stop")])
    session = make_session(
        id="s_mw_alr_once",
        entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")])},
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        provider=faux,
        ids=["ts", "a1", "tf"],
        now=1000,
        middleware=[counter],
    )

    async with runner.run(streaming=streaming) as run:
        _ = [event async for event in run]

    assert counter.calls == 1
    assert runner.session.entries["a1"].parts == [TextContent(text="ok")]


@pytest.mark.parametrize("streaming", [False, True])
async def test_after_llm_response_does_not_fire_for_an_incomplete_response(streaming):
    # No complete assistant message exists, so there is nothing to transform.
    class Counter:
        def __init__(self) -> None:
            self.calls = 0

        def after_llm_response(self, session, conversation_id, message):
            self.calls += 1
            return message

    counter = Counter()
    faux = FauxProvider()
    faux.set_responses([faux_assistant_message([], error=faux_error("provider 500"))])
    session = make_session(
        id="s_mw_alr_none",
        entries={"u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="Hi")])},
        conversations={"c1": conversation("c1", ["u1"], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    runner = DeterministicRunner(
        session,
        provider=faux,
        ids=["ts", "tf"],
        now=1000,
        middleware=[counter],
    )

    with pytest.raises(ClientError):  # the provider failure is re-raised (StreamError when streaming)
        async with runner.run(streaming=streaming) as run:
            _ = [event async for event in run]

    assert counter.calls == 0
    assert main_conversation(runner.session).nodes[-1] == "tf"  # the turn closed ERRORED
