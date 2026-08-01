"""The `post_message` acceptance matrix and the mid-turn message stories.

The matrix tests load a known session cold and assert one post's verdict —
accept (the entry lands exactly where the path says) or reject (the dedicated
exception for the two transient rejections, plain `AgentError` for the
persistent ones). The stories drive the runner end to end: a post between tool
rounds, the close race (a message landing during the final LLM call forces one
extra round instead of a COMPLETED close), the burial rules, and precedence at
the close site.

House style: precondition → one action → full-object postcondition, through
`DeterministicRunner` + `FauxProvider`. Posting *during* a drive is simulated
deterministically — a tool body or an `after_llm_response` middleware posts —
never with timing tricks. Compaction-bracket rejections live with the other
compaction scenarios in `test_runner_compaction.py`; the live subagent-tree
stories in `tests/agent/subagents/test_post_messages.py`.
"""

import pytest

from luca.agent.core.exceptions import (
    AgentError,
    ConversationCancellingError,
    SubagentsActiveError,
)
from luca.agent.core.models import (
    AgentSession,
    ApprovalDecision,
    ApprovalOption,
    ApprovalStatus,
    AssistantMessage,
    CancelRequested,
    ChildConversation,
    ConversationStatus,
    ExecutionResult,
    ExecutionStatus,
    RuntimeConfig,
    SessionConfig,
    TextContent,
    ToolCall,
    ToolExecution,
    TurnFinish,
    TurnOutcome,
    TurnStart,
    Usage,
    UserMessage,
)
from luca.agent.core.projection import CANCELLED_TURN_MARKER
from luca.client.testing import (
    FauxProvider,
    faux_assistant_message,
    faux_text,
    faux_tool_call,
)
from luca.client.types import TextBlock as LucaTextBlock, UserMessage as LucaUserMessage
from tests.agent.scenarios import (
    ADD_SPEC,
    CANCEL_PARKED_SESSION,
    GATED_SESSION,
    MODEL,
    POST_COMPACTION_SESSION,
    POST_FAILURE_SESSION,
    UNDECIDED_SESSION,
    AddTool,
    BinaryArgs,
    DeterministicRunner,
    FakeTool,
    FakeToolRegistry,
    conversation,
    main_conversation,
    make_session,
    spec,
)

ADD_SPEC_ID = ADD_SPEC.spec_id()


def idle_session(session_id: str, **runtime) -> AgentSession:
    """An empty IDLE session; `runtime` feeds `RuntimeConfig` overrides."""
    return make_session(
        id=session_id,
        conversations={"c1": conversation("c1", [])},
        main_conversation_id="c1",
        session_config=SessionConfig(
            llm_config=MODEL,
            runtime_config=RuntimeConfig(**runtime),
        ),
    )


SPAWN_SPEC = spec(
    "spawn_subagent",
    output_schema={
        "type": "object",
        "properties": {"is_subagent_spawn": {"type": "boolean"}},
    },
)
_SPAWN_CALL = ToolCall(
    id="tc1",
    name="spawn_subagent",
    arguments={"prompt": "Research A", "description": "research", "task_id": "t1"},
)

# A parent mid-orchestration, exactly as it sits on disk: the spawn execution
# completed with its handshake payload, the `ChildConversation` link is
# unresolved, and the child's own turn is open. Loading it cold is what proves
# the D13 rejection holds across a reload — the predicate reads only durable
# entries.
SUBAGENTS_ACTIVE_SESSION = make_session(
    id="s_subagents_active",
    entries={
        "u1": UserMessage(id="u1", created_at=500, parts=[TextContent(text="go")]),
        "ts": TurnStart(id="ts", parent_id="u1", created_at=500),
        "a1": AssistantMessage(
            id="a1",
            parent_id="ts",
            created_at=500,
            parts=[_SPAWN_CALL],
            llm_config=MODEL,
            stop_reason="tool_use",
        ),
        "te1": ToolExecution(
            id="te1",
            conversation_id="c1",
            parent_id="a1",
            created_at=500,
            tool_call_id="tc1",
            raw_tool_call=_SPAWN_CALL,
            tool_spec=SPAWN_SPEC,
            status=ExecutionStatus.COMPLETED,
            result=ExecutionResult(
                content=[TextContent(text="spawned")],
                structured_content={
                    "is_subagent_spawn": True,
                    "task_id": "t1",
                    "prompt": "Research A",
                    "description": "research",
                    "process_subagent_result_tool_name": "create_conversation_result",
                },
            ),
            approval_status=ApprovalStatus.ALLOWED,
            approval_decisions=[
                ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=500),
            ],
            started_at=500,
            ended_at=500,
            updated_at=500,
        ),
        "ch1": ChildConversation(
            id="ch1",
            parent_id="te1",
            created_at=500,
            conversation_id="c2",
            tool_execution_id="te1",
        ),
        "u_seed": UserMessage(
            id="u_seed",
            created_at=500,
            parts=[TextContent(text="Research A")],
        ),
        "ts_child": TurnStart(id="ts_child", parent_id="u_seed", created_at=500),
    },
    conversations={
        "c1": conversation("c1", ["u1", "ts", "a1", "te1", "ch1"]),
        "c2": conversation("c2", ["u_seed", "ts_child"], depth=1),
    },
    main_conversation_id="c1",
    session_config=SessionConfig(llm_config=MODEL),
)

# The same tree with the child finished and its link resolved: the parent
# accepts posts again (mid-turn), the finished child refuses them.
SUBAGENTS_RESOLVED_SESSION = SUBAGENTS_ACTIVE_SESSION.model_copy(
    deep=True,
    update={"id": "s_subagents_resolved"},
)
SUBAGENTS_RESOLVED_SESSION.entries["ch1"] = SUBAGENTS_RESOLVED_SESSION.entries["ch1"].model_copy(
    update={"execution_result": ExecutionResult(content=[TextContent(text="A done")])},
)
SUBAGENTS_RESOLVED_SESSION.entries["a_child"] = AssistantMessage(
    id="a_child",
    parent_id="ts_child",
    created_at=500,
    parts=[TextContent(text="A done")],
    llm_config=MODEL,
    stop_reason="stop",
)
SUBAGENTS_RESOLVED_SESSION.entries["tf_child"] = TurnFinish(
    id="tf_child",
    parent_id="a_child",
    created_at=500,
)
SUBAGENTS_RESOLVED_SESSION.conversations["c2"].nodes += ["a_child", "tf_child"]

# A spawned child not yet driven: seed only, no bracket — BUSY with no open
# turn, the queued-behind-the-worker-pool shape.
SEEDED_CHILD_SESSION = SUBAGENTS_ACTIVE_SESSION.model_copy(
    deep=True,
    update={"id": "s_seeded_child"},
)
SEEDED_CHILD_SESSION.entries.pop("ts_child")
SEEDED_CHILD_SESSION.conversations["c2"].nodes.remove("ts_child")

# The active tree with a cancel parked on the parent: CANCELLING outranks the
# subagents check.
CANCELLING_PARENT_SESSION = SUBAGENTS_ACTIVE_SESSION.model_copy(
    deep=True,
    update={"id": "s_cancelling_parent"},
)
CANCELLING_PARENT_SESSION.entries["cr"] = CancelRequested(
    id="cr",
    parent_id="ch1",
    created_at=600,
)
main_conversation(CANCELLING_PARENT_SESSION).nodes.append("cr")


class PostingTool(FakeTool):
    """An `add` whose body posts a steering message — the deterministic
    stand-in for a user typing while a tool runs mid-turn."""

    name = "add"
    description = "Add two numbers."
    Args = BinaryArgs

    def __init__(self) -> None:
        self.post = None  # wired by the test to runner.post_message

    async def _execute(self, args, session, conversation_id, *, cancellation_token) -> str:
        self.post()
        return str(args["a"] + args["b"])


class PostDuringResponse:
    """Middleware double: posts while a response is being handled.
    `after_llm_response` fires before the response is recorded, which is
    indistinguishable from a post landing during the in-flight call — the
    deterministic simulation of the close race. `texts_by_call` maps the
    1-based LLM-call number to the messages to post during it."""

    def __init__(self, texts_by_call: dict[int, list[str]]) -> None:
        self.post = None  # wired by the test to runner.post_message
        self.calls = 0
        self.texts_by_call = texts_by_call

    def after_llm_response(self, message):
        self.calls += 1
        for text in self.texts_by_call.get(self.calls, []):
            self.post(text)
        return message


class PostThenCancelDuringResponse:
    """Posts a message and then cancels, both during the final call — the
    precedence scenario: the unconsumed cancel must control the close."""

    def __init__(self) -> None:
        self.post = None
        self.cancel = None
        self.calls = 0

    def after_llm_response(self, message):
        self.calls += 1
        if self.calls == 1:
            self.post("wait!")
            self.cancel()
        return message


# ── the acceptance matrix ─────────────────────────────────────────────────────


async def test_post_message_queues_behind_an_unanswered_message():
    # Queueing is allowed: a second trailing message lands behind the first
    # and derives nothing new — the next turn answers both.
    session = idle_session("s_queue")
    runner = DeterministicRunner(session, ids=["u1", "u2"], now=1000)

    runner.post_message("First")
    runner.post_message("Second")

    assert runner.busy()
    assert main_conversation(runner.session).nodes == ["u1", "u2"]


async def test_one_turn_answers_two_queued_trailing_messages():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message([faux_text("Both answered.")], finish_reason="stop"),
        ]
    )
    session = idle_session("s_queue_turn")
    runner = DeterministicRunner(
        session,
        provider=faux,
        ids=["u1", "u2", "ts", "a1", "tf"],
        now=1000,
    )
    runner.post_message("First")
    runner.post_message("Second")

    result = await runner.run()

    assert result.outcome == TurnOutcome.COMPLETED
    assert faux.requests[0].messages == [
        LucaUserMessage(content=[LucaTextBlock(text="First")]),
        LucaUserMessage(content=[LucaTextBlock(text="Second")]),
    ]
    assert runner.session == AgentSession(
        id="s_queue_turn",
        entries={
            "u1": UserMessage(
                id="u1",
                parent_id=None,
                created_at=1000,
                parts=[TextContent(text="First")],
                context_tokens=1,  # len("First") // 4
            ),
            "u2": UserMessage(
                id="u2",
                parent_id="u1",
                created_at=1000,
                parts=[TextContent(text="Second")],
                context_tokens=1,  # len("Second") // 4
            ),
            "ts": TurnStart(id="ts", parent_id="u2", created_at=1000),
            "a1": AssistantMessage(
                id="a1",
                parent_id="ts",
                created_at=1000,
                parts=[TextContent(text="Both answered.")],
                llm_config=MODEL,
                stop_reason="stop",
                context_tokens=3,  # len("Both answered.") // 4
            ),
            "tf": TurnFinish(id="tf", parent_id="a1", created_at=1000),
        },
        usages={"c1": {"a1": Usage(conversation_id="c1", entry_id="a1")}},
        conversations={
            "c1": conversation(
                "c1",
                ["u1", "u2", "ts", "a1", "tf"],
                updated_at=1000,
            )
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    assert runner.idle()


async def test_post_message_accepts_while_blocked_at_a_gate():
    # The mid-turn append into a BLOCKED turn. Nothing is unblocked: the
    # message waits with the turn.
    session = GATED_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(session, ids=["um"], now=1000)

    msg_id = runner.post_message("Prefer the safe route")

    assert msg_id == "um"
    assert main_conversation(runner.session).nodes == ["u1", "ts", "a1", "te1", "um"]
    assert runner.blocked()


async def test_post_message_accepts_into_an_open_resumable_bracket():
    # BUSY with an open turn — the mid-turn append is the feature.
    session = UNDECIDED_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(session, ids=["um"], now=1000)

    assert runner.busy()
    runner.post_message("also this")

    assert main_conversation(runner.session).nodes == ["u1", "ts", "a1", "te1", "um"]
    assert runner.busy()


async def test_post_message_rejects_cancelling_with_the_dedicated_exception():
    # The reloaded case: a parked cancel refuses input across a process
    # restart, because the state derives from the durable CancelRequested.
    session = CANCEL_PARKED_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(session, now=1000)

    with pytest.raises(ConversationCancellingError):
        runner.post_message("never mind")

    assert runner.cancelling()
    assert main_conversation(runner.session).nodes == ["u1", "ts", "a1", "te1", "cr"]


async def test_post_message_rejects_immediately_after_a_live_cancel():
    # Esc-then-type: cancel() is synchronous and appends the durable
    # CancelRequested before returning, so the very next post already sees
    # CANCELLING — no window where the cancel is in flight but invisible.
    session = idle_session("s_live_cancel")
    runner = DeterministicRunner(
        session,
        provider=FauxProvider(),
        ids=["u1", "ts", "cr", "tf"],
        now=1000,
    )
    runner.post_message("go")
    run = runner.start()
    runner.cancel()

    with pytest.raises(ConversationCancellingError):
        runner.post_message("wait — do it differently")

    result = await run
    assert result.outcome == TurnOutcome.CANCELLED
    assert main_conversation(runner.session).nodes == ["u1", "ts", "cr", "tf"]


async def test_post_message_is_legal_after_a_failed_turn():
    session = POST_FAILURE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(session, ids=["u2"], now=2000)

    runner.post_message("Take your time, retry.")

    assert runner.busy()
    assert main_conversation(runner.session).nodes == ["u1", "ts", "tf", "u2"]


async def test_post_message_rejects_an_unknown_conversation():
    session = idle_session("s_unknown")
    runner = DeterministicRunner(session, now=1000)

    with pytest.raises(AgentError, match="No conversation"):
        runner.post_message("hi", conversation_id="nope")


async def test_post_message_rejects_while_subagents_are_active():
    # The D13 rejection, loaded cold — the predicate reads the durable
    # `ChildConversation` links, so it holds across a reload exactly like
    # CANCELLING does.
    session = SUBAGENTS_ACTIVE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(session, now=1000)

    with pytest.raises(SubagentsActiveError):
        runner.post_message("change of plans")

    assert main_conversation(runner.session).nodes == ["u1", "ts", "a1", "te1", "ch1"]


async def test_post_message_targets_a_live_subagents_open_turn():
    # `conversation_id` reaches any conversation whose state accepts input —
    # a live subagent included. Main-only input is application policy.
    session = SUBAGENTS_ACTIVE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(session, ids=["um"], now=1000)

    msg_id = runner.post_message("Focus on the docs", conversation_id="c2")

    assert msg_id == "um"
    assert runner.session.conversations["c2"].nodes == ["u_seed", "ts_child", "um"]
    assert runner.session.entries["um"] == UserMessage(
        id="um",
        parent_id="ts_child",
        created_at=1000,
        parts=[TextContent(text="Focus on the docs")],
        context_tokens=4,  # len("Focus on the docs") // 4
    )


async def test_post_message_accepts_the_parent_once_its_children_resolved():
    # Same open turn, children resolved → posting is legal again (and the
    # close-site check then guarantees the message is answered in-turn).
    session = SUBAGENTS_RESOLVED_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(session, ids=["um"], now=1000)

    runner.post_message("now summarize")

    assert main_conversation(runner.session).nodes == ["u1", "ts", "a1", "te1", "ch1", "um"]


async def test_post_message_rejects_a_finished_subagent():
    # Its result is already resolved into the parent; nothing will ever drive
    # it again, so accepting would wedge the message forever.
    session = SUBAGENTS_RESOLVED_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(session, now=1000)

    with pytest.raises(AgentError, match="finished subagent"):
        runner.post_message("one more thing", conversation_id="c2")


async def test_post_message_accepts_a_seeded_undriven_child():
    # BUSY with no open turn: the child is queued behind the worker pool and
    # will be driven; the message projects right behind its seed prompt.
    session = SEEDED_CHILD_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(session, ids=["um"], now=1000)

    runner.post_message("extra context", conversation_id="c2")

    assert runner.session.conversations["c2"].nodes == ["u_seed", "um"]


async def test_cancelling_outranks_the_subagents_check():
    # A cancelled parent mid-wind-down raises the cancellation rejection, not
    # the subagents one — "retry after the flush" is the actionable answer.
    session = CANCELLING_PARENT_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(session, now=1000)

    with pytest.raises(ConversationCancellingError):
        runner.post_message("hello?")


async def test_post_message_rejects_an_archived_conversation_whatever_it_derives():
    # The archived path ends […, u4, ts_c, cmp, tf_c]: skipping the closed
    # compaction bracket leaves the queued question as the leaf, so the
    # ARCHIVED conversation derives BUSY — a status check would accept into a
    # conversation nothing will ever drive. The rejection is identity-based:
    # the target is some conversation's `previous_conversation_id`.
    session = POST_COMPACTION_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(session, now=1000)

    assert session.get_conversation_status("c1").status is ConversationStatus.BUSY
    with pytest.raises(AgentError, match="archived"):
        runner.post_message("hello, old me", conversation_id="c1")


# ── mid-turn stories ──────────────────────────────────────────────────────────


async def test_a_mid_turn_post_between_tool_rounds_is_answered_before_the_close():
    # ── precondition ─────────────────────────────────────────────────────────
    # An empty IDLE session; a tool whose body posts a steering message while
    # it runs; a scripted provider answering the tool round and then the final
    # text. Entry ids in creation order: u1 (post_message), ts (TurnStart),
    # a1 (tool-call AssistantMessage), te1 (ToolExecution), um (the mid-turn
    # post, landed during te1's body), a2 (final AssistantMessage), tf.
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("add", {"a": 1, "b": 2}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("3 — and noted.")], finish_reason="stop"),
        ]
    )
    session = idle_session("s_mid_turn")
    tool = PostingTool()
    runner = DeterministicRunner(
        session,
        tool_registry=FakeToolRegistry([tool]),
        provider=faux,
        ids=["u1", "ts", "a1", "te1", "um", "a2", "tf"],
        now=1000,
    )
    tool.post = lambda: runner.post_message("Keep it short")
    runner.post_message("Add 1 and 2")

    # ── action ───────────────────────────────────────────────────────────────
    result = await runner.run()

    # ── postcondition ────────────────────────────────────────────────────────
    # The message sits INSIDE the turn, after the tool round it landed during;
    # the next projection carried it, the model answered it, and only then did
    # the turn close COMPLETED.
    assert result.outcome == TurnOutcome.COMPLETED
    assert faux.requests[1].messages[-1] == LucaUserMessage(
        content=[LucaTextBlock(text="Keep it short")],
    )
    assert runner.session == AgentSession(
        id="s_mid_turn",
        entries={
            "u1": UserMessage(
                id="u1",
                parent_id=None,
                created_at=1000,
                parts=[TextContent(text="Add 1 and 2")],
                context_tokens=2,  # len("Add 1 and 2") // 4
            ),
            "ts": TurnStart(id="ts", parent_id="u1", created_at=1000),
            "a1": AssistantMessage(
                id="a1",
                parent_id="ts",
                created_at=1000,
                parts=[ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2})],
                llm_config=MODEL,
                stop_reason="tool_use",
                context_tokens=4,  # (name + JSON args) // 4
            ),
            "te1": ToolExecution(
                id="te1",
                conversation_id="c1",
                parent_id="a1",
                created_at=1000,
                tool_call_id="tc1",
                raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
                tool_spec=ADD_SPEC,
                tool_spec_id=ADD_SPEC_ID,
                extras={},
                approval_status=ApprovalStatus.ALLOWED,
                approval_decisions=[
                    ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000),
                ],
                status=ExecutionStatus.COMPLETED,
                result=ExecutionResult(
                    content=[TextContent(text="3")],
                    metadata={},
                    is_error=False,
                ),
                error=None,
                started_at=1000,
                ended_at=1000,
                cancel_signalled_at=None,
                updated_at=1000,
                is_doom_loop_flagged=False,
                context_tokens=0,
            ),
            "um": UserMessage(
                id="um",
                parent_id="te1",
                created_at=1000,
                parts=[TextContent(text="Keep it short")],
                context_tokens=3,  # len("Keep it short") // 4
            ),
            "a2": AssistantMessage(
                id="a2",
                parent_id="um",
                created_at=1000,
                parts=[TextContent(text="3 — and noted.")],
                llm_config=MODEL,
                stop_reason="stop",
                context_tokens=3,  # len("3 — and noted.") // 4
            ),
            "tf": TurnFinish(id="tf", parent_id="a2", created_at=1000),
        },
        tool_specs={ADD_SPEC_ID: ADD_SPEC},
        usages={
            "c1": {
                "a1": Usage(conversation_id="c1", entry_id="a1"),
                "a2": Usage(conversation_id="c1", entry_id="a2"),
            }
        },
        conversations={
            "c1": conversation(
                "c1",
                ["u1", "ts", "a1", "te1", "um", "a2", "tf"],
                updated_at=1000,
            )
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )
    assert runner.idle()


async def test_the_close_race_a_post_during_the_final_call_forces_an_extra_round():
    # The turn must NOT close COMPLETED over a message the model has not seen:
    # the premature final answer stays recorded, and one extra round answers
    # the steering message before the close.
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message([faux_text("Done.")], finish_reason="stop"),
            faux_assistant_message([faux_text("And the follow-up: 30.")], finish_reason="stop"),
        ]
    )
    mw = PostDuringResponse({1: ["Also add 10 and 20"]})
    runner = DeterministicRunner(
        idle_session("s_close_race"),
        provider=faux,
        ids=["u1", "ts", "um", "a1", "a2", "tf"],
        now=1000,
        middleware=[mw],
    )
    mw.post = runner.post_message
    runner.post_message("Say done")

    result = await runner.run()

    assert result.outcome == TurnOutcome.COMPLETED
    assert len(faux.requests) == 2
    # the extra round's projection carried both the steering message and the
    # premature answer (texts only — the wire shape is the projector's)
    assert [m.content[0].text for m in faux.requests[1].messages] == [
        "Say done",
        "Also add 10 and 20",
        "Done.",
    ]
    assert main_conversation(runner.session).nodes == ["u1", "ts", "um", "a1", "a2", "tf"]
    assert runner.session.entries["a1"].parts == [TextContent(text="Done.")]
    assert runner.session.entries["a2"].parts == [TextContent(text="And the follow-up: 30.")]
    assert runner.session.entries["tf"] == TurnFinish(id="tf", parent_id="a2", created_at=1000)
    assert runner.idle()


async def test_two_steering_messages_before_one_close_are_answered_by_one_extra_round():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message([faux_text("Done.")], finish_reason="stop"),
            faux_assistant_message([faux_text("Answering both.")], finish_reason="stop"),
        ]
    )
    mw = PostDuringResponse({1: ["First steer", "Second steer"]})
    runner = DeterministicRunner(
        idle_session("s_two_steers"),
        provider=faux,
        ids=["u1", "ts", "um1", "um2", "a1", "a2", "tf"],
        now=1000,
        middleware=[mw],
    )
    mw.post = runner.post_message
    runner.post_message("Say done")

    result = await runner.run()

    assert result.outcome == TurnOutcome.COMPLETED
    assert len(faux.requests) == 2  # ONE extra round answered both
    assert main_conversation(runner.session).nodes == [
        "u1",
        "ts",
        "um1",
        "um2",
        "a1",
        "a2",
        "tf",
    ]
    assert runner.idle()


async def test_a_post_while_blocked_is_answered_in_the_same_turn_after_the_gate_resolves():
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message([faux_text("3 — briefly.")], finish_reason="stop"),
        ]
    )
    session = GATED_SESSION.model_copy(deep=True)
    registry = FakeToolRegistry(
        [AddTool()],
        decisions=[ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=1000)],
    )
    runner = DeterministicRunner(
        session,
        tool_registry=registry,
        provider=faux,
        ids=["um", "a2", "tf"],
        now=1000,
    )
    runner.post_message("Be brief")

    result = await runner.run()

    assert result.outcome == TurnOutcome.COMPLETED
    # the gate resolved, the tool ran, and the next call carried the message —
    # answered in the SAME turn it landed in
    assert runner.session.entries["te1"].status is ExecutionStatus.COMPLETED
    assert faux.requests[0].messages[-1] == LucaUserMessage(
        content=[LucaTextBlock(text="Be brief")],
    )
    assert main_conversation(runner.session).nodes == ["u1", "ts", "a1", "te1", "um", "a2", "tf"]
    assert runner.idle()


async def test_precedence_an_unconsumed_cancel_beats_the_unseen_message():
    # Both land during the final call: the cancel controls the close, the
    # message is buried — no extra round runs (the script has no response for
    # one, so a wrong precedence fails loudly).
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message([faux_text("Done.")], finish_reason="stop"),
        ]
    )
    mw = PostThenCancelDuringResponse()
    runner = DeterministicRunner(
        idle_session("s_cancel_precedence"),
        provider=faux,
        ids=["u1", "ts", "um", "cr", "a1", "tf"],
        now=1000,
        middleware=[mw],
    )
    mw.post = runner.post_message
    mw.cancel = runner.cancel
    runner.post_message("Say done")

    result = await runner.run()

    assert result.outcome == TurnOutcome.CANCELLED
    assert len(faux.requests) == 1
    assert main_conversation(runner.session).nodes == ["u1", "ts", "um", "cr", "a1", "tf"]
    assert runner.session.entries["tf"] == TurnFinish(
        id="tf",
        parent_id="a1",
        created_at=1000,
        outcome=TurnOutcome.CANCELLED,
    )
    assert runner.idle()  # buried (D6): the closed bracket derives IDLE


async def test_a_buried_message_projects_into_the_next_request():
    # Buried is not lost: projection walks failed brackets, so the message
    # reaches the model with the user's next engagement.
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message([faux_text("Done.")], finish_reason="stop"),
            faux_assistant_message([faux_text("As you said…")], finish_reason="stop"),
        ]
    )
    mw = PostThenCancelDuringResponse()
    runner = DeterministicRunner(
        idle_session("s_buried_projects"),
        provider=faux,
        ids=["u1", "ts", "um", "cr", "a1", "tf", "u2", "ts2", "a2", "tf2"],
        now=1000,
        middleware=[mw],
    )
    mw.post = runner.post_message
    mw.cancel = runner.cancel
    runner.post_message("Say done")
    await runner.run()

    runner.post_message("next question")
    result = await runner.run()

    assert result.outcome == TurnOutcome.COMPLETED
    assert [m.content[0].text for m in faux.requests[1].messages] == [
        "Say done",
        "wait!",  # the buried message, surfaced by the flat walk
        "Done.",
        CANCELLED_TURN_MARKER,
        "next question",
    ]


async def test_precedence_hard_max_steps_closes_errored_over_an_unseen_message():
    # Bounded cost wins: at the hard limit the turn closes ERRORED even with
    # an unseen message — the burial is the D6 case.
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message([faux_text("Done.")], finish_reason="stop"),
        ]
    )
    mw = PostDuringResponse({1: ["too late"]})
    runner = DeterministicRunner(
        idle_session("s_hard_max", hard_max_steps=1),
        provider=faux,
        ids=["u1", "ts", "um", "a1", "tf"],
        now=1000,
        middleware=[mw],
    )
    mw.post = runner.post_message
    runner.post_message("Say done")

    result = await runner.run()

    assert result.outcome == TurnOutcome.ERRORED
    assert len(faux.requests) == 1
    assert runner.session.entries["tf"] == TurnFinish(
        id="tf",
        parent_id="a1",
        created_at=1000,
        outcome=TurnOutcome.ERRORED,
        error="Hard max steps limit reached: 1",
    )
    assert runner.idle()
