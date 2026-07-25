"""`pretty_print`: a durable session rendered as a text transcript.

Every expectation is the WHOLE transcript — the frame, the turn blocks and the
totals are one artifact, and a change anywhere in it should read as a diff.

Timestamps render with the local clock, so the expectations interpolate the
same formatting of the literal `created_at` the precondition carries rather
than hard-coding a zone-dependent string.
"""

from datetime import datetime

from luca.agent.core.models import (
    AgentSession,
    ApprovalDecision,
    ApprovalOption,
    ApprovalStatus,
    AssistantMessage,
    CancelRequested,
    CompactionEntry,
    CompactionSource,
    Conversation,
    ConversationStatus,
    ExecutionResult,
    ExecutionStatus,
    LLMConfig,
    PrunedEntry,
    SessionConfig,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolExecution,
    ToolExecutionError,
    ToolSpec,
    TurnFinish,
    TurnOutcome,
    TurnStart,
    Usage,
    UserMessage,
)
from luca.agent.core.utils import pretty_print

MODEL = LLMConfig(model="test-model", provider="faux")
T = 1_700_000_000_000
STAMP = datetime.fromtimestamp(T / 1000).strftime("%Y-%m-%d %H:%M:%S")
RULE = "─" * 64

# One answered turn: reasoning, text, a completed bracket and the provider
# usage recorded for the assistant entry in this conversation.
ANSWERED_SESSION = AgentSession(
    id="s_answered",
    entries={
        "u1": UserMessage(
            id="u1", created_at=T, parts=[TextContent(text="Hello there")],
        ),
        "ts": TurnStart(id="ts", parent_id="u1", created_at=T),
        "a1": AssistantMessage(
            id="a1", parent_id="ts", created_at=T,
            parts=[
                ThinkingContent(thinking="A greeting.", signature="sig"),
                TextContent(text="Hi! How can I help?"),
            ],
            llm_config=MODEL, stop_reason="stop",
        ),
        "tf": TurnFinish(id="tf", parent_id="a1", created_at=T),
    },
    usages={
        "c1": {
            "a1": Usage(
                conversation_id="c1", entry_id="a1",
                input=10, output=5, total_tokens=15,
            ),
        },
    },
    active_conversation=Conversation(
        id="c1", nodes=["u1", "ts", "a1", "tf"], created_at=T, updated_at=T,
    ),
    session_config=SessionConfig(llm_config=MODEL),
)

# A turn with two tool calls in one assistant message: one denied at the gate,
# one allowed by a rule and completed. The second call's signature outgrows the
# transcript width and breaks into the one-argument-per-line form.
TOOLS_SESSION = AgentSession(
    id="s_tools",
    entries={
        "u1": UserMessage(
            id="u1", created_at=T, parts=[TextContent(text="what's on disk?")],
        ),
        "ts": TurnStart(id="ts", parent_id="u1", created_at=T),
        "a1": AssistantMessage(
            id="a1", parent_id="ts", created_at=T,
            parts=[
                ThinkingContent(thinking="Look around.", redacted=True),
                ToolCall(id="tc1", name="read", arguments={"file_path": "/"}),
                ToolCall(
                    id="tc2", name="read",
                    arguments={
                        "file_path": "/Users/santiagobasulto/code/python-py",
                        "offset": 1,
                        "limit": 200,
                    },
                ),
            ],
            llm_config=MODEL, stop_reason="tool_use",
        ),
        "te1": ToolExecution(
            id="te1", parent_id="a1", created_at=T,
            tool_call_id="tc1",
            raw_tool_call=ToolCall(id="tc1", name="read", arguments={"file_path": "/"}),
            tool_spec=ToolSpec(name="read"),
            status=ExecutionStatus.REJECTED,
            approval_status=ApprovalStatus.REJECTED,
            approval_decisions=[
                ApprovalDecision(
                    decision=ApprovalOption.PENDING, metadata={"via": "mode"},
                    created_at=T,
                ),
                ApprovalDecision(
                    decision=ApprovalOption.DENY, metadata={"via": "user"},
                    created_at=T,
                ),
            ],
            ended_at=T,
        ),
        "te2": ToolExecution(
            id="te2", parent_id="te1", created_at=T,
            tool_call_id="tc2",
            raw_tool_call=ToolCall(
                id="tc2", name="read",
                arguments={
                    "file_path": "/Users/santiagobasulto/code/python-py",
                    "offset": 1,
                    "limit": 200,
                },
            ),
            tool_spec=ToolSpec(name="read"),
            status=ExecutionStatus.COMPLETED,
            result=ExecutionResult(
                content=[TextContent(text="main.py\nluca/\ntests/")],
            ),
            approval_status=ApprovalStatus.ALLOWED,
            approval_decisions=[
                ApprovalDecision(
                    decision=ApprovalOption.ALLOW, metadata={"via": "rule"},
                    created_at=T,
                ),
            ],
            started_at=T, ended_at=T + 2,
        ),
        "a2": AssistantMessage(
            id="a2", parent_id="te2", created_at=T,
            parts=[TextContent(text="Three entries.")],
            llm_config=MODEL, stop_reason="stop",
        ),
        "tf": TurnFinish(id="tf", parent_id="a2", created_at=T),
    },
    tool_executions={"tc1": ["te1"], "tc2": ["te2"]},
    usages={
        "c1": {
            "a1": Usage(
                conversation_id="c1", entry_id="a1", total_tokens=1200,
            ),
            "a2": Usage(
                conversation_id="c1", entry_id="a2", total_tokens=340,
            ),
        },
    },
    active_conversation=Conversation(
        id="c1", nodes=["u1", "ts", "a1", "te1", "te2", "a2", "tf"],
        created_at=T, updated_at=T,
    ),
    session_config=SessionConfig(
        llm_config=MODEL.model_copy(update={"reasoning": "medium"}),
    ),
)

# A tool that raised: the structured ToolExecutionError is the outcome payload,
# and the call never reached the gate (approval_status stays None).
FAILED_SESSION = AgentSession(
    id="s_failed",
    entries={
        "u1": UserMessage(id="u1", created_at=T, parts=[TextContent(text="add")]),
        "ts": TurnStart(id="ts", parent_id="u1", created_at=T),
        "a1": AssistantMessage(
            id="a1", parent_id="ts", created_at=T,
            parts=[ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2})],
            llm_config=MODEL, stop_reason="tool_use",
        ),
        "te1": ToolExecution(
            id="te1", parent_id="a1", created_at=T,
            tool_call_id="tc1",
            raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
            status=ExecutionStatus.FAILED,
            error=ToolExecutionError(
                error_type="ValueError", error_message="kaboom",
            ),
            started_at=T, ended_at=T + 7,
        ),
        "tf": TurnFinish(
            id="tf", parent_id="te1", created_at=T,
            outcome=TurnOutcome.ERRORED, error="the model gave up",
        ),
        # Retry-ready PENDING accepts a post_message: it sits behind the closed
        # bracket with no TurnStart of its own until the next run.
        "u2": UserMessage(id="u2", created_at=T, parts=[TextContent(text="again")]),
    },
    tool_executions={"tc1": ["te1"]},
    active_conversation=Conversation(
        id="c1", nodes=["u1", "ts", "a1", "te1", "tf", "u2"],
        created_at=T, updated_at=T, status=ConversationStatus.PENDING,
    ),
    session_config=SessionConfig(llm_config=MODEL),
)

# A session stopped mid-flight: the turn is still open, its only call parked at
# the approval gate, and a cancel request is waiting to be consumed.
OPEN_SESSION = AgentSession(
    id="s_open",
    entries={
        "u1": UserMessage(id="u1", created_at=T, parts=[TextContent(text="add")]),
        "ts": TurnStart(id="ts", parent_id="u1", created_at=T),
        "a1": AssistantMessage(
            id="a1", parent_id="ts", created_at=T,
            parts=[ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2})],
            llm_config=MODEL, stop_reason="tool_use",
        ),
        "te1": ToolExecution(
            id="te1", parent_id="a1", created_at=T,
            tool_call_id="tc1",
            raw_tool_call=ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
            status=ExecutionStatus.PENDING,
            approval_status=ApprovalStatus.PENDING,
            approval_decisions=[
                ApprovalDecision(
                    decision=ApprovalOption.PENDING, metadata={"via": "mode"},
                    created_at=T,
                ),
            ],
        ),
        "cr": CancelRequested(id="cr", parent_id="te1", created_at=T),
    },
    tool_executions={"tc1": ["te1"]},
    active_conversation=Conversation(
        id="c1", nodes=["u1", "ts", "a1", "te1", "cr"],
        created_at=T, updated_at=T, status=ConversationStatus.CANCELLING,
    ),
    session_config=SessionConfig(llm_config=MODEL),
)

# Context bookkeeping in the path: a compaction that replaced two entries and a
# pruned tool output standing in for the original execution, which stays in the
# store untouched.
COMPACTED_SESSION = AgentSession(
    id="s_compacted",
    entries={
        "cp": CompactionEntry(
            id="cp", created_at=T,
            source=CompactionSource.POLICY,
            parts=[
                TextContent(
                    text="The user asked about seating; the agent answered.",
                ),
            ],
            compacted_nodes=["u0", "a0"],
            llm_config=MODEL,
            started_at=T, ended_at=T,
        ),
        "te1": ToolExecution(
            id="te1", created_at=T,
            tool_call_id="tc1",
            raw_tool_call=ToolCall(id="tc1", name="read", arguments={}),
            status=ExecutionStatus.COMPLETED,
            result=ExecutionResult(content=[TextContent(text="a very long listing")]),
            started_at=T, ended_at=T + 1,
        ),
        "pr": PrunedEntry(
            id="pr", parent_id="te1", created_at=T,
            pruned_entry_type="tool_execution", pruned_entry_id="te1",
            content=[TextContent(text="[tool output has been pruned]")],
        ),
        "u1": UserMessage(id="u1", created_at=T, parts=[TextContent(text="thanks")]),
    },
    tool_executions={"tc1": ["te1"]},
    active_conversation=Conversation(
        id="c1", nodes=["cp", "pr", "u1", "missing"], created_at=T, updated_at=T,
    ),
    session_config=SessionConfig(llm_config=MODEL),
)

# A tool result too big for a transcript: 15 lines, one of them wider than the
# transcript. The durable session keeps all of it.
LISTING = (
    "alpha\nbravo\n" + "c" * 70 + "\ndelta\necho\nfoxtrot\ngolf\nhotel\nindia\n"
    "juliett\nkilo\nlima\nmike\nnovember\noscar"
)

BIG_OUTPUT_SESSION = AgentSession(
    id="s_big",
    entries={
        "a1": AssistantMessage(
            id="a1", created_at=T,
            parts=[ToolCall(id="tc1", name="ls", arguments={})],
            llm_config=MODEL, stop_reason="tool_use",
        ),
        "te1": ToolExecution(
            id="te1", parent_id="a1", created_at=T,
            tool_call_id="tc1",
            raw_tool_call=ToolCall(id="tc1", name="ls", arguments={}),
            status=ExecutionStatus.COMPLETED,
            result=ExecutionResult(content=[TextContent(text=LISTING)]),
            approval_status=ApprovalStatus.ALLOWED,
            started_at=T, ended_at=T + 1,
        ),
    },
    tool_executions={"tc1": ["te1"]},
    active_conversation=Conversation(
        id="c1", nodes=["a1", "te1"], created_at=T, updated_at=T,
    ),
    session_config=SessionConfig(llm_config=MODEL),
)

EMPTY_SESSION = AgentSession(
    id="s_empty",
    active_conversation=Conversation(id="c1", created_at=T, updated_at=T),
    session_config=SessionConfig(llm_config=MODEL),
)


def test_an_answered_turn_renders_frame_messages_and_totals():
    assert pretty_print(ANSWERED_SESSION) == f"""\
LUCA SESSION s_answered
Conversation c1 · idle · 1 turn
Default: faux/test-model
{RULE}

TURN 1 · {STAMP}
User
  Hello there

Assistant · step 1 · faux/test-model
  [thinking hidden]

  Hi! How can I help?

✓ completed · stop · 15 tokens

{RULE}
TOTAL · 1 turn · 1 model call · 0 tool calls · 15 tokens"""


def test_tool_calls_render_as_a_tree_of_approvals_and_outcomes():
    assert pretty_print(TOOLS_SESSION) == f"""\
LUCA SESSION s_tools
Conversation c1 · idle · 1 turn
Default: faux/test-model · reasoning medium
{RULE}

TURN 1 · {STAMP}
User
  what's on disk?

Assistant · step 1 · faux/test-model
  [thinking redacted]

  Tools
  ├─ read(file_path="/")
  │  └─ DENIED · approval denied by user
  │
  └─ read(
       file_path="/Users/santiagobasulto/code/python-py",
       offset=1,
       limit=200
     )
     ├─ ALLOWED · permission rule
     └─ OK · 2 ms
        main.py
        luca/
        tests/

Assistant · step 2 · faux/test-model
  Three entries.

✓ completed · stop · 1,540 tokens

{RULE}
TOTAL · 1 turn · 2 model calls · 2 tool calls · 1,540 tokens"""


def test_a_failure_shows_the_structured_error_and_the_queued_retry_message():
    assert pretty_print(FAILED_SESSION) == f"""\
LUCA SESSION s_failed
Conversation c1 · pending · 1 turn
Default: faux/test-model
{RULE}

TURN 1 · {STAMP}
User
  add

Assistant · step 1 · faux/test-model
  Tools
  └─ add(a=1, b=2)
     └─ FAILED · 7 ms
        ValueError: kaboom

✗ errored · tool_use · 0 tokens · the model gave up

{RULE}

NEXT TURN · {STAMP}
User
  again

{RULE}
TOTAL · 1 turn · 1 model call · 1 tool call · 0 tokens"""


def test_an_open_turn_shows_the_pending_gate_and_no_outcome():
    assert pretty_print(OPEN_SESSION) == f"""\
LUCA SESSION s_open
Conversation c1 · cancelling · 1 turn
Default: faux/test-model
{RULE}

TURN 1 · {STAMP}
User
  add

Assistant · step 1 · faux/test-model
  Tools
  └─ add(a=1, b=2)
     ├─ AWAITING APPROVAL · permission mode
     └─ PENDING

Cancel requested · cancelled

⋯ open · tool_use · 0 tokens

{RULE}
TOTAL · 1 turn · 1 model call · 1 tool call · 0 tokens"""


def test_compaction_pruning_and_a_missing_node_render_in_path_order():
    assert pretty_print(COMPACTED_SESSION) == f"""\
LUCA SESSION s_compacted
Conversation c1 · idle · 0 turns
Default: faux/test-model
{RULE}

NEXT TURN · {STAMP}
Compaction · replaced 2 entries
  The user asked about seating; the agent answered.

Pruned tool_execution te1
  [tool output has been pruned]

User
  thanks

[missing entry missing]

{RULE}
TOTAL · 0 turns · 0 model calls · 0 tool calls · 0 tokens"""


def test_a_compaction_that_produced_nothing_renders_its_header_alone():
    session = COMPACTED_SESSION.model_copy(deep=True, update={"id": "s_scheduled"})
    session.entries["cp"] = session.entries["cp"].model_copy(
        update={"parts": None, "compacted_nodes": None, "ended_at": None},
    )

    assert pretty_print(session) == f"""\
LUCA SESSION s_scheduled
Conversation c1 · idle · 0 turns
Default: faux/test-model
{RULE}

NEXT TURN · {STAMP}
Compaction · replaced 0 entries

Pruned tool_execution te1
  [tool output has been pruned]

User
  thanks

[missing entry missing]

{RULE}
TOTAL · 0 turns · 0 model calls · 0 tool calls · 0 tokens"""


def test_a_big_tool_result_is_clipped_by_line_and_by_width():
    assert pretty_print(BIG_OUTPUT_SESSION) == f"""\
LUCA SESSION s_big
Conversation c1 · idle · 0 turns
Default: faux/test-model
{RULE}

NEXT TURN · {STAMP}
Assistant · step 1 · faux/test-model
  Tools
  └─ ls()
     ├─ ALLOWED
     └─ OK · 1 ms
        alpha
        bravo
        {"c" * 59}…
        delta
        echo
        foxtrot
        golf
        hotel
        india
        juliett
        kilo
        lima
        … (+19 more characters)

{RULE}
TOTAL · 0 turns · 1 model call · 1 tool call · 0 tokens"""


def test_an_empty_session_prints_only_the_frame():
    assert pretty_print(EMPTY_SESSION) == f"""\
LUCA SESSION s_empty
Conversation c1 · idle · 0 turns
Default: faux/test-model
{RULE}
TOTAL · 0 turns · 0 model calls · 0 tool calls · 0 tokens"""
