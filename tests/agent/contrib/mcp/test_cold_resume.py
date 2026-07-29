"""Two regression tests for MCP, both through the runner. Both fail today.

Background. `McpToolRegistry.get_tools()` has to ask each configured MCP server
what tools it offers, and asking is network I/O. It does not wait for the
answer: it starts a background task and returns whatever is cached, which on a
cold registry is an empty list. That background task also runs at most once for
the registry's whole lifetime, so a first attempt that fails is never retried.

Test 1 — a parked approval, reopened.
An application runs several registries behind a `ProxyToolRegistry`, which
learns which child owns which tool name by calling `get_tools()` on each child.
The drive loop calls `decide()` and `prepare()` at the top of the loop and
`get_tools()` further down, just before the model request. So a conversation
saved while parked at an approval prompt, reopened in a fresh process, reaches
the permission gate before anything has listed the MCP servers. The proxy is
told there are no MCP tools, finds no owner for the name, answers ALLOW on its
own without asking the policy, and then fails to route the same name at
dispatch. The call the user was asked to approve is recorded NOT_FOUND and
never runs. That is the composing-registry requirement in
`specs/0002-tool-execution-fixes/prd.md` section 6.4.

Test 2 — a server that was down at launch.
The listing is guarded by a one-shot flag that is set whether the listing
succeeded or not and is never cleared, so a server that was unreachable when
the session started is never asked again. If the machine was offline at launch,
MCP stays dead until the process is restarted.

Shape. Precondition -> one action -> postcondition, with full-object asserts.
The session literals are declared, not produced by driving an earlier turn. "A
fresh process" is modelled by handing the runner a brand new, cold
`McpToolRegistry`. The MCP transport is replaced by scripted stand-ins, so no
subprocess is spawned and nothing depends on timing. The permission policy is a
scripted double that records what it was asked about: a call missing from
`seen` is a call that never reached the gate.
"""

from mcp import types

from luca.agent.contrib.mcp import session as mcp_session
from luca.agent.contrib.mcp.config import StdioServer
from luca.agent.contrib.mcp.registry import McpToolRegistry
from luca.agent.contrib.simple_tool_registry import PermissionPolicy, ProxyToolRegistry
from luca.agent.core import (
    AgentSession,
    ApprovalDecision,
    ApprovalOption,
    ApprovalStatus,
    AssistantMessage,
    Conversation,
    ConversationStatus,
    ExecutionResult,
    ExecutionStatus,
    LLMConfig,
    SessionConfig,
    TextContent,
    ToolCall,
    ToolExecution,
    ToolKind,
    ToolSpec,
    TurnStart,
    UserMessage,
)
from luca.agent.core.events import (
    FinishReason,
    TextBlock,
    ToolExecuted,
    ToolExecutionStarted,
)
from luca.client.testing import FauxProvider, faux_assistant_message, faux_text
from luca.client.types import Tool as LucaTool
from tests.agent.scenarios import DeterministicRunner

MODEL = LLMConfig(model="test-model", provider="faux")

# The stand-ins never run this, so the command does not have to exist.
SERVER = StdioServer(command="never-actually-spawned")

EMPTY_SCHEMA = {"type": "object", "properties": {}}

# What the server reports over the wire...
PING_TOOL = types.Tool(
    name="ping",
    description="Ping the server.",
    inputSchema=EMPTY_SCHEMA,
)

# ...and the `ToolSpec` the registry turns it into: namespaced with the
# server's label, ToolKind.OTHER because the framework cannot know what an
# external tool does.
PING_SPEC = ToolSpec(
    name="srv__ping",
    description="Ping the server.",
    input_schema=EMPTY_SCHEMA,
    tool_kind=ToolKind.OTHER,
)
PING_SPEC_ID = PING_SPEC.spec_id()

# ...and what the model sees on the wire.
PING_WIRE_TOOL = LucaTool(
    name="srv__ping",
    description="Ping the server.",
    parameters=EMPTY_SCHEMA,
)

PING_CALL = ToolCall(id="tc1", name="srv__ping", arguments={})

PENDING_1000 = ApprovalDecision(decision=ApprovalOption.PENDING, created_at=1000)
ALLOW_2000 = ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=2000)

# The call as it sits in a session saved at the approval prompt: born, handed
# to the gate, and deferred.
PARKED_EXECUTION = ToolExecution(
    id="te1",
    parent_id="a1",
    created_at=1000,
    tool_call_id="tc1",
    raw_tool_call=PING_CALL,
    tool_spec=PING_SPEC,
    tool_spec_id=PING_SPEC_ID,
    status=ExecutionStatus.PENDING,
    approval_status=ApprovalStatus.PENDING,
    approval_decisions=[PENDING_1000],
    updated_at=1000,
)

# The whole session around it, as it would be reloaded from disk: the user
# asked, the model requested `srv__ping`, and the turn is holding at the gate.
PARKED_SESSION = AgentSession(
    id="s_mcp_parked",
    entries={
        "u1": UserMessage(id="u1", created_at=1000, parts=[TextContent(text="Ping it")]),
        "ts": TurnStart(id="ts", parent_id="u1", created_at=1000),
        "a1": AssistantMessage(
            id="a1",
            parent_id="ts",
            created_at=1000,
            parts=[PING_CALL],
            llm_config=MODEL,
            stop_reason="tool_use",
        ),
        "te1": PARKED_EXECUTION,
    },
    tool_executions={"tc1": ["te1"]},
    tool_specs={PING_SPEC_ID: PING_SPEC},
    active_conversation=Conversation(
        id="c1",
        nodes=["u1", "ts", "a1", "te1"],
        created_at=1000,
        updated_at=1000,
        status=ConversationStatus.AWAITING_APPROVAL,
    ),
    session_config=SessionConfig(llm_config=MODEL),
)

# One user message and nothing else — a brand new conversation.
FRESH_SESSION = AgentSession(
    id="s_mcp_fresh",
    entries={
        "u1": UserMessage(id="u1", created_at=1000, parts=[TextContent(text="Hello")]),
    },
    active_conversation=Conversation(id="c1", nodes=["u1"], created_at=1000, updated_at=1000),
    session_config=SessionConfig(llm_config=MODEL),
)


# ── stand-ins ──────────────────────────────────────────────────────────────────


class ServerUp:
    """A healthy server: every listing succeeds and reports one tool."""

    async def __call__(self, server, *, auth=None) -> list[types.Tool]:
        return [PING_TOOL]


class ServerDownThenUp:
    """A server that is unreachable on the first listing and healthy on every
    listing after it — the machine-was-offline-at-launch case."""

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, server, *, auth=None) -> list[types.Tool]:
        self.calls += 1
        if self.calls == 1:
            raise ConnectionError("server is down")
        return [PING_TOOL]


class PingAnswers:
    """The tool body over the wire: `ping` answers "pong"."""

    async def __call__(self, server, name, arguments, *, auth=None) -> types.CallToolResult:
        return types.CallToolResult(
            content=[types.TextContent(type="text", text="pong")],
            isError=False,
        )


class ScriptedPolicy(PermissionPolicy):
    """Answers with the next scripted decision, and records the name of every
    call it was asked about. A call missing from `seen` never reached the
    permission gate at all."""

    def __init__(self, decisions: list[ApprovalDecision]) -> None:
        self.decisions = list(decisions)
        self.seen: list[str] = []

    async def decide(
        self,
        session: AgentSession,
        tool_execution: ToolExecution,
    ) -> ApprovalDecision:
        self.seen.append(tool_execution.raw_tool_call.name)
        return self.decisions.pop(0)


# ── 1. a parked approval, reopened in a fresh process ──────────────────────────


async def test_a_parked_mcp_call_is_approved_and_runs_when_the_conversation_is_reopened(monkeypatch):
    monkeypatch.setattr(mcp_session, "list_tools", ServerUp())
    monkeypatch.setattr(mcp_session, "call_tool", PingAnswers())
    faux = FauxProvider()
    # one response, and only one: a completed tool sends its result back to the
    # model, so the turn needs a closing answer to reach idle
    faux.set_responses([faux_assistant_message([faux_text("Pong.")], finish_reason="stop")])
    policy = ScriptedPolicy([ALLOW_2000])
    runner = DeterministicRunner(
        PARKED_SESSION.model_copy(deep=True),
        tool_registry=ProxyToolRegistry(McpToolRegistry({"srv": SERVER}, policy)),
        provider=faux,
        ids=["a2", "tf"],
        now=2000,
    )

    async with runner.run() as run:
        events = [event async for event in run]

    running = PARKED_EXECUTION.model_copy(
        update={
            "status": ExecutionStatus.RUNNING,
            "approval_status": ApprovalStatus.ALLOWED,
            "approval_decisions": [PENDING_1000, ALLOW_2000],
            "started_at": 2000,
            "updated_at": 2000,
        }
    )
    completed = running.model_copy(
        update={
            "status": ExecutionStatus.COMPLETED,
            "result": ExecutionResult(content=[TextContent(text="pong")], is_error=False),
            "ended_at": 2000,
            # stamped by the ContextManager on the terminal transition
            "context_tokens": 1,
        }
    )
    # the permission gate was consulted about the call the user was asked to approve
    assert policy.seen == ["srv__ping"]
    assert events == [
        ToolExecutionStarted(tool_call_id="tc1", execution=running),
        ToolExecuted(
            tool_call_id="tc1",
            execution=completed,
            result_text="pong",
            is_error=False,
        ),
        TextBlock(text="Pong."),
        FinishReason(finish_reason="stop"),
    ]
    assert runner.session.entries["te1"] == completed
    assert runner.idle()


# ── 2. a server that was unreachable when the session started ──────────────────


async def test_an_mcp_server_that_was_down_at_launch_is_picked_up_on_the_next_turn(monkeypatch):
    monkeypatch.setattr(mcp_session, "list_tools", ServerDownThenUp())
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message([faux_text("Hi.")], finish_reason="stop"),
            faux_assistant_message([faux_text("Hi again.")], finish_reason="stop"),
        ]
    )
    runner = DeterministicRunner(
        FRESH_SESSION.model_copy(deep=True),
        tool_registry=ProxyToolRegistry(McpToolRegistry({"srv": SERVER}, ScriptedPolicy([]))),
        provider=faux,
        ids=["ts", "a1", "tf1", "u2", "ts2", "a2", "tf2"],
        now=2000,
    )
    async with runner.run() as run:  # turn one: the server is unreachable
        _ = [event async for event in run]
    assert faux.requests[0].tools is None

    runner.post_message("Hello again")
    async with runner.run() as run:  # turn two: the server is healthy
        _ = [event async for event in run]

    assert faux.requests[1].tools == [PING_WIRE_TOOL]
