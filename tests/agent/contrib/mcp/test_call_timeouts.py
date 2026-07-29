"""How long one of an MCP server's tool CALLS may take.

Background. The listing deadline (`test_listing_timeouts.py`) bounds discovery.
It says nothing about the call itself, and until now nothing did: `_to_tool_spec`
left `ToolSpec.timeout_in_ms` unset, so a dispatch fell through to
`RuntimeConfig.tool_execution_timeout_in_ms`, which defaults to infinity. A
server that accepted a call and never replied ran forever.

The fix needs no new machinery — the runner already reads
`tool_spec.timeout_in_ms` at dispatch and enforces it. It only needed the field
populated, from static server config:

    server.call_timeout_in_ms  >  LUCA_DEFAULT_MCP_CALL_TIMEOUT_MS  >  None

`None`, not a number, is the deliberate default: a tool body's default deadline
is a framework decision (PRD 6.6), and any figure picked for arbitrary
third-party tools would silently kill someone's legitimately slow one. What this
buys is the ability to bound ONE server's calls without bounding every tool in
the agent.

It is kept separate from the listing timeout because a listing is a quick
metadata fetch and a call may legitimately run for minutes — one number would be
too tight for one or too generous for the other.

Shape. Precondition -> one action -> postcondition. The resolver is pure, so its
tests pass an env dict and involve no clock. The dispatch test uses a stand-in
that never replies with a 10ms bound, so it can only ever time out.
"""

import asyncio

import pytest
from mcp import types

from luca.agent.contrib.mcp import session as mcp_session
from luca.agent.contrib.mcp.config import HttpServer, StdioServer
from luca.agent.contrib.mcp.registry import (
    CALL_TIMEOUT_ENV_VAR,
    McpToolRegistry,
    resolve_call_timeout_ms,
)
from luca.agent.contrib.simple_tool_registry import (
    PermissionPolicy,
    ProxyToolRegistry,
    YoloPermissionPolicy,
)
from luca.agent.core import (
    AgentSession,
    ApprovalDecision,
    ApprovalOption,
    ApprovalStatus,
    Conversation,
    ExecutionStatus,
    LLMConfig,
    SessionConfig,
    TextContent,
    ToolCall,
    ToolExecution,
    ToolKind,
    ToolSpec,
    UserMessage,
)
from luca.agent.core.events import ToolExecuted
from luca.client.testing import (
    FauxProvider,
    faux_assistant_message,
    faux_text,
    faux_tool_call,
)
from tests.agent.scenarios import DeterministicRunner

MODEL = LLMConfig(model="test-model", provider="faux")

# The stand-ins never run these, so the commands do not have to exist.
PLAIN_SERVER = StdioServer(command="never-actually-spawned")
SLOW_CALL_SERVER = StdioServer(command="never-actually-spawned", call_timeout_in_ms=10)

EMPTY_SCHEMA = {"type": "object", "properties": {}}

PING_TOOL = types.Tool(
    name="ping",
    description="Ping the server.",
    inputSchema=EMPTY_SCHEMA,
)

# The spec the registry mints for `SLOW_CALL_SERVER` under the label `srv`,
# carrying the call bound the runner enforces at dispatch.
BOUNDED_PING_SPEC = ToolSpec(
    name="srv__ping",
    description="Ping the server.",
    input_schema=EMPTY_SCHEMA,
    tool_kind=ToolKind.OTHER,
    timeout_in_ms=10,
)
BOUNDED_PING_SPEC_ID = BOUNDED_PING_SPEC.spec_id()

PING_CALL = ToolCall(id="tc1", name="srv__ping", arguments={})

ALLOW_2000 = ApprovalDecision(decision=ApprovalOption.ALLOW, created_at=2000)

SESSION = AgentSession(
    id="s_mcp",
    active_conversation=Conversation(id="c1", nodes=[], created_at=500, updated_at=500),
    session_config=SessionConfig(llm_config=MODEL),
)

# One user message and nothing else — a brand new conversation.
FRESH_SESSION = AgentSession(
    id="s_mcp_fresh",
    entries={
        "u1": UserMessage(id="u1", created_at=2000, parts=[TextContent(text="Ping it")]),
    },
    active_conversation=Conversation(id="c1", nodes=["u1"], created_at=2000, updated_at=2000),
    session_config=SessionConfig(llm_config=MODEL),
)


# ── stand-ins ──────────────────────────────────────────────────────────────────


class ServerUp:
    """A healthy server: every listing succeeds and reports one tool."""

    async def __call__(self, server, *, auth=None) -> list[types.Tool]:
        return [PING_TOOL]


class CallNeverAnswers:
    """The server accepts the tool call and then never replies."""

    async def __call__(self, server, name, arguments, *, auth=None) -> types.CallToolResult:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")  # pragma: no cover


class ScriptedPolicy(PermissionPolicy):
    """Answers with the next scripted decision."""

    def __init__(self, decisions: list[ApprovalDecision]) -> None:
        self.decisions = list(decisions)

    async def decide(
        self,
        session: AgentSession,
        tool_execution: ToolExecution,
    ) -> ApprovalDecision:
        return self.decisions.pop(0)


# ── the resolver, as a pure function ───────────────────────────────────────────


def test_a_server_that_configures_nothing_inherits_the_framework_default():
    assert resolve_call_timeout_ms(PLAIN_SERVER, env={}) is None


def test_the_env_var_sets_the_default():
    assert resolve_call_timeout_ms(PLAIN_SERVER, env={CALL_TIMEOUT_ENV_VAR: "5000"}) == 5000


def test_a_per_server_call_timeout_overrides_the_env_var():
    server = StdioServer(command="x", call_timeout_in_ms=1234)

    assert resolve_call_timeout_ms(server, env={CALL_TIMEOUT_ENV_VAR: "5000"}) == 1234


def test_the_listing_timeout_does_not_bound_calls():
    server = StdioServer(command="x", timeout_in_ms=1234)

    assert resolve_call_timeout_ms(server, env={}) is None


def test_an_oauth_server_gets_no_special_call_ceiling():
    server = HttpServer(url="https://example.test/mcp", oauth=True)

    assert resolve_call_timeout_ms(server, env={}) is None


def test_a_malformed_env_override_is_rejected_when_the_registry_is_built(monkeypatch):
    monkeypatch.setenv(CALL_TIMEOUT_ENV_VAR, "soon")

    with pytest.raises(ValueError, match=CALL_TIMEOUT_ENV_VAR):
        McpToolRegistry({"srv": PLAIN_SERVER}, YoloPermissionPolicy())


# ── the bound reaches the spec ─────────────────────────────────────────────────


async def test_the_configured_bound_is_stamped_on_every_spec(monkeypatch):
    monkeypatch.setattr(mcp_session, "list_tools", ServerUp())
    registry = McpToolRegistry({"srv": SLOW_CALL_SERVER}, YoloPermissionPolicy())

    tools = await registry.get_tools(SESSION)

    assert tools == [BOUNDED_PING_SPEC]


async def test_a_server_that_configures_nothing_leaves_the_spec_unbounded(monkeypatch):
    monkeypatch.setattr(mcp_session, "list_tools", ServerUp())
    registry = McpToolRegistry({"srv": PLAIN_SERVER}, YoloPermissionPolicy())

    tools = await registry.get_tools(SESSION)

    assert tools == [
        ToolSpec(
            name="srv__ping",
            description="Ping the server.",
            input_schema=EMPTY_SCHEMA,
            tool_kind=ToolKind.OTHER,
            timeout_in_ms=None,
        )
    ]


# ── through the runner: the runner enforces it at dispatch ─────────────────────


async def test_a_call_that_exceeds_the_bound_is_recorded_timed_out(monkeypatch):
    monkeypatch.setattr(mcp_session, "list_tools", ServerUp())
    monkeypatch.setattr(mcp_session, "call_tool", CallNeverAnswers())
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("srv__ping", {}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("It timed out.")], finish_reason="stop"),
        ]
    )
    runner = DeterministicRunner(
        FRESH_SESSION.model_copy(deep=True),
        tool_registry=ProxyToolRegistry(McpToolRegistry({"srv": SLOW_CALL_SERVER}, ScriptedPolicy([ALLOW_2000]))),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "tf"],
        now=2000,
    )

    async with runner.run() as run:
        events = [event async for event in run]

    # the deadline hard-cancelled the body: TIMED_OUT, resultless, errorless,
    # `cancel_signalled_at` untouched (a deadline is not a run cancellation)
    timed_out = ToolExecution(
        id="te1",
        parent_id="a1",
        created_at=2000,
        tool_call_id="tc1",
        raw_tool_call=PING_CALL,
        tool_spec=BOUNDED_PING_SPEC,
        tool_spec_id=BOUNDED_PING_SPEC_ID,
        approval_status=ApprovalStatus.ALLOWED,
        approval_decisions=[ALLOW_2000],
        status=ExecutionStatus.TIMED_OUT,
        result=None,
        error=None,
        started_at=2000,
        ended_at=2000,
        cancel_signalled_at=None,
        updated_at=2000,
    )
    assert events[3] == ToolExecuted(
        tool_call_id="tc1",
        execution=timed_out,
        result_text="[tool execution timed_out]",
        is_error=True,
    )
    assert runner.session.entries["te1"] == timed_out
    # and the turn ran to completion rather than hanging
    assert runner.idle()


# ── the spec must stay a pure function of the tool definition ──────────────────


async def test_calling_one_tool_twice_files_a_single_spec(monkeypatch):
    monkeypatch.setattr(mcp_session, "list_tools", ServerUp())
    monkeypatch.setattr(mcp_session, "call_tool", CallNeverAnswers())
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("srv__ping", {}, id="tc1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message(
                [faux_tool_call("srv__ping", {}, id="tc2")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("Done.")], finish_reason="stop"),
        ]
    )
    runner = DeterministicRunner(
        FRESH_SESSION.model_copy(deep=True),
        tool_registry=ProxyToolRegistry(
            McpToolRegistry({"srv": SLOW_CALL_SERVER}, ScriptedPolicy([ALLOW_2000, ALLOW_2000]))
        ),
        provider=faux,
        ids=["ts", "a1", "te1", "a2", "te2", "a3", "tf"],
        now=2000,
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    # a timeout read from static config is not volatile, so normalization holds
    assert runner.session.tool_specs == {BOUNDED_PING_SPEC_ID: BOUNDED_PING_SPEC}
