"""How long an MCP server's tool listing may take, and what happens when it
doesn't answer in time.

Background. `get_tools()` waits for its servers to say what they offer, so a
server that never answers would hold the turn open indefinitely. The runner
races the whole `build_tool_list` step against the run's cancellation token, so
a user can always cancel out — but nothing bounds an unattended run, and luca
is a library. Hence a per-server deadline, owned by the registry (the contract
says a registry doing I/O outside the rules owns its own timeout).

Resolution, most specific wins:

    server.timeout_in_ms  >  LUCA_DEFAULT_MCP_TIMEOUT_MS  >  the constant

with one exception: an OAuth server's browser flow runs inside the listing and
waits on a human, so it takes the long ceiling ahead of the env var. Its own
`timeout_in_ms` still overrides even that.

On expiry the server contributes no tools, is recorded in `failures`, and is
retried on the next ask — exactly what a refused connection already does. It
never aborts the run: one unreachable MCP server must not take down the agent.

Shape. Precondition -> one action -> postcondition. The resolver is a pure
function, so its tests pass an env dict and involve no clock at all. The
behavioural tests use a stand-in that never replies together with a 10ms
deadline, so they can only ever time out — deterministic, and 10ms each.
"""

import asyncio

import pytest
from mcp import types

from luca.agent.contrib.mcp import session as mcp_session
from luca.agent.contrib.mcp.config import HttpServer, StdioServer
from luca.agent.contrib.mcp.registry import (
    DEFAULT_LISTING_TIMEOUT_MS,
    OAUTH_LISTING_TIMEOUT_MS,
    TIMEOUT_ENV_VAR,
    McpToolRegistry,
    resolve_listing_timeout_ms,
)
from luca.agent.contrib.simple_tool_registry import (
    PermissionPolicy,
    ProxyToolRegistry,
    YoloPermissionPolicy,
)
from luca.agent.core import (
    AgentSession,
    ApprovalDecision,
    Conversation,
    LLMConfig,
    SessionConfig,
    TextContent,
    ToolExecution,
    ToolKind,
    ToolSpec,
    UserMessage,
)
from luca.client.testing import FauxProvider, faux_assistant_message, faux_text
from luca.client.types import Tool as LucaTool
from tests.agent.scenarios import DeterministicRunner

MODEL = LLMConfig(model="test-model", provider="faux")

# The stand-ins never run these, so the commands do not have to exist. 10ms is
# short enough that a server which never replies times out immediately.
PLAIN_SERVER = StdioServer(command="never-actually-spawned")
SLOW_SERVER = StdioServer(command="never-actually-spawned", timeout_in_ms=10)
OK_SERVER = StdioServer(command="never-actually-spawned-either", timeout_in_ms=10)

EMPTY_SCHEMA = {"type": "object", "properties": {}}

PING_TOOL = types.Tool(
    name="ping",
    description="Ping the server.",
    inputSchema=EMPTY_SCHEMA,
)

SLOW_PING_SPEC = ToolSpec(
    name="slow__ping",
    description="Ping the server.",
    input_schema=EMPTY_SCHEMA,
    tool_kind=ToolKind.OTHER,
)

OK_PING_WIRE_TOOL = LucaTool(
    name="ok__ping",
    description="Ping the server.",
    parameters=EMPTY_SCHEMA,
)

SRV_PING_WIRE_TOOL = LucaTool(
    name="srv__ping",
    description="Ping the server.",
    parameters=EMPTY_SCHEMA,
)

SESSION = AgentSession(
    id="s_mcp",
    active_conversation=Conversation(id="c1", nodes=[], created_at=500, updated_at=500),
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


class ServerNeverAnswers:
    """A server that accepts the connection and then never replies."""

    async def __call__(self, server, *, auth=None) -> list[types.Tool]:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")  # pragma: no cover


class ServerHangsThenAnswers:
    """Never replies on the first listing, answers immediately on every one
    after it — a server that was wedged when the session started."""

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, server, *, auth=None) -> list[types.Tool]:
        self.calls += 1
        if self.calls == 1:
            await asyncio.Event().wait()
        return [PING_TOOL]


class OneServerNeverAnswers:
    """One nominated server never replies; every other one answers."""

    def __init__(self, silent: StdioServer) -> None:
        self._silent = silent

    async def __call__(self, server, *, auth=None) -> list[types.Tool]:
        if server is self._silent:
            await asyncio.Event().wait()
        return [PING_TOOL]


class ScriptedPolicy(PermissionPolicy):
    """Answers with the next scripted decision. Nothing here reaches the gate;
    it exists because the registry requires a policy."""

    def __init__(self, decisions: list[ApprovalDecision]) -> None:
        self.decisions = list(decisions)

    async def decide(
        self,
        session: AgentSession,
        tool_execution: ToolExecution,
    ) -> ApprovalDecision:
        return self.decisions.pop(0)


# ── the resolver, as a pure function ───────────────────────────────────────────


def test_a_server_that_configures_nothing_gets_the_package_default():
    assert resolve_listing_timeout_ms(PLAIN_SERVER, env={}) == DEFAULT_LISTING_TIMEOUT_MS


def test_the_env_var_overrides_the_package_default():
    assert resolve_listing_timeout_ms(PLAIN_SERVER, env={TIMEOUT_ENV_VAR: "5000"}) == 5000


def test_a_per_server_timeout_overrides_the_env_var():
    server = StdioServer(command="x", timeout_in_ms=1234)

    assert resolve_listing_timeout_ms(server, env={TIMEOUT_ENV_VAR: "5000"}) == 1234


def test_an_oauth_server_takes_the_long_ceiling_over_the_env_var():
    server = HttpServer(url="https://example.test/mcp", oauth=True)

    assert resolve_listing_timeout_ms(server, env={TIMEOUT_ENV_VAR: "5000"}) == OAUTH_LISTING_TIMEOUT_MS


def test_an_oauth_server_still_honours_its_own_timeout():
    server = HttpServer(url="https://example.test/mcp", oauth=True, timeout_in_ms=1234)

    assert resolve_listing_timeout_ms(server, env={}) == 1234


def test_the_oauth_ceiling_outlives_the_browser_wait_it_contains():
    from luca.agent.contrib.mcp.oauth import _AUTH_TIMEOUT

    assert OAUTH_LISTING_TIMEOUT_MS > _AUTH_TIMEOUT * 1000


def test_a_malformed_env_override_is_rejected_when_the_registry_is_built(monkeypatch):
    monkeypatch.setenv(TIMEOUT_ENV_VAR, "soon")

    with pytest.raises(ValueError, match=TIMEOUT_ENV_VAR):
        McpToolRegistry({"srv": PLAIN_SERVER}, YoloPermissionPolicy())


# ── the deadline firing, on the registry alone ─────────────────────────────────


async def test_a_server_that_never_answers_contributes_no_tools_and_is_recorded(monkeypatch):
    monkeypatch.setattr(mcp_session, "list_tools", ServerNeverAnswers())
    registry = McpToolRegistry({"slow": SLOW_SERVER}, YoloPermissionPolicy())

    tools = await registry.get_tools(SESSION)

    assert tools == []
    assert registry.failures == {"slow": "listing timed out after 10ms"}
    assert registry.connected_labels == []


async def test_a_server_that_timed_out_is_retried_the_next_time_the_registry_is_asked(monkeypatch):
    monkeypatch.setattr(mcp_session, "list_tools", ServerHangsThenAnswers())
    registry = McpToolRegistry({"slow": SLOW_SERVER}, YoloPermissionPolicy())
    await registry.wait_listed()  # precondition: the first listing timed out
    assert registry.failures == {"slow": "listing timed out after 10ms"}

    tools = await registry.get_tools(SESSION)  # the server answers now

    assert tools == [SLOW_PING_SPEC]
    assert registry.failures == {}
    assert registry.connected_labels == ["slow"]


async def test_a_healthy_server_still_lists_while_a_sibling_times_out(monkeypatch):
    monkeypatch.setattr(mcp_session, "list_tools", OneServerNeverAnswers(SLOW_SERVER))
    registry = McpToolRegistry(
        {"slow": SLOW_SERVER, "ok": OK_SERVER},
        YoloPermissionPolicy(),
    )

    tools = await registry.get_tools(SESSION)

    assert [spec.name for spec in tools] == ["ok__ping"]
    assert registry.failures == {"slow": "listing timed out after 10ms"}
    assert registry.connected_labels == ["ok"]


# ── through the runner, whole session ──────────────────────────────────────────


async def test_a_hung_server_does_not_stall_the_turn_and_is_picked_up_on_the_next_one(monkeypatch):
    monkeypatch.setattr(mcp_session, "list_tools", ServerHangsThenAnswers())
    faux = FauxProvider()
    faux.set_responses(
        [
            faux_assistant_message([faux_text("Hi.")], finish_reason="stop"),
            faux_assistant_message([faux_text("Hi again.")], finish_reason="stop"),
        ]
    )
    registry = McpToolRegistry({"srv": SLOW_SERVER}, ScriptedPolicy([]))
    runner = DeterministicRunner(
        FRESH_SESSION.model_copy(deep=True),
        tool_registry=ProxyToolRegistry(registry),
        provider=faux,
        ids=["ts", "a1", "tf1", "u2", "ts2", "a2", "tf2"],
        now=2000,
    )
    async with runner.run() as run:  # turn one: the server never answers
        _ = [event async for event in run]
    # the turn finished normally rather than hanging or aborting the run
    assert runner.idle()
    assert faux.requests[0].tools is None
    assert registry.failures == {"srv": "listing timed out after 10ms"}

    runner.post_message("Hello again")
    async with runner.run() as run:  # turn two: the server answers
        _ = [event async for event in run]

    assert faux.requests[1].tools == [SRV_PING_WIRE_TOOL]
    assert registry.failures == {}


async def test_one_hung_server_does_not_cost_the_others_their_tools_on_the_wire(monkeypatch):
    monkeypatch.setattr(mcp_session, "list_tools", OneServerNeverAnswers(SLOW_SERVER))
    faux = FauxProvider()
    faux.set_responses([faux_assistant_message([faux_text("Hi.")], finish_reason="stop")])
    registry = McpToolRegistry(
        {"slow": SLOW_SERVER, "ok": OK_SERVER},
        ScriptedPolicy([]),
    )
    runner = DeterministicRunner(
        FRESH_SESSION.model_copy(deep=True),
        tool_registry=ProxyToolRegistry(registry),
        provider=faux,
        ids=["ts", "a1", "tf"],
        now=2000,
    )

    async with runner.run() as run:
        _ = [event async for event in run]

    # the blast radius is the one server, not the agent
    assert faux.requests[0].tools == [OK_PING_WIRE_TOOL]
    assert registry.failures == {"slow": "listing timed out after 10ms"}
    assert runner.idle()
