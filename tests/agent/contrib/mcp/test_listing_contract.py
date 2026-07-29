"""Unit regression tests for how `McpToolRegistry` lists its servers' tools.

These are the narrow, registry-level counterparts to the two runner-level tests
in `test_cold_resume.py`. Same two bugs, isolated to the one object that owns
them, so a failure points straight at the method that is wrong.

Background. `get_tools()` has to ask every configured MCP server what it
offers, and asking is network I/O. Today the registry does not wait for the
answer: it starts a background task and returns whatever is already cached,
which on a cold registry is an empty list. It also starts that background task
at most once for the registry's whole lifetime, so a first attempt that fails —
the machine was offline, the server was restarting — is never retried and the
tool list stays empty until the process is restarted.

What these pin down:

- `get_tools()` answers with the server's tools, not with whatever happens to
  be cached at that instant. Every reader downstream (the `ProxyToolRegistry`
  routing table, the permission gate, the model's tool list) believes that
  answer, so it has to be true when it is given.
- `create_execution()` resolves against the same listing, so a call is not born
  NOT_FOUND just because nothing has listed yet.
- A listing that failed is retried the next time the registry is asked, and a
  server that recovers stops being reported as failed.
- A listing that was cancelled leaves the registry able to try again, rather
  than latched onto a dead task forever.

Every test in this file fails against the current implementation.

Shape. Precondition -> one action -> postcondition, with full-object asserts.
The MCP transport is replaced by a scripted stand-in, so no subprocess is
spawned and nothing depends on how long a real server takes to boot.
"""

import asyncio
import contextlib

from mcp import types

from luca.agent.contrib.mcp import session as mcp_session
from luca.agent.contrib.mcp.config import StdioServer
from luca.agent.contrib.mcp.registry import McpToolRegistry
from luca.agent.contrib.simple_tool_registry import YoloPermissionPolicy
from luca.agent.core import (
    AgentSession,
    Conversation,
    ExecutionStatus,
    LLMConfig,
    SessionConfig,
    ToolCall,
    ToolExecution,
    ToolKind,
    ToolSpec,
)

# The stand-in never runs this, so the command does not have to exist.
SERVER = StdioServer(command="never-actually-spawned")

EMPTY_SCHEMA = {"type": "object", "properties": {}}

# What the server reports over the wire...
PING_TOOL = types.Tool(
    name="ping",
    description="Ping the server.",
    inputSchema=EMPTY_SCHEMA,
)

# ...and the `ToolSpec` the registry must turn it into: namespaced with the
# server's label, ToolKind.OTHER because the framework cannot know what an
# external tool does.
PING_SPEC = ToolSpec(
    name="srv__ping",
    description="Ping the server.",
    input_schema=EMPTY_SCHEMA,
    tool_kind=ToolKind.OTHER,
)

PING_CALL = ToolCall(id="tc1", name="srv__ping", arguments={})

SESSION = AgentSession(
    id="s_mcp",
    active_conversation=Conversation(id="c1", nodes=[], created_at=500, updated_at=500),
    session_config=SessionConfig(llm_config=LLMConfig(model="test-model", provider="faux")),
)


# ── transport stand-ins ────────────────────────────────────────────────────────


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


class ServerUpAfterOneTick:
    """A healthy server whose listing suspends once before answering, so a
    test can cancel it while it is genuinely in flight."""

    async def __call__(self, server, *, auth=None) -> list[types.Tool]:
        await asyncio.sleep(0)
        return [PING_TOOL]


# ── get_tools answers with the server's tools ──────────────────────────────────


async def test_get_tools_answers_with_the_servers_tools_on_a_cold_registry(monkeypatch):
    monkeypatch.setattr(mcp_session, "list_tools", ServerUp())
    registry = McpToolRegistry({"srv": SERVER}, YoloPermissionPolicy())

    tools = await registry.get_tools(SESSION)

    assert tools == [PING_SPEC]


async def test_create_execution_births_pending_on_a_cold_registry(monkeypatch):
    monkeypatch.setattr(mcp_session, "list_tools", ServerUp())
    registry = McpToolRegistry({"srv": SERVER}, YoloPermissionPolicy())

    draft = await registry.create_execution(SESSION, PING_CALL)

    assert draft == ToolExecution(
        tool_call_id="tc1",
        raw_tool_call=PING_CALL,
        tool_spec=PING_SPEC,
        status=ExecutionStatus.PENDING,
    )


# ── a failed listing is retried ────────────────────────────────────────────────


async def test_a_listing_that_failed_is_retried_the_next_time_the_registry_is_asked(monkeypatch):
    monkeypatch.setattr(mcp_session, "list_tools", ServerDownThenUp())
    registry = McpToolRegistry({"srv": SERVER}, YoloPermissionPolicy())
    await registry.wait_listed()  # precondition: the first listing failed
    assert registry.failures == {"srv": "server is down"}
    assert registry.connected_labels == []

    tools = await registry.get_tools(SESSION)  # the server is up now

    assert tools == [PING_SPEC]


async def test_a_server_that_recovers_stops_being_reported_as_failed(monkeypatch):
    monkeypatch.setattr(mcp_session, "list_tools", ServerDownThenUp())
    registry = McpToolRegistry({"srv": SERVER}, YoloPermissionPolicy())
    await registry.wait_listed()  # precondition: the first listing failed
    assert registry.failures == {"srv": "server is down"}

    await registry.get_tools(SESSION)  # the server is up now

    assert registry.failures == {}
    assert registry.connected_labels == ["srv"]


# ── a cancelled listing does not latch ─────────────────────────────────────────


async def test_a_cancelled_listing_leaves_the_registry_able_to_list_again(monkeypatch):
    monkeypatch.setattr(mcp_session, "list_tools", ServerUpAfterOneTick())
    registry = McpToolRegistry({"srv": SERVER}, YoloPermissionPolicy())
    listing = asyncio.ensure_future(registry.wait_listed())
    await asyncio.sleep(0)  # the listing is now in flight
    listing.cancel()  # what the runner does when it kills a raced call
    with contextlib.suppress(asyncio.CancelledError):
        await listing  # and it waits for the teardown to unwind

    tools = await registry.get_tools(SESSION)

    assert tools == [PING_SPEC]
