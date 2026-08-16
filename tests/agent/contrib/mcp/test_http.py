"""Streamable HTTP: required headers, the two response shapes, and both eras.

Driven through `httpx.MockTransport`, which is the established route in this
repo for exercising a wire protocol without a socket. Header assertions are
partial and say so: httpx adds `host`, `accept-encoding` and friends that this
code does not own.
"""

from __future__ import annotations

import json

import httpx
import pytest

from luca.agent.contrib.mcp.connection import ServerConnection
from luca.agent.contrib.mcp.errors import McpAuthRequired, McpServerGone
from luca.agent.contrib.mcp.servers import HttpServer
from luca.agent.contrib.mcp.wire import Era

URL = "https://example.test/mcp"

DISCOVER = {
    "supportedVersions": ["2026-07-28"],
    "capabilities": {"tools": {}},
    "resultType": "complete",
    "ttlMs": 60000,
    "cacheScope": "public",
}
TOOLS = {
    "tools": [{"name": "search", "inputSchema": {"type": "object"}}],
    "resultType": "complete",
    "ttlMs": 60000,
    "cacheScope": "public",
}
CALL = {"content": [{"type": "text", "text": "found"}], "resultType": "complete"}


class Recorder:
    """Answers by method and remembers every request, so a test can assert on
    what was sent as well as what came back."""

    def __init__(self, results: dict, *, status: int = 200, session_id: str | None = None) -> None:
        self.results = results
        self.status = status
        self.session_id = session_id
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = json.loads(request.content)
        headers = {"Mcp-Session-Id": self.session_id} if self.session_id else {}
        if "id" not in body:
            return httpx.Response(202, headers=headers)  # a notification, per the spec
        if self.status != 200:
            return httpx.Response(self.status, json={"error": "no"}, headers=headers)
        result = self.results.get(body["method"])
        if result is None:
            return httpx.Response(
                200,
                json={"jsonrpc": "2.0", "id": body["id"], "error": {"code": -32601, "message": "nope"}},
                headers=headers,
            )
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result}, headers=headers)

    def sent(self, method: str) -> httpx.Request:
        return next(request for request in self.requests if json.loads(request.content)["method"] == method)


def connection(handler, **over) -> tuple[ServerConnection, httpx.AsyncClient]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return ServerConnection(HttpServer(label="remote", url=URL, **over), client=client), client


@pytest.fixture
async def http():
    """Hands out connections over a mock transport and closes both ends."""
    opened: list[tuple[ServerConnection, httpx.AsyncClient]] = []

    def make(handler, **over):
        pair = connection(handler, **over)
        opened.append(pair)
        return pair[0]

    yield make
    for conn, client in opened:
        await conn.aclose()
        await client.aclose()


async def test_a_modern_http_server_is_discovered(http):
    negotiated = await http(Recorder({"server/discover": DISCOVER})).connect()

    assert (negotiated.era, negotiated.protocol_version) == (Era.MODERN, "2026-07-28")


async def test_every_post_carries_the_required_headers(http):
    recorder = Recorder({"server/discover": DISCOVER, "tools/call": CALL})
    await http(recorder).call_tool("search", {"q": "x"})

    sent = recorder.sent("tools/call")

    # Partial on purpose: httpx adds host, accept-encoding and others we do
    # not own and must not pin.
    assert {key: sent.headers[key] for key in ("mcp-protocol-version", "mcp-method", "mcp-name", "accept")} == {
        "mcp-protocol-version": "2026-07-28",
        "mcp-method": "tools/call",
        "mcp-name": "search",
        "accept": "application/json, text/event-stream",
    }


async def test_the_name_header_matches_the_body(http):
    # A server MUST reject a request whose headers disagree with its body, so
    # both are derived from the same frame.
    recorder = Recorder({"server/discover": DISCOVER, "tools/call": CALL})
    await http(recorder).call_tool("search", {})

    sent = recorder.sent("tools/call")

    assert sent.headers["mcp-name"] == json.loads(sent.content)["params"]["name"]


async def test_a_modern_request_carries_its_protocol_metadata_in_the_body(http):
    recorder = Recorder({"server/discover": DISCOVER, "tools/list": TOOLS})
    await http(recorder).list_tools()

    meta = json.loads(recorder.sent("tools/list").content)["params"]["_meta"]

    assert meta["io.modelcontextprotocol/protocolVersion"] == "2026-07-28"


async def test_static_headers_from_the_config_are_sent(http):
    recorder = Recorder({"server/discover": DISCOVER, "tools/list": TOOLS})
    await http(recorder, headers=(("Authorization", "Bearer abc"),)).list_tools()

    assert recorder.sent("tools/list").headers["authorization"] == "Bearer abc"


async def test_an_sse_response_is_read_to_its_final_frame(http):
    # Progress notifications arrive before the answer and must not be mistaken
    # for it.
    stream = (
        b'event: message\r\ndata: {"jsonrpc":"2.0","method":"notifications/progress",'
        b'"params":{"progressToken":"t","progress":1}}\r\n\r\n'
        b'data: {"jsonrpc":"2.0","id":2,"result":' + json.dumps(CALL).encode() + b"}\r\n\r\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["method"] == "server/discover":
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": DISCOVER})
        return httpx.Response(200, content=stream, headers={"content-type": "text/event-stream"})

    result = await http(handler).call_tool("search", {})

    assert [block.text for block in result.content] == ["found"]


async def test_a_stream_that_ends_before_answering_is_a_dead_request(http):
    # This revision removed resumability, so there is nothing to reconnect to.
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["method"] == "server/discover":
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": DISCOVER})
        return httpx.Response(200, content=b": keep-alive\n\n", headers={"content-type": "text/event-stream"})

    with pytest.raises(McpServerGone):
        await http(handler).call_tool("search", {})


async def test_a_401_asks_for_authorization_rather_than_failing_opaquely(http):
    with pytest.raises(McpAuthRequired):
        await http(Recorder({}, status=401)).connect()


async def test_a_legacy_http_server_falls_back_and_keeps_its_session_id(http):
    # The 2025-era shape: no `server/discover`, and a session id the client has
    # to echo on every later request.
    initialize = {
        "protocolVersion": "2025-06-18",
        "capabilities": {"tools": {}},
        "serverInfo": {"name": "old", "version": "1"},
    }
    recorder = Recorder({"initialize": initialize, "tools/list": {"tools": [], "nextCursor": None}})
    recorder.session_id = "session-abc"
    conn = http(recorder)

    await conn.list_tools()

    assert conn.negotiated.era is Era.HANDSHAKE
    assert recorder.sent("tools/list").headers["mcp-session-id"] == "session-abc"


async def test_a_legacy_request_carries_no_protocol_metadata(http):
    initialize = {
        "protocolVersion": "2025-06-18",
        "capabilities": {"tools": {}},
        "serverInfo": {"name": "old", "version": "1"},
    }
    recorder = Recorder({"initialize": initialize, "tools/list": {"tools": [], "nextCursor": None}})
    await http(recorder).list_tools()

    assert "_meta" not in (json.loads(recorder.sent("tools/list").content).get("params") or {})


async def test_an_annotated_argument_is_mirrored_into_a_param_header(http):
    # Supporting `x-mcp-header` is required of clients, and the value stays in
    # the body too because the server validates that they agree.
    schema = {"type": "object", "properties": {"region": {"type": "string", "x-mcp-header": "Region"}}}
    tools = {**TOOLS, "tools": [{"name": "sql", "inputSchema": schema}]}
    recorder = Recorder({"server/discover": DISCOVER, "tools/list": tools, "tools/call": CALL})
    conn = http(recorder)
    await conn.list_tools()

    await conn.call_tool("sql", {"region": "us-west1"}, extra_headers={"Mcp-Param-Region": "us-west1"})

    assert recorder.sent("tools/call").headers["mcp-param-region"] == "us-west1"
