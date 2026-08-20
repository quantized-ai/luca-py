"""JSON-RPC framing for both protocol eras, and version-gated result parsing."""

from __future__ import annotations

import pytest

from luca import __version__
from luca.agent.contrib.mcp.errors import McpError, McpProtocolError
from luca.agent.contrib.mcp.wire import (
    Era,
    decode,
    encode,
    notification_frame,
    parse_result,
    request_frame,
    result_payload,
)

MODERN_META = {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientCapabilities": {},
    "io.modelcontextprotocol/clientInfo": {"name": "luca", "version": __version__},
}


def test_the_latest_version_is_the_modern_era():
    assert Era.of("2026-07-28") is Era.MODERN


@pytest.mark.parametrize("version", ["2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"])
def test_every_pre_2026_version_is_the_handshake_era(version):
    assert Era.of(version) is Era.HANDSHAKE


def test_an_unknown_version_is_rejected():
    with pytest.raises(McpProtocolError, match="Unknown MCP protocol version '1999-01-01'"):
        Era.of("1999-01-01")


def test_a_modern_request_carries_its_own_protocol_metadata():
    frame = request_frame(
        1, "tools/call", {"name": "x", "arguments": {}}, protocol_version="2026-07-28", era=Era.MODERN
    )

    assert frame == {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "x", "arguments": {}, "_meta": MODERN_META},
    }


def test_a_handshake_request_carries_no_metadata():
    frame = request_frame(1, "tools/call", {"name": "x"}, protocol_version="2025-06-18", era=Era.HANDSHAKE)

    assert frame == {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "x"}}


def test_a_modern_request_with_no_params_still_carries_metadata():
    frame = request_frame(2, "tools/list", None, protocol_version="2026-07-28", era=Era.MODERN)

    assert frame == {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {"_meta": MODERN_META}}


def test_a_handshake_request_with_no_params_omits_them():
    frame = request_frame(2, "tools/list", None, protocol_version="2025-06-18", era=Era.HANDSHAKE)

    assert frame == {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}


def test_an_existing_meta_block_is_extended_rather_than_replaced():
    frame = request_frame(
        3,
        "tools/call",
        {"name": "x", "_meta": {"progressToken": "abc"}},
        protocol_version="2026-07-28",
        era=Era.MODERN,
    )

    assert frame["params"]["_meta"] == {"progressToken": "abc", **MODERN_META}


def test_a_notification_has_no_id():
    frame = notification_frame(
        "notifications/cancelled", {"requestId": 4}, protocol_version="2026-07-28", era=Era.MODERN
    )

    assert frame == {
        "jsonrpc": "2.0",
        "method": "notifications/cancelled",
        "params": {"requestId": 4, "_meta": MODERN_META},
    }


def test_a_frame_encodes_to_one_newline_terminated_line():
    assert encode({"jsonrpc": "2.0", "id": 1, "method": "ping"}) == b'{"jsonrpc":"2.0","id":1,"method":"ping"}\n'


def test_a_response_yields_its_result():
    assert result_payload(decode('{"jsonrpc":"2.0","id":1,"result":{"tools":[]}}')) == {"tools": []}


def test_an_error_response_raises_with_its_code_and_data():
    message = decode('{"jsonrpc":"2.0","id":1,"error":{"code":-32602,"message":"bad arg","data":{"field":"q"}}}')

    with pytest.raises(McpError) as caught:
        result_payload(message)

    assert (caught.value.code, caught.value.message, caught.value.data) == (-32602, "bad arg", {"field": "q"})


def test_non_json_output_is_a_protocol_error():
    with pytest.raises(McpProtocolError, match="non-JSON output"):
        decode("this is not json")


def test_json_that_is_not_json_rpc_is_a_protocol_error():
    with pytest.raises(McpProtocolError, match="not JSON-RPC"):
        decode('{"hello":"world"}')


def test_a_tools_list_result_parses_with_its_cache_hints():
    parsed = parse_result(
        "tools/list",
        "2026-07-28",
        {"tools": [{"name": "a", "inputSchema": {"type": "object"}}], "ttlMs": 60000, "cacheScope": "public"},
    )

    assert (parsed.ttl_ms, parsed.cache_scope, [tool.name for tool in parsed.tools]) == (60000, "public", ["a"])


def test_a_modern_listing_omitting_its_required_hints_still_parses():
    # The 2026 schema requires `resultType`, `ttlMs` and `cacheScope`. Losing a
    # server's whole tool listing over a freshness hint this client does not
    # depend on would be the wrong trade, so absence is interpreted.
    parsed = parse_result("tools/list", "2026-07-28", {"tools": []})

    assert (parsed.result_type, parsed.ttl_ms, parsed.cache_scope) == ("complete", 0, "private")


def test_a_stated_cache_hint_is_never_overridden():
    parsed = parse_result("tools/list", "2026-07-28", {"tools": [], "ttlMs": 30000, "cacheScope": "public"})

    assert (parsed.ttl_ms, parsed.cache_scope) == (30000, "public")


def test_a_declared_input_required_result_is_believed():
    # A server asking for more input before it can finish the call. v1 does not
    # answer these, but it has to tell them apart from a completed result.
    parsed = parse_result(
        "tools/call",
        "2026-07-28",
        {"resultType": "input_required", "requestState": "resume-token-1"},
    )

    assert parsed.result_type == "input_required"


def test_a_method_the_version_does_not_define_is_rejected():
    # `server/discover` is 2026-only, so asking a handshake-era server for it
    # fails here rather than being sent and misinterpreted.
    with pytest.raises(McpProtocolError, match="invalid 'server/discover' result for protocol 2025-06-18"):
        parse_result("server/discover", "2025-06-18", {"supportedVersions": ["2025-06-18"], "capabilities": {}})


def test_a_malformed_result_is_rejected():
    with pytest.raises(McpProtocolError, match="invalid 'tools/list' result"):
        parse_result("tools/list", "2026-07-28", {"tools": [{"inputSchema": {}}]})
