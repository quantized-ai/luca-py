"""The Streamable HTTP header layer: standard headers, the base64 sentinel, and
the `x-mcp-header` mirror a client is required to support and to police."""

from __future__ import annotations

import pytest

from luca.agent.contrib.mcp.headers import (
    needs_encoding,
    param_headers,
    request_headers,
    sentinel_decode,
    sentinel_encode,
    validate_header_params,
)

# A schema exercising every mirrorable type plus a nested-but-reachable one.
MIRRORED_SCHEMA = {
    "type": "object",
    "properties": {
        "region": {"type": "string", "x-mcp-header": "Region"},
        "limit": {"type": "integer", "x-mcp-header": "Limit"},
        "dry_run": {"type": "boolean", "x-mcp-header": "DryRun"},
        "query": {"type": "string"},
        "nested": {
            "type": "object",
            "properties": {"tenant": {"type": "string", "x-mcp-header": "Tenant"}},
        },
    },
}


def test_a_plain_ascii_value_is_not_encoded():
    assert sentinel_encode("us-west1") == "us-west1"


def test_a_non_ascii_value_is_encoded():
    assert sentinel_encode("Hello, 世界") == "=?base64?SGVsbG8sIOS4lueVjA==?="


def test_a_padded_value_is_encoded():
    assert sentinel_encode(" padded ") == "=?base64?IHBhZGRlZCA=?="


def test_a_newline_is_encoded():
    assert sentinel_encode("line1\nline2") == "=?base64?bGluZTEKbGluZTI=?="


def test_a_value_shaped_like_the_sentinel_is_encoded():
    # Otherwise a server would decode something the client never encoded.
    assert sentinel_encode("=?base64?literal?=") == "=?base64?PT9iYXNlNjQ/bGl0ZXJhbD89?="


@pytest.mark.parametrize("value", ["us-west1", "Hello, 世界", " padded ", "line1\nline2", "=?base64?literal?=", ""])
def test_encoding_round_trips(value):
    assert sentinel_decode(sentinel_encode(value)) == value


def test_an_unwrapped_value_decodes_to_itself():
    assert sentinel_decode("us-west1") == "us-west1"


def test_a_tab_is_carried_plainly():
    assert needs_encoding("a\tb") is False


def test_a_tools_call_carries_the_name_header():
    assert request_headers("tools/call", {"name": "get_weather"}, protocol_version="2026-07-28") == {
        "MCP-Protocol-Version": "2026-07-28",
        "Mcp-Method": "tools/call",
        "Mcp-Name": "get_weather",
    }


def test_a_resources_read_takes_its_name_from_the_uri():
    assert request_headers("resources/read", {"uri": "file:///a.json"}, protocol_version="2026-07-28") == {
        "MCP-Protocol-Version": "2026-07-28",
        "Mcp-Method": "resources/read",
        "Mcp-Name": "file:///a.json",
    }


def test_a_listing_carries_no_name_header():
    assert request_headers("tools/list", None, protocol_version="2026-07-28") == {
        "MCP-Protocol-Version": "2026-07-28",
        "Mcp-Method": "tools/list",
    }


def test_a_non_ascii_tool_name_is_sentinel_encoded_in_the_header():
    assert request_headers("tools/call", {"name": "поиск"}, protocol_version="2026-07-28") == {
        "MCP-Protocol-Version": "2026-07-28",
        "Mcp-Method": "tools/call",
        "Mcp-Name": "=?base64?0L/QvtC40YHQug==?=",
    }


def test_annotated_arguments_are_mirrored_with_their_types_converted():
    arguments = {"region": "us-west1", "limit": 42, "dry_run": False, "query": "SELECT 1", "nested": {"tenant": "acme"}}

    assert param_headers(MIRRORED_SCHEMA, arguments) == {
        "Mcp-Param-Region": "us-west1",
        "Mcp-Param-Limit": "42",
        "Mcp-Param-DryRun": "false",
        "Mcp-Param-Tenant": "acme",
    }


def test_an_absent_argument_omits_its_header():
    assert param_headers(MIRRORED_SCHEMA, {"query": "SELECT 1"}) == {}


def test_a_null_argument_omits_its_header():
    assert param_headers(MIRRORED_SCHEMA, {"region": None}) == {}


def test_a_valid_schema_has_no_violations():
    assert validate_header_params(MIRRORED_SCHEMA) == []


def test_a_schema_with_no_annotations_has_no_violations():
    assert validate_header_params({"type": "object", "properties": {"q": {"type": "string"}}}) == []


def test_an_annotation_on_a_number_is_a_violation():
    schema = {"type": "object", "properties": {"ratio": {"type": "number", "x-mcp-header": "Ratio"}}}

    assert validate_header_params(schema) == [
        "'x-mcp-header' at ratio is on a 'number' parameter; only string, integer and boolean may be mirrored"
    ]


def test_an_annotation_under_a_composition_keyword_is_a_violation():
    schema = {
        "type": "object",
        "properties": {"either": {"oneOf": [{"type": "string", "x-mcp-header": "Either"}]}},
    }

    assert validate_header_params(schema) == [
        "'x-mcp-header' at either is not reachable through 'properties' alone, so its value cannot be located"
    ]


def test_an_annotation_inside_array_items_is_a_violation():
    schema = {
        "type": "object",
        "properties": {"tags": {"type": "array", "items": {"type": "string", "x-mcp-header": "Tag"}}},
    }

    assert validate_header_params(schema) == [
        "'x-mcp-header' at tags is not reachable through 'properties' alone, so its value cannot be located"
    ]


def test_a_duplicate_header_name_is_a_violation_regardless_of_case():
    schema = {
        "type": "object",
        "properties": {
            "a": {"type": "string", "x-mcp-header": "Region"},
            "b": {"type": "string", "x-mcp-header": "region"},
        },
    }

    assert validate_header_params(schema) == ["'x-mcp-header' 'region' is used by both a and b"]


def test_an_empty_header_name_is_a_violation():
    schema = {"type": "object", "properties": {"a": {"type": "string", "x-mcp-header": ""}}}

    assert validate_header_params(schema) == ["'x-mcp-header' at a must be a non-empty string"]


def test_a_header_name_with_a_newline_is_a_violation():
    schema = {"type": "object", "properties": {"a": {"type": "string", "x-mcp-header": "Bad\r\nInjected"}}}

    assert validate_header_params(schema) == [
        "'x-mcp-header' at a is 'Bad\\r\\nInjected', which is not a valid HTTP field name"
    ]


def test_an_annotation_on_the_schema_root_is_a_violation():
    assert validate_header_params({"type": "object", "x-mcp-header": "Root", "properties": {}}) == [
        "'x-mcp-header' is set on the schema root, which is not a parameter"
    ]


def test_an_integer_outside_the_safe_range_cannot_be_mirrored():
    schema = {"type": "object", "properties": {"n": {"type": "integer", "x-mcp-header": "N"}}}

    with pytest.raises(ValueError, match="outside the range an MCP header may carry"):
        param_headers(schema, {"n": 2**53})
