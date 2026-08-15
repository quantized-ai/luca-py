"""MCP wire shapes to luca core shapes: tool names, specs, results, errors."""

from __future__ import annotations

import mcp_types

from luca.agent.contrib.mcp.errors import McpError, McpServerGone
from luca.agent.contrib.mcp.mapping import (
    approval_context,
    spec_identity,
    to_execution_result,
    to_tool_execution_error,
    to_tool_spec,
    wire_name,
)
from luca.agent.contrib.resource_permissions.mixin import AnswerOption, PermissionRequest, ResourcePermission
from luca.agent.core import (
    ExecutionResult,
    FileContent,
    ImageContent,
    MediaBase64,
    TextContent,
    ToolExecutionError,
    ToolKind,
    ToolSpec,
)

PNG = "iVBORw0KGgo="


def tool(**over) -> mcp_types.Tool:
    """A complete `mcp_types.Tool`, since every field but `name` and
    `inputSchema` is optional and no test is about the difference."""
    return mcp_types.Tool.model_validate({"name": "search", "inputSchema": {"type": "object"}, **over})


def result(**over) -> mcp_types.CallToolResult:
    return mcp_types.CallToolResult.model_validate({"content": [], **over})


def test_a_tool_name_is_prefixed_by_server():
    assert wire_name("github", "create_issue") == "mcp__github__create_issue"


def test_characters_a_provider_rejects_are_replaced():
    assert wire_name("my.server", "get/thing") == "mcp__my_server__get_thing"


def test_a_name_over_the_cap_is_truncated_with_a_digest():
    long = wire_name("github", "a" * 100)

    assert len(long) == 64
    assert long.startswith("mcp__github__aaa")


def test_two_long_names_sharing_a_prefix_stay_distinct():
    assert wire_name("github", "a" * 100) != wire_name("github", "a" * 101)


def test_a_tool_becomes_a_spec_carrying_its_identity():
    assert to_tool_spec("github", tool(description="Find things", outputSchema={"type": "object"})) == ToolSpec(
        name="mcp__github__search",
        title=None,
        description="Find things",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        is_private=False,
        metadata={"mcp": {"server": "github", "tool": "search", "annotations": {}}},
        tool_kind=ToolKind.OTHER,
        namespace="mcp.github",
        version=None,
        timeout_in_ms=None,
    )


def test_a_tool_without_a_description_gets_an_empty_one():
    # `ToolSpec.description` is required and the wire type rejects null, so a
    # server that omits it must not produce an invalid spec.
    assert to_tool_spec("x", tool()).description == ""


def test_annotations_are_carried_for_display_but_not_as_a_tool_kind():
    # A server claiming `readOnlyHint` must not be able to talk itself past a
    # `tool_kind: read` allow-rule.
    spec = to_tool_spec("x", tool(annotations={"title": "Search", "readOnlyHint": True}))

    assert (spec.tool_kind, spec.title, spec.metadata["mcp"]["annotations"]) == (
        ToolKind.OTHER,
        "Search",
        {"title": "Search", "readOnlyHint": True},
    )


def test_a_per_server_call_timeout_lands_on_the_spec():
    assert to_tool_spec("x", tool(), timeout_in_ms=5000).timeout_in_ms == 5000


def test_a_spec_resolves_back_to_its_server_and_remote_name():
    # This is what lets `prepare()` dispatch on a cold resume with no catalog,
    # and why the wire name is never parsed back.
    assert spec_identity(to_tool_spec("github", tool(name="a__b"))) == ("github", "a__b")


def test_a_foreign_spec_has_no_mcp_identity():
    assert spec_identity(ToolSpec(name="read_file", description="", input_schema={})) is None


def test_text_content_maps_straight_through():
    assert to_execution_result(result(content=[{"type": "text", "text": "hello"}])) == ExecutionResult(
        content=[TextContent(text="hello")],
        structured_content=None,
        metadata={},
        is_error=False,
        extras={},
    )


def test_image_content_becomes_a_base64_image_part():
    assert to_execution_result(
        result(content=[{"type": "image", "data": PNG, "mimeType": "image/png"}])
    ) == ExecutionResult(
        content=[ImageContent(source=MediaBase64(data=PNG, media_type="image/png"))],
        structured_content=None,
        metadata={},
        is_error=False,
        extras={},
    )


def test_audio_is_described_and_its_payload_kept_out_of_the_projector():
    # Core has no audio part and the projector raises on anything it does not
    # know, so the block is described rather than carried.
    mapped = to_execution_result(result(content=[{"type": "audio", "data": "AAA=", "mimeType": "audio/wav"}]))

    assert mapped.content == [TextContent(text="[audio content, audio/wav, not rendered]")]
    assert mapped.metadata["mcp_unmapped_content"] == [{"type": "audio", "data": "AAA=", "mimeType": "audio/wav"}]


def test_a_resource_link_is_described_rather_than_fetched():
    mapped = to_execution_result(
        result(content=[{"type": "resource_link", "name": "cfg", "uri": "file:///a.json", "description": "the config"}])
    )

    assert mapped.content == [TextContent(text="[resource file:///a.json] the config")]


def test_an_embedded_text_resource_becomes_text():
    mapped = to_execution_result(
        result(content=[{"type": "resource", "resource": {"uri": "file:///a.txt", "text": "body"}}])
    )

    assert mapped.content == [TextContent(text="body")]


def test_an_embedded_pdf_becomes_a_real_file_part():
    mapped = to_execution_result(
        result(
            content=[
                {
                    "type": "resource",
                    "resource": {"uri": "file:///doc/report.pdf", "blob": "JVBERi0=", "mimeType": "application/pdf"},
                }
            ]
        )
    )

    assert mapped.content == [
        FileContent(source=MediaBase64(data="JVBERi0=", media_type="application/pdf"), name="report.pdf")
    ]


def test_an_embedded_image_blob_becomes_an_image_part():
    mapped = to_execution_result(
        result(content=[{"type": "resource", "resource": {"uri": "x://a", "blob": PNG, "mimeType": "image/png"}}])
    )

    assert mapped.content == [ImageContent(source=MediaBase64(data=PNG, media_type="image/png"))]


def test_an_embedded_blob_of_an_unknown_type_is_described():
    mapped = to_execution_result(
        result(
            content=[
                {"type": "resource", "resource": {"uri": "x://a.bin", "blob": "AA==", "mimeType": "application/x"}}
            ]
        )
    )

    assert mapped.content == [TextContent(text="[resource x://a.bin, application/x, not rendered]")]
    assert "mcp_unmapped_content" in mapped.metadata


def test_structured_content_is_carried_separately():
    mapped = to_execution_result(result(content=[{"type": "text", "text": "ok"}], structuredContent={"count": 2}))

    assert (mapped.content, mapped.structured_content) == ([TextContent(text="ok")], {"count": 2})


def test_non_object_structured_content_is_wrapped_rather_than_dropped():
    # 2026-07-28 loosened `structuredContent` to any JSON value, but the core
    # field is a dict.
    assert to_execution_result(result(structuredContent=[1, 2])).structured_content == {"result": [1, 2]}


def test_a_result_with_no_content_renders_its_structured_output():
    mapped = to_execution_result(result(structuredContent={"count": 2}))

    assert mapped.content == [TextContent(text='{"count":2}')]


def test_a_wholly_empty_result_still_carries_one_block():
    assert to_execution_result(result()).content == [TextContent(text="")]


def test_a_tool_reported_error_completes_with_the_flag_set():
    # `is_error` is the tool's verdict on its own work, not a failure of the
    # call, so the execution still completes.
    mapped = to_execution_result(result(content=[{"type": "text", "text": "no such file"}], isError=True))

    assert (mapped.is_error, mapped.content) == (True, [TextContent(text="no such file")])


def test_result_meta_is_kept_under_its_own_key():
    assert to_execution_result(result(_meta={"traceparent": "00-x"})).metadata == {"mcp_meta": {"traceparent": "00-x"}}


def test_a_jsonrpc_error_keeps_its_code_and_data():
    assert to_tool_execution_error(McpError(-32602, "bad arg", {"field": "q"})) == ToolExecutionError(
        error_type="McpError",
        error_message="bad arg",
        details={"code": -32602, "data": {"field": "q"}},
    )


def test_a_dead_connection_maps_to_its_own_error_type():
    assert to_tool_execution_error(McpServerGone("the server restarted mid-call")) == ToolExecutionError(
        error_type="McpServerGone",
        error_message="the server restarted mid-call",
        details={},
    )


def test_an_unexpected_exception_keeps_its_class_name():
    assert to_tool_execution_error(RuntimeError("boom")) == ToolExecutionError(
        error_type="RuntimeError",
        error_message="boom",
        details={},
    )


def test_the_approval_context_offers_this_tool_and_the_whole_server():
    assert approval_context("github", "create_issue") == {
        "requests": [
            {
                "resources": [{"permission": "mcp", "resource": "github/create_issue"}],
                "answer_options": [
                    {
                        "resource_permissions": [{"permission": "mcp", "resource": "github/create_issue"}],
                        "metadata": {"label": "Allow create_issue from github"},
                    },
                    {
                        "resource_permissions": [{"permission": "mcp", "resource": "github/*"}],
                        "metadata": {"label": "Allow every tool from github"},
                    },
                ],
                "metadata": {"server": "github", "tool": "create_issue"},
            }
        ]
    }


def test_the_approval_context_is_a_real_permission_request():
    # Built as plain dicts so `mapping` stays importable without the
    # resource_permissions package, but the strategy reads them as the output
    # of `PermissionRequest.model_dump()`, so they have to validate as one.
    request = approval_context("github", "create_issue")["requests"][0]

    assert PermissionRequest.model_validate(request) == PermissionRequest(
        resources=[ResourcePermission(permission="mcp", resource="github/create_issue")],
        answer_options=[
            AnswerOption(
                resource_permissions=[ResourcePermission(permission="mcp", resource="github/create_issue")],
                metadata={"label": "Allow create_issue from github"},
            ),
            AnswerOption(
                resource_permissions=[ResourcePermission(permission="mcp", resource="github/*")],
                metadata={"label": "Allow every tool from github"},
            ),
        ],
        metadata={"server": "github", "tool": "create_issue"},
    )
