"""The `mcp` block of luca.json: transport inference, label rules, and the
cross-file merge that the previous attempt got wrong."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from luca.agent.contrib.mcp.config import McpConfigError, McpServerSettings, McpSettings, expand
from luca.agent.contrib.mcp.servers import HttpServer, StdioServer


def test_a_command_makes_a_stdio_server():
    settings = McpSettings.model_validate({"servers": {"fs": {"command": "npx", "args": ["-y", "server-filesystem"]}}})

    assert settings.build(environ={}) == {
        "fs": StdioServer(
            label="fs",
            command="npx",
            args=("-y", "server-filesystem"),
            env=(),
            connect_timeout_in_ms=None,
            list_timeout_in_ms=None,
            call_timeout_in_ms=None,
        )
    }


def test_a_url_makes_an_http_server():
    settings = McpSettings.model_validate(
        {"servers": {"linear": {"url": "https://mcp.linear.app/mcp", "oauth": True, "call_timeout_in_ms": 5000}}}
    )

    assert settings.build(environ={}) == {
        "linear": HttpServer(
            label="linear",
            url="https://mcp.linear.app/mcp",
            headers=(),
            oauth=True,
            client_id=None,
            redirect_port=None,
            connect_timeout_in_ms=None,
            list_timeout_in_ms=None,
            call_timeout_in_ms=5000,
        )
    }


def test_a_merged_redefinition_is_rejected_by_name():
    # `_deep_merge` blends a home and a project luca.json field by field, so a
    # label defined as http in one and stdio in the other arrives here carrying
    # both. Under a discriminated union that lands a mongrel; here it is loud.
    merged = {"servers": {"github": {"url": "https://example.test/mcp", "command": "npx"}}}

    with pytest.raises(ValidationError) as caught:
        McpSettings.model_validate(merged)

    assert "sets both 'command' and 'url'" in str(caught.value)
    assert "must give the whole definition" in str(caught.value)


def test_a_server_with_neither_transport_is_rejected():
    with pytest.raises(ValidationError) as caught:
        McpServerSettings.model_validate({"enabled": True})

    assert "sets neither 'command' (stdio) nor 'url' (http)" in str(caught.value)


def test_http_fields_on_a_stdio_server_are_rejected():
    with pytest.raises(ValidationError) as caught:
        McpServerSettings.model_validate({"command": "npx", "oauth": True})

    assert "is a stdio server but sets 'oauth'" in str(caught.value)


def test_stdio_fields_on_an_http_server_are_rejected():
    with pytest.raises(ValidationError) as caught:
        McpServerSettings.model_validate({"url": "https://example.test/mcp", "env": {"A": "b"}})

    assert "is a http server but sets 'env'" in str(caught.value)


def test_an_empty_collection_is_not_evidence_of_a_transport():
    # pydantic fills `args`, `env` and `headers` in whether or not the user
    # wrote them, so an http server must not be accused of setting `env`.
    settings = McpServerSettings.model_validate({"url": "https://example.test/mcp", "headers": {}})

    assert settings.to_server("x", environ={}) == HttpServer(label="x", url="https://example.test/mcp")


def test_a_disabled_server_is_not_built():
    settings = McpSettings.model_validate({"servers": {"off": {"command": "x", "enabled": False}}})

    assert settings.build(environ={}) == {}


def test_a_label_outside_the_tool_name_charset_is_rejected():
    settings = McpSettings.model_validate({"servers": {"my server": {"command": "x"}}})

    with pytest.raises(McpConfigError) as caught:
        settings.build(environ={})

    assert "'my server'" in str(caught.value)


def test_an_environment_reference_is_expanded_into_headers():
    settings = McpSettings.model_validate(
        {"servers": {"gh": {"url": "https://example.test/mcp", "headers": {"Authorization": "Bearer ${GH_TOKEN}"}}}}
    )

    assert settings.build(environ={"GH_TOKEN": "s3cret"}) == {
        "gh": HttpServer(label="gh", url="https://example.test/mcp", headers=(("Authorization", "Bearer s3cret"),))
    }


def test_an_undefined_environment_reference_names_itself():
    settings = McpSettings.model_validate({"servers": {"gh": {"command": "x", "env": {"TOKEN": "${NOPE}"}}}})

    with pytest.raises(McpConfigError) as caught:
        settings.build(environ={})

    assert str(caught.value) == "mcp server 'gh' env 'TOKEN' references undefined environment variable: NOPE."


def test_every_undefined_reference_is_reported_at_once():
    with pytest.raises(McpConfigError) as caught:
        expand("${A}/${B}", {}, where="somewhere")

    assert str(caught.value) == "somewhere references undefined environment variables: A, B."


def test_a_literal_value_passes_through_expansion_untouched():
    assert expand("no references here", {}, where="somewhere") == "no references here"
