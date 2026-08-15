"""The three hashes on a resolved server.

They exist so the durable tool catalog can decide, at boot and with no network,
whether a cached listing still applies. Each answers a different question, and
the tests below pin which changes move which hash.
"""

from __future__ import annotations

from luca.agent.contrib.mcp.errors import is_modern_protocol_error
from luca.agent.contrib.mcp.servers import HttpServer, StdioServer

STDIO = StdioServer(label="fs", command="npx", args=("-y", "server-filesystem"), env=(("TOKEN", "abc"),))
HTTP = HttpServer(label="gh", url="https://example.test/mcp", headers=(("Authorization", "Bearer abc"),))


def test_renaming_a_label_keeps_the_identity():
    # A rename must not orphan the cached listing or the stored OAuth token.
    assert STDIO.model_copy(update={"label": "files"}).identity() == STDIO.identity()


def test_changing_the_command_changes_the_identity():
    assert STDIO.model_copy(update={"command": "node"}).identity() != STDIO.identity()


def test_changing_a_secret_leaves_the_identity_alone():
    assert STDIO.model_copy(update={"env": (("TOKEN", "different"),)}).identity() == STDIO.identity()


def test_changing_a_secret_changes_the_credential_fingerprint():
    # Which is what drops a `cacheScope: private` listing belonging to someone
    # else's credentials.
    assert STDIO.model_copy(update={"env": (("TOKEN", "different"),)}).credential_fingerprint() != (
        STDIO.credential_fingerprint()
    )


def test_changing_a_secret_leaves_the_definition_hash_alone():
    assert STDIO.model_copy(update={"env": (("TOKEN", "different"),)}).definition_hash() == STDIO.definition_hash()


def test_adding_an_environment_variable_changes_the_definition_hash():
    assert STDIO.model_copy(update={"env": (("TOKEN", "abc"), ("DEBUG", "1"))}).definition_hash() != (
        STDIO.definition_hash()
    )


def test_changing_args_changes_the_definition_hash():
    assert STDIO.model_copy(update={"args": ("-y", "other")}).definition_hash() != STDIO.definition_hash()


def test_a_header_value_is_a_credential_not_a_definition():
    changed = HTTP.model_copy(update={"headers": (("Authorization", "Bearer other"),)})

    assert (changed.definition_hash(), changed.identity()) == (HTTP.definition_hash(), HTTP.identity())
    assert changed.credential_fingerprint() != HTTP.credential_fingerprint()


def test_a_header_name_is_a_definition_and_is_case_insensitive():
    assert HTTP.model_copy(update={"headers": (("authorization", "Bearer abc"),)}).definition_hash() == (
        HTTP.definition_hash()
    )


def test_turning_on_oauth_changes_the_definition_hash():
    assert HTTP.model_copy(update={"oauth": True}).definition_hash() != HTTP.definition_hash()


def test_a_stdio_and_an_http_server_never_share_an_identity():
    assert STDIO.identity() != HTTP.identity()


def test_hashes_are_stable_across_calls():
    assert (STDIO.identity(), STDIO.definition_hash(), STDIO.credential_fingerprint()) == (
        STDIO.identity(),
        STDIO.definition_hash(),
        STDIO.credential_fingerprint(),
    )


def test_the_spec_reserved_codes_prove_a_modern_peer():
    assert [is_modern_protocol_error(code) for code in (-32020, -32021, -32022)] == [True, True, True]


def test_method_not_found_does_not_prove_a_modern_peer():
    # It is the most common answer a legacy server gives to `server/discover`,
    # and reading it as modern would strand every pre-2026 server.
    assert is_modern_protocol_error(-32601) is False
