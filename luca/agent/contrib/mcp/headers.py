"""The HTTP header layer of Streamable HTTP. Pure functions, no I/O.

The transport mirrors selected body fields into headers so gateways can route
without parsing JSON-RPC. Three MUST-level consequences for a client:

1. Every POST carries `MCP-Protocol-Version` and `Mcp-Method`, plus `Mcp-Name`
   for `tools/call`, `resources/read` and `prompts/get`. A server must reject a
   request whose headers disagree with its body, so these are derived from the
   frame rather than passed alongside it.
2. A value that is not plain safe ASCII is base64-encoded in a `=?base64?...?=`
   wrapper. A plain value that looks like the sentinel is encoded too, or a
   server would decode something the client never encoded.
3. `x-mcp-header` annotations are mirrored into `Mcp-Param-{Name}`, and a tool
   whose annotations break the constraints must be EXCLUDED from `tools/list`.
   So `validate_header_params` returns reasons rather than a boolean, for
   `/mcp` to display: a tool that silently vanishes is worse than one that
   explains itself.
"""

from __future__ import annotations

import base64
from collections.abc import Iterator, Mapping
from typing import Any, Final

PROTOCOL_VERSION_HEADER: Final = "MCP-Protocol-Version"
METHOD_HEADER: Final = "Mcp-Method"
NAME_HEADER: Final = "Mcp-Name"
PARAM_HEADER_PREFIX: Final = "Mcp-Param-"

SENTINEL_PREFIX: Final = "=?base64?"
SENTINEL_SUFFIX: Final = "?="

# The methods whose `params.name` (or `params.uri`) the spec mirrors into
# `Mcp-Name`. A method outside this set sends no name header at all.
NAME_SOURCES: Final = {
    "tools/call": "name",
    "prompts/get": "name",
    "resources/read": "uri",
}

# RFC 9110 5.1 token characters. An `x-mcp-header` value has to be a legal HTTP
# field name, and this is that rule spelled out.
_TCHAR: Final = frozenset("!#$%&'*+-.^_`|~0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

_MIRRORABLE_TYPES: Final = frozenset({"string", "integer", "boolean"})

# JavaScript's safe integer range, which the spec imposes so a value survives a
# round trip through an intermediary that parses JSON with doubles.
_MAX_SAFE_INT: Final = 2**53 - 1
_MIN_SAFE_INT: Final = -(2**53) + 1

# Descending into any of these leaves the statically-reachable property chain,
# so an annotation found underneath one cannot be mirrored: the client would
# have to resolve a schema at call time to find the value.
_UNREACHABLE_KEYWORDS: Final = (
    "items",
    "prefixItems",
    "additionalProperties",
    "patternProperties",
    "oneOf",
    "anyOf",
    "allOf",
    "not",
    "if",
    "then",
    "else",
    "$defs",
    "definitions",
)


def needs_encoding(value: str) -> bool:
    """Whether a value has to go in the sentinel wrapper.

    Plain form is only for visible ASCII, space and tab, unpadded: leading or
    trailing whitespace does not survive HTTP field-value parsing.
    """
    if value != value.strip(" \t"):
        return True
    if value.startswith(SENTINEL_PREFIX) and value.endswith(SENTINEL_SUFFIX):
        return True
    return any(character not in " \t" and not ("\x21" <= character <= "\x7e") for character in value)


def sentinel_encode(value: str) -> str:
    """A value in header form, wrapped only when it has to be."""
    if not needs_encoding(value):
        return value
    encoded = base64.b64encode(value.encode("utf-8")).decode("ascii")
    return f"{SENTINEL_PREFIX}{encoded}{SENTINEL_SUFFIX}"


def sentinel_decode(value: str) -> str:
    """The inverse. A value that is not wrapped is returned untouched."""
    if not (value.startswith(SENTINEL_PREFIX) and value.endswith(SENTINEL_SUFFIX)):
        return value
    inner = value[len(SENTINEL_PREFIX) : -len(SENTINEL_SUFFIX)]
    return base64.b64decode(inner).decode("utf-8")


def header_value(value: object) -> str:
    """One argument as a header value, per the spec's type conversion: a string
    as-is, an integer in decimal, a boolean lowercased."""
    if isinstance(value, bool):  # before int — bool IS an int in Python
        return "true" if value else "false"
    if isinstance(value, int):
        if not _MIN_SAFE_INT <= value <= _MAX_SAFE_INT:
            raise ValueError(f"integer {value} is outside the range an MCP header may carry")
        return str(value)
    return sentinel_encode(str(value))


def request_headers(method: str, params: Mapping[str, Any] | None, *, protocol_version: str) -> dict[str, str]:
    """The standard headers for one POST, derived from the frame it accompanies
    so the two cannot disagree."""
    headers = {PROTOCOL_VERSION_HEADER: protocol_version, METHOD_HEADER: method}
    source = NAME_SOURCES.get(method)
    if source is not None and params is not None:
        name = params.get(source)
        if isinstance(name, str):
            headers[NAME_HEADER] = sentinel_encode(name)
    return headers


def validate_header_params(input_schema: Mapping[str, Any]) -> list[str]:
    """Every way this tool's `x-mcp-header` annotations break the rules. Empty
    means safe to expose; non-empty means the client must exclude it."""
    violations: list[str] = []
    seen: dict[str, str] = {}  # lowercased header name -> the path that claimed it
    for path, node, reachable in _walk(input_schema):
        annotation = node.get("x-mcp-header")
        if annotation is None:
            continue
        where = ".".join(path) if path else "the schema root"
        if not path:
            violations.append("'x-mcp-header' is set on the schema root, which is not a parameter")
            continue
        if not reachable:
            violations.append(
                f"'x-mcp-header' at {where} is not reachable through 'properties' alone, so its value cannot be located"
            )
            continue
        if not isinstance(annotation, str) or not annotation:
            violations.append(f"'x-mcp-header' at {where} must be a non-empty string")
            continue
        if not set(annotation) <= _TCHAR:
            violations.append(f"'x-mcp-header' at {where} is {annotation!r}, which is not a valid HTTP field name")
            continue
        declared = node.get("type")
        if declared not in _MIRRORABLE_TYPES:
            violations.append(
                f"'x-mcp-header' at {where} is on a {declared!r} parameter; only string, integer and boolean may be mirrored"
            )
            continue
        claimed = seen.get(annotation.lower())
        if claimed is not None:
            violations.append(f"'x-mcp-header' {annotation!r} is used by both {claimed} and {where}")
            continue
        seen[annotation.lower()] = where
    return violations


def param_headers(input_schema: Mapping[str, Any], arguments: Mapping[str, Any]) -> dict[str, str]:
    """The `Mcp-Param-*` mirror for one call.

    A missing or null argument omits its header rather than sending it empty.
    Assumes the schema already passed `validate_header_params`.
    """
    headers: dict[str, str] = {}
    for path, node, reachable in _walk(input_schema):
        annotation = node.get("x-mcp-header")
        if not (reachable and path and isinstance(annotation, str) and annotation):
            continue
        found, value = _at_path(arguments, path)
        if not found or value is None:
            continue
        headers[f"{PARAM_HEADER_PREFIX}{annotation}"] = header_value(value)
    return headers


def _walk(
    node: Mapping[str, Any],
    path: tuple[str, ...] = (),
    *,
    reachable: bool = True,
) -> Iterator[tuple[tuple[str, ...], Mapping[str, Any], bool]]:
    """Every subschema, with the property path reaching it and whether that path
    is made only of `properties` steps.

    Unreachable branches are still walked: an annotation under `oneOf` is a
    violation to REPORT, not one to skip.
    """
    yield path, node, reachable
    properties = node.get("properties")
    if isinstance(properties, Mapping):
        for name, child in properties.items():
            if isinstance(child, Mapping):
                yield from _walk(child, (*path, name), reachable=reachable)
    for keyword in _UNREACHABLE_KEYWORDS:
        child = node.get(keyword)
        for sub in child if isinstance(child, list) else [child]:
            if isinstance(sub, Mapping):
                yield from _walk(sub, path, reachable=False)


def _at_path(arguments: Mapping[str, Any], path: tuple[str, ...]) -> tuple[bool, Any]:
    """The argument value at a property path, and whether it was there at all."""
    current: Any = arguments
    for step in path:
        if not isinstance(current, Mapping) or step not in current:
            return False, None
        current = current[step]
    return True, current
