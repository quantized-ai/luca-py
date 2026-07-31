"""Structured output (`response_format`) accepts the same three styles as Tool.parameters.

A raw `dict` is ALWAYS a JSON Schema — never a wire-level `response_format`
payload. Providers spell the wire shape differently (OpenAI nests it under
`json_schema`, the Responses API puts it on `text.format`, Anthropic on
`output_config.format`), so one canonical schema in and one projection per
transport out. A caller who needs a hand-written wire payload passes it
through `provider_options`.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, TypeAdapter

ResponseFormat = dict | type | TypeAdapter

DEFAULT_SCHEMA_NAME = "structured_output"

# What providers accept as the schema's wire name: `^[a-zA-Z0-9_-]+$`, 64 max.
_ILLEGAL_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_-]")
_MAX_SCHEMA_NAME = 64


def response_format_to_json_schema(response_format: Any) -> tuple[str, dict]:
    """Convert any of the three response_format forms to `(name, schema)`.

    The name is what providers label the schema with on the wire. A Pydantic
    model lends its class name; the other two forms have none, so they get
    `DEFAULT_SCHEMA_NAME`.
    """
    if isinstance(response_format, dict):
        return DEFAULT_SCHEMA_NAME, response_format
    if isinstance(response_format, type) and issubclass(response_format, BaseModel):
        return sanitize_schema_name(response_format.__name__), response_format.model_json_schema()
    if isinstance(response_format, TypeAdapter):
        return DEFAULT_SCHEMA_NAME, response_format.json_schema()
    raise TypeError(
        f"response_format must be a dict, BaseModel subclass, or TypeAdapter; got {type(response_format).__name__}"
    )


def sanitize_schema_name(name: str) -> str:
    """A class name reduced to what the wire accepts. A Pydantic generic keeps
    its parameters in `__name__` (`Box[int]` literally), which the provider
    rejects — over the label, not the schema. Substituted rather than refused:
    the name is a label the caller never sees."""
    cleaned = _ILLEGAL_NAME_CHARS.sub("_", name)[:_MAX_SCHEMA_NAME]
    return cleaned or DEFAULT_SCHEMA_NAME


def strictify_json_schema(schema: dict, _defs: dict | None = None) -> dict:
    """A copy of `schema` that satisfies strict-mode JSON Schema.

    Strict mode (OpenAI's `strict: true`, Anthropic's compiled grammars)
    requires every object to set `additionalProperties: false` and to list
    EVERY property in `required`. `model_json_schema()` satisfies neither: it
    omits `additionalProperties` and leaves a field with a default out of
    `required`.

    So a property with a default becomes required — the model must emit it.
    That is the cost of strict mode working at all, and it is preferable to
    the alternative: a `strict: true` request that the provider rejects with a
    400 on every call.

    A bare `$ref` is left alone and the `$defs` entry it points at is rewritten
    in place, which is what the reference resolves to. A `$ref` carrying
    SIBLING keys is different: `Field(description=…)` on a nested model
    produces `{"$ref": …, "description": …}`, and OpenAI rejects that outright
    (`$ref cannot have keywords {'description'}`). Those are inlined — the
    target merged in under the siblings, so the description wins — which is
    what OpenAI's own SDK does for the same reason. Inlining goes ONE level:
    refs inside the inlined target stay refs, so a recursive model terminates.
    """
    if not isinstance(schema, dict):
        return schema

    defs = _defs if _defs is not None else _collect_defs(schema)
    schema = _inline_ref_with_siblings(schema, defs)

    out: dict = {}
    for key, value in schema.items():
        if key in ("properties", "$defs", "definitions", "patternProperties") and isinstance(value, dict):
            out[key] = {k: strictify_json_schema(v, defs) for k, v in value.items()}
        elif key in ("anyOf", "oneOf", "allOf", "prefixItems") and isinstance(value, list):
            out[key] = [strictify_json_schema(v, defs) for v in value]
        elif key in ("items", "not", "additionalProperties") and isinstance(value, dict):
            out[key] = strictify_json_schema(value, defs)
        else:
            out[key] = value

    # Gate on the type as well as on `properties`: `{"type": "object"}` with no
    # properties would otherwise sail through unrewritten and still be sent
    # next to `strict: true`, which is the exact 400 this function prevents.
    if isinstance(out.get("properties"), dict) or out.get("type") == "object":
        out["required"] = list(out.get("properties") or {})
        out["additionalProperties"] = False
    return out


# Anthropic's grammar compiler rejects these outright. Their published list;
# their own SDKs strip them the same way rather than failing the request.
_ANTHROPIC_UNSUPPORTED = (
    "minimum",
    "maximum",
    "multipleOf",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minLength",
    "maxLength",
    "pattern",
    "maxItems",
    "uniqueItems",
)
# `format` survives only for the values Anthropic implements.
_ANTHROPIC_FORMATS = frozenset(
    {"date-time", "time", "date", "duration", "email", "hostname", "uri", "ipv4", "ipv6", "uuid"}
)
# `minItems` is supported, but only as 0 or 1.
_ANTHROPIC_MIN_ITEMS = (0, 1)


def strip_unsupported_keywords(schema: dict) -> dict:
    """A copy of `schema` with the constraint keywords Anthropic's grammar
    refuses removed, each appended to that node's `description`.

    Anthropic 400s on what `Field(ge=…)` / `Field(min_length=…)` produce, so
    the same model that works on OpenAI would be rejected. Their own SDKs
    solve it this way: strip from the wire schema, keep the intent in the
    description, validate the reply against the original schema (`parse()`).

    Encodes one provider's published limits; called only from the Anthropic
    projection."""
    if not isinstance(schema, dict):
        return schema

    out: dict = {}
    dropped: list[str] = []
    for key, value in schema.items():
        if key in _ANTHROPIC_UNSUPPORTED:
            dropped.append(f"{key}: {value}")
            continue
        if key == "format" and value not in _ANTHROPIC_FORMATS:
            dropped.append(f"{key}: {value}")
            continue
        if key == "minItems" and value not in _ANTHROPIC_MIN_ITEMS:
            dropped.append(f"{key}: {value}")
            continue

        if key in ("properties", "$defs", "definitions", "patternProperties") and isinstance(value, dict):
            out[key] = {k: strip_unsupported_keywords(v) for k, v in value.items()}
        elif key in ("anyOf", "oneOf", "allOf", "prefixItems") and isinstance(value, list):
            out[key] = [strip_unsupported_keywords(v) for v in value]
        elif key in ("items", "not", "additionalProperties") and isinstance(value, dict):
            out[key] = strip_unsupported_keywords(value)
        else:
            out[key] = value

    if dropped:
        constraints = ", ".join(dropped)
        existing = out.get("description")
        out["description"] = f"{existing} ({constraints})" if existing else f"Constraints: {constraints}"
    return out


def _collect_defs(root: dict) -> dict:
    """The root's definition table, under either spelling."""
    defs: dict = {}
    for key in ("$defs", "definitions"):
        table = root.get(key)
        if isinstance(table, dict):
            defs[key] = table
    return defs


def _inline_ref_with_siblings(node: dict, defs: dict) -> dict:
    """`{"$ref": …, <siblings>}` → the target merged under the siblings.
    Untouched when there are no siblings or the pointer does not resolve."""
    ref = node.get("$ref")
    if not isinstance(ref, str) or len(node) == 1:
        return node
    target = _resolve_local_ref(ref, defs)
    if target is None:
        return node
    siblings = {k: v for k, v in node.items() if k != "$ref"}
    return {**target, **siblings}


def _resolve_local_ref(ref: str, defs: dict) -> dict | None:
    for key, table in defs.items():
        prefix = f"#/{key}/"
        if ref.startswith(prefix):
            target = table.get(ref[len(prefix) :])
            return target if isinstance(target, dict) else None
    return None


def parse_structured_output(text: str, response_format: Any) -> Any:
    """Validate `text` (a JSON string) against `response_format`.

    Returns:
      - dict, when response_format is a raw JSON Schema dict.
      - instance of the model, when response_format is a BaseModel subclass.
      - validated value, when response_format is a TypeAdapter.

    Raises StructuredOutputError on JSON-decode or validation failure.
    """
    import json

    from ..exceptions import StructuredOutputError

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise StructuredOutputError(
            f"Response is not valid JSON: {e}",
            original_exception=e,
        ) from e

    try:
        if isinstance(response_format, dict):
            return data
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            return response_format.model_validate(data)
        if isinstance(response_format, TypeAdapter):
            return response_format.validate_python(data)
    except Exception as e:
        raise StructuredOutputError(
            f"Response did not validate against schema: {e}",
            original_exception=e,
        ) from e

    raise StructuredOutputError(
        f"Unknown response_format type {type(response_format).__name__}; "
        "must be dict, BaseModel subclass, or TypeAdapter."
    )
