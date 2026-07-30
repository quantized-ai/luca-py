"""Structured output (`response_format`) accepts the same three styles as Tool.parameters.

A raw `dict` is ALWAYS a JSON Schema — never a wire-level `response_format`
payload. Providers spell the wire shape differently (OpenAI nests it under
`json_schema`, the Responses API puts it on `text.format`, Anthropic on
`output_config.format`), so one canonical schema in and one projection per
transport out. A caller who needs a hand-written wire payload passes it
through `provider_options`.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, TypeAdapter

ResponseFormat = dict | type | TypeAdapter

DEFAULT_SCHEMA_NAME = "structured_output"


def response_format_to_json_schema(response_format: Any) -> tuple[str, dict]:
    """Convert any of the three response_format forms to `(name, schema)`.

    The name is what providers label the schema with on the wire. A Pydantic
    model lends its class name; the other two forms have none, so they get
    `DEFAULT_SCHEMA_NAME`.
    """
    if isinstance(response_format, dict):
        return DEFAULT_SCHEMA_NAME, response_format
    if isinstance(response_format, type) and issubclass(response_format, BaseModel):
        return response_format.__name__, response_format.model_json_schema()
    if isinstance(response_format, TypeAdapter):
        return DEFAULT_SCHEMA_NAME, response_format.json_schema()
    raise TypeError(
        f"response_format must be a dict, BaseModel subclass, or TypeAdapter; got {type(response_format).__name__}"
    )


def strictify_json_schema(schema: dict) -> dict:
    """A copy of `schema` that satisfies strict-mode JSON Schema.

    Strict mode (OpenAI's `strict: true`, Anthropic's compiled grammars)
    requires every object to set `additionalProperties: false` and to list
    EVERY property in `required`. `model_json_schema()` satisfies neither: it
    omits `additionalProperties` and leaves a field with a default out of
    `required`.

    So a property with a default becomes required — the model must emit it.
    That is the cost of strict mode working at all, and it is preferable to
    the alternative: a `strict: true` request that the provider rejects with a
    400 on every call. `$ref` is left alone; the referenced `$defs` entries
    are rewritten in place, which is what the reference resolves to.
    """
    if not isinstance(schema, dict):
        return schema

    out: dict = {}
    for key, value in schema.items():
        if key in ("properties", "$defs", "definitions", "patternProperties") and isinstance(value, dict):
            out[key] = {k: strictify_json_schema(v) for k, v in value.items()}
        elif key in ("anyOf", "oneOf", "allOf", "prefixItems") and isinstance(value, list):
            out[key] = [strictify_json_schema(v) for v in value]
        elif key in ("items", "not", "additionalProperties") and isinstance(value, dict):
            out[key] = strictify_json_schema(value)
        else:
            out[key] = value

    if isinstance(out.get("properties"), dict):
        out["required"] = list(out["properties"])
        out["additionalProperties"] = False
    return out


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
