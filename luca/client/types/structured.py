"""Structured output (`response_format`) accepts the same three styles as Tool.parameters."""

from __future__ import annotations

from typing import Any, Union

from pydantic import BaseModel, TypeAdapter

ResponseFormat = Union[dict, type, TypeAdapter]


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
        )

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
        )

    raise StructuredOutputError(
        f"Unknown response_format type {type(response_format).__name__}; "
        "must be dict, BaseModel subclass, or TypeAdapter."
    )
