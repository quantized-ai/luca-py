"""Tool definition + tool choice. `Tool.parameters` accepts three styles:
raw JSON Schema dict, Pydantic BaseModel class, or TypeAdapter.

Normalization to JSON Schema for the wire happens in the transport's
_project_tools method; the original is preserved on the Tool instance so
the caller can validate inbound tool-call arguments against it later.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, TypeAdapter

# Accepted at the API boundary; not validated by pydantic (arbitrary types).
ToolParameters = dict | type | TypeAdapter


class Tool(BaseModel):
    name: str
    description: str
    parameters: Any  # one of: dict, type[BaseModel], TypeAdapter
    # A provider-defined tool: the provider owns the schema and the model was
    # trained on it, so the wire carries this type string and the transport
    # skips `description` / `parameters` entirely. `None` is an ordinary tool.
    # A transport that does not know a given type must say so rather than
    # silently send a tool the model cannot use.
    provider_type: str | None = None

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


ToolChoice = Literal["auto", "required", "none"] | dict


def tool_parameters_to_json_schema(parameters: Any) -> dict:
    """Convert any of the three parameter forms to a JSON Schema dict.

    - dict → returned as-is (caller responsibility).
    - BaseModel subclass → `.model_json_schema()`.
    - TypeAdapter → `.json_schema()`.
    """
    if isinstance(parameters, dict):
        return parameters
    if isinstance(parameters, type) and issubclass(parameters, BaseModel):
        return parameters.model_json_schema()
    if isinstance(parameters, TypeAdapter):
        return parameters.json_schema()
    raise TypeError(
        f"Tool.parameters must be a dict, BaseModel subclass, or TypeAdapter; got {type(parameters).__name__}"
    )
