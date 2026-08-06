"""Tool definitions + projection.

`BaseTool` is the contract for everything accepted through `tools=`: `Tool`
(the standard function tool) and provider-native tools (defined in the
transport packages). A tool's `get_projector()` names the `ToolProjector`
that owns its wire shapes; None means the transport's default projector.

`Tool.parameters` accepts three styles: raw JSON Schema dict, Pydantic
BaseModel class, or TypeAdapter. Normalization to JSON Schema for the wire
happens in the projector's `project_tool_to_llm`; the original is preserved
on the Tool instance so the caller can validate inbound tool-call arguments
against it later.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, TypeAdapter

if TYPE_CHECKING:
    from .content import ToolCall
    from .messages import ToolMessage

# Accepted at the API boundary; not validated by pydantic (arbitrary types).
ToolParameters = dict | type | TypeAdapter


class BaseTool(BaseModel):
    """Everything accepted through `tools=`."""

    model_config = ConfigDict(extra="forbid")

    def get_projector(self) -> ToolProjector | None:
        """The projector owning this tool's wire shapes, or None for the
        transport's default (standard function-tool) projector."""
        return None


class Tool(BaseTool):
    name: str
    description: str
    parameters: Any  # one of: dict, type[BaseModel], TypeAdapter

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


ToolChoice = Literal["auto", "required", "none"] | dict


class ToolProjector:
    """All five wire touchpoints of one tool kind for one transport family.

    Concrete class with NotImplementedError hooks — house style, no ABC.
    Each transport defines a base implementing today's standard
    function-tool behavior; a native projector subclasses that base and
    overrides only what differs.
    """

    def project_tool_to_llm(self, tool: BaseTool) -> dict:
        """The tool's declaration — one entry in the wire `tools` list."""
        raise NotImplementedError

    def build_tool_call(self, item: dict) -> ToolCall:
        """A wire call item/block -> the canonical ToolCall."""
        raise NotImplementedError

    def project_tool_call_to_llm(self, tool_call: ToolCall) -> dict:
        """A ToolCall from history -> the wire item/block that replays it."""
        raise NotImplementedError

    def project_tool_message_to_llm(
        self,
        message: ToolMessage,
        tool_call: ToolCall | None,
    ) -> dict:
        """A tool result -> its wire shape. `tool_call` is the originating
        call when the transport found it in history, else None."""
        raise NotImplementedError

    def project_tool_choice_to_llm(self, tool: BaseTool) -> dict:
        """The tool_choice shape that forces this one tool."""
        raise NotImplementedError


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
