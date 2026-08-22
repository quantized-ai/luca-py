"""ChatCompletion request and response DTOs, plus Usage / UsageCost."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

from .catalog import ModelCost, ModelInfo
from .content import TextBlock
from .messages import AssistantMessage, Message
from .reasoning import Reasoning
from .structured import parse_structured_output
from .tools import BaseTool, Tool, ToolChoice


class UsageCost(BaseModel):
    input: float = 0.0
    output: float = 0.0
    cached_input: float = 0.0
    cache_write: float = 0.0
    reasoning: float = 0.0
    total: float = 0.0

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def compute(cls, usage: Usage, cost: ModelCost) -> UsageCost:
        def _per_m(tokens: int | None, rate: float | None) -> float:
            if tokens is None or rate is None:
                return 0.0
            return (tokens / 1_000_000.0) * rate

        inp = _per_m(usage.input_tokens, cost.input_per_million_tokens)
        out = _per_m(usage.output_tokens, cost.output_per_million_tokens)
        cached = _per_m(usage.cached_input_tokens, cost.cached_input_per_million_tokens)
        cwrite = _per_m(usage.cache_write_tokens, cost.cache_write_per_million_tokens)
        reasoning = _per_m(usage.reasoning_tokens, cost.reasoning_per_million_tokens)
        return cls(
            input=inp,
            output=out,
            cached_input=cached,
            cache_write=cwrite,
            reasoning=reasoning,
            total=inp + out + cached + cwrite + reasoning,
        )


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None
    audio_input_tokens: int | None = None
    audio_output_tokens: int | None = None
    image_input_tokens: int | None = None
    # Requests made by provider-hosted tools, normalized to
    # `{tool_name: count}` (e.g. {"web_search": 2}); `provider_tool_usage` is
    # the provider's own payload behind those counts, verbatim.
    tool_requests: dict[str, int] = Field(default_factory=dict)
    provider_tool_usage: dict[str, Any] = Field(default_factory=dict)
    cost: UsageCost | None = None

    model_config = ConfigDict(extra="forbid")


class ChatCompletionRequest(BaseModel):
    model: str
    provider: str | None = None
    messages: list[Message] = Field(default_factory=list)
    system_message: str | list[TextBlock] | None = None
    tools: list[BaseTool] | None = None
    tool_choice: ToolChoice | None = None
    response_format: Any | None = None  # ResponseFormat is Union[dict, type, TypeAdapter] — accept Any

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None

    reasoning: Reasoning | None = None

    cache_retention: Literal["none", "short", "long"] | None = None
    session_id: str | None = None

    parallel_tool_calls: bool | None = None
    user: str | None = None

    model_info: ModelInfo | dict | None = None

    metadata: dict | None = None
    # Raw provider-specific options, keyed by provider name. Only the
    # matching provider's dict is used, and it wins over anything the
    # transport derived.
    provider_options: dict[str, dict] | None = None

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    @field_validator("tools", mode="before")
    @classmethod
    def _coerce_tool_dicts(cls, value: Any) -> Any:
        # Dicts mean STANDARD function tools; native tools are instances and
        # pass through untouched. Same rule as the helper's _coerce_tools.
        if not isinstance(value, list):
            return value
        return [Tool.model_validate(t) if isinstance(t, dict) else t for t in value]


class ChatCompletionResponse(BaseModel):
    """The assistant messages one completion produced, in wire order.

    `messages` is never empty — every transport parses a response into at
    least one `AssistantMessage`, and the terminal state a caller reads
    (finish_reason, provider_finish_reason, usage, provider, model,
    tool_calls, …) lives on the LAST one. There is no attribute forwarding:
    read `response.messages[-1].finish_reason`, not `response.finish_reason`.
    No duplicated storage."""

    messages: list[AssistantMessage] = Field(min_length=1)
    raw: Any = Field(default=None, exclude=True)

    _response_format: Any | None = PrivateAttr(default=None)

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    def parse(self) -> Any:
        if self._response_format is None:
            raise ValueError(
                "No response_format was set on the originating request; "
                "cannot parse(). Set response_format= when calling completion()."
            )
        # Concatenate text blocks across every message; ignore thinking /
        # refusal / tool_call.
        from .content import TextBlock as _TextBlock

        text = "".join(
            block.text for message in self.messages for block in message.content if isinstance(block, _TextBlock)
        )
        return parse_structured_output(text, self._response_format)
