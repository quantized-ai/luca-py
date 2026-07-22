"""ChatCompletion request and response DTOs, plus Usage / UsageCost."""

from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from .catalog import ModelCost, ModelInfo
from .content import TextBlock
from .messages import AssistantMessage, Message
from .reasoning import Reasoning
from .structured import ResponseFormat, parse_structured_output
from .tools import Tool, ToolChoice


class UsageCost(BaseModel):
    input: float = 0.0
    output: float = 0.0
    cached_input: float = 0.0
    cache_write: float = 0.0
    reasoning: float = 0.0
    total: float = 0.0

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def compute(cls, usage: "Usage", cost: ModelCost) -> "UsageCost":
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
    cost: UsageCost | None = None

    model_config = ConfigDict(extra="forbid")


class ChatCompletionRequest(BaseModel):
    model: str
    provider: str | None = None
    messages: list[Message] = Field(default_factory=list)
    system_message: Union[str, list[TextBlock]] | None = None
    tools: list[Tool] | None = None
    tool_choice: ToolChoice | None = None
    response_format: Any | None = None  # ResponseFormat is Union[dict, type, TypeAdapter] — accept Any

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    stop: Union[str, list[str]] | None = None
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

    model_info: Union[ModelInfo, dict] | None = None

    metadata: dict | None = None
    # Raw provider-specific options, keyed by provider name. Only the
    # matching provider's dict is used, and it wins over anything the
    # transport derived.
    provider_options: dict[str, dict] | None = None

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


# pydantic v2 routes its own `__getattribute__` for declared fields and private
# attrs. To forward unknown attribute lookups to `self.message` we override
# `__getattr__`, which Python only calls after normal lookup fails.
_RESPONSE_DECLARED_FIELDS = {"message", "raw"}
_RESPONSE_DECLARED_METHODS = {"parse"}


class ChatCompletionResponse(BaseModel):
    """Wraps an AssistantMessage. Everything callers commonly read off the
    response — finish_reason, provider_finish_reason, usage, provider, model,
    tool_calls, … — lives on self.message and is reached via attribute
    forwarding (__getattr__ below). No duplicated storage."""

    message: AssistantMessage
    raw: Any = Field(default=None, exclude=True)

    _response_format: Any | None = PrivateAttr(default=None)

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    def __getattr__(self, name: str) -> Any:
        # Defer to pydantic's __getattr__ for private attrs, declared fields,
        # declared methods, and dunders — pydantic handles PrivateAttr lookup
        # in __pydantic_private__ and we MUST NOT intercept that.
        if (
            name.startswith("_")
            or name in _RESPONSE_DECLARED_FIELDS
            or name in _RESPONSE_DECLARED_METHODS
        ):
            return super().__getattr__(name)

        # Forward unknown attribute lookups to self.message. Reach the message
        # via object.__getattribute__ to avoid triggering pydantic's recursion
        # safeguards on partially-constructed instances.
        message = object.__getattribute__(self, "__dict__").get("message")
        if message is None:
            raise AttributeError(name)
        try:
            return getattr(message, name)
        except AttributeError:
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r} "
                f"(also not found on self.message: {type(message).__name__})"
            )

    def parse(self) -> Any:
        if self._response_format is None:
            raise ValueError(
                "No response_format was set on the originating request; "
                "cannot parse(). Set response_format= when calling completion()."
            )
        # Concatenate text blocks; ignore thinking / refusal / tool_call.
        from .content import TextBlock as _TextBlock

        text = "".join(
            block.text for block in self.message.content if isinstance(block, _TextBlock)
        )
        return parse_structured_output(text, self._response_format)
