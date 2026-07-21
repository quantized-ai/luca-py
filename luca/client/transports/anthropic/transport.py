"""Anthropic Messages API transport.

Differences from OpenAI worth highlighting:
  - `system` is a top-level field (not a message).
  - Content blocks are first-class on the wire (text / tool_use / tool_result / thinking).
  - `max_tokens` is REQUIRED.
  - Auth header is `x-api-key`, plus `anthropic-version`.
  - URL is `{base_url}/v1/messages`.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

import httpx

from ...exceptions import (
    AuthenticationError,
    BadRequestError,
    ClientError,
    ContextLengthExceededError,
    InvalidModelError,
    ModelNotFoundError,
    ProviderAPIError,
    RateLimitError,
    UnsupportedParameterError,
)
from ...exceptions import ConnectionError as ClientConnectionError
from ...exceptions import TimeoutError as ClientTimeoutError
from ...types.completion import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Usage,
    UsageCost,
)
from ...types.content import (
    ImageBlock,
    RefusalBlock,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolResultBlock,
)
from ...types.media import MediaBase64, MediaFileId, MediaURL
from ...types.messages import AssistantMessage, ToolMessage, UserMessage
from ...types.tools import tool_parameters_to_json_schema
from ..base import BaseTransport, ChatCompletionTransportMixin

_DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
_DEFAULT_MAX_TOKENS = 4096


class AnthropicTransport(BaseTransport, ChatCompletionTransportMixin):
    transport_id = "anthropic"

    # Extended thinking comes in two mutually exclusive shapes and which one a
    # model takes is not discoverable from the wire: adaptive models reject
    # `budget_tokens` and manual models reject `adaptive`. Adaptive is the
    # default because that is where Anthropic is moving; the exceptions are
    # listed. Subclass to change any of it.
    MANUAL_THINKING_MODELS: ClassVar[tuple[str, ...]] = (
        "claude-opus-4-5", "claude-haiku-4-5",
        "claude-sonnet-4-5", "claude-opus-4-1",
        "claude-sonnet-4-0", "claude-opus-4-0",
    )
    NO_THINKING_MODELS: ClassVar[tuple[str, ...]] = (
        "claude-3-5-sonnet", "claude-3-5-haiku", "claude-3-opus",
        "claude-3-sonnet", "claude-3-haiku",
    )
    # Manual mode takes a token budget, not a word. Anthropic requires
    # >= 1024 and strictly < max_tokens.
    EFFORT_BUDGETS: ClassVar[dict[str, int]] = {
        "minimal": 1024, "low": 2048, "medium": 4096,
        "high": 8192, "xhigh": 16384, "auto": 4096,
    }
    # Adaptive takes a word, and accepts only these four.
    ADAPTIVE_EFFORTS: ClassVar[dict[str, str]] = {
        "minimal": "low", "low": "low", "medium": "medium",
        "high": "high", "xhigh": "xhigh",
    }
    MIN_THINKING_BUDGET: ClassVar[int] = 1024
    # Room left for the answer: generous when we size `max_tokens` ourselves,
    # only the bare minimum when squeezing a budget under a caller's cap.
    COMPLETION_HEADROOM: ClassVar[int] = 4096
    MIN_COMPLETION_TOKENS: ClassVar[int] = 1024
    # Adaptive models omit the reasoning text by default and return only the
    # encrypted signature, which leaves nothing to render. Ask for summaries.
    THINKING_DISPLAY: ClassVar[str | None] = "summarized"

    # --- headers / URL ---

    def _headers(self) -> dict[str, str]:
        h = {
            "Content-Type": "application/json",
            "anthropic-version": _DEFAULT_ANTHROPIC_VERSION,
        }
        if self._api_key:
            h["x-api-key"] = self._api_key
        return h

    def _chat_completion_url(self) -> str:
        return f"{self._base_url}/v1/messages"

    # --- payload building ---

    def _build_chat_completion_payload(
        self, request: ChatCompletionRequest, *, stream: bool = False,
    ) -> dict:
        thinking, max_tokens = self._thinking_config(request)
        payload: dict[str, Any] = {
            "model": request.model,
            "max_tokens": max_tokens,
            "messages": self._project_messages(request.messages),
        }
        payload.update(thinking)
        if request.system_message is not None:
            payload["system"] = self._project_system(request.system_message)
        if stream:
            payload["stream"] = True
        self._check_sampling(request, payload)
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.top_k is not None:
            payload["top_k"] = request.top_k
        if request.stop is not None:
            payload["stop_sequences"] = (
                [request.stop] if isinstance(request.stop, str) else list(request.stop)
            )
        if request.tools:
            payload["tools"] = self._project_tools(request.tools)
        if request.tool_choice is not None:
            payload["tool_choice"] = self._project_tool_choice(request.tool_choice)
        if request.metadata is not None:
            payload["metadata"] = request.metadata
        if request.extra_args:
            payload.update(request.extra_args)
        return payload

    def _check_sampling(self, request: ChatCompletionRequest, payload: dict) -> None:
        """Anthropic rejects the sampling controls while thinking is *active*.
        Thinking explicitly `disabled` is not active and leaves them legal, so
        the gate keys off the mode rather than the presence of the key.

        Refused rather than stripped: silently dropping a caller's temperature
        changes their output with nothing to notice."""
        if payload.get("thinking", {}).get("type") in (None, "disabled"):
            return
        conflicting = [
            name for name, value in (
                ("temperature", request.temperature),
                ("top_p", request.top_p),
                ("top_k", request.top_k),
            )
            if value is not None
        ]
        if conflicting:
            raise UnsupportedParameterError(
                f"{', '.join(conflicting)} cannot be set while extended "
                f"thinking is active on {request.model!r}.",
                provider=self._provider,
            )

    def _thinking_mode(self, model: str) -> str:
        """`"manual"`, `"adaptive"` or `"none"` for a wire model id.

        Matches on a normalized id so `-latest`, a dated suffix and a gateway
        prefix (`us.anthropic.claude-…`) all resolve to the same bucket."""
        normalized = model.rsplit(".", 1)[-1].rsplit("/", 1)[-1]
        for prefix in self.NO_THINKING_MODELS:
            if normalized.startswith(prefix):
                return "none"
        for prefix in self.MANUAL_THINKING_MODELS:
            if normalized.startswith(prefix):
                return "manual"
        return "adaptive"

    def _thinking_config(self, request: ChatCompletionRequest) -> tuple[dict, int]:
        """The thinking-related payload keys plus the `max_tokens` to send.

        `reasoning_effort=None` leaves thinking off entirely, so a caller that
        never asked for it keeps the old wire shape."""
        max_tokens = request.max_tokens or _DEFAULT_MAX_TOKENS
        effort = request.reasoning_effort
        if effort is None:
            return {}, max_tokens

        mode = self._thinking_mode(request.model)
        if effort == "none" or mode == "none":
            return {"thinking": {"type": "disabled"}}, max_tokens

        if mode == "adaptive":
            config: dict[str, Any] = {"type": "adaptive"}
            if self.THINKING_DISPLAY is not None:
                config["display"] = self.THINKING_DISPLAY
            thinking: dict[str, Any] = {"thinking": config}
            # "auto" means "let the model decide", which is the absence of an
            # effort key rather than any particular value.
            adaptive_effort = self.ADAPTIVE_EFFORTS.get(effort)
            if adaptive_effort is not None:
                thinking["output_config"] = {"effort": adaptive_effort}
            return thinking, max_tokens

        budget = self.EFFORT_BUDGETS.get(effort, self.MIN_THINKING_BUDGET)
        if request.max_tokens is None:
            max_tokens = budget + self.COMPLETION_HEADROOM
        else:
            # The caller's cap is a billing contract: shrink the budget to fit
            # inside it rather than quietly raising it.
            budget = min(budget, request.max_tokens - self.MIN_COMPLETION_TOKENS)
        if budget < self.MIN_THINKING_BUDGET:
            raise UnsupportedParameterError(
                f"max_tokens={request.max_tokens} leaves no room for extended "
                f"thinking on {request.model!r}: Anthropic requires a budget "
                f"of at least {self.MIN_THINKING_BUDGET} tokens below it.",
                provider=self._provider,
            )
        return {"thinking": {"type": "enabled", "budget_tokens": budget}}, max_tokens

    def _project_system(self, system_message: Any) -> Any:
        if isinstance(system_message, str):
            return system_message
        return [{"type": "text", "text": b.text} for b in system_message if isinstance(b, TextBlock)]

    def _project_messages(self, messages: list) -> list[dict]:
        out: list[dict] = []
        for msg in messages:
            if isinstance(msg, UserMessage):
                out.append(self._project_user_message(msg))
            elif isinstance(msg, AssistantMessage):
                out.append(self._project_assistant_message(msg))
            elif isinstance(msg, ToolMessage):
                # Anthropic represents tool results as a user message with a
                # tool_result content block.
                out.append(self._project_tool_message_as_user(msg))
            else:
                raise BadRequestError(
                    f"Unknown message type {type(msg).__name__}",
                    provider=self._provider,
                )
        return out

    def _project_user_message(self, msg: UserMessage) -> dict:
        if isinstance(msg.content, str):
            return {"role": "user", "content": msg.content}
        wire_blocks: list[dict] = []
        for block in msg.content:
            if isinstance(block, TextBlock):
                wire_blocks.append({"type": "text", "text": block.text})
            elif isinstance(block, ImageBlock):
                wire_blocks.append(self._project_image_block(block))
            else:
                # AudioBlock / FileBlock — best-effort
                wire_blocks.append({"type": "text", "text": str(block)})
        return {"role": "user", "content": wire_blocks}

    def _project_image_block(self, block: ImageBlock) -> dict:
        source = block.source
        if isinstance(source, MediaURL):
            return {"type": "image", "source": {"type": "url", "url": source.url}}
        if isinstance(source, MediaBase64):
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": source.media_type,
                    "data": source.data,
                },
            }
        if isinstance(source, MediaFileId):
            return {"type": "image", "source": {"type": "file", "file_id": source.file_id}}
        raise BadRequestError(f"Unknown image source type", provider=self._provider)

    def _project_assistant_message(self, msg: AssistantMessage) -> dict:
        wire_blocks: list[dict] = []
        for block in msg.content:
            if isinstance(block, TextBlock):
                wire_blocks.append({"type": "text", "text": block.text})
            elif isinstance(block, ThinkingBlock):
                thinking = self._project_thinking_block(block)
                if thinking is not None:
                    wire_blocks.append(thinking)
            elif isinstance(block, ToolCall):
                wire_blocks.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.arguments,
                })
            elif isinstance(block, RefusalBlock):
                # Anthropic doesn't take refusals on the way in; drop.
                continue
        return {"role": "assistant", "content": wire_blocks}

    def _project_thinking_block(self, block: ThinkingBlock) -> dict | None:
        """One thinking block on the way back, or None to omit it.

        An unsigned block is DROPPED rather than sent. Anthropic rejects a
        thinking block whose signature is missing (400 `signature: Field
        required`) but accepts the turn with the block absent, and unsigned
        blocks are reachable: a truncated response never receives its
        `signature_delta`, and a session moved from an OpenAI-compatible host
        carries reasoning text that was never signed. Sending it would make
        the whole conversation permanently unusable; dropping it costs one
        turn's visible reasoning."""
        if block.signature is None:
            return None
        if block.redacted:
            return {"type": "redacted_thinking", "data": block.signature}
        return {
            "type": "thinking",
            "thinking": block.text,
            "signature": block.signature,
        }

    def _project_tool_message_as_user(self, msg: ToolMessage) -> dict:
        if isinstance(msg.content, str):
            content: Any = msg.content
        else:
            # Mixed text/image content — keep as block list.
            content = []
            for b in msg.content:
                if isinstance(b, TextBlock):
                    content.append({"type": "text", "text": b.text})
                elif isinstance(b, ImageBlock):
                    content.append(self._project_image_block(b))
        return {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": msg.tool_call_id,
                "content": content,
                "is_error": msg.is_error,
            }],
        }

    def _project_tools(self, tools: list) -> list[dict]:
        out = []
        for t in tools:
            schema = tool_parameters_to_json_schema(t.parameters)
            out.append({
                "name": t.name,
                "description": t.description,
                "input_schema": schema,
            })
        return out

    def _project_tool_choice(self, choice: Any) -> Any:
        if isinstance(choice, str):
            return {
                "auto": {"type": "auto"},
                "required": {"type": "any"},
                "none": {"type": "none"},
            }.get(choice, {"type": "auto"})
        if isinstance(choice, dict) and "name" in choice:
            return {"type": "tool", "name": choice["name"]}
        return choice

    # --- response parsing ---

    def _parse_chat_completion_response(
        self, response: httpx.Response, request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        data = response.json()
        message = self._parse_assistant_message(data, request)
        message.usage = self._parse_usage(data.get("usage"), request.model_info)

        provider_terminal = data.get("stop_reason")
        canonical, error_message = self._classify_finish(provider_terminal, message)
        message.finish_reason = canonical
        message.provider_finish_reason = provider_terminal
        message.error_message = error_message

        resp = ChatCompletionResponse(message=message, raw=data)
        resp._response_format = request.response_format
        return resp

    def _parse_assistant_message(
        self, data: dict, request: ChatCompletionRequest,
    ) -> AssistantMessage:
        content: list = []
        for block in data.get("content") or []:
            block_type = block.get("type")
            if block_type == "text":
                content.append(TextBlock(text=block.get("text", "")))
            elif block_type == "thinking":
                content.append(ThinkingBlock(
                    text=block.get("thinking", ""),
                    signature=block.get("signature"),
                ))
            elif block_type == "tool_use":
                content.append(ToolCall(
                    id=block["id"],
                    name=block["name"],
                    arguments=block.get("input", {}) or {},
                    complete=True,
                ))
            elif block_type == "redacted_thinking":
                content.append(ThinkingBlock(
                    text="",
                    signature=block.get("data"),
                    redacted=True,
                ))
        return AssistantMessage(
            content=content,
            provider=self._provider,
            model=data.get("model") or request.model,
            response_id=data.get("id"),
        )

    def _parse_usage(self, usage_json: dict | None, model_info: Any) -> Usage:
        if usage_json is None:
            return Usage()
        u = Usage(
            input_tokens=usage_json.get("input_tokens", 0),
            output_tokens=usage_json.get("output_tokens", 0),
            total_tokens=(usage_json.get("input_tokens", 0) + usage_json.get("output_tokens", 0)),
            cached_input_tokens=usage_json.get("cache_read_input_tokens"),
            cache_write_tokens=usage_json.get("cache_creation_input_tokens"),
        )
        if model_info is not None and getattr(model_info, "cost", None) is not None:
            u.cost = UsageCost.compute(u, model_info.cost)
        return u

    # --- finish-reason classification ---

    def _classify_finish(
        self, provider_value: str | None, message: AssistantMessage,
    ) -> tuple[str | None, str | None]:
        if provider_value == "end_turn":
            return ("stop", None)
        if provider_value == "max_tokens":
            return ("length", None)
        if provider_value == "tool_use":
            return ("tool_use", None)
        if provider_value == "stop_sequence":
            return ("stop", None)
        if provider_value == "refusal":
            return ("error", "Anthropic refusal stop reason")
        if provider_value == "sensitive":
            return ("error", "Anthropic safety filter (sensitive content)")
        if provider_value is None:
            return (None, None)
        return (provider_value, None)

    # --- error mapping ---

    def _map_chat_completion_http_error(self, exc: httpx.HTTPError) -> ClientError:
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            body = self._safe_json(exc.response)
            err_obj = (body or {}).get("error") if isinstance(body, dict) else {}
            err_type = (err_obj or {}).get("type", "") if isinstance(err_obj, dict) else ""
            msg = (err_obj or {}).get("message", str(exc)) if isinstance(err_obj, dict) else str(exc)

            if status == 401:
                return AuthenticationError(
                    msg, provider=self._provider, original_exception=exc,
                )
            if status == 429:
                return RateLimitError(
                    msg, provider=self._provider, original_exception=exc,
                    retry_after=self._retry_after(exc.response),
                )
            if status == 400:
                if "context_length" in msg.lower() or "too long" in msg.lower():
                    return ContextLengthExceededError(
                        msg, provider=self._provider, original_exception=exc,
                    )
                if err_type == "invalid_request_error" and "model" in msg.lower():
                    return InvalidModelError(
                        msg, provider=self._provider, original_exception=exc,
                    )
                return BadRequestError(
                    msg, provider=self._provider, original_exception=exc,
                )
            if status == 404:
                return ModelNotFoundError(
                    msg, provider=self._provider, original_exception=exc,
                )
            if 500 <= status < 600:
                return ProviderAPIError(
                    msg, provider=self._provider, original_exception=exc,
                )
            return ProviderAPIError(
                msg, provider=self._provider, original_exception=exc,
            )

        if isinstance(exc, httpx.TimeoutException):
            return ClientTimeoutError(
                str(exc), provider=self._provider, original_exception=exc,
            )
        if isinstance(exc, httpx.NetworkError):
            return ClientConnectionError(
                str(exc), provider=self._provider, original_exception=exc,
            )
        return ProviderAPIError(
            str(exc), provider=self._provider, original_exception=exc,
        )

    @staticmethod
    def _safe_json(response: httpx.Response) -> dict | None:
        try:
            return response.json()
        except Exception:
            return None

    @staticmethod
    def _retry_after(response: httpx.Response) -> float | None:
        val = response.headers.get("retry-after")
        if val is None:
            return None
        try:
            return float(val)
        except ValueError:
            return None

    # --- stream class hooks ---

    def _chat_completion_stream_class(self) -> type:
        from .stream import AnthropicChatCompletionStream
        return AnthropicChatCompletionStream

    def _async_chat_completion_stream_class(self) -> type:
        from .stream import AnthropicAsyncChatCompletionStream
        return AnthropicAsyncChatCompletionStream
