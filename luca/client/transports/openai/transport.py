"""OpenAI Chat Completions transport.

Speaks the OpenAI Chat Completions wire format. Used by OpenAIProvider, and
(via PROVIDERS dict entries -> GenericProvider) by every OpenAI-compatible
host (Groq, DeepSeek, Together, Ollama, Cerebras, Fireworks, ...). The host's
identity is on self._provider.
"""

from __future__ import annotations

import json
from typing import Any

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
)
from ...types.media import MediaBase64, MediaFileId, MediaURL
from ...types.messages import AssistantMessage, ToolMessage, UserMessage
from ...types.tools import tool_parameters_to_json_schema
from ..base import BaseTransport, ChatCompletionTransportMixin


class OpenAITransport(BaseTransport, ChatCompletionTransportMixin):
    transport_id = "openai"

    # --- payload building ---

    def _build_chat_completion_payload(
        self, request: ChatCompletionRequest, *, stream: bool = False,
    ) -> dict:
        wire_messages = self._project_messages(request.messages)
        if request.system_message is not None:
            wire_messages = [self._project_system_message(request.system_message)] + wire_messages

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": wire_messages,
        }
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}

        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.stop is not None:
            payload["stop"] = request.stop
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.presence_penalty is not None:
            payload["presence_penalty"] = request.presence_penalty
        if request.frequency_penalty is not None:
            payload["frequency_penalty"] = request.frequency_penalty
        if request.logprobs is not None:
            payload["logprobs"] = request.logprobs
        if request.top_logprobs is not None:
            payload["top_logprobs"] = request.top_logprobs
        if request.parallel_tool_calls is not None:
            payload["parallel_tool_calls"] = request.parallel_tool_calls
        if request.user is not None:
            payload["user"] = request.user
        if request.reasoning_effort is not None:
            payload["reasoning_effort"] = request.reasoning_effort

        if request.tools:
            payload["tools"] = self._project_tools(request.tools)
        if request.tool_choice is not None:
            payload["tool_choice"] = self._project_tool_choice(request.tool_choice)
        if request.response_format is not None:
            payload["response_format"] = self._project_response_format(request.response_format)

        if request.extra_args:
            payload.update(request.extra_args)
        return payload

    def _project_system_message(self, system_message: Any) -> dict:
        """Canonical str | list[TextBlock] → OpenAI wire-level system entry."""
        if isinstance(system_message, str):
            return {"role": "system", "content": system_message}
        # list[TextBlock] — concatenate into one string (OpenAI doesn't have
        # multi-segment system prompts beyond cache markers, which we don't
        # forward in v1).
        text = "".join(b.text for b in system_message if isinstance(b, TextBlock))
        return {"role": "system", "content": text}

    def _project_messages(self, messages: list) -> list[dict]:
        out: list[dict] = []
        for msg in messages:
            if isinstance(msg, UserMessage):
                out.append(self._project_user_message(msg))
            elif isinstance(msg, AssistantMessage):
                out.append(self._project_assistant_message(msg))
            elif isinstance(msg, ToolMessage):
                out.append(self._project_tool_message(msg))
            else:
                raise BadRequestError(
                    f"Unknown message type {type(msg).__name__}",
                    provider=self._provider,
                )
        return out

    def _project_user_message(self, msg: UserMessage) -> dict:
        if isinstance(msg.content, str):
            wire: dict = {"role": "user", "content": msg.content}
        else:
            wire_content = [self._project_user_block(b) for b in msg.content]
            # If everything is text, flatten to a single string for the common case.
            if all(isinstance(b, TextBlock) for b in msg.content):
                wire = {"role": "user", "content": "".join(b.text for b in msg.content if isinstance(b, TextBlock))}
            else:
                wire = {"role": "user", "content": wire_content}
        if msg.name is not None:
            wire["name"] = msg.name
        return wire

    def _project_user_block(self, block: Any) -> dict:
        if isinstance(block, TextBlock):
            return {"type": "text", "text": block.text}
        if isinstance(block, ImageBlock):
            return {"type": "image_url", "image_url": self._project_media_to_image_url(block.source)}
        # AudioBlock / FileBlock — best-effort projection (OpenAI accepts inline_data
        # for some models). For v1 we use the raw provider's hint or fall back to
        # extra_args; pass through as a dict containing the source.
        return {"type": "text", "text": str(block)}

    def _project_media_to_image_url(self, source: Any) -> dict:
        if isinstance(source, MediaURL):
            entry = {"url": source.url}
            return entry
        if isinstance(source, MediaBase64):
            return {"url": f"data:{source.media_type};base64,{source.data}"}
        if isinstance(source, MediaFileId):
            return {"url": source.file_id}
        raise BadRequestError(
            f"Unknown media source type {type(source).__name__}",
            provider=self._provider,
        )

    def _project_assistant_message(self, msg: AssistantMessage) -> dict:
        wire: dict = {"role": "assistant"}
        text_parts: list[str] = []
        tool_calls_wire: list[dict] = []
        for block in msg.content:
            if isinstance(block, TextBlock):
                text_parts.append(block.text)
            elif isinstance(block, ThinkingBlock):
                # OpenAI doesn't have a public thinking-replay surface; drop the text
                # but keep the structure preserved if the upstream supports it.
                # In v1 we just skip — round-trip is provider-internal. (Future:
                # OpenRouter replay via `reasoning_details` + `signature`.)
                continue
            elif isinstance(block, RefusalBlock):
                wire["refusal"] = block.text
            elif isinstance(block, ToolCall):
                tool_calls_wire.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.arguments) if block.arguments else "{}",
                    },
                })
        if text_parts:
            wire["content"] = "".join(text_parts)
        else:
            wire["content"] = None
        if tool_calls_wire:
            wire["tool_calls"] = tool_calls_wire
        return wire

    def _project_tool_message(self, msg: ToolMessage) -> dict:
        if isinstance(msg.content, str):
            content = msg.content
        else:
            content = "".join(b.text for b in msg.content if isinstance(b, TextBlock))
        wire: dict = {
            "role": "tool",
            "tool_call_id": msg.tool_call_id,
            "content": content,
        }
        if msg.name is not None:
            wire["name"] = msg.name
        return wire

    def _project_tools(self, tools: list) -> list[dict]:
        out = []
        for t in tools:
            schema = tool_parameters_to_json_schema(t.parameters)
            out.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": schema,
                },
            })
        return out

    def _project_tool_choice(self, choice: Any) -> Any:
        if isinstance(choice, str):
            return choice
        if isinstance(choice, dict):
            # {"name": "x"} → {"type": "function", "function": {"name": "x"}}
            if set(choice.keys()) == {"name"}:
                return {"type": "function", "function": {"name": choice["name"]}}
            return choice
        return choice

    def _project_response_format(self, fmt: Any) -> Any:
        if isinstance(fmt, dict):
            return fmt
        try:
            from pydantic import BaseModel, TypeAdapter
        except ImportError:  # pragma: no cover
            return fmt
        if isinstance(fmt, type) and issubclass(fmt, BaseModel):
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": fmt.__name__,
                    "schema": fmt.model_json_schema(),
                    "strict": True,
                },
            }
        if isinstance(fmt, TypeAdapter):
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "schema": fmt.json_schema(),
                    "strict": True,
                },
            }
        return fmt

    # --- response parsing ---

    def _parse_chat_completion_response(
        self, response: httpx.Response, request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        data = response.json()
        choices = data.get("choices") or []
        choice = choices[0] if choices else {}
        message_json = choice.get("message") or {}
        message = self._parse_assistant_message(message_json, request, data)
        message.usage = self._parse_usage(data.get("usage"), request.model_info)

        provider_terminal = choice.get("finish_reason")
        canonical, error_message = self._classify_finish(provider_terminal, message)
        message.finish_reason = canonical
        message.provider_finish_reason = provider_terminal
        message.error_message = error_message

        resp = ChatCompletionResponse(message=message, raw=data)
        resp._response_format = request.response_format
        return resp

    def _parse_assistant_message(
        self, msg_json: dict, request: ChatCompletionRequest, full_data: dict,
    ) -> AssistantMessage:
        content: list = []
        # Reasoning ("thinking") text — present only when the host carries it
        # (OpenRouter: `reasoning`; DeepSeek-style: `reasoning_content`). Pure
        # OpenAI responses omit the field, so this branch is inert for them —
        # data-driven, not a provider check. Prepend so content order is
        # [thinking, text, refusal?, tool_calls...] (reason-then-answer).
        reasoning = msg_json.get("reasoning") or msg_json.get("reasoning_content")
        if reasoning:
            content.append(ThinkingBlock(text=reasoning))
        text = msg_json.get("content")
        if text:
            content.append(TextBlock(text=text))
        refusal = msg_json.get("refusal")
        if refusal:
            content.append(RefusalBlock(text=refusal))
        for tc in msg_json.get("tool_calls") or []:
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except (json.JSONDecodeError, KeyError, TypeError):
                args = {}
            content.append(
                ToolCall(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    arguments=args,
                    complete=True,
                ),
            )
        return AssistantMessage(
            content=content,
            provider=self._provider,
            model=request.model,
            response_id=full_data.get("id"),
        )

    def _parse_usage(self, usage_json: dict | None, model_info: Any) -> Usage:
        if usage_json is None:
            return Usage()
        details = usage_json.get("completion_tokens_details") or {}
        u = Usage(
            input_tokens=usage_json.get("prompt_tokens", 0),
            output_tokens=usage_json.get("completion_tokens", 0),
            total_tokens=usage_json.get("total_tokens", 0),
            reasoning_tokens=details.get("reasoning_tokens"),
        )
        if model_info is not None and getattr(model_info, "cost", None) is not None:
            u.cost = UsageCost.compute(u, model_info.cost)
        return u

    # --- finish-reason classification ---

    def _classify_finish(
        self, provider_value: str | None, message: AssistantMessage,
    ) -> tuple[str | None, str | None]:
        # Strict-mode refusal: upstream "stop" but a RefusalBlock was emitted.
        has_refusal = any(isinstance(b, RefusalBlock) for b in message.content)

        if provider_value == "stop":
            if has_refusal:
                refusal = next(b for b in message.content if isinstance(b, RefusalBlock))
                return ("error", f"OpenAI refusal: {refusal.text}")
            return ("stop", None)
        if provider_value == "length":
            return ("length", None)
        if provider_value == "tool_calls":
            return ("tool_use", None)
        if provider_value == "function_call":
            return ("tool_use", None)
        if provider_value == "content_filter":
            return ("error", "Provider safety filter (content_filter)")
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
                if "context_length" in err_type or "context_length" in msg.lower():
                    return ContextLengthExceededError(
                        msg, provider=self._provider, original_exception=exc,
                    )
                if "model" in err_type.lower():
                    return InvalidModelError(
                        msg, provider=self._provider, original_exception=exc,
                    )
                if "unsupported" in err_type.lower():
                    return UnsupportedParameterError(
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
        from .stream import OpenAIChatCompletionStream
        return OpenAIChatCompletionStream

    def _async_chat_completion_stream_class(self) -> type:
        from .stream import OpenAIAsyncChatCompletionStream
        return OpenAIAsyncChatCompletionStream
