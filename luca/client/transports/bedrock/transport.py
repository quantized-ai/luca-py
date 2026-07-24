"""AWS Bedrock Converse API transport.

One wire schema across every model family on Bedrock (Anthropic, Amazon Nova,
Meta Llama, …). What is new here versus the other transports:

  - The model id lives in the URL path, not the body.
  - `system` is a top-level array of content blocks, not a message.
  - Tool arguments are a real JSON object, not a serialised string.
  - Roles are only `user` / `assistant`; a tool result is a `toolResult`
    block inside a user message, and adjacent same-role messages are merged
    because Converse requires strict alternation.
  - Auth is a plain bearer token (`AWS_BEARER_TOKEN_BEDROCK`); no SigV4.

Anthropic-on-Bedrock reasoning is written from the docs and is unverified —
see `capabilities.py`. Nova and Llama are verified live.
"""

from __future__ import annotations

from typing import Any, ClassVar

import httpx

from ...exceptions import (
    AuthenticationError,
    BadRequestError,
    ClientError,
    ContextLengthExceededError,
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
from .capabilities import check_sampling, get_model_capabilities, resolve_reasoning


class BedrockTransport(BaseTransport, ChatCompletionTransportMixin):
    transport_id = "bedrock"

    # Adaptive Anthropic models omit the reasoning text by default; ask for
    # summaries. Model facts live in `capabilities.py`; this is policy.
    THINKING_DISPLAY: ClassVar[str | None] = "summarized"

    # --- URL (auth is the base-class bearer header) ---

    def _chat_completion_url(
        self, request: ChatCompletionRequest, *, stream: bool = False,
    ) -> str:
        op = "converse-stream" if stream else "converse"
        return f"{self._base_url}/model/{request.model}/{op}"

    # --- payload building ---

    def _build_chat_completion_payload(
        self, request: ChatCompletionRequest, *, stream: bool = False,
    ) -> dict:
        capabilities = get_model_capabilities(request.model)
        options = self._provider_options(request)
        extra_fields, max_tokens = self._reasoning_config(request, capabilities, options)

        payload: dict[str, Any] = {
            "messages": self._project_messages(request.messages),
        }
        if request.system_message is not None:
            payload["system"] = self._project_system(request.system_message)

        check_sampling(
            capabilities, extra_fields,
            temperature=request.temperature, top_p=request.top_p,
            top_k=request.top_k, model=request.model,
        )
        if request.top_k is not None:
            # Converse has no portable top_k slot; where it lives differs per
            # model family. Refuse rather than drop it, and point at the escape
            # hatch that can place it exactly.
            raise UnsupportedParameterError(
                "top_k has no portable slot on Bedrock Converse; pass it via "
                "provider_options['bedrock']['additionalModelRequestFields'].",
            )
        inference = self._inference_config(request, extra_fields, max_tokens)
        if inference:
            payload["inferenceConfig"] = inference

        if request.tools:
            payload["toolConfig"] = self._project_tool_config(request)
        if extra_fields:
            payload["additionalModelRequestFields"] = extra_fields
        # Raw options win outright: a caller who spelled out fields gets them.
        payload.update(options)
        return payload

    def _provider_options(self, request: ChatCompletionRequest) -> dict:
        """This provider's raw options, or nothing. Scoped by provider name so
        one provider's options can never reach another's payload."""
        return (request.provider_options or {}).get(self._provider) or {}

    def _reasoning_config(
        self, request: ChatCompletionRequest, capabilities, options: dict,
    ) -> tuple[dict, int]:
        """The `additionalModelRequestFields` reasoning keys plus the max_tokens
        to send. Raw options carrying that field skip resolution entirely."""
        if "additionalModelRequestFields" in options:
            return {}, request.max_tokens or capabilities.max_output_tokens
        return resolve_reasoning(
            request.reasoning, capabilities, request.max_tokens,
            display=self.THINKING_DISPLAY, model=request.model,
        )

    def _inference_config(
        self, request: ChatCompletionRequest, extra_fields: dict, max_tokens: int,
    ) -> dict:
        inference: dict[str, Any] = {}
        # Converse defaults maxTokens per model, so only send one when the
        # caller asked or when thinking needs the headroom.
        if request.max_tokens is not None:
            inference["maxTokens"] = request.max_tokens
        elif extra_fields:
            inference["maxTokens"] = max_tokens
        if request.temperature is not None:
            inference["temperature"] = request.temperature
        if request.top_p is not None:
            inference["topP"] = request.top_p
        if request.stop is not None:
            inference["stopSequences"] = (
                [request.stop] if isinstance(request.stop, str) else list(request.stop)
            )
        return inference

    def _project_system(self, system_message: Any) -> list[dict]:
        if isinstance(system_message, str):
            return [{"text": system_message}]
        return [{"text": b.text} for b in system_message if isinstance(b, TextBlock)]

    def _project_messages(self, messages: list) -> list[dict]:
        out: list[dict] = []
        for msg in messages:
            if isinstance(msg, UserMessage):
                out.append({"role": "user", "content": self._project_user_content(msg)})
            elif isinstance(msg, AssistantMessage):
                out.append({
                    "role": "assistant",
                    "content": self._project_assistant_content(msg),
                })
            elif isinstance(msg, ToolMessage):
                # Converse has no tool role: a result is a user toolResult block.
                out.append({"role": "user", "content": [self._project_tool_result(msg)]})
            else:
                raise BadRequestError(
                    f"Unknown message type {type(msg).__name__}",
                    provider=self._provider,
                )
        return self._coalesce_roles(out)

    @staticmethod
    def _coalesce_roles(messages: list[dict]) -> list[dict]:
        """Merge adjacent same-role messages. Converse requires strict
        user/assistant alternation, and a turn with several tool results
        arrives as several user messages."""
        merged: list[dict] = []
        for msg in messages:
            if merged and merged[-1]["role"] == msg["role"]:
                merged[-1]["content"].extend(msg["content"])
            else:
                merged.append({"role": msg["role"], "content": list(msg["content"])})
        return merged

    def _project_user_content(self, msg: UserMessage) -> list[dict]:
        if isinstance(msg.content, str):
            return [{"text": msg.content}]
        blocks: list[dict] = []
        for block in msg.content:
            if isinstance(block, TextBlock):
                blocks.append({"text": block.text})
            elif isinstance(block, ImageBlock):
                blocks.append(self._project_image_block(block))
            else:
                blocks.append({"text": str(block)})
        return blocks

    def _project_image_block(self, block: ImageBlock) -> dict:
        source = block.source
        if isinstance(source, MediaBase64):
            return {
                "image": {
                    "format": self._image_format(source.media_type),
                    "source": {"bytes": source.data},
                },
            }
        # Converse takes image bytes or an s3Location, never a URL or file id.
        if isinstance(source, (MediaURL, MediaFileId)):
            raise BadRequestError(
                "Bedrock Converse needs inline image bytes; a URL or file id "
                "cannot be sent.",
                provider=self._provider,
            )
        raise BadRequestError("Unknown image source type", provider=self._provider)

    @staticmethod
    def _image_format(media_type: str | None) -> str:
        if not media_type:
            return "png"
        fmt = media_type.rsplit("/", 1)[-1].lower()
        return "jpeg" if fmt == "jpg" else fmt

    def _project_assistant_content(self, msg: AssistantMessage) -> list[dict]:
        blocks: list[dict] = []
        for block in msg.content:
            if isinstance(block, TextBlock):
                blocks.append({"text": block.text})
            elif isinstance(block, ThinkingBlock):
                reasoning = self._project_thinking_block(block)
                if reasoning is not None:
                    blocks.append(reasoning)
            elif isinstance(block, ToolCall):
                blocks.append({
                    "toolUse": {
                        "toolUseId": block.id,
                        "name": block.name,
                        "input": block.arguments,
                    },
                })
            elif isinstance(block, RefusalBlock):
                continue
        return blocks

    def _project_thinking_block(self, block: ThinkingBlock) -> dict | None:
        """One reasoning block on the way back, or None to omit it.

        An unsigned block is dropped: Bedrock rejects a reasoning block whose
        signature is missing, and an unsigned block is reachable (a truncated
        turn, or a session moved from a host that never signed it). Dropping
        it costs one turn's visible reasoning; sending it breaks the request."""
        if block.signature is None:
            return None
        if block.redacted:
            return {"reasoningContent": {"redactedContent": block.signature}}
        return {
            "reasoningContent": {
                "reasoningText": {"text": block.text, "signature": block.signature},
            },
        }

    def _project_tool_result(self, msg: ToolMessage) -> dict:
        if isinstance(msg.content, str):
            content: list[dict] = [{"text": msg.content}]
        else:
            content = []
            for b in msg.content:
                if isinstance(b, TextBlock):
                    content.append({"text": b.text})
                elif isinstance(b, ImageBlock):
                    content.append(self._project_image_block(b))
        result: dict[str, Any] = {
            "toolUseId": msg.tool_call_id,
            "content": content,
        }
        if msg.is_error:
            result["status"] = "error"
        return {"toolResult": result}

    def _project_tool_config(self, request: ChatCompletionRequest) -> dict:
        tools = []
        for t in request.tools:
            tools.append({
                "toolSpec": {
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": {"json": tool_parameters_to_json_schema(t.parameters)},
                },
            })
        config: dict[str, Any] = {"tools": tools}
        choice = self._project_tool_choice(request.tool_choice)
        if choice is not None:
            config["toolChoice"] = choice
        return config

    @staticmethod
    def _project_tool_choice(choice: Any) -> dict | None:
        # Converse has no "none"; omit toolChoice for it (and anything unknown).
        if isinstance(choice, str):
            return {"auto": {"auto": {}}, "required": {"any": {}}}.get(choice)
        if isinstance(choice, dict) and "name" in choice:
            return {"tool": {"name": choice["name"]}}
        return None

    # --- response parsing ---

    def _parse_chat_completion_response(
        self, response: httpx.Response, request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        data = response.json()
        message = self._parse_assistant_message(data, request)
        message.response_id = response.headers.get("x-amzn-requestid")
        message.usage = self._parse_usage(data.get("usage"), request.model_info)

        provider_terminal = data.get("stopReason")
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
        message = (data.get("output") or {}).get("message") or {}
        for block in message.get("content") or []:
            if "text" in block:
                content.append(TextBlock(text=block["text"]))
            elif "toolUse" in block:
                tool = block["toolUse"]
                content.append(ToolCall(
                    id=tool["toolUseId"],
                    name=tool["name"],
                    arguments=tool.get("input", {}) or {},
                    complete=True,
                ))
            elif "reasoningContent" in block:
                content.append(self._parse_reasoning_block(block["reasoningContent"]))
        return AssistantMessage(
            content=[c for c in content if c is not None],
            provider=self._provider,
            model=request.model,
        )

    @staticmethod
    def _parse_reasoning_block(reasoning: dict) -> ThinkingBlock | None:
        if "reasoningText" in reasoning:
            text = reasoning["reasoningText"]
            return ThinkingBlock(
                text=text.get("text", ""), signature=text.get("signature"),
            )
        if "redactedContent" in reasoning:
            return ThinkingBlock(
                text="", signature=reasoning["redactedContent"], redacted=True,
            )
        return None

    def _parse_usage(self, usage_json: dict | None, model_info: Any) -> Usage:
        if usage_json is None:
            return Usage()
        u = Usage(
            input_tokens=usage_json.get("inputTokens", 0),
            output_tokens=usage_json.get("outputTokens", 0),
            total_tokens=usage_json.get("totalTokens", 0),
            cached_input_tokens=usage_json.get("cacheReadInputTokens"),
            cache_write_tokens=usage_json.get("cacheWriteInputTokens"),
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
        if provider_value == "content_filtered":
            return ("error", "Bedrock content filter")
        if provider_value == "guardrail_intervened":
            return ("error", "Bedrock guardrail intervened")
        if provider_value is None:
            return (None, None)
        return (provider_value, None)

    # --- error mapping ---

    def _map_chat_completion_http_error(self, exc: httpx.HTTPError) -> ClientError:
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            body = self._safe_json(exc.response)
            msg = (body or {}).get("message", str(exc)) if isinstance(body, dict) else str(exc)
            # Bedrock signals the exception class in a header; the value has a
            # `:<url>` suffix, so match the leading token.
            err_type = exc.response.headers.get("x-amzn-errortype", "").split(":", 1)[0]

            if err_type == "AccessDeniedException" or status == 403:
                return AuthenticationError(
                    msg, provider=self._provider, original_exception=exc,
                )
            if err_type == "ThrottlingException" or status == 429:
                return RateLimitError(
                    msg, provider=self._provider, original_exception=exc,
                    retry_after=self._retry_after(exc.response),
                )
            if err_type in ("ResourceNotFoundException", "ModelNotReadyException") or status == 404:
                # A 404 for a model that exists means the account has not
                # submitted the Bedrock use-case form for it — an AWS console
                # step, not a code fix.
                return ModelNotFoundError(
                    msg, provider=self._provider, original_exception=exc,
                )
            if err_type == "ValidationException" or status == 400:
                if "too long" in msg.lower() or "context" in msg.lower():
                    return ContextLengthExceededError(
                        msg, provider=self._provider, original_exception=exc,
                    )
                return BadRequestError(
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
        from .stream import BedrockChatCompletionStream
        return BedrockChatCompletionStream

    def _async_chat_completion_stream_class(self) -> type:
        from .stream import BedrockAsyncChatCompletionStream
        return BedrockAsyncChatCompletionStream
