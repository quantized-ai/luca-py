"""Anthropic Messages API transport.

Differences from OpenAI worth highlighting:
  - `system` is a top-level field (not a message).
  - Content blocks are first-class on the wire (text / tool_use / tool_result / thinking).
  - `max_tokens` is REQUIRED.
  - Auth header is `x-api-key`, plus `anthropic-version`.
  - URL is `{base_url}/v1/messages`.
"""

from __future__ import annotations

from typing import Any, ClassVar

import httpx

from ...exceptions import (
    AuthenticationError,
    BadRequestError,
    ClientError,
    ConnectionError as ClientConnectionError,
    ContextLengthExceededError,
    InvalidModelError,
    ModelNotFoundError,
    ProviderAPIError,
    RateLimitError,
    TimeoutError as ClientTimeoutError,
    UnsupportedParameterError,
)
from ...types.completion import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Usage,
    UsageCost,
)
from ...types.content import (
    FileBlock,
    ImageBlock,
    RefusalBlock,
    TextBlock,
    ThinkingBlock,
    ToolCall,
)
from ...types.media import MediaBase64, MediaFileId, MediaURL
from ...types.messages import AssistantMessage, ToolMessage, UserMessage
from ...types.structured import (
    response_format_to_json_schema,
    strictify_json_schema,
    strip_unsupported_keywords,
)
from ...types.tools import BaseTool, Tool, ToolProjector, tool_parameters_to_json_schema
from ..base import BaseTransport, ChatCompletionTransportMixin
from .capabilities import check_sampling, get_model_capabilities, resolve_reasoning

_DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
_DEFAULT_MAX_TOKENS = 4096


def _project_image_block(block: ImageBlock) -> dict:
    """One image block to its wire shape — shared by user messages (via the
    transport) and tool-result content (via the projector)."""
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
    raise BadRequestError("Unknown image source type")


def _project_file_block(block: FileBlock, *, provider: str | None = None) -> dict:
    """One file block to Anthropic's `document` shape.

    All three sources are supported here, which is unusual — most APIs take
    fewer. `name` is not on the wire: Anthropic infers the media type from
    `media_type` and has no filename field on a document.
    https://platform.claude.com/docs/en/build-with-claude/pdf-support
    """
    source = block.source
    if isinstance(source, MediaURL):
        return {"type": "document", "source": {"type": "url", "url": source.url}}
    if isinstance(source, MediaBase64):
        return {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": source.media_type,
                "data": source.data,
            },
        }
    if isinstance(source, MediaFileId):
        return {"type": "document", "source": {"type": "file", "file_id": source.file_id}}
    raise BadRequestError(f"Unknown file source type {type(source).__name__}", provider=provider)


class AnthropicToolProjector(ToolProjector):
    """Standard function-tool behavior on the Anthropic wire — the default
    projector. Anthropic-NATIVE tools (bash, text editor) differ only in
    declaration: their calls are ordinary tool_use blocks and their results
    ordinary tool_result blocks, so they inherit everything else."""

    def project_tool_to_llm(self, tool: Tool) -> dict:
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool_parameters_to_json_schema(tool.parameters),
        }

    def build_tool_call(self, item: dict) -> ToolCall:
        # `item` is a tool_use content block.
        return ToolCall(
            id=item["id"],
            name=item["name"],
            arguments=item.get("input", {}) or {},
            complete=True,
        )

    def project_tool_call_to_llm(self, tool_call: ToolCall) -> dict:
        return {
            "type": "tool_use",
            "id": tool_call.id,
            "name": tool_call.name,
            "input": tool_call.arguments,
        }

    def project_tool_message_to_llm(
        self,
        message: ToolMessage,
        tool_call: ToolCall | None,
    ) -> dict:
        # Returns the tool_result BLOCK; the transport owns the
        # {"role": "user", "content": [block]} envelope around it.
        return {
            "type": "tool_result",
            "tool_use_id": message.tool_call_id,
            "content": self._result_content(message),
            "is_error": message.is_error,
        }

    def project_tool_choice_to_llm(self, tool: BaseTool) -> dict:
        # {"type": "tool", "name": ...} forces native tools too (they are
        # ordinary named tools) — the native projectors need no override.
        return {"type": "tool", "name": tool.name}

    def _result_content(self, message: ToolMessage) -> str | list[dict]:
        if isinstance(message.content, str):
            return message.content
        # Mixed text/image content — keep as block list.
        content: list[dict] = []
        for b in message.content:
            if isinstance(b, TextBlock):
                content.append({"type": "text", "text": b.text})
            elif isinstance(b, ImageBlock):
                content.append(_project_image_block(b))
        return content


class AnthropicTransport(BaseTransport, ChatCompletionTransportMixin):
    transport_id = "anthropic"

    TOOL_PROJECTOR_BASE: ClassVar[type] = AnthropicToolProjector

    def _default_tool_projector(self) -> ToolProjector:
        return AnthropicToolProjector()

    # Adaptive models omit the reasoning text by default and return only
    # the encrypted signature, which leaves nothing to render. Ask for
    # summaries. Model facts live in `capabilities.py`; this is policy.
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

    def _chat_completion_url(
        self,
        request: ChatCompletionRequest,
        *,
        stream: bool = False,
    ) -> str:
        return f"{self._base_url}/v1/messages"

    # --- payload building ---

    def _build_chat_completion_payload(
        self,
        request: ChatCompletionRequest,
        *,
        stream: bool = False,
    ) -> dict:
        capabilities = get_model_capabilities(request.model)
        options = self._provider_options(request)
        thinking, max_tokens = self._thinking_config(request, capabilities, options)
        payload: dict[str, Any] = {
            "model": request.model,
            "max_tokens": max_tokens,
            "messages": self._project_messages(request.messages, request),
        }
        payload.update(thinking)
        if request.system_message is not None:
            payload["system"] = self._project_system(request.system_message)
        if stream:
            payload["stream"] = True
        check_sampling(
            capabilities,
            thinking,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            model=request.model,
        )
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.top_k is not None:
            payload["top_k"] = request.top_k
        if request.stop is not None:
            payload["stop_sequences"] = [request.stop] if isinstance(request.stop, str) else list(request.stop)
        if request.tools:
            payload["tools"] = self._project_tools(request.tools)
        if request.tool_choice is not None:
            payload["tool_choice"] = self._project_tool_choice(request.tool_choice, request.tools)
        if request.metadata is not None:
            payload["metadata"] = request.metadata
        payload.update(options)
        # After the options merge, deliberately: `update` replaces the whole
        # `output_config` key, so applying the format first let a caller who
        # set `output_config` lose their `response_format` silently.
        if request.response_format is not None:
            self._apply_response_format(payload, request, capabilities)
        return payload

    def _apply_response_format(
        self,
        payload: dict,
        request: ChatCompletionRequest,
        capabilities,
    ) -> None:
        """Structured output lands on `output_config.format`.

        Merged into a COPY: `resolve_reasoning` already puts `effort` there,
        and after the options merge the dict may be the caller's own object.
        A hand-written `format` still wins.

        Refused only for models KNOWN to predate structured outputs — an
        unrecognized id goes through, since refusing it would block every
        model released after this table was written."""
        if capabilities.is_known_model and not capabilities.supports_structured_output:
            raise UnsupportedParameterError(
                f"response_format is not supported on {request.model!r}; "
                "the model predates Anthropic structured outputs.",
                provider=self._provider,
            )
        _, schema = response_format_to_json_schema(request.response_format)
        schema = strip_unsupported_keywords(strictify_json_schema(schema))
        output_config = dict(payload.get("output_config") or {})
        output_config.setdefault("format", {"type": "json_schema", "schema": schema})
        payload["output_config"] = output_config

    def _provider_options(self, request: ChatCompletionRequest) -> dict:
        """This provider's raw options, or nothing. Scoped by provider name so
        one provider's options can never reach another's payload."""
        return (request.provider_options or {}).get(self._provider) or {}

    def _thinking_config(
        self,
        request: ChatCompletionRequest,
        capabilities,
        options: dict,
    ) -> tuple[dict, int]:
        """The thinking-related payload keys plus the `max_tokens` to send.

        Raw provider options win outright rather than being merged into: a
        caller who spelled out `thinking` gets exactly that, and the resolved
        reasoning is skipped entirely."""
        if "thinking" in options or "output_config" in options:
            return {}, request.max_tokens or capabilities.max_output_tokens
        return resolve_reasoning(
            request.reasoning,
            capabilities,
            request.max_tokens,
            display=self.THINKING_DISPLAY,
            model=request.model,
        )

    def _project_system(self, system_message: Any) -> Any:
        if isinstance(system_message, str):
            return system_message
        return [{"type": "text", "text": b.text} for b in system_message if isinstance(b, TextBlock)]

    def _project_messages(self, messages: list, request: ChatCompletionRequest) -> list[dict]:
        """The walk records call lineage — `{call_id: (projector, call)}` —
        so a ToolMessage projects through the projector of the call it
        answers; a dropped foreign call maps to None and takes its result
        down with it.

        An assistant message that projects to ZERO wire blocks (foreign
        thinking + foreign native calls, the typical post-provider-switch
        turn) is omitted entirely — this wire rejects empty content."""
        out: list[dict] = []
        lineage: dict[str, tuple[ToolProjector, ToolCall] | None] = {}
        for msg in messages:
            if isinstance(msg, UserMessage):
                out.append(self._project_user_message(msg))
            elif isinstance(msg, AssistantMessage):
                projected = self._project_assistant_message(msg, request, lineage)
                if projected["content"]:
                    out.append(projected)
            elif isinstance(msg, ToolMessage):
                # Anthropic represents tool results as a user message with a
                # tool_result content block.
                wrapped = self._project_tool_message_as_user(msg, lineage)
                if wrapped is not None:
                    out.append(wrapped)
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
            elif isinstance(block, FileBlock):
                wire_blocks.append(self._project_file_block(block))
            else:
                raise BadRequestError(
                    f"The Anthropic Messages API has no shape for a {type(block).__name__}.",
                    provider=self._provider,
                )
        return {"role": "user", "content": wire_blocks}

    def _project_image_block(self, block: ImageBlock) -> dict:
        return _project_image_block(block)

    def _project_file_block(self, block: FileBlock) -> dict:
        return _project_file_block(block, provider=self._provider)

    def _project_assistant_message(
        self,
        msg: AssistantMessage,
        request: ChatCompletionRequest,
        lineage: dict[str, tuple[ToolProjector, ToolCall] | None] | None = None,
    ) -> dict:
        if lineage is None:
            lineage = {}
        wire_blocks: list[dict] = []
        for block in msg.content:
            if isinstance(block, TextBlock):
                wire_blocks.append({"type": "text", "text": block.text})
            elif isinstance(block, ThinkingBlock):
                thinking = self._project_thinking_block(block, msg, request)
                if thinking is not None:
                    wire_blocks.append(thinking)
            elif isinstance(block, ToolCall):
                entry = self._resolve_call(block)
                lineage[block.id] = entry
                if entry is not None:
                    projector, call = entry
                    wire_blocks.append(projector.project_tool_call_to_llm(call))
            elif isinstance(block, RefusalBlock):
                # Anthropic doesn't take refusals on the way in; drop.
                continue
        return {"role": "assistant", "content": wire_blocks}

    def _project_thinking_block(
        self,
        block: ThinkingBlock,
        msg: AssistantMessage,
        request: ChatCompletionRequest,
    ) -> dict | None:
        """One thinking block on the way back, or None to omit it.

        An unsigned block is DROPPED rather than sent. Anthropic rejects a
        thinking block whose signature is missing (400 `signature: Field
        required`) but accepts the turn with the block absent, and unsigned
        blocks are reachable: a truncated response never receives its
        `signature_delta`, and a session moved from an OpenAI-compatible host
        carries reasoning text that was never signed. Sending it would make
        the whole conversation permanently unusable; dropping it costs one
        turn's visible reasoning.

        A signature minted by a DIFFERENT (provider, model) pair is dropped for
        the same reason — Anthropic 400s on a foreign attestation just as it
        does on a missing one."""
        if block.signature is None:
            return None
        if not self._attestation_is_replayable(msg, request):
            return None
        if block.redacted:
            return {"type": "redacted_thinking", "data": block.signature}
        return {
            "type": "thinking",
            "thinking": block.text,
            "signature": block.signature,
        }

    def _project_tool_message_as_user(
        self,
        msg: ToolMessage,
        lineage: dict[str, tuple[ToolProjector, ToolCall] | None] | None = None,
    ) -> dict | None:
        msg = msg.as_native()
        if lineage is not None and msg.tool_call_id in lineage:
            entry = lineage[msg.tool_call_id]
            if entry is None:
                return None  # its call was dropped (foreign native)
            projector, call = entry
            block = projector.project_tool_message_to_llm(msg, call)
        else:
            block = self._default_tool_projector().project_tool_message_to_llm(msg, None)
        return {"role": "user", "content": [block]}

    def _project_tool_choice(self, choice: Any, tools: list | None) -> Any:
        # Strings are protocol-level and stay here; the {"name": ...} forcing
        # shape resolves through the named tool's projector.
        if isinstance(choice, str):
            return {
                "auto": {"type": "auto"},
                "required": {"type": "any"},
                "none": {"type": "none"},
            }.get(choice, {"type": "auto"})
        if isinstance(choice, dict) and "name" in choice:
            tool = self._declared_tool_named(choice["name"], tools)
            if tool is not None:
                return self._resolve_projector(tool).project_tool_choice_to_llm(tool)
            return {"type": "tool", "name": choice["name"]}
        return choice

    # --- response parsing ---

    def _parse_chat_completion_response(
        self,
        response: httpx.Response,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        data = response.json()
        message = self._parse_assistant_message(data, request)
        message.usage = self._parse_usage(data.get("usage"), request.model_info)

        provider_terminal = data.get("stop_reason")
        canonical, error_message = self._classify_finish(provider_terminal, message)
        message.finish_reason = canonical
        message.provider_finish_reason = provider_terminal
        message.error_message = error_message

        resp = ChatCompletionResponse(messages=[message], raw=data)
        resp._response_format = request.response_format
        return resp

    def _parse_assistant_message(
        self,
        data: dict,
        request: ChatCompletionRequest,
    ) -> AssistantMessage:
        content: list = []
        default = self._default_tool_projector()
        for block in data.get("content") or []:
            block_type = block.get("type")
            if block_type == "text":
                content.append(TextBlock(text=block.get("text", "")))
            elif block_type == "thinking":
                content.append(
                    ThinkingBlock(
                        text=block.get("thinking", ""),
                        signature=block.get("signature"),
                    )
                )
            elif block_type == "tool_use":
                # Native tools (bash, text editor) arrive through the SAME
                # branch — they are ordinary tool_use blocks with names.
                content.append(default.build_tool_call(block))
            elif block_type == "redacted_thinking":
                content.append(
                    ThinkingBlock(
                        text="",
                        signature=block.get("data"),
                        redacted=True,
                    )
                )
        return AssistantMessage(
            content=content,
            provider=self._provider,
            # `model` is what we ASKED for, `response_model` what Anthropic
            # resolved it to (an alias resolves to a dated id). Keeping the
            # requested string here is what makes provenance comparable: it
            # matches the streaming path, every other transport, and the
            # `request.model` that a later replay is checked against.
            model=request.model,
            response_model=data.get("model"),
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
        self,
        provider_value: str | None,
        message: AssistantMessage,
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
                    msg,
                    provider=self._provider,
                    original_exception=exc,
                )
            if status == 429:
                return RateLimitError(
                    msg,
                    provider=self._provider,
                    original_exception=exc,
                    retry_after=self._retry_after(exc.response),
                )
            if status == 400:
                if "context_length" in msg.lower() or "too long" in msg.lower():
                    return ContextLengthExceededError(
                        msg,
                        provider=self._provider,
                        original_exception=exc,
                    )
                if err_type == "invalid_request_error" and "model" in msg.lower():
                    return InvalidModelError(
                        msg,
                        provider=self._provider,
                        original_exception=exc,
                    )
                return BadRequestError(
                    msg,
                    provider=self._provider,
                    original_exception=exc,
                )
            if status == 404:
                return ModelNotFoundError(
                    msg,
                    provider=self._provider,
                    original_exception=exc,
                )
            if 500 <= status < 600:
                return ProviderAPIError(
                    msg,
                    provider=self._provider,
                    original_exception=exc,
                )
            return ProviderAPIError(
                msg,
                provider=self._provider,
                original_exception=exc,
            )

        if isinstance(exc, httpx.TimeoutException):
            return ClientTimeoutError(
                str(exc),
                provider=self._provider,
                original_exception=exc,
            )
        if isinstance(exc, httpx.NetworkError):
            return ClientConnectionError(
                str(exc),
                provider=self._provider,
                original_exception=exc,
            )
        return ProviderAPIError(
            str(exc),
            provider=self._provider,
            original_exception=exc,
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
