"""OpenAI Chat Completions transport.

Speaks the OpenAI Chat Completions wire format. Used by OpenAIProvider, and
(via PROVIDERS dict entries -> GenericProvider) by every OpenAI-compatible
host (Groq, DeepSeek, Together, Ollama, Cerebras, Fireworks, ...). The host's
identity is on self._provider.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

import httpx

from ...exceptions import BadRequestError
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
from ...types.structured import response_format_to_json_schema, strictify_json_schema
from ...types.tools import BaseTool, Tool, ToolProjector, tool_parameters_to_json_schema
from ..base import BaseTransport, ChatCompletionTransportMixin
from .errors import OpenAIErrorMappingMixin


class OpenAIToolProjector(ToolProjector):
    """Standard function-tool behavior on the chat-completions wire — the
    default projector, inherited by every OpenAI-compatible host (OpenRouter
    subclasses the transport and accepts the same projectors). No native
    tools ship on this wire; the projector exists so foreign native calls
    can be detected and dropped, and so third parties can plug in."""

    def project_tool_to_llm(self, tool: Tool) -> dict:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool_parameters_to_json_schema(tool.parameters),
            },
        }

    def build_tool_call(self, item: dict) -> ToolCall:
        # `item` is one entry of the message's `tool_calls` array.
        try:
            args = json.loads(item["function"]["arguments"] or "{}")
        except (json.JSONDecodeError, KeyError, TypeError):
            args = {}
        return ToolCall(
            id=item["id"],
            name=item["function"]["name"],
            arguments=args,
            complete=True,
        )

    def project_tool_call_to_llm(self, tool_call: ToolCall) -> dict:
        # One entry for the assistant message's `tool_calls` array; the
        # transport owns the array and the message envelope.
        return {
            "id": tool_call.id,
            "type": "function",
            "function": {
                "name": tool_call.name,
                "arguments": json.dumps(tool_call.arguments) if tool_call.arguments else "{}",
            },
        }

    def project_tool_message_to_llm(
        self,
        message: ToolMessage,
        tool_call: ToolCall | None,
    ) -> dict:
        # A whole wire message — results ARE top-level on this wire.
        if isinstance(message.content, str):
            content = message.content
        else:
            # Refusing beats dropping: the model would be told the tool call
            # succeeded and handed a result with the image missing.
            unsupported = {type(b).__name__ for b in message.content if not isinstance(b, TextBlock)}
            if unsupported:
                raise BadRequestError(
                    "The chat-completions API allows only text in a tool "
                    f"result; cannot send {', '.join(sorted(unsupported))} "
                    f"for tool_call_id {message.tool_call_id!r}.",
                )
            content = "".join(b.text for b in message.content)
        wire: dict = {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": content,
        }
        if message.name is not None:
            wire["name"] = message.name
        return wire

    def project_tool_choice_to_llm(self, tool: BaseTool) -> dict:
        return {"type": "function", "function": {"name": tool.name}}


class OpenAITransport(BaseTransport, OpenAIErrorMappingMixin, ChatCompletionTransportMixin):
    transport_id = "openai"

    TOOL_PROJECTOR_BASE: ClassVar[type] = OpenAIToolProjector

    def _default_tool_projector(self) -> ToolProjector:
        return OpenAIToolProjector()

    # --- payload building ---

    def _build_chat_completion_payload(
        self,
        request: ChatCompletionRequest,
        *,
        stream: bool = False,
    ) -> dict:
        wire_messages = self._project_messages(request.messages)
        if request.system_message is not None:
            wire_messages = [self._project_system_message(request.system_message), *wire_messages]

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
        # `reasoning_effort` is OpenAI's wire key; `reasoning` is ours.
        # "provider-default" means send nothing.
        if request.reasoning is not None and request.reasoning != "provider-default":
            payload["reasoning_effort"] = request.reasoning

        if request.tools:
            payload["tools"] = self._project_tools(request.tools)
        if request.tool_choice is not None:
            payload["tool_choice"] = self._project_tool_choice(request.tool_choice, request.tools)
        if request.response_format is not None:
            payload["response_format"] = self._project_response_format(request.response_format)

        payload.update(self._provider_options(request))
        return payload

    def _provider_options(self, request: ChatCompletionRequest) -> dict:
        """This provider's raw options, or nothing. Scoped by provider name so
        one provider's options can never reach another's payload."""
        return (request.provider_options or {}).get(self._provider) or {}

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
        """The walk records call lineage — `{call_id: (projector, call)}` —
        so a ToolMessage projects through the projector of the call it
        answers; a dropped foreign call maps to None and takes its result
        down with it. An assistant message left with nothing (no content, no
        tool_calls, no refusal — e.g. a foreign native-only turn) is omitted:
        this wire rejects an empty assistant message."""
        out: list[dict] = []
        lineage: dict[str, tuple[ToolProjector, ToolCall] | None] = {}
        for msg in messages:
            if isinstance(msg, UserMessage):
                out.append(self._project_user_message(msg))
            elif isinstance(msg, AssistantMessage):
                projected = self._project_assistant_message(msg, lineage)
                if projected["content"] is not None or "tool_calls" in projected or "refusal" in projected:
                    out.append(projected)
            elif isinstance(msg, ToolMessage):
                wire = self._project_tool_message(msg, lineage)
                if wire is not None:
                    out.append(wire)
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
            return {"url": source.url}
        if isinstance(source, MediaBase64):
            return {"url": f"data:{source.media_type};base64,{source.data}"}
        if isinstance(source, MediaFileId):
            raise BadRequestError(
                "The chat-completions API cannot take an image by file id "
                f"({source.file_id!r}); pass MediaURL or MediaBase64 instead.",
                provider=self._provider,
            )
        raise BadRequestError(
            f"Unknown media source type {type(source).__name__}",
            provider=self._provider,
        )

    def _project_assistant_message(
        self,
        msg: AssistantMessage,
        lineage: dict[str, tuple[ToolProjector, ToolCall] | None] | None = None,
    ) -> dict:
        if lineage is None:
            lineage = {}
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
                entry = self._resolve_call(block)
                lineage[block.id] = entry
                if entry is not None:
                    projector, call = entry
                    tool_calls_wire.append(projector.project_tool_call_to_llm(call))
        if text_parts:
            wire["content"] = "".join(text_parts)
        else:
            wire["content"] = None
        if tool_calls_wire:
            wire["tool_calls"] = tool_calls_wire
        return wire

    def _project_tool_message(
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
            return projector.project_tool_message_to_llm(msg, call)
        return self._default_tool_projector().project_tool_message_to_llm(msg, None)

    def _project_tool_choice(self, choice: Any, tools: list | None) -> Any:
        # Strings and raw dicts are protocol-level and stay here; the
        # {"name": ...} forcing shape resolves through the named tool's
        # projector.
        if isinstance(choice, str):
            return choice
        if isinstance(choice, dict):
            if set(choice.keys()) == {"name"}:
                tool = self._declared_tool_named(choice["name"], tools)
                if tool is not None:
                    return self._resolve_projector(tool).project_tool_choice_to_llm(tool)
                return {"type": "function", "function": {"name": choice["name"]}}
            return choice
        return choice

    def _project_response_format(self, fmt: Any) -> Any:
        """Chat Completions nests the schema one level down, under
        `json_schema`. Strictified first — `strict: true` alongside a raw
        `model_json_schema()` is a 400 on every current model."""
        name, schema = response_format_to_json_schema(fmt)
        return {
            "type": "json_schema",
            "json_schema": {
                "name": name,
                "schema": strictify_json_schema(schema),
                "strict": True,
            },
        }

    # --- response parsing ---

    def _parse_chat_completion_response(
        self,
        response: httpx.Response,
        request: ChatCompletionRequest,
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

        resp = ChatCompletionResponse(messages=[message], raw=data)
        resp._response_format = request.response_format
        return resp

    def _parse_assistant_message(
        self,
        msg_json: dict,
        request: ChatCompletionRequest,
        full_data: dict,
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
        default = self._default_tool_projector()
        content.extend(default.build_tool_call(tc) for tc in msg_json.get("tool_calls") or [])
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
        self,
        provider_value: str | None,
        message: AssistantMessage,
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

    # --- stream class hooks ---

    def _chat_completion_stream_class(self) -> type:
        from .stream import OpenAIChatCompletionStream

        return OpenAIChatCompletionStream

    def _async_chat_completion_stream_class(self) -> type:
        from .stream import OpenAIAsyncChatCompletionStream

        return OpenAIAsyncChatCompletionStream
