"""OpenAI Responses API transport (`POST /v1/responses`).

The wire protocol OpenAI's own models are developed against, and the only one
that lets reasoning survive a multi-step turn. Differences from Chat
Completions worth highlighting:

  - `input` is a flat list of ITEMS, not messages: one assistant turn becomes a
    `reasoning` item, a `message` item and one `function_call` item per call.
  - `output` is the same item list coming back; there is no `choices`.
  - There is no `finish_reason`. Terminal state is `status` plus
    `incomplete_details.reason`.
  - The system prompt is `instructions`, a top-level field.
  - `max_tokens` is `max_output_tokens`, and `stop` / `seed` / the penalties /
    `logprobs` do not exist at all.
  - Requests are stateless here (`store: false`), so reasoning replays by
    carrying `encrypted_content` in the input rather than by
    `previous_response_id`.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

import httpx

from ...exceptions import BadRequestError, UnsupportedParameterError
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
from ...types.structured import response_format_to_json_schema, strictify_json_schema
from ...types.tools import BaseTool, Tool, ToolProjector, tool_parameters_to_json_schema
from ..base import BaseTransport, ChatCompletionTransportMixin
from ..openai.errors import OpenAIErrorMappingMixin

# Chat-completions parameters the Responses API has no equivalent for. Refused
# rather than dropped: a silently ignored `seed` changes the output with
# nothing to notice.
_UNSUPPORTED_FIELDS: tuple[str, ...] = (
    "stop",
    "seed",
    "presence_penalty",
    "frequency_penalty",
    "logprobs",
    # Accepted by the endpoint but inert: logprobs need an `include` we do not
    # send and could not surface. Refusing matches how `logprobs` is treated.
    "top_logprobs",
)


class OpenAIResponsesToolProjector(ToolProjector):
    """Standard function-tool behavior on the Responses wire — the default
    projector every transport-resolved tool falls back to. Native projectors
    subclass this and override only what differs."""

    def project_tool_to_llm(self, tool: Tool) -> dict:
        # Flat, unlike chat completions — no `function` envelope.
        return {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool_parameters_to_json_schema(tool.parameters),
        }

    def build_tool_call(self, item: dict) -> ToolCall:
        return ToolCall(
            id=item["call_id"],
            name=item["name"],
            arguments=self._parse_arguments(item.get("arguments")),
            complete=True,
        )

    def project_tool_call_to_llm(self, tool_call: ToolCall) -> dict:
        return {
            "type": "function_call",
            "call_id": tool_call.id,
            "name": tool_call.name,
            "arguments": json.dumps(tool_call.arguments) if tool_call.arguments else "{}",
        }

    def project_tool_message_to_llm(
        self,
        message: ToolMessage,
        tool_call: ToolCall | None,
    ) -> dict:
        return {
            "type": "function_call_output",
            "call_id": message.tool_call_id,
            "output": self._output_text(message),
        }

    def project_tool_choice_to_llm(self, tool: BaseTool) -> dict:
        return {"type": "function", "name": tool.name}

    @staticmethod
    def _parse_arguments(raw: str | None) -> dict:
        # `ToolCall.arguments` is a dict, so valid-but-non-object JSON ("null",
        # "[1]") would escape as a raw pydantic ValidationError, not a
        # ClientError.
        try:
            parsed = json.loads(raw or "{}")
        except (json.JSONDecodeError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _output_text(self, message: ToolMessage) -> str:
        if isinstance(message.content, str):
            return message.content
        # Refusing beats dropping: the model would be told the tool call
        # succeeded and handed a result with the image missing.
        unsupported = {type(b).__name__ for b in message.content if not isinstance(b, TextBlock)}
        if unsupported:
            raise BadRequestError(
                "The responses API allows only text in a function call "
                f"output; cannot send {', '.join(sorted(unsupported))} "
                f"for call_id {message.tool_call_id!r}.",
            )
        return "".join(b.text for b in message.content)


class OpenAIResponsesTransport(BaseTransport, OpenAIErrorMappingMixin, ChatCompletionTransportMixin):
    transport_id = "openai-responses"

    TOOL_PROJECTOR_BASE: ClassVar[type] = OpenAIResponsesToolProjector

    def _default_tool_projector(self) -> ToolProjector:
        return OpenAIResponsesToolProjector()

    # --- URL ---

    def _chat_completion_url(
        self,
        request: ChatCompletionRequest,
        *,
        stream: bool = False,
    ) -> str:
        return f"{self._base_url}/responses"

    # --- payload building ---

    def _build_chat_completion_payload(
        self,
        request: ChatCompletionRequest,
        *,
        stream: bool = False,
    ) -> dict:
        self._check_unsupported(request)

        payload: dict[str, Any] = {
            "model": request.model,
            "input": self._project_messages(request.messages, request),
            # Stateless by construction: the full conversation goes up every
            # turn, exactly as on chat completions. Overridable through
            # provider_options for a caller who wants server-side state.
            "store": False,
        }
        if request.system_message is not None:
            payload["instructions"] = self._project_instructions(request.system_message)
        if stream:
            payload["stream"] = True

        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.max_tokens is not None:
            payload["max_output_tokens"] = request.max_tokens
        if request.parallel_tool_calls is not None:
            payload["parallel_tool_calls"] = request.parallel_tool_calls
        if request.user is not None:
            payload["user"] = request.user
        if request.metadata is not None:
            payload["metadata"] = request.metadata

        payload.update(self._reasoning_config(request))

        if request.tools:
            payload["tools"] = self._project_tools(request.tools)
        if request.tool_choice is not None:
            payload["tool_choice"] = self._project_tool_choice(request.tool_choice, request.tools)
        if request.response_format is not None:
            payload["text"] = {"format": self._project_response_format(request.response_format)}

        payload.update(self._provider_options(request))
        return payload

    def _check_unsupported(self, request: ChatCompletionRequest) -> None:
        offending = [name for name in _UNSUPPORTED_FIELDS if getattr(request, name) is not None]
        if offending:
            raise UnsupportedParameterError(
                f"{', '.join(offending)} cannot be sent to the OpenAI Responses "
                "API; it has no equivalent parameter. Use "
                "`transport_class=OpenAITransport` for the chat-completions endpoint.",
                provider=self._provider,
            )

    def _provider_options(self, request: ChatCompletionRequest) -> dict:
        """This provider's raw options, or nothing. Scoped by provider name so
        one provider's options can never reach another's payload."""
        return (request.provider_options or {}).get(self._provider) or {}

    def _reasoning_config(self, request: ChatCompletionRequest) -> dict:
        """The reasoning-related payload keys, or nothing.

        `context` is deliberately NOT sent: only the newest model family
        accepts `all_turns` and every earlier one rejects the whole request,
        while replay under `store: false` runs on the `encrypted_content` we
        put in `input` and does not depend on it either way."""
        if request.reasoning is None or request.reasoning == "provider-default":
            return {}
        return {
            "reasoning": {
                "effort": request.reasoning,
                "summary": "auto",
            },
            "include": ["reasoning.encrypted_content"],
        }

    def _project_instructions(self, system_message: Any) -> str:
        if isinstance(system_message, str):
            return system_message
        return "".join(b.text for b in system_message if isinstance(b, TextBlock))

    # --- input items ---

    def _project_messages(self, messages: list, request: ChatCompletionRequest) -> list[dict]:
        """Canonical messages → the flat input-item list. One assistant
        message expands to several items, emitted in content order so a
        `reasoning` item always precedes the `function_call` it produced.

        Takes the request, which the chat-completions projection does not:
        whether a reasoning item may be replayed depends on the model this
        call is going to.

        The walk records call lineage — `{call_id: (projector, call)}` from
        each assistant message — so a later ToolMessage projects through the
        projector of the call it answers (shell's result must echo fields
        from the call's action). A dropped foreign call maps to None and
        takes its result down with it."""
        out: list[dict] = []
        lineage: dict[str, tuple[ToolProjector, ToolCall] | None] = {}
        for msg in messages:
            if isinstance(msg, UserMessage):
                out.append(self._project_user_message(msg))
            elif isinstance(msg, AssistantMessage):
                out.extend(self._project_assistant_message(msg, request, lineage))
            elif isinstance(msg, ToolMessage):
                item = self._project_tool_message(msg, lineage)
                if item is not None:
                    out.append(item)
            else:
                raise BadRequestError(
                    f"Unknown message type {type(msg).__name__}",
                    provider=self._provider,
                )
        return out

    def _project_user_message(self, msg: UserMessage) -> dict:
        if isinstance(msg.content, str):
            return {"role": "user", "content": [{"type": "input_text", "text": msg.content}]}
        return {
            "role": "user",
            "content": [self._project_user_block(b) for b in msg.content],
        }

    def _project_user_block(self, block: Any) -> dict:
        if isinstance(block, TextBlock):
            return {"type": "input_text", "text": block.text}
        if isinstance(block, ImageBlock):
            return self._project_image_block(block)
        if isinstance(block, FileBlock):
            return self._project_file_block(block)
        raise BadRequestError(
            f"The Responses API has no shape for a {type(block).__name__}.",
            provider=self._provider,
        )

    def _project_file_block(self, block: FileBlock) -> dict:
        """A file on the Responses API: `input_file`, which unlike chat
        completions takes all three sources — an uploaded id, a URL, or inline
        base64 as a data URL. `filename` is required beside inline bytes.
        https://developers.openai.com/api/docs/guides/pdf-files
        """
        source = block.source
        if isinstance(source, MediaFileId):
            return {"type": "input_file", "file_id": source.file_id}
        if isinstance(source, MediaURL):
            return {"type": "input_file", "file_url": source.url}
        if isinstance(source, MediaBase64):
            return {
                "type": "input_file",
                "filename": block.name or "file.pdf",
                "file_data": f"data:{source.media_type};base64,{source.data}",
            }
        raise BadRequestError(
            f"Unknown media source type {type(source).__name__}",
            provider=self._provider,
        )

    def _project_image_block(self, block: ImageBlock) -> dict:
        source = block.source
        if isinstance(source, MediaURL):
            return {"type": "input_image", "image_url": source.url}
        if isinstance(source, MediaBase64):
            return {
                "type": "input_image",
                "image_url": f"data:{source.media_type};base64,{source.data}",
            }
        if isinstance(source, MediaFileId):
            # Unlike chat completions, the Responses API takes a file id here.
            return {"type": "input_image", "file_id": source.file_id}
        raise BadRequestError(
            f"Unknown media source type {type(source).__name__}",
            provider=self._provider,
        )

    def _project_assistant_message(
        self,
        msg: AssistantMessage,
        request: ChatCompletionRequest,
        lineage: dict[str, tuple[ToolProjector, ToolCall] | None] | None = None,
    ) -> list[dict]:
        if lineage is None:
            lineage = {}
        items: list[dict] = []
        for block in msg.content:
            if isinstance(block, ThinkingBlock):
                reasoning = self._project_reasoning_item(block, msg, request)
                if reasoning is not None:
                    items.append(reasoning)
            elif isinstance(block, TextBlock):
                items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": block.text}],
                    }
                )
            elif isinstance(block, ToolCall):
                entry = self._resolve_call(block)
                lineage[block.id] = entry
                if entry is not None:
                    projector, call = entry
                    items.append(projector.project_tool_call_to_llm(call))
            elif isinstance(block, RefusalBlock):
                # A refusal is an output-only shape; the API rejects it on input.
                continue
        return items

    def _project_reasoning_item(
        self,
        block: ThinkingBlock,
        msg: AssistantMessage,
        request: ChatCompletionRequest,
    ) -> dict | None:
        """One reasoning item on the way back, or None to omit it.

        Omitted unless the block carries BOTH halves of its provider identity
        — the `rs_…` id and the encrypted payload — and was minted by the pair
        we are calling (`_attestation_is_replayable`). Every other case is a
        400 that would make the whole conversation permanently unusable: an
        item with no `encrypted_content` under `store: false`, an id the
        account never issued, a summary from Anthropic. Dropping costs one
        turn's visible reasoning, which the model reconstructs."""
        if block.id is None or block.signature is None:
            return None
        if not self._attestation_is_replayable(msg, request):
            return None
        item: dict = {
            "type": "reasoning",
            "id": block.id,
            "encrypted_content": block.signature,
            "summary": [],
        }
        if block.text:
            item["summary"] = [{"type": "summary_text", "text": block.text}]
        return item

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
        # No lineage (hand-built history): standard shape, call unknown.
        return self._default_tool_projector().project_tool_message_to_llm(msg, None)

    def _project_tool_choice(self, choice: Any, tools: list | None) -> Any:
        # Strings and raw dicts are protocol-level and stay here; only the
        # {"name": ...} forcing shape is per-tool and goes through projectors.
        if isinstance(choice, str):
            return choice
        if isinstance(choice, dict):
            if set(choice.keys()) == {"name"}:
                tool = self._declared_tool_named(choice["name"], tools)
                if tool is not None:
                    return self._resolve_projector(tool).project_tool_choice_to_llm(tool)
                return {"type": "function", "name": choice["name"]}
            return choice
        return choice

    def _project_response_format(self, fmt: Any) -> dict:
        """`text.format` takes the schema flat — no nested `json_schema` key."""
        name, schema = response_format_to_json_schema(fmt)
        return {
            "type": "json_schema",
            "name": name,
            "schema": strictify_json_schema(schema),
            "strict": True,
        }

    # --- response parsing ---

    def _parse_chat_completion_response(
        self,
        response: httpx.Response,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        data = response.json()
        message = self._parse_assistant_message(data, request)
        message.usage = self._parse_usage(data.get("usage"), request.model_info)

        provider_terminal = self._provider_finish_reason(data)
        canonical, error_message = self._classify_finish(provider_terminal, message)
        message.finish_reason = canonical
        message.provider_finish_reason = provider_terminal
        # Provider text first: every status that carries an `error` object is
        # one where `_classify_finish` already returns a generic message, so
        # the other order can never surface OpenAI's own code and detail.
        message.error_message = self._response_error_message(data) or error_message

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
        for item in data.get("output") or []:
            item_type = item.get("type")
            if item_type == "reasoning":
                content.append(self._parse_reasoning_item(item))
            elif item_type == "message":
                content.extend(self._parse_message_item(item))
            elif item_type == "function_call":
                content.append(default.build_tool_call(item))
            elif (native := self._native_projector_for_item(item_type)) is not None:
                # The registry replaces a per-request tools scan: an item type
                # can only appear if its tool was declared, and the registry
                # is filled at import by the same module that defines it.
                content.append(native.build_tool_call(item))
            # Hosted-tool items (web_search_call, file_search_call, …) have no
            # canonical block; ignored rather than guessed at.
        return AssistantMessage(
            content=content,
            provider=self._provider,
            model=request.model,
            response_model=data.get("model"),
            response_id=data.get("id"),
        )

    def _parse_reasoning_item(self, item: dict) -> ThinkingBlock:
        """A reasoning item's renderable text is its SUMMARY — the raw chain of
        thought is never returned. Multiple summary parts are joined with a
        blank line, which is how they read as prose."""
        summary = "\n\n".join(
            part.get("text", "") for part in item.get("summary") or [] if part.get("type") == "summary_text"
        )
        return ThinkingBlock(
            text=summary,
            id=item.get("id"),
            signature=item.get("encrypted_content"),
        )

    def _parse_message_item(self, item: dict) -> list:
        blocks: list = []
        for part in item.get("content") or []:
            part_type = part.get("type")
            if part_type == "output_text":
                blocks.append(TextBlock(text=part.get("text", "")))
            elif part_type == "refusal":
                blocks.append(RefusalBlock(text=part.get("refusal", "")))
        return blocks

    def _parse_usage(self, usage_json: dict | None, model_info: Any) -> Usage:
        if usage_json is None:
            return Usage()
        input_details = usage_json.get("input_tokens_details") or {}
        output_details = usage_json.get("output_tokens_details") or {}
        u = Usage(
            input_tokens=usage_json.get("input_tokens", 0),
            output_tokens=usage_json.get("output_tokens", 0),
            total_tokens=usage_json.get("total_tokens", 0),
            cached_input_tokens=input_details.get("cached_tokens"),
            cache_write_tokens=input_details.get("cache_write_tokens"),
            reasoning_tokens=output_details.get("reasoning_tokens"),
        )
        if model_info is not None and getattr(model_info, "cost", None) is not None:
            u.cost = UsageCost.compute(u, model_info.cost)
        return u

    # --- finish-reason classification ---

    @staticmethod
    def _provider_finish_reason(data: dict) -> str | None:
        """The raw terminal state as ONE string, because that is what
        `provider_finish_reason` and `RawFinish` carry. `incomplete` alone says
        nothing, so it is qualified with the reason that explains it."""
        status = data.get("status")
        if status == "incomplete":
            reason = (data.get("incomplete_details") or {}).get("reason")
            return f"incomplete:{reason}" if reason else "incomplete"
        return status

    @staticmethod
    def _response_error_message(data: dict) -> str | None:
        error = data.get("error")
        if isinstance(error, dict):
            return error.get("message")
        return None

    def _classify_finish(
        self,
        provider_value: str | None,
        message: AssistantMessage,
    ) -> tuple[str | None, str | None]:
        if provider_value == "completed":
            if any(isinstance(b, ToolCall) for b in message.content):
                return ("tool_use", None)
            if any(isinstance(b, RefusalBlock) for b in message.content):
                refusal = next(b for b in message.content if isinstance(b, RefusalBlock))
                return ("error", f"OpenAI refusal: {refusal.text}")
            return ("stop", None)
        if provider_value == "incomplete:max_output_tokens":
            return ("length", None)
        if provider_value == "incomplete:content_filter":
            return ("error", "Provider safety filter (content_filter)")
        if provider_value == "failed":
            return ("error", "OpenAI reported the response as failed")
        if provider_value is None:
            return (None, None)
        if provider_value.startswith("incomplete"):
            return ("error", f"OpenAI returned an incomplete response ({provider_value})")
        # `queued` / `in_progress` / `cancelled` are real statuses (background
        # mode). None means "unknown" and keeps the canonical field inside its
        # documented set; the raw value stays on provider_finish_reason.
        return (None, None)

    # --- stream class hooks ---

    def _chat_completion_stream_class(self) -> type:
        from .stream import OpenAIResponsesStream

        return OpenAIResponsesStream

    def _async_chat_completion_stream_class(self) -> type:
        from .stream import OpenAIResponsesAsyncStream

        return OpenAIResponsesAsyncStream
