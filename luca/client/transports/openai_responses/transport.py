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
    PrivateProviderBlock,
    RefusalBlock,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    URLCitationAnnotation,
    WebFetchBlock,
    WebPagePart,
    WebSearchBlock,
)
from ...types.media import MediaBase64, MediaFileId, MediaURL
from ...types.messages import AssistantMessage, ToolMessage, UserMessage
from ...types.structured import response_format_to_json_schema, strictify_json_schema
from ...types.tools import BaseTool, Tool, ToolProjector, tool_parameters_to_json_schema
from ..base import BaseTransport, ChatCompletionTransportMixin, WireFormatMixin
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

    def project_request_params(self, tool: BaseTool) -> dict[str, Any]:
        """Request-level parameters this tool contributes beyond its `tools`
        entry — web search projects its inclusion flags into the top-level
        `include` list. Merged into the payload by
        `_build_chat_completion_payload`; nothing for standard tools."""
        return {}

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


class OpenAIResponsesWireMixin(OpenAIErrorMappingMixin, WireFormatMixin):
    """Responses wire knowledge: payload building, item projection, response
    parsing, finish classification. Shared by OpenAIResponsesTransport
    (non-streaming) and the Responses streamer, by inheritance.
    OpenAIErrorMappingMixin comes first so its `_map_chat_completion_http_error`
    beats WireFormatMixin's stub in the MRO."""

    TOOL_PROJECTOR_BASE: ClassVar[type] = OpenAIResponsesToolProjector

    # The wire format stamped on PrivateProviderBlocks minted here; only this
    # wire replays them.
    WIRE_FORMAT: ClassVar[str | None] = "openai.responses"

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
            # One projector resolution per tool covers both contributions:
            # its `tools` entry and any request-level parameters (web
            # search's `include` values merge with the reasoning include
            # already on the payload).
            declarations: list[dict] = []
            for tool in request.tools:
                projector = self._resolve_projector(tool)
                declarations.append(projector.project_tool_to_llm(tool))
                self._merge_tool_request_params(payload, projector.project_request_params(tool), tool)
            payload["tools"] = declarations
        if request.tool_choice is not None:
            payload["tool_choice"] = self._project_tool_choice(request.tool_choice, request.tools)
        if request.response_format is not None:
            payload["text"] = {"format": self._project_response_format(request.response_format)}

        payload.update(self._provider_options(request))
        return payload

    def _merge_tool_request_params(self, payload: dict, params: dict, tool: BaseTool) -> None:
        """Merge one tool's request-level parameters into the payload.

        Lists combine and deduplicate, identical values are accepted, and a
        conflicting value raises — a tool cannot silently replace core
        fields such as `model` or `input`."""
        for key, value in params.items():
            if key not in payload:
                payload[key] = value
            elif isinstance(payload[key], list) and isinstance(value, list):
                payload[key] = payload[key] + [v for v in value if v not in payload[key]]
            elif payload[key] != value:
                raise BadRequestError(
                    f"{type(tool).__name__} sets request parameter {key!r} to "
                    f"{value!r}, conflicting with {payload[key]!r} already on "
                    "the request.",
                    provider=self._provider,
                )

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
        """An `AudioBlock` raises here rather than falling through by accident:
        the Responses API has no audio input part, and OpenAI's own audio guide
        sends callers to chat completions for it."""
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

    def select_assistant_blocks(
        self,
        message: AssistantMessage,
        request: ChatCompletionRequest,
    ) -> list:
        """The keep/drop rules for this wire, in one place (payload build and
        `effective_messages` both route through here): text always survives;
        a reasoning block survives when both identity halves replay; a
        ToolCall survives when its projector family is this wire's;
        own-format privates replay verbatim — except the `in_progress` stub a
        cancelled stream left behind, an incomplete item this API 400s on —
        foreign privates drop; the portable web blocks are never sent; a
        refusal is an output-only shape."""
        kept: list = []
        for block in message.content:
            if isinstance(block, TextBlock):
                kept.append(block)
            elif isinstance(block, ThinkingBlock):
                if self._project_reasoning_item(block, message, request) is not None:
                    kept.append(block)
            elif isinstance(block, ToolCall):
                if self._resolve_call(block) is not None:
                    kept.append(block)
            elif (
                isinstance(block, PrivateProviderBlock)
                and block.format == self.WIRE_FORMAT
                and block.data.get("status") != "in_progress"
            ):
                kept.append(block)
        return kept

    def _project_assistant_message(
        self,
        msg: AssistantMessage,
        request: ChatCompletionRequest,
        lineage: dict[str, tuple[ToolProjector, ToolCall] | None] | None = None,
    ) -> list[dict]:
        if lineage is None:
            lineage = {}
        # Lineage records EVERY call's fate — dropped ones included, so the
        # ToolMessage answering a dropped foreign call falls with it.
        for block in msg.content:
            if isinstance(block, ToolCall):
                lineage[block.id] = self._resolve_call(block)
        items: list[dict] = []
        for block in self.select_assistant_blocks(msg, request):
            if isinstance(block, ThinkingBlock):
                items.append(self._project_reasoning_item(block, msg, request))
            elif isinstance(block, TextBlock):
                items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": block.text}],
                    }
                )
            elif isinstance(block, ToolCall):
                # Resolved per block, never through the id-keyed lineage dict:
                # duplicate ids would make the dict's last write clobber an
                # earlier kept call's entry.
                projector, call = self._resolve_call(block)
                items.append(projector.project_tool_call_to_llm(call))
            elif isinstance(block, PrivateProviderBlock):
                items.append(block.data)
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
        message.usage = self._parse_usage(data.get("usage"), request.model_info, data.get("tool_usage"))

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
            elif item_type == "web_search_call":
                # Two adjacent views of the one operation: the exact wire
                # item for replay, then its portable meaning.
                content.append(PrivateProviderBlock(format=self.WIRE_FORMAT, data=item))
                synthetic = self._web_block(item)
                if synthetic is not None:
                    content.append(synthetic)
            elif (native := self._native_projector_for_item(item_type)) is not None:
                # The registry replaces a per-request tools scan: an item type
                # can only appear if its tool was declared, and the registry
                # is filled at import by the same module that defines it.
                content.append(native.build_tool_call(item))
            # Other hosted-tool items (file_search_call, …) have no canonical
            # block; ignored rather than guessed at.
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
                blocks.append(
                    TextBlock(
                        text=part.get("text", ""),
                        annotations=self._parse_annotations(part.get("annotations")),
                    )
                )
            elif part_type == "refusal":
                blocks.append(RefusalBlock(text=part.get("refusal", "")))
        return blocks

    # --- web operations (hosted web_search tool) ---

    @staticmethod
    def _parse_annotation(raw: dict) -> URLCitationAnnotation | None:
        """One wire annotation, or None for kinds with no canonical shape."""
        if raw.get("type") != "url_citation":
            return None
        return URLCitationAnnotation(
            url=raw.get("url", ""),
            title=raw.get("title", ""),
            start_index=raw.get("start_index"),
            end_index=raw.get("end_index"),
        )

    def _parse_annotations(self, raw: list | None) -> list[URLCitationAnnotation]:
        return [annotation for entry in raw or [] if (annotation := self._parse_annotation(entry)) is not None]

    def _web_block(self, item: dict) -> WebSearchBlock | WebFetchBlock | None:
        """The portable projection of one web_search_call item, or None when
        its action has no canonical block (find_in_page, image search, an
        action OpenAI adds later) — the private block still preserves it.

        `extras` carries the adjacency-free facts: the operation id
        (`item["id"]`, the `ws_…` id) and, on a failed item,
        `{"status": "failed"}` — OpenAI reports no structured per-operation
        error code on the wire, so the status string is the whole signal."""
        action = item.get("action") or {}
        action_type = action.get("type")
        extras: dict = {}
        if (op_id := item.get("id")) is not None:
            extras["id"] = op_id
        if item.get("status") == "failed":
            extras["error"] = {"status": "failed"}
        if action_type == "search":
            return WebSearchBlock(
                queries=self._web_queries(action),
                results=self._web_search_results(item),
                extras=extras,
            )
        if action_type == "open_page":
            return WebFetchBlock(web_page=self._web_fetched_page(item), extras=extras)
        return None

    @staticmethod
    def _web_queries(action: dict) -> list[str]:
        # `queries` is authoritative; `query` is its legacy singular twin.
        queries = action.get("queries")
        if queries:
            return list(queries)
        query = action.get("query")
        return [query] if query else []

    @staticmethod
    def _web_search_results(item: dict) -> list[WebPagePart] | None:
        """`results` metadata and URL `sources` merged by URL, or None when
        neither was returned. Results come first (they carry title and
        snippet); a source URL no result covers is appended bare."""
        results = item.get("results")
        sources = [s for s in (item.get("action") or {}).get("sources") or [] if s.get("type") == "url"]
        if results is None and not sources:
            return None
        parts: list[WebPagePart] = []
        seen: set[str] = set()
        for result in results or []:
            url = result.get("url", "")
            parts.append(WebPagePart(url=url, title=result.get("title"), content=result.get("snippet")))
            seen.add(url)
        for source in sources:
            if source["url"] not in seen:
                parts.append(WebPagePart(url=source["url"]))
                seen.add(source["url"])
        return parts

    @staticmethod
    def _web_fetched_page(item: dict) -> WebPagePart:
        """The page an open_page action landed on: the result when returned
        (its URL wins — a redirect makes it differ from the action's), else
        just the URL the action opened."""
        action = item.get("action") or {}
        results = item.get("results") or []
        if results:
            result = results[0]
            return WebPagePart(
                url=result.get("url") or action.get("url", ""),
                title=result.get("title"),
                content=result.get("snippet"),
            )
        return WebPagePart(url=action.get("url", ""))

    def _parse_usage(
        self,
        usage_json: dict | None,
        model_info: Any,
        tool_usage_json: dict | None = None,
    ) -> Usage:
        """`usage` plus the sibling `tool_usage` envelope key, which carries
        the hosted-tool request counters."""
        if usage_json is None and tool_usage_json is None:
            return Usage()
        usage_json = usage_json or {}
        input_details = usage_json.get("input_tokens_details") or {}
        output_details = usage_json.get("output_tokens_details") or {}
        u = Usage(
            input_tokens=usage_json.get("input_tokens", 0),
            output_tokens=usage_json.get("output_tokens", 0),
            total_tokens=usage_json.get("total_tokens", 0),
            cached_input_tokens=input_details.get("cached_tokens"),
            cache_write_tokens=input_details.get("cache_write_tokens"),
            reasoning_tokens=output_details.get("reasoning_tokens"),
            tool_requests=self._tool_requests(tool_usage_json),
            provider_tool_usage=tool_usage_json or {},
        )
        if model_info is not None and getattr(model_info, "cost", None) is not None:
            u.cost = UsageCost.compute(u, model_info.cost)
        return u

    @staticmethod
    def _tool_requests(tool_usage_json: dict | None) -> dict[str, int]:
        """One `{tool: count}` entry per hosted tool whose usage reports
        `num_requests` (`web_search` today; `image_gen` reports tokens, not
        requests, and is left to `provider_tool_usage`)."""
        return {
            name: entry["num_requests"]
            for name, entry in (tool_usage_json or {}).items()
            if isinstance(entry, dict) and isinstance(entry.get("num_requests"), int)
        }

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


# Imported here, between the wire mixin and the transport class, because the
# streamer module inherits the mixin above: at the top of the file this would
# be a circular import.
from .streamer import (  # noqa: E402
    AsyncOpenAIResponsesStreamer,
    SyncOpenAIResponsesStreamer,
)


class OpenAIResponsesTransport(BaseTransport, ChatCompletionTransportMixin, OpenAIResponsesWireMixin):
    transport_id = "openai-responses"

    STREAMER = SyncOpenAIResponsesStreamer
    ASYNC_STREAMER = AsyncOpenAIResponsesStreamer
