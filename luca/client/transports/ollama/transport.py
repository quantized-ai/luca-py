"""Ollama's native `/api/chat`.

Not the OpenAI-compatible `/v1` endpoint, for one measured reason: `/v1`
silently ignores `options.num_ctx`. A model whose architecture allows 32k runs
at Ollama's default (4k on this machine) and truncates the conversation with
no error, no warning field, and no `truncated` flag. `/api/chat` honours
`num_ctx`, so it is the only wire on which luca can be sure what window it is
running in.

The window comes from `request.model_info.context_window`, which
`discovery.py` put in the catalog. luca therefore SETS the window it also
reports to the compactor — the two are the same number by construction and
cannot drift. This is the only transport that reads a `model_info` field other
than `cost`: for a local server the context window is a request parameter, not
metadata.

Subclasses `OpenAITransport` because the message shapes agree — same `role` /
`content` / `tool_calls` envelope, so message projection and assistant parsing
are inherited. What differs is the URL, the payload wrapper (`options`,
`think`, an explicit `stream`), the response envelope (a bare `message`, not
`choices[0]`), the usage keys, and NDJSON instead of SSE.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

import httpx

from ...exceptions import (
    ClientError,
    ConnectionError as ClientConnectionError,
    ModelNotFoundError,
    ProviderAPIError,
    TimeoutError as ClientTimeoutError,
)
from ...types.completion import ChatCompletionRequest, ChatCompletionResponse, Usage
from ...types.content import ToolCall
from ...types.messages import AssistantMessage, ToolMessage
from ...types.tools import ToolProjector
from ..openai.transport import OpenAIToolProjector, OpenAITransport
from .discovery import DEFAULT_NUM_CTX_CEILING


class OllamaToolProjector(OpenAIToolProjector):
    """Three deltas from the OpenAI wire: arguments are a real JSON object
    rather than a serialised string (as on Bedrock Converse), a call carries no
    `type` discriminator, and a result correlates by `tool_name` rather than
    `tool_call_id`."""

    def build_tool_call(self, item: dict) -> ToolCall:
        function = item.get("function") or {}
        return ToolCall(
            id=item.get("id") or function.get("name", ""),
            name=function.get("name", ""),
            arguments=function.get("arguments") or {},
            complete=True,
        )

    def project_tool_call_to_llm(self, tool_call: ToolCall) -> dict:
        return {
            "id": tool_call.id,
            "function": {"name": tool_call.name, "arguments": tool_call.arguments or {}},
        }

    def project_tool_message_to_llm(
        self,
        message: ToolMessage,
        tool_call: ToolCall | None,
    ) -> dict:
        wire = super().project_tool_message_to_llm(message, tool_call)
        # Verified against a live daemon: the follow-up turn correlates on
        # `tool_name`. `tool_call_id` alone leaves the result unattributed.
        name = message.name or (tool_call.name if tool_call is not None else None)
        if name:
            wire["tool_name"] = name
        return wire


class OllamaTransport(OpenAITransport):
    transport_id = "ollama"

    TOOL_PROJECTOR_BASE: ClassVar[type] = OpenAIToolProjector

    NUM_CTX_CEILING: ClassVar[int] = DEFAULT_NUM_CTX_CEILING
    """How large a window to ask for when the model would allow more. Policy,
    not a model fact — `provider_options` overrides it per call."""

    def _default_tool_projector(self) -> ToolProjector:
        return OllamaToolProjector()

    # --- URL ---

    def _chat_completion_url(
        self,
        request: ChatCompletionRequest,
        *,
        stream: bool = False,
    ) -> str:
        # One path for both; the `stream` field in the body is what differs.
        return f"{self._base_url}/api/chat"

    # --- payload ---

    def _num_ctx(self, request: ChatCompletionRequest, options: dict) -> int | None:
        """The window to ask for. An explicit `provider_options` value wins;
        otherwise the catalog's, capped."""
        override = (options.get("options") or {}).get("num_ctx")
        if override:
            return int(override)
        window = getattr(request.model_info, "context_window", None)
        return min(int(window), self.NUM_CTX_CEILING) if window else None

    def _build_chat_completion_payload(
        self,
        request: ChatCompletionRequest,
        *,
        stream: bool = False,
    ) -> dict:
        messages = self._project_messages(request.messages)
        if request.system_message is not None:
            messages = [self._project_system_message(request.system_message), *messages]

        options: dict[str, Any] = {}
        num_ctx = self._num_ctx(request, self._provider_options(request))
        if num_ctx is not None:
            options["num_ctx"] = num_ctx
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.top_p is not None:
            options["top_p"] = request.top_p
        if request.top_k is not None:
            options["top_k"] = request.top_k
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        if request.stop is not None:
            options["stop"] = request.stop
        if request.seed is not None:
            options["seed"] = request.seed

        # Explicit: Ollama streams by DEFAULT, so a non-streaming call that
        # omits this gets NDJSON back and the parse fails.
        payload: dict[str, Any] = {"model": request.model, "messages": messages, "stream": stream}
        if options:
            payload["options"] = options
        if request.tools:
            payload["tools"] = self._project_tools(request.tools)
        # Only for a model that advertises it; asking otherwise is a 400.
        # UNVERIFIED: no locally-pulled model on the development machine
        # advertised `thinking`, so this path is written from the docs and has
        # never round-tripped live. The capability flag itself is verified.
        if (
            request.reasoning is not None
            and request.reasoning != "provider-default"
            and getattr(request.model_info, "supports_reasoning", False)
        ):
            payload["think"] = True

        extra = dict(self._provider_options(request))
        extra_options = extra.pop("options", None)
        if extra_options:
            payload.setdefault("options", {}).update(extra_options)
        payload.update(extra)
        return payload

    # --- parsing ---

    def _parse_chat_completion_response(
        self,
        response: httpx.Response,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        data = response.json()
        message_json = dict(data.get("message") or {})
        # `thinking` is Ollama's name for what the inherited parser calls
        # `reasoning`; renaming here reuses that parser wholesale.
        if message_json.get("thinking"):
            message_json["reasoning"] = message_json["thinking"]

        message = self._parse_assistant_message(message_json, request, data)
        message.usage = self._parse_usage(data, request.model_info)

        provider_terminal = data.get("done_reason")
        canonical, error_message = self._classify_finish(provider_terminal, message)
        message.finish_reason = canonical
        message.provider_finish_reason = provider_terminal
        message.error_message = error_message

        resp = ChatCompletionResponse(message=message, raw=data)
        resp._response_format = request.response_format
        return resp

    def _parse_usage(self, usage_json: dict | None, model_info: Any) -> Usage:
        """The WHOLE response, not a `usage` object — Ollama has none, and puts
        the counts at the top level. Only this class's own
        `_parse_chat_completion_response` calls it, so the shape is private
        despite the inherited name."""
        if not usage_json:
            return Usage()
        input_tokens = usage_json.get("prompt_eval_count") or 0
        output_tokens = usage_json.get("eval_count") or 0
        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )

    def _classify_finish(
        self,
        provider_value: str | None,
        message: AssistantMessage,
    ) -> tuple[str | None, str | None]:
        # Ollama says "stop" even when it emitted tool calls, so the canonical
        # reason has to come from the content.
        if provider_value == "stop" and any(isinstance(b, ToolCall) for b in message.content):
            return ("tool_use", None)
        return super()._classify_finish(provider_value, message)

    # --- errors ---

    def _map_chat_completion_http_error(self, exc: httpx.HTTPError) -> ClientError:
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            detail = self._error_detail(exc.response)
            if status == 404:
                return ModelNotFoundError(
                    f"Ollama has no model {detail or 'by that name'}. Pull it first: `ollama pull <model>`.",
                    provider=self._provider,
                    original_exception=exc,
                )
            return ProviderAPIError(
                f"Ollama returned {status}{f': {detail}' if detail else ''}",
                provider=self._provider,
                original_exception=exc,
            )
        if isinstance(exc, httpx.TimeoutException):
            return ClientTimeoutError(str(exc), provider=self._provider, original_exception=exc)
        if isinstance(exc, httpx.NetworkError):
            return ClientConnectionError(
                f"Cannot reach Ollama at {self._base_url} ({exc}). "
                "Is the daemon running? Start it with `ollama serve`.",
                provider=self._provider,
                original_exception=exc,
            )
        return ProviderAPIError(str(exc), provider=self._provider, original_exception=exc)

    @staticmethod
    def _error_detail(response: httpx.Response) -> str | None:
        try:
            return (response.json() or {}).get("error")
        except (json.JSONDecodeError, ValueError):
            return None

    # --- stream class hooks ---

    def _chat_completion_stream_class(self) -> type:
        from .stream import OllamaChatCompletionStream

        return OllamaChatCompletionStream

    def _async_chat_completion_stream_class(self) -> type:
        from .stream import OllamaAsyncChatCompletionStream

        return OllamaAsyncChatCompletionStream
