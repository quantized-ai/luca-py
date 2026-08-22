"""PoC: one streamer design, two providers (OpenAI Responses + Anthropic Messages).

Yields the real events from luca.client.types.streaming.

Layering:
- BaseStreamer            shared state, SSE line parsing, handler dispatch
- AsyncStreamerMixin      async iteration + call_later deadline (cancels the read task)
- SyncStreamerMixin       sync iteration + monotonic deadline (cooperative check per read)
- OpenAIResponsesStreamer         wire knowledge for /v1/responses
- OpenAIChatCompletionsStreamer   wire knowledge for /v1/chat/completions — chunks
                                  carry no event type, so it overrides handle()
                                  and synthesizes block start/end events
- OpenRouterChatCompletionsStreamer  same wire, different URL/provider label
- AnthropicStreamer               wire knowledge for /v1/messages

The concrete classes are just mixin x wire combinations.

Run:  uv run python poc_streamer.py --provider openai
      uv run python poc_streamer.py --provider openai --transport chat_completions
      uv run python poc_streamer.py --provider openrouter --sync
      uv run python poc_streamer.py --provider anthropic --timeout 5   # 5s deadline
"""

import argparse
import asyncio
import json
import os
import time

import dotenv

dotenv.load_dotenv()

import httpx

from luca.client.exceptions import TimeoutError
from luca.client.types import (
    AssistantMessage,
    ErrorEvent,
    FinishEvent,
    StartEvent,
    TextBlock,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    Usage,
    UsageEvent,
)


class BaseStreamer:
    """Shared state, SSE line parsing, and handler dispatch."""

    PROVIDER = None
    URL = None
    HANDLERS_BY_TYPE = {}

    def __init__(self, messages, model, tools=None, api_key=None, httpx_client=None, timeout=None):
        self.messages = messages
        self.model = model
        self.tools = tools
        self.api_key = api_key
        self.timeout = timeout
        self.client = httpx_client
        self.owns_client = httpx_client is None

        self.message = AssistantMessage(content=[], provider=self.PROVIDER, model=model)
        self.response = None
        self.lines = None
        self.finished = False
        self.pending = []  # one wire line can produce several events

    def parse(self, line):
        if not line.startswith("data: ") or line[6:].strip() == "[DONE]":
            return None
        return json.loads(line[6:])

    def handle(self, raw_event):
        handler = self.HANDLERS_BY_TYPE.get(raw_event["type"])
        if handler is None:
            return []
        return getattr(self, handler)(raw_event)

    def timeout_event(self):
        self.finished = True
        error = TimeoutError(f"stream exceeded timeout={self.timeout}s", provider=self.PROVIDER)
        return ErrorEvent(error=error)


class AsyncStreamerMixin:
    """Async iteration; the deadline is a call_later timer that cancels the in-flight read task."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timer = None
        self.last_task = None
        self.expired = False

    async def __aenter__(self):
        if self.client is None:
            self.client = httpx.AsyncClient()

        request = self.build_request()

        if self.timeout is not None:
            self.timer = asyncio.get_running_loop().call_later(self.timeout, self.expire)

        self.last_task = asyncio.create_task(self.client.send(request, stream=True))
        try:
            self.response = await self.last_task
        except asyncio.CancelledError:
            if self.expired:
                raise TimeoutError(
                    f"request did not open within timeout={self.timeout}s", provider=self.PROVIDER
                ) from None
            raise

        if self.response.status_code != 200:
            body = await self.response.aread()
            await self.response.aclose()
            raise RuntimeError(f"HTTP {self.response.status_code}: {body.decode()}")

        self.lines = self.response.aiter_lines()
        return self

    async def __aexit__(self, *exc):
        if self.timer is not None:
            self.timer.cancel()
        if self.response is not None:
            await self.response.aclose()
        if self.owns_client:
            await self.client.aclose()

    def expire(self):
        self.expired = True
        if self.last_task is not None and not self.last_task.done():
            self.last_task.cancel()

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.finished:
            raise StopAsyncIteration

        if self.pending:
            event = self.pending.pop(0)
            if isinstance(event, (FinishEvent, ErrorEvent)):
                self.finished = True
            return event

        while True:
            if self.expired:
                return self.timeout_event()

            self.last_task = asyncio.create_task(anext(self.lines))
            try:
                line = await self.last_task
            except asyncio.CancelledError:
                if self.expired:
                    return self.timeout_event()
                raise
            except StopAsyncIteration:
                self.finished = True
                raise

            raw_event = self.parse(line)
            if raw_event is None:
                continue
            events = self.handle(raw_event)
            if not events:
                continue

            event, *rest = events
            self.pending += rest
            if isinstance(event, (FinishEvent, ErrorEvent)):
                self.finished = True
            return event


class SyncStreamerMixin:
    """Sync iteration; the deadline is a monotonic clock checked before every read (cooperative)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deadline = None

    def __enter__(self):
        if self.client is None:
            self.client = httpx.Client()

        request = self.build_request()

        if self.timeout is not None:
            self.deadline = time.monotonic() + self.timeout

        self.response = self.client.send(request, stream=True)

        if self.expired():
            self.response.close()
            raise TimeoutError(f"request did not open within timeout={self.timeout}s", provider=self.PROVIDER)

        if self.response.status_code != 200:
            body = self.response.read()
            self.response.close()
            raise RuntimeError(f"HTTP {self.response.status_code}: {body.decode()}")

        self.lines = self.response.iter_lines()
        return self

    def __exit__(self, *exc):
        if self.response is not None:
            self.response.close()
        if self.owns_client:
            self.client.close()

    def expired(self):
        return self.deadline is not None and time.monotonic() > self.deadline

    def __iter__(self):
        return self

    def __next__(self):
        if self.finished:
            raise StopIteration

        if self.pending:
            event = self.pending.pop(0)
            if isinstance(event, (FinishEvent, ErrorEvent)):
                self.finished = True
            return event

        while True:
            if self.expired():
                return self.timeout_event()

            try:
                line = next(self.lines)
            except StopIteration:
                self.finished = True
                raise

            raw_event = self.parse(line)
            if raw_event is None:
                continue
            events = self.handle(raw_event)
            if not events:
                continue

            event, *rest = events
            self.pending += rest
            if isinstance(event, (FinishEvent, ErrorEvent)):
                self.finished = True
            return event


class OpenAIResponsesStreamer(BaseStreamer):
    """Wire knowledge for OpenAI /v1/responses."""

    PROVIDER = "openai"
    URL = "https://api.openai.com/v1/responses"

    HANDLERS_BY_TYPE = {
        "response.created": "handle_response_created",
        "response.content_part.added": "handle_response_content_part_added",
        "response.output_text.delta": "handle_response_output_text_delta",
        "response.content_part.done": "handle_response_content_part_done",
        "response.output_item.added": "handle_response_output_item_added",
        "response.function_call_arguments.delta": "handle_response_function_call_arguments_delta",
        "response.output_item.done": "handle_response_output_item_done",
        "response.completed": "handle_response_completed",
    }

    def build_request(self):
        payload = {"model": self.model, "input": self.messages, "stream": True}
        if self.tools:
            payload["tools"] = self.tools
        return self.client.build_request(
            "POST",
            self.URL,
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    def handle_response_created(self, raw_event):
        return [StartEvent()]

    def handle_response_content_part_added(self, raw_event):
        if raw_event["part"]["type"] != "output_text":
            return []
        self.message.content.append(TextBlock(text=""))
        return [TextStartEvent(index=len(self.message.content) - 1)]

    def handle_response_output_text_delta(self, raw_event):
        index = len(self.message.content) - 1
        self.message.content[index].text += raw_event["delta"]
        return [TextDeltaEvent(index=index, delta=raw_event["delta"])]

    def handle_response_content_part_done(self, raw_event):
        if raw_event["part"]["type"] != "output_text":
            return []
        index = len(self.message.content) - 1
        return [TextEndEvent(index=index, content=self.message.content[index].text)]

    def handle_response_output_item_added(self, raw_event):
        if raw_event["item"]["type"] != "function_call":
            return []
        item = raw_event["item"]
        call = ToolCall(id=item["call_id"], name=item["name"], arguments={}, partial_arguments="", complete=False)
        self.message.content.append(call)
        return [ToolCallStartEvent(index=len(self.message.content) - 1, id=call.id, name=call.name)]

    def handle_response_function_call_arguments_delta(self, raw_event):
        index = len(self.message.content) - 1
        self.message.content[index].partial_arguments += raw_event["delta"]
        return [ToolCallDeltaEvent(index=index, arguments_delta=raw_event["delta"])]

    def handle_response_output_item_done(self, raw_event):
        if raw_event["item"]["type"] != "function_call":
            return []
        index = len(self.message.content) - 1
        call = self.message.content[index]
        call.arguments = json.loads(raw_event["item"]["arguments"])
        call.partial_arguments = ""
        call.complete = True
        return [ToolCallEndEvent(index=index, tool_call=call)]

    def handle_response_completed(self, raw_event):
        raw_usage = raw_event["response"].get("usage") or {}
        usage = Usage(
            input_tokens=raw_usage.get("input_tokens", 0),
            output_tokens=raw_usage.get("output_tokens", 0),
            total_tokens=raw_usage.get("total_tokens", 0),
        )
        return [
            UsageEvent(usage=usage),
            FinishEvent(
                message=self.message.model_copy(deep=True),
                finish_reason="tool_use" if self.message.tool_calls else "stop",
                provider_finish_reason=raw_event["response"]["status"],
                usage=usage,
                tool_calls=self.message.tool_calls,
            ),
        ]


class AnthropicStreamer(BaseStreamer):
    """Wire knowledge for Anthropic /v1/messages."""

    PROVIDER = "anthropic"
    URL = "https://api.anthropic.com/v1/messages"

    HANDLERS_BY_TYPE = {
        "message_start": "handle_message_start",
        "content_block_start": "handle_content_block_start",
        "content_block_delta": "handle_content_block_delta",
        "content_block_stop": "handle_content_block_stop",
        "message_delta": "handle_message_delta",
        "message_stop": "handle_message_stop",
    }

    FINISH_REASONS = {"end_turn": "stop", "tool_use": "tool_use", "max_tokens": "max_tokens"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_tokens = 0
        self.output_tokens = 0
        self.stop_reason = None

    def build_request(self):
        payload = {"model": self.model, "max_tokens": 4096, "messages": self.messages, "stream": True}
        if self.tools:
            payload["tools"] = self.tools
        return self.client.build_request(
            "POST",
            self.URL,
            json=payload,
            headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
        )

    def handle_message_start(self, raw_event):
        self.input_tokens = raw_event["message"]["usage"]["input_tokens"]
        return [StartEvent()]

    def handle_content_block_start(self, raw_event):
        index, block = raw_event["index"], raw_event["content_block"]
        if block["type"] == "text":
            self.message.content.append(TextBlock(text=""))
            return [TextStartEvent(index=index)]
        if block["type"] == "tool_use":
            call = ToolCall(id=block["id"], name=block["name"], arguments={}, partial_arguments="", complete=False)
            self.message.content.append(call)
            return [ToolCallStartEvent(index=index, id=call.id, name=call.name)]
        return []

    def handle_content_block_delta(self, raw_event):
        index, delta = raw_event["index"], raw_event["delta"]
        if delta["type"] == "text_delta":
            self.message.content[index].text += delta["text"]
            return [TextDeltaEvent(index=index, delta=delta["text"])]
        if delta["type"] == "input_json_delta":
            self.message.content[index].partial_arguments += delta["partial_json"]
            return [ToolCallDeltaEvent(index=index, arguments_delta=delta["partial_json"])]
        return []

    def handle_content_block_stop(self, raw_event):
        index = raw_event["index"]
        block = self.message.content[index]
        if isinstance(block, TextBlock):
            return [TextEndEvent(index=index, content=block.text)]
        if isinstance(block, ToolCall):
            block.arguments = json.loads(block.partial_arguments) if block.partial_arguments else {}
            block.partial_arguments = ""
            block.complete = True
            return [ToolCallEndEvent(index=index, tool_call=block)]
        return []

    def handle_message_delta(self, raw_event):
        self.stop_reason = raw_event["delta"]["stop_reason"]
        self.output_tokens = raw_event["usage"]["output_tokens"]
        return []

    def handle_message_stop(self, raw_event):
        usage = Usage(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=self.input_tokens + self.output_tokens,
        )
        return [
            UsageEvent(usage=usage),
            FinishEvent(
                message=self.message.model_copy(deep=True),
                finish_reason=self.FINISH_REASONS.get(self.stop_reason, self.stop_reason),
                provider_finish_reason=self.stop_reason,
                usage=usage,
                tool_calls=self.message.tool_calls,
            ),
        ]


class OpenAIChatCompletionsStreamer(BaseStreamer):
    """Wire knowledge for /v1/chat/completions.

    Chunks carry no event type — one chunk mixes text, tool fragments, the
    finish reason, and usage — so this transport overrides handle() instead of
    filling HANDLERS_BY_TYPE. The wire also has no block start/stop events:
    the streamer synthesizes them (start on first delta, end at finish_reason).
    """

    PROVIDER = "openai"
    URL = "https://api.openai.com/v1/chat/completions"

    FINISH_REASONS = {"stop": "stop", "tool_calls": "tool_use", "length": "max_tokens"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.started = False
        self.text_index = None  # the open text block, if any
        self.tool_indices = {}  # wire tool-call index -> message.content index
        self.stop_reason = None

    def build_request(self):
        payload = {
            "model": self.model,
            "messages": self.messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.tools:
            payload["tools"] = self.tools
        return self.client.build_request(
            "POST",
            self.URL,
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    def handle(self, chunk):
        events = []
        if not self.started:
            self.started = True
            events.append(StartEvent())
        if chunk.get("choices"):
            choice = chunk["choices"][0]
            delta = choice.get("delta") or {}
            if delta.get("content"):
                events += self.handle_text_delta(delta["content"])
            for fragment in delta.get("tool_calls") or []:
                events += self.handle_tool_fragment(fragment)
            if choice.get("finish_reason"):
                self.stop_reason = choice["finish_reason"]
                events += self.close_open_blocks()
        if chunk.get("usage"):
            events += self.handle_usage(chunk["usage"])
        return events

    def handle_text_delta(self, text):
        events = []
        if self.text_index is None:
            self.message.content.append(TextBlock(text=""))
            self.text_index = len(self.message.content) - 1
            events.append(TextStartEvent(index=self.text_index))
        self.message.content[self.text_index].text += text
        events.append(TextDeltaEvent(index=self.text_index, delta=text))
        return events

    def handle_tool_fragment(self, fragment):
        events = []
        if fragment.get("id"):  # the first fragment of a call carries id + name
            call = ToolCall(
                id=fragment["id"],
                name=fragment["function"]["name"],
                arguments={},
                partial_arguments="",
                complete=False,
            )
            self.message.content.append(call)
            index = len(self.message.content) - 1
            self.tool_indices[fragment["index"]] = index
            events.append(ToolCallStartEvent(index=index, id=call.id, name=call.name))
        arguments = (fragment.get("function") or {}).get("arguments")
        if arguments:
            index = self.tool_indices[fragment["index"]]
            self.message.content[index].partial_arguments += arguments
            events.append(ToolCallDeltaEvent(index=index, arguments_delta=arguments))
        return events

    def close_open_blocks(self):
        events = []
        if self.text_index is not None:
            block = self.message.content[self.text_index]
            events.append(TextEndEvent(index=self.text_index, content=block.text))
            self.text_index = None
        for index in sorted(self.tool_indices.values()):
            call = self.message.content[index]
            call.arguments = json.loads(call.partial_arguments) if call.partial_arguments else {}
            call.partial_arguments = ""
            call.complete = True
            events.append(ToolCallEndEvent(index=index, tool_call=call))
        self.tool_indices = {}
        return events

    def handle_usage(self, raw_usage):
        usage = Usage(
            input_tokens=raw_usage.get("prompt_tokens", 0),
            output_tokens=raw_usage.get("completion_tokens", 0),
            total_tokens=raw_usage.get("total_tokens", 0),
        )
        return [
            UsageEvent(usage=usage),
            FinishEvent(
                message=self.message.model_copy(deep=True),
                finish_reason=self.FINISH_REASONS.get(self.stop_reason, self.stop_reason),
                provider_finish_reason=self.stop_reason,
                usage=usage,
                tool_calls=self.message.tool_calls,
            ),
        ]


class OpenRouterChatCompletionsStreamer(OpenAIChatCompletionsStreamer):
    """Same wire as OpenAI Chat Completions; only the host and the label change."""

    PROVIDER = "openrouter"
    URL = "https://openrouter.ai/api/v1/chat/completions"


class AsyncOpenAIResponsesStreamer(AsyncStreamerMixin, OpenAIResponsesStreamer):
    pass


class SyncOpenAIResponsesStreamer(SyncStreamerMixin, OpenAIResponsesStreamer):
    pass


class AsyncOpenAIChatCompletionsStreamer(AsyncStreamerMixin, OpenAIChatCompletionsStreamer):
    pass


class SyncOpenAIChatCompletionsStreamer(SyncStreamerMixin, OpenAIChatCompletionsStreamer):
    pass


class AsyncOpenRouterStreamer(AsyncStreamerMixin, OpenRouterChatCompletionsStreamer):
    pass


class SyncOpenRouterStreamer(SyncStreamerMixin, OpenRouterChatCompletionsStreamer):
    pass


class AsyncAnthropicStreamer(AsyncStreamerMixin, AnthropicStreamer):
    pass


class SyncAnthropicStreamer(SyncStreamerMixin, AnthropicStreamer):
    pass


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

ADD_SCHEMA = {
    "type": "object",
    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
    "required": ["a", "b"],
}

RESPONSES_TOOLS = [
    {
        "type": "function",
        "name": "add",
        "description": "Add two numbers",
        "parameters": {**ADD_SCHEMA, "additionalProperties": False},
        "strict": True,
    }
]

CHAT_COMPLETIONS_TOOLS = [
    {
        "type": "function",
        "function": {"name": "add", "description": "Add two numbers", "parameters": ADD_SCHEMA},
    }
]

ANTHROPIC_TOOLS = [{"name": "add", "description": "Add two numbers", "input_schema": ADD_SCHEMA}]

PROVIDERS = {
    "openai": {
        "model": "gpt-5.4-mini",
        "key_env": "OPENAI_API_KEY",
        "default_transport": "responses",
        "transports": {
            "responses": {
                "streamers": {"async": AsyncOpenAIResponsesStreamer, "sync": SyncOpenAIResponsesStreamer},
                "tools": RESPONSES_TOOLS,
            },
            "chat_completions": {
                "streamers": {"async": AsyncOpenAIChatCompletionsStreamer, "sync": SyncOpenAIChatCompletionsStreamer},
                "tools": CHAT_COMPLETIONS_TOOLS,
            },
        },
    },
    "openrouter": {
        "model": "openai/gpt-5.4-mini",
        "key_env": "OPENROUTER_API_KEY",
        "default_transport": "chat_completions",
        "transports": {
            "chat_completions": {
                "streamers": {"async": AsyncOpenRouterStreamer, "sync": SyncOpenRouterStreamer},
                "tools": CHAT_COMPLETIONS_TOOLS,
            },
        },
    },
    "anthropic": {
        "model": "claude-haiku-4-5",
        "key_env": "ANTHROPIC_API_KEY",
        "default_transport": "messages",
        "transports": {
            "messages": {
                "streamers": {"async": AsyncAnthropicStreamer, "sync": SyncAnthropicStreamer},
                "tools": ANTHROPIC_TOOLS,
            },
        },
    },
}


def show(event):
    if event.type == "text_delta":
        print(event.delta, end="", flush=True)
    elif event.type == "tool_call_end":
        print(f"[{event.type}] {event.tool_call.name} {event.tool_call.arguments}")
    elif event.type == "finish":
        print(f"\n[{event.type}] {event.finish_reason}, {event.usage.total_tokens} tokens")
    elif event.type == "error":
        print(f"\n[{event.type}] {event.error}")
    else:
        print(f"[{event.type}]")


async def run_async(streamer):
    async with streamer as stream:
        async for event in stream:
            show(event)


def run_sync(streamer):
    with streamer as stream:
        for event in stream:
            show(event)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=PROVIDERS, required=True)
    parser.add_argument("--transport", default=None, help="defaults to the provider's only/primary transport")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--sync", dest="mode", action="store_const", const="sync")
    mode.add_argument("--async", dest="mode", action="store_const", const="async")
    parser.set_defaults(mode="async")
    parser.add_argument("--timeout", type=float, nargs="?", const=2.0, default=None)
    args = parser.parse_args()

    config = PROVIDERS[args.provider]
    transport_name = args.transport or config["default_transport"]
    if transport_name not in config["transports"]:
        parser.error(f"{args.provider} supports: {', '.join(config['transports'])}")
    transport = config["transports"][transport_name]

    prompt = "Tell me a long story." if args.timeout else "Use the `add` function to compute 2+3"
    streamer = transport["streamers"][args.mode](
        messages=[{"role": "user", "content": prompt}],
        model=config["model"],
        tools=None if args.timeout else transport["tools"],
        api_key=os.environ[config["key_env"]],
        timeout=args.timeout,
    )

    if args.mode == "async":
        asyncio.run(run_async(streamer))
    else:
        run_sync(streamer)


if __name__ == "__main__":
    main()
