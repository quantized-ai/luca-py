"""NDJSON framing → the canonical stream events.

Reassembling lines from network reads is httpx's job (`iter_lines`), so what
is under test here is the translation: which events one frame produces, how a
tool call and a thinking block differ from OpenAI's fragment-by-fragment
shape, and where the token counts come from.
"""

import httpx

from luca.client.types import ChatCompletionRequest, ModelInfo, UserMessage

from ..._helpers.httpx_mocks import make_async_client, make_sync_client, ndjson_response
from ..._helpers.stream_iteration import acollect_events_with_snapshots, collect_events_with_snapshots

REQUEST = ChatCompletionRequest(
    provider="ollama",
    model="llama3.2:latest",
    messages=[UserMessage(content="Hi")],
    model_info=ModelInfo(model="llama3.2:latest", provider="ollama", context_window=8192),
)

TEXT_FRAMES = [
    {"model": "llama3.2:latest", "message": {"role": "assistant", "content": "Hel"}, "done": False},
    {"model": "llama3.2:latest", "message": {"role": "assistant", "content": "lo"}, "done": False},
    {
        "model": "llama3.2:latest",
        "message": {"role": "assistant", "content": ""},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 31,
        "eval_count": 2,
    },
]


def _stream(factory, frames):
    transport = factory(http_client=make_sync_client(ndjson_response(frames)))
    with transport.completion_stream(REQUEST) as stream:
        return collect_events_with_snapshots(stream)


def test_text_frames_open_one_block_and_close_it_on_done(ollama_transport_factory):
    events = _stream(ollama_transport_factory, TEXT_FRAMES)

    assert [type(e).__name__ for e in events] == [
        "StartEvent",
        "TextStartEvent",
        "TextDeltaEvent",
        "TextDeltaEvent",
        "TextEndEvent",
        "UsageEvent",
        "FinishEvent",
    ]
    assert events[-1].message.content[0].text == "Hello"


def test_usage_comes_off_the_done_frame(ollama_transport_factory):
    events = _stream(ollama_transport_factory, TEXT_FRAMES)

    usage = events[-1].message.usage
    # Ollama reports no total; it is the sum, computed here.
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (31, 2, 33)


def test_a_tool_call_arrives_whole_as_one_block(ollama_transport_factory):
    # Unlike OpenAI, arguments are not streamed as fragments — one frame
    # carries the finished object, so the block opens and closes together.
    events = _stream(
        ollama_transport_factory,
        [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "call_1", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
                    ],
                },
                "done": False,
            },
            {"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop"},
        ],
    )

    call = events[-1].message.content[0]
    assert (call.name, call.arguments, call.id) == ("get_weather", {"city": "Paris"}, "call_1")


def test_thinking_text_becomes_a_thinking_block(ollama_transport_factory):
    events = _stream(
        ollama_transport_factory,
        [
            {"message": {"role": "assistant", "thinking": "let me see"}, "done": False},
            {"message": {"role": "assistant", "content": "51"}, "done": False},
            {"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop"},
        ],
    )

    assert [(type(b).__name__, b.text) for b in events[-1].message.content] == [
        ("ThinkingBlock", "let me see"),
        ("TextBlock", "51"),
    ]


def test_a_truncated_body_ends_the_stream_in_an_error_not_a_finish(ollama_transport_factory):
    # A daemon killed mid-frame. Ending in a FinishEvent would be
    # indistinguishable from a real answer that happens to be short.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b'{"message":{"content":"Hi"},"done":fal')

    transport = ollama_transport_factory(http_client=make_sync_client(handler))
    with transport.completion_stream(REQUEST) as stream:
        events = collect_events_with_snapshots(stream)

    assert type(events[-1]).__name__ == "ErrorEvent"
    assert "non-JSON" in str(events[-1].error)


async def test_the_async_stream_parses_the_same_frames(ollama_transport_factory):
    client = make_async_client(ndjson_response(TEXT_FRAMES))
    transport = ollama_transport_factory(async_http_client=client)
    try:
        async with transport.acompletion_stream(REQUEST) as stream:
            events = await acollect_events_with_snapshots(stream)
        assert events[-1].message.content[0].text == "Hello"
    finally:
        await transport.aclose()
