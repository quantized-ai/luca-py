"""OpenAITransport.completion_stream() — sync."""

from dataclasses import dataclass

import pytest

from luca.client.types import (
    AssistantMessage,
    ChatCompletionRequest,
    FinishEvent,
    RefusalBlock,
    RefusalDeltaEvent,
    RefusalEndEvent,
    RefusalStartEvent,
    StartEvent,
    StreamEvent,
    TextBlock,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingBlock,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    Usage,
    UsageEvent,
    UserMessage,
)
from tests.client._helpers.httpx_mocks import make_sync_client, sse_response


def _data(payload: str) -> bytes:
    return f"data: {payload}\n\n".encode()


@dataclass(frozen=True)
class StreamCase:
    name: str
    request: ChatCompletionRequest
    sse_chunks: list[bytes]
    expected_events: list[StreamEvent]


_REQ = ChatCompletionRequest(
    model="gpt-4o",
    provider="openai",
    messages=[UserMessage(content="hi")],
)


CASES = [
    StreamCase(
        name="single_text_block",
        request=_REQ,
        sse_chunks=[
            _data('{"choices":[{"index":0,"delta":{"content":"Hi"}}]}'),
            _data('{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'),
            _data('{"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}'),
            _data("[DONE]"),
        ],
        expected_events=[
            StartEvent(),
            TextStartEvent(index=0),
            TextDeltaEvent(index=0, delta="Hi"),
            TextEndEvent(index=0, content="Hi"),
            UsageEvent(usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2)),
            FinishEvent(
                message=AssistantMessage(
                    content=[TextBlock(text="Hi")],
                    finish_reason="stop",
                    provider_finish_reason="stop",
                    cancelled=False,
                    error_message=None,
                    provider="openai",
                    model="gpt-4o",
                    usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
                ),
                finish_reason="stop",
                provider_finish_reason="stop",
                cancelled=False,
                usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
                tool_calls=[],
            ),
        ],
    ),
    StreamCase(
        name="reasoning_then_text",
        request=_REQ,
        sse_chunks=[
            _data('{"choices":[{"index":0,"delta":{"reasoning":"Let me think."}}]}'),
            _data('{"choices":[{"index":0,"delta":{"content":"Hi"}}]}'),
            _data('{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'),
            _data(
                '{"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":3,"total_tokens":4,"completion_tokens_details":{"reasoning_tokens":2}}}'
            ),
            _data("[DONE]"),
        ],
        expected_events=[
            StartEvent(),
            ThinkingStartEvent(index=0),
            ThinkingDeltaEvent(index=0, delta="Let me think."),
            TextStartEvent(index=1),
            TextDeltaEvent(index=1, delta="Hi"),
            ThinkingEndEvent(index=0, content="Let me think."),
            TextEndEvent(index=1, content="Hi"),
            UsageEvent(usage=Usage(input_tokens=1, output_tokens=3, total_tokens=4, reasoning_tokens=2)),
            FinishEvent(
                message=AssistantMessage(
                    content=[ThinkingBlock(text="Let me think."), TextBlock(text="Hi")],
                    finish_reason="stop",
                    provider_finish_reason="stop",
                    cancelled=False,
                    error_message=None,
                    provider="openai",
                    model="gpt-4o",
                    usage=Usage(input_tokens=1, output_tokens=3, total_tokens=4, reasoning_tokens=2),
                ),
                finish_reason="stop",
                provider_finish_reason="stop",
                cancelled=False,
                usage=Usage(input_tokens=1, output_tokens=3, total_tokens=4, reasoning_tokens=2),
                tool_calls=[],
            ),
        ],
    ),
    StreamCase(
        name="single_tool_call_id_and_name_in_first_chunk",
        request=_REQ,
        sse_chunks=[
            _data(
                '{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_abc",'
                '"function":{"name":"get_weather","arguments":""}}]}}]}'
            ),
            _data('{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"loca"}}]}}]}'),
            _data(
                '{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"tion\\":\\"NYC\\"}"}}]}}]}'
            ),
            _data('{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}'),
            _data("[DONE]"),
        ],
        expected_events=[
            StartEvent(),
            ToolCallStartEvent(index=0, id="call_abc", name="get_weather"),
            ToolCallDeltaEvent(index=0, arguments_delta='{"loca'),
            ToolCallDeltaEvent(index=0, arguments_delta='tion":"NYC"}'),
            ToolCallEndEvent(
                index=0,
                tool_call=ToolCall(
                    id="call_abc",
                    name="get_weather",
                    arguments={"location": "NYC"},
                    partial_arguments="",
                    complete=True,
                ),
            ),
            FinishEvent(
                message=AssistantMessage(
                    content=[
                        ToolCall(
                            id="call_abc",
                            name="get_weather",
                            arguments={"location": "NYC"},
                            partial_arguments="",
                            complete=True,
                        )
                    ],
                    finish_reason="tool_use",
                    provider_finish_reason="tool_calls",
                    cancelled=False,
                    error_message=None,
                    provider="openai",
                    model="gpt-4o",
                    usage=Usage(),
                ),
                finish_reason="tool_use",
                provider_finish_reason="tool_calls",
                cancelled=False,
                usage=Usage(),
                tool_calls=[
                    ToolCall(
                        id="call_abc",
                        name="get_weather",
                        arguments={"location": "NYC"},
                        partial_arguments="",
                        complete=True,
                    )
                ],
            ),
        ],
    ),
]


def test_streamed_refusal_is_preserved_and_classified_as_an_error(openai_transport_factory):
    client = make_sync_client(
        sse_response(
            [
                _data('{"choices":[{"index":0,"delta":{"role":"assistant","content":null,"refusal":""}}]}'),
                _data('{"choices":[{"index":0,"delta":{"refusal":"I\'m sorry"}}]}'),
                _data('{"choices":[{"index":0,"delta":{"refusal":", I cannot assist with that request."}}]}'),
                _data('{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'),
                _data('{"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}'),
                _data("[DONE]"),
            ]
        )
    )
    transport = openai_transport_factory(http_client=client)

    with transport.completion_stream(_REQ) as stream:
        events = list(stream)

    refusal = "I'm sorry, I cannot assist with that request."
    assert events == [
        StartEvent(),
        RefusalStartEvent(index=0),
        RefusalDeltaEvent(index=0, delta="I'm sorry"),
        RefusalDeltaEvent(index=0, delta=", I cannot assist with that request."),
        RefusalEndEvent(index=0, content=refusal),
        UsageEvent(usage=Usage(input_tokens=1, output_tokens=2, total_tokens=3)),
        FinishEvent(
            message=AssistantMessage(
                content=[RefusalBlock(text=refusal)],
                finish_reason="error",
                provider_finish_reason="stop",
                cancelled=False,
                error_message=f"OpenAI refusal: {refusal}",
                provider="openai",
                model="gpt-4o",
                usage=Usage(input_tokens=1, output_tokens=2, total_tokens=3),
            ),
            finish_reason="error",
            provider_finish_reason="stop",
            cancelled=False,
            usage=Usage(input_tokens=1, output_tokens=2, total_tokens=3),
            tool_calls=[],
        ),
    ]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_openai_transport_completion_stream(case, openai_transport_factory):
    client = make_sync_client(sse_response(case.sse_chunks))
    transport = openai_transport_factory(http_client=client)
    with transport.completion_stream(case.request) as s:
        events = list(s)
    assert events == case.expected_events
