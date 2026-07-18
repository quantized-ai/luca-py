"""OpenAITransport.completion_stream() — sync."""

from dataclasses import dataclass

import pytest

from luca.client.types import (
    AssistantMessage,
    ChatCompletionRequest,
    FinishEvent,
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
from tests.client._helpers.stream_iteration import collect_events_with_snapshots


def _data(payload: str) -> bytes:
    return f"data: {payload}\n\n".encode()


@dataclass(frozen=True)
class StreamCase:
    name: str
    request: ChatCompletionRequest
    sse_chunks: list[bytes]
    expected_events: list[StreamEvent]


_REQ = ChatCompletionRequest(
    model="gpt-4o", provider="openai",
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
            StartEvent(partial=AssistantMessage(content=[], provider="openai", model="gpt-4o")),
            TextStartEvent(index=0, partial=AssistantMessage(
                content=[TextBlock(text="")], provider="openai", model="gpt-4o")),
            TextDeltaEvent(index=0, delta="Hi", partial=AssistantMessage(
                content=[TextBlock(text="Hi")], provider="openai", model="gpt-4o")),
            TextEndEvent(index=0, content="Hi", partial=AssistantMessage(
                content=[TextBlock(text="Hi")], provider="openai", model="gpt-4o")),
            UsageEvent(
                usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
                partial=AssistantMessage(
                    content=[TextBlock(text="Hi")], provider="openai", model="gpt-4o"),
            ),
            FinishEvent(
                message=AssistantMessage(
                    content=[TextBlock(text="Hi")],
                    finish_reason="stop", provider_finish_reason="stop",
                    cancelled=False, error_message=None,
                    provider="openai", model="gpt-4o",
                    usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
                ),
                finish_reason="stop", provider_finish_reason="stop",
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
            _data('{"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":3,"total_tokens":4,"completion_tokens_details":{"reasoning_tokens":2}}}'),
            _data("[DONE]"),
        ],
        expected_events=[
            StartEvent(partial=AssistantMessage(content=[], provider="openai", model="gpt-4o")),
            ThinkingStartEvent(index=0, partial=AssistantMessage(
                content=[ThinkingBlock(text="")], provider="openai", model="gpt-4o")),
            ThinkingDeltaEvent(index=0, delta="Let me think.", partial=AssistantMessage(
                content=[ThinkingBlock(text="Let me think.")], provider="openai", model="gpt-4o")),
            TextStartEvent(index=1, partial=AssistantMessage(
                content=[ThinkingBlock(text="Let me think."), TextBlock(text="")],
                provider="openai", model="gpt-4o")),
            TextDeltaEvent(index=1, delta="Hi", partial=AssistantMessage(
                content=[ThinkingBlock(text="Let me think."), TextBlock(text="Hi")],
                provider="openai", model="gpt-4o")),
            ThinkingEndEvent(index=0, content="Let me think.", partial=AssistantMessage(
                content=[ThinkingBlock(text="Let me think."), TextBlock(text="Hi")],
                provider="openai", model="gpt-4o")),
            TextEndEvent(index=1, content="Hi", partial=AssistantMessage(
                content=[ThinkingBlock(text="Let me think."), TextBlock(text="Hi")],
                provider="openai", model="gpt-4o")),
            UsageEvent(
                usage=Usage(input_tokens=1, output_tokens=3, total_tokens=4, reasoning_tokens=2),
                partial=AssistantMessage(
                    content=[ThinkingBlock(text="Let me think."), TextBlock(text="Hi")],
                    provider="openai", model="gpt-4o"),
            ),
            FinishEvent(
                message=AssistantMessage(
                    content=[ThinkingBlock(text="Let me think."), TextBlock(text="Hi")],
                    finish_reason="stop", provider_finish_reason="stop",
                    cancelled=False, error_message=None,
                    provider="openai", model="gpt-4o",
                    usage=Usage(input_tokens=1, output_tokens=3, total_tokens=4, reasoning_tokens=2),
                ),
                finish_reason="stop", provider_finish_reason="stop",
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
            _data(
                '{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"loca"}}]}}]}'
            ),
            _data(
                '{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"tion\\":\\"NYC\\"}"}}]}}]}'
            ),
            _data('{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}'),
            _data("[DONE]"),
        ],
        expected_events=[
            StartEvent(partial=AssistantMessage(content=[], provider="openai", model="gpt-4o")),
            ToolCallStartEvent(
                index=0, id="call_abc", name="get_weather",
                partial=AssistantMessage(
                    content=[ToolCall(
                        id="call_abc", name="get_weather", arguments={},
                        partial_arguments="", complete=False,
                    )],
                    provider="openai", model="gpt-4o",
                ),
            ),
            ToolCallDeltaEvent(
                index=0, arguments_delta='{"loca',
                partial=AssistantMessage(
                    content=[ToolCall(
                        id="call_abc", name="get_weather", arguments={},
                        partial_arguments='{"loca', complete=False,
                    )],
                    provider="openai", model="gpt-4o",
                ),
            ),
            ToolCallDeltaEvent(
                index=0, arguments_delta='tion":"NYC"}',
                partial=AssistantMessage(
                    content=[ToolCall(
                        id="call_abc", name="get_weather", arguments={},
                        partial_arguments='{"location":"NYC"}', complete=False,
                    )],
                    provider="openai", model="gpt-4o",
                ),
            ),
            ToolCallEndEvent(
                index=0,
                tool_call=ToolCall(
                    id="call_abc", name="get_weather",
                    arguments={"location": "NYC"},
                    partial_arguments="", complete=True,
                ),
                partial=AssistantMessage(
                    content=[ToolCall(
                        id="call_abc", name="get_weather",
                        arguments={"location": "NYC"},
                        partial_arguments="", complete=True,
                    )],
                    provider="openai", model="gpt-4o",
                ),
            ),
            FinishEvent(
                message=AssistantMessage(
                    content=[ToolCall(
                        id="call_abc", name="get_weather",
                        arguments={"location": "NYC"},
                        partial_arguments="", complete=True,
                    )],
                    finish_reason="tool_use", provider_finish_reason="tool_calls",
                    cancelled=False, error_message=None,
                    provider="openai", model="gpt-4o",
                    usage=Usage(),
                ),
                finish_reason="tool_use", provider_finish_reason="tool_calls",
                cancelled=False,
                usage=Usage(),
                tool_calls=[ToolCall(
                    id="call_abc", name="get_weather",
                    arguments={"location": "NYC"},
                    partial_arguments="", complete=True,
                )],
            ),
        ],
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_openai_transport_completion_stream(case, openai_transport_factory):
    client = make_sync_client(sse_response(case.sse_chunks))
    transport = openai_transport_factory(http_client=client)
    with transport.completion_stream(case.request) as s:
        events = collect_events_with_snapshots(s)
    assert events == case.expected_events
