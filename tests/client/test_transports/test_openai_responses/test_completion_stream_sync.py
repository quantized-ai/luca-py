"""OpenAIResponsesTransport.completion_stream() — sync.

The Responses stream is a sequence of NAMED events over the whole response, so
these cases exercise the item/content-part bookkeeping that turns them back
into dense, ordered content blocks."""

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


REQ = ChatCompletionRequest(
    model="gpt-5.4",
    provider="openai",
    messages=[UserMessage(content="hi")],
)


CASES = [
    StreamCase(
        name="single_text_block",
        request=REQ,
        sse_chunks=[
            _data('{"type":"response.created","response":{"id":"resp_1","status":"in_progress"}}'),
            _data(
                '{"type":"response.output_item.added","output_index":0,'
                '"item":{"id":"msg_1","type":"message","role":"assistant","content":[]}}'
            ),
            _data(
                '{"type":"response.content_part.added","output_index":0,"content_index":0,'
                '"part":{"type":"output_text","text":""}}'
            ),
            _data('{"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"Hi"}'),
            _data('{"type":"response.content_part.done","output_index":0,"content_index":0}'),
            _data(
                '{"type":"response.completed","response":{"id":"resp_1","status":"completed",'
                '"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}'
            ),
            _data("[DONE]"),
        ],
        expected_events=[
            StartEvent(partial=AssistantMessage(content=[], provider="openai", model="gpt-5.4")),
            TextStartEvent(
                index=0,
                partial=AssistantMessage(content=[TextBlock(text="")], provider="openai", model="gpt-5.4"),
            ),
            TextDeltaEvent(
                index=0,
                delta="Hi",
                partial=AssistantMessage(content=[TextBlock(text="Hi")], provider="openai", model="gpt-5.4"),
            ),
            TextEndEvent(
                index=0,
                content="Hi",
                partial=AssistantMessage(content=[TextBlock(text="Hi")], provider="openai", model="gpt-5.4"),
            ),
            UsageEvent(
                usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
                partial=AssistantMessage(content=[TextBlock(text="Hi")], provider="openai", model="gpt-5.4"),
            ),
            FinishEvent(
                message=AssistantMessage(
                    content=[TextBlock(text="Hi")],
                    finish_reason="stop",
                    provider_finish_reason="completed",
                    provider="openai",
                    model="gpt-5.4",
                    usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
                ),
                finish_reason="stop",
                provider_finish_reason="completed",
                usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
                tool_calls=[],
            ),
        ],
    ),
    StreamCase(
        name="reasoning_summary_parts_join_into_one_block_and_the_payload_lands_at_item_done",
        request=REQ,
        sse_chunks=[
            _data(
                '{"type":"response.output_item.added","output_index":0,'
                '"item":{"id":"rs_1","type":"reasoning","summary":[]}}'
            ),
            _data(
                '{"type":"response.reasoning_summary_part.added","output_index":0,"summary_index":0,'
                '"part":{"type":"summary_text","text":""}}'
            ),
            _data(
                '{"type":"response.reasoning_summary_text.delta","output_index":0,"summary_index":0,"delta":"First."}'
            ),
            _data(
                '{"type":"response.reasoning_summary_part.added","output_index":0,"summary_index":1,'
                '"part":{"type":"summary_text","text":""}}'
            ),
            _data(
                '{"type":"response.reasoning_summary_text.delta","output_index":0,"summary_index":1,"delta":"Second."}'
            ),
            _data(
                '{"type":"response.output_item.done","output_index":0,'
                '"item":{"id":"rs_1","type":"reasoning","encrypted_content":"enc-1","summary":[]}}'
            ),
            _data(
                '{"type":"response.completed","response":{"id":"resp_2","status":"completed",'
                '"usage":{"input_tokens":1,"output_tokens":9,"total_tokens":10,'
                '"output_tokens_details":{"reasoning_tokens":8}}}}'
            ),
            _data("[DONE]"),
        ],
        expected_events=[
            StartEvent(partial=AssistantMessage(content=[], provider="openai", model="gpt-5.4")),
            ThinkingStartEvent(
                index=0,
                partial=AssistantMessage(
                    content=[ThinkingBlock(text="", id="rs_1")],
                    provider="openai",
                    model="gpt-5.4",
                ),
            ),
            ThinkingDeltaEvent(
                index=0,
                delta="First.",
                partial=AssistantMessage(
                    content=[ThinkingBlock(text="First.", id="rs_1")],
                    provider="openai",
                    model="gpt-5.4",
                ),
            ),
            ThinkingDeltaEvent(
                index=0,
                delta="\n\n",
                partial=AssistantMessage(
                    content=[ThinkingBlock(text="First.\n\n", id="rs_1")],
                    provider="openai",
                    model="gpt-5.4",
                ),
            ),
            ThinkingDeltaEvent(
                index=0,
                delta="Second.",
                partial=AssistantMessage(
                    content=[ThinkingBlock(text="First.\n\nSecond.", id="rs_1")],
                    provider="openai",
                    model="gpt-5.4",
                ),
            ),
            ThinkingDeltaEvent(
                index=0,
                delta="",
                partial=AssistantMessage(
                    content=[ThinkingBlock(text="First.\n\nSecond.", id="rs_1", signature="enc-1")],
                    provider="openai",
                    model="gpt-5.4",
                ),
            ),
            ThinkingEndEvent(
                index=0,
                content="First.\n\nSecond.",
                partial=AssistantMessage(
                    content=[ThinkingBlock(text="First.\n\nSecond.", id="rs_1", signature="enc-1")],
                    provider="openai",
                    model="gpt-5.4",
                ),
            ),
            UsageEvent(
                usage=Usage(input_tokens=1, output_tokens=9, total_tokens=10, reasoning_tokens=8),
                partial=AssistantMessage(
                    content=[ThinkingBlock(text="First.\n\nSecond.", id="rs_1", signature="enc-1")],
                    provider="openai",
                    model="gpt-5.4",
                ),
            ),
            FinishEvent(
                message=AssistantMessage(
                    content=[ThinkingBlock(text="First.\n\nSecond.", id="rs_1", signature="enc-1")],
                    finish_reason="stop",
                    provider_finish_reason="completed",
                    provider="openai",
                    model="gpt-5.4",
                    usage=Usage(input_tokens=1, output_tokens=9, total_tokens=10, reasoning_tokens=8),
                ),
                finish_reason="stop",
                provider_finish_reason="completed",
                usage=Usage(input_tokens=1, output_tokens=9, total_tokens=10, reasoning_tokens=8),
                tool_calls=[],
            ),
        ],
    ),
    StreamCase(
        name="a_function_call_item_streams_its_arguments",
        request=REQ,
        sse_chunks=[
            _data(
                '{"type":"response.output_item.added","output_index":0,'
                '"item":{"id":"fc_1","type":"function_call","call_id":"call_1","name":"add","arguments":""}}'
            ),
            _data('{"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\\"a\\":1"}'),
            _data('{"type":"response.function_call_arguments.delta","output_index":0,"delta":",\\"b\\":2}"}'),
            _data(
                '{"type":"response.output_item.done","output_index":0,'
                '"item":{"id":"fc_1","type":"function_call","call_id":"call_1","name":"add"}}'
            ),
            _data(
                '{"type":"response.completed","response":{"id":"resp_3","status":"completed",'
                '"usage":{"input_tokens":2,"output_tokens":2,"total_tokens":4}}}'
            ),
            _data("[DONE]"),
        ],
        expected_events=[
            StartEvent(partial=AssistantMessage(content=[], provider="openai", model="gpt-5.4")),
            ToolCallStartEvent(
                index=0,
                id="call_1",
                name="add",
                partial=AssistantMessage(
                    content=[ToolCall(id="call_1", name="add", complete=False)],
                    provider="openai",
                    model="gpt-5.4",
                ),
            ),
            ToolCallDeltaEvent(
                index=0,
                arguments_delta='{"a":1',
                partial=AssistantMessage(
                    content=[ToolCall(id="call_1", name="add", partial_arguments='{"a":1', complete=False)],
                    provider="openai",
                    model="gpt-5.4",
                ),
            ),
            ToolCallDeltaEvent(
                index=0,
                arguments_delta=',"b":2}',
                partial=AssistantMessage(
                    content=[ToolCall(id="call_1", name="add", partial_arguments='{"a":1,"b":2}', complete=False)],
                    provider="openai",
                    model="gpt-5.4",
                ),
            ),
            ToolCallEndEvent(
                index=0,
                tool_call=ToolCall(id="call_1", name="add", arguments={"a": 1, "b": 2}, complete=True),
                partial=AssistantMessage(
                    content=[ToolCall(id="call_1", name="add", arguments={"a": 1, "b": 2}, complete=True)],
                    provider="openai",
                    model="gpt-5.4",
                ),
            ),
            UsageEvent(
                usage=Usage(input_tokens=2, output_tokens=2, total_tokens=4),
                partial=AssistantMessage(
                    content=[ToolCall(id="call_1", name="add", arguments={"a": 1, "b": 2}, complete=True)],
                    provider="openai",
                    model="gpt-5.4",
                ),
            ),
            FinishEvent(
                message=AssistantMessage(
                    content=[ToolCall(id="call_1", name="add", arguments={"a": 1, "b": 2}, complete=True)],
                    finish_reason="tool_use",
                    provider_finish_reason="completed",
                    provider="openai",
                    model="gpt-5.4",
                    usage=Usage(input_tokens=2, output_tokens=2, total_tokens=4),
                ),
                finish_reason="tool_use",
                provider_finish_reason="completed",
                usage=Usage(input_tokens=2, output_tokens=2, total_tokens=4),
                tool_calls=[ToolCall(id="call_1", name="add", arguments={"a": 1, "b": 2}, complete=True)],
            ),
        ],
    ),
    StreamCase(
        name="unknown_event_types_are_ignored_not_fatal",
        request=REQ,
        sse_chunks=[
            _data('{"type":"response.web_search_call.in_progress","output_index":0,"item_id":"ws_1"}'),
            _data('{"type":"response.some_future_event","output_index":0}'),
            _data(
                '{"type":"response.output_item.added","output_index":1,'
                '"item":{"id":"msg_1","type":"message","role":"assistant","content":[]}}'
            ),
            _data(
                '{"type":"response.content_part.added","output_index":1,"content_index":0,'
                '"part":{"type":"output_text","text":""}}'
            ),
            _data('{"type":"response.output_text.delta","output_index":1,"content_index":0,"delta":"ok"}'),
            _data('{"type":"response.content_part.done","output_index":1,"content_index":0}'),
            _data(
                '{"type":"response.completed","response":{"id":"resp_4","status":"completed",'
                '"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}'
            ),
            _data("[DONE]"),
        ],
        expected_events=[
            StartEvent(partial=AssistantMessage(content=[], provider="openai", model="gpt-5.4")),
            TextStartEvent(
                index=0,
                partial=AssistantMessage(content=[TextBlock(text="")], provider="openai", model="gpt-5.4"),
            ),
            TextDeltaEvent(
                index=0,
                delta="ok",
                partial=AssistantMessage(content=[TextBlock(text="ok")], provider="openai", model="gpt-5.4"),
            ),
            TextEndEvent(
                index=0,
                content="ok",
                partial=AssistantMessage(content=[TextBlock(text="ok")], provider="openai", model="gpt-5.4"),
            ),
            UsageEvent(
                usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
                partial=AssistantMessage(content=[TextBlock(text="ok")], provider="openai", model="gpt-5.4"),
            ),
            FinishEvent(
                message=AssistantMessage(
                    content=[TextBlock(text="ok")],
                    finish_reason="stop",
                    provider_finish_reason="completed",
                    provider="openai",
                    model="gpt-5.4",
                    usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
                ),
                finish_reason="stop",
                provider_finish_reason="completed",
                usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
                tool_calls=[],
            ),
        ],
    ),
    StreamCase(
        name="an_incomplete_terminal_classifies_as_length",
        request=REQ,
        sse_chunks=[
            _data(
                '{"type":"response.output_item.added","output_index":0,'
                '"item":{"id":"msg_1","type":"message","role":"assistant","content":[]}}'
            ),
            _data(
                '{"type":"response.content_part.added","output_index":0,"content_index":0,'
                '"part":{"type":"output_text","text":""}}'
            ),
            _data('{"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"Once"}'),
            _data('{"type":"response.content_part.done","output_index":0,"content_index":0}'),
            _data(
                '{"type":"response.incomplete","response":{"id":"resp_5","status":"incomplete",'
                '"incomplete_details":{"reason":"max_output_tokens"},'
                '"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}'
            ),
            _data("[DONE]"),
        ],
        expected_events=[
            StartEvent(partial=AssistantMessage(content=[], provider="openai", model="gpt-5.4")),
            TextStartEvent(
                index=0,
                partial=AssistantMessage(content=[TextBlock(text="")], provider="openai", model="gpt-5.4"),
            ),
            TextDeltaEvent(
                index=0,
                delta="Once",
                partial=AssistantMessage(content=[TextBlock(text="Once")], provider="openai", model="gpt-5.4"),
            ),
            TextEndEvent(
                index=0,
                content="Once",
                partial=AssistantMessage(content=[TextBlock(text="Once")], provider="openai", model="gpt-5.4"),
            ),
            UsageEvent(
                usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
                partial=AssistantMessage(content=[TextBlock(text="Once")], provider="openai", model="gpt-5.4"),
            ),
            FinishEvent(
                message=AssistantMessage(
                    content=[TextBlock(text="Once")],
                    finish_reason="length",
                    provider_finish_reason="incomplete:max_output_tokens",
                    provider="openai",
                    model="gpt-5.4",
                    usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
                ),
                finish_reason="length",
                provider_finish_reason="incomplete:max_output_tokens",
                usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
                tool_calls=[],
            ),
        ],
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_responses_transport_completion_stream(case, responses_transport_factory):
    client = make_sync_client(sse_response(case.sse_chunks))
    transport = responses_transport_factory(http_client=client)

    with transport.completion_stream(case.request) as s:
        events = collect_events_with_snapshots(s)

    assert events == case.expected_events
