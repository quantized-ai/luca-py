"""Web operations on the Responses stream: the private block opens on the
added stub, everything else — action, results, synthetic block, fact events —
lands at output_item.done, and citations stream as complete annotations.
Handler-level except the final full-wire case; only that one enters the
streamer."""

from luca.client.transports.openai_responses.streamer import SyncOpenAIResponsesStreamer
from luca.client.types import (
    AssistantMessage,
    ChatCompletionRequest,
    FinishEvent,
    PrivateProviderBlock,
    StartEvent,
    TextAnnotationEvent,
    TextBlock,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    URLCitationAnnotation,
    Usage,
    UsageEvent,
    UserMessage,
    WebEndEvent,
    WebFetchBlock,
    WebFetchEvent,
    WebFindEvent,
    WebPagePart,
    WebSearchBlock,
    WebSearchEvent,
    WebSearchResultEvent,
    WebStartEvent,
)
from tests.client._helpers.httpx_mocks import make_sync_client, sse_response

REQUEST = ChatCompletionRequest(
    model="gpt-5.1",
    provider="openai",
    messages=[UserMessage(content="hi")],
)

_STUB = {"id": "ws_1", "type": "web_search_call", "status": "in_progress"}

_ADDED = {"type": "response.output_item.added", "output_index": 0, "item": _STUB}

_SEARCH_DONE_ITEM = {
    "id": "ws_1",
    "type": "web_search_call",
    "status": "completed",
    "action": {"type": "search", "queries": ["apple results"], "query": "apple results"},
    "results": [
        {"type": "text_result", "title": "Apple", "url": "https://apple.com", "snippet": "Apple today..."},
    ],
}

_SEARCH_DONE = {"type": "response.output_item.done", "output_index": 0, "item": _SEARCH_DONE_ITEM}


def _streamer() -> SyncOpenAIResponsesStreamer:
    return SyncOpenAIResponsesStreamer(REQUEST)


def test_the_added_stub_opens_the_private_block():
    s = _streamer()

    events = s.handle(_ADDED)

    assert events == [StartEvent(), WebStartEvent(id="ws_1")]
    assert s.message.content == [PrivateProviderBlock(format="openai.responses", data=_STUB)]


def test_the_progress_events_carry_nothing_and_are_ignored():
    s = _streamer()
    s.handle(_ADDED)

    assert s.handle({"type": "response.web_search_call.in_progress", "item_id": "ws_1", "output_index": 0}) == []
    assert s.handle({"type": "response.web_search_call.searching", "item_id": "ws_1", "output_index": 0}) == []
    assert s.handle({"type": "response.web_search_call.completed", "item_id": "ws_1", "output_index": 0}) == []


def test_the_done_item_completes_both_views_and_states_the_facts():
    s = _streamer()
    s.handle(_ADDED)

    events = s.handle(_SEARCH_DONE)

    results = [WebPagePart(url="https://apple.com", title="Apple", content="Apple today...")]
    assert events == [
        WebSearchEvent(id="ws_1", queries=["apple results"]),
        WebSearchResultEvent(id="ws_1", results=results),
        WebEndEvent(id="ws_1"),
    ]
    assert s.message.content == [
        PrivateProviderBlock(format="openai.responses", data=_SEARCH_DONE_ITEM),
        WebSearchBlock(queries=["apple results"], results=results, extras={"id": "ws_1"}),
    ]


def test_a_duplicate_done_is_ignored():
    s = _streamer()
    s.handle(_ADDED)
    s.handle(_SEARCH_DONE)

    assert s.handle(_SEARCH_DONE) == []
    assert len(s.message.content) == 2


def test_a_search_without_results_emits_no_result_event():
    s = _streamer()
    s.handle(_ADDED)
    item = {
        "id": "ws_1",
        "type": "web_search_call",
        "status": "completed",
        "action": {"type": "search", "queries": ["apple results"], "query": "apple results"},
    }

    events = s.handle({"type": "response.output_item.done", "output_index": 0, "item": item})

    assert events == [
        WebSearchEvent(id="ws_1", queries=["apple results"]),
        WebEndEvent(id="ws_1"),
    ]
    assert s.message.content == [
        PrivateProviderBlock(format="openai.responses", data=item),
        WebSearchBlock(queries=["apple results"], results=None, extras={"id": "ws_1"}),
    ]


def test_an_open_page_states_the_url_it_opens():
    s = _streamer()
    s.handle(_ADDED)
    item = {
        "id": "ws_1",
        "type": "web_search_call",
        "status": "completed",
        "action": {"type": "open_page", "url": "https://apple.com"},
    }

    events = s.handle({"type": "response.output_item.done", "output_index": 0, "item": item})

    assert events == [
        WebFetchEvent(id="ws_1", urls=["https://apple.com"]),
        WebEndEvent(id="ws_1"),
    ]
    assert s.message.content == [
        PrivateProviderBlock(format="openai.responses", data=item),
        WebFetchBlock(web_page=WebPagePart(url="https://apple.com"), extras={"id": "ws_1"}),
    ]


def test_a_find_in_page_states_url_and_pattern_with_no_synthetic():
    s = _streamer()
    s.handle(_ADDED)
    item = {
        "id": "ws_1",
        "type": "web_search_call",
        "status": "completed",
        "action": {"type": "find_in_page", "pattern": "R&D", "url": "https://sec.gov/aapl.htm"},
    }

    events = s.handle({"type": "response.output_item.done", "output_index": 0, "item": item})

    assert events == [
        WebFindEvent(id="ws_1", url="https://sec.gov/aapl.htm", pattern="R&D"),
        WebEndEvent(id="ws_1"),
    ]
    assert s.message.content == [PrivateProviderBlock(format="openai.responses", data=item)]


def test_an_annotation_lands_on_its_text_block_as_it_arrives():
    s = _streamer()
    s.handle(
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"id": "msg_1", "type": "message", "status": "in_progress", "role": "assistant"},
        }
    )
    s.handle(
        {
            "type": "response.content_part.added",
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "annotations": [], "text": ""},
        }
    )
    s.handle(
        {
            "type": "response.output_text.delta",
            "output_index": 0,
            "content_index": 0,
            "delta": "Market update: Apple rose 2.8% today.",
        }
    )

    events = s.handle(
        {
            "type": "response.output_text.annotation.added",
            "output_index": 0,
            "content_index": 0,
            "annotation_index": 0,
            "annotation": {
                "type": "url_citation",
                "url": "https://example.com/apple",
                "title": "Apple shares rise",
                "start_index": 15,
                "end_index": 37,
            },
        }
    )

    annotation = URLCitationAnnotation(
        url="https://example.com/apple",
        title="Apple shares rise",
        start_index=15,
        end_index=37,
    )
    assert events == [TextAnnotationEvent(index=0, annotation=annotation)]
    assert s.message.content == [
        TextBlock(text="Market update: Apple rose 2.8% today.", annotations=[annotation]),
    ]


def test_an_annotation_for_an_unopened_block_is_ignored():
    s = _streamer()

    events = s.handle(
        {
            "type": "response.output_text.annotation.added",
            "output_index": 5,
            "content_index": 0,
            "annotation": {"type": "url_citation", "url": "https://x", "title": "X"},
        }
    )

    assert events == [StartEvent()]
    assert s.message.content == []


def test_a_non_url_annotation_kind_is_ignored():
    s = _streamer()
    s.handle(
        {
            "type": "response.content_part.added",
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "annotations": [], "text": ""},
        }
    )

    events = s.handle(
        {
            "type": "response.output_text.annotation.added",
            "output_index": 0,
            "content_index": 0,
            "annotation": {"type": "file_citation", "file_id": "file_1"},
        }
    )

    assert events == []
    assert s.message.content == [TextBlock(text="")]


def test_a_full_search_stream_narrates_the_operation(responses_transport_factory):
    done_item = {
        "id": "ws_1",
        "type": "web_search_call",
        "status": "completed",
        "action": {"type": "search", "queries": ["apple"], "query": "apple"},
        "results": [{"type": "text_result", "title": "Apple", "url": "https://apple.com", "snippet": "Apple..."}],
    }
    chunks = [
        b'data: {"type":"response.created","response":{"id":"resp_1","status":"in_progress"}}\n\n',
        b'data: {"type":"response.output_item.added","output_index":0,'
        b'"item":{"id":"ws_1","type":"web_search_call","status":"in_progress"}}\n\n',
        b'data: {"type":"response.web_search_call.in_progress","item_id":"ws_1","output_index":0}\n\n',
        b'data: {"type":"response.web_search_call.searching","item_id":"ws_1","output_index":0}\n\n',
        b'data: {"type":"response.web_search_call.completed","item_id":"ws_1","output_index":0}\n\n',
        b'data: {"type":"response.output_item.done","output_index":0,"item":{"id":"ws_1",'
        b'"type":"web_search_call","status":"completed","action":{"type":"search","queries":["apple"],'
        b'"query":"apple"},"results":[{"type":"text_result","title":"Apple","url":"https://apple.com",'
        b'"snippet":"Apple..."}]}}\n\n',
        b'data: {"type":"response.output_item.added","output_index":1,'
        b'"item":{"id":"msg_1","type":"message","status":"in_progress","role":"assistant","content":[]}}\n\n',
        b'data: {"type":"response.content_part.added","output_index":1,"content_index":0,'
        b'"part":{"type":"output_text","annotations":[],"text":""}}\n\n',
        b'data: {"type":"response.output_text.delta","output_index":1,"content_index":0,"delta":"Apple is up."}\n\n',
        b'data: {"type":"response.output_text.annotation.added","output_index":1,"content_index":0,'
        b'"annotation_index":0,"annotation":{"type":"url_citation","url":"https://apple.com",'
        b'"title":"Apple","start_index":0,"end_index":12}}\n\n',
        b'data: {"type":"response.content_part.done","output_index":1,"content_index":0,'
        b'"part":{"type":"output_text","annotations":[],"text":"Apple is up."}}\n\n',
        b'data: {"type":"response.output_item.done","output_index":1,'
        b'"item":{"id":"msg_1","type":"message","status":"completed","role":"assistant"}}\n\n',
        b'data: {"type":"response.completed","response":{"id":"resp_1","status":"completed",'
        b'"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15},'
        b'"tool_usage":{"web_search":{"num_requests":1}}}}\n\n',
    ]
    transport = responses_transport_factory(http_client=make_sync_client(sse_response(chunks)))

    with transport.completion_stream(REQUEST) as s:
        events = list(s)

    annotation = URLCitationAnnotation(url="https://apple.com", title="Apple", start_index=0, end_index=12)
    usage = Usage(
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        tool_requests={"web_search": 1},
        provider_tool_usage={"web_search": {"num_requests": 1}},
    )
    message = AssistantMessage(
        content=[
            PrivateProviderBlock(format="openai.responses", data=done_item),
            WebSearchBlock(
                queries=["apple"],
                results=[WebPagePart(url="https://apple.com", title="Apple", content="Apple...")],
                extras={"id": "ws_1"},
            ),
            TextBlock(text="Apple is up.", annotations=[annotation]),
        ],
        finish_reason="stop",
        provider_finish_reason="completed",
        provider="openai",
        model="gpt-5.1",
        usage=usage,
    )
    assert events == [
        StartEvent(),
        WebStartEvent(id="ws_1"),
        WebSearchEvent(id="ws_1", queries=["apple"]),
        WebSearchResultEvent(
            id="ws_1", results=[WebPagePart(url="https://apple.com", title="Apple", content="Apple...")]
        ),
        WebEndEvent(id="ws_1"),
        TextStartEvent(index=2),
        TextDeltaEvent(index=2, delta="Apple is up."),
        TextAnnotationEvent(index=2, annotation=annotation),
        TextEndEvent(index=2, content="Apple is up."),
        UsageEvent(usage=usage),
        FinishEvent(
            message=message,
            finish_reason="stop",
            provider_finish_reason="completed",
            usage=usage,
            tool_calls=[],
        ),
    ]


def test_the_terminal_usage_carries_the_tool_request_counters():
    s = _streamer()

    events = s.handle(
        {
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "status": "completed",
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                "tool_usage": {"web_search": {"num_requests": 3}},
            },
        }
    )

    assert [e.type for e in events] == ["start", "usage", "finish"]
    assert events[1].usage == Usage(
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        tool_requests={"web_search": 3},
        provider_tool_usage={"web_search": {"num_requests": 3}},
    )


def test_a_streamed_failed_search_stamps_the_status():
    # streaming twin of the invented failed-item fixture: same mixin builder,
    # same stamp
    s = _streamer()
    s.handle(_ADDED)
    item = {
        "id": "ws_1",
        "type": "web_search_call",
        "status": "failed",
        "action": {"type": "search", "queries": ["apple results"], "query": "apple results"},
    }

    s.handle({"type": "response.output_item.done", "output_index": 0, "item": item})

    assert s.message.content == [
        PrivateProviderBlock(format="openai.responses", data=item),
        WebSearchBlock(
            queries=["apple results"],
            results=None,
            extras={"id": "ws_1", "error": {"status": "failed"}},
        ),
    ]
