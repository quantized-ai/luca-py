"""Web operations and citations on the Anthropic stream: server tool blocks
become private blocks with web events at their boundaries, adjacent text
blocks merge into one run, and held citations become ranged annotations at
their span's stop. Handler-level; the streamer is never entered."""

from luca.client.transports.anthropic.streamer import SyncAnthropicStreamer
from luca.client.types import (
    ChatCompletionRequest,
    PrivateProviderBlock,
    StartEvent,
    TextAnnotationEvent,
    TextBlock,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingStartEvent,
    URLCitationAnnotation,
    Usage,
    UsageEvent,
    UserMessage,
    WebEndEvent,
    WebFetchBlock,
    WebFetchEvent,
    WebPagePart,
    WebSearchBlock,
    WebSearchEvent,
    WebSearchResultEvent,
    WebStartEvent,
)

REQUEST = ChatCompletionRequest(
    model="claude-sonnet-5",
    provider="anthropic",
    messages=[UserMessage(content="hi")],
)

_MESSAGE_START = {
    "type": "message_start",
    "message": {"id": "msg_1", "usage": {"input_tokens": 5, "output_tokens": 0}},
}


def _streamer() -> SyncAnthropicStreamer:
    s = SyncAnthropicStreamer(REQUEST)
    s.handle(_MESSAGE_START)
    return s


def _start(index: int, block: dict) -> dict:
    return {"type": "content_block_start", "index": index, "content_block": block}


def _delta(index: int, delta: dict) -> dict:
    return {"type": "content_block_delta", "index": index, "delta": delta}


def _stop(index: int) -> dict:
    return {"type": "content_block_stop", "index": index}


# --- server_tool_use lifecycle ---------------------------------------------


def test_a_direct_search_streams_its_input_onto_the_private_block():
    s = _streamer()

    opened = s.handle(_start(0, {"type": "server_tool_use", "id": "s_1", "name": "web_search", "input": {}}))
    first = s.handle(_delta(0, {"type": "input_json_delta", "partial_json": ""}))
    second = s.handle(_delta(0, {"type": "input_json_delta", "partial_json": '{"query": "AAPL'}))
    third = s.handle(_delta(0, {"type": "input_json_delta", "partial_json": ' stock"}'}))
    stopped = s.handle(_stop(0))

    assert opened == [WebStartEvent(id="s_1")]
    assert first == []
    assert second == []
    assert third == []
    assert stopped == [WebSearchEvent(id="s_1", queries=["AAPL stock"])]
    assert s.message.content == [
        PrivateProviderBlock(
            format="anthropic.messages",
            data={"type": "server_tool_use", "id": "s_1", "name": "web_search", "input": {"query": "AAPL stock"}},
        )
    ]


def test_a_caller_mode_call_arrives_whole_with_no_deltas():
    s = _streamer()
    block = {
        "type": "server_tool_use",
        "id": "s_1",
        "name": "web_fetch",
        "input": {"url": "https://apple.com"},
        "caller": {"type": "code_execution_20260120", "tool_id": "s_0"},
    }

    opened = s.handle(_start(0, block))
    stopped = s.handle(_stop(0))

    assert opened == [WebStartEvent(id="s_1")]
    assert stopped == [WebFetchEvent(id="s_1", urls=["https://apple.com"])]
    assert s.message.content == [PrivateProviderBlock(format="anthropic.messages", data=block)]


def test_malformed_server_tool_input_degrades_instead_of_killing_the_stream():
    # A client tool call's malformed arguments are a StreamError (the caller
    # must execute them); a server tool's input is descriptive — the
    # provider already ran it — so it degrades to the start-time value.
    s = _streamer()
    s.handle(_start(0, {"type": "server_tool_use", "id": "s_1", "name": "web_search", "input": {}}))
    s.handle(_delta(0, {"type": "input_json_delta", "partial_json": '{"query": "trunca'}))

    stopped = s.handle(_stop(0))

    assert stopped == [WebSearchEvent(id="s_1", queries=[])]
    assert s.message.content == [
        PrivateProviderBlock(
            format="anthropic.messages",
            data={"type": "server_tool_use", "id": "s_1", "name": "web_search", "input": {}},
        )
    ]


def test_a_code_execution_call_is_private_only():
    s = _streamer()

    opened = s.handle(_start(0, {"type": "server_tool_use", "id": "s_0", "name": "code_execution", "input": {}}))
    s.handle(_delta(0, {"type": "input_json_delta", "partial_json": '{"code": "1+1"}'}))
    stopped = s.handle(_stop(0))

    assert opened == []
    assert stopped == []
    assert s.message.content == [
        PrivateProviderBlock(
            format="anthropic.messages",
            data={"type": "server_tool_use", "id": "s_0", "name": "code_execution", "input": {"code": "1+1"}},
        )
    ]


# --- result blocks ----------------------------------------------------------

_SEARCH_RESULT = {
    "type": "web_search_tool_result",
    "tool_use_id": "s_1",
    "content": [
        {
            "type": "web_search_result",
            "title": "Apple",
            "url": "https://apple.com",
            "encrypted_content": "opaque",
            "page_age": "2 days ago",
        }
    ],
    "caller": {"type": "direct"},
}


def test_a_search_result_completes_both_views():
    s = _streamer()
    s.handle(_start(0, {"type": "server_tool_use", "id": "s_1", "name": "web_search", "input": {}}))
    s.handle(_delta(0, {"type": "input_json_delta", "partial_json": '{"query": "apple"}'}))
    s.handle(_stop(0))

    arrived = s.handle(_start(1, _SEARCH_RESULT))
    stopped = s.handle(_stop(1))

    results = [WebPagePart(url="https://apple.com", title="Apple")]
    assert arrived == []
    assert stopped == [
        WebSearchResultEvent(id="s_1", results=results),
        WebEndEvent(id="s_1"),
    ]
    assert s.message.content == [
        PrivateProviderBlock(
            format="anthropic.messages",
            data={"type": "server_tool_use", "id": "s_1", "name": "web_search", "input": {"query": "apple"}},
        ),
        PrivateProviderBlock(format="anthropic.messages", data=_SEARCH_RESULT),
        WebSearchBlock(queries=["apple"], results=results, extras={"id": "s_1"}),
    ]


def test_an_errored_search_result_ends_without_a_result_event():
    s = _streamer()
    s.handle(_start(0, {"type": "server_tool_use", "id": "s_1", "name": "web_search", "input": {}}))
    s.handle(_delta(0, {"type": "input_json_delta", "partial_json": '{"query": "apple"}'}))
    s.handle(_stop(0))
    result = {
        "type": "web_search_tool_result",
        "tool_use_id": "s_1",
        "content": {"type": "web_search_tool_result_error", "error_code": "max_uses_exceeded"},
    }

    s.handle(_start(1, result))
    stopped = s.handle(_stop(1))

    assert stopped == [WebEndEvent(id="s_1")]
    assert s.message.content[1:] == [
        PrivateProviderBlock(format="anthropic.messages", data=result),
        WebSearchBlock(
            queries=["apple"],
            results=None,
            extras={
                "id": "s_1",
                "error": {"type": "web_search_tool_result_error", "error_code": "max_uses_exceeded"},
            },
        ),
    ]


def test_parallel_searches_pair_results_by_tool_use_id():
    # Batched calls stream call-call-result-result; each result pairs with
    # ITS call through web_calls, not with the adjacent block.
    s = _streamer()
    s.handle(_start(0, {"type": "server_tool_use", "id": "s_a", "name": "web_search", "input": {}}))
    s.handle(_delta(0, {"type": "input_json_delta", "partial_json": '{"query": "apple"}'}))
    s.handle(_stop(0))
    s.handle(_start(1, {"type": "server_tool_use", "id": "s_b", "name": "web_search", "input": {}}))
    s.handle(_delta(1, {"type": "input_json_delta", "partial_json": '{"query": "nvidia"}'}))
    s.handle(_stop(1))
    result_a = {"type": "web_search_tool_result", "tool_use_id": "s_a", "content": [], "caller": {"type": "direct"}}
    result_b = {"type": "web_search_tool_result", "tool_use_id": "s_b", "content": [], "caller": {"type": "direct"}}

    s.handle(_start(2, result_a))
    first = s.handle(_stop(2))
    s.handle(_start(3, result_b))
    second = s.handle(_stop(3))

    assert first == [WebEndEvent(id="s_a")]
    assert second == [WebEndEvent(id="s_b")]
    assert s.message.content == [
        PrivateProviderBlock(
            format="anthropic.messages",
            data={"type": "server_tool_use", "id": "s_a", "name": "web_search", "input": {"query": "apple"}},
        ),
        PrivateProviderBlock(
            format="anthropic.messages",
            data={"type": "server_tool_use", "id": "s_b", "name": "web_search", "input": {"query": "nvidia"}},
        ),
        PrivateProviderBlock(format="anthropic.messages", data=result_a),
        WebSearchBlock(queries=["apple"], results=[], extras={"id": "s_a"}),
        PrivateProviderBlock(format="anthropic.messages", data=result_b),
        WebSearchBlock(queries=["nvidia"], results=[], extras={"id": "s_b"}),
    ]


def test_a_fetch_result_builds_the_fetch_block():
    s = _streamer()
    call = {"type": "server_tool_use", "id": "f_1", "name": "web_fetch", "input": {"url": "https://apple.com"}}
    s.handle(_start(0, call))
    s.handle(_stop(0))
    result = {
        "type": "web_fetch_tool_result",
        "tool_use_id": "f_1",
        "content": {
            "type": "web_fetch_result",
            "url": "https://apple.com",
            "retrieved_at": "2026-08-15T03:41:54+00:00",
            "content": {
                "type": "document",
                "source": {"type": "text", "media_type": "text/plain", "data": "Apple today announced..."},
                "title": "Apple",
            },
        },
    }

    s.handle(_start(1, result))
    stopped = s.handle(_stop(1))

    assert stopped == [WebEndEvent(id="f_1")]
    assert s.message.content == [
        PrivateProviderBlock(format="anthropic.messages", data=call),
        PrivateProviderBlock(format="anthropic.messages", data=result),
        WebFetchBlock(
            web_page=WebPagePart(url="https://apple.com", title="Apple", content="Apple today announced..."),
            extras={"id": "f_1"},
        ),
    ]


def test_a_code_execution_result_is_private_only():
    s = _streamer()
    s.handle(_start(0, {"type": "server_tool_use", "id": "s_0", "name": "code_execution", "input": {}}))
    s.handle(_stop(0))
    result = {
        "type": "code_execution_tool_result",
        "tool_use_id": "s_0",
        "content": {"type": "code_execution_result", "stdout": "ok", "stderr": "", "return_code": 0, "content": []},
    }

    s.handle(_start(1, result))
    stopped = s.handle(_stop(1))

    assert stopped == []
    assert s.message.content[1] == PrivateProviderBlock(format="anthropic.messages", data=result)
    assert len(s.message.content) == 2


# --- text runs and citations ------------------------------------------------


def test_adjacent_text_blocks_stream_as_one_run():
    s = _streamer()

    first_open = s.handle(_start(0, {"type": "text", "text": ""}))
    s.handle(_delta(0, {"type": "text_delta", "text": "Market update: "}))
    first_stop = s.handle(_stop(0))
    second_open = s.handle(_start(1, {"citations": [], "type": "text", "text": ""}))
    citation = s.handle(
        _delta(
            1,
            {
                "type": "citations_delta",
                "citation": {
                    "type": "web_search_result_location",
                    "url": "https://example.com/apple",
                    "title": "Apple shares rise",
                    "cited_text": "Apple shares rose 2.8%...",
                    "encrypted_index": "opaque",
                },
            },
        )
    )
    delta = s.handle(_delta(1, {"type": "text_delta", "text": "Apple rose 2.8% today."}))
    second_stop = s.handle(_stop(1))

    annotation = URLCitationAnnotation(
        url="https://example.com/apple",
        title="Apple shares rise",
        start_index=15,
        end_index=37,
    )
    assert first_open == [TextStartEvent(index=0)]
    assert first_stop == []  # the run stays open — an adjacent block may continue it
    assert second_open == []  # same run, same index
    assert citation == []  # held until the span's stop fixes its range
    assert delta == [TextDeltaEvent(index=0, delta="Apple rose 2.8% today.")]
    assert second_stop == [TextAnnotationEvent(index=0, annotation=annotation)]
    assert s.message.content == [
        TextBlock(text="Market update: Apple rose 2.8% today.", annotations=[annotation]),
    ]


def test_a_cited_run_keeps_its_wire_blocks_when_it_closes():
    # Privates land at run close — the earliest point the run's cited-ness
    # and full structure are both known — after the merged TextBlock.
    s = _streamer()
    s.handle(_start(0, {"type": "text", "text": ""}))
    s.handle(_delta(0, {"type": "text_delta", "text": "Apple is "}))
    s.handle(_stop(0))
    s.handle(_start(1, {"citations": [], "type": "text", "text": ""}))
    citation = {"type": "web_search_result_location", "url": "https://a", "title": "A", "cited_text": "x"}
    s.handle(_delta(1, {"type": "citations_delta", "citation": citation}))
    s.handle(_delta(1, {"type": "text_delta", "text": "up."}))
    s.handle(_stop(1))

    events = s.handle({"type": "message_stop"})

    annotation = URLCitationAnnotation(url="https://a", title="A", start_index=9, end_index=12)
    assert events[0] == TextEndEvent(index=0, content="Apple is up.")
    assert s.message.content == [
        TextBlock(text="Apple is up.", annotations=[annotation]),
        PrivateProviderBlock(format="anthropic.messages", data={"type": "text", "text": "Apple is "}),
        PrivateProviderBlock(
            format="anthropic.messages",
            data={"citations": [citation], "type": "text", "text": "up."},
        ),
    ]


def test_citations_on_one_block_share_the_derived_range():
    s = _streamer()
    s.handle(_start(0, {"citations": [], "type": "text", "text": ""}))
    s.handle(
        _delta(
            0,
            {
                "type": "citations_delta",
                "citation": {"type": "web_search_result_location", "url": "https://a", "title": "A", "cited_text": "x"},
            },
        )
    )
    s.handle(
        _delta(
            0,
            {
                "type": "citations_delta",
                "citation": {"type": "web_search_result_location", "url": "https://b", "title": "B", "cited_text": "y"},
            },
        )
    )
    s.handle(_delta(0, {"type": "text_delta", "text": "Apple rose."}))

    stopped = s.handle(_stop(0))

    assert stopped == [
        TextAnnotationEvent(
            index=0, annotation=URLCitationAnnotation(url="https://a", title="A", start_index=0, end_index=11)
        ),
        TextAnnotationEvent(
            index=0, annotation=URLCitationAnnotation(url="https://b", title="B", start_index=0, end_index=11)
        ),
    ]


def test_a_non_text_block_closes_the_run_before_it_opens():
    s = _streamer()
    s.handle(_start(0, {"type": "text", "text": ""}))
    s.handle(_delta(0, {"type": "text_delta", "text": "So far."}))
    s.handle(_stop(0))

    events = s.handle(_start(1, {"type": "thinking", "thinking": ""}))

    assert events == [TextEndEvent(index=0, content="So far."), ThinkingStartEvent(index=1)]


def test_message_stop_closes_the_run_before_usage_and_finish():
    s = _streamer()
    s.handle(_start(0, {"type": "text", "text": ""}))
    s.handle(_delta(0, {"type": "text_delta", "text": "Done."}))
    s.handle(_stop(0))
    s.handle(
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {
                "input_tokens": 140,
                "output_tokens": 20,
                "server_tool_use": {"web_search_requests": 1, "web_fetch_requests": 0},
            },
        }
    )

    events = s.handle({"type": "message_stop"})

    usage = Usage(
        # message_delta's input count is authoritative — server tools re-bill
        # input while they loop, so message_start's 5 is stale.
        input_tokens=140,
        output_tokens=20,
        total_tokens=160,
        tool_requests={"web_search": 1, "web_fetch": 0},
        provider_tool_usage={"web_search_requests": 1, "web_fetch_requests": 0},
    )
    assert events[0] == TextEndEvent(index=0, content="Done.")
    assert events[1] == UsageEvent(usage=usage)
    assert events[2].type == "finish"
    assert events[2].usage == usage
    assert events[2].message.content == [TextBlock(text="Done.")]


def test_a_full_search_stream_narrates_the_operation(anthropic_transport_factory):
    from tests.client._helpers.httpx_mocks import make_sync_client, sse_response

    call = {"type": "server_tool_use", "id": "s_1", "name": "web_search", "input": {"query": "apple"}}
    result = {
        "type": "web_search_tool_result",
        "tool_use_id": "s_1",
        "content": [
            {
                "type": "web_search_result",
                "title": "Apple",
                "url": "https://apple.com",
                "encrypted_content": "opaque",
                "page_age": None,
            }
        ],
        "caller": {"type": "direct"},
    }
    chunks = [
        b'data: {"type":"message_start","message":{"id":"msg_1","usage":{"input_tokens":5,"output_tokens":0}}}\n\n',
        b'data: {"type":"content_block_start","index":0,'
        b'"content_block":{"type":"server_tool_use","id":"s_1","name":"web_search","input":{}}}\n\n',
        b'data: {"type":"content_block_delta","index":0,'
        b'"delta":{"type":"input_json_delta","partial_json":"{\\"query\\": \\"apple\\"}"}}\n\n',
        b'data: {"type":"content_block_stop","index":0}\n\n',
        b'data: {"type":"content_block_start","index":1,"content_block":{"type":"web_search_tool_result",'
        b'"tool_use_id":"s_1","content":[{"type":"web_search_result","title":"Apple","url":"https://apple.com",'
        b'"encrypted_content":"opaque","page_age":null}],"caller":{"type":"direct"}}}\n\n',
        b'data: {"type":"content_block_stop","index":1}\n\n',
        b'data: {"type":"content_block_start","index":2,"content_block":{"type":"text","text":""}}\n\n',
        b'data: {"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":"Apple is "}}\n\n',
        b'data: {"type":"content_block_stop","index":2}\n\n',
        b'data: {"type":"content_block_start","index":3,"content_block":{"citations":[],"type":"text","text":""}}\n\n',
        b'data: {"type":"content_block_delta","index":3,"delta":{"type":"citations_delta",'
        b'"citation":{"type":"web_search_result_location","url":"https://apple.com","title":"Apple",'
        b'"cited_text":"up 2.8%","encrypted_index":"opaque"}}}\n\n',
        b'data: {"type":"content_block_delta","index":3,"delta":{"type":"text_delta","text":"up."}}\n\n',
        b'data: {"type":"content_block_stop","index":3}\n\n',
        b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":140,'
        b'"output_tokens":20,"server_tool_use":{"web_search_requests":1,"web_fetch_requests":0}}}\n\n',
        b'data: {"type":"message_stop"}\n\n',
    ]
    transport = anthropic_transport_factory(http_client=make_sync_client(sse_response(chunks)))

    with transport.completion_stream(REQUEST) as s:
        events = list(s)

    annotation = URLCitationAnnotation(url="https://apple.com", title="Apple", start_index=9, end_index=12)
    results = [WebPagePart(url="https://apple.com", title="Apple")]
    usage = Usage(
        input_tokens=140,
        output_tokens=20,
        total_tokens=160,
        tool_requests={"web_search": 1, "web_fetch": 0},
        provider_tool_usage={"web_search_requests": 1, "web_fetch_requests": 0},
    )
    assert [e.type for e in events] == [
        "start",
        "web_start",
        "web_search",
        "web_search_result",
        "web_end",
        "text_start",
        "text_delta",
        "text_delta",
        "text_annotation",
        "text_end",
        "usage",
        "finish",
    ]
    assert events[1:5] == [
        WebStartEvent(id="s_1"),
        WebSearchEvent(id="s_1", queries=["apple"]),
        WebSearchResultEvent(id="s_1", results=results),
        WebEndEvent(id="s_1"),
    ]
    assert events[8] == TextAnnotationEvent(index=3, annotation=annotation)
    assert events[9] == TextEndEvent(index=3, content="Apple is up.")
    assert events[10] == UsageEvent(usage=usage)
    # The cited run keeps its rebuilt wire blocks as private blocks — the
    # same two views the non-streaming parse stores.
    assert events[-1].message.content == [
        PrivateProviderBlock(format="anthropic.messages", data=call),
        PrivateProviderBlock(format="anthropic.messages", data=result),
        WebSearchBlock(queries=["apple"], results=results, extras={"id": "s_1"}),
        TextBlock(text="Apple is up.", annotations=[annotation]),
        PrivateProviderBlock(format="anthropic.messages", data={"type": "text", "text": "Apple is "}),
        PrivateProviderBlock(
            format="anthropic.messages",
            data={
                "citations": [
                    {
                        "type": "web_search_result_location",
                        "url": "https://apple.com",
                        "title": "Apple",
                        "cited_text": "up 2.8%",
                        "encrypted_index": "opaque",
                    }
                ],
                "type": "text",
                "text": "up.",
            },
        ),
    ]


def test_start_still_precedes_everything_through_the_public_gate():
    # `handle()` returns raw handler events; the emit() gate owns the
    # start-first contract, so a web event before message_start still
    # narrates start → web_start when pulled through the iterator.
    s = SyncAnthropicStreamer(REQUEST)

    events = s.handle(_start(0, {"type": "server_tool_use", "id": "s_1", "name": "web_search", "input": {}}))

    assert events == [WebStartEvent(id="s_1")]
    assert s.emit(events[0]) == StartEvent()
    assert s.pending == [WebStartEvent(id="s_1")]


# --- extras stamping, streamed (same mixin builders as the sync parse) ------


def test_a_streamed_search_result_stamps_the_operation_id():
    s = _streamer()
    s.handle(_start(0, {"type": "server_tool_use", "id": "s_1", "name": "web_search", "input": {}}))
    s.handle(_delta(0, {"type": "input_json_delta", "partial_json": '{"query": "apple"}'}))
    s.handle(_stop(0))

    s.handle(_start(1, _SEARCH_RESULT))
    s.handle(_stop(1))

    assert s.message.content[2] == WebSearchBlock(
        queries=["apple"],
        results=[WebPagePart(url="https://apple.com", title="Apple")],
        extras={"id": "s_1"},
    )


def test_a_streamed_failed_search_stamps_the_provider_error():
    s = _streamer()
    s.handle(_start(0, {"type": "server_tool_use", "id": "s_1", "name": "web_search", "input": {}}))
    s.handle(_delta(0, {"type": "input_json_delta", "partial_json": '{"query": "apple"}'}))
    s.handle(_stop(0))
    result = {
        "type": "web_search_tool_result",
        "tool_use_id": "s_1",
        "content": {"type": "web_search_tool_result_error", "error_code": "max_uses_exceeded"},
    }

    s.handle(_start(1, result))
    s.handle(_stop(1))

    assert s.message.content[2] == WebSearchBlock(
        queries=["apple"],
        results=None,
        extras={
            "id": "s_1",
            "error": {"type": "web_search_tool_result_error", "error_code": "max_uses_exceeded"},
        },
    )


def test_a_streamed_failed_fetch_stamps_the_provider_error():
    s = _streamer()
    call = {"type": "server_tool_use", "id": "f_1", "name": "web_fetch", "input": {"url": "https://apple.com"}}
    s.handle(_start(0, call))
    s.handle(_stop(0))
    result = {
        "type": "web_fetch_tool_result",
        "tool_use_id": "f_1",
        "content": {"type": "web_fetch_tool_result_error", "error_code": "unavailable"},
    }

    s.handle(_start(1, result))
    s.handle(_stop(1))

    assert s.message.content[2] == WebFetchBlock(
        web_page=WebPagePart(url="https://apple.com"),
        extras={
            "id": "f_1",
            "error": {"type": "web_fetch_tool_result_error", "error_code": "unavailable"},
        },
    )
