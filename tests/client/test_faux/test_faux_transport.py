"""FauxTransport — scripted responses, error injection, streaming."""

import threading

import pytest

from luca.client.transports.faux import (
    FauxTransport,
    faux_assistant_message,
    faux_error,
    faux_text,
    faux_thinking,
    faux_tool_call,
)
from luca.client.types import (
    ChatCompletionRequest,
    UserMessage,
)


def _req(content: str = "hi") -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test-model",
        messages=[UserMessage(content=content)],
    )


def test_single_scripted_response():
    faux = FauxTransport()
    faux.set_responses(
        [
            faux_assistant_message([faux_text("hello")], finish_reason="stop"),
        ]
    )
    response = faux.completion(_req())
    assert response.messages[-1].finish_reason == "stop"
    assert response.messages[-1].content[0].text == "hello"


def test_native_tools_pass_through_untouched():
    # Faux never projects tools, so a native tool rides along unharmed.
    from luca.client.providers.openai import ApplyPatchTool

    faux = FauxTransport()
    faux.set_responses([faux_assistant_message([faux_text("ok")], finish_reason="stop")])
    native = ApplyPatchTool()
    request = _req()
    request.tools = [native]
    response = faux.completion(request)
    assert response.messages[-1].finish_reason == "stop"
    assert request.tools == [native]


def test_multiple_scripted_responses_in_order():
    faux = FauxTransport()
    faux.set_responses(
        [
            faux_assistant_message([faux_text("first")], finish_reason="stop"),
            faux_assistant_message([faux_text("second")], finish_reason="stop"),
        ]
    )
    assert faux.completion(_req()).messages[-1].content[0].text == "first"
    assert faux.completion(_req()).messages[-1].content[0].text == "second"


def test_exhausted_queue_raises():
    faux = FauxTransport()
    faux.set_responses([faux_assistant_message([faux_text("once")], finish_reason="stop")])
    faux.completion(_req())
    with pytest.raises(RuntimeError, match="no more"):
        faux.completion(_req())


def test_thinking_plus_tool_call_response():
    faux = FauxTransport()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_thinking("Let me check..."), faux_tool_call("get_weather", {"city": "Paris"})],
                finish_reason="tool_use",
            ),
        ]
    )
    response = faux.completion(_req("Weather?"))
    assert response.messages[-1].finish_reason == "tool_use"
    assert response.messages[-1].tool_calls[0].name == "get_weather"
    assert response.messages[-1].tool_calls[0].arguments == {"city": "Paris"}


def test_error_injection_raises_on_completion():
    faux = FauxTransport()
    faux.set_responses(
        [
            faux_assistant_message([faux_text("partial")], error=faux_error("upstream timeout")),
        ]
    )
    with pytest.raises(Exception, match="upstream timeout"):
        faux.completion(_req())


def test_concurrent_completions_consume_distinct_responses():
    n = 8
    faux = FauxTransport()
    faux.set_responses([faux_assistant_message([faux_text(f"r-{i}")], finish_reason="stop") for i in range(n)])
    seen, lock = [], threading.Lock()

    def worker():
        r = faux.completion(_req())
        with lock:
            seen.append(r.messages[-1].content[0].text)

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(seen) == sorted(f"r-{i}" for i in range(n))


def test_streaming_yields_text_deltas_and_terminal_finish():
    faux = FauxTransport()
    faux.set_responses(
        [
            faux_assistant_message([faux_text("Hello world")], finish_reason="stop"),
        ]
    )
    with faux.completion_stream(_req()) as s:
        events = list(s)
    assert events[0].type == "start"
    assert events[-1].type == "finish"
    assert events[-1].finish_reason == "stop"


def test_streaming_error_injection_emits_error_event():
    faux = FauxTransport()
    faux.set_responses(
        [
            faux_assistant_message(
                [faux_text("partial")],
                error=faux_error("upstream timeout"),
            ),
        ]
    )
    with faux.completion_stream(_req()) as s:
        events = list(s)
    assert events[-1].type == "error"
    assert "upstream timeout" in str(events[-1].error)
    assert s.message.content[0].text == "partial"


# --- scripted web content (phase-5 faux extension) --------------------------


def test_scripted_web_blocks_play_back_verbatim():
    # The three real client blocks pass straight through — both modes — so a
    # runner-level test can script a whole web-bearing response.
    from luca.client.types import (
        PrivateProviderBlock,
        TextBlock,
        WebFetchBlock,
        WebPagePart,
        WebSearchBlock,
    )

    blocks = [
        PrivateProviderBlock(format="anthropic.messages", data={"type": "server_tool_use", "id": "s_1"}),
        WebSearchBlock(
            queries=["apple"],
            results=[WebPagePart(url="https://apple.com", title="Apple")],
            extras={"id": "s_1"},
        ),
        WebFetchBlock(web_page=WebPagePart(url="https://apple.com"), extras={"id": "f_1"}),
        faux_text("Done."),
    ]
    faux = FauxTransport()
    faux.set_responses(
        [
            faux_assistant_message(list(blocks), finish_reason="stop"),
            faux_assistant_message(list(blocks), finish_reason="stop"),
        ]
    )

    sync_content = faux.completion(_req()).messages[-1].content
    with faux.completion_stream(_req()) as s:
        list(s)
        streamed_content = s.message.content

    expected = [
        PrivateProviderBlock(format="anthropic.messages", data={"type": "server_tool_use", "id": "s_1"}),
        WebSearchBlock(
            queries=["apple"],
            results=[WebPagePart(url="https://apple.com", title="Apple")],
            extras={"id": "s_1"},
        ),
        WebFetchBlock(web_page=WebPagePart(url="https://apple.com"), extras={"id": "f_1"}),
        TextBlock(text="Done."),
    ]
    assert sync_content == expected
    assert streamed_content == expected


def test_a_scripted_web_search_streams_the_web_event_sequence():
    from luca.client.types import PrivateProviderBlock, WebPagePart, WebSearchBlock
    from luca.client.types.streaming import (
        FinishEvent,
        StartEvent,
        TextDeltaEvent,
        TextEndEvent,
        TextStartEvent,
        WebEndEvent,
        WebSearchEvent,
        WebSearchResultEvent,
        WebStartEvent,
    )

    results = [WebPagePart(url="https://apple.com", title="Apple")]
    faux = FauxTransport()
    faux.set_responses(
        [
            faux_assistant_message(
                [
                    PrivateProviderBlock(format="anthropic.messages", data={"type": "server_tool_use", "id": "s_1"}),
                    WebSearchBlock(queries=["apple"], results=results, extras={"id": "s_1"}),
                    faux_text("Done."),
                ],
                finish_reason="stop",
            ),
        ]
    )

    with faux.completion_stream(_req()) as s:
        events = list(s)

    assert [type(e) for e in events] == [
        StartEvent,
        WebStartEvent,
        WebSearchEvent,
        WebSearchResultEvent,
        WebEndEvent,
        TextStartEvent,
        TextDeltaEvent,
        TextEndEvent,
        FinishEvent,
    ]
    assert events[1] == WebStartEvent(id="s_1")
    assert events[2] == WebSearchEvent(id="s_1", queries=["apple"])
    assert events[3] == WebSearchResultEvent(id="s_1", results=results)
    assert events[4] == WebEndEvent(id="s_1")


def test_scripted_annotations_survive_and_stream():
    from luca.client.types import TextBlock, URLCitationAnnotation
    from luca.client.types.streaming import TextAnnotationEvent

    annotation = URLCitationAnnotation(url="https://apple.com", title="Apple", start_index=0, end_index=12)
    faux = FauxTransport()
    faux.set_responses(
        [
            faux_assistant_message([faux_text("Apple is up.", annotations=[annotation])], finish_reason="stop"),
            faux_assistant_message([faux_text("Apple is up.", annotations=[annotation])], finish_reason="stop"),
        ]
    )

    sync_content = faux.completion(_req()).messages[-1].content
    with faux.completion_stream(_req()) as s:
        events = list(s)
        streamed_content = s.message.content

    assert sync_content == [TextBlock(text="Apple is up.", annotations=[annotation])]
    assert streamed_content == [TextBlock(text="Apple is up.", annotations=[annotation])]
    assert TextAnnotationEvent(index=0, annotation=annotation) in events


def test_an_empty_scripted_result_set_streams_no_result_event():
    # the real streamers suppress the results event for an empty list; the
    # faux must play back the same shape
    from luca.client.types import WebSearchBlock
    from luca.client.types.streaming import WebEndEvent, WebSearchEvent, WebStartEvent

    faux = FauxTransport()
    faux.set_responses(
        [
            faux_assistant_message(
                [WebSearchBlock(queries=["nothing"], results=[], extras={"id": "s_1"})],
                finish_reason="stop",
            ),
        ]
    )

    with faux.completion_stream(_req()) as s:
        events = list(s)

    web_events = [e for e in events if e.type.startswith("web")]
    assert web_events == [
        WebStartEvent(id="s_1"),
        WebSearchEvent(id="s_1", queries=["nothing"]),
        WebEndEvent(id="s_1"),
    ]
