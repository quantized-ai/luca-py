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
    TextBlock,
    UserMessage,
)


def _req(content: str = "hi") -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test-model",
        messages=[UserMessage(content=content)],
    )


def test_single_scripted_response():
    faux = FauxTransport()
    faux.set_responses([
        faux_assistant_message([faux_text("hello")], finish_reason="stop"),
    ])
    response = faux.completion(_req())
    assert response.finish_reason == "stop"
    assert response.message.content[0].text == "hello"


def test_multiple_scripted_responses_in_order():
    faux = FauxTransport()
    faux.set_responses([
        faux_assistant_message([faux_text("first")], finish_reason="stop"),
        faux_assistant_message([faux_text("second")], finish_reason="stop"),
    ])
    assert faux.completion(_req()).message.content[0].text == "first"
    assert faux.completion(_req()).message.content[0].text == "second"


def test_exhausted_queue_raises():
    faux = FauxTransport()
    faux.set_responses([faux_assistant_message([faux_text("once")], finish_reason="stop")])
    faux.completion(_req())
    with pytest.raises(RuntimeError, match="no more"):
        faux.completion(_req())


def test_thinking_plus_tool_call_response():
    faux = FauxTransport()
    faux.set_responses([
        faux_assistant_message(
            [faux_thinking("Let me check..."),
             faux_tool_call("get_weather", {"city": "Paris"})],
            finish_reason="tool_use",
        ),
    ])
    response = faux.completion(_req("Weather?"))
    assert response.finish_reason == "tool_use"
    assert response.tool_calls[0].name == "get_weather"
    assert response.tool_calls[0].arguments == {"city": "Paris"}


def test_error_injection_raises_on_completion():
    faux = FauxTransport()
    faux.set_responses([
        faux_assistant_message([faux_text("partial")],
                                error=faux_error("upstream timeout")),
    ])
    with pytest.raises(Exception, match="upstream timeout"):
        faux.completion(_req())


def test_concurrent_completions_consume_distinct_responses():
    n = 8
    faux = FauxTransport()
    faux.set_responses([
        faux_assistant_message([faux_text(f"r-{i}")], finish_reason="stop")
        for i in range(n)
    ])
    seen, lock = [], threading.Lock()

    def worker():
        r = faux.completion(_req())
        with lock:
            seen.append(r.message.content[0].text)

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert sorted(seen) == sorted(f"r-{i}" for i in range(n))


def test_streaming_yields_text_deltas_and_terminal_finish():
    faux = FauxTransport()
    faux.set_responses([
        faux_assistant_message([faux_text("Hello world")], finish_reason="stop"),
    ])
    with faux.completion_stream(_req()) as s:
        events = list(s)
    assert events[0].type == "start"
    assert events[-1].type == "finish"
    assert events[-1].finish_reason == "stop"


def test_streaming_error_injection_emits_error_event():
    faux = FauxTransport()
    faux.set_responses([
        faux_assistant_message(
            [faux_text("partial")],
            error=faux_error("upstream timeout"),
        ),
    ])
    with faux.completion_stream(_req()) as s:
        events = list(s)
    assert events[-1].type == "error"
    assert "upstream timeout" in str(events[-1].error)
    assert events[-1].partial_message.content[0].text == "partial"
