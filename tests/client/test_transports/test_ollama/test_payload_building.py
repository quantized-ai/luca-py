"""What goes on the wire to `/api/chat`.

`num_ctx` is the reason this transport exists, so most of this file is about
where that number comes from and what may override it.
"""

from dataclasses import dataclass

import pytest

from luca.client.types import (
    AssistantMessage,
    ChatCompletionRequest,
    ModelInfo,
    TextBlock,
    Tool,
    ToolCall,
    ToolMessage,
    UserMessage,
)

TOOL = Tool(
    name="get_weather",
    description="Weather for a city",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
)


def _build(transport, *, stream=False, **kwargs):
    return transport._build_chat_completion_payload(ChatCompletionRequest(provider="ollama", **kwargs), stream=stream)


# ── num_ctx ──────────────────────────────────────────────────────────────────


def test_the_window_comes_from_the_catalog_record(ollama_transport_factory, model_info):
    # Discovery registered 32768; the transport asks Ollama for exactly that,
    # which is what keeps the reported window and the real one the same number.
    payload = _build(
        ollama_transport_factory(),
        model="qwen2.5:14b",
        messages=[UserMessage(content="Hi")],
        model_info=model_info,
    )

    assert payload["options"]["num_ctx"] == 32_768


def test_a_window_above_the_ceiling_is_capped(ollama_transport_factory):
    # llama3.2 advertises 131072; allocating that on a laptop either fails to
    # load or spills to CPU.
    payload = _build(
        ollama_transport_factory(),
        model="llama3.2:latest",
        messages=[UserMessage(content="Hi")],
        model_info=ModelInfo(model="llama3.2:latest", provider="ollama", context_window=131_072),
    )

    assert payload["options"]["num_ctx"] == 32_768


def test_provider_options_override_the_window(ollama_transport_factory, model_info):
    payload = _build(
        ollama_transport_factory(),
        model="qwen2.5:14b",
        messages=[UserMessage(content="Hi")],
        model_info=model_info,
        provider_options={"ollama": {"options": {"num_ctx": 8_192}}},
    )

    assert payload["options"]["num_ctx"] == 8_192


def test_an_unknown_model_asks_for_no_window_at_all(ollama_transport_factory):
    # Better to let Ollama pick than to invent a number: an unregistered model
    # is one discovery never saw.
    payload = _build(
        ollama_transport_factory(),
        model="mystery:latest",
        messages=[UserMessage(content="Hi")],
    )

    assert "options" not in payload


# ── the rest of the payload ──────────────────────────────────────────────────


def test_stream_is_always_explicit(ollama_transport_factory):
    # Ollama streams by DEFAULT. Omitting the field on a non-streaming call
    # returns NDJSON to a parser expecting one object.
    transport = ollama_transport_factory()
    kwargs = {"model": "m", "messages": [UserMessage(content="Hi")]}

    assert _build(transport, **kwargs)["stream"] is False
    assert _build(transport, stream=True, **kwargs)["stream"] is True


def test_the_system_prompt_is_a_message_not_a_field(ollama_transport_factory):
    payload = _build(
        ollama_transport_factory(),
        model="m",
        messages=[UserMessage(content="Hi")],
        system_message="Be brief.",
    )

    assert payload["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
    ]


def test_sampling_settings_go_under_options(ollama_transport_factory):
    payload = _build(
        ollama_transport_factory(),
        model="m",
        messages=[UserMessage(content="Hi")],
        temperature=0.2,
        top_p=0.9,
        max_tokens=256,
    )

    # `max_tokens` is `num_predict` here, and none of these are top-level.
    assert payload["options"] == {"temperature": 0.2, "top_p": 0.9, "num_predict": 256}


def test_tool_arguments_project_as_a_json_object(ollama_transport_factory):
    payload = _build(
        ollama_transport_factory(),
        model="m",
        messages=[
            UserMessage(content="weather?"),
            AssistantMessage(
                content=[ToolCall(id="call_1", name="get_weather", arguments={"city": "Paris"})],
                provider="ollama",
                model="m",
            ),
            ToolMessage(tool_call_id="call_1", name="get_weather", content="18C"),
        ],
        tools=[TOOL],
    )

    # An object, not a serialised string — this wire is like Bedrock Converse,
    # not like OpenAI.
    assert payload["messages"][1]["tool_calls"] == [
        {"id": "call_1", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
    ]
    # Verified live: the follow-up turn correlates on `tool_name`.
    assert payload["messages"][2] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "18C",
        "name": "get_weather",
        "tool_name": "get_weather",
    }


@dataclass(frozen=True)
class ThinkCase:
    name: str
    supports_reasoning: bool
    reasoning: str | None
    expected: bool


THINK_CASES = [
    ThinkCase("asked_and_supported", True, "high", True),
    ThinkCase("asked_but_unsupported", False, "high", False),
    ThinkCase("not_asked", True, None, False),
    ThinkCase("provider_default", True, "provider-default", False),
]


@pytest.mark.parametrize("case", THINK_CASES, ids=lambda c: c.name)
def test_think_is_sent_only_when_asked_for_and_supported(ollama_transport_factory, case):
    # Asking a model that does not advertise `thinking` is a 400.
    payload = _build(
        ollama_transport_factory(),
        model="m",
        messages=[UserMessage(content="Hi")],
        reasoning=case.reasoning,
        model_info=ModelInfo(model="m", provider="ollama", supports_reasoning=case.supports_reasoning),
    )

    assert payload.get("think", False) is case.expected


def test_the_url_is_the_native_chat_endpoint(ollama_transport_factory):
    transport = ollama_transport_factory()
    request = ChatCompletionRequest(provider="ollama", model="m", messages=[UserMessage(content="Hi")])

    assert transport._chat_completion_url(request) == "http://localhost:11434/api/chat"
    assert transport._chat_completion_url(request, stream=True) == "http://localhost:11434/api/chat"


def test_an_assistant_text_message_replays_as_text(ollama_transport_factory):
    payload = _build(
        ollama_transport_factory(),
        model="m",
        messages=[
            UserMessage(content="Hi"),
            AssistantMessage(content=[TextBlock(text="Hello")], provider="ollama", model="m"),
        ],
    )

    assert payload["messages"][1] == {"role": "assistant", "content": "Hello"}
