"""What OpenAIResponsesTransport sends on the wire."""

import json as _json
from dataclasses import dataclass

import httpx
import pytest

from luca.client.exceptions import BadRequestError, UnsupportedParameterError
from luca.client.types import (
    AssistantMessage,
    ChatCompletionRequest,
    ImageBlock,
    MediaBase64,
    MediaFileId,
    MediaURL,
    TextBlock,
    ThinkingBlock,
    Tool,
    ToolCall,
    ToolMessage,
    UserMessage,
)

_EMPTY_RESPONSE = {
    "id": "resp_x",
    "model": "gpt-5.4",
    "status": "completed",
    "output": [],
    "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
}


@dataclass(frozen=True)
class PayloadCase:
    name: str
    request: ChatCompletionRequest
    expected_url: str
    expected_body: dict
    expected_auth: str


CASES = [
    PayloadCase(
        name="a_plain_turn_is_stateless_with_one_input_item",
        request=ChatCompletionRequest(
            model="gpt-5.4",
            provider="openai",
            messages=[UserMessage(content="Hello")],
        ),
        expected_url="https://api.openai.com/v1/responses",
        expected_body={
            "model": "gpt-5.4",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}],
            "store": False,
        },
        expected_auth="Bearer sk-test",
    ),
    PayloadCase(
        name="the_system_prompt_becomes_instructions",
        request=ChatCompletionRequest(
            model="gpt-5.4",
            provider="openai",
            messages=[UserMessage(content="Hello")],
            system_message="You are concise.",
        ),
        expected_url="https://api.openai.com/v1/responses",
        expected_body={
            "model": "gpt-5.4",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}],
            "store": False,
            "instructions": "You are concise.",
        },
        expected_auth="Bearer sk-test",
    ),
    PayloadCase(
        name="sampling_kwargs_use_the_responses_names",
        request=ChatCompletionRequest(
            model="gpt-5.4",
            provider="openai",
            messages=[UserMessage(content="Hi")],
            temperature=0.5,
            top_p=0.9,
            max_tokens=100,
            parallel_tool_calls=False,
            user="u-1",
        ),
        expected_url="https://api.openai.com/v1/responses",
        expected_body={
            "model": "gpt-5.4",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hi"}]}],
            "store": False,
            "temperature": 0.5,
            "top_p": 0.9,
            "max_output_tokens": 100,
            "parallel_tool_calls": False,
            "user": "u-1",
        },
        expected_auth="Bearer sk-test",
    ),
    PayloadCase(
        name="reasoning_asks_for_a_summary_and_the_encrypted_payload",
        request=ChatCompletionRequest(
            model="gpt-5.4",
            provider="openai",
            messages=[UserMessage(content="Hi")],
            reasoning="high",
        ),
        expected_url="https://api.openai.com/v1/responses",
        expected_body={
            "model": "gpt-5.4",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hi"}]}],
            "store": False,
            # No `context`: only the newest family accepts `all_turns` and
            # every earlier reasoning model 400s on it.
            "reasoning": {"effort": "high", "summary": "auto"},
            "include": ["reasoning.encrypted_content"],
        },
        expected_auth="Bearer sk-test",
    ),
    PayloadCase(
        name="provider_default_reasoning_sends_no_reasoning_key",
        request=ChatCompletionRequest(
            model="gpt-5.4",
            provider="openai",
            messages=[UserMessage(content="Hi")],
            reasoning="provider-default",
        ),
        expected_url="https://api.openai.com/v1/responses",
        expected_body={
            "model": "gpt-5.4",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hi"}]}],
            "store": False,
        },
        expected_auth="Bearer sk-test",
    ),
    PayloadCase(
        name="tools_are_flat_with_no_function_envelope",
        request=ChatCompletionRequest(
            model="gpt-5.4",
            provider="openai",
            messages=[UserMessage(content="Weather?")],
            tools=[
                {
                    "name": "get_weather",
                    "description": "Look up weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
            tool_choice={"name": "get_weather"},
        ),
        expected_url="https://api.openai.com/v1/responses",
        expected_body={
            "model": "gpt-5.4",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Weather?"}]}],
            "store": False,
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Look up weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
            "tool_choice": {"type": "function", "name": "get_weather"},
        },
        expected_auth="Bearer sk-test",
    ),
    PayloadCase(
        name="one_assistant_turn_expands_to_reasoning_message_and_function_call_items",
        request=ChatCompletionRequest(
            model="gpt-5.4",
            provider="openai",
            messages=[
                UserMessage(content="Weather?"),
                AssistantMessage(
                    content=[
                        ThinkingBlock(text="Look it up.", id="rs_1", signature="enc-1"),
                        TextBlock(text="Checking."),
                        ToolCall(id="call_1", name="get_weather", arguments={"city": "NYC"}),
                    ],
                    provider="openai",
                    model="gpt-5.4",
                ),
                ToolMessage(tool_call_id="call_1", content="sunny"),
            ],
        ),
        expected_url="https://api.openai.com/v1/responses",
        expected_body={
            "model": "gpt-5.4",
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "Weather?"}]},
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "encrypted_content": "enc-1",
                    "summary": [{"type": "summary_text", "text": "Look it up."}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Checking."}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": '{"city": "NYC"}',
                },
                {"type": "function_call_output", "call_id": "call_1", "output": "sunny"},
            ],
            "store": False,
        },
        expected_auth="Bearer sk-test",
    ),
    PayloadCase(
        name="provider_options_override_anything_derived",
        request=ChatCompletionRequest(
            model="gpt-5.4",
            provider="openai",
            messages=[UserMessage(content="Hi")],
            provider_options={"openai": {"store": True, "mine": 1}, "anthropic": {"theirs": 2}},
        ),
        expected_url="https://api.openai.com/v1/responses",
        expected_body={
            "model": "gpt-5.4",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hi"}]}],
            "store": True,
            "mine": 1,
        },
        expected_auth="Bearer sk-test",
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_responses_transport_outbound_payload(case, responses_transport_factory):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = _json.loads(request.content)
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=_EMPTY_RESPONSE)

    transport = responses_transport_factory(
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    transport.completion(case.request)

    assert captured == {
        "url": case.expected_url,
        "body": case.expected_body,
        "auth": case.expected_auth,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stop", ["END"]),
        ("seed", 42),
        ("presence_penalty", 0.5),
        ("frequency_penalty", 0.5),
        ("logprobs", True),
    ],
)
def test_a_chat_completions_only_parameter_is_refused(field, value, responses_transport_factory):
    # The Responses API has no equivalent. Dropping it would change the output
    # with nothing to notice.
    transport = responses_transport_factory()

    with pytest.raises(UnsupportedParameterError, match=field):
        transport._build_chat_completion_payload(
            ChatCompletionRequest(
                model="gpt-5.4",
                messages=[UserMessage(content="hi")],
                **{field: value},
            ),
        )


def test_every_offending_parameter_is_named_at_once(responses_transport_factory):
    transport = responses_transport_factory()

    with pytest.raises(UnsupportedParameterError, match="stop, seed"):
        transport._build_chat_completion_payload(
            ChatCompletionRequest(
                model="gpt-5.4",
                messages=[UserMessage(content="hi")],
                stop=["END"],
                seed=1,
            ),
        )


def test_image_sources_project_to_input_image(responses_transport_factory):
    transport = responses_transport_factory()

    assert transport._project_user_block(
        ImageBlock(source=MediaURL(url="https://example.com/a.png")),
    ) == {"type": "input_image", "image_url": "https://example.com/a.png"}
    assert transport._project_user_block(
        ImageBlock(source=MediaBase64(data="aGk=", media_type="image/png")),
    ) == {"type": "input_image", "image_url": "data:image/png;base64,aGk="}
    # Unlike chat completions, this endpoint takes an uploaded file by id.
    assert transport._project_user_block(
        ImageBlock(source=MediaFileId(file_id="file-abc123")),
    ) == {"type": "input_image", "file_id": "file-abc123"}


def test_an_image_in_a_tool_result_is_refused(responses_transport_factory):
    # Dropping it would tell the model the call succeeded with nothing in it.
    transport = responses_transport_factory()

    with pytest.raises(BadRequestError, match="only text in a function call output"):
        transport._project_tool_message(
            ToolMessage(
                tool_call_id="call_1",
                content=[
                    TextBlock(text="here"),
                    ImageBlock(source=MediaURL(url="https://example.com/a.png")),
                ],
            ),
        )


def test_structured_output_goes_flat_on_text_format(responses_transport_factory):
    # Flat, unlike chat completions, which nests it under `json_schema`.
    from pydantic import BaseModel

    class Movie(BaseModel):
        title: str
        year: int = 2024

    transport = responses_transport_factory()

    payload = transport._build_chat_completion_payload(
        ChatCompletionRequest(model="gpt-5.4", messages=[UserMessage(content="hi")], response_format=Movie),
    )

    assert payload["text"] == {
        "format": {
            "type": "json_schema",
            "name": "Movie",
            "schema": {
                "additionalProperties": False,
                "properties": {
                    "title": {"title": "Title", "type": "string"},
                    "year": {"default": 2024, "title": "Year", "type": "integer"},
                },
                "required": ["title", "year"],
                "title": "Movie",
                "type": "object",
            },
            "strict": True,
        }
    }


def test_a_dict_response_format_is_a_schema_not_a_wire_payload(responses_transport_factory):
    transport = responses_transport_factory()

    payload = transport._build_chat_completion_payload(
        ChatCompletionRequest(
            model="gpt-5.4",
            messages=[UserMessage(content="hi")],
            response_format={"type": "object", "properties": {"a": {"type": "string"}}},
        ),
    )

    assert payload["text"]["format"]["name"] == "structured_output"
    assert payload["text"]["format"]["schema"]["additionalProperties"] is False


def test_a_generic_model_name_is_sanitized_for_the_wire(responses_transport_factory):
    from typing import Generic, TypeVar

    from pydantic import BaseModel

    T = TypeVar("T")

    class Box(BaseModel, Generic[T]):
        item: T

    transport = responses_transport_factory()

    payload = transport._build_chat_completion_payload(
        ChatCompletionRequest(
            model="gpt-5.4",
            messages=[UserMessage(content="hi")],
            response_format=Box[int],
        ),
    )

    # `Box[int].__name__` is literally "Box[int]", which the wire rejects.
    assert payload["text"]["format"]["name"] == "Box_int_"


# ── provider-defined tools ────────────────────────────────────────────────────

PATCH_TOOL = Tool(
    name="apply_patch",
    description="never sent",
    parameters={"type": "object", "properties": {}},
    provider_type="apply_patch",
)
SHELL_TOOL = Tool(
    name="shell",
    description="never sent",
    parameters={"type": "object", "properties": {}},
    provider_type="shell",
)


def _request(transport, **kwargs):
    return transport._build_chat_completion_payload(
        ChatCompletionRequest(
            model="gpt-5.4",
            provider="openai",
            messages=kwargs.pop("messages", [UserMessage(content="Hi")]),
            **kwargs,
        )
    )


def test_apply_patch_is_sent_as_a_bare_type(responses_transport_factory):
    payload = _request(responses_transport_factory(), tools=[PATCH_TOOL])

    assert payload["tools"] == [{"type": "apply_patch"}]


def test_shell_declares_a_local_environment(responses_transport_factory):
    # the alternative is a container OpenAI provisions, which is not what luca
    # executes
    payload = _request(responses_transport_factory(), tools=[SHELL_TOOL])

    assert payload["tools"] == [{"type": "shell", "environment": {"type": "local"}}]


def test_an_unknown_provider_tool_type_is_refused(responses_transport_factory):
    unknown = Tool(name="x", description="d", parameters={}, provider_type="computer_20250124")

    with pytest.raises(UnsupportedParameterError, match="does not accept the provider tool type"):
        _request(responses_transport_factory(), tools=[unknown])


def test_an_apply_patch_call_is_parsed_as_an_ordinary_tool_call(responses_transport_factory):
    # the item carries no tool name; the name comes from the tool that was
    # offered with that provider type
    transport = responses_transport_factory()
    data = {
        "id": "resp_1",
        "model": "gpt-5.4",
        "output": [
            {
                "type": "apply_patch_call",
                "call_id": "call_1",
                "status": "completed",
                "operation": {"type": "update_file", "path": "lib/fib.py", "diff": "@@\n-a\n+b"},
            }
        ],
    }

    message = transport._parse_assistant_message(
        data,
        ChatCompletionRequest(model="gpt-5.4", provider="openai", messages=[], tools=[PATCH_TOOL]),
    )

    assert message.content == [
        ToolCall(
            id="call_1",
            name="apply_patch",
            arguments={"type": "update_file", "path": "lib/fib.py", "diff": "@@\n-a\n+b"},
            complete=True,
            provider_type="apply_patch",
        )
    ]


def test_a_shell_call_is_parsed_as_an_ordinary_tool_call(responses_transport_factory):
    transport = responses_transport_factory()
    data = {
        "id": "resp_1",
        "model": "gpt-5.4",
        "output": [
            {
                "type": "shell_call",
                "call_id": "call_2",
                "status": "in_progress",
                "action": {"commands": ["ls -l"], "timeout_ms": 120000},
            }
        ],
    }

    message = transport._parse_assistant_message(
        data,
        ChatCompletionRequest(model="gpt-5.4", provider="openai", messages=[], tools=[SHELL_TOOL]),
    )

    assert message.content == [
        ToolCall(
            id="call_2",
            name="shell",
            arguments={"commands": ["ls -l"], "timeout_ms": 120000},
            complete=True,
            provider_type="shell",
        )
    ]


def test_a_native_item_is_ignored_when_nothing_claimed_that_type(responses_transport_factory):
    # exactly what happens to every other hosted-tool item
    transport = responses_transport_factory()
    data = {
        "id": "resp_1",
        "model": "gpt-5.4",
        "output": [{"type": "apply_patch_call", "call_id": "c", "operation": {}}],
    }

    message = transport._parse_assistant_message(
        data, ChatCompletionRequest(model="gpt-5.4", provider="openai", messages=[])
    )

    assert message.content == []


def test_an_apply_patch_result_goes_back_as_a_verdict(responses_transport_factory):
    # a function_call_output answering an apply_patch_call is rejected
    payload = _request(
        responses_transport_factory(),
        tools=[PATCH_TOOL],
        messages=[
            UserMessage(content="fix it"),
            AssistantMessage(
                content=[
                    ToolCall(
                        id="call_1",
                        name="apply_patch",
                        arguments={"type": "delete_file", "path": "a.py"},
                        provider_type="apply_patch",
                    )
                ],
                finish_reason="tool_use",
                provider="openai",
                model="gpt-5.4",
            ),
            ToolMessage(tool_call_id="call_1", content="Success."),
        ],
    )

    assert payload["input"][-1] == {
        "type": "apply_patch_call_output",
        "call_id": "call_1",
        "status": "completed",
        "output": "Success.",
    }


def test_a_failed_patch_reports_failed(responses_transport_factory):
    payload = _request(
        responses_transport_factory(),
        tools=[PATCH_TOOL],
        messages=[
            UserMessage(content="fix it"),
            AssistantMessage(
                content=[
                    ToolCall(
                        id="call_1",
                        name="apply_patch",
                        arguments={"type": "delete_file", "path": "a.py"},
                        provider_type="apply_patch",
                    )
                ],
                finish_reason="tool_use",
                provider="openai",
                model="gpt-5.4",
            ),
            ToolMessage(tool_call_id="call_1", content="File not found", is_error=True),
        ],
    )

    assert payload["input"][-1]["status"] == "failed"


def test_a_shell_result_goes_back_in_the_shell_shape(responses_transport_factory):
    payload = _request(
        responses_transport_factory(),
        tools=[SHELL_TOOL],
        messages=[
            UserMessage(content="list"),
            AssistantMessage(
                content=[
                    ToolCall(
                        id="call_2",
                        name="shell",
                        arguments={"commands": ["ls"]},
                        provider_type="shell",
                    )
                ],
                finish_reason="tool_use",
                provider="openai",
                model="gpt-5.4",
            ),
            ToolMessage(tool_call_id="call_2", content="a.py\n"),
        ],
    )

    assert payload["input"][-1] == {
        "type": "shell_call_output",
        "call_id": "call_2",
        "output": [{"stdout": "a.py\n", "stderr": "", "outcome": {"type": "exit", "exit_code": 0}}],
    }


def test_a_replayed_native_call_keeps_its_item_type(responses_transport_factory):
    # a resumed conversation has to match what the provider recorded
    payload = _request(
        responses_transport_factory(),
        tools=[PATCH_TOOL],
        messages=[
            UserMessage(content="fix it"),
            AssistantMessage(
                content=[
                    ToolCall(
                        id="call_1",
                        name="apply_patch",
                        arguments={"type": "delete_file", "path": "a.py"},
                        provider_type="apply_patch",
                    )
                ],
                finish_reason="tool_use",
                provider="openai",
                model="gpt-5.4",
            ),
        ],
    )

    assert payload["input"][-1] == {
        "type": "apply_patch_call",
        "call_id": "call_1",
        "operation": {"type": "delete_file", "path": "a.py"},
    }


def test_a_call_recorded_before_native_tools_replays_as_a_function_call(responses_transport_factory):
    # luca ships its own `apply_patch`, so a history written with native tools
    # off holds calls that share the name and take completely different
    # arguments. Deriving the item type from the name would rebuild this one as
    # an apply_patch_call whose operation has no type, path or diff.
    payload = _request(
        responses_transport_factory(),
        tools=[PATCH_TOOL],
        messages=[
            UserMessage(content="fix it"),
            AssistantMessage(
                content=[ToolCall(id="call_1", name="apply_patch", arguments={"patch_text": "*** Begin Patch"})],
                finish_reason="tool_use",
                provider="openai",
                model="gpt-5.4",
            ),
            ToolMessage(tool_call_id="call_1", content="Success."),
        ],
    )

    assert payload["input"][1:] == [
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "apply_patch",
            "arguments": '{"patch_text": "*** Begin Patch"}',
        },
        {"type": "function_call_output", "call_id": "call_1", "output": "Success."},
    ]


def test_a_native_call_replays_as_a_function_call_when_the_tool_is_no_longer_offered(
    responses_transport_factory,
):
    # resumed with {"tools": {"native": false}}: asking for an item type this
    # request never declared is worse than replaying it as what it now is
    ordinary = Tool(name="apply_patch", description="Patch.", parameters={"type": "object", "properties": {}})
    payload = _request(
        responses_transport_factory(),
        tools=[ordinary],
        messages=[
            UserMessage(content="fix it"),
            AssistantMessage(
                content=[
                    ToolCall(
                        id="call_1",
                        name="apply_patch",
                        arguments={"type": "delete_file", "path": "a.py"},
                        provider_type="apply_patch",
                    )
                ],
                finish_reason="tool_use",
                provider="openai",
                model="gpt-5.4",
            ),
            ToolMessage(tool_call_id="call_1", content="Success."),
        ],
    )

    assert payload["input"][1:] == [
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "apply_patch",
            "arguments": '{"type": "delete_file", "path": "a.py"}',
        },
        {"type": "function_call_output", "call_id": "call_1", "output": "Success."},
    ]


def test_an_ordinary_tool_is_untouched_alongside_a_native_one(responses_transport_factory):
    ordinary = Tool(name="grep", description="Search.", parameters={"type": "object", "properties": {}})

    payload = _request(responses_transport_factory(), tools=[PATCH_TOOL, ordinary])

    assert payload["tools"] == [
        {"type": "apply_patch"},
        {
            "type": "function",
            "name": "grep",
            "description": "Search.",
            "parameters": {"type": "object", "properties": {}},
        },
    ]
