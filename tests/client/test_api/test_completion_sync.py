"""Helper kwargs end up on the request DTO."""

from dataclasses import dataclass

import pytest

from luca.client import completion
from luca.client.exceptions import BadRequestError
from luca.client.types import (
    AssistantMessage,
    ChatCompletionResponse,
    TextBlock,
    Tool,
    UserMessage,
)


RESP = ChatCompletionResponse(
    message=AssistantMessage(
        content=[TextBlock(text="ok")],
        finish_reason="stop", provider_finish_reason="stop",
        provider="stub", model="m",
    ),
)


@dataclass(frozen=True)
class KwargCase:
    name: str
    kwargs: dict
    field_name: str
    expected_value: object


CASES = [
    KwargCase("system_message", {"system_message": "be brief"}, "system_message", "be brief"),
    KwargCase("temperature", {"temperature": 0.5}, "temperature", 0.5),
    KwargCase("max_tokens", {"max_tokens": 100}, "max_tokens", 100),
    KwargCase("reasoning", {"reasoning": "high"}, "reasoning", "high"),
    KwargCase(
        "provider_options",
        {"provider_options": {"stub": {"x": 1}}},
        "provider_options",
        {"stub": {"x": 1}},
    ),
    KwargCase("metadata", {"metadata": {"trace_id": "t1"}}, "metadata", {"trace_id": "t1"}),
    KwargCase("session_id", {"session_id": "s-1"}, "session_id", "s-1"),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_helper_forwards_kwarg_to_request(stub_provider, case):
    stub_provider.configure(responses=[RESP])
    completion(model="stub:m", messages=[UserMessage(content="hi")], **case.kwargs)
    req = stub_provider.instances[0].calls[0].request
    assert getattr(req, case.field_name) == case.expected_value


def test_dict_messages_coerce_to_typed(stub_provider):
    stub_provider.configure(responses=[RESP])
    completion(model="stub:m", messages=[{"role": "user", "content": "hi"}])
    req = stub_provider.instances[0].calls[0].request
    assert isinstance(req.messages[0], UserMessage)
    assert req.messages[0].content == "hi"


def test_system_role_dict_raises_bad_request(stub_provider):
    stub_provider.configure(responses=[RESP])
    with pytest.raises(BadRequestError, match="system_message"):
        completion(
            model="stub:m",
            messages=[{"role": "system", "content": "be brief"}],
        )


def test_tools_dict_coerce_to_typed(stub_provider):
    stub_provider.configure(responses=[RESP])
    completion(
        model="stub:m",
        messages=[UserMessage(content="hi")],
        tools=[{
            "name": "t", "description": "...", "parameters": {"type": "object"},
        }],
    )
    req = stub_provider.instances[0].calls[0].request
    assert isinstance(req.tools[0], Tool)
    assert req.tools[0].name == "t"
