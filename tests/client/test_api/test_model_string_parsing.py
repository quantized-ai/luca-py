import pytest

from luca.client import completion
from luca.client.types import (
    AssistantMessage,
    ChatCompletionResponse,
    TextBlock,
    UserMessage,
)


RESP = ChatCompletionResponse(
    message=AssistantMessage(
        content=[TextBlock(text="ok")],
        finish_reason="stop", provider_finish_reason="stop",
        provider="stub", model="m",
    ),
)


def test_prefix_form(stub_provider):
    stub_provider.configure(responses=[RESP])
    completion(model="stub:some-model", messages=[UserMessage(content="hi")])
    req = stub_provider.instances[0].calls[0].request
    assert (req.provider, req.model) == ("stub", "some-model")


def test_provider_kwarg_overrides_prefix(stub_provider):
    stub_provider.configure(responses=[RESP])
    completion(
        model="openai:gpt-4o", provider="stub",
        messages=[UserMessage(content="hi")],
    )
    req = stub_provider.instances[0].calls[0].request
    assert (req.provider, req.model) == ("stub", "gpt-4o")


def test_no_provider_no_prefix_raises():
    with pytest.raises(ValueError, match="No provider specified"):
        completion(model="gpt-4o", messages=[UserMessage(content="hi")])


def test_model_id_with_slash_preserved(stub_provider):
    stub_provider.configure(responses=[RESP])
    completion(model="stub:author/model", messages=[UserMessage(content="hi")])
    assert stub_provider.instances[0].calls[0].request.model == "author/model"
