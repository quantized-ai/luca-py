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


def test_two_identical_calls_reuse_one_provider_instance(stub_provider):
    stub_provider.configure(responses=[RESP, RESP])
    completion(model="stub:m", messages=[UserMessage(content="hi")])
    completion(model="stub:m", messages=[UserMessage(content="hi")])
    assert len(stub_provider.instantiations) == 1


def test_different_api_keys_produce_different_instances(stub_provider):
    stub_provider.configure(responses=[RESP, RESP])
    completion(model="stub:m", api_key="a", messages=[UserMessage(content="hi")])
    completion(model="stub:m", api_key="b", messages=[UserMessage(content="hi")])
    assert len(stub_provider.instantiations) == 2


def test_different_base_urls_produce_different_instances(stub_provider):
    stub_provider.configure(responses=[RESP, RESP])
    completion(model="stub:m", base_url="https://a", messages=[UserMessage(content="hi")])
    completion(model="stub:m", base_url="https://b", messages=[UserMessage(content="hi")])
    assert len(stub_provider.instantiations) == 2
