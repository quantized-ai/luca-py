from luca.client import AwsCredentials, completion, get_provider
from luca.client.types import (
    AssistantMessage,
    ChatCompletionResponse,
    TextBlock,
    UserMessage,
)

RESP = ChatCompletionResponse(
    messages=[
        AssistantMessage(
            content=[TextBlock(text="ok")],
            finish_reason="stop",
            provider_finish_reason="stop",
            provider="stub",
            model="m",
        )
    ],
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


def test_different_credentials_produce_different_instances(stub_provider):
    stub_provider.configure(responses=[RESP, RESP])
    one = AwsCredentials(access_key_id="AKIA-1", secret_access_key="s", region="us-east-1")
    two = AwsCredentials(access_key_id="AKIA-2", secret_access_key="s", region="us-east-1")
    completion(model="stub:m", credentials=one, messages=[UserMessage(content="hi")])
    completion(model="stub:m", credentials=two, messages=[UserMessage(content="hi")])
    assert len(stub_provider.instantiations) == 2


def test_get_provider_returns_the_cached_instance(stub_provider):
    stub_provider.configure(responses=[RESP])
    completion(model="stub:m", messages=[UserMessage(content="hi")])

    assert get_provider("stub:m") is stub_provider.instances[0]
    assert len(stub_provider.instances) == 1
