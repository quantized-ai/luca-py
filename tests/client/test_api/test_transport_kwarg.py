from luca.client import completion
from luca.client.types import (
    AssistantMessage,
    ChatCompletionResponse,
    TextBlock,
    UserMessage,
)

from tests.client._helpers.stub_transports import StubTransport


def test_helper_accepts_prebuilt_transport():
    resp = ChatCompletionResponse(
        message=AssistantMessage(
            content=[TextBlock(text="hello")],
            finish_reason="stop", provider_finish_reason="stop",
            provider="custom", model="m",
        ),
    )
    stub = StubTransport(responses=[resp])
    actual = completion(
        model="custom:m",
        messages=[UserMessage(content="hi")],
        transport=stub,
    )
    assert actual is resp
    assert stub.calls[0].method == "completion"
    assert (stub.calls[0].request.provider, stub.calls[0].request.model) == ("custom", "m")
