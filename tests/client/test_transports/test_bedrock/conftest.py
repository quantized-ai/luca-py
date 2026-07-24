import pytest


@pytest.fixture
def bedrock_transport_factory():
    from luca.client.transports import BedrockTransport

    def make(*, http_client=None, async_http_client=None):
        return BedrockTransport(
            provider="bedrock",
            base_url="https://bedrock-runtime.us-east-1.amazonaws.com",
            api_key="bedrock-token-test",
            http_client=http_client,
            async_http_client=async_http_client,
        )

    return make
