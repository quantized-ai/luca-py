import pytest


@pytest.fixture
def anthropic_transport_factory():
    from luca.client.transports import AnthropicTransport

    def make(*, http_client=None, async_http_client=None):
        return AnthropicTransport(
            provider="anthropic",
            base_url="https://api.anthropic.com",
            api_key="sk-ant-test",
            http_client=http_client,
            async_http_client=async_http_client,
        )

    return make
