"""Per-vendor OpenAI Responses helpers."""

import pytest


@pytest.fixture
def responses_transport_factory():
    from luca.client.transports import OpenAIResponsesTransport

    def make(*, http_client=None, async_http_client=None):
        return OpenAIResponsesTransport(
            provider="openai",
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            http_client=http_client,
            async_http_client=async_http_client,
        )

    return make
