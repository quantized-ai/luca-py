"""Per-vendor OpenAI helpers."""

import pytest


@pytest.fixture
def openai_transport_factory():
    from luca.client.transports import OpenAITransport

    def make(*, http_client=None, async_http_client=None):
        return OpenAITransport(
            provider="openai",
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            http_client=http_client,
            async_http_client=async_http_client,
        )

    return make
