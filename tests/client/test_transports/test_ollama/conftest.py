import pytest

from luca.client.types import ModelInfo


@pytest.fixture
def ollama_transport_factory():
    from luca.client.transports import OllamaTransport

    def make(*, http_client=None, async_http_client=None):
        return OllamaTransport(
            provider="ollama",
            base_url="http://localhost:11434",
            api_key=None,
            http_client=http_client,
            async_http_client=async_http_client,
        )

    return make


@pytest.fixture
def model_info():
    """What discovery would have registered for a 32k tool-capable model."""
    return ModelInfo(
        model="qwen2.5:14b",
        provider="ollama",
        context_window=32_768,
        supports_tools=True,
    )
