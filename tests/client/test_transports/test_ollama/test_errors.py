"""Failures a LOCAL provider has that a hosted one does not.

The daemon may simply not be running, and the model may never have been
pulled. Both are one command away from fixed, so both messages say which.
"""

import httpx
import pytest

from luca.client.exceptions import (
    ConnectionError as ClientConnectionError,
    ModelNotFoundError,
    ProviderAPIError,
)
from luca.client.types import ChatCompletionRequest, UserMessage

from ..._helpers.httpx_mocks import make_sync_client

REQUEST = ChatCompletionRequest(
    provider="ollama",
    model="llama3.2:latest",
    messages=[UserMessage(content="Hi")],
)


def test_a_model_that_was_never_pulled_says_how_to_pull_it(ollama_transport_factory):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": 'model "llama3.2:latest" not found'})

    transport = ollama_transport_factory(http_client=make_sync_client(handler))

    with pytest.raises(ModelNotFoundError, match="ollama pull") as exc_info:
        transport.completion(REQUEST)

    assert exc_info.value.provider == "ollama"


def test_a_daemon_that_is_not_running_names_it_and_says_how_to_start_it(ollama_transport_factory):
    # The bare httpx message is "[Errno 61] Connection refused", which names
    # neither Ollama nor the port it was looking for.
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("[Errno 61] Connection refused")

    transport = ollama_transport_factory(http_client=make_sync_client(handler))

    with pytest.raises(ClientConnectionError) as exc_info:
        transport.completion(REQUEST)

    message = str(exc_info.value)
    assert "http://localhost:11434" in message
    assert "ollama serve" in message


def test_another_status_carries_the_daemons_own_error_text(ollama_transport_factory):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "llama runner process has terminated"})

    transport = ollama_transport_factory(http_client=make_sync_client(handler))

    with pytest.raises(ProviderAPIError, match="llama runner process has terminated"):
        transport.completion(REQUEST)
