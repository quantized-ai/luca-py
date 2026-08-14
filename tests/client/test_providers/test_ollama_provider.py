"""The `ollama` registry entry: a local host with no credential."""

import httpx

from luca.client.providers import resolve_provider
from luca.client.transports import OllamaTransport


def test_it_defaults_to_the_local_daemon_on_the_native_api():
    provider = resolve_provider("ollama")
    transport = provider.transport
    try:
        # No `/v1`: the native endpoint is the only one that honours num_ctx.
        assert transport._base_url == "http://localhost:11434"
        assert isinstance(transport, OllamaTransport)
    finally:
        provider.close()


def test_no_credential_is_read_or_sent(monkeypatch):
    # A local daemon has no key, and inventing an env var to read would only
    # create a way to get it wrong.
    provider = resolve_provider("ollama")
    try:
        assert provider.transport._api_key is None
        assert "Authorization" not in provider.transport._headers()
    finally:
        provider.close()


def test_an_explicit_base_url_points_at_a_remote_daemon():
    provider = resolve_provider("ollama", base_url="http://gpu-box.local:11434")
    try:
        assert provider.transport._base_url == "http://gpu-box.local:11434"
    finally:
        provider.close()


def test_the_transport_is_reachable_through_the_registry():
    from luca.client.transports import TRANSPORTS

    assert TRANSPORTS["ollama"] is OllamaTransport


def test_a_custom_http_client_is_used(monkeypatch):
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})))
    provider = resolve_provider("ollama", http_client=client)
    try:
        assert provider.transport._client is client
    finally:
        provider.close()
