"""Provider registry: PROVIDERS dict + register_provider + resolve_provider."""

import pytest

from luca.client.exceptions import ProviderNotFoundError
from luca.client.providers import (
    PROVIDERS,
    GenericProvider,
    OpenAIProvider,
    register_provider,
    resolve_provider,
)
from luca.client.transports import OpenAITransport


def test_resolve_provider_class_entry():
    assert isinstance(resolve_provider("openai", api_key="x"), OpenAIProvider)


def test_resolve_provider_config_dict_entry_builds_generic_provider():
    p = resolve_provider("groq", api_key="x")
    assert isinstance(p, GenericProvider)
    assert p.name == "groq"


def test_resolve_provider_unknown_raises():
    with pytest.raises(ProviderNotFoundError):
        resolve_provider("not-a-real-provider")


def test_register_provider_config_dict(monkeypatch):
    monkeypatch.setitem(PROVIDERS, "my-host", {
        "default_base_url": "https://my-host.test/v1",
        "default_api_key_env_var": "MY_HOST_KEY",
        "default_transport_class": OpenAITransport,
    })
    p = resolve_provider("my-host", api_key="x")
    assert isinstance(p, GenericProvider)
    assert p.name == "my-host"


def test_resolve_with_explicit_transport_and_base_url_for_unknown_provider():
    p = resolve_provider(
        "new-vendor",
        api_key="x",
        base_url="https://new.example/v1",
        transport_class=OpenAITransport,
    )
    assert isinstance(p, GenericProvider)
    assert p.name == "new-vendor"
