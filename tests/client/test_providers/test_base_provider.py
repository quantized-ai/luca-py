"""BaseProvider construction — defaults, env fallback, kwarg overrides."""

from dataclasses import dataclass

import pytest

from luca.client.providers import (
    AnthropicProvider,
    OpenAIProvider,
    OpenRouterProvider,
)
from luca.client.transports import (
    AnthropicTransport,
    OpenAITransport,
    OpenRouterTransport,
)
from tests.client._helpers.stub_transports import StubTransport


@dataclass(frozen=True)
class ProviderCase:
    name: str
    cls: type
    expected_name: str
    expected_base_url: str
    expected_env_var: str
    expected_transport_cls: type


CASES = [
    ProviderCase(
        "openai", OpenAIProvider, "openai",
        "https://api.openai.com/v1", "OPENAI_API_KEY", OpenAITransport,
    ),
    ProviderCase(
        "anthropic", AnthropicProvider, "anthropic",
        "https://api.anthropic.com", "ANTHROPIC_API_KEY", AnthropicTransport,
    ),
    ProviderCase(
        "openrouter", OpenRouterProvider, "openrouter",
        "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", OpenRouterTransport,
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_provider_default_construction(case, monkeypatch):
    monkeypatch.setenv(case.expected_env_var, "key-from-env")
    provider = case.cls()
    t = provider.transport
    assert t._provider == case.expected_name
    assert t._base_url == case.expected_base_url.rstrip("/")
    assert t._api_key == "key-from-env"
    assert isinstance(t, case.expected_transport_cls)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_provider_explicit_api_key_wins_over_env(case, monkeypatch):
    monkeypatch.setenv(case.expected_env_var, "from-env")
    provider = case.cls(api_key="from-arg")
    assert provider.transport._api_key == "from-arg"


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_provider_base_url_kwarg_overrides_default(case):
    provider = case.cls(api_key="x", base_url="https://example.test/v1")
    assert provider.transport._base_url == "https://example.test/v1"


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_provider_transport_kwarg_injects_prebuilt_instance(case):
    stub = StubTransport()
    provider = case.cls(transport=stub)
    assert provider.transport is stub
