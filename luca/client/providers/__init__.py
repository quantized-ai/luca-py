"""Provider registry + first-class provider classes.

PROVIDERS maps host name → BaseProvider subclass OR config dict (the latter
spawns a GenericProvider).
"""

from __future__ import annotations

from typing import TypedDict, Union

from ..exceptions import ProviderNotFoundError
from ..transports import OpenAITransport
from .anthropic import AnthropicProvider
from .base import BaseProvider, ChatCompletionMixin
from .bedrock import BedrockProvider
from .faux import FauxProvider
from .generic import GenericProvider
from .openai import OpenAIProvider
from .openrouter import OpenRouterProvider


class ProviderConfig(TypedDict, total=False):
    default_base_url: str
    default_api_key_env_var: str | None
    default_transport_class: type


PROVIDERS: dict[str, Union[type, ProviderConfig]] = {
    # First-class providers
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "openrouter": OpenRouterProvider,
    "bedrock": BedrockProvider,
    # Long-tail OpenAI-compatible hosts
    "groq": {
        "default_base_url": "https://api.groq.com/openai/v1",
        "default_api_key_env_var": "GROQ_API_KEY",
        "default_transport_class": OpenAITransport,
    },
    "deepseek": {
        "default_base_url": "https://api.deepseek.com/v1",
        "default_api_key_env_var": "DEEPSEEK_API_KEY",
        "default_transport_class": OpenAITransport,
    },
    "ollama": {
        "default_base_url": "http://localhost:11434/v1",
        "default_api_key_env_var": None,
        "default_transport_class": OpenAITransport,
    },
}


def register_provider(name: str, config_or_class) -> None:
    """Register a host. Pass a BaseProvider subclass for custom behavior,
    or a config dict for GenericProvider routing."""
    PROVIDERS[name] = config_or_class


def resolve_provider(
    name: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    transport_class: type | None = None,
    transport=None,
    timeout: float | None = None,
    http_client=None,
    async_http_client=None,
) -> BaseProvider:
    """Build a Provider instance for `name`, applying any per-call overrides."""
    entry = PROVIDERS.get(name)

    common: dict = dict(
        api_key=api_key, base_url=base_url,
        transport_class=transport_class, transport=transport,
        http_client=http_client, async_http_client=async_http_client,
    )
    if timeout is not None:
        common["timeout"] = timeout
    # Drop None entries — BaseProvider has its own defaults for them.
    common = {k: v for k, v in common.items() if v is not None}

    if entry is None:
        # Allow "transport_class= + base_url=" escape hatch even without an entry.
        if transport is not None or (transport_class is not None and base_url is not None):
            return GenericProvider(
                name=name,
                default_base_url=base_url or "",
                default_api_key_env_var=None,
                default_transport_class=transport_class or type(transport),
                **common,
            )
        raise ProviderNotFoundError(
            f"Provider {name!r} is not registered. Either:\n"
            f"  - Register it: register_provider({name!r}, {{...config...}})\n"
            f"  - Or pass `transport_class=` and `base_url=` explicitly."
        )

    if isinstance(entry, type) and issubclass(entry, BaseProvider):
        return entry(**common)
    # entry is a config dict → GenericProvider
    return GenericProvider(name=name, **entry, **common)


__all__ = [
    "BaseProvider", "ChatCompletionMixin",
    "OpenAIProvider", "AnthropicProvider", "OpenRouterProvider", "BedrockProvider",
    "GenericProvider", "FauxProvider",
    "PROVIDERS", "ProviderConfig",
    "register_provider", "resolve_provider",
]
