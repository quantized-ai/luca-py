"""Transport registry + re-exports.

Each transport class speaks one wire protocol; the same class can serve many
hosts via PROVIDERS lookups.
"""

from __future__ import annotations

from .anthropic import AnthropicTransport
from .base import BaseTransport, ChatCompletionTransportMixin
from .bedrock import BedrockTransport
from .faux import FauxTransport
from .openai import OpenAITransport
from .openrouter import OpenRouterTransport

TRANSPORTS: dict[str, type] = {
    "openai": OpenAITransport,
    "anthropic": AnthropicTransport,
    "openrouter": OpenRouterTransport,
    "bedrock": BedrockTransport,
    # FauxTransport is deliberately NOT registered — tests construct it explicitly.
}


def register_transport(name: str, cls: type) -> None:
    TRANSPORTS[name] = cls


__all__ = [
    "BaseTransport",
    "ChatCompletionTransportMixin",
    "OpenAITransport",
    "AnthropicTransport",
    "OpenRouterTransport",
    "BedrockTransport",
    "FauxTransport",
    "TRANSPORTS",
    "register_transport",
]
