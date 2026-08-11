"""Transport registry + re-exports.

Each transport class speaks one wire protocol; the same class can serve many
hosts via PROVIDERS lookups.
"""

from __future__ import annotations

from .anthropic import AnthropicTransport
from .base import BaseTransport, ChatCompletionTransportMixin
from .bedrock import BedrockTransport
from .faux import FauxTransport
from .ollama import OllamaTransport
from .openai import OpenAITransport
from .openai_responses import OpenAIResponsesTransport
from .openrouter import OpenRouterTransport

TRANSPORTS: dict[str, type] = {
    # "openai" is the CHAT COMPLETIONS protocol: what every OpenAI-compatible
    # host speaks. Provider `openai` itself runs on "openai-responses".
    "openai": OpenAITransport,
    "openai-responses": OpenAIResponsesTransport,
    "anthropic": AnthropicTransport,
    "openrouter": OpenRouterTransport,
    "bedrock": BedrockTransport,
    "ollama": OllamaTransport,
    # FauxTransport is deliberately NOT registered — tests construct it explicitly.
}


def register_transport(name: str, cls: type) -> None:
    TRANSPORTS[name] = cls


__all__ = [
    "BaseTransport",
    "ChatCompletionTransportMixin",
    "OpenAITransport",
    "OpenAIResponsesTransport",
    "AnthropicTransport",
    "OpenRouterTransport",
    "BedrockTransport",
    "OllamaTransport",
    "FauxTransport",
    "TRANSPORTS",
    "register_transport",
]
