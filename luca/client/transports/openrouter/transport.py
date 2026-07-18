"""OpenRouter transport — subclass of OpenAITransport with minor header tweaks."""

from __future__ import annotations

from ..openai.transport import OpenAITransport


class OpenRouterTransport(OpenAITransport):
    transport_id = "openrouter"

    def _headers(self) -> dict[str, str]:
        h = super()._headers()
        # HTTP-Referer / X-Title can be added here if needed; OpenRouter accepts
        # but doesn't require them.
        return h
