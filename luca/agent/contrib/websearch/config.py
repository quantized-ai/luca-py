"""The websearch plugin's configuration surface (PRD D7).

Thin wrappers whose per-provider `options` ARE the client's own declaration
classes — not a schema this plugin invents — so every knob the provider
documents is available under its own name and validates where the client
validates it. The first module in the repo to import both providers'
`WebSearchTool`, hence the aliases.

Semantics: instantiating the plugin = feature on with provider defaults; a
provider block customizes or disables its side; at request time the
middleware still intersects with the per-model support table
(`support.py`) — an enabled provider on an unsupported model advertises
nothing. Anthropic's `fetch` is OPT-IN: `None` (the default) means the fetch
tool is never declared.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from luca.client.providers.anthropic import (
    WebFetchTool as AnthropicWebFetchTool,
    WebSearchTool as AnthropicWebSearchTool,
)
from luca.client.providers.openai import WebSearchTool as OpenAIWebSearchTool


class OpenAIWebConfig(BaseModel):
    enabled: bool = True
    options: OpenAIWebSearchTool = OpenAIWebSearchTool()

    model_config = ConfigDict(extra="forbid")


class AnthropicWebConfig(BaseModel):
    enabled: bool = True
    search: AnthropicWebSearchTool = AnthropicWebSearchTool()
    fetch: AnthropicWebFetchTool | None = None  # opt-in

    model_config = ConfigDict(extra="forbid")


class WebSearchConfig(BaseModel):
    """`WebSearchPlugin`'s whole configuration — JSON-shaped, `luca.json`'s
    `websearch` block maps onto it 1:1. `enabled: false` disables everything
    temporarily while the rest of the block stays in place."""

    enabled: bool = True
    openai: OpenAIWebConfig = OpenAIWebConfig()
    anthropic: AnthropicWebConfig = AnthropicWebConfig()

    model_config = ConfigDict(extra="forbid")
