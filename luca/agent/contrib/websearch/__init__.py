"""Provider-hosted web tools for the agent (`luca.agent.contrib.websearch`).

The provider executes web search / web fetch SERVER-SIDE, mid-response —
there is nothing to dispatch and no result to send back — so this plugin is
not a tool package: it appends the hosted declarations when the active
(provider, model) pair supports them, and mints one durable, private,
terminal `ToolExecution` per recorded operation (the structural index that
survives reload and provider switches). The truth stays in the assistant
entry's parts; the synthetic never duplicates private data.

There is deliberately NO approval gate: the provider executes mid-sampling,
so the permission pipeline cannot intercept an operation — advertisement
on/off plus the declaration's own limits (`allowed_domains`, `max_uses`) are
the control surface.

```python
from luca.agent.contrib.plugins import PluginAgentSessionRunner
from luca.agent.contrib.websearch import WebSearchPlugin

runner = PluginAgentSessionRunner(session, plugins=[WebSearchPlugin({
    "anthropic": {"search": {"max_uses": 5}},
})])
```
"""

from __future__ import annotations

from luca.agent.core import AgentSession

from .config import AnthropicWebConfig, OpenAIWebConfig, WebSearchConfig
from .middleware import WebSearchMiddleware
from .specs import WEB_FETCH_SPEC, WEB_SEARCH_SPEC
from .support import supported_web_tools


class WebSearchPlugin:
    """One JSON-shaped dict (or the typed config, or None for the defaults)
    in; one middleware out. A plain class — `get_middleware` is its whole
    plugin surface."""

    def __init__(self, config: dict | WebSearchConfig | None = None) -> None:
        self.config = WebSearchConfig.model_validate(config if config is not None else {})

    def get_middleware(self, session: AgentSession) -> list[WebSearchMiddleware]:
        return [WebSearchMiddleware(session, self.config)]


__all__ = [
    "AnthropicWebConfig",
    "OpenAIWebConfig",
    "WebSearchConfig",
    "WebSearchMiddleware",
    "WebSearchPlugin",
    "WEB_FETCH_SPEC",
    "WEB_SEARCH_SPEC",
    "supported_web_tools",
]
