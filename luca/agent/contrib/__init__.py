"""luca.agent.contrib — optional packages built on the core surface.

The boundary is sharp: `luca.client` is the LLM SDK, `luca.agent.core` is the
agent core (data model, runner, strategy contracts) — everything else ships
here. A contrib package consumes only the public `luca.agent.core` surface,
exactly like application code would; nothing in `core/` imports from contrib.
Import from the specific package, e.g.
`luca.agent.contrib.resource_permissions`; this package level intentionally
exports nothing.
"""
