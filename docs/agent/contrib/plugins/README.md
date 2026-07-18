# Plugins

`BasePlugin` + `PluginAgentSessionRunner` — the composition layer that installs
an agent capability (a tool registry + system-prompt parts + middleware) in one
move. The core runner knows nothing about plugins; this package composes them
at construction over a
[`ProxyToolRegistry`](../simple_tool_registry/README.md). The concept, hooks,
and the `MemoryPlugin` example live in [`09-plugins.md`](../../09-plugins.md) —
this page is the package reference.

```python
from luca.agent.contrib.plugins import BasePlugin, PluginAgentSessionRunner
```

## 1. Install

`PluginAgentSessionRunner` is an `AgentSessionRunner` subclass with one extra
kwarg:

```python
runner = PluginAgentSessionRunner(
    session,
    tool_registry=app_registry,        # optional — your own registry
    plugins=[MemoryPlugin()],
    system_prompt_parts=[SYSTEM_PROMPT],
)
```

Composition (once, at construction): the directly-passed registry and each
plugin's `get_tool_registry(session)` result become children of one
`ProxyToolRegistry` (direct first, plugins in list order; `None` contributes
nothing); `get_system_prompt_parts(session)` / `get_middleware(session)`
results extend the constructor lists after the directly-passed items.

## 2. Write a plugin

A plain class implementing any of the three duck-typed hooks (`BasePlugin` is
an optional base — prefer a plain class):

```python
class ClockPlugin:
    def get_tool_registry(self, agent_session):
        return SimpleToolRegistry(
            tools=[NowTool()], permission_policy=YoloPermissionPolicy(),
        )

    def get_system_prompt_parts(self, agent_session):
        return ["You can read the current time with the now() tool."]
```

The plugin's registry owns its tools' approval policy. A multi-registry plugin
returns its own `ProxyToolRegistry`; a plugin whose tools should be gated by
the *application's* rules shouldn't ship a registry at all — expose the tools
and let the app compose them.

> ⚠️ **Construction-time only.** Hooks run once. Per-call dynamism belongs in
> a registry's `get_tools`, [middleware](../../07-middleware.md), or a callable
> prompt part.

Next: [`resource_permissions/README.md`](../resource_permissions/README.md).
