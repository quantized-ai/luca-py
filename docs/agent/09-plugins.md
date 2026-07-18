# Plugins

A plugin bundles the pieces one agent capability usually ships together — a
tool registry, system-prompt parts, middleware — behind a single object, so you
install the capability in one move. Plugins are a **contrib** concept: the core
runner knows nothing about them; `PluginAgentSessionRunner`
(`luca.agent.contrib.plugins`) is the composition layer:

```python
from luca.agent.contrib.plugins import PluginAgentSessionRunner
from luca.agent.contrib.memory import MemoryPlugin

runner = PluginAgentSessionRunner(
    session,
    tool_registry=registry,            # your app's own registry (optional)
    plugins=[MemoryPlugin()],
    system_prompt_parts=[SYSTEM_PROMPT],
)
```

## 1. The hooks

A plugin is a plain Python class — the same duck-typed convention as
[middleware](07-middleware.md): implement only the hooks you need, `hasattr`
skips the rest. Every hook receives the `AgentSession` the runner is being
built over.

| Hook | Returns | Composes into |
|---|---|---|
| `get_tool_registry(agent_session)` | a `ToolRegistry` or `None` | a child of the runner's one `ProxyToolRegistry` (after the directly-passed registry) |
| `get_system_prompt_parts(agent_session)` | list of parts — any constructor form: `str` / dict / `SystemPromptPart` / callable (see [06](06-system-prompts.md)) | `system_prompt_parts` |
| `get_middleware(agent_session)` | list of middleware instances | `middleware` |

Contributions land **after** the directly-passed items, plugins in list order.
Plugin prompt parts are coerced exactly like constructor parts. `BasePlugin`
(registry `None`, empty lists) is an optional base for partial overrides —
prefer a plain class.

A plugin's tools ride in the plugin's **own registry**, so the plugin also owns
their approval policy — a multi-registry plugin returns its own
`ProxyToolRegistry`. Want a plugin's tools gated by *your* rules instead?
Compose your own registry over the plugin's tools rather than installing the
plugin's registry.

> ⚠️ **Construction-time only.** Hooks run once, inside
> `PluginAgentSessionRunner.__init__` — never again. Per-call behavior (varying
> the tool list by turn, injecting context per LLM call) belongs in your
> registry's dynamic `get_tools`, [middleware](07-middleware.md), or a callable
> prompt part.

## 2. Example: the memory plugin

`luca.agent.contrib.memory` is a working example that is also a real feature —
two in-memory capabilities: a **scratchpad** (the agent's private working
memory) and a **todo list** (its task tracker). Four tools plus the prompt
parts that teach the model to use them:

| Tool | Args | Does |
|---|---|---|
| `read_scratchpad` | — | returns the current content (empty string at first) |
| `write_scratchpad` | `content: str` | replaces the whole content |
| `read_todo` | — | returns the todo list, a `repr()` of `{"content", "status"}` dicts (`[]` at first) |
| `update_todos` | `todos: list[TodoItem]` | replaces the whole list — the model re-sends every item, including unchanged ones |

A `TodoItem` is `{"content": str, "status": "pending" | "in_progress" |
"completed" | "cancelled"}` — validated by the tool's `Args` schema, so a bad
status becomes a clean tool error.

The plugin itself is the whole pattern in ~25 lines
([full source](../../luca/agent/contrib/memory/plugin.py)):

```python
class MemoryPlugin:
    def __init__(self) -> None:
        self.scratchpad_store: dict = {}
        self.todo_store: dict = {}

    def get_tools(self):
        return [
            ReadScratchPadTool(self.scratchpad_store),
            WriteScratchPadTool(self.scratchpad_store),
            ReadTodoTool(self.todo_store),
            UpdateTodosTool(self.todo_store),
        ]

    def get_tool_registry(self, agent_session):
        return SimpleToolRegistry(
            tools=self.get_tools(), permission_policy=YoloPermissionPolicy(),
        )

    def get_system_prompt_parts(self, agent_session):
        return [SCRATCHPAD_SYSTEM_PROMPT, TODO_SYSTEM_PROMPT]
```

The plugin owns the shared state (one store handed to each tool pair) — the
piece you couldn't express by passing loose tools — and its Yolo registry means
memory tools auto-run regardless of how the app gates its own tools.

> ⚠️ **Not persisted.** The stores live on the plugin instance, not the
> session: one `MemoryPlugin()` = one scratchpad + one todo list, and a new
> instance starts blank. Durable memory would write through a store you
> persist yourself.

## 3. Equivalence

A plugin is sugar, not a runtime actor. A runner built with a plugin **is**
the runner built with the same objects composed directly — and
`AgentSessionRunner.__eq__` compares that effective configuration:

```python
from luca.agent.core import AgentSessionRunner
from luca.agent.contrib.simple_tool_registry import ProxyToolRegistry

plugin = MemoryPlugin()
plugin_registry = plugin.get_tool_registry(session)

class StoredRegistryPlugin:                       # hand the SAME registry back
    def get_tool_registry(self, agent_session):
        return plugin_registry
    def get_system_prompt_parts(self, agent_session):
        return plugin.get_system_prompt_parts(agent_session)

with_plugin = PluginAgentSessionRunner(session, plugins=[StoredRegistryPlugin()])
explicit = AgentSessionRunner(
    session,
    tool_registry=ProxyToolRegistry(plugin_registry),
    system_prompt_parts=plugin.get_system_prompt_parts(session),
)

assert with_plugin == explicit
```

Next: [`10-projection.md`](10-projection.md).
