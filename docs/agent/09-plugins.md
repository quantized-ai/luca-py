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
> the tool list by turn or by conversation, injecting context per LLM call)
> belongs in your registry's dynamic `get_tools(session, conversation_id)`
> ([03](03-tools.md)), [middleware](07-middleware.md), or a callable prompt
> part.

## 2. Example: the memory plugin

`luca.agent.contrib.memory` is a working example that is also a real feature —
two in-memory capabilities: a **scratchpad** (the agent's private working
memory) and a **todo list** (its task tracker). Four tools plus the prompt
parts that teach the model to use them:

| Tool | Args | Does |
|---|---|---|
| `read_scratchpad` | — | returns this conversation's content (empty string at first) |
| `write_scratchpad` | `content: str` | replaces the whole content |
| `read_todo` | — | returns this conversation's todo list as JSON (`[]` at first) |
| `update_todos` | `todos: list[TodoItem]` | replaces the whole list — the model re-sends every item, including unchanged ones |

A `TodoItem` is `{"content": str, "status": "pending" | "in_progress" |
"completed" | "cancelled"}` — validated by the tool's `Args` schema, so a bad
status is born INVALID and the body never runs.

**Ids are assigned by the store, not by the model.** `update_todos` matches the
items it is given against the list it is replacing: same content keeps its
number, new content takes the next one. Both todo tools return the numbered
list on `ExecutionResult.structured_content` (a `TodoListResult`), which is
what an application renders — a model asked to preserve its own numbering does
not reliably do so, and a number the user can point at ("complete #2") has to
survive a reordering.

Replacement is the whole lifecycle. Nothing sweeps a finished plan up on a
schedule — the next `update_todos` overwrites it, and since the counter is
never reset, the todo after two completed ones is `#3`, not a second `#1`.
The plugin ships no middleware: the stores are written by the four tools and
by nothing else.

The plugin itself is the whole pattern
([full source](../../luca/agent/contrib/memory/plugin.py)):

```python
class MemoryPlugin:
    def __init__(self, scratchpad_store=None, todo_store=None) -> None:
        self.scratchpad_store: dict = {} if scratchpad_store is None else scratchpad_store
        self.todo_store: dict = {} if todo_store is None else todo_store

    def get_tools(self) -> list[Tool]:          # not a hook — see below
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

> ⚠️ **Two different `get_tools`.** The plugin's takes no arguments and returns
> contrib `Tool` objects (`from luca.agent.contrib.tools import Tool`) — a
> convenience for anyone dropping these tools into their own registry. The
> `ToolRegistry` contract's is
> `async get_tools(session, conversation_id) -> list[ToolSpec]`
> ([03](03-tools.md)); a plugin never implements it, its registry does.

The plugin routes the shared state (one store handed to each tool pair) — the
piece you couldn't express by passing loose tools — and its Yolo registry means
memory tools auto-run regardless of how the app gates its own tools.

Each store is keyed **by conversation**:

```python
self.store.setdefault(conversation_id, {})["content"] = args["content"]
```

One plugin instance is shared by the main agent and every subagent
([13](13-subagents.md)), so a flat dict would mean the parent and its children
overwriting each other's private working memory — silently, the moment a second
conversation exists. Anything a plugin holds needs the same treatment.

### The stores are yours

`MemoryPlugin()` makes its own dicts: everything works and nothing outlives the
process. That is the right default for a script and the wrong one for anything
a user comes back to. **Pass dicts in and they are the memory** — the tools
mutate what they are handed:

```python
plugin = MemoryPlugin(
    scratchpad_store=session.extras.setdefault("scratchpad", {}),
    todo_store=session.extras.setdefault("todos", {}),
)
```

`AgentSession.extras` is a free-form dict the core stores verbatim and never
reads ([02](02-data-model.md)), so a store kept there is saved with the session
and a resumed run picks the plan up mid-flight with its numbering intact — no
sidecar file and no second save path. An application that persists differently
passes its own dicts instead; the plugin cannot tell.

> ⚠️ **Compaction re-keys.** A compaction installs a NEW conversation id, and
> the slots are still filed under the old one. An application that persists the
> stores moves them when it sees `CompactionFinished.new_conversation_id`; one
> that does not, does not care. Nothing in the plugin watches for this — it
> never sees an event.

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
