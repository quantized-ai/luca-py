# Subagents

`luca.agent.contrib.subagents` is the two tools that make parallel subagents
usable, plus the plugin that installs them. The mechanism — child conversations,
the tree of runs, approvals, cancellation — is core and is documented in
[`13-subagents.md`](../../13-subagents.md); this package is the *policy*: what
the model is offered, and how a finished child becomes an answer.

```python
from luca.agent.contrib.subagents import SubagentsPlugin
```

## 1. Install

```python
from luca.agent.contrib.plugins import PluginAgentSessionRunner
from luca.agent.core import AgentSessionRunner, LLMConfig, RuntimeConfig

session = AgentSessionRunner.new_session(
    LLMConfig(model="openai/gpt-4o-mini", provider="openrouter"),
    runtime_config=RuntimeConfig(subagents_enabled=True),     # ← required
)
runner = PluginAgentSessionRunner(session, plugins=[SubagentsPlugin(), *others])
```

> ⚠️ **Installing is not enabling.** `subagents_enabled` defaults to `False`,
> and with it off the spawn tool is withheld and the prompt part stays silent.
> The capability is configuration, not installation.

## 2. What it ships

| | |
|---|---|
| `SpawnSubagent` (`spawn_subagent`) | starts one subagent. Args: `prompt` (self-contained instructions), `description` (for the user), optional `task_id` |
| `CreateConversationResult` (`create_conversation_result`) | **private** — the runtime calls it when a child finishes, to turn that conversation into one result |
| `SPAWNING_PROMPT` | the system-prompt text that teaches spawning, contributed as a **callable** part |
| `SubagentToolRegistry` | `SimpleToolRegistry` + the depth gate |
| `spawn_gate_open(session, conversation_id)` | the one predicate everything derives from |

Both tools ship auto-approved, in their own always-allowing registry — the
pattern `MemoryPlugin` uses. That is the right call, not a shortcut: the
dangerous work is not in the spawn, it is in the tools the *child* calls, and
those are gated inside the child by their own registries. Prompting on the spawn
would ask about the wrapper and never the payload.

## 3. The gate

`spawn_gate_open` is the single predicate:

```python
def spawn_gate_open(session, conversation_id) -> bool:
    config = session.session_config.runtime_config
    return (
        config.subagents_enabled
        and session.conversations[conversation_id].depth < config.subagents_max_depth
    )
```

Both the tool list and the system prompt derive from it, and that identity is
the whole reason the prompt part is a callable. A static part would tell a
subagent at the cap that it can spawn while `get_tools` withholds the tool — and
the model would try.

`SubagentToolRegistry.get_tools` withholds any spec whose `output_schema`
declares `is_subagent_spawn` — **by declaration, never by name**, which is what
makes a custom spawn tool gate correctly too. The private result tool is never
withheld: the runtime resolves it by name, and the wire filter is the runner's
job.

## 4. The handshake

The contract between this package and the runner is the tool's **structured
output** — declared on the spec, populated on the result:

| | What is read | When | Meaning |
|---|---|---|---|
| **Gate** | `ToolSpec.output_schema` declares `is_subagent_spawn` | before the model call, from the spec alone | this tool *can* spawn → it is subject to the depth cap |
| **Handshake** | `structured_content["is_subagent_spawn"] is True` | after the execution completes | this call *did* spawn → create the child |

```python
class SubagentSpawn(BaseModel):
    is_subagent_spawn: bool = True
    task_id: str
    prompt: str
    description: str
    process_subagent_result_tool_name: str
```

The asymmetry is what the `bool` buys: a spawn tool that decides at runtime *not*
to spawn returns the payload with `is_subagent_spawn=False` and no child is
created — while still being gated, because the gate read the declaration rather
than the outcome.

The payload is **free**: `structured_content` never reaches the model and is
never counted toward context, so the prompt, the task id and the result tool's
name cost the parent conversation nothing. The model sees one short status line
(`"Spawned subagent: read the changelog"`).

Which tool derives the result travels *in the payload*, so an application can
pair its own spawn tool with its own summarizer without the core knowing either
name:

```python
SubagentsPlugin(result_tool_name="my_summarizer")
```

## 5. What a finished child returns

`CreateConversationResult` takes the child's **last assistant message**, text
parts only — reasoning and tool calls are the child's working, not its answer.
The last *node* is the closing turn marker, so it searches backwards for the
message rather than reading the end of the path.

A child that was cancelled, errored, or ran out of steps has no final message.
Instead of returning nothing it hands the parent a readable transcript
(`pretty_print` of that conversation) with `is_error=True` — a finished child
whose result says what happened, never an exception travelling upward.

## 6. Coexistence

Nothing here negotiates with the other plugins. There is no global permission
gate anywhere: each registry answers `decide()` for its own tools, and
`PluginAgentSessionRunner` makes every plugin registry a child of one
`ProxyToolRegistry`. Shell keeps its `PermissionStrategy`, memory keeps Yolo,
subagents keeps Yolo.

> ⚠️ **Tool names are globally unique.** `ProxyToolRegistry` routes by
> `spec.name`; `namespace` does not disambiguate. The failure is loud — the
> proxy raises on a duplicate — so it is a naming rule, not a hazard.

The demo wires all of this by default: `uv run python main.py`
([`tui/`](../tui/README.md)).

Next: [`simple_context_manager/README.md`](../simple_context_manager/README.md).
