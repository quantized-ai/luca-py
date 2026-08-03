# Subagents

`luca.agent.contrib.subagents` is the four tools that make parallel subagents
usable, plus the plugin that installs them. The mechanism — child conversations,
the tree of runs, wake rounds, approvals, cancellation — is core and is
documented in [`13-subagents.md`](../../13-subagents.md); this package is the
*policy*: what the model is offered, how a finished child becomes an answer,
and how the model manages the tasks it started.

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
| `StopSubagent` (`stop_subagent`) | asks the runner to cancel one of this turn's tasks. Args: `task_id` (must match), optional `reason`. Best effort — the child reports back one final time as stopped |
| `ListSubagents` (`list_subagents`) | read-only roster of the turn's tasks: id, status (`pending` / `cancelling` / `completed` / `failed`), description, prompt |
| `SPAWNING_PROMPT` / `CONTROL_PROMPT` | the system-prompt texts that teach spawning and task management, each contributed as a **callable** part |
| `SubagentToolRegistry` | `SimpleToolRegistry` + the spawn gate (enabled, depth, spawn budget) + the control-tool withholding (no tasks → no `stop_subagent` / `list_subagents`) |
| `spawn_gate_open(session, conversation_id, *, exclude=None)` | the spawn predicate everything spawn-side derives from |
| `open_turn_children(session, conversation_id)` | the roster read the control side derives from |

All four ship auto-approved, in their own always-allowing registry — the
pattern `MemoryPlugin` uses. That is the right call, not a shortcut: the
dangerous work is not in the spawn, it is in the tools the *child* calls, and
those are gated inside the child by their own registries; stopping only ever
cancels work this conversation itself started, and listing reads. Prompting on
the spawn would ask about the wrapper and never the payload.

## 3. The gate

`spawn_gate_open` is the single predicate — enabled, then the depth cap, then
the per-turn spawn budget:

```python
def spawn_gate_open(session, conversation_id, *, exclude=None) -> bool:
    config = session.session_config.runtime_config
    if not config.subagents_enabled:
        return False
    conversation = session.conversations.get(conversation_id)
    if conversation is None:
        return False
    if config.subagents_max_depth != Inf and conversation.depth >= config.subagents_max_depth:
        return False
    limit = config.subagents_max_per_turn
    return limit == Inf or spawns_committed(session, conversation_id, exclude=exclude) < limit
```

`spawns_committed` (exported by `luca.agent.core`) counts the open turn's
committed subagents from durable entries alone — settled spawns that actually
spawned, plus in-flight spawn calls as reservations — so the gate survives a
reload mid-turn. `exclude` mirrors the core's own gate; no contrib caller
needs it.

Both the tool list and the system prompt derive from it, and that identity is
the whole reason the prompt part is a callable. A static part would tell a
subagent at the cap that it can spawn while `get_tools` withholds the tool — and
the model would try.

`SubagentToolRegistry.get_tools` withholds any spec whose `output_schema`
declares `is_subagent_spawn` — **by declaration, never by name**, which is what
makes a custom spawn tool gate correctly too. The private result tool is never
withheld: the runtime resolves it by name, and the wire filter is the runner's
job.

The control tools ride a **second** prompt/tool-list identity:
`stop_subagent` and `list_subagents` are withheld until the conversation's
open turn has a `ChildConversation`, and `CONTROL_PROMPT`'s part
(`control_prompt_part`) speaks on exactly that predicate. It is a separate
part from the spawning one on purpose — a spent spawn budget silences the
spawn teaching while this turn's tasks still need managing, and one part
gated on either predicate would let the prompt and the tool list disagree.
(Matched by name, not declaration: this is presentation policy over the
registry's own tools, not a cap the runner verifies.)

## 4. The handshake

The contract between this package and the runner is the tool's **structured
output** — declared on the spec, populated on the result:

| | What is read | When | Meaning |
|---|---|---|---|
| **Gate** | `ToolSpec.output_schema` declares `is_subagent_spawn` | before the model call, from the spec alone | this tool *can* spawn → it is subject to the depth cap and the spawn budget |
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
never counted toward context, so the prompt and the result tool's name cost the
parent conversation nothing. The model sees one short status line, carrying the
task id so it can correlate the answer with the task it asked for
(`"Spawned subagent with id a3f01b2c: read the changelog"`) — which matters most
when the model left `task_id` out and the tool made one up.

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

It is **outcome-aware**: a child whose turn closed cancelled / errored /
timed out gets its last words prefixed with the outcome ("The subagent was
cancelled before finishing. Its last message: …") and `is_error=True` — a
stopped task's progress report must not read as an answer. A child with no
final message at all instead hands the parent a readable transcript
(`pretty_print` of that conversation), also with `is_error=True` — a finished
child whose result says what happened, never an exception travelling upward.

## 5b. Stopping

`StopSubagent` mirrors the spawn contract, value-side only: its declared
`SubagentStop` payload (`is_subagent_stop`, `task_id`, `reason`) rides
`structured_content`, and the runner consumes a completed `True` payload by
cancelling the matching direct child — matched only among tasks spawned
*before* the stop was issued (a later task reusing the id is safe); with
duplicate ids the target is the first match still unresolved at issue time,
and a match that resolved after the stop consumes the signal, so a replayed
stop can never fall through to a same-id sibling. The tool
validates against the live session first and returns the payload with the
flag **down** (plus an honest message) for an unknown, finished, or
already-stopping target — exactly like a declining spawn, so the runner does
nothing. The runner re-checks anyway; a child can finish between the tool
body and the handler.

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

Next: [`skills/README.md`](../skills/README.md).
