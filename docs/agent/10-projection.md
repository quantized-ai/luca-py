# Conversation projection

The message list the model sees is **derived state** — recomputed from the
durable session on every call, never stored. The strategy that derives it is
public: `ConversationProjector`. Subclass it to own history policy end to end:
what projects, how tool outcomes read, what gets dropped or injected.

```python
from luca.agent.core import ConversationProjector

runner = AgentSessionRunner(
    session, tool_registry=registry,
    conversation_projector=MyProjector(),     # None → the default projector
)
```

The projector is a runtime collaborator like the permission policy: it lives on
the runner, is never serialized, and is one object — not a stack (plugins don't
contribute projectors). It is **not middleware**; `before_llm_call`
([07](07-middleware.md)) stays downstream for last-mile request edits.

## 1. What the default does

`project(conversation, entries)` walks the conversation path in order and maps
each entry to a canonical `luca.client` message (provider wire formats are the
client's job — the projector never builds OpenAI dicts or Anthropic blocks):

| Entry | Projects to |
|---|---|
| `UserMessage` | client `UserMessage`, content in order — text and image blocks |
| `AssistantMessage` | client `AssistantMessage` — text / thinking / tool-call blocks in order |
| `ToolExecution` (terminal) | one correlated client `ToolMessage` (below) |
| `CompactionEntry` | a synthetic user message carrying the summary |
| `PrunedEntry` | its replacement content, under the *original* entry's role and correlation ([11](11-context-and-usage.md)) |
| `TurnFinish(CANCELLED)` | a synthetic user message: `[Request interrupted by user]` |
| `TurnFinish` (other), `TurnStart`, `CancelRequested` | nothing — bookkeeping |

Every per-entry method takes `(entry, entries)` — the resolved entry plus the
read-only entry mapping, so a projection can resolve cross-entry references
(that's how `project_pruned` finds its original). No merging, trimming, or
token counting happens by default — that's yours to add by overriding
`project`.

## 2. Tool outcomes on the wire

`project_tool_execution(execution, entries)` is the single customization point
for every tool status. `COMPLETED` projects the tool's own `result.content` and preserves
`result.is_error`. Every other terminal status derives error text (always
`is_error=True`) from `status` + the structured `error`:

| Status | Default output |
|---|---|
| `NOT_FOUND` | `Unknown tool: 'read_database'.` |
| `INVALID` | `Arguments for tool 'add' are invalid.` + the validation errors as JSON |
| `FAILED` | `Tool execution failed: ConnectionError: …` |
| `REJECTED` / `CANCELLED` / `INTERRUPTED` / `TIMED_OUT` | `[tool execution rejected]` etc. |

All of this wording lives **on the class** — swap a placeholder without
touching the method, or replace the method wholesale:

```python
class MyProjector(ConversationProjector):
    CANCELLED_TURN_MARKER = "[User stopped the previous request]"
    STATUS_ONLY_OUTPUTS = {
        **ConversationProjector.STATUS_ONLY_OUTPUTS,
        ExecutionStatus.REJECTED: "The user declined this tool call.",
    }
```

## 3. One projection, two consumers

The same `project_tool_execution` output feeds the correlated `ToolMessage` in
the next LLM request **and** the `ToolExecuted` event's `result_text` /
`is_error` — so what your UI renders is exactly what the model is told:

```python
class Redacting(ConversationProjector):
    def project_tool_execution(self, entry, entries):
        message = super().project_tool_execution(entry, entries)
        return message.model_copy(update={"content": redact(message.content)})
```

That is why the projector must be **deterministic** for the same durable
execution: no wall clock, no live registry, no transient state. (The event may
fire now and the request re-project after a reload — they must agree.)

## 4. History policy — override `project`

Trimming, synthetic context, translations — anything that used to be a
"message middleware" belongs here:

```python
class KeepRecent(ConversationProjector):
    def project(self, conversation, entries):
        return super().project(conversation, entries)[-40:]
```

## 5. Rewriting image media

`_image_block(part)` maps an `ImageContent` to the client's `ImageBlock`.
Override it to rewrite media without touching the rest of the projection —
uploading base64 bytes once and sending an id instead, for example:

```python
class Uploading(ConversationProjector):
    def _image_block(self, part):
        return ImageBlock(source=MediaFileId(file_id=upload(part.source)))
```

`part.metadata` is application-owned and is dropped on the way to the wire.

## 6. Fail-loud rules

Projection never papers over broken state — errors raise `ProjectionError`
instead of producing invented content:

- a conversation node missing from the entry store;
- an entry type the projector doesn't know;
- a `PENDING` or `RUNNING` tool execution (the runtime never calls the model
  mid-execution);
- a `COMPLETED` execution without a result;
- a `PrunedEntry` whose referent is missing, whose `pruned_entry_type`
  disagrees with the referent, or whose referent has no pruned projection.

> ⚠️ **Correlation is sacred.** Every projected `ToolMessage` must keep
> `execution.tool_call_id` as its `tool_call_id`, and every model tool call
> must end up with exactly one correlated output. Rewrite content freely;
> never drop or re-key a tool message.

Next: [`11-context-and-usage.md`](11-context-and-usage.md).
