# Quickstart — build an agent, step by step

A guided tour for someone who has never used `luca`. You'll start with an agent
that can only talk, then add tools, swap models, write a system prompt, gate
tool calls behind real permissions, and install ready-made capabilities
(memory, shell access). Every step is a complete program you can run, followed
by what you should see when you do.

**The whole mental model, up front:**

> **`AgentSession` is data. `AgentSessionRunner` is behavior.**
> The session is one Pydantic object holding the entire conversation —
> messages, tool calls, tool results, reasoning, turn boundaries. It serializes
> to JSON and reloads to resume exactly where it stopped. The runner is the
> async engine that drives it forward: call the model, record the answer, run
> the tools it asked for, loop. Everything transient — the tool registry, the
> live model call, cancellation — lives on the runner, never in the session.

Two more things worth knowing before the code:

- **The runner is async.** There is no synchronous agent loop; you drive it
  from `asyncio.run(...)`.
- **Everything here is non-streaming**, so you see whole events arriving one at
  a time instead of a token firehose. [Section 12](#12-streaming) shows the
  one-flag switch to token deltas.

| | |
|---|---|
| [1. Setup](#1-setup) | [7. Contrib: where the rest lives](#7-contrib--where-everything-else-lives) |
| [2. Your first agent](#2-your-first-agent-no-tools) | [8. Real permissions](#8-real-permissions) |
| [3. Add tools](#3-add-tools) | [9. The memory plugin](#9-the-memory-plugin--built-in-tools) |
| [4. Switch models and providers](#4-switch-models-and-providers) | [10. Shell tools](#10-shell-tools) |
| [5. The system prompt](#5-the-system-prompt) | [11. Extras](#11-extras) |
| [6. Save and resume](#6-save-and-resume) | [13. Where to go next](#13-where-to-go-next) |

---

## 1. Setup

```bash
pip install luca-ai          # or: uv add luca-ai
```

The agent needs an API key for whichever model host you use. Put it in a `.env`
file next to your script:

```bash
# .env
OPENROUTER_API_KEY=sk-or-v1-…
```

`luca` reads the key from the environment; it does not load `.env` for you.
Either export the variable yourself, or install `python-dotenv` and load it at
startup (that's what every example below assumes):

```bash
pip install python-dotenv
```

```python
from dotenv import load_dotenv
load_dotenv()
```

Which variable you need depends on the `provider` you configure — see the table
in [section 4](#4-switch-models-and-providers). The examples use
`provider="openrouter"`, so `OPENROUTER_API_KEY`.

---

## 2. Your first agent (no tools)

The smallest useful agent is three objects: an `LLMConfig` (which model),
an `AgentSession` (the state), and an `AgentSessionRunner` (the engine).

```python
# agent.py
import asyncio

from dotenv import load_dotenv

from luca.agent.core import AgentSessionRunner, LLMConfig
from luca.agent.core.events import FinishReason, TextBlock

load_dotenv()


async def main() -> None:
    # 1. A fresh, empty session configured for one model.
    session = AgentSessionRunner.new_session(
        LLMConfig(model="openai/gpt-4o-mini", provider="openrouter"),
    )

    # 2. The engine that drives that session.
    runner = AgentSessionRunner(session)

    # 3. Queue a user message, then drive one turn and print the events.
    runner.post_message("Say hi and tell me what you are, in two sentences.")

    async with runner.run() as run:
        async for event in run:
            match event:
                case TextBlock(text=text):
                    print(f"[text] {text}")
                case FinishReason(finish_reason=reason):
                    print(f"[finish] {reason}")


asyncio.run(main())
```

```bash
python agent.py
```

### What you'll see

```
[text] Hi! I'm an AI assistant running inside a luca agent. I can hold a
conversation and, once you give me tools, use them to get real work done.
[finish] stop
```

Two events, in this order:

| Event | Meaning |
|---|---|
| `TextBlock` | One complete block of assistant text. Fires once per text block in the answer. |
| `FinishReason` | Why the model stopped (`"stop"` for a normal answer, `"tool_use"` when it asked for tools). Always the last event of a model response. |

If the model produced reasoning, a `ReasoningBlock` would arrive before the
text. That's the entire non-streaming event vocabulary for a toolless agent.

### What just happened

`post_message` appended a user message and armed the runner. `run()` returned a
**handle** — nothing ran yet. Iterating that handle *is* the engine: each
`async for` step pulls the agent forward. When the model answered without
asking for tools, the turn closed and iteration ended.

Look at the session afterwards and you'll find the whole thing recorded:

```python
for entry in session.entries.values():
    print(entry.type, entry.id)
```

```
user       c0a9cdf1
turn_start 6e418080
assistant  2473f7b8
turn_finish 1b5802ea
```

Everything the agent does becomes an **entry** in `session.entries`. A turn is
just a `turn_start` / `turn_finish` pair bracketing the work in between — there
is no "Turn" object. `session.active_conversation.nodes` is the ordered list of
entry ids that makes up the current conversation.

For a readable dump of any session, use the built-in transcript printer:

```python
from luca.agent.core import pretty_print
print(pretty_print(session))
```

```
LUCA SESSION 4cdb5f25
Conversation d4521e80 · idle · 1 turn
Default: openrouter/openai/gpt-4o-mini
────────────────────────────────────────────────────────────────

TURN 1 · 2026-07-28 10:10:39
User
  Say hi and tell me what you are, in two sentences.

Assistant · step 1 · openrouter/openai/gpt-4o-mini
  Hi! I'm an AI assistant running inside a luca agent. …

✓ completed · stop · 0 tokens
```

### Status — how you know what to do next

The runner always knows what it needs from you. Poll it:

```python
runner.status          # ConversationStatus enum
runner.idle()          # nothing queued → post_message()
runner.pending()       # work queued → run()
runner.awaiting_approval()   # paused at a tool-approval gate → resolve, then run()
runner.cancelling()    # a cancel is parked → run() flushes it
```

A fresh session is `IDLE`; `post_message` makes it `PENDING`; a finished turn
returns to `IDLE`. That's the loop a real app polls, and it's the same loop
across a process restart, because status is re-derived from the entries when a
runner takes ownership of a session.

---

## 3. Add tools

A tool is a Python class: a name, a description, a Pydantic `Args` model (which
becomes the JSON Schema the model is shown), and an `_execute` body.

Tools don't go to the runner directly. They go into a **tool registry**, which
owns the whole tool lifecycle — listing, resolving, validating arguments,
approving, dispatching. `SimpleToolRegistry` is the batteries-included one, and
`YoloPermissionPolicy` is its "approve everything" policy (we'll replace that in
[section 8](#8-real-permissions)).

```python
# agent_tools.py
import asyncio

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from luca.agent.contrib.simple_tool_registry import SimpleToolRegistry, YoloPermissionPolicy
from luca.agent.contrib.tools import Tool
from luca.agent.core import AgentSession, AgentSessionRunner, CancellationToken, LLMConfig
from luca.agent.core.events import FinishReason, TextBlock, ToolCallReceived, ToolExecuted, ToolExecutionStarted

load_dotenv()


class AddArgs(BaseModel):
    a: float = Field(description="First number.")
    b: float = Field(description="Second number.")


class AddTool(Tool):
    name = "add"
    description = "Add two numbers and return the sum."
    Args = AddArgs

    async def _execute(
        self, args: dict, session: AgentSession, *, cancellation_token: CancellationToken,
    ) -> str:
        return str(args["a"] + args["b"])


class MultiplyArgs(BaseModel):
    a: float = Field(description="First number.")
    b: float = Field(description="Second number.")


class MultiplyTool(Tool):
    name = "multiply"
    description = "Multiply two numbers and return the product."
    Args = MultiplyArgs

    async def _execute(
        self, args: dict, session: AgentSession, *, cancellation_token: CancellationToken,
    ) -> str:
        return str(args["a"] * args["b"])


async def main() -> None:
    session = AgentSessionRunner.new_session(
        LLMConfig(model="openai/gpt-4o-mini", provider="openrouter"),
    )
    registry = SimpleToolRegistry(
        tools=[AddTool(), MultiplyTool()],
        permission_policy=YoloPermissionPolicy(),   # approve every call
    )
    runner = AgentSessionRunner(session, tool_registry=registry)

    runner.post_message("Add 21 and 21, then double the result.")

    async with runner.run() as run:
        async for event in run:
            match event:
                case ToolCallReceived(execution=ex):
                    call = ex.raw_tool_call
                    print(f"[call] {call.name}({call.arguments})")
                case ToolExecutionStarted(execution=ex):
                    print(f"[start] {ex.raw_tool_call.name}")
                case ToolExecuted(execution=ex, result_text=text, is_error=err):
                    print(f"[done] {ex.raw_tool_call.name} -> {text}{' (error)' if err else ''}")
                case TextBlock(text=text):
                    print(f"[text] {text}")
                case FinishReason(finish_reason=reason):
                    print(f"[finish] {reason}")


asyncio.run(main())
```

### What you'll see

```
[finish] tool_use
[call] add({'a': 21, 'b': 21})
[start] add
[done] add -> 42.0
[finish] tool_use
[call] multiply({'a': 42, 'b': 2})
[start] multiply
[done] multiply -> 84.0
[text] 21 + 21 = 42, and doubling that gives 84.
[finish] stop
```

**One `run()` drove three model calls and two tool executions.** The agent loops
on its own: model → tool calls → results → model → … until the model answers
without asking for a tool.

The order within each round is exact and worth memorizing:

| Order | Event | Fires |
|---|---|---|
| 1 | `ReasoningBlock` / `TextBlock` | per block of the model's answer |
| 2 | `FinishReason` | once, closing that model response |
| 3 | `ToolCallReceived` | once per requested call, right after it's recorded |
| 4 | `ToolExecutionStarted` | only if the tool body actually gets dispatched |
| 5 | `ToolExecuted` | once per call, at its final outcome |

> ⚠️ **`ToolExecuted` fires for every outcome, not just success.** A tool that
> fails, is denied, times out, or is cancelled still produces exactly one
> `ToolExecuted`. Check `is_error` and `execution.status`. The invariant is
> absolute: **every tool call produces exactly one tool result** — otherwise the
> next model call would be malformed.

`result_text` is what the *model* will be told; `execution` is what was
*recorded*. They come from the same place, so your UI can never disagree with
the model's view.

### About the tool body

```python
async def _execute(self, args, session, *, cancellation_token) -> str
```

- `args` is a plain dict, already validated against `Args`.
- `session` is the live session — **read-only**. The runner owns every write.
  Read `session.id` or `session.session_config.llm_config` from it. Per-call
  state belongs in your own object (`self`), not the session.
- `cancellation_token` lets a long-running tool notice it should stop early.
- Return a string for the simple path. Override `execute` instead if you need a
  rich result (`is_error=True`, metadata, multiple content blocks).

---

## 4. Switch models and providers

Everything about model choice lives in one object, stored on the session:

```python
from luca.agent.core import LLMConfig

LLMConfig(
    model="anthropic/claude-sonnet-4.5",   # the model id, as the host names it
    provider="openrouter",                 # which host to call
    reasoning="medium",                    # optional reasoning effort
)
```

Same agent as before — only the config changes:

```python
# Anthropic directly, with reasoning
session = AgentSessionRunner.new_session(
    LLMConfig(model="claude-sonnet-4-5", provider="anthropic", reasoning="high"),
)

# OpenAI directly
session = AgentSessionRunner.new_session(
    LLMConfig(model="gpt-4o-mini", provider="openai"),
)

# A local Ollama server — no key needed
session = AgentSessionRunner.new_session(
    LLMConfig(model="llama3.1", provider="ollama"),
)
```

| `provider` | Env var it reads |
|---|---|
| `openrouter` | `OPENROUTER_API_KEY` |
| `openai` | `OPENAI_API_KEY` |
| `anthropic` | `ANTHROPIC_API_KEY` |
| `bedrock` | `AWS_BEARER_TOKEN_BEDROCK` (+ `AWS_REGION`) |
| `groq` | `GROQ_API_KEY` |
| `deepseek` | `DEEPSEEK_API_KEY` |
| `ollama` | — (local) |

`reasoning` accepts `"none"`, `"minimal"`, `"low"`, `"medium"`, `"high"`,
`"xhigh"`, or `"provider-default"`. `None` (the default) sends nothing. Hosts
that don't support reasoning ignore it.

**Switching mid-session** is a plain assignment — the config is session state,
read fresh before every model call:

```python
session.session_config.llm_config = LLMConfig(
    model="openai/gpt-4o-mini", provider="openrouter",   # cheap model for the next turn
)
```

Each assistant entry records the config that produced it, so a session that
switched models halfway stays auditable.

### What you'll see with reasoning on

```
[reasoning] The user wants 6 times 7. That's 42.
[text] 42.
[finish] stop
```

```python
case ReasoningBlock(text=text):
    print(f"[reasoning] {text}")
```

Reasoning is durable — it's stored in the session and survives a reload.

---

## 5. The system prompt

There is no `system_prompt=` string. The runner takes **`system_prompt_parts`**:
a list of fragments, resolved fresh before every model call, sorted, and joined.

Why parts instead of one string? Because real prompts come from several places
at once — your persona, the project's rules file, a plugin's tool instructions,
a runtime nudge — and each one wants to be owned and ordered independently.

```python
from luca.agent.core import AgentSessionRunner, SystemPromptPart

runner = AgentSessionRunner(
    session,
    tool_registry=registry,
    system_prompt_parts=[
        # 1. a full part: text + provenance + explicit ordering
        SystemPromptPart(text="You are Ada, a terse assistant.", source="persona", priority=0),

        # 2. a dict — same thing, validated strictly
        {"text": "Always answer in one sentence.", "source": "house-rules", "priority": 10},

        # 3. a bare string — priority defaults to -1 ("unranked", sorts first)
        "Prefer tools over guessing.",

        # 4. a callable — re-evaluated before every model call
        wrap_up_nudge,
    ],
)
```

A callable part receives the live session config and a freshly computed runtime
status, so the prompt can react to what the turn is doing:

```python
def wrap_up_nudge(session_config, runtime_status):
    if runtime_status.step_count > 8:                       # model calls in this turn
        return {"text": "You've taken many steps — converge to an answer.", "priority": 99}
    return ""                                               # contributes nothing
```

`runtime_status` carries `step_count`, `turn_count` and `status`.

### What you'll see

Inspect the assembled prompt at any time:

```python
print(runner.build_system_message())
```

```
Prefer tools over guessing.

You are Ada, a terse assistant.
Always answer in one sentence.
```

Parts sort by `priority` ascending (`-1` first), then join with newlines. The
blank line is the empty part the callable returned. Omit `system_prompt_parts`
entirely and the runner sends **no system message at all** — a perfectly valid
agent.

Need different assembly (XML tags per source, deduplication)? Pass a
`system_prompt_assembler=` with your own `assemble_system_prompt(parts) -> str`.

---

## 6. Save and resume

The session *is* the state. Save it as JSON; reload it and keep going:

```python
# save
open(f"{session.id}.json", "w").write(session.model_dump_json(indent=2))

# resume — the runner re-derives the status from the entries
from luca.agent.core import AgentSession

session = AgentSession.model_validate_json(open("4cdb5f25.json").read())
runner = AgentSessionRunner(session, tool_registry=registry)
runner.post_message("What did I ask you before?")
```

The registry is **not** saved — it's a runtime collaborator, so you supply it
again when you rebuild the runner. This is also why a crashed process recovers
cleanly: a session that died mid-turn reloads into the right status and the next
`run()` picks up where it stopped.

---

## 7. Contrib — where everything else lives

Everything so far except the tools came from `luca.agent.core`. That's the whole
core: the data model, the runner, and a handful of strategy contracts. It's
deliberately small and stable.

**Everything else is `luca.agent.contrib`** — optional packages built on the
core's public surface, exactly the way *your* application code is. Nothing in
the core imports from contrib. You already used two of them without noticing:
`contrib.tools` (the `Tool` base class) and `contrib.simple_tool_registry`.

| What the core knows | What contrib supplies |
|---|---|
| `ToolSpec` — plain JSON tool data | `Tool` — the ergonomic Python tool class |
| `ToolRegistry` — a 4-method contract | `SimpleToolRegistry`, `ProxyToolRegistry` |
| nothing about approval | `PermissionPolicy`, rule-based `PermissionStrategy` |
| nothing about plugins | `BasePlugin`, `PluginAgentSessionRunner` |
| `CompactionPolicy` — a contract | `SummarizingCompactionPolicy` |
| — | the shell tools, the memory plugin, a Textual TUI |

The practical consequence: **the framework is a specification, not a
dependency**. A registry that fronts a remote tool server never imports `Tool`.
A permission model you don't like gets replaced with your own. The rest of this
guide is contrib — and it's all code you could have written yourself against the
same public surface.

| Package | What it gives you |
|---|---|
| `contrib.tools` | `Tool`, `tool()`, `tool_class()` |
| `contrib.simple_tool_registry` | `SimpleToolRegistry`, `ProxyToolRegistry`, `YoloPermissionPolicy` |
| `contrib.resource_permissions` | Rule-based, interactive approvals ([§8](#8-real-permissions)) |
| `contrib.plugins` | Bundle a registry + prompt parts + middleware in one object |
| `contrib.memory` | Scratchpad + todo-list tools ([§9](#9-the-memory-plugin--built-in-tools)) |
| `contrib.shell` | read / glob / grep / edit / write / apply_patch / bash ([§10](#10-shell-tools)) |
| `contrib.compaction` | A ready-made summarizing compaction policy |
| `contrib.tui` | A full Textual terminal UI |

---

## 8. Real permissions

`YoloPermissionPolicy` approved everything. A real agent needs to ask. The
`resource_permissions` package models approval as **coverage of
`(permission, resource)` pairs**:

- A tool declares what a call needs: `("write", "/tmp/note.txt")`.
- Rules, and answers your user gives, grant pairs — often as globs
  (`("write", "/tmp/*")`).
- A call runs only when **every** required pair is covered by an ALLOW. One DENY
  kills it; one uncovered pair leaves it pending.

### The tool side: `get_approval_context`

A tool describes its own approval needs by mixing in
`ResourcePermissionToolMixin` and implementing one method. The mixin turns your
typed requests into the approval context the strategy reads (it serializes them
into `execution.extras["approval_context"]` — the free-form channel registries
use to pass anything they want to a policy).

```python
from pathlib import Path

from luca.agent.contrib.resource_permissions import (
    AnswerOption, PermissionRequest, ResourcePermission, ResourcePermissionToolMixin,
)
from luca.agent.contrib.tools import Tool
from luca.agent.core import ToolKind


class WriteNoteArgs(BaseModel):
    path: str = Field(description="Where to write the note.")
    text: str = Field(description="What to write.")


class WriteNoteTool(ResourcePermissionToolMixin, Tool):
    name = "write_note"
    description = "Write a short note to a file."
    Args = WriteNoteArgs
    tool_kind = ToolKind.EDIT

    def build_permission_requests(self, args, session):
        return [PermissionRequest(
            # what this call requires
            resources=[ResourcePermission(permission="write", resource=args["path"])],
            # grants worth offering the user, usually broader than the call
            answer_options=[AnswerOption(
                resource_permissions=[ResourcePermission(permission="write", resource="/tmp/*")],
                metadata={"preview": "Always allow writes under /tmp"},
            )],
            metadata={"preview": f"Write {len(args['text'])} chars to {args['path']}"},
        )]

    async def _execute(self, args, session, *, cancellation_token) -> str:
        Path(args["path"]).write_text(args["text"])
        return f"wrote {args['path']}"
```

`metadata` is **UX only** — previews and labels for your prompt. The strategy
never reads it.

> ⚠️ **`build_permission_requests` is synchronous on purpose.** It runs in a
> worker thread, because deciding what a call needs permission for is usually
> filesystem work, and a blocking syscall on the event loop would stall the
> whole agent.

### The app side: the approval loop

```python
from luca.agent.contrib.resource_permissions import (
    AnswerDecision, AnswerOption, AnswerScope, ApprovalAnswer,
    PermissionMode, PermissionStrategy, ResourcePermission,
)
from luca.agent.contrib.simple_tool_registry import SimpleToolRegistry
from luca.agent.core.events import ApprovalRequired

strategy = PermissionStrategy(mode=PermissionMode.ASK)      # keep your own reference!
registry = SimpleToolRegistry(tools=[WriteNoteTool()], permission_policy=strategy)
runner = AgentSessionRunner(session, tool_registry=registry)

runner.post_message("Write 'hello' to /tmp/note.txt")

async with runner.run() as run:
    async for event in run:
        match event:
            case ToolCallReceived(execution=ex):
                print(f"[call] {ex.raw_tool_call.name}({ex.raw_tool_call.arguments})")
            case ApprovalRequired(executions=execs):
                print(f"[gate] {len(execs)} call(s) need approval")
            case ToolExecuted(result_text=text):
                print(f"[done] {text}")
            case TextBlock(text=text):
                print(f"[text] {text}")

# the run stopped at the gate — resolve it
if runner.awaiting_approval():
    for execution in runner.pending_approvals():
        for request in strategy.pending_requests(execution):     # only uncovered steps
            print("  ?", request.metadata["preview"])
            for option in request.answer_options:
                print("    -", option.metadata["preview"])

        # your UI decides; here we accept the suggested "always" grant
        strategy.apply_answer(execution, [ApprovalAnswer(
            answer_option=AnswerOption(resource_permissions=[
                ResourcePermission(permission="write", resource="/tmp/*"),
            ]),
            decision=AnswerDecision.APPROVE,
            scope=AnswerScope.ALWAYS,
        )])

# run again — the agent picks up exactly where it paused
async with runner.run() as run:
    async for event in run:
        ...
```

### What you'll see

First run:

```
[finish] tool_use
[call] write_note({'path': '/tmp/note.txt', 'text': 'hello'})
[gate] 1 call(s) need approval
  ? Write 5 chars to /tmp/note.txt
    - Always allow writes under /tmp
```

The run **ends there**. `ApprovalRequired` is always the last event before a
gate, and `runner.status` is now `awaiting_approval`.

Second run, after `apply_answer`:

```
[start] write_note
[done] wrote /tmp/note.txt
[text] Done — the note is written.
[finish] stop
```

The paused turn resumed — same `turn_start`, no re-asking the model. Because the
answer was `ALWAYS`-scoped, the strategy now holds a persistent rule:

```python
print(strategy.rules)
# [ToolRule(tool_name='write_note',
#           resource_permission=ResourcePermission(permission='write', resource='/tmp/*'),
#           decision=ApprovalOption.ALLOW)]
```

The next `/tmp` write won't prompt at all.

### Three things to internalize

1. **Answers are not replies.** An `ApprovalAnswer` is a free-standing verdict
   over pairs — no ids, no addressing. The tool's `answer_options` are
   suggestions; you can construct your own. Whether a call runs is *emergent*
   from whether its pairs ended up covered.
2. **Recording an answer doesn't advance anything.** The session stays
   `AWAITING_APPROVAL` until the next `run()` asks the strategy again. An answer
   that covers too little just means you get asked again — the failure mode is a
   re-ask, never a false approval.
3. **`mode=PermissionMode.YOLO`** promotes anything uncovered to ALLOW while
   explicit DENY rules still block. Same code, no prompts.

Seed rules up front to skip whole categories:

```python
from luca.agent.core import ApprovalOption
from luca.agent.contrib.resource_permissions import ToolKindRule, ToolRule

strategy = PermissionStrategy(
    mode=PermissionMode.ASK,
    rules=[
        ToolKindRule(tool_kind=ToolKind.READ, decision=ApprovalOption.ALLOW),   # all reads
        ToolRule(                                                              # …except these
            resource_permission=ResourcePermission(permission="read", resource="/etc/*"),
            decision=ApprovalOption.DENY,
        ),
    ],
)
```

Rules are one ordered list and the **last** match wins.

---

## 9. The memory plugin — built-in tools

A **plugin** bundles what a capability usually ships together — a tool registry,
system-prompt parts, and middleware — behind one object. Install plugins with
`PluginAgentSessionRunner`, a drop-in replacement for `AgentSessionRunner`:

```python
from luca.agent.contrib.memory import MemoryPlugin
from luca.agent.contrib.plugins import PluginAgentSessionRunner

memory = MemoryPlugin()
runner = PluginAgentSessionRunner(session, plugins=[memory])
```

That's the whole installation. The plugin contributes four tools and the prompt
text that teaches the model to use them:

| Tool | Does |
|---|---|
| `read_scratchpad` / `write_scratchpad` | A private working-memory string; each write replaces it entirely |
| `read_todo` / `update_todos` | A task list; `update_todos` replaces the whole list in one call |

```python
print([t.name for t in memory.get_tools()])
# ['read_scratchpad', 'write_scratchpad', 'read_todo', 'update_todos']

print(runner.build_system_message())
# The following tools are available:
# ### Scratchpad (read_scratchpad / write_scratchpad)
# Your private working memory.
# …
# ### Todo list (read_todo / update_todos)
# …
```

### What you'll see

```
runner.post_message("Track 'Buy milk' on my todo list.")
```

```
[finish] tool_use
[call] update_todos({'todos': [{'content': 'Buy milk', 'status': 'pending'}]})
[start] update_todos
[done] update_todos -> Todo list updated successfully
[text] Added to the list.
[finish] stop
```

The state lives on the plugin instance, not the session:

```python
print(memory.todo_store)
# {'todos': [{'content': 'Buy milk', 'status': 'pending'}]}
```

Plugins compose — `plugins=[memory, shell, my_plugin]` merges every registry
behind one router and concatenates every prompt part. Writing one is just a
plain class with any of three methods: `get_tool_registry(session)`,
`get_system_prompt_parts(session)`, `get_middleware(session)`.

> **A plugin is only construction-time sugar.** A runner built with plugins is
> equivalent to one built by composing the same objects by hand.

---

## 10. Shell tools

`ShellAccessPlugin` gives the agent a real filesystem and a shell — seven tools
scoped to a workspace directory, with permissions already wired.

```python
from pathlib import Path

from luca.agent.contrib.memory import MemoryPlugin
from luca.agent.contrib.plugins import PluginAgentSessionRunner
from luca.agent.contrib.shell import ShellAccessPlugin

shell = ShellAccessPlugin(workspace=Path("."))     # mode="ask" by default
runner = PluginAgentSessionRunner(session, plugins=[shell, MemoryPlugin()])
```

| Tool | Kind | Does |
|---|---|---|
| `read` | READ | Numbered text pages, directory listings, images |
| `glob` | SEARCH | Find files by pattern |
| `grep` | SEARCH | Search file contents |
| `edit` | EDIT | Exact-match replacement, returns a diff |
| `write` | EDIT | Full-content write, creates parent dirs |
| `apply_patch` | EDIT | Multi-file patch envelope, verify-then-commit |
| `bash` | EXECUTE | Fresh shell per call; timeout and cancellation kill the process group |

### What you'll see

```
runner.post_message("What is in hello.txt?")
```

```
[finish] tool_use
[call] read({'file_path': 'hello.txt'})
[start] read
[done] read -> <path>/Users/you/project/hello.txt</path>
<type>file</type>
<content>
1: hello from luca

(End of file - total 1 lines)
</content>
[text] The file says: hello from luca
```

**No approval prompt** — and that's the interesting part. The plugin seeds ALLOW
rules for the whole read tier (`read`, `glob`, `grep`, plus directory access)
over the workspace and everything beneath it. Every call declares two steps:

```
access_directory <the directory the call touches>
<verb> <resource>                    # read hello.txt, bash "git status", …
```

| Call | Prompts for |
|---|---|
| `read tests.py` inside the workspace | nothing — covered by the seeded rules |
| `edit tests.py` inside the workspace | the `edit` step only |
| `read ../../secrets.txt` outside it | both steps |

The plugin exposes its strategy so your approval prompt is the exact same loop
from [section 8](#8-real-permissions):

```python
strategy = shell.permission_strategy
for execution in runner.pending_approvals():
    steps = strategy.pending_requests(execution)     # only the uncovered steps
    strategy.apply_answer(execution, ask_user(execution, steps))
```

Two behaviors worth knowing:

- **Read-first contract.** `edit` and `write` refuse to modify an existing file
  the agent never `read`. The plugin shares one tracker across the tools to make
  that work.
- **Domain failures are results, not crashes.** A missing file or a non-zero
  exit comes back as `ExecutionResult(is_error=True)` — the model sees it and
  can react, and the run continues.

> ⚠️ **A permission gate is not a sandbox.** An approved `bash` command can
> touch anything the process can. `mode="yolo"` is full-disk access for all
> seven tools.

---

## 11. Extras

### 11a. `ContextManager` — reshape what tools return

Tool output is often huge, and the model pays for all of it. The
`ContextManager` is the runner's context-accounting strategy, and its
`process_tool_output` hook transforms a result *before* it becomes durable — the
saved session, the `ToolExecuted` event, and the model's view all see the
processed version.

```python
from luca.agent.core import ContextManager, ExecutionResult, TextContent


class TruncatingContextManager(ContextManager):
    MAX_CHARS = 30_000

    def process_tool_output(self, session, execution, result):
        # branch by tool: truncate `bash`, never touch `read`
        if execution.raw_tool_call.name != "bash":
            return result
        content = []
        for part in result.content:
            if isinstance(part, TextContent) and len(part.text) > self.MAX_CHARS:
                dropped = len(part.text) - self.MAX_CHARS
                content.append(TextContent(
                    text=f"{part.text[: self.MAX_CHARS]}\n… [{dropped} characters truncated]",
                ))
            else:
                content.append(part)
        return ExecutionResult(content=content, metadata=result.metadata, is_error=result.is_error)


runner = AgentSessionRunner(session, tool_registry=registry,
                            context_manager=TruncatingContextManager())
```

```
[done] bash -> total 48
drwxr-xr-x  12 you  staff   384 Jul 28 10:11 .
… [109431 characters truncated]
```

> ⚠️ **The execution you receive is mid-transition.** Read it for *identity* —
> `raw_tool_call.name`, `raw_tool_call.arguments`, `tool_spec` — never for
> outcome: its status is still `RUNNING` and the result isn't attached yet.

The same class also owns `calculate_context(session, entry)`, which stamps each
entry's estimated `context_tokens` (the default is one token per 4 characters).
Override it to plug in a real tokenizer. That number is what a compaction policy
usually gauges against — which is the next section.

### 11b. `CompactionPolicy` — surviving a full context window

Long conversations eventually fill the model's context window. **Compaction**
replaces the older span with a stand-in and keeps going. Nothing is destroyed:
compaction opens a *new conversation inside the same session*, archives the old
path in `conversation_history`, and records exactly which entries were replaced.

The policy owns the entire decision: when to compact, what replaces the old
span, and what survives. Here is the simplest possible one — it doesn't even
call a model, it just drops the oldest entries and leaves a note:

```python
from luca.agent.core import CompactionPlan, CompactionPolicy, TextContent


class DropOldestCompactionPolicy(CompactionPolicy):
    """The most basic compaction there is: keep the last N nodes, replace
    everything older with a one-line placeholder."""

    def __init__(self, *, max_context_tokens: int = 100_000, keep_last: int = 6) -> None:
        self.max_context_tokens = max_context_tokens
        self.keep_last = keep_last

    def should_compact(self, session) -> bool:
        # Consulted at the top of every run. Your gauge, your threshold.
        nodes = session.active_conversation.nodes
        used = sum(session.entries[node].context_tokens for node in nodes)
        return used > self.max_context_tokens

    async def compact(self, session, nodes, entry):
        # `nodes` is the path you may rewrite; `entry` is your (copied)
        # compaction entry to fill in. Return None for "nothing to do".
        carried = [node for node in nodes if node != entry.id]
        if len(carried) <= self.keep_last:
            return None

        kept = carried[-self.keep_last:]
        dropped = len(carried) - len(kept)

        entry.parts = [TextContent(text=f"[{dropped} earlier entries were dropped to free up context.]")]
        return CompactionPlan(entry=entry, nodes=[entry.id, *kept])


runner = AgentSessionRunner(session, compaction_policy=DropOldestCompactionPolicy())
```

`plan.nodes` is the new conversation, written before ids exist: a string carries
an existing node over; an entry object creates a new one there. The only rules
are structural — it can't be empty, it must include the compaction entry, and it
can only reference ids you were offered.

> ⚠️ **A blind cut can split a tool call from its result.** The framework
> validates the plan's *structure*, never its meaning: it will happily commit a
> path that keeps a tool result whose originating call got folded away, and the
> provider will reject the next request. A production policy cuts on exchange
> boundaries —
> which is exactly what `SummarizingCompactionPolicy` does by walking back to a
> `TurnStart` and the user message that prompted it.

Compaction runs at the top of a drive, *before* the turn:

```python
await runner.run()                # compacts if the policy says so, then answers

runner.schedule_compaction()      # or force one; durable + idempotent
await runner.run()
```

#### What you'll see

```
[compaction] scheduled
[compaction] started
[compaction] finished COMPLETED → new conversation efd24de9
[text] …the answer to the current question…
[finish] stop
```

```python
from luca.agent.core.events import CompactionFinished, CompactionScheduled, CompactionStarted

match event:
    case CompactionScheduled(): print("[compaction] scheduled")
    case CompactionStarted():   print("[compaction] started")
    case CompactionFinished(entry=e, outcome=o, conversation_id=cid):
        print(f"[compaction] finished {o.value} → new conversation {cid}")
```

Afterwards:

```python
print(session.active_conversation.nodes)             # the placeholder + the kept tail
print([c.id for c in session.conversation_history])  # ['28a72c7e'] — the old path, archived intact
print(len(session.entries))                          # unchanged — nothing was deleted
```

`CompactionFinished` fires whatever the outcome — including failure. For a
policy-triggered compaction (which degrades silently so the user's turn
survives), it's your only signal that something went wrong.

For a real one, `contrib.compaction.SummarizingCompactionPolicy` asks a model
for a dense summary and keeps the last N exchanges verbatim.

> ⚠️ **`post_message` raises while a compaction is scheduled or in flight.** It
> requires a closed turn bracket, and a compaction has one of its own. Schedule
> immediately before driving.

### 11c. Running mechanisms — a quick reference

`run()` and `start()` both return an `AgentRun` handle. Everything below is that
handle being consumed differently.

```python
# (a) await → drive to the next stopping point, get a RunResult
result = await runner.run()
result.status              # where it stopped: IDLE / AWAITING_APPROVAL / PENDING
result.outcome             # how the turn closed, if it closed one
result.pending_approvals   # the executions waiting on you

# (b) iterate → render events as they happen. Iteration REQUIRES `async with`.
async with runner.run() as run:
    async for event in run:
        render(event)

# (c) callback → fires for every event even when you only await
await runner.run(on_event=lambda e: log.write(e.model_dump_json()))

# (d) eager → starts immediately in the background, finishes whether you watch or not
run = runner.start()
result = await run                       # join later
run.cancel()                             # or stop it

# (e) from sync code — the loop is async-only
asyncio.run(main())
```

| | `run()` | `start()` |
|---|---|---|
| When work begins | on first `await` / iteration | immediately |
| If you never consume it | nothing happens | it still completes |
| Stop iterating | the agent stops | it keeps going |
| Use it for | UI-paced rendering | fire-and-forget work |

Behaviors worth knowing:

- **Iterating a lazy run *is* the engine.** Pulling the next event is what
  advances the agent; `break` stops it. Exiting the `async with` block
  *suspends* — it never advances anything — and a later `runner.run()` resumes
  the open turn. A handle you exited is spent; make a fresh one.
- **One handle, one logical pass.** A second `await` returns the cached result.
- **A run stops at the next thing that needs you**, which is why the canonical
  app loop is status-driven:

```python
while True:
    if runner.idle():
        runner.post_message(input("> "))
    elif runner.awaiting_approval():
        resolve(runner.pending_approvals())        # §8
    async with runner.run() as run:                # PENDING / CANCELLING → make progress
        async for event in run:
            render(event)
    save(runner.session)
```

---

## 12. Streaming

Everything so far used block events for readability. Pass `streaming=True` to
**add** token-level events. This changes the event vocabulary only — the session
that gets recorded is byte-for-byte identical either way.

```python
from luca.agent.core.events import TextDelta, TextStart

async with runner.run(streaming=True) as run:
    async for event in run:
        match event:
            case TextStart():
                print("assistant: ", end="", flush=True)
            case TextDelta(text=text):
                print(text, end="", flush=True)
```

| Extra event | Carries |
|---|---|
| `TextStart` / `TextDelta` | `.text` — the delta |
| `ReasoningStart` / `ReasoningDelta` | `.text` — the delta |
| `ToolCallStart` | `.tool_call_id`, `.name` — a call is being assembled |

You still receive every block event too, in this order:

```
TextStart
TextDelta 'Hello'
TextDelta ' there'
TextBlock 'Hello there'          ← the completed block
FinishReason 'stop'
```

> ⚠️ **Handle deltas *or* blocks, not both.** A streaming renderer prints the
> deltas and treats `TextBlock` / `ReasoningBlock` as no-ops — otherwise every
> answer prints twice.

---

## 13. Where to go next

Two capabilities this guide deliberately skipped, both worth reading before you
ship anything real:

**Middleware** — ten hook points around the pipeline: rewrite a user message
before it's stored, inspect or alter the tool list sent to the model, intercept
a tool call before it dispatches, observe every outcome, adjust each entry
before it's written. Plain classes implementing whichever hooks they need,
passed as `middleware=[...]`. This is the seam for logging, auditing, redaction,
and injecting your own policy without subclassing anything.
→ [`07-middleware.md`](07-middleware.md)

**Cancellation and runtime control** — `runner.cancel()` is a durable,
synchronous signal that works in any state, including on a reloaded session that
crashed mid-turn: it records the intent, stops the work at the next safe
boundary, and closes the turn cleanly, with a configurable grace window for
in-flight tools. Alongside it, `RuntimeConfig` (persisted with the session)
carries `hard_max_steps` / `soft_max_steps` to bound a runaway turn,
`doom_loop_threshold` to catch an agent repeating the same call, and per-phase
timeouts for model calls and tool bodies.
→ [`04-runner.md`](04-runner.md) §9, [`08-runtime-config.md`](08-runtime-config.md)

### The full reference

| Page | Topic |
|---|---|
| [`02-data-model.md`](02-data-model.md) | `AgentSession`, entries, turns, forking |
| [`03-tools.md`](03-tools.md) | `ToolSpec`, the tool contract, rich results |
| [`04-runner.md`](04-runner.md) | Runs, events, status machine, cancellation |
| [`05-permissions.md`](05-permissions.md) | The `ToolRegistry` contract in full |
| [`06-system-prompts.md`](06-system-prompts.md) | Parts, priorities, assemblers |
| [`08-runtime-config.md`](08-runtime-config.md) | Timeouts, step limits, doom loops |
| [`09-plugins.md`](09-plugins.md) | Writing your own plugin |
| [`10-projection.md`](10-projection.md) | Own the message history the model sees |
| [`11-context-and-usage.md`](11-context-and-usage.md) | `context_tokens`, usage, pruning |
| [`12-compaction.md`](12-compaction.md) | The full compaction contract |
| [`contrib/`](contrib/README.md) | Every contrib package |

Next: [`02-data-model.md`](02-data-model.md).
