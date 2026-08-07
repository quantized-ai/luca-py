# 0010 — Provider-native tools in `luca.agent`

`luca.client` can declare, parse and project provider-native tools (OpenAI
`apply_patch` / `shell`, Anthropic `text_editor` / `bash`). This spec is how
`luca.agent` uses them: what changes so a single session can run native tools
on one provider, switch to another (with its own natives, or none), and keep
projecting the whole history — every turn, deterministically, losslessly.

The mechanism, in one sentence: **the core stores and projects everything
provider-blind; the plugin that owns a native tool adapts the outgoing request
(tools and messages) and adopts the incoming calls, through ordinary
middleware.** No provider knowledge enters the core.

The promise, in one timeline:

```
T0  session starts on gpt-5.4 (openai, native)   → apply_patch + shell on the wire
T1  model edits files and runs tests natively    → stored once, canonically
T4  switch to claude-opus-5 (anthropic, native)  → text_editor + bash on the wire
    the OpenAI turns replay as ordinary tool calls; a pending approval executes fine
T8  switch to kimi-2.7 (native off)              → generic read/write/edit/bash
T9  switch back to gpt-5.4                       → the T1 calls replay NATIVE again,
    byte-identical to their first emission, rebuilt from storage
```

---

## Ground truth

Two facts everything below builds on. Both are shipped or live-validated.

### 1. The client's generic form (shipped, commit 93b6df1)

A native call or result has two equivalent representations in `luca.client`:
the typed subclass, and the canonical base type carrying the native identity in
`extras` under the one reserved key `custom_type`:

```python
# These are the SAME call:
ApplyPatchToolCall(id="tc1", name="apply_patch", item_id="apc_88", status="completed",
                   arguments={"type": "update_file", "path": "main.py", "diff": "@@…"})

ToolCall(id="tc1", name="apply_patch",
         arguments={"type": "update_file", "path": "main.py", "diff": "@@…"},
         extras={"custom_type": "apply_patch_call", "item_id": "apc_88", "status": "completed"})

# And these are the SAME result:
ShellToolMessage(tool_call_id="tc2", results=[ShellCommandResult(stdout="2 passed", …)])

ToolMessage(tool_call_id="tc2", content="",
            extras={"custom_type": "shell_call_output",
                    "results": [{"stdout": "2 passed", "stderr": "",
                                 "outcome": {"type": "exit", "exit_code": 0}}]})
```

`as_native()` / `as_generic()` convert either way, losslessly. The transports
normalize before projecting, so the two forms produce byte-identical wire
payloads. Consequence: **the generic form needs no provider imports** — a
plain `ToolCall`/`ToolMessage` + `extras` dict is storable and projectable
with no native class in scope. This is the agent's storage format.

### 2. Providers accept undeclared tool names in history (validated live)

The default projection of a foreign call is an ordinary `tool_use` /
`function_call` whose name is not in this request's `tools`. Both providers
accept it — even alongside native declarations in the same request:

```
anthropic  history: tool_use  name="openai_apply_patch", name="openai_shell"  (undeclared)
           tools:   [text_editor_20250728, bash_20250124, delete_file]
           → 200; the model answers accurately FROM the foreign turns

openai     history: function_call  name="anthropic_text_editor_20250728", …   (undeclared)
           tools:   [{type:"apply_patch"}, {type:"shell"}, {type:"function", name:"read"}]
           → 200; same
```

Probes: `poc_tests/anthropic_undeclared_history.py` and
`poc_tests/openai_undeclared_history.py`, run 2026-08-07 against
claude-sonnet-4-5 and gpt-5.1 (Responses wire). Additionally validated in the
client (`openai_responses/native_tools.py:62`): OpenAI accepts a full native
item replay — `id` and `status` included — on every later turn under
`store: false`.

---

## Principles

1. **A native tool is a tool.** `openai_apply_patch` is a real `ToolSpec`, a
   real `Tool` with a real `execute()`, resolved and permissioned like any
   other. Nothing about it is a special case anywhere in the core.
2. **`ToolSpec.name` is a stable internal identity.** Every framework lookup —
   resolution, approvals, doom-loop, middleware, `spec_id()` — keys on it.
   The DEFAULT projection of any stored call is this name, verbatim, as an
   ordinary function call — valid on every provider (Ground truth 2), with no
   tool class, adapter or plugin required.
3. **Native is a last-mile upgrade, applied by its owner.** The projector is
   provider-blind and untouched. The plugin that owns a native tool rewrites
   the outgoing request — the tool list and its own calls/results — through
   middleware, and adopts its own inbound calls the same way. An uninstalled
   plugin means no upgrade: the session still projects (the default is the
   verbatim form).
4. **One call → one execution → one result, all the way to the wire.**
5. **Native payloads are captured at birth, in the client's generic form.**
   `extras` on the call (written at adoption), `extras` on the result (written
   by `execute()`). Opaque everywhere in the core: stored, never interpreted.
6. **The active LLM config — and the native-tools flag — live on the
   `AgentSession`.** Anything handed the session — `get_tools`, every
   middleware — reads what the next call will use from it. No `llm_cfg`
   parameter threading, no config captured at construction. (Mechanism in
   Layer 4.)

---

## Layer 1 — the application surface

```python
plugin = ShellAccessPlugin(workspace, ...)
runner = PluginAgentSessionRunner(session, plugins=[plugin], ...)
# native on/off is SESSION state, stamped by the runner together with the
# active config: session.update_llm_config(model_string, use_native_tools=…)
```

Native on/off is an adaptation input, nothing more: the same entries stay
valid for every provider, and because the flag is re-stamped every drive
iteration (Layer 4), flipping it mid-session just works. What the model sees
per request:

```
openai,    natives supported →  apply_patch + shell (native items) + read glob grep
anthropic, natives supported →  text_editor + bash (native items)  + read glob grep delete_file
anything else, or native off →  read glob grep edit write apply_patch delete_file bash
```

The TUI renders `ToolSpec.title` (`"Apply patch"`, `"Shell"`) via
`display_name`; internal names (`openai_apply_patch`) are identities, shown
only in debug surfaces.

### The tool set per mode — settled

> Supersedes the guess this section used to carry — `[apply_patch, shell,
> read]` on OpenAI, `[text_editor, bash, delete_file]` on Anthropic — worked
> capability by capability against the four natives' real command sets and
> the generics as they exist in `contrib/shell/tools.py`.

Two things moved. **The generic set is eight, not seven**: `delete_file` is a
real generic tool (`ToolKind.DELETE`, landed with this section), not an
Anthropic-only prop. And **search survives everywhere**: no native replaces
`glob` / `grep`.

#### native off, or a provider/model with no natives

```
read  glob  grep  edit  write  apply_patch  delete_file  bash
```

The whole suite. `delete_file` belongs HERE and not only under Anthropic: a
tool that exists in one provider mode and not another is exactly the
provider-shaped behavior difference this spec exists to remove, and it closes
a real gap — before it, the only ways to remove a file were a
`*** Begin Patch / *** Delete File:` envelope and `bash rm`, the second of
which spends an EXECUTE approval over a command string to perform a DELETE
over a path.

#### openai + native

```
openai_apply_patch  openai_shell  read  glob  grep
```

| kept | why |
|---|---|
| `read` | `apply_patch` cannot read, and `shell`'s `cat` is not a substitute: `read` returns images and PDFs as attachments, caps line length and bytes, pages by offset, and is the only tool that feeds the `FileReadTracker` WITHOUT writing the file (`edit` and `write` record into it too — `tools.py:420`, `911`, `961`, `1047`). It also sits in the read tier `ShellAccessPlugin` auto-allows, where a `shell` call never will. |
| `glob` `grep` | Nothing native replaces search. `rg` through `shell` costs an approval prompt per search, has no result cap, and is a differently-spelled argv every time. Two declarations buy determinism, output caps, and silence inside the workspace. |

| dropped | why |
|---|---|
| `edit` `write` | `update_file` is exact-context replacement and `create_file` is a whole-file write. Two edit grammars in one request is the worst overlap available. Accepted losses: `edit.replace_all` (N calls, or `sed` through `shell`), and — a REQUIREMENT on the native tool, not an assumption — `create_file` over an existing path must overwrite, or a large file has no cheap full rewrite. |
| `apply_patch` (generic) | Same verb, narrower unit: the native takes ONE operation, the generic an envelope of many. Multi-file atomicity is what native mode costs, and it costs it on both providers. |
| `bash` | `shell` is a superset — a LIST of commands, `timeout_ms`, `max_output_length`. `workdir` becomes `cd … && …`. |
| `delete_file` | `apply_patch`'s `delete_file` operation. |

#### anthropic + native

```
anthropic_text_editor_20250728  anthropic_bash_20250124  read  glob  grep  delete_file
```

| kept | why |
|---|---|
| `read` | `text_editor view` reads text and lists directories; it cannot return an image or a PDF, and it records nothing into the `FileReadTracker`. Partial overlap is not replacement. The cost is one duplicated read path; the benefit is that "look at this screenshot" still works on the provider with the best vision. |
| `glob` `grep` | As above. |
| `delete_file` | `text_editor` has no delete command at all — this is the mode that proves the tool has to exist. |

| dropped | why |
|---|---|
| `edit` `write` | `str_replace` and `create` are the same two verbs under different parameter names. Accepted loss: `replace_all`. |
| `apply_patch` (generic) | The one real judgement call. It survives on capability — multi-file atomic edits, and `*** Move to:` would be Anthropic-native's only rename path — and is dropped anyway: a V4A envelope sitting next to `str_replace`/`insert` is two edit grammars again, and the model is trained on the second one. Rename falls to `bash mv`, which is what a shell is for. |
| `bash` | `bash_20250124` is one `command` string; the timeout, the output caps and the `workdir` are the EXECUTOR's, and the executor is still ours. |

`{"restart": true}` has no session to restart — our bash is one process per
call — so the native bash tool answers it with a plain SUCCESSFUL result
saying so. An error would teach the model the tool is broken; the intent
("give me a clean shell") is already true by construction.

#### `REPLACES` — per native tool, never per provider

For each native tool, the generic names it makes redundant. This is the whole
table the middleware consumes:

| native tool | replaces |
|---|---|
| `openai_apply_patch` | `edit` `write` `apply_patch` `delete_file` |
| `openai_shell` | `bash` |
| `anthropic_text_editor_20250728` | `edit` `write` `apply_patch` |
| `anthropic_bash_20250124` | `bash` |

The generic `apply_patch` is on BOTH edit rows. On OpenAI it loses to the
native of the same name; on Anthropic it loses to `str_replace`/`insert` — the
"dropped" table above says so, and it is the edit tool that must say it,
because a native shell replaces nothing about patching.

```python
# contrib/shell/native/: `supported_native_tools` already answers "which
# natives does this (provider, model) pair support" — the flag gates it.
active  = supported_native_tools(session.llm_config) if session.use_native_tools else frozenset()
dropped = {generic for name in active for generic in REPLACES[name]}

visible = [s for s in specs
           if s.name not in dropped                              # a replaced generic
           and (s.name not in REPLACES or s.name in active)]      # an inactive native
```

It has to be per NATIVE TOOL because support is per MODEL: `gpt-5.5-pro` has
`shell` and no `apply_patch`, `gpt-5.4-pro` has `apply_patch` and no `shell`.
Under a per-provider `keep_generic` list both come out wrong.

Every row below was re-derived by hand-executing the snippet against the eight
generics and the four natives (2026-08-08); `dropped` is the union of
`REPLACES` over the active natives, and `advertised` is what survives both
filters:

| model | active natives | dropped generics | advertised |
|---|---|---|---|
| `gpt-5.6` | apply_patch, shell | edit write apply_patch delete_file bash | `openai_apply_patch openai_shell read glob grep` |
| `gpt-5.5-pro` | shell | bash | `openai_shell read glob grep edit write apply_patch delete_file` |
| `gpt-5.4-pro` | apply_patch | edit write apply_patch delete_file | `openai_apply_patch read glob grep bash` |
| `claude-opus-5` | text_editor, bash | edit write apply_patch bash | `anthropic_text_editor_20250728 anthropic_bash_20250124 read glob grep delete_file` |
| anything on `openrouter` | none | — | the eight generics |

The two full-support rows reproduce the settled sets above exactly — five
tools on OpenAI, six on Anthropic. Both partial rows are the answer you want —
full generic file editing beside a native shell, and native patching beside a
generic bash — and neither is expressible per provider.

#### The read-before-write guard, under natives

`edit` and `write` are the two tools that enforce the `FileReadTracker`
contract, and in native mode neither is advertised, so the guard stops
applying to the edits the model actually makes. That is smaller than it
sounds and the fix is narrow:

- The in-place native paths that REPLACE existing text — `apply_patch
  update_file` and `str_replace` — need the exact current content, which the
  model can only have from a read. The guard is nearly redundant on those two.
- `insert` is the exception, and it is a real one: it takes `insert_line` and
  `insert_text` and nothing else (`client/native/text_editor.py:64`), so a
  model can insert into a file it has never read. It is left outside the guard
  anyway, and the reason is not redundancy but blast radius: `insert` is purely
  additive — every original line survives it — so a blind insert produces a
  broken file, never a lost one, and `read`-then-`insert` is what a model does
  in practice because it needs a line number.
- The path that genuinely clobbers is whole-file creation over an existing
  file (`create_file`, `create`). **Recommendation (unchanged by the `insert`
  correction above)**: the two native EDIT tools take the plugin's shared
  `FileReadTracker` and enforce read-first on exactly that path, and
  `text_editor view` RECORDS into it the way `read` does. Nothing else — a
  guard duplicated in four tools is a guard nobody trusts.
- `delete_file` is outside the contract in every mode, deliberately: `read`
  refuses binaries, so a read-first delete could never remove a stray `.pyc`,
  archive or screenshot, and having read 2000 lines says nothing about the
  whole file being destroyed. Its containment is its approval step, which the
  plugin never auto-allows.

#### Left for later

- The shell system prompt still enumerates all eight generics in every mode;
  it should name what this request actually advertises. Harmless (an extra
  name in prose), so it follows rather than blocks.
- `replace_all` has no native equivalent. If models start emitting five
  `str_replace` calls for one rename, the answer is to keep `edit` on
  Anthropic — revisit with evidence, not now.
- Multi-file atomic patching is gone in both native modes. Same rule: bring
  the generic `apply_patch` back if a real case turns up.
- Read-first on the native create paths (recommended above) is not
  implemented in this pass.

---

## Layer 2 — `luca/agent/contrib/shell/`: the tools and their middleware

```
luca/agent/contrib/shell/
├── tools.py            read glob grep edit write apply_patch delete_file bash  (generic)
├── native/
│   ├── support.py      supported_native_tools(llm_config) — per-MODEL capability
│   ├── openai.py       OpenAIApplyPatchTool  OpenAIShellTool
│   ├── anthropic.py    AnthropicTextEditorTool  AnthropicBashTool
│   └── middleware.py   ShellNativeMiddleware — the whole provider fork
└── plugin.py           ShellAccessPlugin
                          get_tool_registry  → SimpleToolRegistry (UNCHANGED)
                          get_middleware     → [ShellNativeMiddleware(session)]
```

### The four native tools

Real tools: their own `Args`, `execute()`, `build_permission_requests()`.

```python
class OpenAIApplyPatchTool(ShellTool):
    name = "openai_apply_patch"          # internal identity
    title = "Apply patch"
    tool_kind = ToolKind.EDIT

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")
        type: Literal["create_file", "update_file", "delete_file"]
        path: str
        diff: str | None = None          # absent for delete_file

    async def execute(self, args, ...):
        # V4A is the grammar shell/patch.py already parses; reuse parse_patch.
        ...


class OpenAIShellTool(ShellTool):
    name = "openai_shell"
    title = "Shell"

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")
        commands: list[str] = Field(min_length=1)
        timeout_ms: int | None = None
        max_output_length: int | None = None

    async def execute(self, args, ...):
        # One subprocess per command, in order. The native result shape is
        # captured AT BIRTH (principle 5) — the only moment the per-command
        # split exists:
        results = [await self._run(cmd, args.timeout_ms) for cmd in args.commands]
        return ExecutionResult(
            content=[TextContent(text=self._render(results))],   # the model-facing text
            extras={"custom_type": "shell_call_output",
                    "results": [{"stdout": r.stdout, "stderr": r.stderr,
                                 "outcome": r.outcome} for r in results]},
        )


class AnthropicTextEditorTool(ShellTool):
    name = "anthropic_text_editor_20250728"
    title = "Edit file"
    tool_kind = ToolKind.EDIT

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")
        command: Literal["view", "create", "str_replace", "insert"]
        path: str
        file_text: str | None = None
        old_str: str | None = None
        new_str: str | None = None
        insert_line: int | None = None
        insert_text: str | None = None
        view_range: list[int] | None = None


class AnthropicBashTool(ShellTool):
    name = "anthropic_bash_20250124"
    title = "Bash"

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")
        command: str | None = None
        restart: bool | None = None
```

### `ShellNativeMiddleware` — every provider fork, in one plugin-owned object

Registered through `Plugin.get_middleware(agent_session)` (the existing
protocol), so it holds the session and reads the active config from it
(principle 6). Four methods over four slots — three exist today, one is new
(`adapt_tool_declarations`, Layer 4).

```python
class ShellNativeMiddleware:
    """The plugin's entire provider knowledge. The core never sees any of
    these tables."""

    WIRE_NAMES = {                         # internal → provider-facing, OUT
        "openai_apply_patch":               "apply_patch",
        "openai_shell":                     "shell",
        "anthropic_text_editor_20250728":   "str_replace_based_edit_tool",
        "anthropic_bash_20250124":          "bash",
    }
    ADOPT_BY_CUSTOM_TYPE = {               # inbound OpenAI: typed calls
        "apply_patch_call": "openai_apply_patch",
        "shell_call":       "openai_shell",
    }
    ADOPT_BY_WIRE_NAME = {                 # inbound Anthropic: plain calls
        "str_replace_based_edit_tool": "anthropic_text_editor_20250728",
        "bash":                        "anthropic_bash_20250124",
    }
    DECLARATIONS = {
        "openai_apply_patch":               openai_client.ApplyPatchTool,
        "openai_shell":                     openai_client.LocalShellTool,
        "anthropic_text_editor_20250728":   anthropic_client.TextEditorTool,
        "anthropic_bash_20250124":          anthropic_client.BashTool,
    }
    REPLACES = {                           # native tool → the generics it makes
        "openai_apply_patch":               ("edit", "write", "apply_patch", "delete_file"),
        "openai_shell":                     ("bash",),                    # redundant
        "anthropic_text_editor_20250728":   ("edit", "write", "apply_patch"),
        "anthropic_bash_20250124":          ("bash",),
    }

    def __init__(self, session):
        self.session = session                 # the ONLY state — everything per-call
                                               # is read from it (principle 6)

    def _active(self):
        """The natives this request may advertise. Capability is per MODEL
        (native/support.py); the flag is the application's. Both facts were
        stamped on the session by the runner before any middleware runs
        (Layer 4)."""
        if not self.session.use_native_tools:
            return frozenset()
        return supported_native_tools(self.session.llm_config)

    # ── OUT: the advertised tools — two slots, two vocabularies ──────────────

    def build_tool_list(self, conversation_id, specs):
        """Spec vocabulary — the EXISTING slot (runner.py:2252). Decides WHICH
        of my tools this request advertises: drop only, never reshape. Two
        drops, and nothing else is mine to touch: a generic some ACTIVE native
        replaces, and a native that is not active."""
        active = self._active()
        dropped = {generic for name in active for generic in self.REPLACES[name]}
        return [s for s in specs
                if s.name not in dropped
                and (s.name not in self.REPLACES or s.name in active)]

    def adapt_tool_declarations(self, conversation_id, tools):
        """Client vocabulary — the NEW slot, after the spec→declaration
        conversion. Swaps my surviving native function declarations for the
        client's native items."""
        return [self.DECLARATIONS[t.name]() if t.name in self.DECLARATIONS else t
                for t in tools]

    # ── OUT: the projected messages ──────────────────────────────────────────

    def before_llm_call(self, conversation_id, messages, system_message):
        """The projector already produced the DEFAULT form: every call under
        its internal name, no extras — valid as-is on every provider
        (Ground truth 2). This upgrades MY active calls to native. Not-mine,
        foreign-family and native-off calls are untouched: verbatim is the
        default, not a branch."""
        active = self._active()
        if not active:
            return messages, system_message
        out = []
        for message in messages:
            if isinstance(message, ClientAssistantMessage):
                message = self._upgrade_calls(message, active)
            elif isinstance(message, ClientToolMessage):
                message = self._upgrade_result(message, active)
            out.append(message)
        return out, system_message

    def _upgrade_calls(self, message, active):
        blocks = []
        for block in message.content:
            if isinstance(block, ClientToolCall) and block.name in active:
                execution = self.session.get_tool_execution(block.id)
                block = block.model_copy(update={
                    "name": self.WIRE_NAMES[block.name],
                    "extras": execution.raw_tool_call.extras,     # stored at adoption
                })
            blocks.append(block)
        return message.model_copy(update={"content": blocks})

    def _upgrade_result(self, message, active):
        execution = self.session.get_tool_execution(message.tool_call_id)
        if execution is None or execution.raw_tool_call.name not in active:
            return message
        result = execution.result
        if result is not None and result.extras:
            return message.model_copy(update={"extras": result.extras})   # stored at birth
        if execution.raw_tool_call.name == "openai_shell":
            # DERIVED outcome (rejected / failed / awaiting approval): no
            # ExecutionResult was ever stored, but shell's native result MUST
            # be structured. The owner synthesizes it from the message the
            # projector derived — this is the whole "denied native call"
            # problem, solved where the knowledge lives.
            text = "".join(b.text for b in message.content)
            return message.model_copy(update={"extras": {
                "custom_type": "shell_call_output",
                "results": [{"stdout": "", "stderr": text,
                             "outcome": {"type": "exit", "exit_code": 1}}]}})
        return message    # apply_patch & anthropic natives: plain text is the native form

    # ── IN: adoption ─────────────────────────────────────────────────────────

    def after_llm_response(self, conversation_id, message):
        """Rewrites MY native call blocks to the canonical form under the
        INTERNAL name, generic extras attached — before the runner records
        anything. Everything downstream (recording, executions, doom-loop,
        approvals) sees an ordinary ToolCall."""
        blocks = []
        for block in message.content:
            if isinstance(block, ClientToolCall):
                block = self._adopt(block) or block
            blocks.append(block)
        return message.model_copy(update={"content": blocks})

    def _adopt(self, block):
        active = self._active()
        generic = block.as_generic()           # typed subclass → base + extras (lossless)
        name = self.ADOPT_BY_CUSTOM_TYPE.get(generic.extras.get("custom_type")) or (
            self.ADOPT_BY_WIRE_NAME.get(generic.name)
            if generic.name in {self.WIRE_NAMES[n] for n in active}
            else None
        )
        if name is None:
            return None
        return generic.model_copy(update={"name": name})
```

`as_generic()` is the whole inbound trick: `item_id`, `status` and any field a
native class grows later land in `extras` mechanically. The plugin imports
client declaration classes (they have no generic form) and nothing else
provider-specific from the client.

The registry is untouched: `get_tool_registry` returns today's
`SimpleToolRegistry` with all twelve tools. Resolution, validation, approvals
and dispatch always see the full list — which is what lets a pending
`openai_apply_patch` approval execute after a switch to Anthropic.

---

## Layer 3 — `luca/agent/contrib/simple_tool_registry/` and the plugin protocol

No changes. `get_tools(session, conversation_id)` keeps its signature — the
session now carries the active config (Layer 4), so a registry that wants to
vary by model reads it there. `Plugin.get_middleware` already exists and
already receives the session.

---

## Layer 4 — the core

### The data model: two fields

```python
class ToolCall(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str                                     # ALWAYS the internal ToolSpec.name
    arguments: dict = Field(default_factory=dict)

    extras: dict = Field(default_factory=dict)
    # The client's generic-form extras, stored verbatim at adoption
    # (`custom_type`, `item_id`, `status`, …). Opaque: the core stores it and
    # copies it; only the owning plugin's middleware ever reads it back.
    # Excluded from doom-loop comparison for free: _is_doom_loop compares
    # name + arguments only (runner.py:4602).

    model_config = ConfigDict(extra="forbid")


class ExecutionResult(BaseModel):
    content: list[ContentPart]                    # what the LLM sees — unchanged
    structured_content: dict | None = None        # unchanged, docstring unchanged
    metadata: dict = Field(default_factory=dict)  # unchanged
    is_error: bool = False                        # unchanged

    extras: dict = Field(default_factory=dict)
    # The client's generic-form RESULT extras — the mirror of ToolCall.extras.
    # Written once by the tool's execute() (today: OpenAIShellTool); opaque to
    # the core; read back only by the owning plugin's middleware.
```

`ToolSpec` gains only presentation: `title: str | None` + `display_name`
(excluded from `spec_id()`, so adding it does not invalidate stored ids).
Nothing else. No projection flags, no wire aliases — the core has no concept
of "native".

`AgentSession` gains one lookup — TO IMPLEMENT, it does not exist today:

```python
def get_tool_execution(self, tool_call_id: str) -> ToolExecution | None:
    """The ToolExecution correlated to one tool call id (its tool_call_id
    field, models.py:543). What the adaptation middleware uses to read the
    stored extras back — raw_tool_call.extras for a call, result.extras (or
    result=None, the derived-outcome signal) for a result."""
```

A linear scan over `entries` is fine to start; index later if projection-time
scans ever show up in profiles.

### The session owns the active LLM config — HARD REQUIREMENT

Today `_drive` derives a per-call config into a local and threads it around:

```python
llm_cfg = self.session.session_config.llm_config
model_string = self.build_model_string(conversation_id, llm_cfg)   # middleware may route
effective_cfg = self.effective_llm_config(llm_cfg, model_string)   # a LOCAL — invisible
```

That local exists for exactly one reason: `build_model_string` middleware may
rewrite the model string, and the runner must record the config the call
actually ran under (provenance drives thinking-signature replay). But keeping
the result in a local makes it invisible to everything that was handed the
session — which defeats the reason the session is passed everywhere.

The mechanism — two notions, one of which already exists:

```python
session.session_config.llm_config     # CONFIGURED — durable, what the user set (today)
session.llm_config                    # ACTIVE — what the next call will use (new)
```

```python
# _drive, top of each iteration — replaces the effective_cfg local
model_string = self.build_model_string(conversation_id, self.session.session_config.llm_config)
self.session.update_llm_config(model_string, use_native_tools=use_native_tools)
# same derivation as effective_llm_config (split at the first colon, preserve
# reasoning/extras), moved to the session — and the native flag is stamped in
# the same call (its source is the application's configured value: session
# config / TUI toggle). From here on, session.llm_config and
# session.use_native_tools ARE what this call will use; get_tools, every
# middleware, and provenance recording read them from the session.
```

Properties:

- The derivation always starts from the CONFIGURED config, so a middleware
  routing one turn to a cheaper model does not drift the session — same as
  today, except the result is visible on the session instead of trapped in a
  local.
- `active` needs no persistence: recomputed at the top of every iteration,
  and provenance is already durably recorded per turn (`TurnStart.llm_config`).
- If middleware ever returns different strings for concurrently-driving
  conversations, the single active field is last-writer-wins. That case is
  undefined today and stays undefined — no per-conversation slot until a real
  need exists.

### The runner: four touches

```python
# 1. IN — the adoption slot already exists (runner.py:3248); middleware runs
#    BEFORE anything is recorded, so adoption needs zero new plumbing:
message = self._run_middlewares("after_llm_response", conversation_id, message)

# 2. IN — adapter.py: dispatch on isinstance, and keep every ToolCall field.
#    Fixes a live bug: adapter.py:40 tests block.type == "tool_call", but a
#    typed native call's type is "apply_patch_call" — today the execution is
#    created while the assistant PART is dropped, and the next projection
#    emits a result correlating to nothing. (Moot for adopted calls once the
#    middleware rewrites them, but the default path must be correct without
#    any middleware installed.)
elif isinstance(block, ClientToolCall):
    generic = block.as_generic()
    parts.append(ToolCall(id=generic.id, name=generic.name,
                          arguments=generic.arguments, extras=generic.extras))

# 3. IN — _receive_executions: model_copy instead of field-by-field rebuild,
#    so extras (and any future ToolCall field) ride into raw_tool_call:
raw = call.model_copy(deep=True)

# 4. OUT — _collect_tools gains ONE new middleware slot. `build_tool_list`
#    (runner.py:2252) stays exactly where it is, spec-level — the documented
#    reason holds (runner.py:2242): the client Tool DTO drops tool_kind /
#    namespace / metadata, so a middleware handed DTOs could not filter on
#    them. The new slot runs after the conversion, in client vocabulary,
#    for declaration swaps:
specs = await self.resolve_tool_specs(conversation_id)      # unchanged — full list
self._verify_gate(conversation_id, specs)                   # unchanged
visible = self.build_tool_list(conversation_id, specs)      # unchanged — is_private filter,
                                                            #   then spec-level middleware (drops)
tools = [adapter.tool_spec_to_luca_tool(spec) for spec in visible]
return specs, self._run_middlewares("adapt_tool_declarations", conversation_id, tools)
```

`before_llm_call` (runner.py:2347) is untouched — it already receives the
projected messages. The `ConversationProjector` is untouched — it keeps
producing the provider-blind default, which is exactly the verbatim form.
`get_tools` is untouched. There is no adapter layer, no wire-hook contract,
no specs threading.

---

## The path, end to end

```
OUT  specs ── build_tool_list (spec vocab) ── middleware DROPS:
                    │   replaced generics + inactive natives
                    ▼
              spec → function-declaration conversion
                    │ adapt_tool_declarations (client vocab, NEW) ── middleware SWAPS:
                    ▼   active natives → native declaration items
              acompletion(tools=…)

OUT  entries ──► ConversationProjector (UNTOUCHED) ──► default form:
                    every call under its internal name, no extras — valid everywhere
                    │ before_llm_call middleware (plugin): upgrade MY active calls
                    │   name → wire name, extras from storage;
                    │   results: extras from storage, or synthesized for derived outcomes
                    ▼
              acompletion(messages=…)        client as_native() rebuilds the typed items

IN   client message ── after_llm_response middleware (plugin):
                    my native blocks → as_generic(), internal name, extras kept
                    │
                    ▼
     runner records AssistantMessage.parts + ToolExecution.raw_tool_call (deep copy)
     create_execution ──► decide ──► prepare ──► execute
                                                  └─► ExecutionResult(content, extras)
```

---

## Worked example

```
════════════════════════════════════════════════════════════════════════════
T0   openai, native — the two tool slots
════════════════════════════════════════════════════════════════════════════
IN   get_tools → ALL specs (generics + 4 natives), internal names

build_tool_list (MW, specs):        drop edit/write/apply_patch/delete_file (REPLACES
                                      of openai_apply_patch) + bash (of openai_shell)
                                    drop the anthropic natives (not active)
                                    → [openai_apply_patch, openai_shell, read, glob, grep]
conversion:                         five function declarations
adapt_tool_declarations (MW):       → [ApplyPatchTool(), LocalShellTool(),
                                       Tool("read", {…}), Tool("glob", {…}), Tool("grep", {…})]

OUT  acompletion tools = [{type:"apply_patch"}, {type:"shell", environment:{type:"local"}},
                          {type:"function", name:"read", parameters:{…}}, glob, grep]
```

```
════════════════════════════════════════════════════════════════════════════
T2   native calls arrive and execute
════════════════════════════════════════════════════════════════════════════
WIRE {type:"apply_patch_call", id:"apc_88", call_id:"tc1", status:"completed",
      operation:{type:"update_file", path:"main.py", diff:"@@…@@…"}}
     {type:"shell_call", id:"sh_12", call_id:"tc2", status:"completed",
      action:{commands:["pytest -q","ruff check"], timeout_ms:60000}}

     after_llm_response: as_generic() → custom_type → internal name

STORED
  AssistantMessage(e3, parts=[
    ToolCall("tc1", "openai_apply_patch", {type:"update_file", …},
             extras={custom_type:"apply_patch_call", item_id:"apc_88", status:"completed"}),
    ToolCall("tc2", "openai_shell", {commands:[…], timeout_ms:60000},
             extras={custom_type:"shell_call", item_id:"sh_12", status:"completed"})])
  ToolExecution(e4, tc1, COMPLETED, result=ExecutionResult(
      content=[Text("Updated main.py (2 hunks)")]))
  ToolExecution(e5, tc2, COMPLETED, result=ExecutionResult(
      content=[Text("2 passed\nAll checks passed")],
      extras={custom_type:"shell_call_output",
              results:[{stdout:"2 passed", stderr:"", outcome:{type:"exit", exit_code:0}},
                       {stdout:"All checks passed", stderr:"", outcome:{type:"exit", exit_code:0}}]}))
```

```
════════════════════════════════════════════════════════════════════════════
T4   switch to anthropic, native — the same entries project by DEFAULT
════════════════════════════════════════════════════════════════════════════
projector OUT (untouched): {type:"tool_use", id:"tc1", name:"openai_apply_patch", input:{…}}
                           {type:"tool_use", id:"tc2", name:"openai_shell", input:{…}}
                           plain tool_results from ExecutionResult.content
before_llm_call: openai natives are NOT active on anthropic → NO-OP for them
WIRE tools (build_tool_list): [text_editor_20250728, bash_20250124,
                               read, glob, grep, delete_file]
→ the request shape validated by poc_tests/anthropic_undeclared_history.py
  (which probed the natives + delete_file subset of it)

A PENDING openai_apply_patch approved now: resolves (registry list never
shrank), validates, executes. The switch is invisible to dispatch.
```

```
════════════════════════════════════════════════════════════════════════════
T9   switch back to openai, native — before_llm_call upgrades
════════════════════════════════════════════════════════════════════════════
tc1: name → "apply_patch",  extras ← raw_tool_call.extras
tc2: name → "shell",        extras ← raw_tool_call.extras
     result tc2: extras ← execution.result.extras
WIRE {type:"apply_patch_call", id:"apc_88", call_id:"tc1", status:"completed", operation:{…}}
     {type:"shell_call", id:"sh_12", call_id:"tc2", status:"completed", action:{…}}
     {type:"shell_call_output", call_id:"tc2",
      output:[{stdout:"2 passed", …}, {stdout:"All checks passed", …}]}
     — byte-identical to T2.
```

```
════════════════════════════════════════════════════════════════════════════
The formerly-hard case: a DENIED shell call, openai, native
════════════════════════════════════════════════════════════════════════════
ToolExecution(tc8, status=REJECTED, result=None)      ← no result row exists;
projector derives ToolMessage("[tool execution rejected]", is_error=True)
before_llm_call → _upgrade_result: mine, active, no stored extras, shell →
  extras = {custom_type:"shell_call_output",
            results:[{stdout:"", stderr:"[tool execution rejected]",
                      outcome:{type:"exit", exit_code:1}}]}
WIRE a valid shell_call_output. Not an edge case — the same method that
     upgrades every other result.
```

---

## Recorded decisions

**Provider adaptation is plugin middleware, not core machinery.** The core
stores canonically and projects the verbatim default; the owning plugin
upgrades outgoing requests and adopts inbound calls through the middleware
protocol (`get_middleware`; slots: `build_tool_list`,
`adapt_tool_declarations`, `before_llm_call`, `after_llm_response` — all
existing except `adapt_tool_declarations`). This replaces an earlier design
that threaded
projection flags and wire aliases through the core (see Appendix).

**The default projection is the verbatim form, and it is always valid.**
Internal name, no extras, ordinary function call — validated live on both
providers (Ground truth 2, probes in `poc_tests/`). Fail-safe by
construction: with the plugin uninstalled, nothing upgrades, everything still
projects.

**`extras` carry the client's generic native form, end to end.** Written at
adoption (call) and at execution (result); opaque to the core; read back only
by the owning plugin's middleware. Derived outcomes (REJECTED, FAILED,
awaiting-approval) have no stored result, so their native form — where one is
required at all — is synthesized by the same middleware at projection time.

**`get_tools` returns every resolvable tool, always.** Advertisement is not
its concern. This is what lets a call born under one provider resolve,
validate and execute under another (the pending-approval case), and it means
an inbound call naming a currently-inactive native (a model copying
`openai_apply_patch` out of history while on Anthropic) resolves and
**executes** — it is a real tool that works.

**Advertisement is computed per NATIVE TOOL, through `REPLACES`.** A native
tool names the generics it makes redundant; the request drops the union of
those for the natives that are actually ACTIVE, plus every native that is not.
Per-provider "keep these generics" lists cannot express partial support, and
partial support is real (`gpt-5.5-pro`: shell, no apply_patch). Full table and
the mode-by-mode reasoning in Layer 1.

**`delete_file` is a generic tool in every mode, not an Anthropic prop.**
`text_editor` has no delete command, which is what forced the tool to exist —
but a tool that appears only under one provider is the behavior difference
this spec removes. It ships in the generic eight and is dropped only where
`openai_apply_patch` (whose `delete_file` operation covers it) is active.

**Permission grants do not survive a provider switch.** An ALWAYS grant
records the calling tool's name (`strategy.py:214`), so a grant under
`openai_apply_patch` does not match `anthropic_text_editor_20250728`.
`ToolKindRule(tool_kind=EDIT)` does survive. Known, accepted: the user
re-approves once per switch.

**A tool version bump behaves like a provider switch.** When Anthropic ships
`text_editor_20260401`, that is a new tool with a new internal name; the old
one leaves the middleware's active tables and its calls project verbatim from
then on.

**Doom-loop detection is untouched.** `_is_doom_loop` compares name +
arguments only; per-call `extras` (item ids) cannot defeat it.

**The session exposes the active LLM config AND the native-tools flag;
`effective_cfg` threading is eliminated.** Configured vs. active split; the
runner stamps both at the top of each drive iteration in one call —
`session.update_llm_config(model_string, use_native_tools=…)` (Layer 4). The
per-iteration re-stamp is also what makes a mid-session native flip just
work. Last-writer-wins if middleware routes concurrent conversations
differently — an undefined case, left undefined.

---

## Pending

Implementation order:

```
1. core/models.py            ToolCall.extras; ExecutionResult.extras; ToolSpec.title;
                             AgentSession.get_tool_execution()
2. core session/runner       session-owned active config (Layer 4 — designed, implement)
3. core/adapter.py           isinstance dispatch + as_generic() on the IN default path
4. core/runner.py            _receive_executions model_copy; new
                             adapt_tool_declarations slot in _collect_tools
5. contrib/shell/tools.py    delete_file — DONE (the generic set is eight)
6. contrib/shell/native/     four native tools; ShellNativeMiddleware (REPLACES)
7. contrib/shell/plugin.py   use_native_tools; get_middleware wiring

tests: middleware assertions are the spec's use cases — given entries +
session config, assert the exact (tools, messages) acompletion receives.
```

---

## Appendix — considered and rejected

**Core-threaded projection machinery** (`ToolSpec.should_project` +
`wire_name`, `get_tools(llm_cfg)`, a `WireAdapter` contract with
`declare`/`adopt_call`/`project_call`/`project_result` hooks, a
`LucaClientAdapter.to_wire` pass, specs threaded through
`prepare_llm_call`/`build_messages`, the tools/messages ordering flip in
`_drive`). A full prior iteration of this spec. Rejected because every piece
duplicated a decision the owning plugin can make at the existing middleware
seams with knowledge it already has — and because its projection fork keyed on
the spec alone, which broke on executions that never produce a result (a
denied native shell call crashed projection; under the middleware design that
case is one more branch of the owner's upgrade method).

**NOT_FOUND for inbound calls naming an inactive native.** Made sense when a
per-request `should_project` flag existed; without it, the registry resolves
the tool and it simply executes, which also serves the pending-approval case.

**A narration fallback for foreign history.** The live probes settled
undeclared-name acceptance; the fallback would be dead code.

**`ExecutionResult.structured_content` for native shell results.** Its
docstring forbids exactly this ("ConversationProjector never puts it on the
wire"); `extras` is model-facing by definition.

**`ExecutionResult.metadata` for native shell results.** Viable (it is
plugin-space data), but a dedicated `extras` field keeps the symmetric
contract with `ToolCall.extras` explicit instead of overloading free-form
bookkeeping.

**Result content as N ContentParts, one per command.** Breaks the moment a
command emits both stdout and stderr.

**Converting stored calls between provider shapes** (rewriting an `insert` as
a diff, fanning one call into many). Lossy, O(history) per request, breaks
the 1:1 call/result invariant. The verbatim default is lossless and O(1).

**Storing hand-picked extras keys** (`item_id`, `status`). Superseded by
storing `as_generic().extras` whole: mechanical, lossless, and any field a
native class grows later rides along without agent changes.
