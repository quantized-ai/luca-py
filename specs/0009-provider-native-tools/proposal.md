# 0009 — Provider-native tools in `luca.agent`

Step two of 0009. The client already declares, parses and projects native
tools; this is how the **agent** uses them, with everything new living in
`luca.agent.contrib.shell.native` and exactly **one** line of new surface in
the core.

---

## 0. Scope

What is actually available, per family:

| Our tool | `anthropic` | `openai` (Responses) |
|---|---|---|
| `bash` | `bash_20250124`, wire name `bash` | `shell` (local), wire name `shell` |
| `apply_patch` | — no native equivalent | `apply_patch`, wire name `apply_patch` |
| `read` / `edit` / `write` | (`text_editor_20250728` — appendix) | — |
| `glob` / `grep` | — stays a function tool | — stays a function tool |

So: **three bindings ship.** Anthropic gets native bash; OpenAI gets native
shell + apply_patch. Everything else stays a function tool. The mechanism is
generic — the appendix shows the text-editor binding as ~25 more lines.

The load-bearing discovery: **OpenAI's `apply_patch` diff body is V4A, which
`shell/patch.py` already parses.** A `create_file` diff is all-`+` lines, an
`update_file` diff is context/`-`/`+` with `@@` anchors. So the whole
translation is rendering the `*** Begin Patch` envelope around it.

---

## 1. The one core change

Native tools change four wire touchpoints. The agent already owns three of
them through public seams. It owns the fourth through no seam at all:

| Touchpoint | Seam today |
|---|---|
| Inbound call parsing | the client's own registry — **agent does nothing** |
| Call replay from history | `before_llm_call` middleware — **public** |
| Result shape | `before_llm_call` middleware — **public** |
| **Declaration (`tools=`)** | `_collect_tools` → `adapter.tool_spec_to_luca_tool` — **none** |

A native declaration is a different client CLASS with a different projector
(`{"type": "bash_20250124", "name": "bash"}`), and a `ToolSpec` cannot express
it — `ToolSpec` is name/description/JSON-Schema, i.e. a function tool by
construction. So one hook, mirroring `build_tool_list` exactly one step
downstream:

```python
# luca/agent/core/runner.py  — new public method, same shape as build_tool_list
def build_wire_tools(self, conversation_id: str, specs: list[ToolSpec]) -> list[LucaBaseTool]:
    """The WIRE form of the model-visible catalog, threaded through any
    `build_wire_tools` middleware.

    Split from `build_tool_list` for the reason that one is split from
    `resolve_tool_specs`: a different question in a different vocabulary.
    `build_tool_list` decides WHICH tools the model sees; this decides HOW
    each is declared. A provider-native tool is not a function tool and has
    no `ToolSpec` spelling — it is a different client class carrying a
    different `ToolProjector`."""
    tools = [adapter.tool_spec_to_luca_tool(spec) for spec in specs]
    return self._run_middlewares("build_wire_tools", conversation_id, tools)


# _collect_tools — the only call-site change
async def _collect_tools(self, conversation_id: str) -> tuple[list[ToolSpec], list[LucaBaseTool]]:
    specs = await self.resolve_tool_specs(conversation_id)
    self._verify_gate(conversation_id, specs)
    visible = self.build_tool_list(conversation_id, specs)
-   return specs, [adapter.tool_spec_to_luca_tool(spec) for spec in visible]
+   return specs, self.build_wire_tools(conversation_id, visible)
```

```python
# luca/agent/core/middleware.py — the 13th hook
def build_wire_tools(
    self,
    session: AgentSession,
    conversation_id: str,
    tools: list[LucaBaseTool],
) -> list[LucaBaseTool]:
    """The last-mile tool hook: client tool DTOs, after `build_tool_list` and
    the adapter, immediately before the request. Client vocabulary on
    purpose — same as `before_llm_call`, and the only vocabulary in which a
    provider-native declaration exists. Match by `tool.name`."""
    return tools
```

That is the entire core diff: one method, one call site, one hook stub
(+ docs). **Nothing else in the core, the data model, the projector, the
adapter or the registry contract moves.**

---

## 2. Layer 1 — the application

```python
from luca.agent.contrib.plugins import PluginAgentSessionRunner
from luca.agent.contrib.shell import ShellAccessPlugin

runner = PluginAgentSessionRunner(
    session,
    plugins=[ShellAccessPlugin(workspace=".", native_tools=True)],
)

async with runner.run() as run:
    async for event in run:
        ...
```

One boolean. That is the whole public API change.

`native_tools=True` means *"use the native binding when this session's model
family has one, otherwise the function tool."* Default `False`.

Everything downstream is unchanged and stays unchanged: the same
`AgentSession`, the same `ToolExecution` lifecycle, the same
`ApprovalRequired` gates, the same `ToolCallReceived` / `ToolExecutionStarted`
/ `ToolExecuted` events, the same TUI rendering, the same permission rules on
disk. **A user who granted `Always allow 'git *'` keeps that grant when the
tool goes native** — see §5.

Switching model mid-session (`/model`) switches the tool set on the next call.
No session surgery.

---

## 3. Layer 2 — `ShellAccessPlugin`

```python
class ShellAccessPlugin:
    def __init__(self, workspace, additional_directories=None, mode=PermissionMode.ASK,
                 extra_rules=None, native_tools: bool = False) -> None:
        ...                                             # unchanged
        self.tools = [ReadTool(...), GlobTool(...), ..., BashTool(...)]   # the 7, unchanged
        self.bindings: dict[str, list[NativeBinding]] = (
            build_bindings(self.workspace, self.tracker) if native_tools else {}
        )

    # ── hook 1: the registry ────────────────────────────────────────────────
    def get_tool_registry(self, session) -> ToolRegistry:
        function_registry = SimpleToolRegistry(list(self.tools), self.permission_strategy)
        if not self.bindings:
            return function_registry                    # today's behavior, byte for byte
        return ShellToolRegistry(
            function_registry,
            {
                family: SimpleToolRegistry(variant(self.tools, bindings), self.permission_strategy)
                for family, bindings in self.bindings.items()
            },
        )

    # ── hook 2: the middleware ──────────────────────────────────────────────
    def get_middleware(self, session) -> list:
        return [NativeToolsMiddleware(self.bindings)] if self.bindings else []

    # ── hook 3: the prompt (now a CALLABLE part — resolved per LLM call) ────
    def get_system_prompt_parts(self, session) -> list:
        return [self._prompt_part]

    def _prompt_part(self, session, conversation_id) -> str:
        names = ", ".join(sorted(t.name for t in self._variant_for(session)))
        return SHELL_SYSTEM_PROMPT_TEMPLATE.format(
            workspace=self.workspace, tools=names, additional=self._additional(),
        )
```

The prompt part becomes a callable because the tool NAMES change with the
family (`bash` vs `shell`). Callable parts are already a supported form; a
static string naming `bash` while the wire declares `shell` is exactly the
prompt/tool-list disagreement the subagents gate exists to avoid.

The `PermissionStrategy` and the `FileReadTracker` are **the same instances**
across every variant — that is what makes the native/function switch invisible
to approvals and to the read-first contract.

---

## 4. Layer 3 — the binding table

One value object. Everything provider-specific in the package is data in this
table.

```python
# luca/agent/contrib/shell/native/bindings.py

@dataclass(frozen=True)
class NativeBinding:
    tool: Tool                      # agent-side executor; `.name` IS the wire name
    declaration: ClientBaseTool     # client-side native tool (what `tools=` carries)
    supersedes: tuple[str, ...]     # base tool names this replaces
    build_result: Callable | None = None    # ToolMessage -> native ToolMessage (shell only)


def build_bindings(workspace, tracker) -> dict[str, list[NativeBinding]]:
    return {
        "anthropic": [
            NativeBinding(
                tool=AnthropicBashTool(workdir=workspace),
                declaration=anthropic_client.BashTool(),
                supersedes=("bash",),
            ),
        ],
        "openai": [
            NativeBinding(
                tool=OpenAIShellTool(workdir=workspace),
                declaration=openai_client.LocalShellTool(),
                supersedes=("bash",),
                build_result=shell_tool_message,
            ),
            NativeBinding(
                tool=OpenAIApplyPatchTool(workdir=workspace),
                declaration=openai_client.ApplyPatchTool(),
                supersedes=("apply_patch",),
            ),
        ],
    }


FAMILY_BY_PROVIDER = {"anthropic": "anthropic", "openai": "openai"}

def native_family(session) -> str | None:
    """THE predicate. Consulted identically by the registry, the three
    middleware hooks and the prompt part — one rule, no cache, no
    per-conversation state, same answer before and after a reload."""
    return FAMILY_BY_PROVIDER.get(session.session_config.llm_config.provider)


def variant(base_tools: list[Tool], bindings: list[NativeBinding]) -> list[Tool]:
    superseded = {name for b in bindings for name in b.supersedes}
    return [t for t in base_tools if t.name not in superseded] + [b.tool for b in bindings]
```

Resulting tool sets:

```
none        read glob grep edit write apply_patch bash
anthropic   read glob grep edit write apply_patch bash*          (* native)
openai      read glob grep edit write apply_patch* shell*
```

---

## 5. Layer 4 — the native tools (argument translation)

**A native tool is an ordinary contrib `Tool` whose `Args` is the provider's
wire schema and whose body delegates to the shell tool it supersedes.** That
single decision buys everything else for free: approvals, timeouts,
cancellation, output caps, temp-file spillover, `ShellToolError` → `is_error`,
context accounting, events, TUI rendering.

### `bash_20250124`

```python
class AnthropicBashTool(BashTool):
    name = "bash"                       # the wire name Anthropic mandates
    version = "bash_20250124"

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")
        command: str | None = None
        restart: bool = False

        @model_validator(mode="after")
        def _exactly_one(self):
            if bool(self.command) == self.restart:
                raise ValueError("provide exactly one of `command` or `restart`")
            return self

    def _delegate(self, args: dict) -> dict:
        return {"command": args["command"], "timeout": None, "workdir": None}

    def build_permission_requests(self, args, session, conversation_id):
        if args.get("restart"):
            return []                   # touches nothing — no gate
        return super().build_permission_requests(self._delegate(args), session, conversation_id)

    async def _run(self, args, session, conversation_id, *, cancellation_token):
        if args.get("restart"):
            return ExecutionResult(content=[TextContent(text="A fresh shell runs per command; nothing to restart.")])
        return await super()._run(self._delegate(args), session, conversation_id,
                                  cancellation_token=cancellation_token)
```

### `shell` (local)

Wire action, from the live capture:
`{"commands": ["printf 'luca' | wc -c"], "max_output_length": 10240, "timeout_ms": 8972}`.

```python
class ShellResults(BaseModel):                  # the declared output_schema
    results: list[ShellCommandResult]           # the CLIENT's model, reused verbatim


class OpenAIShellTool(BashTool):
    name = "shell"
    version = "shell.local"
    output_schema = ShellResults

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")
        commands: list[str] = Field(min_length=1)
        timeout_ms: int | None = None
        max_output_length: int | None = None

    def _delegate(self, command: str, args: dict) -> dict:
        return {"command": command, "timeout": args.get("timeout_ms"), "workdir": None}

    def build_permission_requests(self, args, session, conversation_id):
        # One access step (same workdir for every command) + one verb step per command,
        # each IDENTICAL to what the function `bash` tool would have emitted.
        access, verbs = None, []
        for command in args["commands"]:
            steps = BashTool.build_permission_requests(self, self._delegate(command, args),
                                                       session, conversation_id)
            access = access or steps[0]
            verbs.append(steps[1])
        return [access, *verbs]

    async def _run(self, args, session, conversation_id, *, cancellation_token):
        results, transcript = [], []
        for command in args["commands"]:
            single = await BashTool._run(self, self._delegate(command, args), session,
                                         conversation_id, cancellation_token=cancellation_token)
            text = single.content[0].text
            exit_code = single.metadata["exit"]          # None ⇒ timed out / cancelled
            results.append({
                "stdout": _cap(text, args.get("max_output_length")),
                "stderr": "",                            # our bash merges stderr into stdout
                "outcome": {"type": "exit", "exit_code": exit_code} if exit_code is not None
                           else {"type": "timeout"},
            })
            transcript.append(f"$ {command}\n{text}")
        return ExecutionResult(
            content=[TextContent(text="\n\n".join(transcript))],   # model-facing + TUI + Anthropic replay
            structured_content={"results": results},               # the shell wire's channel
            metadata={"commands": args["commands"]},
            is_error=any(r["outcome"].get("exit_code") not in (0,) for r in results),
        )
```

`structured_content` is exactly the right channel and it is already specified
as such: *"`content` is what the LLM sees, `structured_content` is what the
APPLICATION reads"* — never projected, never counted, declared by
`output_schema`. The middleware in §7 is the application that reads it.

### `apply_patch`

```python
HEADERS = {"create_file": ADD_PREFIX, "update_file": UPDATE_PREFIX, "delete_file": DELETE_PREFIX}

def render_envelope(args: dict) -> str:
    """OpenAI's operation -> the V4A envelope `shell/patch.py` already parses.
    The DIFF BODY needs no translation at all: create_file is all-`+` lines,
    update_file is context/`-`/`+` with `@@` anchors — the same grammar."""
    body = "" if args["type"] == "delete_file" else "\n" + args["diff"].rstrip("\n")
    return f"{BEGIN_MARKER}\n{HEADERS[args['type']]}{args['path']}{body}\n{END_MARKER}\n"


class OpenAIApplyPatchTool(ApplyPatchTool):
    name = "apply_patch"
    version = "apply_patch.v4a"

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")
        type: Literal["create_file", "update_file", "delete_file"]
        path: str
        diff: str = ""

    def _delegate(self, args: dict) -> dict:
        return {"patch_text": render_envelope(args)}

    def build_permission_requests(self, args, session, conversation_id):
        return super().build_permission_requests(self._delegate(args), session, conversation_id)

    async def _run(self, args, session, conversation_id, *, cancellation_token):
        return await super()._run(self._delegate(args), session, conversation_id,
                                  cancellation_token=cancellation_token)
```

**Permission parity is the design guarantee.** Every native tool routes
`build_permission_requests` through its base with the delegated args, so the
emitted `ResourcePermission` pairs (`access_directory` + `bash`/`apply_patch`
over the same absolute paths and command strings) are byte-identical to the
function tool's. Seeded rules, saved grants and the TUI approval prompt all
keep working, unchanged, across the switch. This is asserted directly in the
tests (§9).

---

## 6. Layer 5 — the registry

Two problems, one answer: `bash` (function) and `bash` (Anthropic native) share
a name with different `Args`; so do the two `apply_patch`es.

```python
class ShellToolRegistry(ToolRegistry):
    """Pre-built registries, one per family plus the function fallback. ONE
    predicate picks which answers. Stateless — no route cache, no per-call
    state on `self`, identical answer on a cold resume (rule 13 for free)."""

    def __init__(self, function_registry, native_registries: dict[str, ToolRegistry]) -> None:
        self.function_registry = function_registry
        self.native_registries = native_registries

    def _delegate(self, session) -> ToolRegistry:
        return self.native_registries.get(native_family(session), self.function_registry)

    async def get_tools(self, session, conversation_id):
        return await self._delegate(session).get_tools(session, conversation_id)

    async def create_execution(self, session, conversation_id, call):
        return await self._delegate(session).create_execution(session, conversation_id, call)

    async def decide(self, session, conversation_id, execution):
        return await self._delegate(session).decide(session, conversation_id, execution)

    async def prepare(self, session, conversation_id, execution):
        return await self._delegate(session).prepare(session, conversation_id, execution)
```

Twenty lines, zero cleverness. The inner registries are plain
`SimpleToolRegistry`s sharing one `PermissionStrategy`.

---

## 7. Layer 6 — the middleware

Three hooks, one per wire touchpoint the agent owns. **Anthropic needs only
the first**; the other two exist for OpenAI's typed item shapes.

```python
NATIVE_CALLS = "contrib.shell.native_calls"          # AgentSession.extras key
BASE_FIELDS = set(ClientToolCall.model_fields)


class NativeToolsMiddleware:

    def __init__(self, bindings: dict[str, list[NativeBinding]]) -> None:
        self.bindings = bindings

    def _for(self, session) -> list[NativeBinding]:
        return self.bindings.get(native_family(session), [])
```

### 7.1 Declaration — swap the DTO

```python
    def build_wire_tools(self, session, conversation_id, tools):
        declarations = {b.tool.name: b.declaration for b in self._for(session)}
        return [declarations.get(t.name, t) for t in tools]
```

`LucaTool(name="shell", parameters={commands…})` → `LocalShellTool()`. The
client's `_resolve_projector` then emits `{"type": "shell", "environment":
{"type": "local"}}`. Done.

### 7.2 Identity — remember what the core `ToolCall` cannot carry

The core's `ToolCall` is `{id, name, arguments}`. OpenAI's items carry an extra
`item_id` (`apc_…` / `sh_…`) and a `status`, needed to replay the item on every
later turn. `AgentSession.extras` is the sanctioned durable home for exactly
this — plugin-owned state that outlives the process without a second file
(`MemoryPlugin`'s todo store is the reference case).

```python
    def after_llm_response(self, session, conversation_id, message):
        """The one place the native subclass is still intact — the adapter
        flattens it to the core ToolCall immediately after this."""
        family = native_family(session)
        record = session.extras.setdefault(NATIVE_CALLS, {})
        for block in message.content:
            if isinstance(block, ClientToolCall) and type(block).projector_class is not None:
                dumped = block.model_dump()
                record[block.id] = {
                    "family": family,
                    "type": block.type,                                     # registry key
                    "wire": {k: v for k, v in dumped.items() if k not in BASE_FIELDS},
                }
        return message
```

`wire` is *"every field this subclass declares beyond the base"* — generic, so
a native type added later needs no change here.

### 7.3 Last mile — rebuild the native shapes on the way out

The client picks the replay projector from `type(tool_call).projector_class`
and the result projector from the call it answers. So the projected history
must carry the real subclasses. `before_llm_call` is the documented last-mile
hook and receives exactly those messages.

```python
    def before_llm_call(self, session, conversation_id, messages, system_message):
        record = session.extras.get(NATIVE_CALLS) or {}
        family = native_family(session)
        native_calls: dict[str, ClientToolCall] = {}
        result_builders = {b.tool.name: b.build_result for b in self._for(session) if b.build_result}
        executions = None
        out = []

        for msg in messages:
            if isinstance(msg, ClientAssistantMessage):
                msg = self._upgrade_assistant(msg, record, family, native_calls)
            elif isinstance(msg, ClientToolMessage) and msg.tool_call_id in native_calls:
                call = native_calls[msg.tool_call_id]
                build = result_builders.get(call.name)
                if build is not None:
                    executions = executions if executions is not None else _executions(session)
                    msg = build(msg, executions.get(msg.tool_call_id))
            out.append(msg)
        return out, system_message

    def _upgrade_assistant(self, msg, record, family, native_calls):
        blocks = []
        for block in msg.content:
            meta = record.get(block.id) if isinstance(block, ClientToolCall) else None
            if meta is None or meta["family"] != family:
                blocks.append(block)          # foreign family ⇒ replay as a plain function call
                continue
            native = NATIVE_TOOL_CALL_TYPES[meta["type"]](
                id=block.id, name=block.name, arguments=block.arguments,
                complete=True, **meta["wire"],
            )
            native_calls[native.id] = native
            blocks.append(native)
        return msg.model_copy(update={"content": blocks})


def _executions(session) -> dict[str, ToolExecution]:
    """One pass, built lazily and only when a structured result is needed."""
    return {e.tool_call_id: e for e in session.entries.values() if isinstance(e, ToolExecution)}


def shell_tool_message(message, execution):
    """ToolMessage -> ShellToolMessage. Reads the DURABLE execution, so a
    pruned or compacted output stays consistent with the wire."""
    payload = (execution.result.structured_content if execution and execution.result else None) or {}
    results = payload.get("results") or [{                    # gated / rejected / timed out / failed
        "stdout": "", "stderr": tool_message_text(message),
        "outcome": {"type": "exit", "exit_code": 1},
    }]
    return ShellToolMessage(
        tool_call_id=message.tool_call_id, name=message.name,
        content=message.content, results=[ShellCommandResult(**r) for r in results],
    )
```

`apply_patch` needs no `build_result`: its projector takes a plain
`ToolMessage` and turns `is_error` into the wire status. Anthropic needs
neither hook — its native calls are ordinary `tool_use` blocks answered by
ordinary `tool_result` blocks.

---

## 8. One full round trip

**OpenAI, `shell`:**

```
registry.get_tools        → [read…, apply_patch(native), shell(native)]  (family = "openai")
build_tool_list           → same, private specs dropped
build_wire_tools          → [Tool(read)…, ApplyPatchTool(), LocalShellTool()]   ← §7.1
  wire tools[]              {"type":"shell","environment":{"type":"local"}}
─ model responds ─
transport parses          → ShellToolCall(id="call_x", name="shell", item_id="sh_1",
                                          arguments={"commands":["ls"],"timeout_ms":8972})
after_llm_response        → extras["contrib.shell.native_calls"]["call_x"] =
                              {"family":"openai","type":"shell_call",
                               "wire":{"item_id":"sh_1","status":"completed"}}     ← §7.2
adapter.message_to_parts  → core ToolCall(id="call_x", name="shell", arguments={…})
─ ordinary agent machinery: RECEIVED → create_execution → decide (gate!) → prepare → run ─
OpenAIShellTool._run      → BashTool._run per command
                            ExecutionResult(content=[transcript],
                                            structured_content={"results":[{stdout,stderr,outcome}]})
─ next LLM call ─
projector                 → ClientToolCall(id="call_x", name="shell") + ToolMessage(transcript)
before_llm_call           → ShellToolCall(…, item_id="sh_1") + ShellToolMessage(results=[…])  ← §7.3
transport                 → {"type":"shell_call", "id":"sh_1", "call_id":"call_x", …}
                            {"type":"shell_call_output", "call_id":"call_x",
                             "output":[{stdout,stderr,outcome}], "max_output_length":10240}
```

**Anthropic, `bash`:** identical down to `build_wire_tools`, then nothing else
fires — the call arrives as a base `ToolCall` named `bash`, the result goes
back as a base `ToolMessage`. §7.2 and §7.3 are no-ops (no native subclass, no
record).

---

## 9. Edge cases (each one a test)

| Case | Behavior |
|---|---|
| Gated / rejected / timed-out / failed `shell` call | `shell_tool_message` synthesizes one `ShellCommandResult` carrying the derived error text and `exit_code=1` — every terminal status stays projectable |
| `restart: true` (Anthropic) | benign success, **zero permission requests** (nothing is touched) |
| `commands: []` | `Args` `min_length=1` ⇒ terminal `INVALID` birth, structured error, never reaches `decide()` |
| `/model` switches family mid-session | registry answers the new variant on the next call; history records mismatch on `family` ⇒ **no upgrade** ⇒ old native calls replay as ordinary function calls rather than being dropped |
| Reload mid-turn | `extras` restores the records with the session; nothing else is needed |
| `native_tools=False` | `ShellToolRegistry` and the middleware are never constructed — today's path, byte for byte |
| Subagents | registry + middleware are stateless; `extras` writes are sync with no `await` between, so no lock is needed (rule 13) |
| Compaction / pruning | `extras` records may outlive their entries — harmless orphans keyed by call id; the shell result is read from the DURABLE execution, so a pruned output never resurrects on the wire |
| `build_model_string` middleware reroutes the model | the family predicate reads the session config, so a routing middleware must route the natives too. Documented limit, not handled |

---

## 10. Files and tests

```
luca/agent/contrib/shell/
├── plugin.py                       # + native_tools flag, callable prompt part
└── native/
    ├── __init__.py                 # NativeBinding, build_bindings, native_family, the 3 tools
    ├── bindings.py                 # the table + native_family + variant
    ├── tools.py                    # AnthropicBashTool, OpenAIShellTool, OpenAIApplyPatchTool
    ├── envelope.py                 # render_envelope (operation -> V4A)
    ├── registry.py                 # ShellToolRegistry
    └── middleware.py               # NativeToolsMiddleware + shell_tool_message

tests/agent/contrib/shell/native/
├── test_bindings.py                # the table, native_family, variant assembly per family
├── test_tools.py                   # arg translation + PERMISSION PARITY (identical requests
│                                   #   to the function tool for the same effective call)
├── test_envelope.py                # render_envelope × 3 ops, round-tripped through parse_patch
├── test_registry.py                # which registry answers per llm_config; cold-resume identity
└── test_middleware.py              # the 3 hooks as pure functions over message lists,
                                    #   incl. every non-COMPLETED shell result shape

tests/agent/test_runner_middleware.py   # + the build_wire_tools hook (core)
tests/agent/contrib/shell/test_plugin.py # + native wiring, callable prompt part
```

One end-to-end via `FauxProvider` scripting an `AssistantMessage(content=[ShellToolCall(…)])`
proves the whole chain lands in the session and back on the wire.

Docs: `docs/agent/contrib/shell/native.md`, plus the new hook in
`docs/agent/07-middleware.md` and `AGENTS.agent.md`.

---

## 11. Rejected alternatives

1. **Native `ToolCall` subclasses in the agent core data model** (the PRD's
   "step two" note). Puts provider wire shapes into the core's ONE serializable
   session and forces `SerializeAsAny` through `AssistantMessage.parts`.
   `session.extras` gets the same durability with zero core surface.
2. **A `ConversationProjector` subclass** for the OpenAI shapes. Correct, but
   `conversation_projector=` is a runner constructor argument, so a plugin
   cannot install one — and two plugins could not both. `before_llm_call`
   reaches the same messages and composes.
3. **`ToolSpec.native_declaration: dict` + a data-driven client `NativeTool`.**
   The most honest option for HISTORY (an old session would say it was
   advertised as `bash_20250124`). Costs a new client type plus a passthrough
   projector per transport family, and makes the core route by provider.
   Revisit when a third family appears; until then `ToolSpec.version`
   (`"bash_20250124"`) records the same fact in an existing field.
4. **A `NativeToolsRunner` overriding `_collect_tools`.** Makes a private
   method de-facto public and forces a runner class on the application.

---

## Appendix — the text-editor binding (not in scope, ~25 lines)

```python
class AnthropicTextEditorTool(EditTool):
    name = "str_replace_based_edit_tool"
    version = "text_editor_20250728"

    class Args(BaseModel):
        command: Literal["view", "create", "str_replace", "insert"]
        path: str
        view_range: list[int] | None = None
        file_text: str | None = None
        old_str: str | None = None
        new_str: str | None = None
        insert_line: int | None = None
        insert_text: str | None = None

    async def _run(self, args, session, conversation_id, *, cancellation_token):
        match args["command"]:
            case "view":        return await self.read.execute({...}, ...)      # ReadTool
            case "create":      return await self.write.execute({...}, ...)     # WriteTool
            case "str_replace": return await EditTool._run(self, {...}, ...)
            case "insert":      return await self.write.execute({...}, ...)
```

`supersedes=("read", "edit", "write")`, `declaration=anthropic_client.TextEditorTool(max_characters=…)`,
and it needs neither §7.2 nor §7.3 — same as bash.
