# 0010 — Phase 1 implementation plan: validate the architecture with integration tests

Self-contained. Read [draft.md](draft.md) (the design) and
[use_cases.md](use_cases.md) (the traced flows — they are the expected values
of these tests) before starting. This plan tells you what to build, in what
order, and every test to write. Follow the repo's AGENTS.md for test style:
full-object assertions, declarative test bodies (precondition → one action →
postcondition), fixtures in conftest, no logic in tests.

## Objective

Prove the design end to end with **integration tests only**, both directions
(wire-in → stored data model; stored data model → what `acompletion()`
receives). Explicitly NOT the goal: production-quality native tools. Handlers
are mocked, results are dummies, the tool set is a throwaway test subset.
Phase 1 succeeds when the whole battery below is green; real
`contrib/shell/native` tools come later and only replace the test fixtures'
tables.

## The test tool subset

Fixed for the whole phase, declared once at the top of the test package as
plain config (when the real implementation lands, only this config changes):

```
GENERICS  read{file_path}   delete_file{path}   apply_patch{patch_text}   bash{command}
NATIVES   openai_apply_patch      {type, path, diff}       wire "apply_patch"
          openai_shell            {commands, timeout_ms}   wire "shell"
          anthropic_text_editor_20250728  {command, path, …}  wire "str_replace_based_edit_tool"
          anthropic_bash_20250124        {command}             wire "bash"

ADVERTISED (the plugin middleware's tables):
  openai,    native on  → [openai_apply_patch, openai_shell, read]
  anthropic, native on  → [anthropic_text_editor_20250728, anthropic_bash_20250124, delete_file]
  else / native off     → every generic, no natives
```

Every tool's `execute()` returns a hardcoded dummy `ExecutionResult`
(`content=[Text("<name> executed")]`), EXCEPT `openai_shell`, which also
writes the structured result extras — that asymmetry is part of what phase 1
validates:

```python
ExecutionResult(
    content=[TextContent(text="shell executed")],
    extras={"custom_type": "shell_call_output",
            "results": [{"stdout": "shell executed", "stderr": "",
                         "outcome": {"type": "exit", "exit_code": 0}}]},
)
```

Tests that need to observe an execution patch `execute()` with a mock and
assert the validated `Args` it received.

## Step 0 — verify the mocking boundary (do this FIRST)

The tests must control what the LLM "returns". Three candidate boundaries,
in order of preference — verify which one works and settle it before writing
any test:

1. **Mock `acompletion` / `acompletion_stream` at the runner's import site**
   (`luca/agent/core/runner.py` — confirm the import name and patch there).
   The mock (a) captures its kwargs — `messages`, `tools`, `model` — for
   assertions, and (b) returns a prebuilt client response whose message
   contains **typed native ToolCall subclasses** (`ApplyPatchToolCall`,
   `ShellToolCall` from `luca.client.providers.openai`), because that is what
   the real client delivers after parsing. This is expected to be sufficient
   for the entire battery: client-side parsing/projection is already covered
   by `tests/client/` (spec 0009), so re-testing it here adds nothing.
2. **The Faux provider** (`--faux` in main.py). Verify whether it can return
   provider-native typed calls. It almost certainly cannot (native parsing is
   per-transport). If it can, it may replace option 1 for some tests; do not
   spend more than an hour here.
3. **httpx-level mocking with real transports** (respx or a custom
   `httpx.Client`), replaying captured wire payloads like the one in test B1
   below. Only needed if a test must exercise the real client parse path —
   default to NOT doing this in phase 1.

Also verify (V1): the `after_llm_response` middleware runs on the
**streaming** path (`acompletion_stream`) as well as the non-streaming one —
adoption lives in that hook, and the TUI streams by default. If it does not
run there today, fixing that is in scope for phase 1.

## Step 1 — core changes (all small; see draft.md Layer 4 for full text)

1. `luca/agent/core/models.py`
   - `ToolCall.extras: dict = Field(default_factory=dict)` — opaque carrier
     of the client's generic-form native payload.
   - `ExecutionResult.extras: dict = Field(default_factory=dict)` — same,
     for results, written by `execute()`.
   - `ToolSpec.title: str | None` + `display_name` property; `title` excluded
     from `spec_id()`.
   - `AgentSession.get_tool_execution(tool_call_id) -> ToolExecution | None` —
     linear scan over `entries` matching `ToolExecution.tool_call_id`.
2. Session-owned active config (replaces the `effective_cfg` local):
   - `session.update_llm_config(model_string, use_native_tools: bool)` —
     the derivation currently in `AgentSessionRunner.effective_llm_config`
     (runner.py:2193: split at first colon, preserve `reasoning`/`extras`),
     starting from `session.session_config.llm_config` (the CONFIGURED value,
     which never mutates). Stamps `session.llm_config` (ACTIVE) and
     `session.use_native_tools`.
   - `_drive` calls it at the top of each iteration (where
     `effective_cfg = self.effective_llm_config(...)` sits today,
     runner.py:3090) and every current reader of `effective_cfg` (provenance
     recording included) reads the session instead. No `llm_cfg` parameter is
     added to any signature.
   - The ACTIVE values are not persisted; they are recomputed every iteration
     (this is also what makes a mid-session native on/off flip work).
3. `luca/agent/core/adapter.py` — the IN path dispatches on
   `isinstance(block, ToolCall)` (today it tests `block.type == "tool_call"`,
   which drops typed native subclasses — a live bug), and the default
   conversion goes through `block.as_generic()` so `extras` are captured even
   with no middleware installed.
4. `luca/agent/core/runner.py`
   - `_receive_executions`: `raw = call.model_copy(deep=True)` instead of the
     field-by-field rebuild (extras must reach `raw_tool_call`).
   - `_collect_tools`: one NEW middleware slot after the spec→declaration
     conversion. `build_tool_list` (runner.py:2252) stays exactly as is:

     ```python
     visible = self.build_tool_list(conversation_id, specs)      # unchanged
     tools = [adapter.tool_spec_to_luca_tool(s) for s in visible]
     return specs, self._run_middlewares("adapt_tool_declarations", conversation_id, tools)
     ```

## Step 2 — test fixtures (NOT in `luca/`, test-local)

Under `tests/agent/test_native_tools/` (or equivalent):

- The eight tools from the subset, as real `Tool` classes with real `Args`
  models and dummy `execute()`s as specified above.
- `NativeToolsMiddleware` — the phase-1 stand-in for the design's
  `ShellNativeMiddleware` (draft.md Layer 2), holding the four tables
  (`WIRE_NAMES`, `ADOPT_BY_CUSTOM_TYPE`, `ADOPT_BY_WIRE_NAME`,
  `DECLARATIONS`, `NATIVE_BY_PROVIDER`) and the four methods:
  - `build_tool_list(cid, specs)` — DROP only, spec vocabulary.
  - `adapt_tool_declarations(cid, tools)` — SWAP only, client declarations.
  - `before_llm_call(cid, messages, system_message)` — upgrade own active
    calls (name → wire name, extras from
    `session.get_tool_execution(id).raw_tool_call.extras`) and own results
    (extras from `execution.result.extras`; if the execution has no result —
    REJECTED/FAILED/awaiting — and the tool is `openai_shell`, synthesize
    `shell_call_output` from the derived message text, exit_code 1).
    Everything not-mine or not-active: untouched (verbatim is the default the
    projector already produced).
  - `after_llm_response(cid, message)` — adopt own inbound calls:
    `block.as_generic()`, then custom-type match, then wire-name match but
    ONLY for wire names of currently-active natives; rename to the internal
    name, keep `extras`.
  All per-call state read from `session.llm_config` /
  `session.use_native_tools`; the middleware's only constructor arg is the
  session.
- A test plugin exposing `get_tool_registry` (plain `SimpleToolRegistry` with
  all eight tools) and `get_middleware` (the middleware above), wired through
  `PluginAgentSessionRunner`.
- A controllable permission policy: modes `allow_all`, `require_approval`,
  `deny_all`.
- conftest helpers: `make_session(model, native)`, `drive(runner, response)`
  (posts a message / redrives with the acompletion mock primed), plus
  builders for the typed native call objects.

## Step 3 — the battery

Assert FULL objects everywhere: the complete `tools` list, the complete
`messages` list `acompletion` received, the complete stored entry. The
expected values are literally the OUT blocks of use_cases.md.

### Battery A — advertisement (UC0)

| id | Given | When | Then (`acompletion(tools=…)`) |
|---|---|---|---|
| A1 | empty conversation, `openai:gpt-5.1`, native on | run() | `[ApplyPatchTool(), LocalShellTool(), Tool("read")]` |
| A2 | empty, `anthropic:claude-sonnet-4-5`, native on | run() | `[TextEditorTool(), BashTool(), Tool("delete_file")]` |
| A3 | empty, `openrouter:kimi-2.7`, native on | run() | all four generics as function tools |
| A4 | empty, openai, native OFF | run() | all four generics as function tools |
| A5 | A1 ran once; flip native off | run() again | second call: generics only (the per-iteration re-stamp) |

### Battery B — inbound + execution (UC1, UC2, UC4, UC10, UC11-part1)

B1 — the example, spelled out (use this exact payload as the mock's typed
return; if step 0 chose the httpx boundary, it is the raw wire item):

```
Given  empty conversation, openai, native on, policy=require_approval
When   run(); mocked acompletion returns a message containing
       ApplyPatchToolCall(id="call_Rjs…", item_id="apc_08f3…",
           status="completed", name="apply_patch",
           arguments={type:"update_file", path:"lib/fib.py", diff:"@@…fibonacci…"})
Then   STORED AssistantMessage.parts == [ToolCall(
           id="call_Rjs…", name="openai_apply_patch",
           arguments={type:"update_file", path:"lib/fib.py", diff:"@@…"},
           extras={custom_type:"apply_patch_call", item_id:"apc_08f3…",
                   status:"completed"})]
       STORED ToolExecution: status PENDING (awaiting approval),
           raw_tool_call == the part (deep copy, not aliased)
When   approve; redrive with mocked OpenAIApplyPatchTool.execute
Then   execute received validated Args(type="update_file", path="lib/fib.py", diff=…)
       STORED ToolExecution COMPLETED, result == the dummy
```

| id | Given / When | Then |
|---|---|---|
| B2 | openai native on; mocked response = `ShellToolCall` (`item_id="sh_12"`, `commands=["pytest -q"]`); allow_all | stored `ToolCall(name="openai_shell", extras={custom_type:"shell_call", item_id:"sh_12", …})`; execution COMPLETED with `result.extras` = the shell_call_output dummy |
| B3 | anthropic native on; response = plain `ToolCall(name="bash", {command:"ls"})` | adopted: stored name `anthropic_bash_20250124`, `extras={}` |
| B4 | anthropic native OFF; generic `bash` advertised; response = `ToolCall(name="bash", …)` | NOT adopted: stored name `bash`, resolves to the generic tool (the gating edge case) |
| B5 | any provider; response = `ToolCall(name="read", …)` | pipeline byte-identical to today: no extras, no renames, executes (UC10) |
| B6 | openai native on; policy=deny; response = `ShellToolCall` | execution REJECTED, `result is None` |
| B7 | anthropic; response = `ToolCall(name="openai_apply_patch", …)` (model copied an inactive native from history) | not adopted (already internal), resolves, EXECUTES (UC7) |

### Battery C — outbound projection (UC3, UC5, UC8, UC9, UC11-part2)

Each test: build the session by running Battery-B-style turns first (or by
constructing entries directly), then post a user message, run, and assert the
full `messages` kwarg the mocked acompletion received.

| id | Given | Then (`acompletion(messages=…)`) |
|---|---|---|
| C1 | B2's session; still openai, native on | call: `ToolCall(id, name="shell", arguments, extras={custom_type:"shell_call", item_id:"sh_12", …})`; result: `ToolMessage(…, extras={custom_type:"shell_call_output", …})` — byte-identical to the original |
| C2 | same session; switch model to anthropic, native on | call: `name="openai_shell"`, NO extras; result: flat text, NO extras; tools per A2 (validated live: `poc_tests/`) |
| C3 | same session; switch to kimi | everything verbatim, generic tools |
| C4 | same session; switch BACK to openai native | identical to C1's expectation (the round trip) |
| C5 | session with both openai-born (B2) and anthropic-born (B3) calls; on openai native | openai pair upgraded, anthropic pair verbatim under `anthropic_bash_20250124`, in the same messages list (UC8) |
| C6 | B6's session (rejected shell); openai, native on | call upgraded; result: `ToolMessage(content="[tool execution rejected]", is_error=True, extras={custom_type:"shell_call_output", results:[{stdout:"", stderr:"[tool execution rejected]", outcome:{type:"exit", exit_code:1}}]})` (UC11 — the formerly-broken case) |
| C7 | B2's session; runner WITHOUT the plugin/middleware | everything verbatim, no crash, no extras anywhere (UC9 — fail-safe default) |

### Battery D — cross-cutting

| id | Scenario | Assertion |
|---|---|---|
| D1 | execution PENDING under openai; switch to anthropic; approve | mocked `execute` invoked with the validated Args; result stored (UC6) |
| D2 | model repeats the identical native call (same name+arguments, different `item_id`) | doom-loop flag set on the repeat — extras cannot defeat it |
| D3 | serialize the C5 session to JSON, reload, re-project | acompletion receives the identical messages list |
| D4 | streaming run (mock `acompletion_stream`) delivering a native call | adoption applied identically to B2 (this is verification V1 as a pinned test) |

## Explicit non-goals for phase 1

Real patch application, real subprocesses, the real
`contrib/shell/native` package, TUI rendering of `title`, native-specific
permission-request modeling, performance of `get_tool_execution`, and any
change to `luca/client`.

## Done when

`uv run py.test tests/` green (including `filterwarnings=error`), with the
full battery above present, plus `uv run ruff check --fix && uv run ruff
format` clean. The design is then considered validated; phase 2 ports the
test middleware/tables into `contrib/shell/native/` as real tools.
