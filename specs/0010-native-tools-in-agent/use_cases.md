# 0010 — use cases, traced

Every flow in [draft.md](draft.md), run as a black box: given the stored data
model and the session's active config, what the plugin's middleware does, and
exactly what `acompletion(messages=…, tools=…)` receives — or what gets
stored. Actors:

```
PROJ   ConversationProjector — UNTOUCHED, provider-blind
MW     ShellNativeMiddleware — the plugin's middleware (build_tool_list,
       adapt_tool_declarations, before_llm_call, after_llm_response);
       reads session.llm_config + session.use_native_tools, both stamped by
       the runner at the top of each drive iteration
RUNNER AgentSessionRunner          REG    SimpleToolRegistry — UNTOUCHED
TOOL   the executing tool          STORED session entries
```

Session data is the draft's worked example: `tc1`/`tc2` born on OpenAI,
`tc5`/`tc6` born on Anthropic. The boundary of every OUT trace is the
arguments handed to `acompletion()` — the client's own projection is not the
agent's concern.

---

## UC0 — the advertised tools, per active config

```
IN   REG.get_tools() → ALL twelve specs (generics + 4 natives) — same list, always
     RUNNER: is_private filter

MW.build_tool_list(cid, specs):            spec vocabulary — DROP only
  openai, native     → [openai_apply_patch, openai_shell, read]
  anthropic, native  → [anthropic_text_editor_20250728, anthropic_bash_20250124, delete_file]
  else               → every generic, no natives

RUNNER: spec → function-declaration conversion (internal names)

MW.adapt_tool_declarations(cid, tools):    client vocabulary — SWAP only
  my native declarations → the client's native items; everything else untouched

OUT  acompletion(tools=…)
  openai, native     [ApplyPatchTool(), LocalShellTool(), Tool("read", {…})]
  anthropic, native  [TextEditorTool(), BashTool(), Tool("delete_file", {path})]
  kimi / native off  [read, write, edit, glob, grep, bash, apply_patch, delete_file]
```

---

## UC1 — an OpenAI native call arrives

```
IN   CLIENT returns AssistantMessage(content=[
       ApplyPatchToolCall(id="tc1", name="apply_patch", item_id="apc_88",
                          status="completed",
                          arguments={type:"update_file", path:"main.py", diff:"@@…"})])

MW.after_llm_response(cid, message):
     block.as_generic()
       → ToolCall(id="tc1", name="apply_patch", arguments={…},
                  extras={custom_type:"apply_patch_call", item_id:"apc_88",
                          status:"completed"})
     ADOPT_BY_CUSTOM_TYPE["apply_patch_call"] → rename to "openai_apply_patch"
RUNNER records: message_to_parts (isinstance dispatch), then
     _receive_executions: raw = call.model_copy(deep=True)
     _is_doom_loop compares name + arguments only — extras can't defeat it

OUT  STORED
     AssistantMessage(e3, parts=[
       ToolCall(id="tc1", name="openai_apply_patch",
                arguments={type:"update_file", path:"main.py", diff:"@@…"},
                extras={custom_type:"apply_patch_call", item_id:"apc_88",
                        status:"completed"})])
     ToolExecution(e4, tool_call_id="tc1", raw_tool_call=<deep copy>, RECEIVED)
```

---

## UC2 — a native call executes (shell: result captured structured, at birth)

```
IN   ToolExecution(e5, tc2, RECEIVED)
     raw_tool_call = ToolCall("tc2", "openai_shell",
                              {commands:["pytest -q","ruff check"], timeout_ms:60000},
                              extras={custom_type:"shell_call", item_id:"sh_12", …})

REG  create_execution → resolve("openai_shell") → the spec
TOOL OpenAIShellTool.Args.model_validate(arguments) → VALID
REG  decide → ALLOW;  prepare → PreparedCall
TOOL execute: one subprocess per command, in order

OUT  STORED
     ToolExecution(e5, tc2, COMPLETED, result=ExecutionResult(
       content=[Text("2 passed\nAll checks passed")],              # model-facing text
       extras={custom_type:"shell_call_output",                    # native shape, at birth
               results:[{stdout:"2 passed",          stderr:"", outcome:{type:"exit", exit_code:0}},
                        {stdout:"All checks passed", stderr:"", outcome:{type:"exit", exit_code:0}}]}))
```

---

## UC3 — same-provider replay: stored → acompletion, native

```
IN   STORED e3 + e5          session.llm_config → openai, native on

PROJ (untouched) → the DEFAULT form:
       ToolCall(id="tc2", name="openai_shell", arguments={…})        # no extras
       ToolMessage(tool_call_id="tc2",
                   content=[Text("2 passed\nAll checks passed")])    # no extras
MW.before_llm_call — openai_shell IS active → upgrade:
       call:   name → WIRE_NAMES["openai_shell"] = "shell"
               extras ← session.get_tool_execution("tc2").raw_tool_call.extras
       result: extras ← execution.result.extras

OUT  acompletion(messages=[…,
       AssistantMessage(content=[
         ToolCall(id="tc2", name="shell",
                  arguments={commands:["pytest -q","ruff check"], timeout_ms:60000},
                  extras={custom_type:"shell_call", item_id:"sh_12", status:"completed"})]),
       ToolMessage(tool_call_id="tc2", content=[Text("2 passed\nAll checks passed")],
                   extras={custom_type:"shell_call_output", results:[{…}, {…}]}),
     ])
     # the client's generic form — equivalent by construction to the typed
     # ShellToolCall / ShellToolMessage (Ground truth 1); replay is
     # byte-identical to the original emission
```

---

## UC4 — an Anthropic native call arrives (wire name is the identity)

```
IN   CLIENT returns AssistantMessage(content=[
       ToolCall(id="tc6", name="bash", arguments={command:"pytest -q"})])  # base class

MW.after_llm_response:
     as_generic() → identity (already generic, extras={})
     no custom_type; "bash" is an active native's wire name on anthropic
     ADOPT_BY_WIRE_NAME["bash"] → rename to "anthropic_bash_20250124"

OUT  STORED ToolCall(id="tc6", name="anthropic_bash_20250124",
                     arguments={command:"pytest -q"}, extras={})
```

---

## UC5 — provider switch: OpenAI history, Anthropic request

```
IN   STORED tc1..tc4 (OpenAI-born)          session.llm_config → anthropic, native on

PROJ → the default form: internal names, no extras, results as flat text
MW.before_llm_call — openai natives are NOT active on anthropic → NO-OP
     # verbatim is not a branch; it is the default the projector already produced

OUT  acompletion(
       tools=[TextEditorTool(), BashTool(), Tool("delete_file", {path})],   # UC0
       messages=[…,
         AssistantMessage(content=[
           ToolCall(id="tc1", name="openai_apply_patch",
                    arguments={type:"update_file", path:"main.py", diff:"@@…"}),
           ToolCall(id="tc2", name="openai_shell",
                    arguments={commands:["pytest -q","ruff check"], timeout_ms:60000})]),
         ToolMessage(tool_call_id="tc1", content=[Text("Updated main.py (2 hunks)")]),
         ToolMessage(tool_call_id="tc2", content=[Text("2 passed\nAll checks passed")]),
       ])
     # undeclared names in history — ACCEPTED, validated live 2026-08-07:
     #   poc_tests/anthropic_undeclared_history.py  (this direction)
     #   poc_tests/openai_undeclared_history.py     (mirror direction)
```

---

## UC6 — a pending approval executes after the switch

```
IN   ToolExecution(e9, tc4, PENDING)                 born under openai
     raw_tool_call = ToolCall("tc4", "openai_apply_patch",
                              {type:"delete_file", path:"secrets.env"}, extras={…})
     session.llm_config → anthropic

user approves → dispatch
REG  resolve("openai_apply_patch") → found        # get_tools never shrank
TOOL Args.model_validate → VALID;  execute → the file is deleted

OUT  STORED ToolExecution(e9, tc4, COMPLETED,
                          result=ExecutionResult(content=[Text("Deleted secrets.env")]))
     # replays from now on via UC5 — the switch is invisible to dispatch
```

---

## UC7 — the model copies an inactive native's name out of history

```
IN   CLIENT returns ToolCall(id="tc7", name="openai_apply_patch",
                             arguments={type:"update_file", path:"main.py", diff:"@@…"})
     session.llm_config → anthropic

MW.after_llm_response: no custom_type; "openai_apply_patch" is no wire name → NO-OP
RUNNER stores it as-is (it already carries the internal name)
REG  resolve("openai_apply_patch") → found → validate → decide → execute

OUT  STORED ToolExecution(e14, tc7, COMPLETED, result=…)
     # it is a real tool that works; not being advertised is not an error
```

---

## UC8 — switch back: native rebuilt from storage

```
IN   STORED tc1 tc2 (OpenAI-born), tc5 tc6 (Anthropic-born)
     session.llm_config → openai, native on

PROJ → default form for all four
MW.before_llm_call:
     tc1, tc2 → active → upgraded (UC3): wire names + stored extras, call and result
     tc5, tc6 → not active on openai → NO-OP → stay
       ToolCall(id="tc6", name="anthropic_bash_20250124", arguments={command:"pytest -q"})
       ToolMessage(tool_call_id="tc6", content=[Text("2 passed")])

OUT  acompletion(messages=[ upgraded tc1+tc2 pairs, verbatim tc5+tc6 pairs, … ])
```

---

## UC9 — session resumed with the plugin uninstalled

```
IN   STORED tc1..tc6          no ShellAccessPlugin → no middleware registered

PROJ → the default form — which needs no tool class, no plugin, no config
(no MW)

OUT  acompletion(messages=[ every call under its internal name, every result
                            as flat text ])          # exactly UC5's shape
     # fail-safe by construction: the default is always valid
```

---

## UC10 — an ordinary function tool (the 99% path)

```
IN   CLIENT returns ToolCall(id="tc9", name="read",
                             arguments={file_path:"main.py"})          extras={}

MW.after_llm_response: as_generic() → identity; no custom_type, no wire-name match → NO-OP
STORED as-is;  executes;  ExecutionResult(content=[Text("1|def fib(n):…")], extras={})
PROJ → ToolCall("tc9", "read", {…}) + ToolMessage(content=[Text("1|…")])
MW.before_llm_call: "read" is not a native → NO-OP
MW.build_tool_list: "read" kept

OUT  acompletion identical to today, byte for byte. Zero special cases.
```

---

## UC11 — a native shell call is DENIED

```
IN   CLIENT returns ShellToolCall(id="tc8", …, arguments={commands:["rm -rf build"]})
     → adopted (UC1) → STORED ToolCall("tc8", "openai_shell", {…},
                                       extras={custom_type:"shell_call", item_id:"sh_44", …})
REG  decide → user answers NO
     → STORED ToolExecution(e20, tc8, status=REJECTED, result=None)
     # no ExecutionResult exists — execute() never ran; nothing wrote result extras

next request:   session.llm_config → openai, native on

PROJ → ToolCall("tc8", "openai_shell", {…})                          # default form
       ToolMessage(tool_call_id="tc8",
                   content=[Text("[tool execution rejected]")],       # DERIVED from
                   is_error=True)                                     # status (projection.py:166)
MW.before_llm_call → _upgrade_result:
     mine, active, execution.result is None → shell needs a structured result →
     SYNTHESIZE from the derived message:
       extras = {custom_type:"shell_call_output",
                 results:[{stdout:"", stderr:"[tool execution rejected]",
                           outcome:{type:"exit", exit_code:1}}]}
     call upgraded as in UC3 (wire name + stored extras)

OUT  acompletion(messages=[…,
       ToolCall(id="tc8", name="shell", arguments={…}, extras={custom_type:"shell_call", …}),
       ToolMessage(tool_call_id="tc8", content=[Text("[tool execution rejected]")],
                   is_error=True,
                   extras={custom_type:"shell_call_output", results:[{…exit_code:1}]}),
     ])
     # same method that upgrades every result — one more branch, not an edge case
```
