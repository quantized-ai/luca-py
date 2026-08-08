Guidance for `luca.agent.contrib.shell`. Read this whenever you're working in
`luca/agent/contrib/shell/` or `tests/agent/contrib/shell/`.

## What this package is

The shell tool suite: eight generic filesystem/process tools (`read`, `glob`,
`grep`, `edit`, `write`, `apply_patch`, `delete_file`, `bash`) modeled on
Claude Code / OpenCode behavior, plus four PROVIDER-NATIVE ones in `native/`
(`openai_apply_patch`, `openai_shell`, `anthropic_text_editor_20250728`,
`anthropic_bash_20250124`) and the middleware that projects them. The
behavioral contract — exact output formats, error strings, scenario cases — is
pinned by the tests in `tests/agent/contrib/shell/`, which assert the strings
verbatim; the LLM-facing descriptions live on the tool classes.

`ShellAccessPlugin` (`plugin.py`) bundles all twelve behind one workspace:
absolute roots fixed at construction, ONE shared `FileReadTracker`, and one
seeded `PermissionStrategy` exposed as `permission_strategy` (the app's
approval prompt feeds `pending_requests()` / `apply_answer()` on it). ASK mode
seeds ALLOW rules for the read tier (`access_directory`, `read`, `glob`,
`grep`) over the workspace and each additional directory; YOLO allows
everything. It is a permission gate, not a sandbox — approval is the only
containment.

The registry always holds all twelve — a call born under one provider must
still resolve, validate and execute after a switch — and
`ShellNativeMiddleware` (registered through `get_middleware`) decides which of
them each REQUEST advertises. See the "Provider-native tools" section below.

## File layout

```
luca/agent/contrib/shell/
├── __init__.py   # public surface: the 8 generic tools, the 4 natives, the
│                 #   middleware, ShellTool, ShellToolError, FileReadTracker,
│                 #   ShellAccessPlugin
├── tools.py      # ALL generic tool classes + shared machinery (base class, tracker, locks, constants)
├── plugin.py     # ShellAccessPlugin: workspace/additional dirs, shared tracker,
│                 #   seeded PermissionStrategy, the shell-session pool + aclose(),
│                 #   registry + middleware + system-prompt hooks
├── session.py    # ShellSession/ShellSessionPool: the LIVE bash process behind
│                 #   anthropic_bash_20250124, one per conversation lineage
├── replace.py    # edit's 9 replacement-candidate strategies + the replace() driver (pure, no IO)
├── patch.py      # apply_patch's parser + hunk applier (pure, no IO — the tool owns all filesystem access)
└── native/       # the provider-native tools — see the section below
    ├── __init__.py    # the 4 tools, ShellNativeMiddleware, active_natives,
    │                  #   supported_native_tools
    ├── support.py     # WHICH (provider, model) pairs support which native — a
    │                  #   dated, hardcoded table
    ├── openai.py      # OpenAIApplyPatchTool, OpenAIShellTool
    ├── anthropic.py   # AnthropicTextEditorTool, AnthropicBashTool
    ├── middleware.py  # ShellNativeMiddleware: the four adaptation slots
    └── _files.py      # BOM/CRLF-preserving read/write + the metadata diff

tests/agent/contrib/shell/
├── conftest.py           # `run` / `perm` fixtures: validate args through Args like the registry would
├── test_plugin.py        # ShellAccessPlugin wiring, seeded rules, decide/pending flows
├── tools/test_<name>.py  # one file per generic tool, one section per behavior scenario
└── native/               # test_support.py, test_tools.py, test_middleware.py
```

The runner-level story — what `acompletion()` actually receives across
provider switches — is a separate battery at `tests/agent/test_native_tools/`.

## Shared machinery (tools.py)

- **`ShellTool(ResourcePermissionToolMixin, Tool)`** — the base. Every tool:
  - takes `workdir` at construction (defaults to `Path.cwd()`); relative
    argument paths resolve against it via `_resolve()` (normpath, no symlink
    resolution). The `AgentSession` carries no cwd — the workdir is instance
    state.
  - implements `_run(args, session, *, cancellation_token) -> ExecutionResult`;
    the base `execute` wrapper catches `ShellToolError` and returns it as an
    `ExecutionResult(is_error=True)`. Domain failures (missing file, ambiguous
    edit, bad regex, non-zero exit) are results, never raised to the runner.
  - implements `build_permission_requests()` (the mixin's override point).
    Synchronous on purpose: the mixin runs it in `asyncio.to_thread`, so the
    path stats it does (`_access_scope` calls `is_dir()` on every target) stay
    off the event loop. It is awaited inside the registry's `create_execution`,
    where a blocking syscall on a hung mount would otherwise stall the run.
- **`FileReadTracker`** — a set of resolved paths behind the read-first
  contract: `read` records every text file it returns; `edit`/`write` refuse
  to mutate an existing file that was never recorded, and record their own
  writes. One tracker instance MUST be shared across read/edit/write for the
  contract to hold (constructor arg `tracker=`; `ShellAccessPlugin` owns this).
- **Per-file locks** — module-level `asyncio.Lock` per resolved path
  serializes concurrent edit/write to the same file.
- **Blocking IO** runs in `asyncio.to_thread`; process-spawning tools use
  `start_new_session=True` and kill the process group on
  `asyncio.CancelledError` (the cancellation contract on `Tool`, in
  `luca/agent/contrib/tools.py`).

## Per-tool summary

| Tool | Kind | Key behavior |
|------|------|--------------|
| `ReadTool` | READ | Text page: `N: line` numbering, 2000-line / 2000-char-per-line / 50 KiB caps, tail note with continuation offset (`(End of file - total N lines)` / `(Showing lines A-B of N. …)` / `(Output capped at 50 KB. …)`), `<path>/<type>/<content>` envelope. Directories: sorted non-recursive `<entries>` page, `dir/` suffix. Images (jpeg/png/gif/webp by mime) → real `ImageContent` in `result.content` (base64, `metadata` naming the file); PDFs still a text stub plus `metadata={"attachment": {...}}`. Binary rejected by extension list OR NUL/30%-control sample sniff. Missing path suggests up to 3 close siblings. Records reads on the tracker. |
| `GlobTool` | SEARCH | `rg --files --hidden --glob '!**/.git/**' --glob <pattern>` with cwd = search root. Path arg must be an existing directory. Absolute paths out, capped at 100 (exactly 100 ⇒ treated as truncated). Empty ⇒ exactly `No files found`. |
| `GrepTool` | SEARCH | `rg --json` parsed for `match` events; one result per line, grouped by file (`Found N matches` header, `  Line X: preview`). Cap 100 with `(more matches available)` only when a 101st exists. Invalid regex ⇒ rg's stderr as an error result. 2000-char preview cap. |
| `EditTool` | EDIT | Unique exact replacement; `replace_all` for every occurrence; `old_string=""` creates a missing file (fails against an existing one). Fuzzy correction via `replace.py`'s strategy chain — a candidate is only applied if it literally occurs in the file. Preserves BOM and LF/CRLF convention (normalizes to LF internally, restores on write). Read-first enforced. Exact error strings for identical/not-found/ambiguous (asserted verbatim in tests). Returns unified diff + replacement count in metadata. |
| `WriteTool` | EDIT | Full-content write, creates parents. Read-first enforced for existing targets. Exactly one BOM preserved (existing file's or content's, never duplicated). Content round-trips exactly (empty, NUL, CRLF, no-final-newline). `metadata={"existed": bool}`. |
| `ApplyPatchTool` | EDIT | `*** Begin/End Patch` envelope with Add/Delete/Update(+Move) ops; heredoc wrapper accepted. Verify-everything-then-commit: a failing op leaves ALL files untouched (no rollback once commit starts — deliberate). Four line-matching passes (exact → rstrip → strip → unicode-punctuation), `@@ context` seek, `*** End of File` tail anchor. Kept context lines are copied from the original file. Updated files end with a newline; BOM preserved. Output `Success. Updated the following files:` + `A/M/D <path>` (moves show the destination); per-file diff/additions/deletions/move_to in `metadata["files"]`. |
| `DeleteFileTool` | DELETE | Removes ONE file; refuses a missing path and a directory (a symlink to a directory is removed as the link it is). Deliberately OUTSIDE the read-first contract — `read` refuses binaries, so a guard would make a stray `.pyc` undeletable. Its `delete_file` verb is not in the seeded read tier, so every call prompts. |
| `BashTool` | EXECUTE | Fresh `shell -c <command>` per call (shell from ctor/`$SHELL`/`/bin/bash`), stdin disabled, stderr merged into stdout, streamed. Tool-enforced timeout (default 120 000 ms) and cooperative cancellation both kill the process group and return partial output with a `<shell_metadata>` block (the note per outcome is `SHELL_METADATA_NOTES`); `metadata={"exit": int|None, "truncated", "output_path"}`. Output over 2000 lines / 50 KiB ⇒ tail preview + full output saved to a temp file (`output_dir=` ctor override). Non-zero exit is a result (`is_error=True`), not an exception. Description is a `.format` template rendered per instance with os/shell/tmp/limits. `_render` and `_truncate` are shared with the persistent native bash; `_effective_workdir` is the seam that one overrides. |

## Permission requests (what `build_permission_requests` returns)

Every tool returns TWO ordered requests: an `access_directory` step, then
its verb step. The access step (`ShellTool._access_request`) lists each
distinct directory the call touches, with one answer option per directory
granting `[<dir>, <dir>/*]` ("Always allow access to <dir>"; fnmatch `*`
crosses `/`, so the glob is recursive). Directory per tool: `read`/`grep` —
target if it is an existing directory, else its parent
(`ShellTool._access_scope`); `glob` — the search root; `edit`/`write` — the
target's parent; `apply_patch` — every touched path's parent (deduped);
`bash` — the effective workdir (`workdir` arg resolved, else the instance
workdir).

The verb steps:

- File tools (`read`/`edit`/`write`): `resources=[<resolved abs path>]`,
  metadata preview `"<Verb> <path>"`, one answer option `<parent>/*`.
- `glob`/`grep`: `resources=[<resolved search root or file>]`, answer option
  `<dir>/*` (a file target suggests its parent).
- `apply_patch`: every touched path — op sources plus move destinations —
  resolved absolute, `answer_options=[]`; unparseable patch text ⇒ ONE
  request with `resources=[]` and no access step (never raises).
- `bash`: `resources=[<stripped command string>]`, answer option `"<head> *"`
  (e.g. `git *`). Matching is the strategy's `fnmatch` — command globs are
  coarse by design.
- `delete_file`: `resources=[<resolved abs path>]` under the `delete_file`
  verb, answer option `<parent>/*`. Access scope is `_access_scope(path)`, so
  a directory target scopes to itself.

The natives deliberately reuse the GENERIC verbs, so a rule or an "always
allow" answer keeps meaning across a provider switch: `openai_apply_patch` →
`apply_patch`, `openai_shell` / `anthropic_bash_20250124` → `bash` (one
resource per command for the shell's list), and
`anthropic_text_editor_20250728` → the verb its COMMAND means: `view` → `read`
(so it lands in the auto-allowed read tier, exactly like the generic `read`),
`create` → `write`, `str_replace` / `insert` → `edit`. A bash `{"restart":
true}` declares only the access step — it runs nothing.

## Provider-native tools (`native/`)

Four ordinary `Tool`s under stable internal names. Nothing about them is a
special case in the core: they resolve, validate, gate and dispatch like any
other tool, and `ToolSpec.name` stays the identity every lookup keys on.
`title` (`"Apply patch"`, `"Shell"`, …) is presentation only, rendered by the
TUI through `ToolSpec.display_name`.

| Internal name | Wire | Replaces (generics dropped when active) |
|---|---|---|
| `openai_apply_patch` | `{"type": "apply_patch"}` | `edit` `write` `apply_patch` `delete_file` |
| `openai_shell` | `{"type": "shell", "environment": {"type": "local"}}` | `bash` |
| `anthropic_text_editor_20250728` | `{"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"}` | `edit` `write` `apply_patch` |
| `anthropic_bash_20250124` | `{"type": "bash_20250124", "name": "bash"}` | `bash` |

`read`, `glob` and `grep` survive every mode (nothing native reads an image or
searches a repo), and `delete_file` survives on Anthropic (`text_editor`
cannot delete).

**`anthropic_bash_20250124` is STATEFUL, and that is not optional.** A native
declaration has no description field, so the model cannot be told anything about
the tool it does not already believe — and what Anthropic taught it is that one
bash process stays alive across calls with every command running inside it.
`session.py` is that process:

- `ShellSession` — one `/bin/bash -s`, spawned on first use. Each command is
  written to a temp FILE and SOURCED with `< /dev/null` (so `cd`/`export`
  persist, heredocs survive verbatim, and a bare `cat` cannot eat the next
  command off our pipe), followed by a per-command uuid sentinel carrying `$?`
  and `$PWD`.
- `ShellSessionPool` — one session per conversation LINEAGE (`lineage_root`
  walks `previous_conversation_id`, so a compaction keeps its shell and a
  subagent gets its own). Owned and closed by `ShellAccessPlugin`.
- **Nothing is serialized.** A restart — `restart: true`, a timeout, a cancel,
  a command that runs `exit`, a resumed session — means a new, empty shell, and
  the plugin's `bash_restart_part` tells the model instead of rebuilding it.
  That notice is spent by USE, not by being emitted (`ShellSession.fresh`): the
  system prompt is re-sent whole every call, so a notice the model read and
  never acted on has to still be there when it finally runs a command.
- The permission scope follows the SHELL's cwd, not the workspace — otherwise
  the approval prompt names the wrong directory for a command the model already
  said it would run somewhere else.
- `ShellAccessPlugin.aclose()` kills the live processes. There is no teardown
  hook on `Tool`/`ToolRegistry`/`BasePlugin`, so the application calls it on its
  graceful paths (the TUI's `_quit` / `_reset_session`). A hard kill needs
  nothing: the shell's command pipe is held only by us, so it gets EOF and
  exits.

**The `REPLACES` table is per NATIVE TOOL, never per provider**, because
support is per MODEL: `gpt-5.5-pro` has the native shell and NO native
`apply_patch`, so it keeps every generic editor. `support.py` is the dated,
hardcoded capability table; `active_natives(session)` is where it meets the
session's `use_native_tools` switch, and it is the ONLY thing every decision
below reads.

**The editors reuse `luca.client.native`'s PURE transforms** (`apply_diff`,
`str_replace`, `insert`, `view`) and do their own IO in `_files.py`. That is
deliberate and worth not undoing: the client's filesystem executors confine
every path under one `root`, while here the APPROVAL is the containment (a
path outside the workspace prompts, it is never refused), and `_files.py`
preserves a BOM and CRLF endings exactly as `edit`/`write` do, so an edit made
natively and one made generically leave the same bytes.

**`openai_shell` is the only tool in the suite with a native RESULT shape.**
Per-command stdout/stderr/outcome cannot ride prose, so `execute()` captures
them into `ExecutionResult.extras` at birth — the only moment the per-command
split exists — and the middleware puts that dict back on the wire. Every other
native's result is plain text, which IS its native form.

That has one consequence worth stating loudly, because it is easy to
reintroduce: **on this tool the prose never reaches the model.** The projector
builds the wire item from `extras["results"]` alone and never reads `content`,
so any cap applied to the text is inert. `_cap` therefore clips the CAPTURED
streams — `bash`'s own 2000-line / 50 KiB budget, tightened by the model's
`max_output_length` when it set one — spills the rest to one temp file the
clipped text names, and `_assemble` renders the prose from those same capped
streams so both channels agree. `ContextManager` sizes an execution from
`content`, so keeping them equal is also what keeps context accounting honest.

**`ShellNativeMiddleware` owns all four adaptation slots** and is the only
place a wire exists:

| Slot | Vocabulary | Does |
|---|---|---|
| `build_tool_list` | `ToolSpec` | drops replaced generics and inactive natives |
| `adapt_tool_declarations` | client tools | swaps the survivors for the native declaration items |
| `before_llm_call` | client messages | upgrades MY active calls/results to their stored native payloads; synthesizes `shell_call_output` for a shell call that has no result (denied / failed / awaiting approval) |
| `after_llm_response` | client message | adopts inbound native calls under their internal names, extras kept |

Everything else projects VERBATIM — internal name, no extras, an ordinary
function call, which both providers accept even undeclared. That is the
fail-safe: uninstall the plugin and every session still projects.

`visible_names(session, names)` is the drop rule as a classmethod, shared by
`build_tool_list` and the plugin's system-prompt part (which is a CALLABLE, so
the prompt names the tools the model is about to be shown — including their
provider-facing names).

## Testing conventions

- Tests are self-scoped unit tests: no runner, no registry, no session. The
  `run` fixture (conftest) validates raw args through `Args` then calls
  `execute` with a fresh `CancellationToken`; `perm` builds the permission
  requests the same way. One test file per tool, sections grouped by
  scenario.
- **glob/grep mock the subprocess** at the `_run_ripgrep(argv, cwd)` boundary
  (there may be no `rg` binary on PATH — construct with a fake `rg_path`).
  Ignore/hidden/.git behavior is asserted as argv flags, not real traversal.
- read/edit/write/apply_patch use real `tmp_path` files; bash spawns real
  short-lived processes (the timeout/cancel tests sleep-and-kill, ~1s total).
- **A test that starts a persistent shell must close it.** `test_session.py`'s
  `shell` / `pool` fixtures and `test_tools.py`'s `bash` fixture are the only
  doors; `-W error::ResourceWarning` turns a leaked process into a failure,
  which is the point.
- `filterwarnings = error` applies: close every file handle
  (`Path.read_text`, not bare `open()`), await every spawned task.
- Every scenario with a distinct resource shape also asserts
  `build_permission_requests`.

## When touching this package

- Behavior questions → the tests are the contract: exact output and error
  strings are asserted verbatim in `tests/agent/contrib/shell/`. Changing a
  string is a contract change — update both deliberately.
- Edit matching bugs → `replace.py` (the strategy order is deliberate; the
  driver rejects candidates not literally present, and disproportionately
  large ones).
- Patch matching bugs → `patch.py` (`_PASSES`, `_find_sequence`, `apply_update`).
- New tool → subclass `ShellTool`, implement `_run` + `build_permission_requests`,
  raise `ShellToolError` for domain failures, add `tests/agent/contrib/shell/tools/test_<name>.py`.
  If it can be replaced by a native, add it to that native's `REPLACES` row.
- A provider ships a NEW native, or a new model supports one → `native/support.py`
  (the table plus a test in the same commit). A tool VERSION bump is a NEW tool
  with a new internal name: the old one leaves the tables and its stored calls
  project verbatim from then on — never rename one in place.
- Plugin changes → `plugin.py` + `tests/agent/contrib/shell/test_plugin.py`.
  Keep the invariants: roots stored absolute with the same
  normpath-no-symlink convention as `ShellTool._resolve` (mixed conventions
  break rule matching), ONE `FileReadTracker` and one workdir across the
  tools, seeded rules derived from the roots only, and one `ShellSessionPool`
  that `aclose()` releases.
- Persistent-shell changes → `session.py` + `tests/agent/contrib/shell/test_session.py`.
  The three load-bearing details are the sourced script (not stdin), the
  `< /dev/null` on it, and the per-command random sentinel — each one exists
  because dropping it silently corrupts a command's output rather than failing.
