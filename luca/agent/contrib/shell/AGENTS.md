Guidance for `luca.agent.contrib.shell`. Read this whenever you're working in
`luca/agent/contrib/shell/` or `tests/agent/contrib/shell/`.

## What this package is

The shell tool suite: seven filesystem/process tools (`read`, `glob`, `grep`,
`edit`, `write`, `apply_patch`, `bash`) modeled on Claude Code / OpenCode
behavior. The behavioral contract — exact output formats, error strings,
scenario cases — is pinned by the tests in `tests/agent/contrib/shell/`,
which assert the strings verbatim; the LLM-facing descriptions live on the
tool classes.

`ShellAccessPlugin` (`plugin.py`) bundles the seven tools behind one
workspace: absolute roots fixed at construction, ONE shared
`FileReadTracker`, and one seeded `PermissionStrategy` exposed as
`permission_strategy` (the app's approval prompt feeds
`pending_requests()` / `apply_answer()` on it). ASK mode seeds ALLOW rules
for the read tier (`access_directory`, `read`, `glob`, `grep`) over the
workspace and each additional directory; YOLO allows everything. It is a
permission gate, not a sandbox — approval is the only containment.

## File layout

```
luca/agent/contrib/shell/
├── __init__.py   # public surface: the 7 tools, ShellTool, ShellToolError,
│                 #   FileReadTracker, ShellAccessPlugin
├── tools.py      # ALL tool classes + shared machinery (base class, tracker, locks, constants)
├── plugin.py     # ShellAccessPlugin: workspace/additional dirs, shared tracker,
│                 #   seeded PermissionStrategy, registry + system-prompt hooks
├── replace.py    # edit's 9 replacement-candidate strategies + the replace() driver (pure, no IO)
└── patch.py      # apply_patch's parser + hunk applier (pure, no IO — the tool owns all filesystem access)

tests/agent/contrib/shell/
├── conftest.py           # `run` / `perm` fixtures: validate args through Args like the registry would
├── test_plugin.py        # ShellAccessPlugin wiring, seeded rules, decide/pending flows
└── tools/test_<name>.py  # one file per tool, one section per behavior scenario
```

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
| `BashTool` | EXECUTE | Fresh `shell -c <command>` per call (shell from ctor/`$SHELL`/`/bin/bash`), stdin disabled, stderr merged into stdout, streamed. Tool-enforced timeout (default 120 000 ms) and cooperative cancellation both kill the process group and return partial output with a `<shell_metadata>` block; `metadata={"exit": int|None, "truncated", "output_path"}`. Output over 2000 lines / 50 KiB ⇒ tail preview + full output saved to a temp file (`output_dir=` ctor override). Non-zero exit is a result (`is_error=True`), not an exception. Description is a `.format` template rendered per instance with os/shell/tmp/limits. |

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
- Plugin changes → `plugin.py` + `tests/agent/contrib/shell/test_plugin.py`.
  Keep the invariants: roots stored absolute with the same
  normpath-no-symlink convention as `ShellTool._resolve` (mixed conventions
  break rule matching), ONE `FileReadTracker` and one workdir across the
  tools, seeded rules derived from the roots only.

## Provider-defined tools

Some providers define their own tools and train the model on the schema.
`native.py` holds them. The request carries a type string instead of a schema,
the model emits an ordinary `tool_use` block, and LUCA still executes it — so
approval, the read-first guard, the recorded execution and replay are unchanged.

`NativeTextEditorTool` is Anthropic's `str_replace_based_edit_tool`. It does not
reimplement anything: each command translates the provider's arguments into
luca's and delegates to `read` / `edit` / `write`, including for
`build_permission_requests`, so a native call is gated by exactly the rules a
plain call would produce. Only `insert` has no counterpart, and even that
splices and then writes through `write` so the guard, the lock and the BOM
handling stay in one place.

`NativeBashTool` is Anthropic's `bash`, and it is a different tool rather than a
rename: the provider specifies ONE shell session whose working directory,
environment and background processes survive between calls, with a `restart`
that starts clean. luca's own `bash` is a fresh subprocess per call.
`session_shell.py` holds the session — a long-lived shell driven over its stdin,
each command followed by a per-call random marker carrying `$?`. One shell PER
CONVERSATION, because a tool instance is shared by the main agent and every
subagent and a single session would mean one conversation's `cd` relocating
another's next command. `ShellAccessPlugin.close()` releases them; the TUI calls
it on quit and on every session swap, or a long run accumulates idle shells.

The editor REPLACES `read` / `edit` / `write` rather than joining them — two tools for
one job is a choice the model should not have to make. `glob`, `grep`,
`apply_patch` and `bash` have no native equivalent and always stay.

`native_editor_type(provider, model)` decides, and it keys on the TRANSPORT, not
the model family. Bedrock and OpenRouter both serve Claude models and neither
speaks the Messages API, so a "is this a Claude?" check would confidently send a
tool the API rejects. Anthropic's editor version is keyed to the model
generation, which is why the type is an instance attribute rather than a
ClassVar.
