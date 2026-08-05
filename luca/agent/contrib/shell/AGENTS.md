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
├── patch.py      # apply_patch's parser + hunk applier (pure, no IO — the tool owns all filesystem access)
├── native.py     # the provider-defined tools + which route gets which
└── session_shell.py  # PersistentShell: one long-lived shell, sentinel protocol

tests/agent/contrib/shell/
├── conftest.py           # `run` / `perm` fixtures: validate args through Args like the registry would
├── test_plugin.py        # ShellAccessPlugin wiring, seeded rules, decide/pending flows
├── test_native.py        # the native tools, their permissions, and the route table
├── test_session_shell.py # PersistentShell against real shell processes
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
plain call would produce. Only `insert` has no counterpart, and it is the one
command that owns its own read-modify-write: it mirrors `EditTool` byte for
byte (read bytes, strip the BOM, detect CRLF, work in LF, restore on the way
out) under the same per-path lock, because routing through `write` meant a
round trip to text, and that is where a file's CRLF endings, its BOM and its
line NUMBERING were being lost. Split on `\n` only — `str.splitlines` breaks
on form feeds and Unicode separators that `read` never counted as lines, so
the model's line numbers and the splice disagreed.

`NativeBashTool` is Anthropic's `bash`, and it is a different tool rather than a
rename: the provider specifies ONE shell session whose working directory,
environment and background processes survive between calls, with a `restart`
that starts clean. luca's own `bash` is a fresh subprocess per call.
`session_shell.py` holds the session — a long-lived shell driven over its stdin,
each command written BETWEEN a per-call random start marker and an end marker
carrying `$?`. Both markers matter. The start one exists because a backgrounded
job keeps the shell's stdout after the foreground command returns, so what it
writes between calls is sitting in the pipe when the next command runs and gets
read as that command's output, exit code and all; everything before the start
marker is dropped. Reads take CHUNKS, never lines: `readline()` is bounded by
the stream limit and raises on the first line past 64 KiB, which one `cat` of a
bundled asset produces, and that exception used to escape `run()` and strand
the sibling reader on the pipe for the life of the session. The shell is
`/bin/bash`, never `$SHELL` — the wrapper is Bourne-specific, and under fish or
tcsh no marker is ever printed and every command sits until its timeout.

A timeout or a user cancel takes the session down with it: the command runs IN
the shell, so there is nothing to signal that leaves the shell standing. Both
return the partial output and SAY the session was reset, because the silent
version is what makes the next command land in a directory the model did not
choose.

One shell PER CONVERSATION, because a tool instance is shared by the main agent
and every subagent and a single session would mean one conversation's `cd`
relocating another's next command. A finished subagent's shell is released on
the next call; `ShellAccessPlugin.close()` releases the rest, and the TUI calls
it on quit and on every session swap. A handle left over from a CLOSED event
loop is reaped by hand rather than reused — its pipes belong to that loop, so an
embedder calling `asyncio.run` once per turn would otherwise get
`Future attached to a different loop` forever.

The editor REPLACES `read` / `edit` / `write` rather than joining them — two tools for
one job is a choice the model should not have to make. `glob`, `grep`,
`apply_patch` and `bash` have no native equivalent and always stay.

OpenAI's two are a different shape. `apply_patch` and `shell` do not arrive as
`tool_use` blocks at all: the model emits `apply_patch_call` / `shell_call`
items and expects `apply_patch_call_output` / `shell_call_output` back, so the
Responses transport parses them into ordinary `ToolCall`s on the way in and
rebuilds the matching item on the way out. Above the transport nothing knows.
The item carries no tool name, so an INCOMING call is resolved from whichever
tool was offered with that provider type on the same request. Replaying a
recorded one does not work that way: `ToolCall.provider_type` records what the
call arrived as, and the projection reads that. Names are not durable
identifiers — luca ships its own `apply_patch` too, so a history written before
native tools existed, or with them turned off, would replay as an
`apply_patch_call` whose operation has no type, path or diff. The reverse is
guarded as well: a native call replays as a plain function call when its type
is not among the tools this request declares.

Streaming is a SEPARATE parser (`stream.py`) and it needs its own handling —
`_item_added` / `_item_done` — or the item is dropped and the turn ends with no
content at all. Streaming is the TUI default, so a gap there is the whole
feature missing while every non-streaming test passes.

`NativeApplyPatchTool` wraps the provider's per-file operation in a
`*** Begin Patch` envelope and hands it to `ApplyPatchTool` — the hunk syntax is
the same family, so the parser, the fuzzy context matching and the diff
rendering are the ones that already ship. `NativeShellTool` runs the call's
command LIST in one `PersistentShell`, stopping at the first failure, and asks
approval for every command rather than only the first.

`shell` declares `environment: {"type": "local"}`. The alternative is a
container OpenAI provisions, which is a different product and not what luca
executes.

`native_editor_type` / `native_bash_type` / `native_openai_tool_types` decide,
and `ShellAccessPlugin` calls them PER REQUEST rather than at construction —
`/model` changes the route mid-session and nothing rebuilds the composition, so
a set frozen at build time keeps sending Anthropic's editor to the Responses
API. They key on the TRANSPORT, not the model family. Bedrock and OpenRouter both
serve Claude models and neither speaks the Messages API, so a "is this a
Claude?" check would confidently send a tool the API rejects. The same on the
other side: groq and deepseek serve GPT-shaped ids over chat completions, where
`shell` does not exist at all. Anthropic's editor version is keyed to the model
generation, which is why the type is an instance attribute rather than a
ClassVar — and the NAME moves with it, since `text_editor_20250728` pairs with
`str_replace_based_edit_tool` and `text_editor_20250124` with the older
`str_replace_editor`, and changing one without the other is a 400.

Both native shells truncate like luca's `bash` does (2000 lines / 50 KiB, full
output spilled to a temp file), and `shell` honours the `max_output_length` its
schema advertises. Uncapped, one `cat` of a build log goes into the request,
the response and every later save of the session file.

`view_range` is validated on `Args` rather than in `_view`, and that placement
is the point: luca does not own this schema, so malformed values arrive and
the message has to name a field the model can resend. A short list used to die
on `window[1]`, a start below 1 failed against `ReadTool`'s own `offset` (a
field the native schema cannot send), and `[5, 2]` was worse than either — it
returned one line and told the model it had shown lines 5 to 5.

`ShellAccessPlugin.native_key_for` and `.install_tools` are the OVERRIDE
POINTS, and they are public for that reason: the first decides which provider
tools a route gets, the second composes the list. A developer with a host luca
does not know, or a policy of their own, subclasses one of them rather than
patching a module function. `ModelAwareRegistry` is exported too, and its
`sync` is public because a subclass overriding `get_tools` still has to call
it. Both are covered by tests that actually subclass — an override point with
no such test is a claim, not a seam.

