# TUI

A [Textual](https://textual.textualize.io/) terminal UI for the agent loop —
the interactive counterpart of the classic REPL: a chat transcript, live
streaming, a modal approval gate, Esc cancellation, and per-run session
persistence. It is also the reference for wiring a real interactive app on
top of the runner: one drive worker, one shared
[`PermissionStrategy`](../resource_permissions/README.md), sessions saved as
`~/.luca/projects/<encoded-project-path>/<session-id>.json`. Requires the `tui` dependency group (installed by
default with `uv sync`).

```python
from luca.agent.contrib.tui import AgentApp, build_runner, main
```

## 1. Run it

```bash
uv run python main.py                     # fresh session (needs OPENROUTER_API_KEY)
uv run python main.py --faux              # offline scripted demo — no key, no network
uv run python main.py --resume            # pick a past session for this project
uv run python main.py --conversation <id> # resume that session (--fork to branch)
uv run python main.py --no-streaming      # block-level events instead of deltas
uv run python main.py --no-subagents      # stop it spawning subagents that work in parallel
uv run python main.py --subagents-max-depth 1     # no nesting (the app default is 3)
uv run python main.py --subagents-max-per-turn 5  # spawn budget per turn (default: none)
uv run python main.py --subagents-max-workers 3   # how many work at once (default: no cap)
uv run python main.py --model moonshotai/kimi-k2.7-code --reasoning high
uv run python main.py --config ./ci.json  # use THIS config file, skip discovery
uv run python main.py --conversation <id> --pretty-print  # transcript, then exit
```

`--model` / `--provider` / `--reasoning` update the session's `LLMConfig`;
they persist with the session and override the stored values on a resume.

Subagents are on by default: the app installs `SubagentsPlugin` **and** sets
`subagents_enabled` on the session ([`subagents/`](../subagents/README.md)) —
both are required, since installing the tools is not the same as switching the
capability on. A subagent gets the same shell and memory tools the main agent
has, each keyed by conversation so the two never overwrite each other.
`--no-subagents` withholds the plugin and clears the flag on the session, so a
resumed session that had subagents on comes back with them off. The three
limit flags write the same way — every launch, resumed sessions included, so
the flags always describe the run being started: depth defaults to 3 (main
plus three levels of nesting), the per-turn spawn budget and the worker cap
default to no limit ([`08-runtime-config.md`](../../08-runtime-config.md)).

`--pretty-print` replaces the app: it loads the session from the store, writes the
[`pretty_print`](../../02-data-model.md#13-read-a-saved-session) transcript to
stdout and exits, so it requires `--conversation` (a usage error without one)
and ignores every other flag — nothing is started, nothing is saved.

> ⚠️ **Old session files do not load.** A `<id>.json` written before tool-spec
> normalization fails validation on `--conversation` and on `--pretty-print`.
> There is no migration — start a fresh session.

Defaults, custom providers, permission mode, runtime limits, compaction, and
the workspace can all live in a `luca.json` file instead of flags. See
[`config.md`](config.md). Precedence is CLI flag > `./luca.json` >
`~/.config/luca/luca.json` > the persisted session > built-in default.

`main.py` is a thin dotenv launcher over `python -m luca.agent.contrib.tui`
(same flags).

## 2. What's on screen

| Piece | Behavior |
|---|---|
| Transcript cells | One bordered cell per block: `you`, `assistant`, `thinking`, `tool` (call → running → result, clipped; `running` shows only once the body is dispatched, so a denied or unresolved call jumps straight to its status), `compacted` (a summary, subtitled with how many entries it replaced), `notice` (cancels, failures). Assistant and thinking cells render markdown (bold, lists, fenced code); tool-call argument values are clipped to a one-line preview so a large `write`/`edit` does not dump its whole payload |
| Subagent panels | One indented `SubagentPanel` per subagent, titled with the task it was given and subtitled `waiting… / running… / done / failed` — `waiting…` while queued behind `--subagents-max-workers` or parked at a gate, driven by the subagent lifecycle events. Everything that subagent produces — its text, its reasoning, its own tool cells — is mounted **inside** its panel. See §2.1 |
| Input box | A multiline `PromptInput` (grows with its content, capped at 10 rows), enabled **even while the agent works**: submitting mid-turn posts into the open turn (rendered immediately, answered before the turn closes), and a rejection — cancelling, subagents active — shows a notice and keeps the draft. Enter posts the message (starting the drive worker when none is running); Alt+Enter, Shift+Enter, Ctrl+J, or Enter after a trailing `\` insert a newline (the modified Enters need a terminal honoring the kitty keyboard protocol — under tmux, `extended-keys`); a multiline paste lands verbatim without sending. A line starting with a known `/command` runs that command instead of sending it (commands stay idle-only — mid-turn they notice and keep the draft), and typing `/` completes command names (Right accepts) |
| Status line | The header shows `session <id> · <provider>:<model> · <status>` (plus the reasoning level when set), so the live model is always visible |
| Context bar | A one-line gauge under the transcript showing context utilization (`▐████░░░░▌ 42% 84k/200k`), colored toward red as it nears the compaction threshold. Reads the `calculate_context_used` / `get_context_window_size` gauge from `contrib/simple_context_manager` |
| `Ctrl+V` | Attaches the clipboard's image to the next message; the transcript shows `[image: pasted-1.png]` |
| Approval modal | One screen per uncovered permission step: Approve once / tool-suggested ALWAYS grants / Deny / Abandon — pick by button or digit key. A gate raised by a subagent names it; the main agent's are unlabelled |
| `Esc` | Cancels the live run (`run.cancel()`); the wind-down renders live and the turn closes `CANCELLED`, cascading to every live subagent |
| `Ctrl+D` | Saves the session and quits |

### 2.1 How subagents are drawn

A run's stream is its whole conversation subtree's, so several subagents'
events arrive interleaved. **Attribution therefore cannot be positional** — a
flat transcript would splice three conversations into one column. Every event
carries `conversation_id`, and the app keys both the live-cell state and the
MOUNT TARGET by it:

```
╭─ assistant ──────────────────────────────────────────╮
│ Three files, three helpers. Spawning them now.       │
╰──────────────────────────────────────────────────────╯

   ╭─ subagent · read alpha.txt ───────────────────────╮
   │ Read alpha.txt and say what it is.                │
   │  ╭─ tool · read ─────────────────────────────╮    │
   │  │ read(file_path='…/alpha.txt')             │    │
   │  ╰──────────────────────────────────── done ─╯    │
   │  ╭─ assistant ───────────────────────────────╮    │
   │  │ alpha.txt is a shopping list.             │    │
   │  ╰───────────────────────────────────────────╯    │
   ╰─────────────────────────────────────────── done ─╯
```

Two tool calls per subagent never appear, and that is deliberate: the **spawn**
tool renders as the panel its child got (a cell beside it would say the same
thing twice), and the **result** tool is private — runtime-invoked, never shown
to the model — so drawing it as a call the model appears to have made would be
a lie. Both are matched by DECLARATION (`is_private`, an `output_schema` that
declares `is_subagent_spawn`), never by name, so an application that ships its
own spawn/result pair renders exactly the same way.

A panel closes off the resolved `ChildConversation`, not off the result tool's
event. That distinction is load-bearing: a cancelled subagent is resolved by
the parent's wind-down **without** the result tool ever running, so a panel
driven by the event alone would sit on `running…` forever on precisely the
path where you most need to see it stop.

Resume replays it identically. `ChildConversation` mounts the panel and the
child's own path replays inside it, so a reloaded session shows the delegated
work rather than a gap where it happened. The seed prompt is not replayed as a
cell — the panel's border already carries it.

## 3. Slash commands

Type a command at the prompt while idle. The first word is matched against the
registry in `commands.py`; anything else (a real path like `/etc/hosts`, a
typo) is sent to the agent as a normal message, so nothing is swallowed.

| Command | Effect |
|---|---|
| `/help` | List every command (rendered from the registry, so it never drifts) |
| `/model [provider:model]` | No arg drills down: pick a provider, then one of its models. `provider:model` switches both, a bare id switches only the model. Takes effect next turn |
| `/reasoning [level]` | No arg opens a picker of the reasoning levels; an arg sets it directly |
| `/compact` | Summarize the history and continue on a new conversation ([12](../../12-compaction.md)). Needs a `context_manager=` on the app that implements `compact()`; with the accounting-only default the drive reports a turn failure |
| `/model` | Pick a provider, then a model. Providers come from the catalog intersected with the transports luca has; the model step filters as you type, since openrouter alone lists hundreds. `/model provider:model` still switches to anything, listed or not |
| `/new` | Save the current session, then start a fresh one with the same model and an empty transcript. The old one stays in the store |
| `/resume` | Pick another session for this project and switch to it, replaying its transcript. The one being left is saved first, unless it is still empty |
| `/quit` | Save and exit (same as `Ctrl+D`) |

The pickers are `PickerScreen` modals (arrow keys to move, Enter to select, Esc
to cancel), with the current value pre-highlighted and marked "(current)".
`/model` is two steps (provider, then model) so a provider is never left with a
mismatched model, and Esc at either step changes nothing. The model step ends
with a "← Back to providers" entry that returns to the provider step so the
provider can be changed without starting over. The list is the curated
`RECOMMENDED_MODELS` in `wiring.py` grouped by provider, not the client catalog
(too sparse to drive a picker); `/model <provider:model>` still switches to
anything off the list, including providers not shown in the picker. The
direct form rejects an empty half (`/model openai:`), and `/new` carries the
session's runtime config (timeouts, step limits) into the fresh session.

Adding a command is one `SlashCommand` entry in `commands.py`; `/help` picks it
up automatically.

## 4. Structure

The Textual-free modules hold everything worth unit-testing; the widgets stay
thin:

| Module | Role |
|---|---|
| `wiring.py` | `build_runner(session, workspace=, provider=, mode=, context_manager=, additional_directories=, extra_rules=, subagents=)` — shell + memory plugins, the demo math tools ([`contrib.tools.Tool`](../tools/README.md) subclasses), one shared strategy; `build_faux_provider()` scripts the `--faux` conversation |
| `approvals.py` | `build_approval_prompts(execution, strategy, main_conversation_id=, subagent_labels=)` — pending steps → `ApprovalPrompt`s whose options carry fully-built `ApprovalAnswer`s (the whole gate policy, no UI). The main id is what lets a prompt say which subagent is asking; `subagent_labels` is what lets it say so in the subagent's own terms, since two can gate at the same moment and an id only answers "which one" to someone who already knows |
| `sessions.py` | The store: `resolve_session_directory` (root + encoded project path), `list_sessions` for the `/resume` picker, and load / save / fork — the save is atomic (temp file + `os.replace`), which is the application's job since the core owns no persistence |
| `render.py` | Pure formatting and session reads: `format_tool_call`, `clip_text`, `status_label`, `user_transcript_text`, `compaction_transcript_text` (the live and replayed transcript share them, so they cannot drift), plus `is_runtime_plumbing`, `subagent_task` and `child_links` for the panels |
| `clipboard.py` | `read_clipboard_image()` — the clipboard's image as PNG bytes, or `None` |
| `cells.py` / `screens.py` / `prompt.py` / `app.py` | Transcript widgets (incl. `SubagentPanel`, the one container), the modals (`ApprovalScreen`, `PickerScreen`), `PromptInput` (the multiline prompt box), `AgentApp` (drive worker + one event handler for both streaming and block tiers) |
| `context_bar.py` | The context-utilization gauge under the transcript; `render_context_bar` is the pure formatter |
| `commands.py` | Slash command registry + `dispatch` (called from `on_prompt_input_submitted` before the message is sent) |
| `config.py` | `LucaConfig` + `load_luca_config` (home+project `luca.json` merge, or one file named by `resolve_config_path` from `--config` / `LUCA_CONFIG_PATH`) and the precedence resolvers, incl. `build_context_manager` — see [`config.md`](config.md) |
| `cli.py` | argparse entry point; the `--pretty-print` transcript path (never builds an app), and loads `luca.json`, threading it (incl. the `SummarizingContextManager`) through the seams |

Attach an image with `Ctrl+V`, then type and press Enter — the image leads the
message. The status bar shows how many are attached, and `Esc` clears them
while nothing is running. A message can be image-only.

> ⚠️ **The clipboard is read directly, not pasted.** A terminal only ever
> transmits text, so image bytes never arrive as a key event. `Ctrl+V` shells
> out instead: `osascript` on macOS, `wl-paste` or `xclip` on Linux,
> PowerShell on Windows. Where none of those exist you get a notice, and
> nothing is attached. This is also why paste cannot work over SSH — the
> clipboard it reads is the one on the machine running the TUI.

> ⚠️ **Textual cannot draw images.** The transcript shows a placeholder line,
> never the picture.

The drive worker is the REPL loop verbatim: answer the gate, then fall
*through* to a run — recording answers on the strategy never advances the
runner, so the approval branch is always followed by `runner.run()`.

## 5. Test with the faux client

`provider=` is the same zero-logic passthrough the runner exposes, so the app
is drivable headless with a scripted
[`FauxProvider`](../../../client/12-testing.md) and Textual's `run_test()`
Pilot — no network, no keys:

```python
provider = FauxProvider()
provider.set_responses([faux_assistant_message([faux_text("Hello!")])])
app = AgentApp(session, provider=provider, workspace=tmp_path, session_dir=tmp_path)

async with app.run_test() as pilot:
    app.query_one("#prompt", PromptInput).load_text("hi")
    await pilot.press("enter")
    ...
    assert [c.text for c in app.query(AssistantCell)] == ["Hello!"]
```

Cells expose plain state (`.text`, `.status`, `.result_text`, `.is_error`) so
tests assert on attributes, not rendered output. See
`tests/agent/contrib/tui/` for the full patterns: approval flows by digit
key, `faux_hang()` + Esc for cancellation, reload-and-replay for resume.

For subagents assert on the transcript TREE, not on `app.query(Cell)` — nesting
is the behavior, and a flat query passes just as happily with a subagent's
output spliced into the main column. Note also that `app.query` is scoped to
the TOP of the screen stack, so a test reading the transcript while the
approval modal is up must reach through `app.screen_stack[0]`.

> ⚠️ **The app owns the wiring.** `AgentApp` builds its runner via
> `build_runner` — inject behavior through `provider=`, `workspace=`,
> `mode=` ("ask" / "yolo" / "auto") and `context_manager=`, not by passing a
> runner.

Next: [`subagents/README.md`](../subagents/README.md).
