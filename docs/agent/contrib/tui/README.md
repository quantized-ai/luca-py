# TUI

A [Textual](https://textual.textualize.io/) terminal UI for the agent loop,
built as a **design system**: a full-screen app with one scrolling
conversation column, an inline approval prompt, overlay lists (command
palette, context picker), and modal screens for sessions, settings and cost.
Every visual state is expressible as declarative data (`state.ScreenState`),
renderable without a live agent (the gallery), and snapshot-tested. It is
also the reference for wiring a real interactive app on top of the runner:
one drive worker, one shared
[`PermissionStrategy`](../resource_permissions/README.md), sessions saved as
`~/.luca/projects/<encoded-project-path>/<session-id>.json`. Requires the
`tui` dependency group (installed by default with `uv sync`).

```python
from luca.agent.contrib.tui import AgentApp, GalleryApp, LucaApp, build_runner, main
```

## 1. Run it

```bash
uv run python main.py                     # fresh session (needs OPENROUTER_API_KEY)
uv run python main.py --faux              # offline scripted demo — no key, no network
uv run python main.py --resume            # pick a past session for this project
uv run python main.py --conversation <id> # resume that session (--fork to branch)
uv run python main.py --gallery           # the design-system gallery — no agent at all
uv run python main.py --no-streaming      # block-level events instead of deltas
uv run python main.py --no-subagents      # stop it spawning subagents that work in parallel
uv run python main.py --model moonshotai/kimi-k2.7-code --reasoning high
uv run python main.py --config ./ci.json  # use THIS config file, skip discovery
uv run python main.py --conversation <id> --pretty-print  # transcript, then exit
```

`--model` / `--provider` / `--reasoning` update the session's `LLMConfig`;
they persist with the session and override the stored values on a resume.
Subagent flags (`--subagents-max-depth`, `--subagents-max-per-turn`,
`--subagents-max-workers`) and `--pretty-print` behave as before — see
[`08-runtime-config.md`](../../08-runtime-config.md) and
[`config.md`](config.md) for the `luca.json` equivalents.

> ⚠️ **Old session files do not load.** A `<id>.json` written before tool-spec
> normalization fails validation on `--conversation` and on `--pretty-print`.
> There is no migration — start a fresh session.

## 2. What's on screen

One `Vertical`: a custom status bar (`◧ luca · cwd · model · branch* ·
tokens · $cost`), the single rule, the scrolling transcript, the dock
(composer OR approval prompt OR overlay list), and the hint legend — which is
always visible and always reflects the focused context. No stock
Header/Footer, no panel borders; blocks are separated by blank rows.

Transcript block vocabulary (`blocks.py`, rendered from `state.py` models):

| Block | Rendering |
|---|---|
| user | `›` in accent, then the message (images as `[image: name]` lines) |
| thinking | `∴ thought for 3s` collapsed; `∴ <activity>` while in progress |
| text | assistant prose at an 84-column measure; `` `code spans` `` in accent; streaming ends with a block cursor |
| tool | `▸ name arg` header (6-cell name column), then the result in a `1+│+2` gutter: a faint summary (`84 lines`, `exit 0 · removed 12 paths`), a diff stat (`+7 −1`), or verbatim output with `^o expand`. Denied calls turn `▸` and `denied` to `$error`; **every automatic permission decision is stated** (`approved by rule`, `denied · by rule`) |
| list | plan / active skills / context set / consumers: a faint label row + glyph rows (`☑ ◉ ☐ ✓`) |
| diff | unified diff in the gutter: `[5 line-no][3 sign][code]`, added/removed row backgrounds spanning the block |
| notice | one faint (or `$error`) row: cancellations, turn failures |
| task | a subagent: `task · <description> · <status>` label, the child's own blocks nested in a gutter (see §2.1) |

Interactions: the composer stays enabled while the agent works (mid-turn
posts queue into the open turn; the placeholder flips to `working…` and the
legend to `enter queue`); `esc` interrupts a run, backs out of a modal, and
selects Cancel turn at an approval prompt — it never quits (`ctrl+q` does,
saving first). `^p` palette, `^t` jump to the plan, `^s` skills, `^o` expand
the last clipped output, `ctrl+v` attach a clipboard image. A drive failure
renders as a `▸ model` error block plus a recovery prompt (retry / switch
model / cancel turn; `^r` retries).

### 2.1 How subagents are drawn

Several subagents' events arrive interleaved on one stream, so attribution
cannot be positional: every event carries `conversation_id`, and both the
live-widget state and the mount target are keyed by it. Each child renders as
a `task` block — its label carries the spawn's description and its status
(`waiting / running / done / failed`), and everything the child produces
mounts inside its gutter. The spawn tool renders as the task block itself and
the private result tool renders as nothing (both matched by DECLARATION —
`is_private`, `is_subagent_spawn` — never by name). A task settles off the
resolved `ChildConversation`, not the result tool's event, so a cancelled
subagent stops saying `running`. Resume replays the same tree.

## 3. Slash commands

Type `/` in the empty composer (or `^p`) for the palette; typing filters, and
a full `/name arg` line submits directly while idle. Unknown commands are
sent to the agent as a normal message, so `/etc/hosts` is never swallowed.

| Command | Effect |
|---|---|
| `/skill [name]` | Sends `/skill <name>` to the agent (the model loads it with the skill tool); the palette inserts `/skill ` for the argument |
| `/session`, `/resume` | The sessions screen: `↑↓` move, `enter` resume, `f` fork, `d` (twice) delete, with a last-turn preview per row |
| `/context` | The `@` context picker (also opened by typing `@` inline) — mock: the committed set renders as a context list block, display-only for now |
| `/cost` | The cost screen: estimated spend per category with cost-proportional meters, the context window bar, biggest consumers; `^k` compacts |
| `/settings` | The settings screen: `← →` adjusts model / reasoning / streaming / theme / counter live; `esc` saves and closes |
| `/clear`, `/new` | Save, then start a fresh session carrying the runtime config |
| `/model [provider:model]` | No arg: provider then model, in the overlay menu. Takes effect next turn |
| `/reasoning [level]` | No arg opens the level menu |
| `/theme` | Pick any registered theme (`luca-dark` ships; see `theme.py`) |
| `/compact` | Summarize the history and continue ([12](../../12-compaction.md)) |
| `/help` | Every command, as a list block |
| `/quit` | Save and exit |

Dollar figures anywhere (status counter, sessions, cost screen) are estimates
from the small `usage.PRICING` table and are omitted for unlisted models.

## 4. The design system

The design source of truth is the handoff in `design_handoff_luca_tui/` (the
eleven screens `1a`–`1k`). The system that implements it:

- **`theme.py`** — the palette as a registered `Theme`; the only hex source.
  The contrast floor is `$text-faint` (#7E7E7E) — nothing functional dimmer.
- **`app.tcss`** — every layout, spacing, border and color assignment, tokens
  only, commented by the screen ids each section serves.
- **`state.py`** — the view-model: `ScreenState` (status, transcript blocks,
  dock, modal, hints). Fixtures, the live app and the replay all produce it.
- **`gallery.py` + `fixtures/`** — the component catalog:

```bash
uv run python main.py --gallery                    # browse every state
uv run python main.py --gallery 1c_diff_approval   # one handoff screen
uv run python main.py --gallery my_state.yaml      # your own fixture
```

A fixture is YAML validated as `ScreenState` — so "what does the approval
prompt look like over a 200-line diff?" is a file, not a live-model hunt:

```yaml
name: my_state
status: {cwd: ~/quantized/luca, model: sonnet-4.5, branch: main, tokens: 12.4k}
transcript:
  - {kind: user, text: run the tests}
  - {kind: tool, tool: bash, arg: pytest -q, result: {summary: exit 0 · 1.2s}}
composer: {placeholder: "ask, or / for commands"}
hints: [enter send, ⇥ complete, ^p palette]
```

Every bundled fixture has a committed SVG snapshot
(`tests/agent/contrib/tui/test_snapshots.py`); `pytest --snapshot-update`
regenerates them after an intentional visual change — review the SVG diff
like code. Add a fixture for any new component or state.

> ⚠️ **YAML gotchas.** Quote strings containing commas inside `{…}` flow
> mappings, and diff line numbers use the key `num` — bare `no` is a YAML
> boolean.

## 5. Structure

| Module | Role |
|---|---|
| `theme.py` / `app.tcss` / `state.py` / `format.py` | The design system's contract: palette, geometry, view-model, pure text helpers (token spans, humanized figures, hint legends) |
| `render.py` | Pure derivations: entries/events → view-models (`tool_block`, `plan_block`, `preview_rows`, `subagent_task`, …) — live and replay share them, so they cannot drift |
| `blocks.py` / `chrome.py` / `shells.py` / `modals.py` | The widgets: transcript blocks, status bar + legend + composer, the shared selection treatment + approval prompt + overlay list + modal base, the three modal screens |
| `frame.py` | `LucaApp` — the frame; `apply_state(ScreenState)` renders any state |
| `gallery.py` | Fixture loading + `GalleryApp` (`--gallery`) |
| `app.py` | `AgentApp(LucaApp)` — the drive worker and one event handler for both streaming and block tiers |
| `wiring.py` | `build_runner(...)` — shell + memory plugins, demo math tools, one shared strategy; `build_faux_provider()` scripts the `--faux` conversation |
| `approvals.py` | Pending permission steps → the fixed 4-option prompt model (`Approve once / Approve always — <scope> / Deny / Cancel turn`), no UI |
| `usage.py` / `gitinfo.py` / `files.py` | Status counter + cost-screen state; branch/dirty; `@`-picker file listing |
| `sessions.py` / `commands.py` / `config.py` / `cli.py` / `prompt.py` / `clipboard.py` | The store (atomic save, summaries with previews), the command registry + live modal-state builders, `luca.json`, argparse, the composer's TextArea, clipboard image read |

## 6. Test with the faux client

```python
provider = FauxProvider()
provider.set_responses([faux_assistant_message([faux_text("Hello!")])])
app = AgentApp(session, provider=provider, workspace=tmp_path,
               session_dir=tmp_path, skills=False, instructions=False)

async with app.run_test(size=(105, 35)) as pilot:
    app.query_one(PromptInput).load_text("hi")
    await pilot.press("enter")
    ...
    assert [t.text for t in app.query(AssistantText)] == ["Hello!"]
```

Widgets keep their view-models on `.model`, so tests assert on data, not on
rendered output. Approval flows are inline (`ApprovalPromptView` in the dock
— press a digit), cancellation is `faux_hang()` + `esc`, and subagent tests
assert on the transcript TREE (blocks inside `TaskBlockView.body`), never a
flat query. See `tests/agent/contrib/tui/` for the full patterns and
`test_snapshots.py` for the fixture snapshots.

> ⚠️ **The app owns the wiring.** `AgentApp` builds its runner via
> `build_runner` — inject behavior through `provider=`, `workspace=`,
> `mode=` ("ask" / "yolo" / "auto") and `context_manager=`, not by passing a
> runner.

Next: [`subagents/README.md`](../subagents/README.md).
