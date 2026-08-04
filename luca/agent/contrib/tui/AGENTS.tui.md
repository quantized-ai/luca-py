Guidance for the Textual TUI. Read `AGENTS.agent.md` first for the rules shared by the whole `luca.agent` layer.

## What this package is

`luca/agent/contrib/tui/` is the Textual terminal UI, built as a design system: every visual state is expressible as declarative data, renderable without a live agent, and snapshot-tested. The visual design is specified by the handoff in `design_handoff_luca_tui/` at the repo root (eleven screens, `1a`–`1k`); treat that README as the design source of truth.

## The design-system layers

```
theme.py      the palette as a registered Textual Theme — THE ONLY HEX SOURCE
app.tcss      every layout, spacing, border and color assignment ($tokens only)
state.py      the view-model: ScreenState + the block vocabulary (pure Pydantic)
format.py     pure text helpers: token spans, humanized figures, hint legends
render.py     pure derivations: agent entries/events → view-models
blocks.py     the transcript block widgets (user, thinking, text, tool, list, diff, notice, task)
chrome.py     StatusBar, HintLegend, Composer
shells.py     SelectRow (THE selection treatment, defined once), ApprovalPromptView,
              OverlayListView (palette/picker/menu), LucaModalScreen
modals.py     SessionsScreen, SettingsScreen, CostScreen
frame.py      LucaApp — the global frame; apply_state(ScreenState) renders any state
gallery.py    fixture loading + GalleryApp (`--gallery`), the component catalog
fixtures/     the declarative screen states: 1a–1k + components/ sheets
app.py        AgentApp(LucaApp) — the live agent wiring (drive worker, events)
```

Supporting live modules: `usage.py` (token totals, estimated cost, the 1k cost state), `gitinfo.py` (branch/dirty), `files.py` (@-picker file listing), `prompt_files/` (`@`-mention expansion: `parse_prompt` → content parts, via a first-match-wins handler chain — add a format by adding a handler above `BinaryHandler`, nothing else changes), `sessions.py` (session store + summaries), `commands.py` (the 14 slash commands + live modal-state builders), `approvals.py` (the 4-option prompt model), `config.py` (`luca.json`), `cli.py`, `prompt.py` (the composer's TextArea).

## Hard rules (from the design handoff)

- **No hex literal outside `theme.py`. No geometry outside `app.tcss`.** No `styles.x = ...` in Python for anything a stylesheet can express. Widgets that need concrete colors resolve them through `theme.resolve_tokens(app)`.
- **Contrast floor `$text-faint` (#7E7E7E).** Nothing functional dimmer; if something seems to need a dimmer grey, remove the element.
- **No animation** — no spinners, progress bars, or transitions. The only moving thing is the terminal's cursor.
- **Selection never reflows** — `SelectRow` always allocates the bar and caret columns; the treatment exists once (`shells.SelectRow` + `.select-row` in TCSS).
- **Plain keys** — arrows, tab, enter, esc, digits, documented `^` pairs. The hint legend always reflects the focused context.
- **Every automatic permission decision is stated in the transcript** (`approved by rule`, `denied · by rule`).
- Don't invent UI the handoff doesn't specify. The two sanctioned extensions are `NoticeBlock` (turn-level notices) and `TaskBlock` (subagent conversations, built from the handoff's gutter idiom).
- `@` file mentions render as ordinary `ToolBlock`s (`tool: read`) under the user turn — same idiom, no new block kind. They are NOT tool calls: no `ToolExecution` backs them, so they never carry an approval note. A declined mention (too long, binary, a directory) is `status: error` with a `[error]×[/]` summary; a path that does not resolve gets no row at all and stays prose, which is what keeps `@property` and `@types/node` from being read as files. See `fixtures/components/mentions.yaml`, the requirements sheet for the feature.

## The gallery is the catalog

```bash
uv run python -m luca.agent.contrib.tui --gallery                # browse everything
uv run python -m luca.agent.contrib.tui --gallery 1a_agent_loop  # one screen
```

A fixture is a YAML file validated as `state.ScreenState`. The bundled set covers the eleven handoff screens plus `components/` sheets showing every block and shell variant. **Any new component or state gets a fixture**, and every fixture gets a snapshot test (`tests/agent/contrib/tui/test_snapshots.py`, `pytest --snapshot-update` regenerates the committed SVGs). To prototype a state ("what does 90% context look like?"), write a fixture — never drive a live model to reproduce a screen.

YAML gotchas: quote strings containing commas inside `{...}` flow mappings, and diff line numbers use the key `num` (`no` is a YAML boolean).

## Agent integration details

- The core default for `subagents_max_depth` is 1; the TUI opts into 3.
- `/model` can switch models mid-session; the whole history replays to the new model (reasoning-attestation provenance rules apply).
- Sessions live in `~/.luca/projects/<encoded-project-path>/<id>.json`, keyed on the WORKSPACE. `resolve_session_directory` in `sessions.py` computes it.
- `_reset_session` is the "switch to this session" primitive behind `/clear`, `/session` resume and fork: rebuild the runner, wipe the transcript, replay.
- The sessions screen parses every stored session to build its rows (~3ms per 500KB session); no index file to keep in sync.
- The `@` picker commits by writing `@path` mentions into the composer (`format.inline_paths`) — it does not attach file contents, so the paths reach the model as text and it reads them with its own tools.
- Mock-only surfaces (until the agent grows the feature): the settings screen's `approval mode` row is read-only; dollar figures are estimates from `usage.PRICING` and are omitted for unlisted models.

## Testing

Tests live in `tests/agent/contrib/tui/`. Pure-module coverage: `test_state.py`, `test_format.py`, `test_render.py`, `test_approvals.py`, `test_sessions.py`, `test_config.py`, `test_wiring.py`, `test_cli.py`, `test_commands.py`, `test_gallery.py`. Headless Pilot tests drive `AgentApp` with a scripted `FauxProvider` in `test_app*.py` and `test_subagents.py`. `test_snapshots.py` renders every bundled fixture at 105×35 and compares committed SVGs. The directory skips itself when Textual is missing.
