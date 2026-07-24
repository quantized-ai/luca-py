# TUI

A [Textual](https://textual.textualize.io/) terminal UI for the agent loop —
the interactive counterpart of the classic REPL: a chat transcript, live
streaming, a modal approval gate, Esc cancellation, and per-run session
persistence. It is also the reference for wiring a real interactive app on
top of the runner: one drive worker, one shared
[`PermissionStrategy`](../resource_permissions/README.md), sessions saved as
`<session-id>.json`. Requires the `tui` dependency group (installed by
default with `uv sync`).

```python
from luca.agent.contrib.tui import AgentApp, build_runner, main
```

## 1. Run it

```bash
luca                                      # installed command — TUI in the current dir
luca --faux                               # offline scripted demo — no key, no network
uv run python main.py                     # same thing from a checkout (needs OPENROUTER_API_KEY)
uv run python main.py --conversation <id> # resume <id>.json (--fork to branch)
uv run python main.py --no-streaming      # block-level events instead of deltas
uv run python main.py --model moonshotai/kimi-k2.7-code --reasoning high
```

`--model` / `--reasoning` update the session's `LLMConfig` (provider
stays openrouter); they persist with the session and override the stored
values on a resume.

`main.py`, the `luca` command, and `python -m luca` all delegate to
`luca.cli:main` (load `.env`, then launch the TUI). Install the command with
`uv tool install ".[tui]"` — see the top-level README.

## 2. What's on screen

| Piece | Behavior |
|---|---|
| Transcript cells | One bordered cell per block: `you`, `assistant`, `thinking`, `tool` (call → running → result, clipped), `notice` (cancels, failures) |
| Input box | Enabled while the runner is `IDLE`; Enter posts the message and starts the drive worker |
| `Ctrl+V` | Attaches the clipboard's image to the next message; the transcript shows `[image: pasted-1.png]` |
| Approval modal | One screen per uncovered permission step: Approve once / tool-suggested ALWAYS grants / Deny / Abandon — pick by button or digit key |
| `Esc` | Cancels the live run (`run.cancel()`); the wind-down renders live and the turn closes `CANCELLED` |
| `Ctrl+D` | Saves the session and quits |

## 3. Structure

The Textual-free modules hold everything worth unit-testing; the widgets stay
thin:

| Module | Role |
|---|---|
| `wiring.py` | `build_runner(session, workspace=, provider=, mode=)` — shell + memory plugins, the demo math tools, one shared strategy; `build_faux_provider()` scripts the `--faux` conversation |
| `approvals.py` | `build_approval_prompts(execution, strategy)` — pending steps → `ApprovalPrompt`s whose options carry fully-built `ApprovalAnswer`s (the whole gate policy, no UI) |
| `sessions.py` | `<session-id>.json` load / save / fork |
| `render.py` | Pure formatting: `format_tool_call`, `clip_text`, `status_label`, `user_transcript_text` (the live and replayed transcript share it, so they cannot drift) |
| `clipboard.py` | `read_clipboard_image()` — the clipboard's image as PNG bytes, or `None` |
| `cells.py` / `screens.py` / `app.py` | Transcript widgets, the modal, `AgentApp` (drive worker + one event handler for both streaming and block tiers) |
| `cli.py` | argparse entry point |

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

## 4. Test with the faux client

`provider=` is the same zero-logic passthrough the runner exposes, so the app
is drivable headless with a scripted
[`FauxProvider`](../../../client/12-testing.md) and Textual's `run_test()`
Pilot — no network, no keys:

```python
provider = FauxProvider()
provider.set_responses([faux_assistant_message([faux_text("Hello!")])])
app = AgentApp(session, provider=provider, workspace=tmp_path, session_dir=tmp_path)

async with app.run_test() as pilot:
    app.query_one("#prompt", Input).value = "hi"
    await pilot.press("enter")
    ...
    assert [c.text for c in app.query(AssistantCell)] == ["Hello!"]
```

Cells expose plain state (`.text`, `.status`, `.result_text`, `.is_error`) so
tests assert on attributes, not rendered output. See
`tests/agent/contrib/tui/` for the full patterns: approval flows by digit
key, `faux_hang()` + Esc for cancellation, reload-and-replay for resume.

> ⚠️ **The app owns the wiring.** `AgentApp` builds its runner via
> `build_runner` — inject behavior through `provider=`, `workspace=`, and
> `mode=` ("ask" / "yolo"), not by passing a runner.

Next: back to the [contrib index](../README.md).
