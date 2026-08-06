"""Anthropic native tools, end to end, streaming.

    uv run python specs/0009-provider-native-tools/examples/anthropic_example_streaming.py

Same task and same handlers as `anthropic_example_non_streaming.py`, driven
through `completion_stream`. What the streaming variant shows: on this wire
native calls stream like any other tool call — `tool_call_start`, full JSON
`tool_call_delta` fragments, then `tool_call_end` with the arguments parsed.
Executing them does not change at all.

The turn's result comes off the terminal `finish` event: `.message` is what
you append to history, `.tool_calls` is what you execute.

WARNING: this executes model-written shell commands on your machine. They
run with cwd set to a throwaway workspace, but nothing sandboxes them.
"""

import subprocess
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from luca.client import completion_stream
from luca.client.providers.anthropic import BashTool, TextEditorTool
from luca.client.types import TextBlock, ToolCall, ToolMessage, UserMessage

load_dotenv()  # ANTHROPIC_API_KEY

MODEL = "anthropic:claude-sonnet-4-5"
MAX_TURNS = 16

# --- the workspace the model will work in ----------------------------------

workspace = Path(tempfile.mkdtemp(prefix="luca-anthropic-native-")).resolve()
(workspace / "notes.txt").write_text("alpha\nbeta\ngamma\n")
(workspace / "scratch.txt").write_text("throwaway\n")

SYSTEM = (
    "You are a coding agent working inside a single workspace directory.\n"
    f"The workspace is {workspace}. Bash commands already run with that "
    "directory as the working directory.\n"
    "Use str_replace_based_edit_tool for anything touching file CONTENT "
    "(view to read, create to write a new file, str_replace to change text "
    "in place, insert to add a line). Use bash only for shell work: "
    "counting, deleting, listing.\n"
    "Take one step at a time and check your work as you go."
)

TASK = (
    "1. Read notes.txt in full, then re-read just lines 2 to 3 of it, and tell me what is in it.\n"
    "2. Replace the line 'beta' with 'BETA'.\n"
    "3. Add a new line 'delta' after 'gamma' — use the editor's insert command for this one.\n"
    "4. Count the lines of notes.txt with the shell, then write a new file "
    "summary.txt whose only line is 'lines=<N>'.\n"
    "5. Delete scratch.txt.\n"
    "6. List the workspace to confirm, then summarise what you did in one line."
)

# --- the handlers: this is where a native tool is actually implemented ------
#
# Native calls arrive as ORDINARY ToolCalls on this wire — the wire name
# points at the tool, the payload is `arguments` — and are answered with
# ordinary ToolMessages. Raising is fine: the loop turns any exception into
# an is_error result the model can recover from.


def handle_text_editor(tc: ToolCall) -> ToolMessage:
    """The text_editor_20250728 command set, against real files."""
    args = tc.arguments
    path = (workspace / args["path"]).resolve()  # an absolute path replaces the left side
    path.relative_to(workspace)  # refuse to touch anything outside the workspace

    match args["command"]:
        case "view" if path.is_dir():
            detail = "\n".join(sorted(p.name for p in path.iterdir()))
        case "view":
            lines = path.read_text().splitlines()
            start, end = args.get("view_range") or [1, len(lines)]
            end = len(lines) if end == -1 else end
            # `cat -n` style: these numbers are what view_range and
            # insert_line are counted in.
            detail = "\n".join(f"{n:6}\t{lines[n - 1]}" for n in range(start, end + 1))
        case "create":
            path.write_text(args["file_text"])
            detail = f"created {path.name}"
        case "str_replace":
            content = path.read_text()
            if content.count(args["old_str"]) != 1:
                raise ValueError(f"old_str must match exactly once in {path.name}")
            path.write_text(content.replace(args["old_str"], args["new_str"]))
            detail = f"edited {path.name}"
        case "insert":
            lines = path.read_text().splitlines(keepends=True)
            # text_editor_20250728 names this `insert_text`, NOT `new_str`.
            lines.insert(args["insert_line"], args["insert_text"].rstrip("\n") + "\n")
            path.write_text("".join(lines))
            detail = f"inserted after line {args['insert_line']} of {path.name}"
        case other:
            raise ValueError(f"unknown text editor command {other!r}")

    return ToolMessage(tool_call_id=tc.id, name=tc.name, content=[TextBlock(text=detail)])


def handle_bash(tc: ToolCall) -> ToolMessage:
    """One real subprocess per command. A production harness owns ONE
    persistent session instead, and tears it down on {"restart": true}."""
    if tc.arguments.get("restart"):
        return ToolMessage(tool_call_id=tc.id, name=tc.name, content=[TextBlock(text="shell session restarted")])

    proc = subprocess.run(
        tc.arguments["command"],
        shell=True,
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return ToolMessage(
        tool_call_id=tc.id,
        name=tc.name,
        content=[TextBlock(text=proc.stdout + proc.stderr or "(no output)")],
        is_error=proc.returncode != 0,
    )


HANDLERS = {TextEditorTool.name: handle_text_editor, BashTool.name: handle_bash}

# --- the agent loop --------------------------------------------------------

print(f"workspace: {workspace}\n")

messages = [UserMessage(content=[TextBlock(text=TASK)])]
tools = [TextEditorTool(max_characters=10_000), BashTool()]

for turn in range(1, MAX_TURNS + 1):
    finish = None

    with completion_stream(MODEL, messages, system_message=SYSTEM, tools=tools, max_tokens=2048) as stream:
        for event in stream:
            if event.type == "text_delta":
                print(event.delta, end="", flush=True)
            elif event.type == "text_end":
                print()
            elif event.type == "tool_call_start":
                print(f"[call] {event.name} ", end="", flush=True)
            elif event.type == "tool_call_delta":
                # Anthropic native calls stream FULL JSON argument fragments,
                # exactly like a function tool. (OpenAI's do not.)
                print(event.arguments_delta, end="", flush=True)
            elif event.type == "tool_call_end":
                print()
            elif event.type == "finish":
                finish = event
            elif event.type == "error":
                raise event.error

    messages.append(finish.message)  # replay bookkeeping ends here

    if finish.finish_reason != "tool_use":
        print(f"[done] finish_reason={finish.finish_reason} turns={turn}")
        break

    for tc in finish.tool_calls:
        try:
            result = HANDLERS[tc.name](tc)
        except Exception as exc:
            # Honest failures are the caller's job — say what actually broke.
            text = f"{type(exc).__name__}: {exc}"
            result = ToolMessage(tool_call_id=tc.id, name=tc.name, content=[TextBlock(text=text)], is_error=True)
        print(f"[result] is_error={result.is_error} {result.content[0].text!r}\n")
        messages.append(result)
else:
    print(f"[stopped] hit MAX_TURNS={MAX_TURNS}")

# --- what actually happened on disk ----------------------------------------

print("\n--- final workspace ---")
for path in sorted(workspace.iterdir()):
    print(f"\n{path.name}:\n{path.read_text()}", end="")
