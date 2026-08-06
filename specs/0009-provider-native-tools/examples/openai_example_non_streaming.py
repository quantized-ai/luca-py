"""OpenAI native tools, end to end, no streaming.

    uv run python specs/0009-provider-native-tools/examples/openai_example_non_streaming.py

Declares `ApplyPatchTool()` and `LocalShellTool()` and runs a real agent loop
against gpt-5.1 on the Responses wire. The client only DECLARES the tools and
surfaces the calls; the handlers below EXECUTE them: every apply_patch
operation (create_file / update_file / delete_file) against real files, every
shell command in a real subprocess. The task is written so all three
operations fire — apply_patch has no read command, so reading is the shell's
job.

WARNING: this executes model-written shell commands on your machine. They
run with cwd set to a throwaway workspace, but nothing sandboxes them.
"""

import subprocess
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from luca.client import completion
from luca.client.providers.openai import (
    ApplyPatchTool,
    ApplyPatchToolCall,
    LocalShellTool,
    ShellCommandResult,
    ShellExitOutcome,
    ShellTimeoutOutcome,
    ShellToolCall,
    ShellToolMessage,
)
from luca.client.types import TextBlock, ToolMessage, UserMessage

load_dotenv()  # OPENAI_API_KEY

MODEL = "openai:gpt-5.1"
MAX_TURNS = 16

# --- the workspace the model will work in ----------------------------------

workspace = Path(tempfile.mkdtemp(prefix="luca-openai-native-")).resolve()
(workspace / "notes.txt").write_text("alpha\nbeta\ngamma\n")
(workspace / "scratch.txt").write_text("throwaway\n")

SYSTEM = (
    "You are a coding agent working inside a single workspace directory.\n"
    f"The workspace is {workspace}. Shell commands already run with that "
    "directory as the working directory, and apply_patch paths are resolved "
    "against it.\n"
    "Use apply_patch for every file creation, modification and deletion. Use "
    "shell for reading files and inspecting the directory.\n"
    "Take one step at a time and check your work as you go."
)

TASK = (
    "1. Read notes.txt and tell me what is in it.\n"
    "2. Replace the line 'beta' with 'BETA'.\n"
    "3. Add a new line 'delta' after 'gamma'.\n"
    "4. Count the lines of notes.txt with the shell, then write a new file "
    "summary.txt whose only line is 'lines=<N>'.\n"
    "5. Delete scratch.txt.\n"
    "6. List the workspace to confirm, then summarise what you did in one line."
)

# --- the handlers: this is where a native tool is actually implemented ------
#
# Native calls arrive as TYPED ToolCall subclasses on this wire, under names
# the client synthesizes ("apply_patch", "shell") because the wire items
# carry none. apply_patch answers with a plain ToolMessage — is_error IS the
# wire status — and shell with a structured ShellToolMessage.


def apply_v4a(original: str, diff: str) -> str:
    """Minimal V4A: "+" lines are emitted, " " context and "-" deletions are
    located by scanning forward from a cursor, "@@" anchors are implied by
    that scan. The real thing — fuzzy context matching, *** End of File,
    CRLF — is openai-agents-python's src/agents/apply_diff.py."""
    lines, merged, cursor = original.split("\n"), [], 0
    for raw in diff.splitlines():
        if raw.startswith(("@@", "***")):
            continue
        mark, text = raw[:1] or " ", raw[1:]
        if mark == "+":
            merged.append(text)
            continue
        while cursor < len(lines) and lines[cursor] != text:
            merged.append(lines[cursor])
            cursor += 1
        if cursor >= len(lines):
            raise ValueError(f"context line not found: {text!r}")
        if mark == " ":
            merged.append(text)
        cursor += 1
    return "\n".join([*merged, *lines[cursor:]])


def handle_apply_patch(tc: ApplyPatchToolCall) -> ToolMessage:
    """The three apply_patch operations, against real files."""
    operation = tc.operation  # typed accessor over .arguments
    path = (workspace / operation["path"]).resolve()  # an absolute path replaces the left side
    path.relative_to(workspace)  # refuse to touch anything outside the workspace

    match operation["type"]:
        case "create_file":
            # A create diff is the whole file, every line "+"-prefixed.
            path.write_text("".join(line[1:] + "\n" for line in operation["diff"].splitlines()))
            detail = f"created {path.name}"
        case "update_file":
            path.write_text(apply_v4a(path.read_text(), operation["diff"]))
            detail = f"updated {path.name}"
        case "delete_file":
            path.unlink()
            detail = f"deleted {path.name}"
        case other:
            raise ValueError(f"unknown apply_patch operation {other!r}")

    return ToolMessage(tool_call_id=tc.id, name=tc.name, content=[TextBlock(text=detail)])


def handle_shell(tc: ShellToolCall) -> ShellToolMessage:
    """One real subprocess per command, in order, one result each."""
    action = tc.action  # typed accessor over .arguments
    results = []
    for command in action["commands"]:
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=(action.get("timeout_ms") or 30_000) / 1000,
            )
            stdout, stderr, outcome = proc.stdout, proc.stderr, ShellExitOutcome(exit_code=proc.returncode)
        except subprocess.TimeoutExpired as expired:
            stdout, stderr, outcome = expired.stdout or "", expired.stderr or "", ShellTimeoutOutcome()
        results.append(ShellCommandResult(stdout=stdout, stderr=stderr, outcome=outcome))

    # Structured results are REQUIRED here: per-command stdout/stderr/exit
    # codes cannot ride prose, and a plain ToolMessage raises BadRequestError
    # at projection. `content` is an optional human-readable echo.
    echo = "; ".join(f"{r.outcome.type} {(r.stdout + r.stderr).strip()!r}" for r in results)
    return ShellToolMessage(tool_call_id=tc.id, name=tc.name, results=results, content=[TextBlock(text=echo)])


HANDLERS = {ApplyPatchTool.name: handle_apply_patch, LocalShellTool.name: handle_shell}

# --- the agent loop --------------------------------------------------------

print(f"workspace: {workspace}\n")

messages = [UserMessage(content=[TextBlock(text=TASK)])]
tools = [ApplyPatchTool(), LocalShellTool()]

for turn in range(1, MAX_TURNS + 1):
    response = completion(MODEL, messages, system_message=SYSTEM, tools=tools)
    messages.append(response.message)  # replay bookkeeping ends here

    for block in response.message.content:
        if isinstance(block, TextBlock) and block.text.strip():
            print(f"[assistant] {block.text.strip()}\n")

    if response.finish_reason != "tool_use":
        print(f"[done] finish_reason={response.finish_reason} turns={turn}")
        break

    for tc in response.tool_calls:
        print(f"[call] {type(tc).__name__} {tc.arguments}")
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
