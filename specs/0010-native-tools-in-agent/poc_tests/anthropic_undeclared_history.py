"""N1 probe — Anthropic: undeclared tool names in history.

    uv run python specs/0010-native-tools-in-agent/poc_tests/anthropic_undeclared_history.py

The one empirical check spec 0010 rests on, Anthropic side. The verbatim
projection replays a foreign-provider call as an ordinary `tool_use` block
whose `name` (the stored internal spec name, e.g. `openai_apply_patch`) is
NOT in this request's `tools`. This probe fabricates exactly that history —
the T5 moment of the worked example: a session born on OpenAI-native tools,
switched to Anthropic-native — and asks whether the API accepts it.

ACCEPTED  → verbatim works as designed; nothing changes.
REJECTED  → Anthropic falls back to narrating foreign history; the
            projection rule itself is unchanged.

No files are touched and no tools are executed — the history is fabricated
and the follow-up question forbids tool use.
"""

import sys

from dotenv import load_dotenv

from luca.client import completion
from luca.client.exceptions import ClientError
from luca.client.providers.anthropic import BashTool, TextEditorTool
from luca.client.types import (
    AssistantMessage,
    TextBlock,
    Tool,
    ToolCall,
    ToolMessage,
    UserMessage,
)

load_dotenv()  # ANTHROPIC_API_KEY

MODEL = "anthropic:claude-sonnet-4-5"

DIFF = (
    "@@\n"
    "-def fib(n):\n"
    "+def fibonacci(n):\n"
    "     if n <= 1:\n"
    "         return n\n"
    "@@\n"
    "-    return fib(n-1) + fib(n-2)\n"
    "+    return fibonacci(n-1) + fibonacci(n-2)\n"
)

# The history a session born on OpenAI-native tools carries, projected the
# way the agent's verbatim rule would: plain ToolCalls under the stored
# INTERNAL names, results as flat text, extras never emitted.
messages = [
    UserMessage(content=[TextBlock(text="Rename fib to fibonacci in main.py and run the tests.")]),
    AssistantMessage(
        content=[
            TextBlock(text="I'll update the file, then run the tests."),
            ToolCall(
                id="toolu_verbatim_1",
                name="openai_apply_patch",
                arguments={"type": "update_file", "path": "main.py", "diff": DIFF},
            ),
            ToolCall(
                id="toolu_verbatim_2",
                name="openai_shell",
                arguments={"commands": ["pytest -q"], "timeout_ms": 60000},
            ),
        ],
    ),
    ToolMessage(
        tool_call_id="toolu_verbatim_1",
        name="openai_apply_patch",
        content=[TextBlock(text="Updated main.py (2 hunks)")],
    ),
    ToolMessage(
        tool_call_id="toolu_verbatim_2",
        name="openai_shell",
        content=[TextBlock(text="2 passed")],
    ),
    UserMessage(
        content=[
            TextBlock(
                text=(
                    "What did you change and how do you know the tests pass? Answer in one line. Do not call any tools."
                ),
            ),
        ],
    ),
]

# This request's tools — the T4/T5 advertised set after the switch: both
# Anthropic natives plus delete_file as a plain function. Neither
# `openai_apply_patch` nor `openai_shell` is declared.
tools = [
    TextEditorTool(),
    BashTool(),
    Tool(
        name="delete_file",
        description="Delete one file from the workspace.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    ),
]

try:
    response = completion(MODEL, messages, tools=tools)
except ClientError as exc:
    print(f"REJECTED  {type(exc).__name__}: {exc}")
    sys.exit(1)

print(f"ACCEPTED  finish_reason={response.finish_reason}")
for block in response.message.content:
    if isinstance(block, TextBlock):
        print(f"[assistant] {block.text.strip()}")
for tc in response.tool_calls:
    # A call here would be the T7 scenario — the model copying an undeclared
    # name out of history. Worth knowing, not a failure of this probe.
    print(f"[unexpected tool call] {tc.name} {tc.arguments}")
