"""N1 probe — OpenAI Responses: undeclared tool names in history.

    uv run python specs/0010-native-tools-in-agent/poc_tests/openai_undeclared_history.py

The one empirical check spec 0010 rests on, OpenAI side. The verbatim
projection replays a foreign-provider call as an ordinary `function_call`
item whose `name` (the stored internal spec name, e.g.
`anthropic_text_editor_20250728`) is NOT in this request's `tools`. This
probe fabricates exactly that history — the T9 moment of the worked example:
a session that ran Anthropic-native turns, switched back to OpenAI — and
asks whether the API accepts it.

ACCEPTED  → verbatim works as designed; nothing changes.
REJECTED  → OpenAI falls back to narrating foreign history; the projection
            rule itself is unchanged.

No files are touched and no tools are executed — the history is fabricated
and the follow-up question forbids tool use.
"""

import sys

from dotenv import load_dotenv

from luca.client import completion
from luca.client.exceptions import ClientError
from luca.client.providers.openai import ApplyPatchTool, LocalShellTool
from luca.client.types import (
    AssistantMessage,
    TextBlock,
    Tool,
    ToolCall,
    ToolMessage,
    UserMessage,
)

load_dotenv()  # OPENAI_API_KEY

MODEL = "openai:gpt-5.1"

# The history a session's Anthropic-native turns carry, projected the way
# the agent's verbatim rule would: plain ToolCalls under the stored INTERNAL
# names, results as flat text.
messages = [
    UserMessage(content=[TextBlock(text="Add a docstring to fibonacci in main.py and re-run the tests.")]),
    AssistantMessage(
        content=[
            TextBlock(text="I'll insert the docstring, then run the tests."),
            ToolCall(
                id="call_verbatim_1",
                name="anthropic_text_editor_20250728",
                arguments={
                    "command": "insert",
                    "path": "main.py",
                    "insert_line": 1,
                    "insert_text": '    """Return the nth Fibonacci number."""\n',
                },
            ),
            ToolCall(
                id="call_verbatim_2",
                name="anthropic_bash_20250124",
                arguments={"command": "pytest -q"},
            ),
        ],
    ),
    ToolMessage(
        tool_call_id="call_verbatim_1",
        name="anthropic_text_editor_20250728",
        content=[TextBlock(text="Inserted 1 line at main.py:1")],
    ),
    ToolMessage(
        tool_call_id="call_verbatim_2",
        name="anthropic_bash_20250124",
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

# This request's tools — the T9 advertised set after switching back: both
# OpenAI natives plus read as a plain function. Neither
# `anthropic_text_editor_20250728` nor `anthropic_bash_20250124` is declared.
tools = [
    ApplyPatchTool(),
    LocalShellTool(),
    Tool(
        name="read",
        description="Read one file from the workspace.",
        parameters={
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"],
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
