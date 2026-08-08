"""A PARKED provider-native tool call on the wire (spec 0007, decision 14).

Context, from the top. A provider-native tool (OpenAI's `apply_patch` /
`shell`, Anthropic's `text_editor` / `bash`) is stored and projected by the
core exactly like any other function call — internal name, plain text result,
no provider knowledge anywhere. `ShellNativeMiddleware` upgrades that default
to the provider's own item shapes on the way out, and for `shell` results it
branches on `execution.result is None`: with a stored result it re-attaches the
structured extras captured at birth; without one — a rejected, failed or (since
0008) gated call — it SYNTHESIZES a `shell_call_output` from whatever text the
projector derived, because OpenAI's shell result has no free-text form.

A PARKED call (`AWAITING_RESULT`) has no `ExecutionResult` either, so it lands
in that same derived branch, and the only exit code the synthesizer can write
for a call with no outcome is 1. The model therefore sees "not finished yet" as
a FAILED COMMAND: exit 1, placeholder text in `stderr`.

THAT IS A KNOWING COMPROMISE, NOT A BUG. `shell_call_output` carries per-command
stdout / stderr / exit code and nothing else — there is no "pending" variant and
no honest third answer available in the wire format. It is unreachable in
practice: all four shipped native tools (`apply_patch`, `shell`, `text_editor`,
`bash`) are local and synchronous and cannot defer, so only somebody writing a
deferring native tool of their own can reach it. These two tests pin the
behaviour so the compromise is a recorded decision rather than a surprise, and
so a future honest encoding replaces a known value instead of an accident.

Everything else is unchanged: the generic `ToolMessage` still carries
`is_error=False` (an error result is exactly what makes a model retry), and
every non-shell native keeps the placeholder verbatim — plain text IS their
native result form.

The LLM boundary and the fixture tools are 0010's battery machinery
(`tests/agent/test_native_tools/`), with the two deferring natives swapped in.
"""

from __future__ import annotations

import pytest

import luca.agent.core.runner as runner_module
from luca.agent.contrib.plugins import PluginAgentSessionRunner
from luca.agent.contrib.shell import ShellNativeMiddleware
from luca.agent.contrib.simple_tool_registry import SimpleToolRegistry
from luca.agent.core import AgentSession, CancellationToken, ExecutionDeferred, ExecutionStatus
from luca.agent.core.projection import ConversationProjector
from luca.client.types import (
    AssistantMessage as ClientAssistantMessage,
    TextBlock,
    ToolCall as ClientToolCall,
    ToolMessage as ClientToolMessage,
    UserMessage as ClientUserMessage,
)
from tests.agent.test_native_tools.conftest import (
    MockLLM,
    assistant,
    make_session,
    plain_call,
    shell_call,
    text,
)
from tests.agent.test_native_tools.plugin import ControllablePermissionPolicy
from tests.agent.test_native_tools.tools import (
    ALL_TOOL_CLASSES,
    AnthropicBashTool,
    OpenAIShellTool,
)

AWAITING = ConversationProjector.AWAITING_RESULT_OUTPUT


# ── the two deferring natives ────────────────────────────────────────────────


class DeferringShellTool(OpenAIShellTool):
    """`openai_shell`, answering "not yet" forever. No shipped native can do
    this — that is the whole point of the compromise below."""

    async def execute(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
        *,
        tool_name: str,
        tool_call_id: str,
        cancellation_token: CancellationToken,
    ) -> ExecutionDeferred:
        return ExecutionDeferred()


class DeferringBashTool(AnthropicBashTool):
    """`anthropic_bash_20250124`, answering "not yet" forever — the control
    case: no structure is required of its result, so nothing is synthesized."""

    async def execute(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
        *,
        tool_name: str,
        tool_call_id: str,
        cancellation_token: CancellationToken,
    ) -> ExecutionDeferred:
        return ExecutionDeferred()


DEFERRING_TOOL_CLASSES = tuple(
    {OpenAIShellTool: DeferringShellTool, AnthropicBashTool: DeferringBashTool}.get(cls, cls)
    for cls in ALL_TOOL_CLASSES
)


class DeferringNativeToolsPlugin:
    """0010's battery plugin with the two deferring natives swapped in: the
    same eight fixture tools, the same REAL `ShellNativeMiddleware`."""

    def __init__(self) -> None:
        self.policy = ControllablePermissionPolicy()

    def get_tool_registry(self, agent_session: AgentSession) -> SimpleToolRegistry:
        return SimpleToolRegistry(
            tools=[cls() for cls in DEFERRING_TOOL_CLASSES],
            permission_policy=self.policy,
        )

    def get_middleware(self, agent_session: AgentSession) -> list:
        return [ShellNativeMiddleware(agent_session)]


def make_deferring_runner(session: AgentSession) -> PluginAgentSessionRunner:
    return PluginAgentSessionRunner(session, plugins=[DeferringNativeToolsPlugin()])


@pytest.fixture
def llm(monkeypatch):
    """The battery's mocked LLM boundary, re-bound for this module (fixtures
    do not cross a package boundary). An unconsumed script means a drive
    stopped early — a parked drive that should have run one round and did
    not — so it fails the test."""
    mock = MockLLM()
    monkeypatch.setattr(runner_module, "acompletion", mock.acompletion)
    monkeypatch.setattr(runner_module, "acompletion_stream", mock.acompletion_stream)
    yield mock
    assert mock.responses == [], "scripted LLM responses left unconsumed"


# ── the compromise ───────────────────────────────────────────────────────────


async def test_a_parked_native_shell_call_reaches_the_model_as_exit_1(llm):
    # ── precondition ─────────────────────────────────────────────────────────
    # A native shell call adopted, approved, dispatched — and parked. No
    # `ExecutionResult` exists, and the drive stops there: only a post makes a
    # parked call project at all.
    session = make_session("openai:gpt-5.1", native=True)
    runner = make_deferring_runner(session)
    llm.queue(assistant(shell_call("tc8", "sh_44", {"commands": ["rm -rf build"]}), finish_reason="tool_use"))
    runner.post_message("clean up")
    await runner.run()
    assert session.get_tool_execution("tc8").status == ExecutionStatus.AWAITING_RESULT

    # ── action ───────────────────────────────────────────────────────────────
    llm.queue(assistant(text("still waiting, then")))
    runner.post_message("again")
    await runner.run()

    # ── postcondition ────────────────────────────────────────────────────────
    # THE KNOWING COMPROMISE. `shell_call_output` has stdout, stderr and an
    # exit code and nothing else — no "pending" variant — so the parked call
    # reaches the model as a command that FAILED, with the placeholder in
    # `stderr` and exit code 1, even though the generic `ToolMessage` around it
    # correctly says `is_error=False`. It is only reachable by writing a
    # deferring native tool: none of the four shipped ones can defer. The wire
    # format offers no honest third answer, so this is pinned rather than
    # fixed.
    assert llm.calls[-1]["messages"] == [
        ClientUserMessage(content=[TextBlock(text="clean up")]),
        ClientAssistantMessage(
            content=[
                ClientToolCall(
                    id="tc8",
                    name="shell",
                    arguments={"commands": ["rm -rf build"]},
                    extras={"custom_type": "shell_call", "item_id": "sh_44", "status": "completed"},
                ),
            ],
            provider="openai",
            model="gpt-5.1",
        ),
        ClientToolMessage(
            tool_call_id="tc8",
            content=[TextBlock(text=AWAITING)],
            is_error=False,
            extras={
                "custom_type": "shell_call_output",
                "results": [
                    {
                        "stdout": "",
                        "stderr": AWAITING,
                        "outcome": {"type": "exit", "exit_code": 1},
                    },
                ],
            },
        ),
        ClientUserMessage(content=[TextBlock(text="again")]),
    ]
    assert session.get_tool_execution("tc8").status == ExecutionStatus.AWAITING_RESULT


async def test_a_parked_native_bash_call_keeps_the_placeholder_verbatim(llm):
    # ── precondition ─────────────────────────────────────────────────────────
    session = make_session("anthropic:claude-sonnet-4-5", native=True)
    runner = make_deferring_runner(session)
    llm.queue(assistant(plain_call("tc6", "bash", {"command": "ls"}), finish_reason="tool_use"))
    runner.post_message("list files")
    await runner.run()
    assert session.get_tool_execution("tc6").status == ExecutionStatus.AWAITING_RESULT

    # ── action ───────────────────────────────────────────────────────────────
    llm.queue(assistant(text("still waiting, then")))
    runner.post_message("again")
    await runner.run()

    # ── postcondition ────────────────────────────────────────────────────────
    # The control case that bounds the compromise: the synthesizer fires for
    # `openai_shell` and for nothing else, because only shell's result must be
    # structured. Every other native's plain text IS its native form, so the
    # call is upgraded to its wire name and the placeholder rides through
    # untouched — no invented exit code, no invented failure.
    assert llm.calls[-1]["messages"] == [
        ClientUserMessage(content=[TextBlock(text="list files")]),
        ClientAssistantMessage(
            content=[ClientToolCall(id="tc6", name="bash", arguments={"command": "ls"}, extras={})],
            provider="anthropic",
            model="claude-sonnet-4-5",
        ),
        ClientToolMessage(tool_call_id="tc6", content=[TextBlock(text=AWAITING)], is_error=False),
        ClientUserMessage(content=[TextBlock(text="again")]),
    ]
    assert session.get_tool_execution("tc6").status == ExecutionStatus.AWAITING_RESULT
