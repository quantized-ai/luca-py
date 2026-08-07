"""Shared machinery for the 0010 phase-1 battery.

The LLM boundary is mocked at the runner's import site
(`luca.agent.core.runner.acompletion` / `acompletion_stream`, step 0 of the
plan): the mock records every call's kwargs — `messages`, `tools`, `model` —
for the battery's full-object assertions, and returns prebuilt client
responses whose messages carry TYPED native tool calls
(`ApplyPatchToolCall`, `ShellToolCall`), because that is what the real client
delivers after parsing. Client-side parsing/projection is covered by
`tests/client/` (spec 0009) and is not re-tested here.
"""

from __future__ import annotations

import pytest

import luca.agent.core.runner as runner_module
from luca.agent.contrib.plugins import PluginAgentSessionRunner
from luca.agent.contrib.tools import Tool
from luca.agent.core import AgentSessionRunner
from luca.agent.core.models import (
    AgentSession,
    AssistantMessage,
    Conversation,
    LLMConfig,
    RuntimeConfig,
    SessionConfig,
)
from luca.client.providers.openai import ApplyPatchToolCall, ShellToolCall
from luca.client.types import (
    AssistantMessage as ClientAssistantMessage,
    ChatCompletionResponse,
    TextBlock,
    Tool as ClientTool,
    ToolCall as ClientToolCall,
    Usage as ClientUsage,
)
from luca.client.types.streaming import FinishEvent

from .plugin import ControllablePermissionPolicy, NativeToolsPlugin

# ── response builders ────────────────────────────────────────────────────────


def assistant(*blocks, finish_reason: str = "stop") -> ClientAssistantMessage:
    return ClientAssistantMessage(content=list(blocks), finish_reason=finish_reason)


def text(t: str) -> TextBlock:
    return TextBlock(text=t)


def apply_patch_call(
    call_id: str,
    item_id: str,
    arguments: dict,
    status: str = "completed",
) -> ApplyPatchToolCall:
    """A typed OpenAI apply_patch call, exactly as the client parses one."""
    return ApplyPatchToolCall(
        id=call_id,
        name="apply_patch",
        arguments=arguments,
        item_id=item_id,
        status=status,
    )


def shell_call(
    call_id: str,
    item_id: str,
    arguments: dict,
    status: str = "completed",
) -> ShellToolCall:
    """A typed OpenAI shell call, exactly as the client parses one."""
    return ShellToolCall(
        id=call_id,
        name="shell",
        arguments=arguments,
        item_id=item_id,
        status=status,
    )


def plain_call(call_id: str, name: str, arguments: dict) -> ClientToolCall:
    """A base-class tool call — how Anthropic natives and every ordinary
    function call arrive."""
    return ClientToolCall(id=call_id, name=name, arguments=arguments)


def wire_tool(tool_cls: type[Tool]) -> ClientTool:
    """The client function declaration the adapter builds from a fixture
    tool's spec — the expected value for an un-swapped tool in
    `acompletion(tools=…)`."""
    return ClientTool(
        name=tool_cls.name,
        description=tool_cls.description,
        parameters=tool_cls.Args.model_json_schema(),
    )


# ── the mocked LLM boundary ──────────────────────────────────────────────────


class _FakeStream:
    """The minimal shape the runner's streaming path consumes: an async
    context manager whose iterator yields one FinishEvent."""

    def __init__(self, message: ClientAssistantMessage) -> None:
        self._message = message

    async def __aenter__(self) -> _FakeStream:
        return self

    async def __aexit__(self, *exc_info) -> bool:
        return False

    def __aiter__(self):
        return self._events()

    async def _events(self):
        yield FinishEvent(
            message=self._message,
            finish_reason=self._message.finish_reason,
            provider_finish_reason=None,
            usage=ClientUsage(),
        )


class MockLLM:
    """Scripted stand-in for `acompletion` / `acompletion_stream`.

    `queue()` scripts responses in call order; each call pops one (an
    exhausted script raises — the test's script no longer matches what the
    runner did). `calls` records every call's kwargs for assertions."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.responses: list[ClientAssistantMessage] = []

    def queue(self, *messages: ClientAssistantMessage) -> None:
        self.responses.extend(messages)

    async def acompletion(self, **kwargs) -> ChatCompletionResponse:
        self.calls.append(kwargs)
        return ChatCompletionResponse(message=self.responses.pop(0))

    def acompletion_stream(self, **kwargs) -> _FakeStream:
        self.calls.append(kwargs)
        return _FakeStream(self.responses.pop(0))


@pytest.fixture
def llm(monkeypatch):
    mock = MockLLM()
    monkeypatch.setattr(runner_module, "acompletion", mock.acompletion)
    monkeypatch.setattr(runner_module, "acompletion_stream", mock.acompletion_stream)
    yield mock
    # An exhausted script raises on its own; a script left OVER would pass
    # silently — a drive that stopped early (a lost redrive, a halted loop)
    # must fail the test that scripted more rounds than it ran.
    assert mock.responses == [], "scripted LLM responses left unconsumed"


# ── session / runner factories ───────────────────────────────────────────────


def make_session(
    model: str = "openai:gpt-5.1",
    native: bool = True,
    **runtime_overrides,
) -> AgentSession:
    provider, _, model_id = model.partition(":")
    return AgentSession(
        id="s1",
        conversations={"c1": Conversation(id="c1", nodes=[], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(
            llm_config=LLMConfig(model=model_id, provider=provider),
            runtime_config=RuntimeConfig(**runtime_overrides),
            use_native_tools=native,
        ),
    )


def make_runner(
    session: AgentSession,
    policy: ControllablePermissionPolicy | None = None,
) -> PluginAgentSessionRunner:
    return PluginAgentSessionRunner(
        session,
        plugins=[NativeToolsPlugin(policy or ControllablePermissionPolicy())],
    )


def make_bare_runner(session: AgentSession) -> AgentSessionRunner:
    """A runner with NO plugin, NO middleware, NO registry — the UC9
    fail-safe composition."""
    return AgentSessionRunner(session)


def switch_model(session: AgentSession, model: str) -> None:
    """Change the CONFIGURED model; the next drive iteration re-stamps the
    ACTIVE config from it."""
    provider, _, model_id = model.partition(":")
    session.session_config.llm_config = LLMConfig(model=model_id, provider=provider)


async def seed_openai_shell_turn(llm: MockLLM, runner: AgentSessionRunner) -> None:
    """One B2-style turn: "run tests" → typed ShellToolCall (tc2 / sh_12) →
    executes → final "done". The canonical OpenAI-born pair for Battery C."""
    llm.queue(
        assistant(shell_call("tc2", "sh_12", {"commands": ["pytest -q"]}), finish_reason="tool_use"),
        assistant(text("done")),
    )
    runner.post_message("run tests")
    await runner.run()


async def seed_anthropic_bash_turn(llm: MockLLM, runner: AgentSessionRunner) -> None:
    """One B3-style turn: "now bash" → plain bash call (tc6) → adopted as
    `anthropic_bash_20250124` → executes → final "done2". The canonical
    Anthropic-born pair for Battery C."""
    llm.queue(
        assistant(plain_call("tc6", "bash", {"command": "ls"}), finish_reason="tool_use"),
        assistant(text("done2")),
    )
    runner.post_message("now bash")
    await runner.run()


def stored_assistant_parts(session: AgentSession, nth: int = 0) -> list:
    """The parts of the nth AssistantMessage on the main conversation's
    path — the STORED side of an adoption assertion."""
    conversation = session.conversations[session.main_conversation_id]
    messages = [
        session.entries[node_id]
        for node_id in conversation.nodes
        if isinstance(session.entries[node_id], AssistantMessage)
    ]
    return messages[nth].parts
