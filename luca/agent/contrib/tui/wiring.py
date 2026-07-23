"""Agent composition for the TUI.

`build_runner` reproduces the demo wiring in one place: the shell plugin's
tools scoped to a workspace, the memory plugin, the three demo math tools,
and ONE `PermissionStrategy` (built and seeded by `ShellAccessPlugin`)
shared by every registry so a single approval gate serves everything.

`build_faux_provider` scripts an offline conversation (`--faux`) so the TUI
can be exercised end-to-end with no key and no network — the same
`FauxProvider` the tests inject.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

from luca.agent.contrib.memory import MemoryPlugin
from luca.agent.contrib.plugins import PluginAgentSessionRunner
from luca.agent.contrib.resource_permissions import PermissionStrategy
from luca.agent.contrib.shell import ShellAccessPlugin
from luca.agent.contrib.simple_tool_registry import SimpleToolRegistry
from luca.agent.core import Tool
from luca.agent.core.context import CancellationToken, ToolContext
from luca.agent.core.models import AgentSession, LLMConfig
from luca.client.testing import (
    FauxProvider,
    faux_assistant_message,
    faux_text,
    faux_thinking,
    faux_tool_call,
)


# ── demo math tools ────────────────────────────────────────────────────────────
# Resourceless tools without the permission mixin: the approval layer
# synthesizes a plain "run <name>" request for them, exercising the
# no-approval-context path of the gate.


class BinaryOp(BaseModel):
    a: float = Field(description="The first operand.")
    b: float = Field(description="The second operand.")


class AddTool(Tool):
    name = "add"
    description = "Add two numbers and return the sum."
    Args = BinaryOp

    async def _execute(
        self, args: dict, context: ToolContext,
        *, cancellation_token: CancellationToken,
    ) -> str:
        return str(args["a"] + args["b"])


class SubtractTool(Tool):
    name = "subtract"
    description = "Subtract b from a and return the difference."
    Args = BinaryOp

    async def _execute(
        self, args: dict, context: ToolContext,
        *, cancellation_token: CancellationToken,
    ) -> str:
        return str(args["a"] - args["b"])


class MultiplyTool(Tool):
    name = "multiply"
    description = "Multiply two numbers and return the product."
    Args = BinaryOp

    async def _execute(
        self, args: dict, context: ToolContext,
        *, cancellation_token: CancellationToken,
    ) -> str:
        return str(args["a"] * args["b"])


SYSTEM_PROMPT = (
    "You're a helpful assistant. Use the provided tools for any arithmetic and "
    "for any filesystem or shell work — don't compute results or invent file "
    "contents yourself."
)


def default_model() -> LLMConfig:
    return LLMConfig(
        model="openai/gpt-5.4-mini", provider="openrouter",
        reasoning="medium",
    )


# The `/model` picker's models, grouped by provider so `/model` can drill down:
# pick a provider, then pick one of its models. A short curated set (the client
# catalog is too stale to drive a picker). `/model provider:model` still
# switches to anything off this list, including providers not shown here.
RECOMMENDED_MODELS: dict[str, tuple[str, ...]] = {
    "anthropic": (
        "claude-opus-4-8",
        "claude-sonnet-5",
        "claude-haiku-4-5-20251001",
        "claude-fable-5",
    ),
    "openrouter": (
        "openai/gpt-5.4-mini",
        "openai/gpt-5.4",
        "anthropic/claude-sonnet-5",
        "anthropic/claude-opus-4-8",
        "moonshotai/kimi-k2.7-code",
        "meta-llama/llama-3.3-70b-instruct",
        "deepseek/deepseek-r1",
    ),
    "openai": (
        "gpt-5.4",
        "gpt-5.4-mini",
    ),
    "bedrock": (
        "us.amazon.nova-lite-v1:0",
        "us.amazon.nova-pro-v1:0",
        "us.amazon.nova-micro-v1:0",
        "us.meta.llama3-3-70b-instruct-v1:0",
        "us.meta.llama4-maverick-17b-instruct-v1:0",
        "us.deepseek.r1-v1:0",
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    ),
    "groq": (
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
    ),
    "deepseek": (
        "deepseek-chat",
        "deepseek-reasoner",
    ),
}


def faux_model() -> LLMConfig:
    return LLMConfig(model="fake-model", provider="faux")


def build_runner(
    session: AgentSession,
    *,
    workspace: str | os.PathLike[str] = ".",
    provider=None,
    mode: str = "ask",
) -> tuple[PluginAgentSessionRunner, PermissionStrategy]:
    """The full demo composition: shell + memory plugins, the math tools, one
    shared strategy. `provider=` is the zero-logic passthrough the tests use
    to inject a `FauxProvider`."""
    shell = ShellAccessPlugin(workspace=Path(workspace), mode=mode)
    strategy = shell.permission_strategy
    registry = SimpleToolRegistry(
        tools=[AddTool(), SubtractTool(), MultiplyTool()],
        permission_policy=strategy,
    )
    runner = PluginAgentSessionRunner(
        session,
        tool_registry=registry,
        plugins=[MemoryPlugin(), shell],
        system_prompt_parts=[SYSTEM_PROMPT],
        provider=provider,
    )
    return runner, strategy


def build_faux_provider() -> FauxProvider:
    """Scripted offline conversation for `--faux`: one turn — thinking, a
    gated `multiply` call, then the wrap-up. A second user message exhausts
    the script (the faux raises), which the app surfaces as a turn error."""
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [
                faux_thinking(
                    "The user wants arithmetic — I should multiply.",
                    signature="faux-signature",
                ),
                faux_tool_call("multiply", {"a": 6, "b": 7}, id="tc_faux_1"),
            ],
            finish_reason="tool_use",
        ),
        faux_assistant_message(
            [faux_text("The product is 42 (via the multiply tool).")],
            finish_reason="stop",
        ),
    ])
    return faux
