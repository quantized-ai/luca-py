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
from luca.agent.contrib.skills import SkillsPlugin
from luca.agent.contrib.subagents import SPAWN_TOOL_NAME, SubagentsPlugin
from luca.agent.contrib.tools import Tool
from luca.agent.core.context import CancellationToken
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
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
        *,
        cancellation_token: CancellationToken,
    ) -> str:
        return str(args["a"] + args["b"])


class SubtractTool(Tool):
    name = "subtract"
    description = "Subtract b from a and return the difference."
    Args = BinaryOp

    async def _execute(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
        *,
        cancellation_token: CancellationToken,
    ) -> str:
        return str(args["a"] - args["b"])


class MultiplyTool(Tool):
    name = "multiply"
    description = "Multiply two numbers and return the product."
    Args = BinaryOp

    async def _execute(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
        *,
        cancellation_token: CancellationToken,
    ) -> str:
        return str(args["a"] * args["b"])


SYSTEM_PROMPT = (
    "You're a helpful assistant. Use the provided tools for any arithmetic and "
    "for any filesystem or shell work — don't compute results or invent file "
    "contents yourself."
)


def default_model() -> LLMConfig:
    return LLMConfig(
        model="openai/gpt-5.4-mini",
        provider="openrouter",
        reasoning="medium",
    )


# The `/model` picker's models, grouped by provider so `/model` can drill down:
# pick a provider, then pick one of its models. Only providers registered on
# this branch with models verified live are listed (bedrock ships in a separate
# PR; groq/deepseek/ollama need keys). quantized is listed and needs
# QUANTIZED_API_KEY. `/model provider:model` still switches to anything off this
# list, so an unlisted provider is still reachable by hand.
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
    "quantized": (
        "anthropic/claude-opus-4.6",
        "anthropic/claude-sonnet-4.6",
        "anthropic/claude-haiku-4.5",
        "openai/gpt-5.1",
        "openai/gpt-4o-mini",
        "deepseek/deepseek-v3.2",
        "deepseek/deepseek-r1-0528",
        "qwen/qwen3-235b-a22b",
        "meta-llama/llama-3.3-70b-instruct",
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
    context_manager=None,
    additional_directories: list | None = None,
    extra_rules: list | None = None,
    subagents: bool = True,
    skills: bool = True,
    extra_skill_locations: list[str] | None = None,
) -> tuple[PluginAgentSessionRunner, PermissionStrategy]:
    """The full demo composition: shell + memory plugins, the math tools, one
    shared strategy, and — unless `subagents=False` — the subagent tools. `provider=` is the zero-logic passthrough the tests use
    to inject a `FauxProvider`; `context_manager=` is the same for context
    accounting and compaction — `None` falls back to core's default, which
    accounts but never compacts, so `/compact` fails until one that implements
    `compact()` is passed here."""
    # Skill roots become read-granted directories on the shell plugin below, so
    # bundled files open without a prompt. Read tier only.
    skills_plugin = SkillsPlugin(workspace=workspace, extra_locations=extra_skill_locations) if skills else None
    readable = [*(additional_directories or [])]
    if skills_plugin is not None:
        readable.extend(str(directory) for directory in skills_plugin.skill_directories)
    shell = ShellAccessPlugin(
        workspace=Path(workspace),
        mode=mode,
        additional_directories=readable,
        extra_rules=extra_rules,
    )
    strategy = shell.permission_strategy
    registry = SimpleToolRegistry(
        tools=[AddTool(), SubtractTool(), MultiplyTool()],
        permission_policy=strategy,
    )
    plugins: list = [MemoryPlugin(), shell]
    if skills_plugin is not None:
        plugins.append(skills_plugin)
    if subagents:
        # Installing the plugin is not on its own enough: `subagents_enabled`
        # still has to be True on the session's RuntimeConfig. The capability
        # is configuration, not installation — and a subagent gets the same
        # shell and memory tools the main agent has, each keyed by conversation
        # so the two never overwrite each other.
        plugins.append(SubagentsPlugin())
    runner = PluginAgentSessionRunner(
        session,
        tool_registry=registry,
        plugins=plugins,
        system_prompt_parts=[SYSTEM_PROMPT],
        provider=provider,
        context_manager=context_manager,
    )
    return runner, strategy


def build_faux_provider() -> FauxProvider:
    """Scripted offline conversation for `--faux`: one turn — thinking, a
    gated `multiply` call, a subagent that does its own gated `multiply`, then
    the wrap-up. A second user message exhausts the script (the faux raises),
    which the app surfaces as a turn error.

    ONE subagent, not several, and that is a limit of the script rather than a
    preference: responses are served FIFO from a single queue, so with two
    children racing for the next one there is no way to say which gets which.
    Parallel panels need a real model.

    The subagent's own `multiply` gates, which is the point of scripting one at
    all — it is the shape where the approval modal has to name WHICH
    conversation is asking."""
    faux = FauxProvider()
    faux.set_responses(
        [
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
                [
                    faux_text("Let me have a helper check that independently."),
                    faux_tool_call(
                        SPAWN_TOOL_NAME,
                        {
                            "prompt": "Multiply 6 by 7 with the multiply tool and report the product.",
                            "description": "check the arithmetic",
                            "task_id": "faux-check",
                        },
                        id="tc_faux_2",
                    ),
                ],
                finish_reason="tool_use",
            ),
            # the subagent's own turn — its cells mount inside its panel
            faux_assistant_message(
                [faux_tool_call("multiply", {"a": 6, "b": 7}, id="tc_faux_3")],
                finish_reason="tool_use",
            ),
            faux_assistant_message(
                [faux_text("Confirmed: 6 × 7 = 42.")],
                finish_reason="stop",
            ),
            faux_assistant_message(
                [faux_text("The product is 42 (via the multiply tool), and my helper agrees.")],
                finish_reason="stop",
            ),
        ]
    )
    return faux
