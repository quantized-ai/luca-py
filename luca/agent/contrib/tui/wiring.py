"""Agent composition for the TUI.

`build_runner` reproduces the demo wiring in one place: the prompt plugins
(the model's base prompt, the environment, the project's instruction files),
the shell plugin's tools scoped to a workspace, the memory plugin, the
questions plugin (`ask_user`), the three demo math tools, and ONE
`PermissionStrategy` (built and seeded by `ShellAccessPlugin`) shared by every
registry so a single approval gate serves everything.

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
from luca.agent.contrib.prompts import InstructionsPlugin, SystemPromptPlugin
from luca.agent.contrib.questions import QuestionsPlugin
from luca.agent.contrib.resource_permissions import PermissionStrategy
from luca.agent.contrib.shell import ShellAccessPlugin
from luca.agent.contrib.simple_tool_registry import SimpleToolRegistry
from luca.agent.contrib.skills import SkillsPlugin
from luca.agent.contrib.subagents import SPAWN_TOOL_NAME, SubagentsPlugin
from luca.agent.contrib.tools import Tool
from luca.agent.contrib.websearch import WebSearchPlugin
from luca.agent.core.context import CancellationToken
from luca.agent.core.models import AgentSession, LLMConfig
from luca.client.testing import (
    FauxProvider,
    faux_assistant_message,
    faux_text,
    faux_thinking,
    faux_tool_call,
)

from .render import SCRATCHPAD_STORE_KEY, TODO_STORE_KEY

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
        tool_name: str,
        tool_call_id: str,
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
        tool_name: str,
        tool_call_id: str,
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
        tool_name: str,
        tool_call_id: str,
        cancellation_token: CancellationToken,
    ) -> str:
        return str(args["a"] * args["b"])


def default_model() -> LLMConfig:
    return LLMConfig(
        model="openai/gpt-5.4-mini",
        provider="openrouter",
        model_options={"reasoning": "medium"},
    )


def faux_model() -> LLMConfig:
    return LLMConfig(model="fake-model", provider="faux")


def build_runner(
    session: AgentSession,
    *,
    workspace: str | os.PathLike[str] = ".",
    provider=None,
    api_key: str | None = None,
    mode: str = "ask",
    context_manager=None,
    additional_directories: list | None = None,
    extra_rules: list | None = None,
    subagents: bool = True,
    skills: bool = True,
    extra_skill_locations: list[str] | None = None,
    instructions: bool = True,
    extra_instructions: list[str] | None = None,
    questions_store: dict | None = None,
    websearch=None,
) -> tuple[PluginAgentSessionRunner, PermissionStrategy, QuestionsPlugin]:
    """The full demo composition: shell + memory + questions plugins, the math
    tools, one shared strategy, and — unless `subagents=False` — the subagent
    tools. `provider=` is the zero-logic passthrough the tests use to inject a
    `FauxProvider`, and `api_key=` the credential the app resolved for the
    session's provider (None = let the client read its env var);
    `context_manager=` is the same for context accounting and compaction —
    `None` falls back to core's default, which accounts but never compacts, so
    `/compact` fails until one that implements `compact()` is passed here.

    `questions_store=` is the dict `QuestionsTool` keeps its outstanding jobs
    in. The app hands in the one it loaded from the session's `.tui.json`
    sidecar, so a question parked when the process died comes back parked;
    passing nothing starts clean, which is right for a script and wrong for
    anything a user comes back to. The plugin is returned alongside the runner
    because the driver needs a stable reference to `answer()` on — resolving a
    deferred tool belongs to the tool and its driver, never to the runner."""
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
    # First in the list, so the persona and the model addendum open the
    # assembled prompt and every plugin's tool blurb follows them. The env and
    # instruction parts carry positive priorities and sort to the tail.
    # THE APP OWNS THE MEMORY. The plugin takes whatever dicts it is handed and
    # mutates them in place; putting them on the session is what makes the todo
    # list and the scratchpad survive a resume, because `save_session` writes
    # the session and they are part of it. `setdefault` so a reloaded session
    # is picked up as-is and a fresh one starts empty.
    memory = MemoryPlugin(
        scratchpad_store=session.extras.setdefault(SCRATCHPAD_STORE_KEY, {}),
        todo_store=session.extras.setdefault(TODO_STORE_KEY, {}),
    )
    # THE `ask_user` TOOL, and the one deferred tool the demo ships. Its store
    # is the app's, not the session's: outstanding questions are the
    # INTERFACE's state, and a second driver loading this session has no use
    # for a prompt only this TUI knows how to render.
    questions = QuestionsPlugin(store=questions_store)
    plugins: list = [SystemPromptPlugin(workspace=workspace), memory, questions, shell]
    if skills_plugin is not None:
        plugins.append(skills_plugin)
    if instructions:
        plugins.append(InstructionsPlugin(workspace=workspace, extra=extra_instructions))
    if subagents:
        # Installing the plugin is not on its own enough: `subagents_enabled`
        # still has to be True on the session's RuntimeConfig. The capability
        # is configuration, not installation — and a subagent gets the same
        # shell and memory tools the main agent has, each keyed by conversation
        # so the two never overwrite each other.
        plugins.append(SubagentsPlugin())
    if websearch is not None:
        # Provider-hosted web tools: the block IS the plugin's config (D7),
        # threaded as one typed kwarg — nothing is persisted, and the
        # per-model support table inside the middleware still has the last
        # word on whether anything is advertised.
        plugins.append(WebSearchPlugin(websearch))
    runner = PluginAgentSessionRunner(
        session,
        tool_registry=registry,
        plugins=plugins,
        provider=provider,
        api_key=api_key,
        context_manager=context_manager,
    )
    return runner, strategy, questions


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
