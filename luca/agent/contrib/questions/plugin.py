"""`QuestionsPlugin` — the store owner and the registry for `QuestionsTool`."""

from __future__ import annotations

from luca.agent.contrib.simple_tool_registry import (
    SimpleToolRegistry,
    YoloPermissionPolicy,
)
from luca.agent.core import AgentSession

from .tool import QuestionsTool


class QuestionsPlugin:
    """Bundles `QuestionsTool` in its own auto-allowing registry, and owns the
    store it is built over. The same shape `MemoryPlugin` has, minus the
    system-prompt parts.

    ONE TOOL INSTANCE, HELD ON THE PLUGIN. `MemoryPlugin` builds fresh tools
    per `get_tools()` call because its tools are stateless over a shared store.
    Here the application needs a STABLE REFERENCE to call `answer()` on, so the
    plugin builds it once and exposes it as `plugin.tool`. (The store is shared
    either way, so this is ergonomics, not correctness.)

    `tool_class=` IS HOW A SUBCLASS GETS IN. The whole point of
    `QuestionsTool.generate_content_string_from_answers` being overridable is
    that a driver replaces it; passing the subclass here is how that instance
    reaches the registry.

    `YoloPermissionPolicy`. Asking a question is not an action to approve — a
    gate in front of it would mean the user approving BEING ASKED something. An
    application that wants one composes its own registry over `self.tool`,
    exactly as `MemoryPlugin`'s docstring describes.

    NO `get_system_prompt_parts`. The tool's `description` already carries every
    instruction the model needs, and a prompt part would repeat it in a second
    place that can drift. The hook is duck-typed, so an application that wants
    to add "prefer asking over guessing" as global policy subclasses and adds
    one.

    Wiring, in full:

        plugin = QuestionsPlugin(store=saved)
        runner = PluginAgentSessionRunner(session=session, plugins=[plugin, ...])
        app.questions = plugin.tool          # the app keeps the reference

    To make the store ride the session instead of a file of the application's
    own, hand in `session.extras.setdefault("questions", {})` — nothing else
    changes: no serialization code, no second file, no re-keying."""

    def __init__(
        self,
        store: dict | None = None,
        tool_class: type[QuestionsTool] = QuestionsTool,
    ) -> None:
        self.store: dict = {} if store is None else store
        self.tool: QuestionsTool = tool_class(store=self.store)

    def get_tools(self) -> list[QuestionsTool]:
        return [self.tool]

    def get_tool_registry(self, agent_session: AgentSession) -> SimpleToolRegistry:
        return SimpleToolRegistry(
            tools=self.get_tools(),
            permission_policy=YoloPermissionPolicy(),
        )
