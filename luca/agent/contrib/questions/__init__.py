"""luca.agent.contrib.questions — ask the user, park the turn, read the answers.

The first real application of deferred tool calling (spec 0007). `QuestionsTool`
lets the model ask up to four questions and WAIT: `execute()` returns
`ExecutionDeferred()` while the answers are missing, the runner parks the open
turn at `AWAITING_RESULT`, and the driver — a TUI, a websocket server, whatever
installed the tool — renders the questions, collects the answers out of band,
and drives again. The next drive re-dispatches the same call, which now finds an
answer and returns its one `ExecutionResult`.

Two surfaces, and the split is the design. The FRAMEWORK-facing side is
`execute()`, called through the registry's prepared callable. The APP-facing
side is `pending()` / `is_answered()` / `answer()` — ordinary Python methods the
framework has never heard of, because resolving a deferred tool belongs to the
tool and its driver, never to the runner or the registry.

    plugin = QuestionsPlugin(store=session.extras.setdefault("questions", {}))
    runner = PluginAgentSessionRunner(session=session, plugins=[plugin])
    ...
    for execution in runner.pending_deferred_tool_executions():
        answers = await ask_the_human(plugin.tool.pending(execution.tool_call_id))
        plugin.tool.answer(execution.tool_call_id, answers)

The store is a constructor argument and is JSON-shaped all the way down, keyed
by `tool_call_id` — hand in a dict you persist and a resumed session picks the
outstanding questions back up. See `QuestionsTool`'s docstring.
"""

from .plugin import QuestionsPlugin
from .tool import (
    DESCRIPTION,
    QUESTIONS_NAMESPACE,
    OptionsType,
    Question,
    QuestionsTool,
)

__all__ = [
    "DESCRIPTION",
    "QUESTIONS_NAMESPACE",
    "OptionsType",
    "Question",
    "QuestionsPlugin",
    "QuestionsTool",
]
