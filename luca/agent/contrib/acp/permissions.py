"""Gated tool calls → `session/request_permission`, and the answers back.

The runner parks a turn when a call needs approval it cannot resolve from
rules. The APPROVAL EVENT IS NOT THE SIGNAL: `ApprovalRequired` is emitted as
the last event before parking, but it can be superseded, so the durable read
`runner.pending_approvals()` after the run drains is what a driver acts on.
The TUI does the same thing for the same reason.

One execution can require SEVERAL approvals — a shell tool asks for directory
access and then for its own verb — and ACP's `session/request_permission`
carries one tool call and one flat option list. So each step becomes its own
request, and the client sees "may I read this directory?" then "may I read
this file?". Answers are collected for the whole execution and applied in one
go, exactly as `apply_answer` expects.

Coverage is emergent: an answer that does not cover every required pair leaves
the call PENDING and the gate re-arms on the next drive. `answered` tracks what
the user has already been asked so a repeat is visible rather than looking like
a hang.
"""

from __future__ import annotations

import logging

from acp.helpers import update_tool_call
from acp.schema import PermissionOption

from luca.agent.contrib.app.approvals import ApprovalPromptModel, build_approval_prompts
from luca.agent.contrib.resource_permissions import PermissionStrategy
from luca.agent.core.models import ToolExecution

from .stream import tool_title

logger = logging.getLogger(__name__)

# Our four prompt options onto ACP's four kinds. `cancel` has no counterpart
# and needs none: ACP models "stop the whole turn" as the request's OUTCOME
# (`cancelled`), not as something the agent offers in the list.
OPTION_KINDS = {
    "approve": "allow_once",
    "always": "allow_always",
    "deny": "reject_once",
}


def permission_options(prompt: ApprovalPromptModel) -> list[PermissionOption]:
    """One approval step's選 choices, in the order the prompt built them."""
    return [
        PermissionOption(option_id=str(index), name=option.label, kind=OPTION_KINDS[option.kind])
        for index, option in enumerate(prompt.options)
        if option.kind in OPTION_KINDS
    ]


class Cancelled(Exception):
    """The client answered a permission request with `cancelled`. The turn is
    over; the driver winds the runner down and reports `cancelled`."""


class PermissionBridge:
    """Asks the client about every gated call, then writes the answers to the
    strategy. One instance per session, because `answered` is per session."""

    def __init__(self, connection, session_id: str, strategy: PermissionStrategy) -> None:
        self._connection = connection
        self._session_id = session_id
        self._strategy = strategy
        self.answered: set[str] = set()

    async def resolve(self, executions: list[ToolExecution], main_conversation_id: str) -> None:
        """Ask about each execution and apply what comes back.

        Answers are applied AFTER the whole loop, so a cancel part-way through
        applies nothing: the turn is ending and half-granting permissions on
        the way out would leave rules the user never agreed to."""
        collected: list[tuple[ToolExecution, list]] = []
        for execution in executions:
            if execution.tool_call_id in self.answered:
                logger.info(
                    "asking again about %s: the previous answer did not cover the request",
                    execution.tool_call_id,
                )
            answers = await self._ask_about(execution, main_conversation_id)
            self.answered.add(execution.tool_call_id)
            collected.append((execution, answers))
        for execution, answers in collected:
            self._strategy.apply_answer(execution, answers)

    async def _ask_about(self, execution: ToolExecution, main_conversation_id: str) -> list:
        answers: list = []
        for prompt in build_approval_prompts(
            execution,
            self._strategy,
            main_conversation_id=main_conversation_id,
        ):
            options = permission_options(prompt)
            response = await self._connection.request_permission(
                session_id=self._session_id,
                tool_call=update_tool_call(
                    execution.tool_call_id,
                    title=tool_title(execution),
                    raw_input=execution.raw_tool_call.arguments,
                ),
                options=options,
            )
            outcome = response.outcome
            if getattr(outcome, "outcome", None) != "selected":
                raise Cancelled
            chosen = prompt.options[int(outcome.option_id)]
            answers.append(chosen.answer)
            if chosen.is_deny:
                # The remaining steps of this execution are moot: it is not
                # going to run.
                break
        return answers
