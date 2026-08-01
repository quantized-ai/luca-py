"""Pure approval-prompt model — the gate policy without the UI.

Translates a pending `ToolExecution`'s uncovered `PermissionRequest`s (via
`PermissionStrategy.pending_requests` — rule-covered steps stay silent) into
display-ready `ApprovalPrompt`s whose options carry fully-built
`ApprovalAnswer`s. The modal only shows labels and returns the picked option;
every decision-policy choice lives here, unit-testable without Textual.

The policy (an app choice, not framework behavior — same as the classic REPL
demo): "Approve once" / "Deny" answer with the request's exact pairs at scope
ONCE; picking a tool-suggested option applies it at scope ALWAYS (writing
rules). An option with `answer=None` means "abandon the turn" — the caller
cancels instead of answering. A resourceless tool without the permission
mixin gets a synthesized single-step request.

Deny semantics are the caller's job: a DENY answer makes the whole call dead,
so the remaining prompts of that execution must be skipped.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict

from luca.agent.contrib.resource_permissions import (
    AnswerDecision,
    AnswerOption,
    AnswerScope,
    ApprovalAnswer,
    PermissionRequest,
    PermissionStrategy,
    ResourcePermission,
)
from luca.agent.core.models import ToolExecution


class PromptOption(BaseModel):
    """One selectable answer. `answer=None` is the abandon-turn option."""

    label: str
    answer: ApprovalAnswer | None = None

    model_config = ConfigDict(extra="forbid")

    @property
    def is_abandon(self) -> bool:
        return self.answer is None

    @property
    def is_deny(self) -> bool:
        return self.answer is not None and self.answer.decision is AnswerDecision.DENY


class ApprovalPrompt(BaseModel):
    """One approval step of one execution, ready for display."""

    tool_name: str
    step: int
    total_steps: int
    resources: list[str]
    preview: str
    options: list[PromptOption]
    # Which conversation is asking, when it is not the main one. `None` for the
    # main agent — the ordinary case, where naming it would be noise. The
    # execution carries this itself (`ToolExecution.conversation_id`), which is
    # exactly why no wrapper type is needed to attribute a gate.
    conversation_id: str | None = None
    # The same subagent, said in words. An id answers "which one" only if you
    # already know which is which, and two subagents can gate at the same
    # moment — so the thing the user decides with is the TASK, not the key.
    # None whenever `conversation_id` is, and whenever nothing named it.
    conversation_label: str | None = None

    model_config = ConfigDict(extra="forbid")


ABANDON_LABEL = "Abandon turn (cancel — unanswered calls never run)"


def _pair_label(pair: ResourcePermission) -> str:
    if pair.resource is None:
        return pair.permission
    return f"{pair.permission}:{pair.resource}"


def _option_label(option: AnswerOption) -> str:
    preview = option.metadata.get("preview")
    if preview:
        return preview
    return ", ".join(_pair_label(pair) for pair in option.resource_permissions)


def _build_options(request: PermissionRequest) -> list[PromptOption]:
    exact = AnswerOption(resource_permissions=request.resources)
    options = [
        PromptOption(
            label="Approve once",
            answer=ApprovalAnswer(
                answer_option=exact,
                decision=AnswerDecision.APPROVE,
            ),
        ),
    ]
    options.extend(
        PromptOption(
            label=_option_label(option),
            answer=ApprovalAnswer(
                answer_option=option,
                decision=AnswerDecision.APPROVE,
                scope=AnswerScope.ALWAYS,
            ),
        )
        for option in request.answer_options
    )
    options.append(
        PromptOption(
            label="Deny",
            answer=ApprovalAnswer(
                answer_option=exact,
                decision=AnswerDecision.DENY,
            ),
        )
    )
    options.append(PromptOption(label=ABANDON_LABEL, answer=None))
    return options


def build_approval_prompts(
    execution: ToolExecution,
    strategy: PermissionStrategy,
    *,
    main_conversation_id: str | None = None,
    subagent_labels: Mapping[str, str] | None = None,
) -> list[ApprovalPrompt]:
    """The execution's UNCOVERED approval steps as display-ready prompts.

    `main_conversation_id`, when given, is what makes "which conversation is
    asking" answerable: a gate from anything else is labelled as a subagent's.
    Now that `pending_approvals()` is subtree-scoped, a flat list can mix the
    main agent's requests with several subagents' — and the execution
    attributes itself, so no wrapper type is needed.

    `subagent_labels` maps a conversation id to what that subagent was asked to
    do. Optional, because attribution and naming are different jobs: the id is
    always available here, the task only to a caller that tracks its
    subagents."""
    name = execution.raw_tool_call.name
    requests = strategy.pending_requests(execution)
    if not requests:  # resourceless tool without the mixin (add/subtract/…)
        requests = [
            PermissionRequest(
                resources=[ResourcePermission(permission=name)],
                metadata={"preview": f"Run {name}"},
            )
        ]
    asking = (
        execution.conversation_id
        if main_conversation_id is not None and execution.conversation_id != main_conversation_id
        else None
    )
    label = (subagent_labels or {}).get(asking) if asking is not None else None
    prompts: list[ApprovalPrompt] = []
    for index, request in enumerate(requests):
        resources = [_pair_label(pair) for pair in request.resources]
        preview = request.metadata.get("preview") or ", ".join(resources)
        prompts.append(
            ApprovalPrompt(
                tool_name=name,
                step=index + 1,
                total_steps=len(requests),
                resources=resources,
                preview=preview,
                options=_build_options(request),
                conversation_id=asking,
                conversation_label=label,
            )
        )
    return prompts
