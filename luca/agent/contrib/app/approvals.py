"""Pure approval-prompt model — the gate policy, with no front end in it.

Translates a pending `ToolExecution`'s uncovered `PermissionRequest`s (via
`PermissionStrategy.pending_requests` — rule-covered steps stay silent) into
the design's fixed prompt shape:

    ❯  1  Approve once
       2  Approve always — <exactly what is being widened>
       3  Deny — tell Luca what to do instead
       4  Cancel turn                                        esc

Option 2 only exists when the tool suggested a widening (`answer_options`);
its suffix is generated from that suggestion — a bare "Approve always" is
never rendered. Choosing Deny answers DENY and gives the composer back;
Cancel turn aborts the whole turn (`esc` binds to it). All decision policy
lives here, unit-testable without Textual.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

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

OptionKind = Literal["approve", "always", "deny", "cancel"]

DENY_LABEL = "Deny — tell Luca what to do instead"
CANCEL_LABEL = "Cancel turn"


class PromptOption(BaseModel):
    """One selectable answer. `answer=None` is the cancel-turn option."""

    label: str
    kind: OptionKind
    answer: ApprovalAnswer | None = None
    key_hint: str | None = None

    model_config = ConfigDict(extra="forbid")

    @property
    def is_cancel(self) -> bool:
        return self.kind == "cancel"

    @property
    def is_deny(self) -> bool:
        return self.kind == "deny"


class ApprovalPromptModel(BaseModel):
    """One approval step of one execution, ready for display."""

    tool_name: str
    question: str  # theme-token spans allowed
    step: int
    total_steps: int
    options: list[PromptOption]
    conversation_id: str | None = None
    conversation_label: str | None = None

    model_config = ConfigDict(extra="forbid")


def _pair_label(pair: ResourcePermission) -> str:
    if pair.resource is None:
        return pair.permission
    return f"{pair.permission}:{pair.resource}"


def _scope_label(option: AnswerOption) -> str:
    preview = option.metadata.get("preview")
    if preview:
        return str(preview)
    return ", ".join(_pair_label(pair) for pair in option.resource_permissions)


def _build_options(request: PermissionRequest) -> list[PromptOption]:
    exact = AnswerOption(resource_permissions=request.resources)
    options = [
        PromptOption(
            label="Approve once",
            kind="approve",
            answer=ApprovalAnswer(answer_option=exact, decision=AnswerDecision.APPROVE),
        ),
    ]
    if request.answer_options:
        # The FIRST suggestion is the offered widening; the prompt stays four
        # options by design. The user always sees exactly what they widen.
        suggested = request.answer_options[0]
        options.append(
            PromptOption(
                label=f"Approve always — {_scope_label(suggested)}",
                kind="always",
                answer=ApprovalAnswer(
                    answer_option=suggested,
                    decision=AnswerDecision.APPROVE,
                    scope=AnswerScope.ALWAYS,
                ),
            )
        )
    options.append(
        PromptOption(
            label=DENY_LABEL,
            kind="deny",
            answer=ApprovalAnswer(answer_option=exact, decision=AnswerDecision.DENY),
        )
    )
    options.append(PromptOption(label=CANCEL_LABEL, kind="cancel", key_hint="esc"))
    return options


def _question(tool_name: str, request: PermissionRequest, step: int, total: int) -> str:
    preview = request.metadata.get("preview")
    body = str(preview) if preview else f"Run {tool_name}"
    question = body.rstrip("?.") + "?"
    if total > 1:
        question += f"  (step {step} of {total})"
    return question


def build_approval_prompts(
    execution: ToolExecution,
    strategy: PermissionStrategy,
    *,
    main_conversation_id: str | None = None,
    subagent_labels: Mapping[str, str] | None = None,
) -> list[ApprovalPromptModel]:
    """The execution's UNCOVERED approval steps as display-ready prompts.
    A gate raised by a subagent names the task it belongs to in the question,
    so two simultaneous gates stay tellable apart."""
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
    prompts: list[ApprovalPromptModel] = []
    for index, request in enumerate(requests):
        question = _question(name, request, index + 1, len(requests))
        if asking is not None:
            question = f"[faint]task · {label or asking} —[/] {question}"
        prompts.append(
            ApprovalPromptModel(
                tool_name=name,
                question=question,
                step=index + 1,
                total_steps=len(requests),
                options=_build_options(request),
                conversation_id=asking,
                conversation_label=label,
            )
        )
    return prompts
