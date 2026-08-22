"""The approval prompt as a view-model.

The policy — which steps are still uncovered, what each option grants — lives
in `luca.agent.contrib.app.approvals` and is shared with every other front end.
This is the one line of it that is Textual's business: turning a prompt into
the `ApprovalState` the dock renders.
"""

from __future__ import annotations

from luca.agent.contrib.app.approvals import ApprovalPromptModel

from . import state as vm


def approval_state(prompt: ApprovalPromptModel) -> vm.ApprovalState:
    return vm.ApprovalState(
        question=prompt.question,
        options=[vm.ApprovalOption(label=o.label, key_hint=o.key_hint) for o in prompt.options],
        selected=0,
    )
