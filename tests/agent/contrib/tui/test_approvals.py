"""`build_approval_prompts` — the pure approval-prompt policy.

Each test: a known execution + strategy, one call, full-object asserts on the
resulting `ApprovalPromptModel`s (options carry fully-built `ApprovalAnswer`s).

Which steps are still pending is the strategy's job (asserted in
`tests/agent/contrib/test_resource_permissions.py`); what this module owns is
the translation of a pending request into the design's fixed prompt shape —
the option set, the labels, the question wording, and the answers each option
carries.
"""

from luca.agent.contrib.app.approvals import (
    CANCEL_LABEL,
    DENY_LABEL,
    ApprovalPromptModel,
    PromptOption,
    build_approval_prompts,
)
from luca.agent.contrib.resource_permissions import (
    AnswerDecision,
    AnswerOption,
    AnswerScope,
    ApprovalAnswer,
    PermissionStrategy,
    ResourcePermission,
)
from luca.agent.contrib.tui import state as vm
from luca.agent.contrib.tui.approvals import approval_state
from luca.agent.core.models import (
    ApprovalOption,
    ExecutionStatus,
    ToolCall,
    ToolExecution,
    ToolKind,
)
from tests.agent.scenarios import spec

# ── execution literals ────────────────────────────────────────────────────────
# A call only reaches the approval gate once its tool resolved, so every
# literal carries the `tool_spec` snapshot the strategy reads the kind from.
# They stand alone — never put in a session — so none carries a `tool_spec_id`.

READ_EXECUTION = ToolExecution(
    id="te1",
    created_at=500,
    tool_call_id="tc1",
    raw_tool_call=ToolCall(id="tc1", name="read", arguments={"path": "/tmp/notes.txt"}),
    tool_spec=spec("read", tool_kind=ToolKind.READ),
    status=ExecutionStatus.PENDING,
    extras={
        "approval_context": {
            "requests": [
                {
                    "resources": [{"permission": "read", "resource": "/tmp/notes.txt"}],
                    "answer_options": [
                        {
                            "resource_permissions": [{"permission": "read", "resource": "/tmp/*"}],
                            "metadata": {"preview": "all reads under /tmp/*"},
                        }
                    ],
                    "metadata": {"preview": "Read /tmp/notes.txt"},
                }
            ]
        }
    },
)

MATH_EXECUTION = ToolExecution(
    id="te2",
    created_at=500,
    tool_call_id="tc2",
    raw_tool_call=ToolCall(id="tc2", name="add", arguments={"a": 1, "b": 2}),
    tool_spec=spec("add"),
    status=ExecutionStatus.PENDING,
)

TWO_STEP_EXECUTION = ToolExecution(
    id="te3",
    created_at=500,
    tool_call_id="tc3",
    raw_tool_call=ToolCall(id="tc3", name="edit", arguments={"path": "/etc/hosts"}),
    tool_spec=spec("edit", tool_kind=ToolKind.EDIT),
    status=ExecutionStatus.PENDING,
    extras={
        "approval_context": {
            "requests": [
                {
                    "resources": [{"permission": "access_directory", "resource": "/etc"}],
                    "metadata": {"preview": "Access /etc"},
                },
                {
                    "resources": [{"permission": "edit", "resource": "/etc/hosts"}],
                    "metadata": {"preview": "Edit /etc/hosts"},
                },
            ]
        }
    },
)

# A request the tool wrote no previews for — neither on the step nor on its
# suggested option — so the question falls back to the tool name and the
# widening label to the pairs themselves.
UNLABELLED_EXECUTION = ToolExecution(
    id="te4",
    created_at=500,
    tool_call_id="tc4",
    raw_tool_call=ToolCall(id="tc4", name="write", arguments={"path": "/srv/app.log"}),
    tool_spec=spec("write", tool_kind=ToolKind.EDIT),
    status=ExecutionStatus.PENDING,
    extras={
        "approval_context": {
            "requests": [
                {
                    "resources": [{"permission": "write", "resource": "/srv/app.log"}],
                    "answer_options": [
                        {
                            "resource_permissions": [
                                {"permission": "write", "resource": "/srv/*"},
                                {"permission": "create", "resource": "/srv/*"},
                            ]
                        }
                    ],
                }
            ]
        }
    },
)

# One step over two resources, so a rule can cover half of it.
MULTI_PAIR_EXECUTION = ToolExecution(
    id="te5",
    created_at=500,
    tool_call_id="tc5",
    raw_tool_call=ToolCall(
        id="tc5",
        name="read",
        arguments={"paths": ["/etc/hosts", "/tmp/scratch"]},
    ),
    tool_spec=spec("read", tool_kind=ToolKind.READ),
    status=ExecutionStatus.PENDING,
    extras={
        "approval_context": {
            "requests": [
                {
                    "resources": [
                        {"permission": "read", "resource": "/etc/hosts"},
                        {"permission": "read", "resource": "/tmp/scratch"},
                    ],
                    "metadata": {"preview": "Read two files"},
                }
            ]
        }
    },
)

# A preview whose trailing punctuation must not double up in the question.
DOTTED_EXECUTION = ToolExecution(
    id="te6",
    created_at=500,
    tool_call_id="tc6",
    raw_tool_call=ToolCall(id="tc6", name="bash", arguments={"command": "rm -r build"}),
    tool_spec=spec("bash", tool_kind=ToolKind.EXECUTE),
    status=ExecutionStatus.PENDING,
    extras={
        "approval_context": {
            "requests": [
                {
                    "resources": [{"permission": "execute", "resource": "rm"}],
                    "metadata": {"preview": "Remove build artifacts."},
                }
            ]
        }
    },
)

# The same gate raised from inside a subagent's conversation.
SUBAGENT_EXECUTION = READ_EXECUTION.model_copy(update={"conversation_id": "c_child"})
MAIN_EXECUTION = READ_EXECUTION.model_copy(update={"conversation_id": "c_main"})


def test_resourced_request_builds_the_full_option_set():
    strategy = PermissionStrategy()

    prompts = build_approval_prompts(READ_EXECUTION, strategy)

    exact = AnswerOption(
        resource_permissions=[
            ResourcePermission(permission="read", resource="/tmp/notes.txt"),
        ]
    )
    suggested = AnswerOption(
        resource_permissions=[
            ResourcePermission(permission="read", resource="/tmp/*"),
        ],
        metadata={"preview": "all reads under /tmp/*"},
    )
    assert prompts == [
        ApprovalPromptModel(
            tool_name="read",
            question="Read /tmp/notes.txt?",
            step=1,
            total_steps=1,
            options=[
                PromptOption(
                    label="Approve once",
                    kind="approve",
                    answer=ApprovalAnswer(
                        answer_option=exact,
                        decision=AnswerDecision.APPROVE,
                    ),
                ),
                PromptOption(
                    label="Approve always — all reads under /tmp/*",
                    kind="always",
                    answer=ApprovalAnswer(
                        answer_option=suggested,
                        decision=AnswerDecision.APPROVE,
                        scope=AnswerScope.ALWAYS,
                    ),
                ),
                PromptOption(
                    label=DENY_LABEL,
                    kind="deny",
                    answer=ApprovalAnswer(
                        answer_option=exact,
                        decision=AnswerDecision.DENY,
                    ),
                ),
                PromptOption(label=CANCEL_LABEL, kind="cancel", key_hint="esc"),
            ],
            conversation_id=None,
            conversation_label=None,
        )
    ]


def test_resourceless_tool_gets_a_synthesized_three_option_prompt():
    strategy = PermissionStrategy()

    prompts = build_approval_prompts(MATH_EXECUTION, strategy)

    exact = AnswerOption(
        resource_permissions=[
            ResourcePermission(permission="add"),
        ]
    )
    assert prompts == [
        ApprovalPromptModel(
            tool_name="add",
            question="Run add?",
            step=1,
            total_steps=1,
            options=[
                PromptOption(
                    label="Approve once",
                    kind="approve",
                    answer=ApprovalAnswer(
                        answer_option=exact,
                        decision=AnswerDecision.APPROVE,
                    ),
                ),
                PromptOption(
                    label=DENY_LABEL,
                    kind="deny",
                    answer=ApprovalAnswer(
                        answer_option=exact,
                        decision=AnswerDecision.DENY,
                    ),
                ),
                PromptOption(label=CANCEL_LABEL, kind="cancel", key_hint="esc"),
            ],
            conversation_id=None,
            conversation_label=None,
        )
    ]


def test_previewless_widening_is_labelled_by_its_pairs():
    strategy = PermissionStrategy()

    prompts = build_approval_prompts(UNLABELLED_EXECUTION, strategy)

    exact = AnswerOption(
        resource_permissions=[
            ResourcePermission(permission="write", resource="/srv/app.log"),
        ]
    )
    suggested = AnswerOption(
        resource_permissions=[
            ResourcePermission(permission="write", resource="/srv/*"),
            ResourcePermission(permission="create", resource="/srv/*"),
        ]
    )
    assert prompts == [
        ApprovalPromptModel(
            tool_name="write",
            question="Run write?",
            step=1,
            total_steps=1,
            options=[
                PromptOption(
                    label="Approve once",
                    kind="approve",
                    answer=ApprovalAnswer(
                        answer_option=exact,
                        decision=AnswerDecision.APPROVE,
                    ),
                ),
                PromptOption(
                    label="Approve always — write:/srv/*, create:/srv/*",
                    kind="always",
                    answer=ApprovalAnswer(
                        answer_option=suggested,
                        decision=AnswerDecision.APPROVE,
                        scope=AnswerScope.ALWAYS,
                    ),
                ),
                PromptOption(
                    label=DENY_LABEL,
                    kind="deny",
                    answer=ApprovalAnswer(
                        answer_option=exact,
                        decision=AnswerDecision.DENY,
                    ),
                ),
                PromptOption(label=CANCEL_LABEL, kind="cancel", key_hint="esc"),
            ],
            conversation_id=None,
            conversation_label=None,
        )
    ]


def test_partially_covered_step_answers_only_over_its_pending_pairs():
    strategy = PermissionStrategy()
    strategy.add_rule(
        None,
        ResourcePermission(permission="read", resource="/etc/*"),
        ApprovalOption.ALLOW,
    )

    prompts = build_approval_prompts(MULTI_PAIR_EXECUTION, strategy)

    exact = AnswerOption(
        resource_permissions=[
            ResourcePermission(permission="read", resource="/tmp/scratch"),
        ]
    )
    assert prompts == [
        ApprovalPromptModel(
            tool_name="read",
            question="Read two files?",
            step=1,
            total_steps=1,
            options=[
                PromptOption(
                    label="Approve once",
                    kind="approve",
                    answer=ApprovalAnswer(
                        answer_option=exact,
                        decision=AnswerDecision.APPROVE,
                    ),
                ),
                PromptOption(
                    label=DENY_LABEL,
                    kind="deny",
                    answer=ApprovalAnswer(
                        answer_option=exact,
                        decision=AnswerDecision.DENY,
                    ),
                ),
                PromptOption(label=CANCEL_LABEL, kind="cancel", key_hint="esc"),
            ],
            conversation_id=None,
            conversation_label=None,
        )
    ]


# The step-framing and question-wording tests below assert the question line
# only: each prompt's option set is identical to the ones asserted whole
# above, and repeating it per step would bury the one thing they are about.


def test_multi_step_context_numbers_each_question():
    strategy = PermissionStrategy()

    prompts = build_approval_prompts(TWO_STEP_EXECUTION, strategy)

    assert [(p.step, p.total_steps, p.question) for p in prompts] == [
        (1, 2, "Access /etc?  (step 1 of 2)"),
        (2, 2, "Edit /etc/hosts?  (step 2 of 2)"),
    ]


def test_rule_covered_steps_stay_silent_and_the_count_shrinks():
    strategy = PermissionStrategy()
    strategy.add_rule(
        None,
        ResourcePermission(permission="access_directory", resource="/etc"),
        ApprovalOption.ALLOW,
    )

    prompts = build_approval_prompts(TWO_STEP_EXECUTION, strategy)

    assert [(p.step, p.total_steps, p.question) for p in prompts] == [
        (1, 1, "Edit /etc/hosts?"),
    ]


def test_question_never_doubles_trailing_punctuation():
    strategy = PermissionStrategy()

    [prompt] = build_approval_prompts(DOTTED_EXECUTION, strategy)

    assert prompt.question == "Remove build artifacts?"


def test_a_subagent_gate_names_its_task_in_the_question():
    strategy = PermissionStrategy()

    [prompt] = build_approval_prompts(
        SUBAGENT_EXECUTION,
        strategy,
        main_conversation_id="c_main",
        subagent_labels={"c_child": "audit the docs"},
    )

    assert (prompt.question, prompt.conversation_id, prompt.conversation_label) == (
        "[faint]task · audit the docs —[/] Read /tmp/notes.txt?",
        "c_child",
        "audit the docs",
    )


def test_an_unlabelled_subagent_gate_falls_back_to_the_conversation_id():
    strategy = PermissionStrategy()

    [prompt] = build_approval_prompts(SUBAGENT_EXECUTION, strategy, main_conversation_id="c_main")

    assert (prompt.question, prompt.conversation_id, prompt.conversation_label) == (
        "[faint]task · c_child —[/] Read /tmp/notes.txt?",
        "c_child",
        None,
    )


def test_a_main_conversation_gate_carries_no_task_prefix():
    strategy = PermissionStrategy()

    [prompt] = build_approval_prompts(MAIN_EXECUTION, strategy, main_conversation_id="c_main")

    assert (prompt.question, prompt.conversation_id, prompt.conversation_label) == (
        "Read /tmp/notes.txt?",
        None,
        None,
    )


def test_to_state_maps_labels_and_the_esc_hint():
    strategy = PermissionStrategy()

    [prompt] = build_approval_prompts(READ_EXECUTION, strategy)

    assert approval_state(prompt) == vm.ApprovalState(
        question="Read /tmp/notes.txt?",
        options=[
            vm.ApprovalOption(label="Approve once"),
            vm.ApprovalOption(label="Approve always — all reads under /tmp/*"),
            vm.ApprovalOption(label=DENY_LABEL),
            vm.ApprovalOption(label=CANCEL_LABEL, key_hint="esc"),
        ],
        selected=0,
    )


def test_option_flags():
    strategy = PermissionStrategy()

    [prompt] = build_approval_prompts(READ_EXECUTION, strategy)

    assert [(option.kind, option.is_cancel, option.is_deny) for option in prompt.options] == [
        ("approve", False, False),
        ("always", False, False),
        ("deny", False, True),
        ("cancel", True, False),
    ]
