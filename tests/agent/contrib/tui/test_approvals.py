"""`build_approval_prompts` — the pure approval-prompt policy.

Each test: a known execution + strategy, one call, full-object asserts on the
resulting `ApprovalPrompt`s (options carry fully-built `ApprovalAnswer`s).
"""

from luca.agent.contrib.resource_permissions import (
    AnswerDecision,
    AnswerOption,
    AnswerScope,
    ApprovalAnswer,
    PermissionStrategy,
    ResourcePermission,
)
from luca.agent.contrib.tui.approvals import (
    ABANDON_LABEL,
    ApprovalPrompt,
    PromptOption,
    build_approval_prompts,
)
from luca.agent.core.models import (
    ApprovalOption,
    ExecutionStatus,
    ToolCall,
    ToolExecution,
)

READ_EXECUTION = ToolExecution(
    id="te1", created_at=500,
    tool_call_id="tc1",
    raw_tool_call=ToolCall(id="tc1", name="read", arguments={"path": "/tmp/notes.txt"}),
    status=ExecutionStatus.PENDING,
    extras={"approval_context": {"requests": [{
        "resources": [{"permission": "read", "resource": "/tmp/notes.txt"}],
        "answer_options": [{
            "resource_permissions": [{"permission": "read", "resource": "/tmp/*"}],
            "metadata": {"preview": "Allow all reads under /tmp/*"},
        }],
        "metadata": {"preview": "Read /tmp/notes.txt"},
    }]}},
)

MATH_EXECUTION = ToolExecution(
    id="te2", created_at=500,
    tool_call_id="tc2",
    raw_tool_call=ToolCall(id="tc2", name="add", arguments={"a": 1, "b": 2}),
    status=ExecutionStatus.PENDING,
)

TWO_STEP_EXECUTION = ToolExecution(
    id="te3", created_at=500,
    tool_call_id="tc3",
    raw_tool_call=ToolCall(id="tc3", name="edit", arguments={"path": "/etc/hosts"}),
    status=ExecutionStatus.PENDING,
    extras={"approval_context": {"requests": [
        {
            "resources": [{"permission": "access_directory", "resource": "/etc"}],
            "metadata": {"preview": "Access /etc"},
        },
        {
            "resources": [{"permission": "edit", "resource": "/etc/hosts"}],
            "metadata": {"preview": "Edit /etc/hosts"},
        },
    ]}},
)


def test_resourced_request_builds_the_full_option_set():
    strategy = PermissionStrategy()

    prompts = build_approval_prompts(READ_EXECUTION, strategy)

    exact = AnswerOption(resource_permissions=[
        ResourcePermission(permission="read", resource="/tmp/notes.txt"),
    ])
    suggested = AnswerOption(
        resource_permissions=[
            ResourcePermission(permission="read", resource="/tmp/*"),
        ],
        metadata={"preview": "Allow all reads under /tmp/*"},
    )
    assert prompts == [ApprovalPrompt(
        tool_name="read",
        step=1, total_steps=1,
        resources=["read:/tmp/notes.txt"],
        preview="Read /tmp/notes.txt",
        options=[
            PromptOption(
                label="Approve once",
                answer=ApprovalAnswer(
                    answer_option=exact, decision=AnswerDecision.APPROVE,
                ),
            ),
            PromptOption(
                label="Allow all reads under /tmp/*",
                answer=ApprovalAnswer(
                    answer_option=suggested,
                    decision=AnswerDecision.APPROVE,
                    scope=AnswerScope.ALWAYS,
                ),
            ),
            PromptOption(
                label="Deny",
                answer=ApprovalAnswer(
                    answer_option=exact, decision=AnswerDecision.DENY,
                ),
            ),
            PromptOption(label=ABANDON_LABEL, answer=None),
        ],
    )]


def test_resourceless_tool_gets_a_synthesized_prompt():
    strategy = PermissionStrategy()

    prompts = build_approval_prompts(MATH_EXECUTION, strategy)

    exact = AnswerOption(resource_permissions=[
        ResourcePermission(permission="add"),
    ])
    assert prompts == [ApprovalPrompt(
        tool_name="add",
        step=1, total_steps=1,
        resources=["add"],
        preview="Run add",
        options=[
            PromptOption(
                label="Approve once",
                answer=ApprovalAnswer(
                    answer_option=exact, decision=AnswerDecision.APPROVE,
                ),
            ),
            PromptOption(
                label="Deny",
                answer=ApprovalAnswer(
                    answer_option=exact, decision=AnswerDecision.DENY,
                ),
            ),
            PromptOption(label=ABANDON_LABEL, answer=None),
        ],
    )]


def test_multi_step_context_yields_one_prompt_per_request():
    strategy = PermissionStrategy()

    prompts = build_approval_prompts(TWO_STEP_EXECUTION, strategy)

    assert [(p.step, p.total_steps, p.preview) for p in prompts] == [
        (1, 2, "Access /etc"),
        (2, 2, "Edit /etc/hosts"),
    ]


def test_rule_covered_steps_stay_silent():
    strategy = PermissionStrategy()
    strategy.add_rule(
        None,
        ResourcePermission(permission="access_directory", resource="/etc"),
        ApprovalOption.ALLOW,
    )

    prompts = build_approval_prompts(TWO_STEP_EXECUTION, strategy)

    assert [(p.step, p.total_steps, p.preview) for p in prompts] == [
        (1, 1, "Edit /etc/hosts"),
    ]


def test_option_flags():
    strategy = PermissionStrategy()

    [prompt] = build_approval_prompts(MATH_EXECUTION, strategy)

    assert [
        (option.is_abandon, option.is_deny) for option in prompt.options
    ] == [(False, False), (False, True), (True, False)]
