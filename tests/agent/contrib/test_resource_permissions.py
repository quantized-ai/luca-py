"""Self-scoped tests for `luca.agent.contrib.resource_permissions`.

Everything outside the package is an invariant: no runner, no session, no
engine. Each test is GIVEN a known strategy state (mode + match mode + rules
+ recorded answers) and a known `ToolExecution` literal, WHEN one call
(`decide` / `apply_answer` / `add_rule` / the mixin's `get_approval_context`),
THEN one asserted outcome. `ApprovalDecision.created_at` self-stamps
wall-clock, so decide() asserts read `(decision, metadata)` instead of the
full object.
"""

import pytest
from pydantic import BaseModel, ValidationError

from luca.agent.contrib.resource_permissions import (
    AnswerDecision,
    AnswerOption,
    AnswerScope,
    ApprovalAnswer,
    PermissionMatchMode,
    PermissionMode,
    PermissionRequest,
    PermissionStrategy,
    ResourcePermission,
    ResourcePermissionToolMixin,
    ToolKindRule,
    ToolRule,
)
from luca.agent.core import (
    ApprovalOption,
    ExecutionStatus,
    LLMConfig,
    Tool,
    ToolCall,
    ToolContext,
    ToolExecution,
    ToolKind,
    ToolSpec,
)

# ── execution literals (created_at=500, matching tests/agent/scenarios.py) ────
# The strategy reads the tool NAME from `raw_tool_call`, the KIND from the
# resolved `tool_spec` snapshot, and the approval requests — each a list of
# (permission, resource) pairs plus suggested answer options — from the
# `{"requests": [...]}` dict `SimpleToolRegistry` stores under
# `extras["approval_context"]`.


def _pair(permission: str, resource: str | None) -> dict:
    return {"permission": permission, "resource": resource}


READ_EXECUTION = ToolExecution(
    id="x_read",
    created_at=500,
    tool_call_id="c_read",
    raw_tool_call=ToolCall(
        id="c_read", name="read_file", arguments={"path": "/etc/hosts"},
    ),
    tool_spec=ToolSpec(name="read_file", tool_kind=ToolKind.READ),
    extras={
        "approval_context": {
            "requests": [
                {
                    "resources": [_pair("read", "/etc/hosts")],
                    "answer_options": [
                        {
                            "resource_permissions": [_pair("read", "/etc/*")],
                            "metadata": {"preview": "Approve all reads in /etc/*"},
                        },
                    ],
                    "metadata": {"preview": "Read file /etc/hosts"},
                },
            ],
        },
    },
    status=ExecutionStatus.PENDING,
)

SIBLING_EXECUTION = ToolExecution(
    id="x_sibling",
    created_at=500,
    tool_call_id="c_sibling",
    raw_tool_call=ToolCall(
        id="c_sibling", name="read_file", arguments={"path": "/etc/passwd"},
    ),
    tool_spec=ToolSpec(name="read_file", tool_kind=ToolKind.READ),
    extras={
        "approval_context": {
            "requests": [
                {
                    "resources": [_pair("read", "/etc/passwd")],
                    "answer_options": [],
                    "metadata": {"preview": "Read file /etc/passwd"},
                },
            ],
        },
    },
    status=ExecutionStatus.PENDING,
)

MULTI_RESOURCE_EXECUTION = ToolExecution(
    id="x_multi",
    created_at=500,
    tool_call_id="c_multi",
    raw_tool_call=ToolCall(
        id="c_multi", name="read_file",
        arguments={"paths": ["/etc/hosts", "/tmp/scratch"]},
    ),
    tool_spec=ToolSpec(name="read_file", tool_kind=ToolKind.READ),
    extras={
        "approval_context": {
            "requests": [
                {
                    "resources": [
                        _pair("read", "/etc/hosts"),
                        _pair("read", "/tmp/scratch"),
                    ],
                    "answer_options": [],
                    "metadata": {"preview": "Read two files"},
                },
            ],
        },
    },
    status=ExecutionStatus.PENDING,
)

SWAP_EXECUTION = ToolExecution(
    id="x_swap",
    created_at=500,
    tool_call_id="c_swap",
    raw_tool_call=ToolCall(
        id="c_swap", name="swap_files",
        arguments={"from": "/src/main.py", "to": "/src/new_main.py"},
    ),
    tool_spec=ToolSpec(name="swap_files", tool_kind=ToolKind.EDIT),
    extras={
        "approval_context": {
            "requests": [
                {
                    "resources": [_pair("read", "/src/main.py")],
                    "answer_options": [
                        {
                            "resource_permissions": [_pair("read", "/src/*")],
                            "metadata": {"preview": "Approve all reads in /src/*"},
                        },
                    ],
                    "metadata": {"preview": "Read /src/main.py"},
                },
                {
                    "resources": [_pair("write", "/src/new_main.py")],
                    "answer_options": [
                        {
                            "resource_permissions": [_pair("write", "/src/*")],
                            "metadata": {"preview": "Approve all writes in /src/*"},
                        },
                    ],
                    "metadata": {"preview": "Write /src/new_main.py"},
                },
            ],
        },
    },
    status=ExecutionStatus.PENDING,
)

RESOURCELESS_EXECUTION = ToolExecution(
    id="x_add",
    created_at=500,
    tool_call_id="c_add",
    raw_tool_call=ToolCall(id="c_add", name="add", arguments={"a": 1, "b": 2}),
    tool_spec=ToolSpec(name="add"),
    extras={},
    status=ExecutionStatus.PENDING,
)

EMPTY_REQUEST_EXECUTION = ToolExecution(
    id="x_empty",
    created_at=500,
    tool_call_id="c_empty",
    raw_tool_call=ToolCall(
        id="c_empty", name="apply_patch", arguments={"patch_text": "nope"},
    ),
    tool_spec=ToolSpec(name="apply_patch", tool_kind=ToolKind.EDIT),
    extras={
        "approval_context": {
            "requests": [
                {
                    "resources": [],
                    "answer_options": [],
                    "metadata": {"preview": "Apply patch (invalid patch text)"},
                },
            ],
        },
    },
    status=ExecutionStatus.PENDING,
)


# ── decide(): modes ───────────────────────────────────────────────────────────


async def test_ask_mode_without_rules_leaves_resourceful_call_pending():
    strategy = PermissionStrategy(mode=PermissionMode.ASK)
    decision = await strategy.decide(READ_EXECUTION)
    assert (decision.decision, decision.metadata) == (
        ApprovalOption.PENDING, {"via": "mode"},
    )


async def test_ask_mode_without_rules_leaves_resourceless_call_pending():
    strategy = PermissionStrategy(mode=PermissionMode.ASK)
    decision = await strategy.decide(RESOURCELESS_EXECUTION)
    assert (decision.decision, decision.metadata) == (
        ApprovalOption.PENDING, {"via": "mode"},
    )


async def test_yolo_mode_promotes_unresolved_to_allow():
    strategy = PermissionStrategy(mode=PermissionMode.YOLO)
    decision = await strategy.decide(READ_EXECUTION)
    assert (decision.decision, decision.metadata) == (
        ApprovalOption.ALLOW, {"via": "mode"},
    )


async def test_auto_mode_promotes_unresolved_to_allow():
    strategy = PermissionStrategy(mode=PermissionMode.AUTO)
    decision = await strategy.decide(RESOURCELESS_EXECUTION)
    assert (decision.decision, decision.metadata) == (
        ApprovalOption.ALLOW, {"via": "mode"},
    )


async def test_explicit_deny_rule_blocks_even_in_yolo_mode():
    strategy = PermissionStrategy(
        mode=PermissionMode.YOLO,
        rules=[
            ToolRule(
                resource_permission=ResourcePermission(
                    permission="read", resource="/etc/*",
                ),
                decision=ApprovalOption.DENY,
            ),
        ],
    )
    decision = await strategy.decide(READ_EXECUTION)
    assert (decision.decision, decision.metadata) == (
        ApprovalOption.DENY, {"via": "rule"},
    )


# ── decide(): rule matching ───────────────────────────────────────────────────


async def test_tool_rule_glob_allows_matching_pair():
    strategy = PermissionStrategy(
        mode=PermissionMode.ASK,
        rules=[
            ToolRule(
                resource_permission=ResourcePermission(
                    permission="read", resource="/etc/*",
                ),
                decision=ApprovalOption.ALLOW,
            ),
        ],
    )
    decision = await strategy.decide(READ_EXECUTION)
    assert (decision.decision, decision.metadata) == (
        ApprovalOption.ALLOW, {"via": "rule"},
    )


async def test_tool_rule_requires_matching_permission():
    strategy = PermissionStrategy(
        mode=PermissionMode.ASK,
        rules=[
            ToolRule(
                resource_permission=ResourcePermission(
                    permission="write", resource="/etc/*",
                ),
                decision=ApprovalOption.ALLOW,
            ),
        ],
    )
    decision = await strategy.decide(READ_EXECUTION)
    assert (decision.decision, decision.metadata) == (
        ApprovalOption.PENDING, {"via": "mode"},
    )


async def test_relaxed_match_mode_ignores_the_rule_tool_name():
    strategy = PermissionStrategy(
        mode=PermissionMode.ASK,
        rules=[
            ToolRule(
                tool_name="another_tool",
                resource_permission=ResourcePermission(
                    permission="read", resource="/etc/*",
                ),
                decision=ApprovalOption.ALLOW,
            ),
        ],
    )
    decision = await strategy.decide(READ_EXECUTION)
    assert (decision.decision, decision.metadata) == (
        ApprovalOption.ALLOW, {"via": "rule"},
    )


async def test_strict_match_mode_requires_the_rule_tool_name():
    strategy = PermissionStrategy(
        mode=PermissionMode.ASK,
        permission_mode=PermissionMatchMode.STRICT,
        rules=[
            ToolRule(
                tool_name="another_tool",
                resource_permission=ResourcePermission(
                    permission="read", resource="/etc/*",
                ),
                decision=ApprovalOption.ALLOW,
            ),
        ],
    )
    decision = await strategy.decide(READ_EXECUTION)
    assert (decision.decision, decision.metadata) == (
        ApprovalOption.PENDING, {"via": "mode"},
    )


async def test_strict_match_mode_matches_the_calling_tool():
    strategy = PermissionStrategy(
        mode=PermissionMode.ASK,
        permission_mode=PermissionMatchMode.STRICT,
        rules=[
            ToolRule(
                tool_name="read_file",
                resource_permission=ResourcePermission(
                    permission="read", resource="/etc/*",
                ),
                decision=ApprovalOption.ALLOW,
            ),
        ],
    )
    decision = await strategy.decide(READ_EXECUTION)
    assert (decision.decision, decision.metadata) == (
        ApprovalOption.ALLOW, {"via": "rule"},
    )


async def test_strict_match_mode_treats_none_tool_name_as_any_tool():
    strategy = PermissionStrategy(
        mode=PermissionMode.ASK,
        permission_mode=PermissionMatchMode.STRICT,
        rules=[
            ToolRule(
                resource_permission=ResourcePermission(
                    permission="read", resource="/etc/*",
                ),
                decision=ApprovalOption.ALLOW,
            ),
        ],
    )
    decision = await strategy.decide(READ_EXECUTION)
    assert (decision.decision, decision.metadata) == (
        ApprovalOption.ALLOW, {"via": "rule"},
    )


async def test_last_matching_rule_wins():
    strategy = PermissionStrategy(
        mode=PermissionMode.ASK,
        rules=[
            ToolRule(
                resource_permission=ResourcePermission(
                    permission="read", resource="/etc/*",
                ),
                decision=ApprovalOption.ALLOW,
            ),
            ToolRule(
                resource_permission=ResourcePermission(
                    permission="read", resource="/etc/hosts",
                ),
                decision=ApprovalOption.DENY,
            ),
        ],
    )
    decision = await strategy.decide(READ_EXECUTION)
    assert (decision.decision, decision.metadata) == (
        ApprovalOption.DENY, {"via": "rule"},
    )


async def test_tool_kind_rule_matches_by_kind():
    strategy = PermissionStrategy(
        mode=PermissionMode.ASK,
        rules=[ToolKindRule(tool_kind=ToolKind.READ, decision=ApprovalOption.ALLOW)],
    )
    decision = await strategy.decide(READ_EXECUTION)
    assert (decision.decision, decision.metadata) == (
        ApprovalOption.ALLOW, {"via": "kind_default"},
    )


async def test_resourceless_call_matches_a_none_resource_rule_by_tool_name():
    strategy = PermissionStrategy(
        mode=PermissionMode.ASK,
        rules=[
            ToolRule(
                resource_permission=ResourcePermission(permission="add"),
                decision=ApprovalOption.ALLOW,
            ),
        ],
    )
    decision = await strategy.decide(RESOURCELESS_EXECUTION)
    assert (decision.decision, decision.metadata) == (
        ApprovalOption.ALLOW, {"via": "rule"},
    )


async def test_resourceless_rule_does_not_cover_resourceful_call():
    strategy = PermissionStrategy(
        mode=PermissionMode.ASK,
        rules=[
            ToolRule(
                resource_permission=ResourcePermission(permission="read"),
                decision=ApprovalOption.ALLOW,
            ),
        ],
    )
    decision = await strategy.decide(READ_EXECUTION)
    assert (decision.decision, decision.metadata) == (
        ApprovalOption.PENDING, {"via": "mode"},
    )


async def test_glob_rule_does_not_cover_resourceless_call():
    strategy = PermissionStrategy(
        mode=PermissionMode.ASK,
        rules=[
            ToolRule(
                resource_permission=ResourcePermission(
                    permission="add", resource="*",
                ),
                decision=ApprovalOption.ALLOW,
            ),
        ],
    )
    decision = await strategy.decide(RESOURCELESS_EXECUTION)
    assert (decision.decision, decision.metadata) == (
        ApprovalOption.PENDING, {"via": "mode"},
    )


async def test_empty_resources_request_degrades_to_the_tool_name_pair():
    strategy = PermissionStrategy(
        mode=PermissionMode.ASK,
        rules=[
            ToolRule(
                resource_permission=ResourcePermission(permission="apply_patch"),
                decision=ApprovalOption.ALLOW,
            ),
        ],
    )
    decision = await strategy.decide(EMPTY_REQUEST_EXECUTION)
    assert (decision.decision, decision.metadata) == (
        ApprovalOption.ALLOW, {"via": "rule"},
    )


# ── decide(): aggregation (DENY > PENDING > ALLOW over the flat pair list) ────


async def test_deny_on_one_pair_beats_allow_on_another():
    strategy = PermissionStrategy(
        mode=PermissionMode.ASK,
        rules=[
            ToolRule(
                resource_permission=ResourcePermission(
                    permission="read", resource="*",
                ),
                decision=ApprovalOption.ALLOW,
            ),
            ToolRule(
                resource_permission=ResourcePermission(
                    permission="read", resource="/tmp/*",
                ),
                decision=ApprovalOption.DENY,
            ),
        ],
    )
    decision = await strategy.decide(MULTI_RESOURCE_EXECUTION)
    assert (decision.decision, decision.metadata) == (
        ApprovalOption.DENY, {"via": "rule"},
    )


async def test_one_unmatched_pair_keeps_the_call_pending():
    strategy = PermissionStrategy(
        mode=PermissionMode.ASK,
        rules=[
            ToolRule(
                resource_permission=ResourcePermission(
                    permission="read", resource="/etc/*",
                ),
                decision=ApprovalOption.ALLOW,
            ),
        ],
    )
    decision = await strategy.decide(MULTI_RESOURCE_EXECUTION)
    assert (decision.decision, decision.metadata) == (
        ApprovalOption.PENDING, {"via": "mode"},
    )


async def test_covering_only_one_request_keeps_the_call_pending():
    strategy = PermissionStrategy(
        mode=PermissionMode.ASK,
        rules=[
            ToolRule(
                resource_permission=ResourcePermission(
                    permission="read", resource="/src/*",
                ),
                decision=ApprovalOption.ALLOW,
            ),
        ],
    )
    decision = await strategy.decide(SWAP_EXECUTION)
    assert (decision.decision, decision.metadata) == (
        ApprovalOption.PENDING, {"via": "mode"},
    )


async def test_rules_covering_every_request_allow_the_call():
    strategy = PermissionStrategy(
        mode=PermissionMode.ASK,
        rules=[
            ToolRule(
                resource_permission=ResourcePermission(
                    permission="read", resource="/src/*",
                ),
                decision=ApprovalOption.ALLOW,
            ),
            ToolRule(
                resource_permission=ResourcePermission(
                    permission="write", resource="/src/*",
                ),
                decision=ApprovalOption.ALLOW,
            ),
        ],
    )
    decision = await strategy.decide(SWAP_EXECUTION)
    assert (decision.decision, decision.metadata) == (
        ApprovalOption.ALLOW, {"via": "rule"},
    )


async def test_deny_rule_on_one_request_denies_the_call():
    strategy = PermissionStrategy(
        mode=PermissionMode.ASK,
        rules=[
            ToolRule(
                resource_permission=ResourcePermission(
                    permission="read", resource="*",
                ),
                decision=ApprovalOption.ALLOW,
            ),
            ToolRule(
                resource_permission=ResourcePermission(
                    permission="write", resource="*",
                ),
                decision=ApprovalOption.DENY,
            ),
        ],
    )
    decision = await strategy.decide(SWAP_EXECUTION)
    assert (decision.decision, decision.metadata) == (
        ApprovalOption.DENY, {"via": "rule"},
    )


# ── apply_answer(): scopes and decisions ──────────────────────────────────────


async def test_approve_once_with_the_exact_option_allows_the_call():
    strategy = PermissionStrategy(mode=PermissionMode.ASK)
    strategy.apply_answer(READ_EXECUTION, [ApprovalAnswer(
        answer_option=AnswerOption(resource_permissions=[
            ResourcePermission(permission="read", resource="/etc/hosts"),
        ]),
        decision=AnswerDecision.APPROVE,
        scope=AnswerScope.ONCE,
    )])
    decision = await strategy.decide(READ_EXECUTION)
    assert (decision.decision, decision.metadata) == (
        ApprovalOption.ALLOW, {"via": "user"},
    )


def test_approve_once_writes_no_rule():
    strategy = PermissionStrategy(mode=PermissionMode.ASK)
    strategy.apply_answer(READ_EXECUTION, [ApprovalAnswer(
        answer_option=AnswerOption(resource_permissions=[
            ResourcePermission(permission="read", resource="/etc/hosts"),
        ]),
        decision=AnswerDecision.APPROVE,
    )])
    assert strategy.rules == []


async def test_approve_once_does_not_cover_a_sibling_call():
    strategy = PermissionStrategy(mode=PermissionMode.ASK)
    strategy.apply_answer(READ_EXECUTION, [ApprovalAnswer(
        answer_option=AnswerOption(resource_permissions=[
            ResourcePermission(permission="read", resource="/etc/hosts"),
        ]),
        decision=AnswerDecision.APPROVE,
    )])
    decision = await strategy.decide(SIBLING_EXECUTION)
    assert (decision.decision, decision.metadata) == (
        ApprovalOption.PENDING, {"via": "mode"},
    )


async def test_approve_once_with_a_glob_option_covers_the_requirement():
    strategy = PermissionStrategy(mode=PermissionMode.ASK)
    strategy.apply_answer(READ_EXECUTION, [ApprovalAnswer(
        answer_option=AnswerOption(resource_permissions=[
            ResourcePermission(permission="read", resource="/etc/*"),
        ]),
        decision=AnswerDecision.APPROVE,
    )])
    decision = await strategy.decide(READ_EXECUTION)
    assert (decision.decision, decision.metadata) == (
        ApprovalOption.ALLOW, {"via": "user"},
    )


async def test_approve_once_with_a_glob_option_still_does_not_leak_to_a_sibling():
    strategy = PermissionStrategy(mode=PermissionMode.ASK)
    strategy.apply_answer(READ_EXECUTION, [ApprovalAnswer(
        answer_option=AnswerOption(resource_permissions=[
            ResourcePermission(permission="read", resource="/etc/*"),
        ]),
        decision=AnswerDecision.APPROVE,
    )])
    decision = await strategy.decide(SIBLING_EXECUTION)
    assert (decision.decision, decision.metadata) == (
        ApprovalOption.PENDING, {"via": "mode"},
    )


async def test_approve_once_covering_one_pair_leaves_the_call_pending():
    strategy = PermissionStrategy(mode=PermissionMode.ASK)
    strategy.apply_answer(SWAP_EXECUTION, [ApprovalAnswer(
        answer_option=AnswerOption(resource_permissions=[
            ResourcePermission(permission="read", resource="/src/main.py"),
        ]),
        decision=AnswerDecision.APPROVE,
    )])
    decision = await strategy.decide(SWAP_EXECUTION)
    assert (decision.decision, decision.metadata) == (
        ApprovalOption.PENDING, {"via": "mode"},
    )


async def test_answers_covering_every_pair_in_one_call_allow_the_call():
    strategy = PermissionStrategy(mode=PermissionMode.ASK)
    strategy.apply_answer(SWAP_EXECUTION, [
        ApprovalAnswer(
            answer_option=AnswerOption(resource_permissions=[
                ResourcePermission(permission="read", resource="/src/main.py"),
            ]),
            decision=AnswerDecision.APPROVE,
        ),
        ApprovalAnswer(
            answer_option=AnswerOption(resource_permissions=[
                ResourcePermission(permission="write", resource="/src/new_main.py"),
            ]),
            decision=AnswerDecision.APPROVE,
        ),
    ])
    decision = await strategy.decide(SWAP_EXECUTION)
    assert (decision.decision, decision.metadata) == (
        ApprovalOption.ALLOW, {"via": "user"},
    )


async def test_answers_accumulate_across_apply_answer_calls():
    strategy = PermissionStrategy(mode=PermissionMode.ASK)
    strategy.apply_answer(SWAP_EXECUTION, [ApprovalAnswer(
        answer_option=AnswerOption(resource_permissions=[
            ResourcePermission(permission="read", resource="/src/main.py"),
        ]),
        decision=AnswerDecision.APPROVE,
    )])
    strategy.apply_answer(SWAP_EXECUTION, [ApprovalAnswer(
        answer_option=AnswerOption(resource_permissions=[
            ResourcePermission(permission="write", resource="/src/new_main.py"),
        ]),
        decision=AnswerDecision.APPROVE,
    )])
    decision = await strategy.decide(SWAP_EXECUTION)
    assert (decision.decision, decision.metadata) == (
        ApprovalOption.ALLOW, {"via": "user"},
    )


async def test_deny_once_denies_the_call():
    strategy = PermissionStrategy(mode=PermissionMode.ASK)
    strategy.apply_answer(READ_EXECUTION, [ApprovalAnswer(
        answer_option=AnswerOption(resource_permissions=[
            ResourcePermission(permission="read", resource="/etc/hosts"),
        ]),
        decision=AnswerDecision.DENY,
    )])
    decision = await strategy.decide(READ_EXECUTION)
    assert (decision.decision, decision.metadata) == (
        ApprovalOption.DENY, {"via": "user"},
    )


async def test_a_deny_verdict_beats_a_coexisting_allow_verdict_on_the_same_pair():
    strategy = PermissionStrategy(mode=PermissionMode.ASK)
    strategy.apply_answer(READ_EXECUTION, [
        ApprovalAnswer(
            answer_option=AnswerOption(resource_permissions=[
                ResourcePermission(permission="read", resource="/etc/hosts"),
            ]),
            decision=AnswerDecision.APPROVE,
        ),
        ApprovalAnswer(
            answer_option=AnswerOption(resource_permissions=[
                ResourcePermission(permission="read", resource="/etc/hosts"),
            ]),
            decision=AnswerDecision.DENY,
        ),
    ])
    decision = await strategy.decide(READ_EXECUTION)
    assert (decision.decision, decision.metadata) == (
        ApprovalOption.DENY, {"via": "user"},
    )


async def test_an_explicit_deny_pins_the_call_against_a_rule_from_the_same_pause():
    strategy = PermissionStrategy(mode=PermissionMode.ASK)
    strategy.apply_answer(SIBLING_EXECUTION, [ApprovalAnswer(
        answer_option=AnswerOption(resource_permissions=[
            ResourcePermission(permission="read", resource="/etc/passwd"),
        ]),
        decision=AnswerDecision.DENY,
    )])
    strategy.apply_answer(READ_EXECUTION, [ApprovalAnswer(
        answer_option=AnswerOption(resource_permissions=[
            ResourcePermission(permission="read", resource="/etc/*"),
        ]),
        decision=AnswerDecision.APPROVE,
        scope=AnswerScope.ALWAYS,
    )])
    sibling = await strategy.decide(SIBLING_EXECUTION)
    read = await strategy.decide(READ_EXECUTION)
    assert (
        (sibling.decision, sibling.metadata),
        (read.decision, read.metadata),
    ) == (
        (ApprovalOption.DENY, {"via": "user"}),
        (ApprovalOption.ALLOW, {"via": "rule"}),
    )


def test_approve_always_writes_an_allow_rule_per_pair_of_the_option():
    strategy = PermissionStrategy(mode=PermissionMode.ASK)
    strategy.apply_answer(MULTI_RESOURCE_EXECUTION, [ApprovalAnswer(
        answer_option=AnswerOption(resource_permissions=[
            ResourcePermission(permission="read", resource="/etc/hosts"),
            ResourcePermission(permission="read", resource="/tmp/scratch"),
        ]),
        decision=AnswerDecision.APPROVE,
        scope=AnswerScope.ALWAYS,
    )])
    assert strategy.rules == [
        ToolRule(
            tool_name="read_file",
            resource_permission=ResourcePermission(
                permission="read", resource="/etc/hosts",
            ),
            decision=ApprovalOption.ALLOW,
        ),
        ToolRule(
            tool_name="read_file",
            resource_permission=ResourcePermission(
                permission="read", resource="/tmp/scratch",
            ),
            decision=ApprovalOption.ALLOW,
        ),
    ]


async def test_approve_always_resolves_the_call_via_the_written_rule():
    strategy = PermissionStrategy(mode=PermissionMode.ASK)
    strategy.apply_answer(READ_EXECUTION, [ApprovalAnswer(
        answer_option=AnswerOption(resource_permissions=[
            ResourcePermission(permission="read", resource="/etc/*"),
        ]),
        decision=AnswerDecision.APPROVE,
        scope=AnswerScope.ALWAYS,
    )])
    decision = await strategy.decide(READ_EXECUTION)
    assert (decision.decision, decision.metadata) == (
        ApprovalOption.ALLOW, {"via": "rule"},
    )


async def test_deny_always_writes_a_deny_rule_that_blocks_the_call():
    strategy = PermissionStrategy(mode=PermissionMode.ASK)
    strategy.apply_answer(READ_EXECUTION, [ApprovalAnswer(
        answer_option=AnswerOption(resource_permissions=[
            ResourcePermission(permission="read", resource="/etc/*"),
        ]),
        decision=AnswerDecision.DENY,
        scope=AnswerScope.ALWAYS,
    )])
    decision = await strategy.decide(READ_EXECUTION)
    assert strategy.rules == [
        ToolRule(
            tool_name="read_file",
            resource_permission=ResourcePermission(
                permission="read", resource="/etc/*",
            ),
            decision=ApprovalOption.DENY,
        ),
    ]
    assert (decision.decision, decision.metadata) == (
        ApprovalOption.DENY, {"via": "rule"},
    )


async def test_deny_always_blocks_even_in_yolo_mode():
    strategy = PermissionStrategy(mode=PermissionMode.YOLO)
    strategy.apply_answer(READ_EXECUTION, [ApprovalAnswer(
        answer_option=AnswerOption(resource_permissions=[
            ResourcePermission(permission="read", resource="/etc/*"),
        ]),
        decision=AnswerDecision.DENY,
        scope=AnswerScope.ALWAYS,
    )])
    decision = await strategy.decide(READ_EXECUTION)
    assert (decision.decision, decision.metadata) == (
        ApprovalOption.DENY, {"via": "rule"},
    )


async def test_approve_always_on_a_resourceless_call_writes_a_none_resource_rule():
    strategy = PermissionStrategy(mode=PermissionMode.ASK)
    strategy.apply_answer(RESOURCELESS_EXECUTION, [ApprovalAnswer(
        answer_option=AnswerOption(resource_permissions=[
            ResourcePermission(permission="add"),
        ]),
        decision=AnswerDecision.APPROVE,
        scope=AnswerScope.ALWAYS,
    )])
    decision = await strategy.decide(RESOURCELESS_EXECUTION)
    assert strategy.rules == [
        ToolRule(
            tool_name="add",
            resource_permission=ResourcePermission(permission="add", resource=None),
            decision=ApprovalOption.ALLOW,
        ),
    ]
    assert (decision.decision, decision.metadata) == (
        ApprovalOption.ALLOW, {"via": "rule"},
    )


async def test_a_non_covering_answer_leaves_the_call_pending():
    strategy = PermissionStrategy(mode=PermissionMode.ASK)
    strategy.apply_answer(READ_EXECUTION, [ApprovalAnswer(
        answer_option=AnswerOption(resource_permissions=[
            ResourcePermission(permission="read", resource="/var/*"),
        ]),
        decision=AnswerDecision.APPROVE,
    )])
    decision = await strategy.decide(READ_EXECUTION)
    assert (decision.decision, decision.metadata) == (
        ApprovalOption.PENDING, {"via": "mode"},
    )


async def test_a_custom_option_the_tool_never_emitted_is_accepted():
    strategy = PermissionStrategy(mode=PermissionMode.ASK)
    strategy.apply_answer(READ_EXECUTION, [ApprovalAnswer(
        answer_option=AnswerOption(resource_permissions=[
            ResourcePermission(permission="read", resource="*"),
        ]),
        decision=AnswerDecision.APPROVE,
    )])
    decision = await strategy.decide(READ_EXECUTION)
    assert (decision.decision, decision.metadata) == (
        ApprovalOption.ALLOW, {"via": "user"},
    )


# ── models ────────────────────────────────────────────────────────────────────


def test_approval_answer_scope_defaults_to_once():
    answer = ApprovalAnswer(
        answer_option=AnswerOption(resource_permissions=[
            ResourcePermission(permission="read", resource="/etc/hosts"),
        ]),
        decision=AnswerDecision.APPROVE,
    )
    assert answer.scope == AnswerScope.ONCE


def test_tool_rule_rejects_a_pending_decision():
    with pytest.raises(ValidationError):
        ToolRule(
            resource_permission=ResourcePermission(permission="read"),
            decision=ApprovalOption.PENDING,
        )


def test_add_rule_dedupes_identical_rules():
    strategy = PermissionStrategy(mode=PermissionMode.ASK)
    strategy.add_rule(
        "read_file",
        ResourcePermission(permission="read", resource="/etc/*"),
        ApprovalOption.ALLOW,
    )
    strategy.add_rule(
        "read_file",
        ResourcePermission(permission="read", resource="/etc/*"),
        ApprovalOption.ALLOW,
    )
    assert strategy.rules == [
        ToolRule(
            tool_name="read_file",
            resource_permission=ResourcePermission(
                permission="read", resource="/etc/*",
            ),
            decision=ApprovalOption.ALLOW,
        ),
    ]


def test_extra_fields_are_forbidden():
    with pytest.raises(ValidationError):
        ResourcePermission(permission="read", resource="/etc/hosts", preview="nope")


# ── the tool mixin ────────────────────────────────────────────────────────────


class PathArgs(BaseModel):
    path: str


class SwapArgs(BaseModel):
    source: str
    target: str


class StubReadTool(ResourcePermissionToolMixin, Tool):
    name = "read_file"
    description = "Read a file."
    Args = PathArgs
    tool_kind = ToolKind.READ

    def build_permission_requests(
        self, args: dict, context: ToolContext,
    ) -> list[PermissionRequest]:
        return [PermissionRequest(
            resources=[
                ResourcePermission(permission="read", resource=args["path"]),
            ],
            answer_options=[
                AnswerOption(
                    resource_permissions=[
                        ResourcePermission(permission="read", resource="/etc/*"),
                    ],
                    metadata={"preview": "Approve all reads in /etc/*"},
                ),
            ],
            metadata={"preview": f"Read {args['path']}"},
        )]


class StubSwapTool(ResourcePermissionToolMixin, Tool):
    name = "swap_files"
    description = "Swap two files."
    Args = SwapArgs
    tool_kind = ToolKind.EDIT

    def build_permission_requests(
        self, args: dict, context: ToolContext,
    ) -> list[PermissionRequest]:
        return [
            PermissionRequest(
                resources=[
                    ResourcePermission(permission="read", resource=args["source"]),
                ],
                metadata={"preview": f"Read {args['source']}"},
            ),
            PermissionRequest(
                resources=[
                    ResourcePermission(permission="write", resource=args["target"]),
                ],
                metadata={"preview": f"Write {args['target']}"},
            ),
        ]


class NoOptionsTool(ResourcePermissionToolMixin, Tool):
    name = "list_files"
    description = "List files."
    Args = PathArgs
    tool_kind = ToolKind.READ

    def build_permission_requests(
        self, args: dict, context: ToolContext,
    ) -> list[PermissionRequest]:
        return [PermissionRequest(
            resources=[
                ResourcePermission(permission="list", resource=args["path"]),
            ],
        )]


class UnimplementedTool(ResourcePermissionToolMixin, Tool):
    name = "broken"
    description = "Forgot to implement build_permission_requests."
    Args = PathArgs


def tool_context() -> ToolContext:
    return ToolContext(
        session_id="s1",
        model=LLMConfig(model="test-model", provider="faux"),
    )


async def test_mixin_serializes_the_typed_requests_to_the_wire_dict():
    context = await StubReadTool().get_approval_context(
        {"path": "/etc/hosts"}, tool_context(),
    )
    assert context == {
        "requests": [
            {
                "resources": [{"permission": "read", "resource": "/etc/hosts"}],
                "answer_options": [
                    {
                        "resource_permissions": [
                            {"permission": "read", "resource": "/etc/*"},
                        ],
                        "metadata": {"preview": "Approve all reads in /etc/*"},
                    },
                ],
                "metadata": {"preview": "Read /etc/hosts"},
            },
        ],
    }


async def test_mixin_preserves_the_request_order():
    context = await StubSwapTool().get_approval_context(
        {"source": "/src/a.py", "target": "/src/b.py"}, tool_context(),
    )
    assert [request["metadata"]["preview"] for request in context["requests"]] == [
        "Read /src/a.py", "Write /src/b.py",
    ]


async def test_mixin_defaults_answer_options_and_metadata_in_the_dump():
    context = await NoOptionsTool().get_approval_context(
        {"path": "/srv"}, tool_context(),
    )
    assert context == {
        "requests": [
            {
                "resources": [{"permission": "list", "resource": "/srv"}],
                "answer_options": [],
                "metadata": {},
            },
        ],
    }


async def test_mixin_without_build_permission_requests_raises():
    with pytest.raises(NotImplementedError):
        await UnimplementedTool().get_approval_context(
            {"path": "/etc/hosts"}, tool_context(),
        )


async def test_mixin_output_hydrates_and_drives_the_strategy():
    context = await StubSwapTool().get_approval_context(
        {"source": "/src/a.py", "target": "/src/b.py"}, tool_context(),
    )
    execution = ToolExecution(
        id="x_mixin",
        created_at=500,
        tool_call_id="c_mixin",
        raw_tool_call=ToolCall(
            id="c_mixin", name="swap_files",
            arguments={"source": "/src/a.py", "target": "/src/b.py"},
        ),
        tool_spec=StubSwapTool().get_tool_spec(),
        extras={"approval_context": context},
        status=ExecutionStatus.PENDING,
    )
    strategy = PermissionStrategy(mode=PermissionMode.ASK)
    strategy.add_rule(
        "swap_files",
        ResourcePermission(permission="read", resource="/src/*"),
        ApprovalOption.ALLOW,
    )
    strategy.add_rule(
        "swap_files",
        ResourcePermission(permission="write", resource="/src/*"),
        ApprovalOption.ALLOW,
    )
    decision = await strategy.decide(execution)
    assert strategy.permission_requests(execution) == StubSwapTool(
    ).build_permission_requests(
        {"source": "/src/a.py", "target": "/src/b.py"}, tool_context(),
    )
    assert (decision.decision, decision.metadata) == (
        ApprovalOption.ALLOW, {"via": "rule"},
    )


# ── permission_requests() ─────────────────────────────────────────────────────


def test_permission_requests_is_empty_for_a_resourceless_call():
    strategy = PermissionStrategy(mode=PermissionMode.ASK)
    assert strategy.permission_requests(RESOURCELESS_EXECUTION) == []


def test_permission_requests_returns_the_hydrated_typed_models():
    strategy = PermissionStrategy(mode=PermissionMode.ASK)
    assert strategy.permission_requests(SWAP_EXECUTION) == [
        PermissionRequest(
            resources=[
                ResourcePermission(permission="read", resource="/src/main.py"),
            ],
            answer_options=[
                AnswerOption(
                    resource_permissions=[
                        ResourcePermission(permission="read", resource="/src/*"),
                    ],
                    metadata={"preview": "Approve all reads in /src/*"},
                ),
            ],
            metadata={"preview": "Read /src/main.py"},
        ),
        PermissionRequest(
            resources=[
                ResourcePermission(permission="write", resource="/src/new_main.py"),
            ],
            answer_options=[
                AnswerOption(
                    resource_permissions=[
                        ResourcePermission(permission="write", resource="/src/*"),
                    ],
                    metadata={"preview": "Approve all writes in /src/*"},
                ),
            ],
            metadata={"preview": "Write /src/new_main.py"},
        ),
    ]


# ── pending_requests() ────────────────────────────────────────────────────────


def test_pending_requests_returns_every_request_when_nothing_is_covered():
    strategy = PermissionStrategy(mode=PermissionMode.ASK)
    assert strategy.pending_requests(SWAP_EXECUTION) == (
        strategy.permission_requests(SWAP_EXECUTION)
    )


def test_pending_requests_drops_a_rule_covered_request():
    strategy = PermissionStrategy(
        mode=PermissionMode.ASK,
        rules=[
            ToolRule(
                resource_permission=ResourcePermission(
                    permission="read", resource="/src/*",
                ),
                decision=ApprovalOption.ALLOW,
            ),
        ],
    )
    assert strategy.pending_requests(SWAP_EXECUTION) == [
        strategy.permission_requests(SWAP_EXECUTION)[1],
    ]


def test_pending_requests_drops_a_deny_covered_request():
    strategy = PermissionStrategy(
        mode=PermissionMode.ASK,
        rules=[
            ToolRule(
                resource_permission=ResourcePermission(
                    permission="read", resource="/src/*",
                ),
                decision=ApprovalOption.DENY,
            ),
        ],
    )
    assert strategy.pending_requests(SWAP_EXECUTION) == [
        strategy.permission_requests(SWAP_EXECUTION)[1],
    ]


def test_pending_requests_is_empty_when_rules_cover_everything():
    strategy = PermissionStrategy(
        mode=PermissionMode.ASK,
        rules=[
            ToolRule(
                resource_permission=ResourcePermission(
                    permission="read", resource="/src/*",
                ),
                decision=ApprovalOption.ALLOW,
            ),
            ToolRule(
                resource_permission=ResourcePermission(
                    permission="write", resource="/src/*",
                ),
                decision=ApprovalOption.ALLOW,
            ),
        ],
    )
    assert strategy.pending_requests(SWAP_EXECUTION) == []


def test_pending_requests_is_empty_in_yolo_mode():
    strategy = PermissionStrategy(mode=PermissionMode.YOLO)
    assert strategy.pending_requests(SWAP_EXECUTION) == []


def test_pending_requests_keeps_only_the_unresolved_pairs_of_a_request():
    strategy = PermissionStrategy(
        mode=PermissionMode.ASK,
        rules=[
            ToolRule(
                resource_permission=ResourcePermission(
                    permission="read", resource="/etc/*",
                ),
                decision=ApprovalOption.ALLOW,
            ),
        ],
    )
    assert strategy.pending_requests(MULTI_RESOURCE_EXECUTION) == [
        PermissionRequest(
            resources=[
                ResourcePermission(permission="read", resource="/tmp/scratch"),
            ],
            answer_options=[],
            metadata={"preview": "Read two files"},
        ),
    ]


def test_pending_requests_drops_a_verdict_covered_request():
    strategy = PermissionStrategy(mode=PermissionMode.ASK)
    strategy.apply_answer(SWAP_EXECUTION, [
        ApprovalAnswer(
            answer_option=AnswerOption(resource_permissions=[
                ResourcePermission(permission="read", resource="/src/main.py"),
            ]),
            decision=AnswerDecision.APPROVE,
        ),
    ])
    assert strategy.pending_requests(SWAP_EXECUTION) == [
        strategy.permission_requests(SWAP_EXECUTION)[1],
    ]


def test_pending_requests_is_empty_for_a_resourceless_call():
    strategy = PermissionStrategy(mode=PermissionMode.ASK)
    assert strategy.pending_requests(RESOURCELESS_EXECUTION) == []


def test_pending_requests_keeps_an_empty_resources_request_while_pending():
    strategy = PermissionStrategy(mode=PermissionMode.ASK)
    assert strategy.pending_requests(EMPTY_REQUEST_EXECUTION) == (
        strategy.permission_requests(EMPTY_REQUEST_EXECUTION)
    )


def test_pending_requests_drops_an_empty_resources_request_covered_by_its_implicit_pair():
    strategy = PermissionStrategy(
        mode=PermissionMode.ASK,
        rules=[
            ToolRule(
                resource_permission=ResourcePermission(
                    permission="apply_patch", resource=None,
                ),
                decision=ApprovalOption.ALLOW,
            ),
        ],
    )
    assert strategy.pending_requests(EMPTY_REQUEST_EXECUTION) == []
