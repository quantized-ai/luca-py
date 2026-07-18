"""luca.agent.contrib.resource_permissions — resource-based tool approval.

A full-featured, rule-based `PermissionPolicy` built on
`luca.agent.contrib.simple_tool_registry`'s strategy contract: permission
modes (ASK / YOLO / AUTO), one ordered last-match-wins rule list (each
`ToolRule` binding a decision to a `(permission, resource)` pair —
optionally tool-scoped in STRICT match mode — plus `ToolKind` defaults),
answer-decoupled interactive approval (`ApprovalAnswer` verdicts —
approve/deny × once/always — over `AnswerOption` pair sets, resolved by
coverage rather than request addressing), and a
`ResourcePermissionToolMixin` so tools declare typed `PermissionRequest`s
serialized to the wire shape the strategy reads (stored under
`extras["approval_context"]`).
"""

from .mixin import (
    AnswerOption,
    PermissionRequest,
    ResourcePermission,
    ResourcePermissionToolMixin,
)
from .strategy import (
    AnswerDecision,
    AnswerScope,
    ApprovalAnswer,
    PermissionMatchMode,
    PermissionMode,
    PermissionStrategy,
    ToolKindRule,
    ToolRule,
)

__all__ = [
    "AnswerDecision",
    "AnswerOption",
    "AnswerScope",
    "ApprovalAnswer",
    "PermissionMatchMode",
    "PermissionMode",
    "PermissionRequest",
    "PermissionStrategy",
    "ResourcePermission",
    "ResourcePermissionToolMixin",
    "ToolKindRule",
    "ToolRule",
]
