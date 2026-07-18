"""Rule-based, resource-aware permission strategy — modes, rules, answers.

Implements the `PermissionPolicy` strategy contract from
`luca.agent.contrib.simple_tool_registry`: a `SimpleToolRegistry` hands each
unresolved `ToolExecution` to `decide()` and the runner appends the returned
`ApprovalDecision`. Modes, rules, resource globs, interactive answers — that
whole vocabulary is application logic, packaged here.

The tools and this strategy communicate through the free-form approval
context stored under `ToolExecution.extras["approval_context"]` (emitted by
the tool's duck-typed `get_approval_context`, stored by `SimpleToolRegistry`,
never interpreted by the core). The convention this strategy reads:

    {
        "requests": [        # ordered approval steps; absent/[] = resource-less
            {
                "resources":      [{"permission": str, "resource": str | None}],
                "answer_options": [  # suggested grants, usually glob-valued
                    {"resource_permissions": [...], "metadata": {...}},
                ],
                "metadata": {...},   # UX-only; never read by the strategy
            },
        ],
    }

Tools shouldn't hand-build that dict — mix in `ResourcePermissionToolMixin`
(`mixin.py`) and return a list of typed `PermissionRequest`s instead; the
mixin serializes them to exactly this shape. Applications read the typed
models back via `permission_requests()`, never the raw dicts.

Answers are decoupled from requests: users answer with free-standing
verdicts (approve/deny × once/always) over `AnswerOption`s — the tool's
options are suggestions, not a closed set. Whether a call runs is emergent
from coverage: every required (permission, resource) pair (or the implicit
resource-less pair, whose permission is the tool name) must be covered by an
ALLOW verdict or rule; any DENY kills it; anything uncovered stays PENDING —
the runner pauses, the application prompts its user, the answers land back
here via `apply_answer()`, and the next `run()` asks `decide()` again.
"""

from __future__ import annotations

from enum import Enum
from fnmatch import fnmatchcase

from pydantic import BaseModel, ConfigDict, field_validator

from luca.agent.core.models import (
    ApprovalDecision,
    ApprovalOption,
    ToolExecution,
    ToolKind,
)
from luca.agent.contrib.simple_tool_registry import PermissionPolicy

from .mixin import AnswerOption, PermissionRequest, ResourcePermission


class PermissionMode(str, Enum):
    ASK = "ask"  # rules decide; anything unresolved asks the user
    YOLO = "yolo"  # promote unresolved to ALLOW (explicit DENY rules still block)
    AUTO = "auto"  # same promotion as YOLO (reserved for divergence later)


class PermissionMatchMode(str, Enum):
    RELAXED = "relaxed"  # rules match on (permission, resource); tool_name is ignored
    STRICT = "strict"  # a rule's tool_name (when set) must also match the calling tool


# ── rules (one unified, ordered list; LAST match wins) ───────────────────────


class ToolKindRule(BaseModel):
    """Resource-agnostic: applies to every call of a given ToolKind."""

    tool_kind: ToolKind
    decision: ApprovalOption

    model_config = ConfigDict(extra="forbid")


class ToolRule(BaseModel):
    """Binds a decision to a (permission, resource) pair, optionally scoped
    to a tool. The pair's `permission` matches exactly against each of the
    call's required pairs; its `resource` is matched against the pair's
    resource: a glob ("*", "/etc/*") matches a resourceful pair that
    fnmatches it; `None` matches only the resource-less invocation; mixed
    never matches (so "always allow `add`" with no resources won't suddenly
    cover an `add` that later reports one). `tool_name` only participates in
    STRICT match mode, where `None` means any tool; rules written by
    `apply_answer` always record it."""

    tool_name: str | None = None
    resource_permission: ResourcePermission
    decision: ApprovalOption

    model_config = ConfigDict(extra="forbid")

    @field_validator("decision")
    @classmethod
    def _no_pending(cls, value: ApprovalOption) -> ApprovalOption:
        if value == ApprovalOption.PENDING:
            raise ValueError("a ToolRule decision must be ALLOW or DENY")
        return value


# ── interactive answers (what the application posts back onto the strategy) ──


class AnswerDecision(str, Enum):
    APPROVE = "approve"
    DENY = "deny"


class AnswerScope(str, Enum):
    ONCE = "once"  # verdict scoped to the answered execution
    ALWAYS = "always"  # persistent rule for the strategy's lifetime


class ApprovalAnswer(BaseModel):
    """One free-standing verdict over an `AnswerOption`'s pairs. Answers are
    NOT replies to requests — no ids, no addressing; the tool's
    `answer_options` are suggestions and custom-built options are legal.
    Whether the call resolves is emergent: the next `decide()` checks whether
    every required pair is covered."""

    answer_option: AnswerOption
    decision: AnswerDecision
    scope: AnswerScope = AnswerScope.ONCE

    model_config = ConfigDict(extra="forbid")


# ── the strategy ──────────────────────────────────────────────────────────────


class PermissionStrategy(PermissionPolicy):
    """Rule-based, interactive permission strategy.

    `decide()` is a pure query of the strategy's state: recorded ephemeral
    verdicts first, then the rule list + mode. It returns PENDING for
    anything uncovered — the runner pauses and the app calls `apply_answer()`
    before running again. Because the runner re-asks `decide()` for every
    still-unresolved call, a newly written rule also clears matching
    siblings, and an answer that doesn't cover every required pair simply
    leaves the call PENDING (the approval loop)."""

    def __init__(
        self,
        mode: PermissionMode = PermissionMode.ASK,
        rules: list[ToolKindRule | ToolRule] | None = None,
        permission_mode: PermissionMatchMode = PermissionMatchMode.RELAXED,
    ) -> None:
        self.mode = mode
        self.permission_mode = permission_mode
        self.rules = list(rules or [])
        # execution id → ephemeral verdicts recorded by ONCE-scoped answers
        self._verdicts: dict[str, list[tuple[ResourcePermission, ApprovalOption]]] = {}

    def permission_requests(self, execution: ToolExecution) -> list[PermissionRequest]:
        """The typed approval requests stored on this execution ([] if none).
        Pure hydration for the app's UI (and for `decide` internally); order
        is presentation order."""
        stored = (execution.extras.get("approval_context") or {}).get("requests") or []
        return [PermissionRequest.model_validate(raw) for raw in stored]

    def pending_requests(self, execution: ToolExecution) -> list[PermissionRequest]:
        """The requests still awaiting the user: hydrated like
        `permission_requests` but filtered to the pairs `decide()` would
        leave PENDING. A request whose pairs are all covered (by verdicts,
        rules, or mode promotion) drops out; a partially covered one keeps
        only its unresolved pairs. This is the prompt feed for an
        interactive app — show these steps, not every stored one. [] for a
        requestless tool (there is nothing stored to hydrate)."""
        implicit = ResourcePermission(permission=execution.raw_tool_call.name)
        pending: list[PermissionRequest] = []
        for request in self.permission_requests(execution):
            pairs = request.resources or [implicit]
            unresolved = [
                pair for pair in pairs
                if self._resolve_pair(execution, pair)[0] == ApprovalOption.PENDING
            ]
            if not unresolved:
                continue
            if request.resources:
                pending.append(request.model_copy(update={"resources": unresolved}))
            else:
                pending.append(request)
        return pending

    async def decide(self, tool_execution: ToolExecution) -> ApprovalDecision:
        resolved = [
            self._resolve_pair(tool_execution, pair)
            for pair in self._required_pairs(tool_execution)
        ]
        decision, via = _aggregate(resolved)
        return ApprovalDecision(decision=decision, metadata={"via": via})

    def apply_answer(
        self, execution: ToolExecution, answers: list[ApprovalAnswer],
    ) -> None:
        """Record user answers for one pending execution. Answers are
        verdicts over pairs, NOT replies to requests: ALWAYS-scoped answers
        append rules (recording the calling tool's name); ONCE-scoped answers
        record verdicts for this execution only. Answers accumulate across
        calls; no membership or coverage validation — an answer that covers
        nothing is a no-op verdict and the call simply stays PENDING.
        Resolution is emergent — the next `decide()` checks coverage."""
        for answer in answers:
            decision = (
                ApprovalOption.ALLOW
                if answer.decision == AnswerDecision.APPROVE
                else ApprovalOption.DENY
            )
            for pair in answer.answer_option.resource_permissions:
                if answer.scope == AnswerScope.ALWAYS:
                    self.add_rule(execution.raw_tool_call.name, pair, decision)
                else:
                    self._verdicts.setdefault(execution.id, []).append((pair, decision))

    def add_rule(
        self,
        tool_name: str | None,
        resource_permission: ResourcePermission,
        decision: ApprovalOption,
    ) -> None:
        """Append a `ToolRule`, deduping identical rules."""
        rule = ToolRule(
            tool_name=tool_name,
            resource_permission=resource_permission,
            decision=decision,
        )
        if rule not in self.rules:
            self.rules.append(rule)

    # ── resolution internals ─────────────────────────────────────────────────

    def _resolve_pair(
        self, execution: ToolExecution, pair: ResourcePermission,
    ) -> tuple[ApprovalOption, str]:
        """One pair's (decision, via): ephemeral verdicts first, then the
        rule list with the mode's promotion applied."""
        verdict = self._match_verdicts(execution.id, pair)
        if verdict is not None:
            return verdict
        return self._apply_mode(*self._match_rules(execution, pair))

    def _required_pairs(self, execution: ToolExecution) -> list[ResourcePermission]:
        """Flatten every request's pairs. A request with empty `resources` —
        or a call with no requests at all — contributes the implicit
        resource-less pair `(permission=<tool_name>, resource=None)`.
        Requests are presentation grouping only; resolution is per pair."""
        implicit = ResourcePermission(permission=execution.raw_tool_call.name)
        requests = self.permission_requests(execution)
        if not requests:
            return [implicit]
        pairs: list[ResourcePermission] = []
        for request in requests:
            pairs.extend(request.resources or [implicit])
        return pairs

    def _match_verdicts(
        self, execution_id: str, pair: ResourcePermission,
    ) -> tuple[ApprovalOption, str] | None:
        """The ephemeral verdict covering `pair`, DENY beating ALLOW among
        matches. Matching uses the same algorithm as rule-resource matching
        (exact permission, fnmatch resource, None↔None) but ignores tool_name
        and match mode — verdicts are execution-scoped; the tool is implied.
        Verdicts precede rules in `decide()` so an explicitly-answered call
        stays pinned against rules written in the same approval pause."""
        matched: ApprovalOption | None = None
        for verdict_pair, decision in self._verdicts.get(execution_id, []):
            if verdict_pair.permission != pair.permission:
                continue
            if not _resource_matches(pair.resource, verdict_pair.resource):
                continue
            if decision == ApprovalOption.DENY:
                return ApprovalOption.DENY, "user"
            matched = decision
        if matched is None:
            return None
        return matched, "user"

    def _match_rules(
        self, execution: ToolExecution, pair: ResourcePermission,
    ) -> tuple[ApprovalOption, str]:
        """Last-match-wins over the unified rule list. Returns (decision, via);
        no match → PENDING via 'mode'. The tool name comes from the call
        (`raw_tool_call`); the kind from the resolved `tool_spec` snapshot
        (always present here — a call whose tool didn't resolve is terminal at
        birth and never reaches the policy)."""
        tool_kind = (
            execution.tool_spec.tool_kind
            if execution.tool_spec is not None
            else None
        )
        matched: tuple[ApprovalOption, str] | None = None
        for rule in self.rules:
            if isinstance(rule, ToolKindRule):
                if rule.tool_kind == tool_kind:
                    matched = (rule.decision, "kind_default")
            elif isinstance(rule, ToolRule):
                if (
                    self._tool_matches(rule, execution.raw_tool_call.name)
                    and rule.resource_permission.permission == pair.permission
                    and _resource_matches(
                        pair.resource, rule.resource_permission.resource,
                    )
                ):
                    matched = (rule.decision, "rule")
        if matched is None:
            return ApprovalOption.PENDING, "mode"
        return matched

    def _tool_matches(self, rule: ToolRule, tool_name: str) -> bool:
        if self.permission_mode == PermissionMatchMode.RELAXED:
            return True
        return rule.tool_name is None or rule.tool_name == tool_name

    def _apply_mode(
        self, base: ApprovalOption, via: str,
    ) -> tuple[ApprovalOption, str]:
        if self.mode == PermissionMode.ASK:
            return base, via
        # YOLO / AUTO: promote a residual PENDING to ALLOW; explicit DENY blocks.
        if base in (ApprovalOption.DENY, ApprovalOption.ALLOW):
            return base, via
        return ApprovalOption.ALLOW, "mode"


def _aggregate(
    resolved: list[tuple[ApprovalOption, str]],
) -> tuple[ApprovalOption, str]:
    """Collapse (decision, via) pairs to one, DENY > PENDING > ALLOW."""
    for option in (ApprovalOption.DENY, ApprovalOption.PENDING):
        for decision, via in resolved:
            if decision == option:
                return decision, via
    return resolved[0]  # all ALLOW — keep the first's via


def _resource_matches(call_resource: str | None, rule_resource: str | None) -> bool:
    """A `None` rule matches only a resource-less pair; a glob matches only a
    resourceful pair (fnmatch); mixed never matches."""
    if call_resource is None or rule_resource is None:
        return call_resource is None and rule_resource is None
    return fnmatchcase(call_resource, rule_resource)
