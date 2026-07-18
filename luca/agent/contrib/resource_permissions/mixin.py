"""Typed approval-request contract between tools and `PermissionStrategy`.

The strategy reads the approval context stored under
`ToolExecution.extras["approval_context"]`: a `{"requests": [...]}` dict
whose entries each describe one approval step — the `(permission, resource)`
pairs the step requires, suggested grants the user may answer with, and
UX-only metadata. `SimpleToolRegistry` stores whatever the tool's duck-typed
`get_approval_context` returns, so nothing enforces the shape — a tool that
misspells a key or forgets "resources" silently degrades the whole strategy.
This mixin closes the gap: a tool mixes it in, implements
`build_permission_requests()` returning a list of `PermissionRequest`s, and
the mixin's `get_approval_context()` serializes them to exactly the dict the
strategy expects.

Most tools need a single request whose permission is simply the tool's name;
a tool that performs several distinguishable actions (read here, write
there) returns one request per action, in the order it wants them presented:

    class ReadFileTool(ResourcePermissionToolMixin, Tool):
        ...
        def build_permission_requests(self, args, context):
            return [PermissionRequest(
                resources=[
                    ResourcePermission(permission="read", resource=args["path"]),
                ],
                metadata={"preview": f"Read {args['path']}"},
            )]

The mixin defines the `get_approval_context` convention `SimpleToolRegistry`
reads (there is no base-class method to override — `Tool` doesn't declare
one).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from luca.agent.core import ToolContext


class ResourcePermission(BaseModel):
    """One (permission, resource) pair — the single unit of vocabulary. The
    same type expresses what a tool requires, what a user grants, and what a
    rule stores. `resource=None` denotes the resource-less invocation. By
    convention a single-action tool's permission is the tool name."""

    permission: str
    resource: str | None = None

    model_config = ConfigDict(extra="forbid")


class AnswerOption(BaseModel):
    """A set of pairs offered (or constructed) as one selectable answer.
    `metadata` is UX-only — previews, labels — and is NEVER read by the
    strategy."""

    resource_permissions: list[ResourcePermission]
    metadata: dict = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class PermissionRequest(BaseModel):
    """One approval step a tool declares: the pairs it needs, suggested
    grants, and UX metadata."""

    resources: list[ResourcePermission]
    answer_options: list[AnswerOption] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class ResourcePermissionToolMixin:
    """Mix into a `Tool` and implement `build_permission_requests` — the one
    override point. Receives the VALIDATED arguments, exactly like the
    `get_approval_context` convention `SimpleToolRegistry` reads."""

    def build_permission_requests(
        self, args: dict, context: ToolContext,
    ) -> list[PermissionRequest]:
        raise NotImplementedError

    async def get_approval_context(self, args: dict, context: ToolContext) -> dict:
        requests = self.build_permission_requests(args, context)
        return {"requests": [request.model_dump() for request in requests]}
