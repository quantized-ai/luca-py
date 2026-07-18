"""Agent-level Tool definitions — the runtime extension point.

A `Tool` describes itself to the LLM (the adapter builds a `luca.client.Tool`
from its `name` / `description` / `Args`), declares its classification
`tool_kind`, and executes. This is the EXECUTION contract only: everything
around it — resolution, argument validation, approval — is owned by the
`ToolRegistry` the runner is constructed with (`tool_registry.py`).

A registry drives each tool call: it snapshots the tool into a `ToolSpec`
(including `timeout_in_ms`) at birth, validates the LLM-produced arguments
against `Args`, and — only once the call is approved — calls `execute`.
Subclasses cooperate by overriding `_execute` (the simple text path) or
`execute` (for is_error / metadata / multi-block results).

`tool()` / `tool_class()` (bottom of this module) build a `Tool` on the fly
from plain callables — convenience helpers for runtime-constructed tools.
Subclassing `Tool` with an `Args` model stays the recommended mechanism.

Cancellation & deadline contract. The runner races every execution against
the run's `CancellationToken` (passed explicitly as the keyword-only
`cancellation_token`) and an optional outside deadline (the birth
`ToolSpec.timeout_in_ms`, else `RuntimeConfig.tool_execution_timeout_in_ms`):

- A cancelled run hard-cancels the tool task once the grace period
  (`RuntimeConfig.tool_cancellation_grace_period`, default 0) expires —
  recorded INTERRUPTED, resultless. A cooperative tool may watch the
  `cancellation_token` and return early within the grace window:
  whatever it returns is its real result (say "cut short" in the content and
  choose `is_error` yourself).
- Deadline expiry hard-cancels the same way — recorded TIMED_OUT.
- Tools that spawn processes MUST kill their process group on
  `asyncio.CancelledError` (`start_new_session=True` + `os.killpg`) — the
  hard cancel for cancellation and timeout is identical. Blocking sync work
  belongs in `asyncio.to_thread` (a hard cancel cannot interrupt a syscall;
  the runner records the outcome on time and detaches the worker).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, create_model

from .context import CancellationToken, ToolContext
from .models import ExecutionResult, TextContent, ToolKind, ToolSpec


class Tool:
    """Base tool. Subclasses set the `ClassVar`s and override `_execute`.

    `Args` is a Pydantic model describing the call arguments; the adapter turns
    it into the wire JSON schema. `tool_kind` / `namespace` / `version` feed the
    `ToolSpec` snapshot so a saved conversation stays identifiable forever.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    Args: ClassVar[type[BaseModel]]

    tool_kind: ClassVar[ToolKind] = ToolKind.OTHER
    namespace: ClassVar[str | None] = None
    version: ClassVar[str | None] = None
    # Per-tool execution deadline (ms), snapshotted into ToolSpec at birth.
    # Beats RuntimeConfig.tool_execution_timeout_in_ms; None defers to it;
    # -1 (Inf) disables. Expiry hard-cancels the call and records TIMED_OUT
    # (resultless).
    timeout_in_ms: ClassVar[int | None] = None

    def get_tool_spec(self) -> ToolSpec:
        """A self-contained snapshot of this tool's identity, independent of
        any argument payload (arguments live on `ToolExecution.raw_tool_call`)."""
        return ToolSpec(
            name=self.name,
            description=self.description,
            tool_kind=self.tool_kind,
            namespace=self.namespace,
            version=self.version,
            timeout_in_ms=self.timeout_in_ms,
        )

    async def _execute(
        self, args: dict, context: ToolContext,
        *, cancellation_token: CancellationToken,
    ) -> str:
        """The simple override point: do the work, return text for the LLM.
        The `context` carries the session id and active model; the
        `cancellation_token` may be checked cooperatively (V1 tools may
        ignore it)."""
        raise NotImplementedError

    async def execute(
        self, args: dict, context: ToolContext,
        *, cancellation_token: CancellationToken,
    ) -> ExecutionResult:
        """Run the tool and wrap the output. Override for is_error / metadata /
        multi-block results. Timing is recorded by the runner on the
        `ToolExecution` (`started_at` / `ended_at`), not on the result."""
        output = await self._execute(
            args, context, cancellation_token=cancellation_token,
        )
        return ExecutionResult(content=[TextContent(text=output)])


def tool_class(
    *,
    name: str,
    description: str,
    arguments: type[BaseModel] | dict[str, Any],
    execute: Callable[[dict, ToolContext], Awaitable[str]],
    tool_kind: ToolKind = ToolKind.OTHER,
    get_approval_context: Callable[[dict, ToolContext], Awaitable[dict]] | None = None,
    bases: tuple[type, ...] = (Tool,),
    class_attrs: dict[str, Any] | None = None,
) -> type[Tool]:
    """Build a `Tool` subclass at runtime from plain callables.

    Convenience for runtime-constructed tools; subclassing `Tool` with an
    `Args` model stays the recommended mechanism (and the escape hatch for
    anything this signature doesn't cover: rich `ExecutionResult` output,
    validators, per-instance `__init__` state).

    - `arguments` is either a ready `BaseModel` class (used as `Args` as-is)
      or a `create_model` field spec — `{"path": (str, Field(default="."))}` —
      compiled into an `extra="forbid"` model.
    - `execute` becomes `_execute`, the simple text path: an async
      `(args, context) -> str`. Per-instance configuration belongs in the
      callable's closure, not on the class.
    - `get_approval_context`, when given, overrides any inherited one —
      including a `bases` mixin's (class dict beats MRO); passing both is
      almost certainly a mistake.
    - `bases` must contain a `Tool` subclass; MRO order is the caller's job
      (mixins before `Tool`).
    - `class_attrs` merges extra class attributes into the definition —
      mixin-required attributes, or the remaining `Tool` ClassVars
      (`namespace`, `version`, `timeout_in_ms`). They are part of the tool's
      definition, shared by every instance. Colliding with a factory-managed
      attribute raises.

    The returned class is a fresh anonymous type on every call: it has no
    importable qualname (so instances don't pickle) and no stable identity
    (two identical calls produce distinct classes). Hand-write the class if
    you need either.
    """
    if not any(isinstance(base, type) and issubclass(base, Tool) for base in bases):
        raise TypeError("tool_class() bases must include Tool or a Tool subclass")

    if isinstance(arguments, type) and issubclass(arguments, BaseModel):
        args_model = arguments
    else:
        args_model = create_model(
            f"{name}_args", __config__=ConfigDict(extra="forbid"), **arguments,
        )

    async def _execute(self: Tool, args: dict, context: ToolContext) -> str:
        return await execute(args, context)

    ns: dict[str, Any] = {
        "name": name,
        "description": description,
        "Args": args_model,
        "tool_kind": tool_kind,
        "_execute": _execute,
    }
    if get_approval_context is not None:

        async def _get_approval_context(
            self: Tool, args: dict, context: ToolContext,
        ) -> dict:
            return await get_approval_context(args, context)

        ns["get_approval_context"] = _get_approval_context

    collisions = ns.keys() & (class_attrs or {}).keys()
    if collisions:
        raise ValueError(
            f"class_attrs collide with factory-managed attributes: {sorted(collisions)}",
        )
    ns.update(class_attrs or {})

    return type(f"{name}_tool", bases, ns)


def tool(
    *,
    name: str,
    description: str,
    arguments: type[BaseModel] | dict[str, Any],
    execute: Callable[[dict, ToolContext], Awaitable[str]],
    tool_kind: ToolKind = ToolKind.OTHER,
    get_approval_context: Callable[[dict, ToolContext], Awaitable[dict]] | None = None,
    bases: tuple[type, ...] = (Tool,),
    class_attrs: dict[str, Any] | None = None,
) -> Tool:
    """Build a tool on the fly and return it ready to pass to the runner —
    `tool_class(...)()`. Same parameters and caveats as `tool_class`."""
    return tool_class(
        name=name,
        description=description,
        arguments=arguments,
        execute=execute,
        tool_kind=tool_kind,
        get_approval_context=get_approval_context,
        bases=bases,
        class_attrs=class_attrs,
    )()
