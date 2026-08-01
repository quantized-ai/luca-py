"""`Tool` — the ergonomic way to write a tool in Python.

The CORE knows nothing about this module. Its only tool type is `ToolSpec`
(`luca.agent.core.models`): plain, language-neutral, JSON-serializable data.
`Tool` is contrib — a convenience for people whose tools live in this process
and who would rather declare a Pydantic `Args` model than hand-write JSON
Schema. A registry backed by a remote tool server never needs it.

A `Tool` describes itself to the LLM (`name` / `description` / `Args`),
declares its classification `tool_kind`, and executes. This is the EXECUTION
contract only: everything around it — resolution, argument validation,
approval — is owned by the `ToolRegistry` the runner is constructed with
(`luca.agent.core.tool_registry`).

A registry drives each tool call: it snapshots the tool into a `ToolSpec` via
`get_tool_spec()` at birth, validates the LLM-produced arguments against
`Args`, and — only once the call is approved — resolves and validates again in
`prepare()` and returns a callable over `execute`. Subclasses cooperate by
overriding `_execute` (the simple text path) or `execute` (for is_error /
metadata / multi-block results).

It lives here rather than inside a registry package because `resource_
permissions`, `shell` and `memory` all build on `Tool` without going through
any particular registry.

`tool()` / `tool_class()` (bottom of this module) build a `Tool` on the fly
from plain callables — convenience helpers for runtime-constructed tools.
Subclassing `Tool` with an `Args` model stays the recommended mechanism.

WHAT A TOOL RECEIVES. `execute` / `_execute` / the duck-typed
`get_approval_context` take the live `AgentSession` — read `session.id` and
`session.session_config.llm_config` — and the `conversation_id` the call
belongs to. Treat the session as READ-ONLY: the runner owns every write to
session state. Per-run application state is not the framework's concern; a tool
is application code and can hold its own references or read a
`contextvars.ContextVar`.

WHY A TOOL GETS A CONVERSATION. A session can hold several conversations at
once — the main agent's and one per subagent — and a tool instance is SHARED by
all of them. Without the id a body cannot tell whether it is running for the
main agent or for subagent B, so any per-conversation state it keeps
(a scratchpad, a todo list, a read-before-edit ledger) silently becomes shared
and the conversations overwrite each other. Key such state by
`conversation_id`; tool dispatch within one conversation is sequential, so a
per-conversation slot needs no lock. State deliberately shared ACROSS
conversations does — see rule 13 in `luca.agent.core.tool_registry`.

The conversation reaches the body through the CLOSURE the registry builds in
`prepare()`, not through a new core parameter: `PreparedTool` still takes only
the cancellation token, so the core still depends on no Python tool class.

WHAT A TOOL RETURNS. `_execute` is the simple text path. `execute` is the rich
one, and the only place a tool can set `structured_content` — the
machine-readable payload an application reads, which never reaches the model
(`content` is the sole model-facing channel). Declaring an `output_schema`
model advertises the SHAPE of that payload; it does not produce one, and
nothing validates the payload against it.

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
- Deadline expiry hard-cancels the same way — recorded TIMED_OUT. The
  deadline bounds the BODY only: a tool configured with `timeout_in_ms=5000`
  is NOT bounded end to end, since resolution, validation and approval
  preflight all happen outside it.
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

from luca.agent.core import (
    AgentSession,
    CancellationToken,
    ExecutionResult,
    TextContent,
    ToolKind,
    ToolSpec,
)


class Tool:
    """Base tool. Subclasses set the `ClassVar`s and override `_execute`.

    `Args` is a Pydantic model describing the call arguments; `get_tool_spec()`
    turns it into the `input_schema` the model is shown. `tool_kind` /
    `namespace` / `version` feed the `ToolSpec` snapshot so a saved
    conversation stays identifiable forever.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    Args: ClassVar[type[BaseModel]]
    # Optional: the Pydantic MODEL CLASS describing the machine-readable result
    # this tool can produce — usually a module-level model bound here
    # (`output_schema = WeatherReport`), which keeps it importable so a
    # consumer can validate a payload back through it. `get_tool_spec()` turns
    # it into `ToolSpec.output_schema`, the JSON Schema DICT derived from it:
    # same name, different type, one level apart.
    #
    # Declaring it populates NOTHING. A tool that produces a payload overrides
    # `execute()` and sets `structured_content=` on the returned
    # `ExecutionResult`; nothing validates one against the other.
    output_schema: ClassVar[type[BaseModel] | None] = None

    # Never advertised to the model. The RUNTIME still resolves, prepares and
    # dispatches it — `get_tools()` returns it exactly like any other tool —
    # but the runner omits it from the wire list, and a model call naming it
    # records NOT_FOUND. Write a private tool exactly like any other.
    is_private: ClassVar[bool] = False

    tool_kind: ClassVar[ToolKind] = ToolKind.OTHER
    namespace: ClassVar[str | None] = None
    version: ClassVar[str | None] = None
    # Per-tool execution deadline (ms), snapshotted into ToolSpec at birth.
    # Beats RuntimeConfig.tool_execution_timeout_in_ms; None defers to it;
    # -1 (Inf) disables. Expiry hard-cancels the call and records TIMED_OUT
    # (resultless). It bounds the BODY only.
    timeout_in_ms: ClassVar[int | None] = None

    def get_tool_spec(self) -> ToolSpec:
        """A self-contained snapshot of this tool's identity and its argument
        schema, independent of any argument payload (arguments live on
        `ToolExecution.raw_tool_call`).

        `input_schema` is `Args.model_json_schema()` verbatim — byte-identical
        to what the transport used to derive from `Args` at send time, so the
        wire payload is unchanged; the schema is simply computed at snapshot
        time instead.

        `output_schema` is derived from the ClassVar of the same name exactly
        as `input_schema` is derived from `Args`, and is None when the tool
        declares no output model. Mind the types: the ClassVar is a Pydantic
        model CLASS, the spec field is a JSON Schema DICT.

        It must stay a pure function of the tool DEFINITION: specs are stored
        once per session under a content hash, so anything call-scoped here
        would mint a new stored row on every call."""
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=self.Args.model_json_schema(),
            output_schema=(self.output_schema.model_json_schema() if self.output_schema is not None else None),
            is_private=self.is_private,
            tool_kind=self.tool_kind,
            namespace=self.namespace,
            version=self.version,
            timeout_in_ms=self.timeout_in_ms,
        )

    async def _execute(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
        *,
        cancellation_token: CancellationToken,
    ) -> str:
        """The simple override point: do the work, return text for the LLM.
        `session` is the live session — read-only; `conversation_id` is the
        conversation this call belongs to (key any per-conversation state by
        it); the `cancellation_token` may be checked cooperatively (V1 tools
        may ignore it)."""
        raise NotImplementedError

    async def execute(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
        *,
        cancellation_token: CancellationToken,
    ) -> ExecutionResult:
        """Run the tool and wrap the output. Override for is_error / metadata /
        multi-block results. Timing is recorded by the runner on the
        `ToolExecution` (`started_at` / `ended_at`), not on the result."""
        output = await self._execute(
            args,
            session,
            conversation_id,
            cancellation_token=cancellation_token,
        )
        return ExecutionResult(content=[TextContent(text=output)])


def tool_class(
    *,
    name: str,
    description: str,
    arguments: type[BaseModel] | dict[str, Any],
    execute: Callable[[dict, AgentSession, str], Awaitable[str]],
    output: type[BaseModel] | dict[str, Any] | None = None,
    is_private: bool = False,
    tool_kind: ToolKind = ToolKind.OTHER,
    get_approval_context: Callable[[dict, AgentSession, str], Awaitable[dict]] | None = None,
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
      `(args, session, conversation_id) -> str`. Per-instance configuration
      belongs in the callable's closure, not on the class.
    - `output`, when given, becomes `output_schema` and takes the same two
      forms as `arguments` — a ready `BaseModel` class used as-is, or a
      `create_model` field spec compiled into an `extra="forbid"` model. A
      dict is ALWAYS a field spec here too, never a raw JSON Schema.

      ⚠️ The factories wire only the text path, so a factory-built tool can
      DECLARE an output schema but cannot populate `structured_content` —
      that needs an `execute` override. Hand-write the class, or pass a
      `bases=` mixin that overrides `execute`, when you need both.
    - `is_private`, when True, keeps the tool off the wire: the runtime still
      resolves and dispatches it, but the model is never shown it and a call
      naming it records NOT_FOUND.
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
            f"{name}_args",
            __config__=ConfigDict(extra="forbid"),
            **arguments,
        )

    output_model: type[BaseModel] | None = None
    if output is not None:
        if isinstance(output, type) and issubclass(output, BaseModel):
            output_model = output
        else:
            output_model = create_model(
                f"{name}_output",
                __config__=ConfigDict(extra="forbid"),
                **output,
            )

    # Matches `Tool.execute`'s call exactly — including the keyword-only
    # `cancellation_token` it always passes. The wrapped callable keeps the
    # two-argument shape the factory advertises.
    async def _execute(
        self: Tool,
        args: dict,
        session: AgentSession,
        conversation_id: str,
        *,
        cancellation_token: CancellationToken,
    ) -> str:
        return await execute(args, session, conversation_id)

    # `output_schema` goes in UNCONDITIONALLY (None when not given, which is
    # what the base class declares anyway) so it takes part in the collision
    # check below — passing `output=` and `class_attrs={"output_schema": …}`
    # together has to raise, not silently pick one.
    ns: dict[str, Any] = {
        "name": name,
        "description": description,
        "Args": args_model,
        "output_schema": output_model,
        "is_private": is_private,
        "tool_kind": tool_kind,
        "_execute": _execute,
    }
    if get_approval_context is not None:

        async def _get_approval_context(
            self: Tool,
            args: dict,
            session: AgentSession,
            conversation_id: str,
        ) -> dict:
            return await get_approval_context(args, session, conversation_id)

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
    execute: Callable[[dict, AgentSession, str], Awaitable[str]],
    output: type[BaseModel] | dict[str, Any] | None = None,
    is_private: bool = False,
    tool_kind: ToolKind = ToolKind.OTHER,
    get_approval_context: Callable[[dict, AgentSession, str], Awaitable[dict]] | None = None,
    bases: tuple[type, ...] = (Tool,),
    class_attrs: dict[str, Any] | None = None,
) -> Tool:
    """Build a tool on the fly and return it ready to pass to a registry —
    `tool_class(...)()`. Same parameters and caveats as `tool_class`."""
    return tool_class(
        name=name,
        description=description,
        arguments=arguments,
        execute=execute,
        output=output,
        is_private=is_private,
        tool_kind=tool_kind,
        get_approval_context=get_approval_context,
        bases=bases,
        class_attrs=class_attrs,
    )()
