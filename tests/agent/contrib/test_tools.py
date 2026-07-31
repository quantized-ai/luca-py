"""Unit tests for contrib's `Tool` base contract plus the `tool()` /
`tool_class()` factory helpers.

`Tool` lives in `luca.agent.contrib.tools`: the core's only tool type is
`ToolSpec`, and the ergonomic Python base class is a convenience on top of it.
The tests pin the two halves of that surface — `get_tool_spec()` stamping the
identity ClassVars AND the `input_schema` derived from `Args`, and
`execute()` threading the live `AgentSession` plus the keyword-only
`cancellation_token` into `_execute` — then the factories, which build the
same surface from plain callables and must run end to end through it.

Declarative — hardcoded invariant in, full expected out. No logic, no helpers.
"""

from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from luca.agent.contrib.tools import Tool, tool, tool_class
from luca.agent.core.context import CancellationToken
from luca.agent.core.models import (
    AgentSession,
    ExecutionResult,
    LLMConfig,
    TextContent,
    ToolKind,
    ToolSpec,
)
from luca.agent.core.runner import AgentSessionRunner

# A tool is handed the LIVE session, so the invariant every test sets up is a
# real one built the way an application builds it.
SESSION = AgentSessionRunner.new_session(
    LLMConfig(model="faux-model", provider="faux"),
    session_id="s1",
)


class PathArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = "."


# `PathArgs.model_json_schema()` verbatim — what `get_tool_spec()` stamps as
# `input_schema` for every hand-written tool below.
PATH_ARGS_SCHEMA = {
    "additionalProperties": False,
    "properties": {"path": {"default": ".", "title": "Path", "type": "string"}},
    "title": "PathArgs",
    "type": "object",
}

# The same schema as the factories compile it: `create_model` titles the
# generated model `<name>_args`.
LIST_FILES_ARGS_SCHEMA = {
    "additionalProperties": False,
    "properties": {"path": {"default": ".", "title": "Path", "type": "string"}},
    "title": "list_files_args",
    "type": "object",
}


# The preferred way to declare an output model: a module-level model bound to
# the `output_schema` ClassVar, so a consumer can import it and validate a
# payload back through it.
class WeatherReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    degrees_in_celsius: int
    wind_direction: Literal["north", "south", "east", "west"]


# `WeatherReport.model_json_schema()` verbatim — what `get_tool_spec()` stamps
# as `output_schema`. Note the type flip: the ClassVar is the model CLASS, the
# spec field is this DICT.
WEATHER_REPORT_SCHEMA = {
    "additionalProperties": False,
    "properties": {
        "degrees_in_celsius": {"title": "Degrees In Celsius", "type": "integer"},
        "wind_direction": {
            "enum": ["north", "south", "east", "west"],
            "title": "Wind Direction",
            "type": "string",
        },
    },
    "required": ["degrees_in_celsius", "wind_direction"],
    "title": "WeatherReport",
    "type": "object",
}


async def list_files(args: dict, session: AgentSession) -> str:
    return f"listed {args['path']}"


async def describe_call(args: dict, session: AgentSession) -> dict:
    return {"resources": [args["path"]], "preview": f"List {args['path']}"}


class StaticApprovalMixin:
    async def get_approval_context(self, args: dict, session: AgentSession) -> dict:
        return {"resources": ["from-mixin"]}


# ── the Tool base contract ────────────────────────────────────────────────────


class EchoTool(Tool):
    name = "echo"
    description = "Echo the path back."
    Args = PathArgs
    tool_kind = ToolKind.READ

    async def _execute(
        self,
        args: dict,
        session: AgentSession,
        *,
        cancellation_token: CancellationToken,
    ) -> str:
        return f"echo {args['path']}"


class DeadlineTool(EchoTool):
    name = "deadline"
    description = "Echo, with a declared deadline."
    timeout_in_ms = 5000


class WeatherTool(Tool):
    """Declares an output model AND produces the payload — the two are separate
    acts, which is why this overrides `execute` rather than `_execute`."""

    name = "get_weather"
    description = "Get the current weather for a city."
    Args = PathArgs
    output_schema = WeatherReport
    tool_kind = ToolKind.WEB_FETCH

    async def execute(
        self,
        args: dict,
        session: AgentSession,
        *,
        cancellation_token: CancellationToken,
    ) -> ExecutionResult:
        reading = WeatherReport(degrees_in_celsius=25, wind_direction="south")
        return ExecutionResult(
            content=[TextContent(text="25°C, wind from the south.")],
            structured_content=reading.model_dump(),
        )


class CapturingTool(Tool):
    name = "capture"
    description = "Records everything the body was handed."
    Args = PathArgs

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def _execute(
        self,
        args: dict,
        session: AgentSession,
        *,
        cancellation_token: CancellationToken,
    ) -> str:
        self.calls.append((args, session, cancellation_token))
        return "captured"


def test_get_tool_spec_stamps_the_identity_classvars_and_the_args_schema():
    # a tool declaring no output model stamps output_schema=None
    assert EchoTool().get_tool_spec() == ToolSpec(
        name="echo",
        description="Echo the path back.",
        input_schema=PATH_ARGS_SCHEMA,
        output_schema=None,
        metadata=None,
        tool_kind=ToolKind.READ,
        namespace=None,
        version=None,
        timeout_in_ms=None,
    )


def test_get_tool_spec_stamps_the_declared_timeout():
    assert DeadlineTool().get_tool_spec() == ToolSpec(
        name="deadline",
        description="Echo, with a declared deadline.",
        input_schema=PATH_ARGS_SCHEMA,
        output_schema=None,
        metadata=None,
        tool_kind=ToolKind.READ,
        namespace=None,
        version=None,
        timeout_in_ms=5000,
    )


def test_get_tool_spec_stamps_the_declared_output_schema():
    assert WeatherTool().get_tool_spec() == ToolSpec(
        name="get_weather",
        description="Get the current weather for a city.",
        input_schema=PATH_ARGS_SCHEMA,
        output_schema=WEATHER_REPORT_SCHEMA,
        metadata=None,
        tool_kind=ToolKind.WEB_FETCH,
        namespace=None,
        version=None,
        timeout_in_ms=None,
    )


async def test_a_tool_may_return_structured_content_alongside_its_text():
    result = await WeatherTool().execute(
        {"path": "Berlin"},
        SESSION,
        cancellation_token=CancellationToken(),
    )

    assert result == ExecutionResult(
        content=[TextContent(text="25°C, wind from the south.")],
        structured_content={"degrees_in_celsius": 25, "wind_direction": "south"},
        metadata={},
        is_error=False,
    )


async def test_execute_wraps_the_simple_text_path():
    result = await EchoTool().execute(
        {"path": "src"},
        SESSION,
        cancellation_token=CancellationToken(),
    )

    assert result == ExecutionResult(
        content=[TextContent(text="echo src")],
        structured_content=None,
        metadata={},
        is_error=False,
    )


async def test_execute_threads_the_session_and_token_to_the_body():
    instance = CapturingTool()
    token = CancellationToken()

    await instance.execute({"path": "."}, SESSION, cancellation_token=token)

    assert instance.calls == [({"path": "."}, SESSION, token)]


# ── the tool() / tool_class() factories ───────────────────────────────────────


def test_tool_class_matches_hand_written_surface():
    cls = tool_class(
        name="list_files",
        description="List the files in a directory.",
        arguments={"path": (str, Field(default="."))},
        execute=list_files,
        tool_kind=ToolKind.READ,
    )

    assert issubclass(cls, Tool)
    assert cls.__name__ == "list_files_tool"
    assert cls().get_tool_spec() == ToolSpec(
        name="list_files",
        description="List the files in a directory.",
        input_schema=LIST_FILES_ARGS_SCHEMA,
        output_schema=None,
        metadata=None,
        tool_kind=ToolKind.READ,
        namespace=None,
        version=None,
        timeout_in_ms=None,
    )


def test_generated_args_schema():
    cls = tool_class(
        name="list_files",
        description="List the files in a directory.",
        arguments={"path": (str, Field(default=".", description="Directory to list."))},
        execute=list_files,
    )

    assert cls.Args.model_json_schema() == {
        "additionalProperties": False,
        "properties": {
            "path": {
                "default": ".",
                "description": "Directory to list.",
                "title": "Path",
                "type": "string",
            },
        },
        "title": "list_files_args",
        "type": "object",
    }


def test_generated_args_forbid_extra():
    cls = tool_class(
        name="list_files",
        description="List the files in a directory.",
        arguments={"path": (str, Field(default="."))},
        execute=list_files,
    )

    with pytest.raises(ValidationError):
        cls.Args.model_validate({"path": ".", "recursive": True})


def test_arguments_accepts_existing_model_as_is():
    cls = tool_class(
        name="list_files",
        description="List the files in a directory.",
        arguments=PathArgs,
        execute=list_files,
    )

    assert cls.Args is PathArgs


def test_output_accepts_an_existing_model_as_is():
    cls = tool_class(
        name="get_weather",
        description="Get the current weather for a city.",
        arguments=PathArgs,
        execute=list_files,
        output=WeatherReport,
    )

    assert cls.output_schema is WeatherReport


def test_omitting_output_leaves_the_spec_output_schema_null():
    cls = tool_class(
        name="list_files",
        description="List the files in a directory.",
        arguments=PathArgs,
        execute=list_files,
    )

    assert cls.output_schema is None
    assert cls().get_tool_spec().output_schema is None


def test_generated_output_schema():
    # a dict is a create_model field spec here too, never a raw JSON Schema;
    # `create_model` titles the generated model `<name>_output`
    cls = tool_class(
        name="get_weather",
        description="Get the current weather for a city.",
        arguments=PathArgs,
        execute=list_files,
        output={"degrees_in_celsius": (int, Field(description="Current temperature."))},
    )

    assert cls().get_tool_spec().output_schema == {
        "additionalProperties": False,
        "properties": {
            "degrees_in_celsius": {
                "description": "Current temperature.",
                "title": "Degrees In Celsius",
                "type": "integer",
            },
        },
        "required": ["degrees_in_celsius"],
        "title": "get_weather_output",
        "type": "object",
    }


def test_generated_output_forbids_extra():
    cls = tool_class(
        name="get_weather",
        description="Get the current weather for a city.",
        arguments=PathArgs,
        execute=list_files,
        output={"degrees_in_celsius": (int, ...)},
    )

    with pytest.raises(ValidationError):
        cls.output_schema.model_validate({"degrees_in_celsius": 25, "humidity": 40})


def test_tool_returns_instance_with_expected_spec():
    instance = tool(
        name="list_files",
        description="List the files in a directory.",
        arguments={"path": (str, Field(default="."))},
        execute=list_files,
        tool_kind=ToolKind.READ,
    )

    assert isinstance(instance, Tool)
    assert instance.get_tool_spec() == ToolSpec(
        name="list_files",
        description="List the files in a directory.",
        input_schema=LIST_FILES_ARGS_SCHEMA,
        metadata=None,
        tool_kind=ToolKind.READ,
        namespace=None,
        version=None,
        timeout_in_ms=None,
    )


async def test_execute_runs_the_wired_callable():
    instance = tool(
        name="list_files",
        description="List the files in a directory.",
        arguments={"path": (str, Field(default="."))},
        execute=list_files,
    )

    result = await instance.execute(
        {"path": "src"},
        SESSION,
        cancellation_token=CancellationToken(),
    )

    assert result == ExecutionResult(
        content=[TextContent(text="listed src")],
        metadata={},
        is_error=False,
    )


def test_no_approval_context_leaves_the_hook_absent():
    instance = tool(
        name="list_files",
        description="List the files in a directory.",
        arguments={"path": (str, Field(default="."))},
        execute=list_files,
    )

    # The hook is duck-typed: a registry decides whether to gather approval
    # context with `hasattr`, so "no context" must mean no attribute at all.
    assert not hasattr(instance, "get_approval_context")


async def test_explicit_approval_context_callable_is_used():
    instance = tool(
        name="list_files",
        description="List the files in a directory.",
        arguments={"path": (str, Field(default="."))},
        execute=list_files,
        get_approval_context=describe_call,
    )

    assert await instance.get_approval_context({"path": "src"}, SESSION) == {
        "resources": ["src"],
        "preview": "List src",
    }


async def test_a_tool_with_an_approval_context_still_executes():
    instance = tool(
        name="list_files",
        description="List the files in a directory.",
        arguments={"path": (str, Field(default="."))},
        execute=list_files,
        get_approval_context=describe_call,
    )

    result = await instance.execute(
        {"path": "src"},
        SESSION,
        cancellation_token=CancellationToken(),
    )

    assert result == ExecutionResult(
        content=[TextContent(text="listed src")],
        metadata={},
        is_error=False,
    )


async def test_mixin_in_bases_provides_approval_context():
    instance = tool(
        name="list_files",
        description="List the files in a directory.",
        arguments={"path": (str, Field(default="."))},
        execute=list_files,
        bases=(StaticApprovalMixin, Tool),
    )

    assert await instance.get_approval_context({"path": "."}, SESSION) == {
        "resources": ["from-mixin"],
    }


async def test_explicit_approval_context_beats_mixin():
    instance = tool(
        name="list_files",
        description="List the files in a directory.",
        arguments={"path": (str, Field(default="."))},
        execute=list_files,
        get_approval_context=describe_call,
        bases=(StaticApprovalMixin, Tool),
    )

    assert await instance.get_approval_context({"path": "src"}, SESSION) == {
        "resources": ["src"],
        "preview": "List src",
    }


def test_bases_without_tool_raises():
    with pytest.raises(TypeError):
        tool_class(
            name="list_files",
            description="List the files in a directory.",
            arguments={"path": (str, Field(default="."))},
            execute=list_files,
            bases=(StaticApprovalMixin,),
        )


def test_class_attrs_land_on_the_class():
    cls = tool_class(
        name="list_files",
        description="List the files in a directory.",
        arguments={"path": (str, Field(default="."))},
        execute=list_files,
        class_attrs={"namespace": "fs", "version": "1.2", "timeout_in_ms": 5000},
    )

    assert cls().get_tool_spec() == ToolSpec(
        name="list_files",
        description="List the files in a directory.",
        input_schema=LIST_FILES_ARGS_SCHEMA,
        output_schema=None,
        metadata=None,
        tool_kind=ToolKind.OTHER,
        namespace="fs",
        version="1.2",
        timeout_in_ms=5000,
    )


def test_class_attrs_collision_raises():
    with pytest.raises(ValueError, match="class_attrs collide with factory-managed attributes"):
        tool_class(
            name="list_files",
            description="List the files in a directory.",
            arguments={"path": (str, Field(default="."))},
            execute=list_files,
            class_attrs={"name": "other_name"},
        )


def test_output_schema_via_class_attrs_collides_with_the_managed_name():
    # `output_schema` is factory-managed whether or not `output=` was passed,
    # so class_attrs can never be the back door around it
    with pytest.raises(ValueError, match="class_attrs collide with factory-managed attributes"):
        tool_class(
            name="get_weather",
            description="Get the current weather for a city.",
            arguments={"path": (str, Field(default="."))},
            execute=list_files,
            output=WeatherReport,
            class_attrs={"output_schema": WeatherReport},
        )


def test_each_call_mints_a_distinct_class():
    first = tool(
        name="list_files",
        description="List the files in a directory.",
        arguments={"path": (str, Field(default="."))},
        execute=list_files,
    )
    second = tool(
        name="list_files",
        description="List the files in a directory.",
        arguments={"path": (str, Field(default="."))},
        execute=list_files,
    )

    assert type(first) is not type(second)


def test_the_tool_helpers_are_not_on_the_core_package_surface():
    import luca.agent.core

    assert not hasattr(luca.agent.core, "Tool")
    assert not hasattr(luca.agent.core, "tool")
    assert not hasattr(luca.agent.core, "tool_class")
    assert not hasattr(luca.agent.core, "ToolContext")
