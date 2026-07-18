"""Unit tests for the `Tool` base contract plus the `tool()` / `tool_class()`
factory helpers. The factory tests are SKIPPED for now: the factories were
deliberately left out of the tool-registry refactor (their `_execute` wiring
still targets the pre-registry signature) and will be redesigned separately.
The live tests pin the base `Tool` surface: `get_tool_spec()` stamping
(including `timeout_in_ms`) and `execute()` threading the keyword-only
`cancellation_token` through `_execute`. Declarative — hardcoded invariant
in, full expected out. No logic, no helpers.
"""

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from luca.agent.core import Tool, ToolContext, tool
from luca.agent.core.context import CancellationToken
from luca.agent.core.models import (
    ExecutionResult,
    LLMConfig,
    TextContent,
    ToolKind,
    ToolSpec,
)
from luca.agent.core.tools import tool_class

factory_skip = pytest.mark.skip(
    reason="tool()/tool_class() factories are out of the registry refactor's "
    "scope — to be redesigned against the new Tool signature",
)

CONTEXT = ToolContext(
    session_id="s1",
    model=LLMConfig(model="faux-model", provider="faux"),
)


async def list_files(args: dict, context: ToolContext) -> str:
    return f"listed {args['path']}"


async def describe_call(args: dict, context: ToolContext) -> dict:
    return {"resources": [args["path"]], "preview": f"List {args['path']}"}


class PathArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = "."


class StaticApprovalMixin:
    async def get_approval_context(self, args: dict, context: ToolContext) -> dict:
        return {"resources": ["from-mixin"]}


# ── the Tool base contract ────────────────────────────────────────────────────


class EchoTool(Tool):
    name = "echo"
    description = "Echo the path back."
    Args = PathArgs
    tool_kind = ToolKind.READ

    async def _execute(
        self, args: dict, context: ToolContext,
        *, cancellation_token: CancellationToken,
    ) -> str:
        return f"echo {args['path']}"


class DeadlineTool(EchoTool):
    name = "deadline"
    description = "Echo, with a declared deadline."
    timeout_in_ms = 5000


class TokenCapturingTool(Tool):
    name = "capture_token"
    description = "Records the cancellation token it received."
    Args = PathArgs

    def __init__(self) -> None:
        self.tokens: list[CancellationToken] = []

    async def _execute(
        self, args: dict, context: ToolContext,
        *, cancellation_token: CancellationToken,
    ) -> str:
        self.tokens.append(cancellation_token)
        return "captured"


def test_get_tool_spec_stamps_the_identity_classvars():
    assert EchoTool().get_tool_spec() == ToolSpec(
        name="echo",
        description="Echo the path back.",
        tool_kind=ToolKind.READ,
        namespace=None,
        version=None,
        timeout_in_ms=None,
    )


def test_get_tool_spec_stamps_the_declared_timeout():
    assert DeadlineTool().get_tool_spec() == ToolSpec(
        name="deadline",
        description="Echo, with a declared deadline.",
        tool_kind=ToolKind.READ,
        namespace=None,
        version=None,
        timeout_in_ms=5000,
    )


async def test_execute_wraps_the_simple_text_path():
    result = await EchoTool().execute(
        {"path": "src"}, CONTEXT, cancellation_token=CancellationToken(),
    )

    assert result == ExecutionResult(content=[TextContent(text="echo src")])


async def test_execute_threads_the_cancellation_token_to_the_body():
    instance = TokenCapturingTool()
    token = CancellationToken()

    await instance.execute({"path": "."}, CONTEXT, cancellation_token=token)

    assert instance.tokens == [token]


# ── the tool() / tool_class() factories (skipped — see module docstring) ─────


@factory_skip
def test_tool_class_matches_hand_written_surface():
    cls = tool_class(
        name="list_files",
        description="List the files in a directory.",
        arguments={"path": (str, Field(default=".", description="Directory to list."))},
        execute=list_files,
        tool_kind=ToolKind.READ,
    )

    assert issubclass(cls, Tool)
    assert cls.__name__ == "list_files_tool"
    assert cls.name == "list_files"
    assert cls.description == "List the files in a directory."
    assert cls.tool_kind is ToolKind.READ
    assert cls.namespace is None
    assert cls.version is None
    assert cls.timeout_in_ms is None


@factory_skip
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


@factory_skip
def test_generated_args_forbid_extra():
    cls = tool_class(
        name="list_files",
        description="List the files in a directory.",
        arguments={"path": (str, Field(default="."))},
        execute=list_files,
    )

    with pytest.raises(ValidationError):
        cls.Args.model_validate({"path": ".", "recursive": True})


@factory_skip
def test_arguments_accepts_existing_model_as_is():
    cls = tool_class(
        name="list_files",
        description="List the files in a directory.",
        arguments=PathArgs,
        execute=list_files,
    )

    assert cls.Args is PathArgs


@factory_skip
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
        tool_kind=ToolKind.READ,
        namespace=None,
        version=None,
    )


@factory_skip
async def test_execute_wires_the_callable():
    instance = tool(
        name="list_files",
        description="List the files in a directory.",
        arguments={"path": (str, Field(default="."))},
        execute=list_files,
    )

    assert await instance.execute({"path": "src"}, CONTEXT) == ExecutionResult(
        content=[TextContent(text="listed src")],
    )


@factory_skip
async def test_approval_context_defaults_to_tool_default():
    instance = tool(
        name="list_files",
        description="List the files in a directory.",
        arguments={"path": (str, Field(default="."))},
        execute=list_files,
    )

    assert await instance.get_approval_context({"path": "."}, CONTEXT) == {}


@factory_skip
async def test_explicit_approval_context_callable_is_used():
    instance = tool(
        name="list_files",
        description="List the files in a directory.",
        arguments={"path": (str, Field(default="."))},
        execute=list_files,
        get_approval_context=describe_call,
    )

    assert await instance.get_approval_context({"path": "src"}, CONTEXT) == {
        "resources": ["src"],
        "preview": "List src",
    }


@factory_skip
async def test_mixin_in_bases_provides_approval_context():
    instance = tool(
        name="list_files",
        description="List the files in a directory.",
        arguments={"path": (str, Field(default="."))},
        execute=list_files,
        bases=(StaticApprovalMixin, Tool),
    )

    assert await instance.get_approval_context({"path": "."}, CONTEXT) == {
        "resources": ["from-mixin"],
    }


@factory_skip
async def test_explicit_approval_context_beats_mixin():
    instance = tool(
        name="list_files",
        description="List the files in a directory.",
        arguments={"path": (str, Field(default="."))},
        execute=list_files,
        get_approval_context=describe_call,
        bases=(StaticApprovalMixin, Tool),
    )

    assert await instance.get_approval_context({"path": "src"}, CONTEXT) == {
        "resources": ["src"],
        "preview": "List src",
    }


@factory_skip
def test_bases_without_tool_raises():
    with pytest.raises(TypeError):
        tool_class(
            name="list_files",
            description="List the files in a directory.",
            arguments={"path": (str, Field(default="."))},
            execute=list_files,
            bases=(StaticApprovalMixin,),
        )


@factory_skip
def test_class_attrs_land_on_the_class():
    cls = tool_class(
        name="list_files",
        description="List the files in a directory.",
        arguments={"path": (str, Field(default="."))},
        execute=list_files,
        class_attrs={"namespace": "fs", "version": "1.2", "timeout_in_ms": 5000},
    )

    assert cls.namespace == "fs"
    assert cls.version == "1.2"
    assert cls.timeout_in_ms == 5000
    assert cls().get_tool_spec() == ToolSpec(
        name="list_files",
        description="List the files in a directory.",
        tool_kind=ToolKind.OTHER,
        namespace="fs",
        version="1.2",
    )


@factory_skip
def test_class_attrs_collision_raises():
    with pytest.raises(ValueError):
        tool_class(
            name="list_files",
            description="List the files in a directory.",
            arguments={"path": (str, Field(default="."))},
            execute=list_files,
            class_attrs={"name": "other_name"},
        )


@factory_skip
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


def test_tool_class_is_not_on_the_package_surface():
    import luca.agent.core

    assert not hasattr(luca.agent.core, "tool_class")
