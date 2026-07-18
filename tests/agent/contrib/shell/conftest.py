"""Shared fixtures for the shell tool unit tests.

`run` executes a tool the way `SimpleToolRegistry` would — validate the raw
arguments against `Args`, then call `execute` with a fresh cancellation
token. `perm` builds the tool's ordered `PermissionRequest` list from the
same validated arguments.
"""

import pytest

from luca.agent.core import CancellationToken, LLMConfig, ToolContext

MODEL = LLMConfig(model="test-model", provider="faux")


@pytest.fixture
def context() -> ToolContext:
    return ToolContext(session_id="s_shell", model=MODEL)


@pytest.fixture
def run(context):
    async def _run(tool, arguments, *, cancellation_token=None):
        validated = tool.Args.model_validate(arguments).model_dump()
        return await tool.execute(
            validated,
            context,
            cancellation_token=cancellation_token or CancellationToken(),
        )

    return _run


@pytest.fixture
def perm(context):
    def _perm(tool, arguments):
        validated = tool.Args.model_validate(arguments).model_dump()
        return tool.build_permission_requests(validated, context)

    return _perm
