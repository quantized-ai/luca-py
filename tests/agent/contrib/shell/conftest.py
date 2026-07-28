"""Shared fixtures for the shell tool unit tests.

`session` is a real, empty `AgentSession` — the live session every tool hook
now receives (`ToolContext` is gone). The shell tools never read it, so a
fresh session per test is a faithful stand-in and nothing more.

`run` executes a tool the way `SimpleToolRegistry.prepare()`'s closure would —
validate the raw arguments against `Args`, then call `execute` with the
session and a fresh cancellation token. `perm` builds the tool's ordered
`PermissionRequest` list from the same validated arguments and the same
session.
"""

import pytest

from luca.agent.core import (
    AgentSession,
    AgentSessionRunner,
    CancellationToken,
    LLMConfig,
)

MODEL = LLMConfig(model="test-model", provider="faux")


@pytest.fixture
def session() -> AgentSession:
    return AgentSessionRunner.new_session(MODEL, session_id="s_shell")


@pytest.fixture
def run(session):
    async def _run(tool, arguments, *, cancellation_token=None):
        validated = tool.Args.model_validate(arguments).model_dump()
        return await tool.execute(
            validated,
            session,
            cancellation_token=cancellation_token or CancellationToken(),
        )

    return _run


@pytest.fixture
def perm(session):
    def _perm(tool, arguments):
        validated = tool.Args.model_validate(arguments).model_dump()
        return tool.build_permission_requests(validated, session)

    return _perm
