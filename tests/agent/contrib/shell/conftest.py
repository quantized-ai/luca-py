"""Shared fixtures for the shell tool unit tests.

`session` is a real, empty `AgentSession` — the live session every tool hook
now receives (`ToolContext` is gone). The shell tools never read it, so a
fresh session per test is a faithful stand-in and nothing more.

`run` executes a tool the way `SimpleToolRegistry.prepare()`'s closure would —
validate the raw arguments against `Args`, then call `execute` with the
session, the conversation the call belongs to, and a fresh cancellation token.
`perm` builds the tool's ordered `PermissionRequest` list from the same
validated arguments, session and conversation.

`conversation_id` defaults to the session's own conversation; a test that
exercises the per-conversation isolation of `FileReadTracker` passes a second
one explicitly.
"""

import pytest

from luca.agent.core import (
    AgentSession,
    AgentSessionRunner,
    CancellationToken,
    LLMConfig,
)
from tests.agent.scenarios import (
    main_conversation,
)

MODEL = LLMConfig(model="test-model", provider="faux")

# The one conversation these unit tests run in. Fixed rather than generated so
# a test file can pre-seed a `FileReadTracker` for it without a fixture.
CONVERSATION = "c_shell"


@pytest.fixture
def session() -> AgentSession:
    return AgentSessionRunner.new_session(
        MODEL,
        session_id="s_shell",
        conversation_id=CONVERSATION,
    )


@pytest.fixture
def conversation_id(session) -> str:
    return main_conversation(session).id


@pytest.fixture
def run(session, conversation_id):
    async def _run(tool, arguments, *, cancellation_token=None, conversation=None):
        validated = tool.Args.model_validate(arguments).model_dump()
        return await tool.execute(
            validated,
            session,
            conversation or conversation_id,
            cancellation_token=cancellation_token or CancellationToken(),
        )

    return _run


@pytest.fixture
def perm(session, conversation_id):
    def _perm(tool, arguments, *, conversation=None):
        validated = tool.Args.model_validate(arguments).model_dump()
        return tool.build_permission_requests(
            validated,
            session,
            conversation or conversation_id,
        )

    return _perm
