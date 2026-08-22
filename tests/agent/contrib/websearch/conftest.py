"""Shared machinery for the websearch battery.

The advertisement tests mock the LLM boundary at the runner's import site
(`luca.agent.core.runner.acompletion`) with kwargs capture — the
`tests/agent/test_native_tools/conftest.py` pattern — because asserting WHAT
was advertised needs the call's `tools=` kwarg, which the FauxProvider does
not capture-assert the same way. The synthesis/runner-integration tests use
the FauxProvider's phase-5 web scripting instead.
"""

from __future__ import annotations

import pytest

import luca.agent.core.runner as runner_module
from luca.client.types import (
    AssistantMessage as ClientAssistantMessage,
    ChatCompletionResponse,
    Usage as ClientUsage,
)
from luca.client.types.streaming import FinishEvent


class _FakeStream:
    """The minimal shape the runner's streaming path consumes."""

    def __init__(self, message: ClientAssistantMessage) -> None:
        self._message = message

    async def __aenter__(self) -> _FakeStream:
        return self

    async def __aexit__(self, *exc_info) -> bool:
        return False

    def __aiter__(self):
        return self._events()

    async def _events(self):
        yield FinishEvent(
            message=self._message,
            finish_reason=self._message.finish_reason,
            provider_finish_reason=None,
            usage=ClientUsage(),
        )


class MockLLM:
    """Scripted stand-in for `acompletion` / `acompletion_stream` with
    per-call kwargs capture."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.responses: list[ClientAssistantMessage] = []

    def queue(self, *messages: ClientAssistantMessage) -> None:
        self.responses.extend(messages)

    async def acompletion(self, **kwargs) -> ChatCompletionResponse:
        self.calls.append(kwargs)
        return ChatCompletionResponse(messages=[self.responses.pop(0)])

    def acompletion_stream(self, **kwargs) -> _FakeStream:
        self.calls.append(kwargs)
        return _FakeStream(self.responses.pop(0))


@pytest.fixture
def llm(monkeypatch):
    mock = MockLLM()
    monkeypatch.setattr(runner_module, "acompletion", mock.acompletion)
    monkeypatch.setattr(runner_module, "acompletion_stream", mock.acompletion_stream)
    yield mock
    assert mock.responses == [], "scripted LLM responses left unconsumed"
