"""Runtime tool context — generated on the fly, never persisted.

`ToolContext` is the ambient, information-only state a registry or tool
receives per run: the session id and the active `LLMConfig`. The runner
builds one per `AgentRun` and hands the same instance to every registry call
in that run.

`CancellationToken` lives here too but is no longer carried by the context:
the runner passes it explicitly as the keyword-only `cancellation_token` to
`registry.execute(...)` / `tool.execute(...)`. The token actually trips:
`runner.cancel()` / `run.cancel()` sets it after appending the durable
`CancelRequested` entry — it is the runtime wake-up only; the entry is the
truth. The runner races every tool execution and LLM call against it;
cooperative tools may also watch it themselves (`cancelled` /
`wait_cancelled()`) to return partial output within the cancellation grace
window.
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, ConfigDict

from .models import LLMConfig
from .exceptions import CancelledError


class CancellationToken:
    """A cooperative cancellation signal shared across a run's tool executions.

    Wraps an `asyncio.Event`: once `cancel()` fires, `cancelled` is True forever
    and any awaiter of `wait_cancelled()` wakes. Not a Pydantic model and not
    serializable — purely runtime."""

    def __init__(self) -> None:
        self._event = asyncio.Event()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()

    async def wait_cancelled(self) -> None:
        await self._event.wait()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise CancelledError()


class ToolContext(BaseModel):
    """Ambient, information-only context passed to registry and tool calls.
    Transient — built per `run()`, never stored on the session."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    session_id: str
    model: LLMConfig
