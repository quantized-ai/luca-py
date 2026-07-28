"""`CancellationToken` — the run's cooperative cancellation signal.

The token actually trips: `runner.cancel()` / `run.cancel()` sets it after
appending the durable `CancelRequested` entry — it is the runtime wake-up
only; the entry is the truth.

The runner races EVERY collaborator await against it: all four `ToolRegistry`
calls (`get_tools`, `create_execution`, `decide`, `prepare`), the prepared
tool callable, the LLM call and its streaming steps, and compaction. When the
token wins, the runner kills the await and continues; the callee never learns
that a token exists.

Only the prepared tool callable is HANDED the token (as the keyword-only
`cancellation_token`), and only because a cooperative body can return partial
output within the cancellation grace window (`cancelled` /
`wait_cancelled()`). There is no partial answer worth having from listing
tools, minting an execution record, deciding an approval or preparing a
dispatch, so those four are raced at the runner level instead — with no grace
window, and awaiting the killed task's unwinding so a registry's `finally`
completes before the run moves on.

There is no `ToolContext`: registries, tools and context managers receive the
live `AgentSession`, which already carries everything the old context copied
(`session.id`, `session.session_config.llm_config`). Per-run application state
is not the framework's concern — registries and tools are application code and
can hold their own references or read a `contextvars.ContextVar`.
"""

from __future__ import annotations

import asyncio

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
