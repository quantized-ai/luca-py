"""`SubAgentManager` — runs read-only sub-agents as background tasks.

Each sub-agent is a separate `AgentSessionRunner` over its own `AgentSession`:
one runner drives one run at a time, and a session is mutated in place with no
locks, so the isolation is mandatory, not a choice. `spawn` is synchronous and
non-blocking — it builds the child, posts the prompt, and launches a
supervising `asyncio.Task` that drains the child under a concurrency cap and a
per-task timeout. Progress and results flow out through `updates()` as
immutable `SubAgentTask` snapshots; the caller (the TUI) renders them and
injects the final result back into the parent conversation.

The manager owns teardown. `aclose()` cancels every running and queued child
and awaits the unwind, so no task or child engine is ever orphaned — a leaked
task is a test failure here. Cancelling a supervisor mid-drive tears the child
engine down through the run's `async with` on the way out; the child session is
discarded, so its interrupted state never matters.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from luca.agent.core import (
    AgentSession,
    AgentSessionRunner,
    AssistantMessage,
    LLMConfig,
    RuntimeConfig,
    TextContent,
)

from .types import BUILTIN_AGENT_TYPES, SubAgentType


class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SubAgentTask(BaseModel):
    """An immutable snapshot of one sub-agent task — safe to hand to the UI and
    to assert on whole in tests."""

    model_config = ConfigDict(extra="forbid")

    id: str
    agent_type: str
    title: str
    prompt: str
    status: TaskStatus
    steps: int = 0
    result: str | None = None
    error: str | None = None


# (agent_type, child_session) -> the runner that drives the child.
RunnerFactory = Callable[[str, AgentSession], AgentSessionRunner]

_SENTINEL = object()


def _uuid8() -> str:
    return uuid4().hex[:8]


class SubAgentManager:
    """Owns the background sub-agent runs and their snapshots.

    `runner_factory` builds a child runner from a child session — the seam
    tests use to inject a per-child `FauxProvider` (concurrent children cannot
    share one faux; its scripted queue interleaves). `child_model` is the model
    every child inherits (the parent session's, in production)."""

    def __init__(
        self,
        *,
        runner_factory: RunnerFactory,
        child_model: LLMConfig,
        agent_types: dict[str, SubAgentType] | None = None,
        max_concurrency: int = 2,
        per_task_timeout_s: float = 300.0,
        generate_id: Callable[[], str] = _uuid8,
    ) -> None:
        self._runner_factory = runner_factory
        self._child_model = child_model
        self._agent_types = agent_types or BUILTIN_AGENT_TYPES
        self._per_task_timeout_s = per_task_timeout_s
        self._generate_id = generate_id
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._snapshots: dict[str, SubAgentTask] = {}
        self._runners: dict[str, AgentSessionRunner] = {}
        self._supervisors: dict[str, asyncio.Task] = {}
        self._queue: asyncio.Queue = asyncio.Queue()
        self._closed = False

    @property
    def agent_types(self) -> dict[str, SubAgentType]:
        return dict(self._agent_types)

    # ── caller-facing ────────────────────────────────────────────────────────

    def spawn(self, agent_type: str, prompt: str, title: str | None = None) -> str:
        """Build a child, post the prompt, and launch its supervisor. Cheap and
        non-blocking: the network work happens inside the background task, not
        here, so a tool body can call this and return at once."""
        if self._closed:
            raise RuntimeError("SubAgentManager is closed; cannot spawn.")
        if agent_type not in self._agent_types:
            raise ValueError(f"Unknown sub-agent type {agent_type!r}; known types: {sorted(self._agent_types)}.")
        profile = self._agent_types[agent_type]
        task_id = self._generate_id()
        child_session = AgentSessionRunner.new_session(
            self._child_model,
            runtime_config=RuntimeConfig(
                soft_max_steps=profile.soft_max_steps,
                hard_max_steps=profile.hard_max_steps,
            ),
        )
        runner = self._runner_factory(agent_type, child_session)
        runner.post_message(prompt)
        self._runners[task_id] = runner
        self._publish(
            SubAgentTask(
                id=task_id,
                agent_type=agent_type,
                title=title or prompt[:60],
                prompt=prompt,
                status=TaskStatus.QUEUED,
            )
        )
        self._supervisors[task_id] = asyncio.ensure_future(self._supervise(task_id))
        return task_id

    def tasks(self) -> list[SubAgentTask]:
        return list(self._snapshots.values())

    def get(self, task_id: str) -> SubAgentTask:
        return self._snapshots[task_id]

    async def updates(self) -> AsyncIterator[SubAgentTask]:
        """Yield each task snapshot as it transitions; return when `aclose()`
        signals the end of the stream."""
        while True:
            item = await self._queue.get()
            if item is _SENTINEL:
                return
            yield item

    def extract_result(self, session: AgentSession) -> str:
        """The sub-agent's answer: the text of the last assistant message.
        Public so an application can override how a child's result is read."""
        entries = session.entries
        for node_id in reversed(session.active_conversation.nodes):
            entry = entries.get(node_id)
            if isinstance(entry, AssistantMessage):
                return "\n".join(part.text for part in entry.parts if isinstance(part, TextContent)).strip()
        return ""

    async def aclose(self) -> None:
        """Cancel every running and queued child and await the unwind, then
        close the update stream. Idempotent."""
        if self._closed:
            return
        self._closed = True
        supervisors = list(self._supervisors.values())
        for task in supervisors:
            task.cancel()
        if supervisors:
            await asyncio.gather(*supervisors, return_exceptions=True)
        self._queue.put_nowait(_SENTINEL)

    # ── internals ────────────────────────────────────────────────────────────

    async def _supervise(self, task_id: str) -> None:
        runner = self._runners[task_id]
        try:
            async with self._semaphore:  # cap + queue: QUEUED until a slot frees
                self._transition(task_id, TaskStatus.RUNNING)
                await asyncio.wait_for(self._drain(task_id, runner), self._per_task_timeout_s)
            self._transition(task_id, TaskStatus.DONE, result=self.extract_result(runner.session))
        except TimeoutError:
            self._transition(task_id, TaskStatus.FAILED, error="timed out")
        except asyncio.CancelledError:
            self._transition(task_id, TaskStatus.CANCELLED, error="cancelled")
            raise
        except Exception as exc:
            self._transition(task_id, TaskStatus.FAILED, error=str(exc))

    async def _drain(self, task_id: str, runner: AgentSessionRunner) -> None:
        """Drive the child to completion, closing its engine on the way out —
        even under cancellation/timeout — via the run's `async with`."""
        run = runner.run(streaming=False)
        async with run:
            async for _ in run:
                self._bump_steps(task_id, runner.session)

    def _bump_steps(self, task_id: str, session: AgentSession) -> None:
        steps = session.session_runtime_status.step_count
        current = self._snapshots[task_id]
        if steps != current.steps:
            self._transition(task_id, current.status, steps=steps)

    def _transition(self, task_id: str, status: TaskStatus, **fields) -> None:
        snapshot = self._snapshots[task_id].model_copy(update={"status": status, **fields})
        self._publish(snapshot)

    def _publish(self, snapshot: SubAgentTask) -> None:
        self._snapshots[snapshot.id] = snapshot
        self._queue.put_nowait(snapshot)
