"""`SubAgentManager` — background lifecycle, concurrency cap, timeout, teardown,
and the overridable result seam."""

from __future__ import annotations

from luca.agent.contrib.subagents import SubAgentManager, SubAgentTask, TaskStatus
from luca.client.testing import faux_assistant_message, faux_hang, faux_text

from .support import FAUX_MODEL, factory_over, scripted, until


async def test_spawn_records_a_queued_task(make_manager):
    manager = make_manager(
        runner_factory=factory_over([scripted(faux_assistant_message([faux_text("done")]))]),
        child_model=FAUX_MODEL,
        generate_id=iter(["t1"]).__next__,
    )

    task_id = manager.spawn("explore", "map the runner", title="runner")

    assert task_id == "t1"
    assert manager.get("t1") == SubAgentTask(
        id="t1",
        agent_type="explore",
        title="runner",
        prompt="map the runner",
        status=TaskStatus.QUEUED,
    )


async def test_concurrency_cap_queues_beyond_the_limit(make_manager):
    manager = make_manager(
        runner_factory=factory_over([scripted(faux_assistant_message([faux_hang()])) for _ in range(3)]),
        child_model=FAUX_MODEL,
        max_concurrency=2,
        generate_id=iter(["a", "b", "c"]).__next__,
    )

    for _ in range(3):
        manager.spawn("explore", "hang")
    await until(lambda: sum(t.status is TaskStatus.RUNNING for t in manager.tasks()) == 2)

    assert [t.status for t in manager.tasks()] == [
        TaskStatus.RUNNING,
        TaskStatus.RUNNING,
        TaskStatus.QUEUED,
    ]


async def test_a_child_that_never_answers_times_out(make_manager):
    manager = make_manager(
        runner_factory=factory_over([scripted(faux_assistant_message([faux_hang()]))]),
        child_model=FAUX_MODEL,
        per_task_timeout_s=0.05,
        generate_id=iter(["t1"]).__next__,
    )

    manager.spawn("explore", "hang")
    await until(lambda: manager.get("t1").status is TaskStatus.FAILED)

    assert manager.get("t1") == SubAgentTask(
        id="t1",
        agent_type="explore",
        title="hang",
        prompt="hang",
        status=TaskStatus.FAILED,
        error="timed out",
    )


async def test_aclose_cancels_running_children(make_manager):
    manager = make_manager(
        runner_factory=factory_over([scripted(faux_assistant_message([faux_hang()])) for _ in range(3)]),
        child_model=FAUX_MODEL,
        max_concurrency=3,
        generate_id=iter(["a", "b", "c"]).__next__,
    )
    for _ in range(3):
        manager.spawn("explore", "hang")
    await until(lambda: sum(t.status is TaskStatus.RUNNING for t in manager.tasks()) == 3)

    await manager.aclose()

    assert [t.status for t in manager.tasks()] == [TaskStatus.CANCELLED] * 3


async def test_extract_result_is_overridable(make_manager):
    class LoudManager(SubAgentManager):
        def extract_result(self, session):
            return "OVERRIDDEN"

    manager = make_manager(
        cls=LoudManager,
        runner_factory=factory_over([scripted(faux_assistant_message([faux_text("real answer")]))]),
        child_model=FAUX_MODEL,
        generate_id=iter(["t1"]).__next__,
    )

    manager.spawn("general", "question")
    await until(lambda: manager.get("t1").status is TaskStatus.DONE)

    assert manager.get("t1").result == "OVERRIDDEN"
