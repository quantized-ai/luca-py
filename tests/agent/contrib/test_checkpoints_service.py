"""`CheckpointIndex` and `CheckpointService`: the conversation half, and the
two halves joined.

The store's own behavior is covered in `test_checkpoints_store.py`. What is
under test here is the ANCHORING — that a checkpoint taken before a message
rewinds the whole turn that message opened — plus the index's durability across
a save/load and its refusal to offer a checkpoint the current path can no
longer reach.
"""

import asyncio
import shutil
from pathlib import Path

import pytest

from luca.agent.contrib.checkpoints import (
    CHECKPOINTS_KEY,
    Checkpoint,
    CheckpointIndex,
    CheckpointService,
    ShadowGitStore,
    read_index,
    restorable,
    write_index,
)
from luca.agent.core.exceptions import AgentError
from luca.agent.core.models import AgentSession, SessionConfig, TextContent, UserMessage
from luca.client.testing import FauxProvider, faux_assistant_message, faux_text
from tests.agent.scenarios import (
    GATED_SESSION,
    MODEL,
    RICH_IDLE_SESSION,
    DeterministicRunner,
    conversation,
    main_conversation,
    make_session,
)

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="checkpoints need a git binary")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    space = tmp_path / "workspace"
    space.mkdir()
    (space / "kept.py").write_text("original\n")
    return space


@pytest.fixture
def service(workspace: Path, tmp_path: Path) -> CheckpointService:
    return CheckpointService(ShadowGitStore(workspace, tmp_path / "store" / "checkpoints.git"))


def empty_session() -> AgentSession:
    return make_session(
        id="s_cp",
        entries={},
        conversations={"c1": conversation("c1", [], created_at=500, updated_at=500)},
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )


# ── the index ─────────────────────────────────────────────────────────────────


def test_the_index_round_trips_through_a_serialized_session():
    session = empty_session()
    write_index(session, CheckpointIndex(checkpoints=[Checkpoint(commit="abc", anchor_entry_id="u1", label="hi")]))

    reloaded = AgentSession.model_validate_json(session.model_dump_json())

    assert read_index(reloaded) == CheckpointIndex(
        checkpoints=[Checkpoint(commit="abc", anchor_entry_id="u1", created_at=0, label="hi")]
    )


def test_an_unreadable_index_is_dropped_rather_than_raising():
    session = empty_session()
    session.extras[CHECKPOINTS_KEY] = {"checkpoints": "not a list"}

    assert read_index(session) == CheckpointIndex()


def test_a_session_with_no_index_reads_as_empty():
    assert read_index(empty_session()) == CheckpointIndex()


def test_only_checkpoints_anchored_on_the_current_path_are_offered():
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    index = CheckpointIndex(
        checkpoints=[
            Checkpoint(commit="a", anchor_entry_id="tf1"),
            Checkpoint(commit="b", anchor_entry_id="u0"),  # on the archived c0 only
            Checkpoint(commit="c", anchor_entry_id=None),  # rewind-to-empty is always reachable
        ]
    )

    assert [c.commit for c in restorable(session, index)] == ["c", "a"]


# ── taking a checkpoint ───────────────────────────────────────────────────────


@needs_git
async def test_take_anchors_at_the_current_path_leaf(service: CheckpointService):
    session = RICH_IDLE_SESSION.model_copy(deep=True)

    checkpoint = await service.take(session, label="do the thing")

    assert checkpoint is not None
    assert checkpoint.anchor_entry_id == main_conversation(session).nodes[-1]
    assert checkpoint.label == "do the thing"
    assert read_index(session).checkpoints == [checkpoint]


@needs_git
async def test_take_on_an_empty_conversation_anchors_at_none(service: CheckpointService):
    session = empty_session()

    checkpoint = await service.take(session, label="first")

    assert checkpoint is not None
    assert checkpoint.anchor_entry_id is None


@needs_git
async def test_overlapping_takes_each_keep_their_row(service: CheckpointService):
    """The index is a read-modify-write across an `await`, so two `take()`
    calls overlapping on their snapshot must not lose one another."""
    session = empty_session()

    await asyncio.gather(service.take(session, "one"), service.take(session, "two"))

    recorded = read_index(session).checkpoints
    assert [c.label for c in recorded] == ["one", "two"]
    assert len({c.commit for c in recorded}) == 2


async def test_take_is_a_no_op_when_checkpoints_are_disabled(workspace: Path, tmp_path: Path):
    service = CheckpointService(
        ShadowGitStore(workspace, tmp_path / "store" / "checkpoints.git"),
        enabled=False,
    )
    session = empty_session()

    assert await service.take(session, label="x") is None
    assert service.checkpoints(session) == []
    assert CHECKPOINTS_KEY not in session.extras


# ── restoring: files and conversation together ────────────────────────────────


@needs_git
async def test_restoring_rewinds_the_turn_and_reverts_the_files(service: CheckpointService, workspace: Path):
    faux = FauxProvider()
    faux.set_responses([faux_assistant_message([faux_text("Done.")], finish_reason="stop")])
    session = empty_session()
    runner = DeterministicRunner(session, provider=faux, ids=["u1", "ts", "a1", "tf", "c2"], now=1000)

    checkpoint = await service.take(session, label="edit the file")
    runner.post_message([TextContent(text="edit the file")])
    async with runner.run() as run:
        async for _ in run:
            pass
    (workspace / "kept.py").write_text("the agent changed this\n")

    restored = await service.restore(runner, checkpoint)

    assert restored is True
    assert (workspace / "kept.py").read_text() == "original\n"
    assert main_conversation(session).nodes == []
    assert session.main_conversation_id == "c2"


@needs_git
async def test_the_rewound_turn_is_archived_not_deleted(service: CheckpointService, workspace: Path):
    faux = FauxProvider()
    faux.set_responses([faux_assistant_message([faux_text("Done.")], finish_reason="stop")])
    session = empty_session()
    runner = DeterministicRunner(session, provider=faux, ids=["u1", "ts", "a1", "tf", "c2"], now=1000)

    checkpoint = await service.take(session)
    runner.post_message([TextContent(text="edit the file")])
    async with runner.run() as run:
        async for _ in run:
            pass

    await service.restore(runner, checkpoint)

    assert session.conversations["c1"].nodes == ["u1", "ts", "a1", "tf"]
    assert session.conversations["c2"].previous_conversation_id == "c1"
    assert isinstance(session.entries["u1"], UserMessage)


@needs_git
async def test_restoring_an_unknown_commit_leaves_everything_alone(service: CheckpointService, workspace: Path):
    session = RICH_IDLE_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(session, ids=["c2"], now=1000)
    (workspace / "kept.py").write_text("changed\n")

    restored = await service.restore(runner, Checkpoint(commit="0" * 40, anchor_entry_id="tf2"))

    assert restored is False
    assert session.main_conversation_id == "c1"
    assert (workspace / "kept.py").read_text() == "changed\n"


@needs_git
async def test_restoring_with_an_open_turn_raises_and_leaves_the_files_alone(
    service: CheckpointService,
    workspace: Path,
):
    """`rewind_to`'s guard is the caller's to satisfy. The workspace is rolled
    back to the safety snapshot, so a refused restore leaves nothing
    half-done — neither the session NOR the files move."""
    session = GATED_SESSION.model_copy(deep=True)
    runner = DeterministicRunner(session, ids=["c2"], now=1000)
    checkpoint = await service.take(session)
    (workspace / "kept.py").write_text("changed\n")

    with pytest.raises(AgentError, match="requires a closed turn"):
        await service.restore(runner, checkpoint)

    assert session.main_conversation_id == GATED_SESSION.main_conversation_id
    assert (workspace / "kept.py").read_text() == "changed\n"
