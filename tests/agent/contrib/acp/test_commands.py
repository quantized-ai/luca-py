"""Slash commands: parsing, the registry, and the round trip through a client.

ACP has no invoke method. The agent advertises a list, the client sends
`/name args` back as ordinary prompt text, and the agent parses it. So the
tests come in two halves: the parse and the registry as plain functions, then
the whole path through `session/prompt`.
"""

from __future__ import annotations

import pytest
from acp.helpers import text_block

from luca.agent.contrib.acp import commands as slash
from luca.agent.contrib.acp.commands import Invocation, Prompt

from .conftest import RecordingClient
from .test_agent import connected


def write_command(workspace, name: str, body: str) -> None:
    directory = workspace / ".claude" / "commands"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.md").write_text(body)


# ── parsing ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("/help", Invocation(name="help", args="")),
        ("/review the parser", Invocation(name="review", args="the parser")),
        ("/my-command arg", Invocation(name="my-command", args="arg")),
        ("/my_command arg", Invocation(name="my_command", args="arg")),
        ("/help\nand more", Invocation(name="help", args="and more")),
    ],
)
def test_a_leading_slash_name_parses(text, expected):
    assert slash.parse(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "what does /help do?",
        " /help",  # a leading space is the documented escape hatch
        "//not a command",
        "/",
        "/tmp/notes.txt is the file",  # a path, not a command
        "no slash at all",
    ],
)
def test_ordinary_prose_is_not_a_command(text):
    assert slash.parse(text) is None


# ── the registry ─────────────────────────────────────────────────────────────


def test_the_built_ins_are_always_there(workspace):
    names = [command.name for command in slash.build(workspace)]

    assert names == ["compact", "cost", "help"]


def test_a_user_command_joins_the_registry_with_its_hint(workspace):
    write_command(
        workspace,
        "review",
        "---\ndescription: review a file\nargument-hint: '[path]'\n---\nReview $ARGUMENTS carefully.",
    )

    [command] = [c for c in slash.build(workspace) if c.name == "review"]

    assert (command.name, command.description, command.hint) == ("review", "review a file", "[path]")


def test_a_user_command_cannot_shadow_a_built_in(workspace):
    """A stray file in a cloned repo must not redefine /compact."""
    write_command(workspace, "compact", "Do something else entirely.")

    compacts = [c for c in slash.build(workspace) if c.name == "compact"]

    assert len(compacts) == 1
    assert compacts[0].description == "summarize the history and continue"


def test_commands_can_be_turned_off_entirely(workspace):
    write_command(workspace, "review", "Review it.")

    assert slash.build(workspace, enabled=False) == ()


def test_help_lists_every_command_including_itself_and_the_users(workspace):
    write_command(workspace, "review", "---\ndescription: review a file\n---\nReview it.")
    registry = slash.build(workspace)

    [help_command] = [c for c in registry if c.name == "help"]
    text = help_command.run(None, "").text

    for name in ("compact", "cost", "help", "review"):
        assert f"/{name}" in text


# ── the advertisement ────────────────────────────────────────────────────────


def test_the_advertisement_carries_a_hint_only_when_there_is_one(workspace):
    write_command(workspace, "review", "---\ndescription: review\nargument-hint: '[path]'\n---\nReview $ARGUMENTS.")
    advertised = {c.name: c for c in slash.advertisement(slash.build(workspace))}

    assert advertised["review"].input.root.hint == "[path]"
    assert advertised["compact"].input is None


# ── dispatch ─────────────────────────────────────────────────────────────────


def test_a_user_command_expands_into_a_prompt(workspace):
    write_command(workspace, "review", "Review $ARGUMENTS for bugs.")
    registry = slash.build(workspace)

    outcome = slash.dispatch(None, registry, Invocation(name="review", args="parser.py"))

    assert outcome == Prompt("Review parser.py for bugs.")


def test_an_unknown_name_dispatches_to_nothing(workspace):
    """Not an error: prose that starts with a slash still belongs to the
    model, and the client already refuses the names it knows nothing about."""
    assert slash.dispatch(None, slash.build(workspace), Invocation(name="nope", args="")) is None


# ── through the protocol ─────────────────────────────────────────────────────


async def test_a_client_is_told_what_commands_exist(agent, workspace):
    write_command(workspace, "review", "---\ndescription: review a file\n---\nReview it.")
    client = RecordingClient()
    instance = await connected(agent, client)
    await instance.initialize(protocol_version=1)

    await instance.new_session(cwd=str(workspace))

    [update] = client.of("available_commands_update")
    assert [c.name for c in update.available_commands] == ["compact", "cost", "help", "review"]


async def test_loading_a_session_re_advertises_them(agent, workspace):
    """A reload is a fresh client with an empty palette; without this the
    commands vanish on every restart."""
    client = RecordingClient()
    instance = await connected(agent, client)
    await instance.initialize(protocol_version=1)
    session = (await instance.new_session(cwd=str(workspace))).session_id

    reader = RecordingClient()
    second = await connected(agent, reader)
    await second.initialize(protocol_version=1)
    await second.load_session(cwd=str(workspace), session_id=session)

    assert reader.of("available_commands_update")


async def test_a_reporting_command_answers_without_calling_the_model(agent, workspace):
    client = RecordingClient()
    instance = await connected(agent, client)
    await instance.initialize(protocol_version=1)
    session = (await instance.new_session(cwd=str(workspace))).session_id

    response = await instance.prompt(session_id=session, prompt=[text_block("/cost")])

    assert response.stop_reason == "end_turn"
    assert "total" in client.text()
    # The faux script would have produced a tool call had the model been asked.
    assert client.of("tool_call") == []


async def test_a_user_command_reaches_the_model_as_its_expansion(agent, workspace):
    write_command(workspace, "ask", "What is $ARGUMENTS?")
    client = RecordingClient(approve="allow_once")
    instance = await connected(agent, client)
    await instance.initialize(protocol_version=1)
    session = (await instance.new_session(cwd=str(workspace))).session_id

    await instance.prompt(session_id=session, prompt=[text_block("/ask 6 times 7")])

    posted = instance._sessions[session].session
    user_text = [
        part.text
        for entry in posted.entries.values()
        for part in getattr(entry, "parts", [])
        if getattr(part, "type", None) == "text"
    ]
    assert "What is 6 times 7?" in user_text


async def test_an_unknown_command_is_sent_to_the_model_rather_than_refused(agent, workspace):
    client = RecordingClient(approve="allow_once")
    instance = await connected(agent, client)
    await instance.initialize(protocol_version=1)
    session = (await instance.new_session(cwd=str(workspace))).session_id

    response = await instance.prompt(session_id=session, prompt=[text_block("/nonexistent do a thing")])

    assert response.stop_reason == "end_turn"
    assert client.text()


async def test_the_no_commands_flag_advertises_an_empty_list(agent, workspace):
    write_command(workspace, "review", "Review it.")
    client = RecordingClient()
    instance = await connected(agent, client, commands=False)
    await instance.initialize(protocol_version=1)

    await instance.new_session(cwd=str(workspace))

    [update] = client.of("available_commands_update")
    assert update.available_commands == []
