"""The protocol surface, driven end to end against the offline conversation.

`--faux` scripts a turn that exercises most of the mapping at once: thinking, a
gated `multiply` call, a subagent spawn, the child's own tool call, and the
wrap-up text. So a single `session/prompt` here covers the pieces that would
otherwise need a live model.
"""

from __future__ import annotations

import pytest
from acp import RequestError
from acp.helpers import text_block
from acp.schema import ClientCapabilities, ElicitationCapabilities, Implementation

from luca.agent.contrib.acp import LucaAgent

from .conftest import RecordingClient


async def connected(agent_factory, client: RecordingClient, **overrides) -> LucaAgent:
    agent = agent_factory(**overrides)
    agent.on_connect(client)
    return agent


# ── initialize ───────────────────────────────────────────────────────────────


async def test_initialize_advertises_what_luca_can_actually_take(agent):
    client = RecordingClient()
    instance = await connected(agent, client)

    response = await instance.initialize(protocol_version=1, client_capabilities=ClientCapabilities())

    assert response.protocol_version == 1
    assert response.agent_capabilities.load_session is True
    assert (
        response.agent_capabilities.prompt_capabilities.image,
        response.agent_capabilities.prompt_capabilities.audio,
        response.agent_capabilities.prompt_capabilities.embedded_context,
    ) == (True, True, True)
    assert response.auth_methods == []
    assert response.agent_info.name == "luca"


async def test_initialize_never_claims_a_version_above_its_own(agent):
    """A client asking for 2 gets 1 back, which is the negotiation the spec
    describes: answer with your own latest when theirs is not one you speak."""
    instance = await connected(agent, RecordingClient())

    response = await instance.initialize(protocol_version=2)

    assert response.protocol_version == 1


# ── session/new ──────────────────────────────────────────────────────────────


async def test_new_session_returns_an_id_and_the_two_modes(agent, workspace):
    instance = await connected(agent, RecordingClient())
    await instance.initialize(protocol_version=1)

    response = await instance.new_session(cwd=str(workspace))

    assert response.session_id
    assert response.modes.current_mode_id == "ask"
    assert [mode.id for mode in response.modes.available_modes] == ["ask", "yolo"]


async def test_new_session_ignores_mcp_servers_and_says_so(agent, workspace, caplog):
    """Nothing in luca speaks MCP yet. Dropping the servers silently would make
    a user's configured tools vanish with no explanation anywhere."""
    instance = await connected(agent, RecordingClient())
    await instance.initialize(protocol_version=1)

    with caplog.at_level("WARNING", logger="luca.agent.contrib.acp.agent"):
        response = await instance.new_session(
            cwd=str(workspace),
            mcp_servers=[{"name": "fs", "command": "/bin/true", "args": [], "env": []}],
        )

    assert response.session_id
    assert "does not speak MCP" in caplog.text


async def test_a_session_is_saved_the_moment_it_is_created(agent, workspace, tmp_path):
    """So `session/load` can find it, and a client that creates a session and
    crashes has not lost it."""
    instance = await connected(agent, RecordingClient())
    await instance.initialize(protocol_version=1)

    response = await instance.new_session(cwd=str(workspace))

    stored = list((tmp_path / "home" / ".luca" / "projects").rglob(f"{response.session_id}.json"))
    assert len(stored) == 1


# ── session/prompt ───────────────────────────────────────────────────────────


async def test_a_prompt_turn_streams_text_thinking_and_tool_calls(agent, workspace):
    client = RecordingClient(approve="allow_once")
    instance = await connected(agent, client)
    await instance.initialize(protocol_version=1)
    session = (await instance.new_session(cwd=str(workspace))).session_id

    response = await instance.prompt(session_id=session, prompt=[text_block("what is 6 times 7?")])

    assert response.stop_reason == "end_turn"
    assert "42" in client.text()
    assert "multiply" in client.thoughts()
    assert [update.tool_call_id for update in client.of("tool_call")]


async def test_a_gated_call_asks_the_client_and_then_runs(agent, workspace):
    client = RecordingClient(approve="allow_once")
    instance = await connected(agent, client)
    await instance.initialize(protocol_version=1)
    session = (await instance.new_session(cwd=str(workspace))).session_id

    await instance.prompt(session_id=session, prompt=[text_block("what is 6 times 7?")])

    assert client.permission_requests, "the multiply call is gated and should have been asked about"
    tool_call, options = client.permission_requests[0]
    assert tool_call.title.startswith("multiply")
    assert [option.kind for option in options] == ["allow_once", "reject_once"]


async def test_cancelling_at_the_permission_prompt_ends_the_turn(agent, workspace):
    """ACP models "stop everything" as the request's outcome, not as an option
    the agent offers, so a `cancelled` outcome winds the turn down."""
    client = RecordingClient(approve=None)
    instance = await connected(agent, client)
    await instance.initialize(protocol_version=1)
    session = (await instance.new_session(cwd=str(workspace))).session_id

    response = await instance.prompt(session_id=session, prompt=[text_block("what is 6 times 7?")])

    assert response.stop_reason == "cancelled"


async def test_an_empty_prompt_is_rejected_rather_than_posted(agent, workspace):
    instance = await connected(agent, RecordingClient())
    await instance.initialize(protocol_version=1)
    session = (await instance.new_session(cwd=str(workspace))).session_id

    with pytest.raises(RequestError):
        await instance.prompt(session_id=session, prompt=[])


async def test_prompting_an_unknown_session_is_an_invalid_param(agent):
    instance = await connected(agent, RecordingClient())
    await instance.initialize(protocol_version=1)

    with pytest.raises(RequestError):
        await instance.prompt(session_id="nope", prompt=[text_block("hi")])


# ── subagents ────────────────────────────────────────────────────────────────


async def test_a_subagents_output_arrives_on_the_call_that_spawned_it(agent, workspace):
    """ACP has one stream per session. The child's text must not appear as a
    second voice; it belongs to the spawn call, as progress on it."""
    client = RecordingClient(approve="allow_once")
    instance = await connected(agent, client)
    await instance.initialize(protocol_version=1)
    session = (await instance.new_session(cwd=str(workspace))).session_id

    await instance.prompt(session_id=session, prompt=[text_block("what is 6 times 7?")])

    spawns = [update for update in client.of("tool_call") if update.title.startswith("spawn_subagent")]
    assert spawns, "the faux script spawns a subagent"
    folded = [
        update
        for update in client.of("tool_call_update")
        if update.tool_call_id == spawns[0].tool_call_id and update.content
    ]
    assert folded, "the child's output should have been folded onto the spawn call"
    assert "Confirmed" in "".join(
        block.content.text for update in folded for block in update.content if block.type == "content"
    )


# ── session/set_mode ─────────────────────────────────────────────────────────


async def test_switching_to_yolo_moves_the_live_strategy(agent, workspace):
    instance = await connected(agent, RecordingClient())
    await instance.initialize(protocol_version=1)
    session = (await instance.new_session(cwd=str(workspace))).session_id

    await instance.set_session_mode(session_id=session, mode_id="yolo")

    application = instance._sessions[session]
    assert (application.mode, application.strategy.mode.value) == ("yolo", "yolo")


async def test_an_unknown_mode_is_rejected(agent, workspace):
    instance = await connected(agent, RecordingClient())
    await instance.initialize(protocol_version=1)
    session = (await instance.new_session(cwd=str(workspace))).session_id

    with pytest.raises(RequestError):
        await instance.set_session_mode(session_id=session, mode_id="reckless")


# ── session/load ─────────────────────────────────────────────────────────────


async def test_loading_replays_the_conversation_before_answering(agent, workspace):
    client = RecordingClient(approve="allow_once")
    instance = await connected(agent, client)
    await instance.initialize(protocol_version=1)
    session = (await instance.new_session(cwd=str(workspace))).session_id
    await instance.prompt(session_id=session, prompt=[text_block("what is 6 times 7?")])

    reader = RecordingClient()
    second = await connected(agent, reader)
    await second.initialize(protocol_version=1)
    response = await second.load_session(cwd=str(workspace), session_id=session)

    assert response.modes.current_mode_id == "ask"
    assert [update.content.text for update in reader.of("user_message_chunk")] == ["what is 6 times 7?"]
    assert "42" in "".join(update.content.text for update in reader.of("agent_message_chunk"))


async def test_loading_an_unknown_session_fails_rather_than_inventing_one(agent, workspace):
    instance = await connected(agent, RecordingClient())
    await instance.initialize(protocol_version=1)

    with pytest.raises(FileNotFoundError):
        await instance.load_session(cwd=str(workspace), session_id="deadbeef")


# ── cancellation ─────────────────────────────────────────────────────────────


async def test_cancel_on_an_unknown_session_is_ignored(agent):
    """`session/cancel` is a notification: there is nobody to report an error
    to, and a client racing a cancel against a closed session is normal."""
    instance = await connected(agent, RecordingClient())

    assert await instance.cancel(session_id="nope") is None


# ── elicitation ──────────────────────────────────────────────────────────────


async def test_elicitation_is_only_used_when_the_client_offers_it(agent, workspace):
    client = RecordingClient()
    instance = await connected(agent, client)

    await instance.initialize(
        protocol_version=1,
        client_capabilities=ClientCapabilities(elicitation=ElicitationCapabilities()),
    )

    assert instance._client_capabilities.elicitation is not None


async def test_client_info_is_accepted_and_ignored(agent):
    instance = await connected(agent, RecordingClient())

    response = await instance.initialize(
        protocol_version=1,
        client_info=Implementation(name="zed", version="1.0"),
    )

    assert response.protocol_version == 1
