"""A stored session as `session/update` notifications, for `session/load`.

ACP's contract for loading is that the agent streams the whole conversation
back before it answers the request, so the client can rebuild the thread and
carry on as if nothing had been interrupted.

The replay walks the MAIN conversation's path — `conversations[main].nodes` —
rather than the flat entry store, because the store also holds archived
predecessors from every compaction and rewind. The path is what the agent
would actually send to the model, so it is what the user was last looking at.

Every tool call replays as a `tool_call` followed by one terminal
`tool_call_update`, which is the same pair the live stream produces. A client
that renders the two identically therefore cannot tell a reload from having
been there.
"""

from __future__ import annotations

from acp.helpers import (
    start_tool_call,
    text_block,
    tool_content,
    update_agent_message,
    update_agent_thought,
    update_plan,
    update_tool_call,
    update_user_message,
)

from luca.agent.contrib.memory import is_todo_tool, is_todo_update
from luca.agent.core.models import (
    AgentSession,
    AssistantMessage,
    TextContent,
    ThinkingContent,
    ToolExecution,
    UserMessage,
)
from luca.agent.core.projection import IMAGE_BLOCK_MARKER

from .stream import (
    TOOL_STATUSES,
    plan_entries,
    tool_diffs,
    tool_kind,
    tool_locations,
    tool_title,
)


def _user_blocks(entry: UserMessage) -> list:
    """A user turn's parts as content blocks.

    Images, audio and files replay as a placeholder line rather than as their
    bytes. Re-sending a 16MB screenshot on every reload to redraw one bubble is
    not a trade worth making, and ACP has no way to say "this was an image you
    already saw"."""
    blocks = []
    for part in entry.parts:
        if isinstance(part, TextContent):
            if part.text:
                blocks.append(text_block(part.text))
        else:
            name = getattr(part, "name", None) or (part.metadata or {}).get("name") or part.type
            blocks.append(text_block(f"{IMAGE_BLOCK_MARKER} [{part.type}: {name}]"))
    return blocks


def replay(session: AgentSession) -> list:
    """The whole main conversation as ordered `session/update` payloads."""
    conversation = session.conversations[session.main_conversation_id]
    updates: list = []
    message_index = 0
    for node_id in conversation.nodes:
        entry = session.entries.get(node_id)
        if isinstance(entry, UserMessage):
            for block in _user_blocks(entry):
                chunk = update_user_message(block)
                chunk.message_id = f"user_{node_id}"
                updates.append(chunk)
        elif isinstance(entry, AssistantMessage):
            message_index += 1
            for part in entry.parts:
                if isinstance(part, ThinkingContent):
                    if part.thinking and not part.redacted:
                        thought = update_agent_thought(text_block(part.thinking))
                        thought.message_id = f"msg_{message_index}"
                        updates.append(thought)
                elif isinstance(part, TextContent) and part.text:
                    chunk = update_agent_message(text_block(part.text))
                    chunk.message_id = f"msg_{message_index}"
                    updates.append(chunk)
        elif isinstance(entry, ToolExecution):
            updates.extend(_tool_updates(entry))
    return updates


def _tool_updates(execution: ToolExecution) -> list:
    """One past tool call as the pair the live stream would have sent.

    A todo write replays as the plan it produced, not as a tool row — the same
    rule the live translation follows, so the plan a user sees after a reload
    is the one they had."""
    if is_todo_tool(execution):
        entries = plan_entries(execution) if is_todo_update(execution) else None
        return [update_plan(entries)] if entries else []
    text = execution.result.content[0].text if execution.result and execution.result.content else None
    content = tool_diffs(execution) or ([tool_content(text_block(text))] if isinstance(text, str) and text else None)
    return [
        start_tool_call(
            execution.tool_call_id,
            tool_title(execution),
            kind=tool_kind(execution),
            status="pending",
            raw_input=execution.raw_tool_call.arguments,
        ),
        update_tool_call(
            execution.tool_call_id,
            status=TOOL_STATUSES.get(execution.status, "failed"),
            content=content,
            locations=tool_locations(execution) or None,
        ),
    ]


def is_replayable(session: AgentSession) -> bool:
    """Whether the session has anything worth streaming back. A session saved
    before its first turn has an empty path and replays as nothing, which is
    correct and not an error."""
    conversation = session.conversations.get(session.main_conversation_id)
    return bool(conversation and conversation.nodes)


__all__ = ["is_replayable", "replay"]
