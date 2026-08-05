"""Middleware across a subagent tree: one instance, two conversations.

This is the whole reason every hook carries `(session, conversation_id)`. A
middleware instance is shared by the runner and serves the main conversation
AND every subagent concurrently. Without the id a hook cannot tell whose
operation it is looking at — an application middleware written for a single
conversation would get no error, just wrong behavior.

What these pin: the id reaching each hook is the conversation whose operation
invoked it (never "the main one", never the last one to run), and the session
is the LIVE runner-held object rather than a copy.
"""

from pydantic import BaseModel, ConfigDict

from luca.agent.core.models import ExecutionStatus
from tests.agent.scenarios import FakeTool
from tests.agent.subagents.conftest import (
    DeterministicRunner,
    SubagentRegistry,
    faux_assistant_message,
    faux_text,
    faux_tool_call,
    spawn_call,
    subagent_session,
)

IDS = [f"x{n}" for n in range(60)]


class _NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProbeTool(FakeTool):
    """A plain, non-spawn tool, so a subagent's own conversation contains a
    tool outcome that is unambiguously the CHILD's."""

    name = "probe"
    description = "Probe."
    Args = _NoArgs

    async def _execute(self, args, session, conversation_id, *, cancellation_token) -> str:
        return "probed"


class ScopeRecorder:
    """Records `(hook, conversation_id)` for the hooks a spawn round drives,
    and every session identity it was handed."""

    def __init__(self) -> None:
        self.trace: list[tuple[str, str]] = []
        self.sessions: list[int] = []

    def _note(self, hook, session, conversation_id):
        self.trace.append((hook, conversation_id))
        self.sessions.append(id(session))

    def build_model_string(self, session, conversation_id, model_string, llm_cfg):
        self._note("build_model_string", session, conversation_id)
        return model_string

    def build_tool_list(self, session, conversation_id, tools):
        self._note("build_tool_list", session, conversation_id)
        return tools

    def before_llm_call(self, session, conversation_id, messages, system_message):
        self._note("before_llm_call", session, conversation_id)
        return messages, system_message

    def after_llm_response(self, session, conversation_id, message):
        self._note("after_llm_response", session, conversation_id)
        return message

    def before_tool_creation(self, session, conversation_id, call):
        self._note("before_tool_creation", session, conversation_id)
        return call

    def after_tool_creation(self, session, conversation_id, execution, exception=None):
        self._note("after_tool_creation", session, conversation_id)
        return execution

    def before_tool_execution(self, session, conversation_id, execution):
        self._note("before_tool_execution", session, conversation_id)
        return execution

    def after_tool_execution(self, session, conversation_id, execution, exception=None):
        self._note("after_tool_execution", session, conversation_id)
        return execution


async def test_every_hook_receives_the_conversation_whose_operation_it_serves(faux):
    # One spawn round: the parent calls the model and dispatches the spawn
    # tool, the child then runs its own model round, and the parent is woken
    # by the resolution. Every hook must be attributed to the conversation
    # that actually performed the operation.
    faux.set_responses(
        [
            faux_assistant_message(
                [spawn_call("Research A", "research A", call_id="tc1", task_id="t1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("A is fine.")], finish_reason="stop"),
            faux_assistant_message([faux_text("A is fine, then.")], finish_reason="stop"),
        ]
    )
    recorder = ScopeRecorder()
    session = subagent_session()
    runner = DeterministicRunner(
        session,
        tool_registry=SubagentRegistry(),
        provider=faux,
        ids=list(IDS),
        now=1000,
        middleware=[recorder],
    )
    runner.post_message("go")

    async with runner.run() as run:
        _ = [event async for event in run]

    child_id = next(cid for cid in runner.session.conversations if cid != "c1")
    conversations = {cid for _, cid in recorder.trace}

    assert conversations == {"c1", child_id}
    # every hook saw the SAME live session object the runner holds
    assert set(recorder.sessions) == {id(runner.session)}
    # the child's own model round is attributed to the child, and the private
    # result execution the runner mints to resolve it belongs to the PARENT —
    # it is the parent's path the update lands on
    assert ("before_llm_call", child_id) in recorder.trace
    assert ("after_llm_response", child_id) in recorder.trace
    assert [hook for hook, cid in recorder.trace if cid == child_id] == [
        "build_model_string",
        "before_llm_call",
        "build_tool_list",
        "after_llm_response",
    ]
    # the spawn tool call is the parent's throughout its lifecycle
    assert [
        hook
        for hook, cid in recorder.trace
        if cid == "c1" and hook in ("before_tool_creation", "after_tool_creation", "before_tool_execution")
    ] == [
        "before_tool_creation",  # the spawn call
        "after_tool_creation",
        "before_tool_execution",
        "before_tool_creation",  # the runtime-minted result call
        "after_tool_creation",
        "before_tool_execution",
    ]


async def test_a_middleware_can_route_a_subagent_to_a_different_model(faux):
    # The practical payoff of a conversation-scoped `build_model_string`:
    # cheap model for the children, the configured one for the main
    # conversation. Impossible before the id existed.
    faux.set_responses(
        [
            faux_assistant_message(
                [spawn_call("Research A", "research A", call_id="tc1", task_id="t1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("A is fine.")], finish_reason="stop"),
            faux_assistant_message([faux_text("A is fine, then.")], finish_reason="stop"),
        ]
    )

    class CheapSubagents:
        def build_model_string(self, session, conversation_id, model_string, llm_cfg):
            if session.conversations[conversation_id].depth > 0:
                return "openai:gpt-5-nano"
            return model_string

    session = subagent_session()
    runner = DeterministicRunner(
        session,
        tool_registry=SubagentRegistry(),
        provider=faux,
        ids=list(IDS),
        now=1000,
        middleware=[CheapSubagents()],
    )
    runner.post_message("go")

    async with runner.run() as run:
        _ = [event async for event in run]

    child_id = next(cid for cid in runner.session.conversations if cid != "c1")
    # the child's assistant entries record the routed model; the parent's do not
    child_nodes = runner.session.conversations[child_id].nodes
    child_assistants = [
        runner.session.entries[node] for node in child_nodes if runner.session.entries[node].type == "assistant"
    ]
    main_assistants = [
        runner.session.entries[node]
        for node in runner.session.conversations["c1"].nodes
        if runner.session.entries[node].type == "assistant"
    ]

    assert [(a.llm_config.provider, a.llm_config.model) for a in child_assistants] == [("openai", "gpt-5-nano")]
    assert {(a.llm_config.provider, a.llm_config.model) for a in main_assistants} == {("faux", "test-model")}


async def test_before_entry_written_attributes_each_write_to_its_conversation(faux):
    # An entry may be referenced by more than one conversation; the id says
    # which conversation's OPERATION caused this write, which is what a hook
    # doing per-conversation bookkeeping needs.
    faux.set_responses(
        [
            faux_assistant_message(
                [spawn_call("Research A", "research A", call_id="tc1", task_id="t1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("A is fine.")], finish_reason="stop"),
            faux_assistant_message([faux_text("A is fine, then.")], finish_reason="stop"),
        ]
    )

    class WriteRecorder:
        def __init__(self) -> None:
            self.seen: list[tuple[str, str]] = []

        def before_entry_written(self, session, conversation_id, entry):
            self.seen.append((entry.type, conversation_id))
            return entry

    recorder = WriteRecorder()
    session = subagent_session()
    runner = DeterministicRunner(
        session,
        tool_registry=SubagentRegistry(),
        provider=faux,
        ids=list(IDS),
        now=1000,
        middleware=[recorder],
    )
    runner.post_message("go")

    async with runner.run() as run:
        _ = [event async for event in run]

    child_id = next(cid for cid in runner.session.conversations if cid != "c1")

    # the child's seed message and its whole turn are written under the child
    assert ("user", child_id) in recorder.seen
    assert ("assistant", child_id) in recorder.seen
    assert ("turn_start", child_id) in recorder.seen
    assert ("turn_finish", child_id) in recorder.seen
    # the link into the parent's path is the PARENT's write, both times (it is
    # appended unresolved, then mutated when the child finishes)
    assert [cid for kind, cid in recorder.seen if kind == "child_conversation"] == ["c1", "c1"]
    # and no write is attributed to a conversation that does not exist
    assert {cid for _, cid in recorder.seen} == {"c1", child_id}


async def test_a_subagents_tool_outcome_carries_the_child_id(faux):
    # `after_tool_execution` is where per-conversation cost/telemetry lands.
    # A tool dispatched inside a subagent must not report as the parent's.
    faux.set_responses(
        [
            faux_assistant_message(
                [spawn_call("Research A", "research A", call_id="tc1", task_id="t1")],
                finish_reason="tool_use",
            ),
            faux_assistant_message(
                [faux_text("checking"), faux_tool_call("probe", {}, id="tc9")],
                finish_reason="tool_use",
            ),
            faux_assistant_message([faux_text("A is fine.")], finish_reason="stop"),
            faux_assistant_message([faux_text("A is fine, then.")], finish_reason="stop"),
        ]
    )

    class OutcomeRecorder:
        def __init__(self) -> None:
            self.seen: list[tuple[str, str, ExecutionStatus]] = []

        def after_tool_execution(self, session, conversation_id, execution, exception=None):
            self.seen.append((execution.raw_tool_call.name, conversation_id, execution.status))
            return execution

    recorder = OutcomeRecorder()
    session = subagent_session()
    runner = DeterministicRunner(
        session,
        tool_registry=SubagentRegistry([ProbeTool()]),
        provider=faux,
        ids=list(IDS),
        now=1000,
        middleware=[recorder],
    )
    runner.post_message("go")

    async with runner.run() as run:
        _ = [event async for event in run]

    child_id = next(cid for cid in runner.session.conversations if cid != "c1")

    assert ("probe", child_id, ExecutionStatus.COMPLETED) in recorder.seen
    assert [cid for name, cid, _ in recorder.seen if name == "probe"] == [child_id]
