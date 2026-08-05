"""Self-scoped tests for `luca.agent.contrib.plugins`:
`PluginAgentSessionRunner` composes each plugin's contributions at
construction — the directly-passed registry and every plugin registry land as
children of ONE `ProxyToolRegistry` (in order), prompt parts and middleware
extend the constructor lists after the directly-passed items, hooks run once
with the session — and the result compares equal to a directly-configured
`AgentSessionRunner`.

Construction-level only; the composed registry's behavior is covered by
`test_simple_tool_registry.py`. The one listing here is the observable payoff
of composition: `get_tools` is async and answers `ToolSpec`s, so a plugin may
contribute ANY `ToolRegistry` — contrib's `SimpleToolRegistry` over Python
`Tool` classes, or a registry that has none at all.
"""

from pydantic import BaseModel, ConfigDict

from luca.agent.contrib.plugins import BasePlugin, PluginAgentSessionRunner
from luca.agent.contrib.simple_tool_registry import (
    ProxyToolRegistry,
    SimpleToolRegistry,
    YoloPermissionPolicy,
)
from luca.agent.contrib.tools import Tool
from luca.agent.core import (
    AgentSession,
    AgentSessionRunner,
    CancellationToken,
    LLMConfig,
    SessionConfig,
    SystemPromptPart,
    ToolRegistry,
    ToolSpec,
)
from tests.agent.scenarios import (
    conversation,
    main_conversation,
)

MODEL = LLMConfig(model="test-model", provider="faux")


def make_session() -> AgentSession:
    return AgentSession(
        id="s_plugins",
        conversations={
            "c1": conversation(
                "c1",
                [],
                created_at=500,
                updated_at=500,
            )
        },
        main_conversation_id="c1",
        session_config=SessionConfig(llm_config=MODEL),
    )


class NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


NO_ARGS_SCHEMA = {
    "additionalProperties": False,
    "properties": {},
    "title": "NoArgs",
    "type": "object",
}


class PingTool(Tool):
    name = "ping"
    description = "Answer pong."
    Args = NoArgs

    async def _execute(
        self,
        args: dict,
        session: AgentSession,
        conversation_id: str,
        *,
        cancellation_token: CancellationToken,
    ) -> str:
        return "pong"


class EchoTool(PingTool):
    name = "echo"
    description = "Echo back."


# The specs the composed proxy advertises for the tools above — written out
# rather than derived, so the expected listing is the literal wire data.
PING_SPEC = ToolSpec(
    name="ping",
    description="Answer pong.",
    input_schema=NO_ARGS_SCHEMA,
)
ECHO_SPEC = ToolSpec(
    name="echo",
    description="Echo back.",
    input_schema=NO_ARGS_SCHEMA,
)
# What a registry backed by a remote tool server hands back: JSON Schema it
# already has, with no Python tool class anywhere behind it.
REMOTE_SPEC = ToolSpec(
    name="remote",
    description="Runs somewhere else.",
    input_schema={"type": "object", "properties": {}},
)


class RemoteToolRegistry(ToolRegistry):
    """A plugin registry that is NOT contrib's `SimpleToolRegistry`: it owns
    no `Tool` classes and answers the listing with a hand-written `ToolSpec`.
    Only `get_tools` is reached at composition time — the three lifecycle
    methods stay the base's `NotImplementedError`."""

    async def get_tools(self, session: AgentSession, conversation_id: str) -> list[ToolSpec]:
        return [REMOTE_SPEC]


class RecordingMiddleware:
    def before_post_message(self, session, conversation_id, parts: list) -> list:
        return parts


class FullPlugin:
    """Implements every hook; holds ONE registry instance so a directly-
    configured runner can share it (equality compares collaborator state)."""

    def __init__(self) -> None:
        self.registry = SimpleToolRegistry(
            tools=[PingTool()],
            permission_policy=YoloPermissionPolicy(),
        )
        self.middleware = RecordingMiddleware()
        self.sessions_seen: list[AgentSession] = []

    def get_tool_registry(self, agent_session: AgentSession):
        self.sessions_seen.append(agent_session)
        return self.registry

    def get_system_prompt_parts(self, agent_session: AgentSession) -> list:
        return ["plugin part"]

    def get_middleware(self, agent_session: AgentSession) -> list:
        return [self.middleware]


class PartsOnlyPlugin:
    """A plain class implementing a single hook — duck typing skips the rest."""

    def get_system_prompt_parts(self, agent_session: AgentSession) -> list:
        return ["parts only"]


class RegistrylessPlugin:
    """get_tool_registry returning None contributes no registry."""

    def get_tool_registry(self, agent_session: AgentSession):
        return None


class RemotePlugin:
    """Contributes a bare `ToolRegistry` — the hook's contract is the core
    four-method one, not contrib's `Tool`-backed registry."""

    def __init__(self) -> None:
        self.registry = RemoteToolRegistry()

    def get_tool_registry(self, agent_session: AgentSession):
        return self.registry


# ── composition ───────────────────────────────────────────────────────────────


def test_user_registry_and_plugin_registries_land_in_one_proxy_in_order():
    user_registry = SimpleToolRegistry(
        tools=[EchoTool()],
        permission_policy=YoloPermissionPolicy(),
    )
    plugin = FullPlugin()

    runner = PluginAgentSessionRunner(
        make_session(),
        tool_registry=user_registry,
        plugins=[plugin],
    )

    assert type(runner.tool_registry) is ProxyToolRegistry
    assert runner.tool_registry.registries == [user_registry, plugin.registry]


async def test_the_composed_proxy_lists_every_contributed_registrys_tools():
    user_registry = SimpleToolRegistry(
        tools=[EchoTool()],
        permission_policy=YoloPermissionPolicy(),
    )
    session = make_session()
    runner = PluginAgentSessionRunner(
        session,
        tool_registry=user_registry,
        plugins=[FullPlugin(), RemotePlugin()],
    )

    specs = await runner.tool_registry.get_tools(session, main_conversation(session).id)

    assert specs == [ECHO_SPEC, PING_SPEC, REMOTE_SPEC]


def test_without_a_user_registry_the_proxy_holds_only_plugin_registries():
    plugin = FullPlugin()

    runner = PluginAgentSessionRunner(make_session(), plugins=[plugin])

    assert runner.tool_registry.registries == [plugin.registry]


def test_none_registry_contributes_nothing():
    runner = PluginAgentSessionRunner(
        make_session(),
        plugins=[RegistrylessPlugin()],
    )

    assert runner.tool_registry.registries == []


def test_parts_and_middleware_flatten_after_directly_passed_items():
    plugin = FullPlugin()
    direct_middleware = RecordingMiddleware()

    runner = PluginAgentSessionRunner(
        make_session(),
        system_prompt_parts=["direct part"],
        middleware=[direct_middleware],
        plugins=[plugin],
    )

    assert runner.system_prompt_parts == [
        SystemPromptPart(text="direct part"),
        SystemPromptPart(text="plugin part"),
    ]
    assert runner.middleware == [direct_middleware, plugin.middleware]


def test_hooks_are_duck_typed_and_receive_the_session():
    plugin = FullPlugin()
    session = make_session()

    runner = PluginAgentSessionRunner(
        session,
        plugins=[PartsOnlyPlugin(), plugin],
    )

    assert plugin.sessions_seen == [session]
    assert runner.system_prompt_parts == [
        SystemPromptPart(text="parts only"),
        SystemPromptPart(text="plugin part"),
    ]


def test_base_plugin_defaults_contribute_nothing():
    runner = PluginAgentSessionRunner(make_session(), plugins=[BasePlugin()])

    assert runner.tool_registry.registries == []
    assert runner.system_prompt_parts == []
    assert runner.middleware == []


# ── equality with a directly-configured runner ────────────────────────────────


def test_plugin_runner_equals_a_directly_configured_runner():
    user_registry = SimpleToolRegistry(
        tools=[EchoTool()],
        permission_policy=YoloPermissionPolicy(),
    )
    plugin = FullPlugin()
    session = make_session()

    with_plugins = PluginAgentSessionRunner(
        session,
        tool_registry=user_registry,
        system_prompt_parts=["direct part"],
        plugins=[plugin],
    )
    direct = AgentSessionRunner(
        session,
        tool_registry=ProxyToolRegistry(user_registry, plugin.registry),
        system_prompt_parts=["direct part", "plugin part"],
        middleware=[plugin.middleware],
    )

    assert with_plugins == direct
