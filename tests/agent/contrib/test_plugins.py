"""Self-scoped tests for `luca.agent.contrib.plugins`:
`PluginAgentSessionRunner` composes each plugin's contributions at
construction — the directly-passed registry and every plugin registry land as
children of ONE `ProxyToolRegistry` (in order), prompt parts and middleware
extend the constructor lists after the directly-passed items, hooks run once
with the session — and the result compares equal to a directly-configured
`AgentSessionRunner`. Construction-level only; the composed registry's
behavior is covered by `test_simple_tool_registry.py`.
"""

from pydantic import BaseModel, ConfigDict

from luca.agent.contrib.plugins import BasePlugin, PluginAgentSessionRunner
from luca.agent.contrib.simple_tool_registry import (
    ProxyToolRegistry,
    SimpleToolRegistry,
    YoloPermissionPolicy,
)
from luca.agent.core import (
    AgentSession,
    AgentSessionRunner,
    CancellationToken,
    Conversation,
    LLMConfig,
    SessionConfig,
    SystemPromptPart,
    Tool,
    ToolContext,
)

MODEL = LLMConfig(model="test-model", provider="faux")


def make_session() -> AgentSession:
    return AgentSession(
        id="s_plugins",
        active_conversation=Conversation(
            id="c1", nodes=[], created_at=500, updated_at=500,
        ),
        session_config=SessionConfig(llm_config=MODEL),
    )


class NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PingTool(Tool):
    name = "ping"
    description = "Answer pong."
    Args = NoArgs

    async def _execute(
        self, args: dict, context: ToolContext,
        *, cancellation_token: CancellationToken,
    ) -> str:
        return "pong"


class EchoTool(PingTool):
    name = "echo"
    description = "Echo back."


class RecordingMiddleware:
    def before_post_message(self, parts: list) -> list:
        return parts


class FullPlugin:
    """Implements every hook; holds ONE registry instance so a directly-
    configured runner can share it (equality compares collaborator state)."""

    def __init__(self) -> None:
        self.registry = SimpleToolRegistry(
            tools=[PingTool()], permission_policy=YoloPermissionPolicy(),
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


# ── composition ───────────────────────────────────────────────────────────────


def test_user_registry_and_plugin_registries_land_in_one_proxy_in_order():
    user_registry = SimpleToolRegistry(
        tools=[EchoTool()], permission_policy=YoloPermissionPolicy(),
    )
    plugin = FullPlugin()
    session = make_session()

    runner = PluginAgentSessionRunner(
        session, tool_registry=user_registry, plugins=[plugin],
    )

    assert type(runner.tool_registry) is ProxyToolRegistry
    assert runner.tool_registry.registries == [user_registry, plugin.registry]
    assert [tool.name for tool in runner.tool_registry.get_tools(session)] == [
        "echo", "ping",
    ]


def test_without_a_user_registry_the_proxy_holds_only_plugin_registries():
    plugin = FullPlugin()

    runner = PluginAgentSessionRunner(make_session(), plugins=[plugin])

    assert runner.tool_registry.registries == [plugin.registry]


def test_none_registry_contributes_nothing():
    runner = PluginAgentSessionRunner(
        make_session(), plugins=[RegistrylessPlugin()],
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
        session, plugins=[PartsOnlyPlugin(), plugin],
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
        tools=[EchoTool()], permission_policy=YoloPermissionPolicy(),
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
