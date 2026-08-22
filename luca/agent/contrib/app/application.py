"""`AgentApplication` — one configured agent, with no front end attached.

What a front end needs before it can render a single character: the composed
runner, the permission strategy every gate is answered through, the
`ask_user` plugin a deferred call is resolved on, the checkpoint service, the
user's slash commands, and the two halves of persistence (the session, and the
sidecar the session does not own).

This is the object that used to be `AgentApp.__init__`. It moved out because a
`textual.App` subclass is a bad place to keep the answer to "which agent am I
driving" — the TUI is one front end, the ACP server is another, and the
composition is identical for both.

WHAT IS NOT HERE: the drive loop. Advancing the agent means rendering events,
asking approval questions and collecting answers, all of which are the front
end's own business and look nothing alike between a terminal and a JSON-RPC
socket. This object hands you `runner`, `strategy` and `questions`; what you do
with them is yours.

LIFETIME. The service outlives any single session: `/clear` and `/resume` swap
the runner, not the workspace, so `use(session)` rebuilds the runner in place
and the checkpoint store carries over.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from luca.agent.contrib.checkpoints import CheckpointService, ShadowGitStore
from luca.agent.contrib.plugins import PluginAgentSessionRunner
from luca.agent.contrib.questions import QuestionsPlugin
from luca.agent.contrib.resource_permissions import PermissionStrategy
from luca.agent.core.models import AgentSession, LLMConfig

from .auth import api_key_for
from .sessions import (
    QUESTIONS_STORE_KEY,
    load_app_state,
    save_app_state,
    save_session,
)
from .wiring import SCRATCHPAD_STORE_KEY, TODO_STORE_KEY, build_runner


class AgentApplication:
    """Compose one agent and keep it. `runner`, `strategy` and `questions` are
    the three references a driver works from; they are re-bound by `use()` and
    must therefore be read off this object rather than cached."""

    def __init__(
        self,
        session: AgentSession,
        *,
        provider=None,
        auth: dict | None = None,
        workspace: str | os.PathLike[str] = ".",
        session_dir: str | os.PathLike[str] = ".",
        mode: str = "ask",
        context_manager=None,
        additional_directories: list | None = None,
        permission_rules: list | None = None,
        model_options: Callable[[LLMConfig], LLMConfig] | None = None,
        subagents: bool = True,
        skills: bool = True,
        extra_skill_locations: list[str] | None = None,
        instructions: bool = True,
        extra_instructions: list[str] | None = None,
        checkpoints: bool = True,
    ) -> None:
        self.workspace = workspace
        self.session_dir = Path(session_dir)
        self.provider = provider
        # `auth.json`, read once at boot. Kept as the whole map rather than as
        # one resolved key because a model switch can move the session to
        # another provider mid-run, and the key has to follow it.
        self.auth = dict(auth or {})
        self.mode = mode
        self.context_manager = context_manager
        self.additional_directories = additional_directories
        self.permission_rules = permission_rules
        # A callable, not the LucaConfig: a front end needs "re-resolve options
        # for this pair", not the file the answer came from. Identity when
        # nothing configured any.
        self.model_options = model_options or (lambda llm_config: llm_config)
        self.subagents = subagents
        self.skills = skills
        self.extra_skill_locations = extra_skill_locations
        self.instructions = instructions
        self.extra_instructions = extra_instructions
        # One shadow repo per PROJECT, beside that project's sessions. Built
        # here rather than in `use()` because it outlives any single session.
        self.checkpoints = CheckpointService(
            ShadowGitStore(workspace, self.session_dir / "checkpoints.git"),
            enabled=checkpoints,
        )
        # NO CUSTOM COMMANDS HERE. Discovery is one call —
        # `discover_commands(resolve_locations(workspace, extra))` — but the
        # `reserved` set that stops a user shadowing a built-in is the front
        # end's own vocabulary, and what a command becomes afterwards is too
        # (the TUI turns each into a `SlashCommand`). Both live in
        # `custom_commands`, which every front end can reach; neither belongs
        # to the composition.
        self._app_state: dict = {}
        self._questions_store: dict = {}
        self.runner: PluginAgentSessionRunner
        self.strategy: PermissionStrategy
        self.questions: QuestionsPlugin
        self.use(session)

    # ── composition ───────────────────────────────────────────────────────────

    def use(self, session: AgentSession) -> None:
        """Drive this session from now on, rebuilding the runner around it, and
        load ITS sidecar.

        The question store lives in the sidecar rather than on
        `AgentSession.extras` because outstanding questions are the INTERFACE's
        state: another driver loading this session has no use for a prompt only
        this front end knows how to render. Losing the sidecar is survivable —
        `ask_user` re-seeds from `raw_tool_call.arguments` and defers again, so
        the user is asked a second time rather than left wedged."""
        self._app_state = load_app_state(session.id, self.session_dir)
        # The store is this object's own reference, NOT a key `setdefault`-ed
        # into `_app_state` — that would make the state permanently truthy and
        # write a sidecar beside every session that never asked a question.
        # `save()` files it back only when it holds something.
        stored = self._app_state.get(QUESTIONS_STORE_KEY)
        self._questions_store = stored if isinstance(stored, dict) else {}
        self.runner, self.strategy, self.questions = build_runner(
            session,
            workspace=self.workspace,
            provider=self.provider,
            api_key=self.api_key_for(session.session_config.llm_config.provider),
            mode=self.mode,
            context_manager=self.context_manager,
            additional_directories=self.additional_directories,
            extra_rules=self.permission_rules,
            subagents=self.subagents,
            skills=self.skills,
            extra_skill_locations=self.extra_skill_locations,
            instructions=self.instructions,
            extra_instructions=self.extra_instructions,
            questions_store=self._questions_store,
        )

    @property
    def session(self) -> AgentSession:
        return self.runner.session

    @property
    def questions_store(self) -> dict:
        """`QuestionsTool`'s job store for the live session, as the sidecar
        holds it. Exposed so a front end can persist an answer the moment it is
        given rather than at the end of the drive."""
        return self._questions_store

    # ── credentials ───────────────────────────────────────────────────────────

    def api_key_for(self, provider: str) -> str | None:
        """This provider's key from `auth.json`, or None to let the client fall
        back to its own environment variable."""
        return api_key_for(self.auth, provider)

    def repoint_api_key(self, provider: str) -> None:
        """Move the live runner (and its context manager) onto another
        provider's credential. Called on every model switch: a session
        configured for openrouter must stop sending openrouter's key the moment
        it is pointed at openai."""
        key = self.api_key_for(provider)
        self.runner.api_key = key
        if self.context_manager is not None:
            self.context_manager.api_key = key

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist both halves of a session: the conversation and the sidecar.
        Called wherever `save_session` alone would be, so the two can never
        drift out of step."""
        save_session(self.session, self.session_dir)
        state = dict(self._app_state)
        if self._questions_store:
            state[QUESTIONS_STORE_KEY] = self._questions_store
        else:
            state.pop(QUESTIONS_STORE_KEY, None)
        save_app_state(self.session.id, state, self.session_dir)

    # ── compaction fallout ────────────────────────────────────────────────────

    def move_memory_stores(self, outgoing: str, incoming: str) -> None:
        """Compaction installs a NEW conversation id, and both memory stores
        are keyed by the old one. Nothing in the agent moves them: the stores
        are the application's, handed to the plugin at construction, so keeping
        them addressable is the application's job. Mutated IN PLACE — the
        plugin's tools hold these same dicts by reference."""
        for key in (TODO_STORE_KEY, SCRATCHPAD_STORE_KEY):
            store = self.session.extras.get(key)
            if isinstance(store, dict) and outgoing in store:
                store[incoming] = store.pop(outgoing)
