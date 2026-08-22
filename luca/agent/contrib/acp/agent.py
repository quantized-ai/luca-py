"""`LucaAgent` — luca behind the Agent Client Protocol.

One `acp.Agent` implementation over `AgentApplication`, so a session driven
from Zed, Nori, Pool or acp-ui is composed exactly the way the TUI composes
one. What this class adds is the protocol surface and the drive loop; every
translation lives next door in `stream`, `permissions`, `questions` and
`replay`.

THE DRIVE LOOP IS THE TUI'S, with `session/update` where it mounts widgets:

    while not runner.idle():
        async with runner.run(streaming=True) as run:
            async for event in run:
                send(translate(event))
        save()
        if runner.blocked():
            resolve approvals
            resolve parked questions

Two properties of that loop are load-bearing and neither is obvious.

THE DRIVE COMES BEFORE THE PROMPT. Answering an approval writes to the
permission strategy, and only a drive consumes it, so a conversation still
BLOCKED after a drive has a genuinely unanswered gate rather than an
unprocessed answer.

THE APPROVAL EVENT IS NOT THE SIGNAL. `ApprovalRequired` is emitted before the
run parks but can be superseded; `runner.blocked()` plus
`runner.pending_approvals()` after the run drains is the durable read, and it
is the one the TUI uses too.

WHAT THIS AGENT DOES NOT DO, all optional in the protocol: it ignores the
`mcpServers` a client passes to `session/new` (nothing in luca speaks MCP yet
— issue #25), it never calls `fs/read_text_file` or `fs/write_text_file`, so
edits go to disk and unsaved editor buffers are invisible to it, and it runs
its own subprocesses rather than the client's terminals, so a long command
reports only when it finishes.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

from acp import Agent, RequestError
from acp.schema import (
    AgentCapabilities,
    ClientCapabilities,
    Implementation,
    InitializeResponse,
    LoadSessionResponse,
    NewSessionResponse,
    PromptCapabilities,
    PromptResponse,
    SessionMode,
    SessionModeState,
    SetSessionModeResponse,
)

from luca.agent.contrib.app import (
    DEFAULT_LOG_LEVEL,
    AgentApplication,
    LucaConfig,
    apply_model_options,
    boot,
    build_context_manager,
    build_permission_rules,
    build_session,
    credentials,
    load_session,
    log_path,
    pick,
    setup_logging,
)
from luca.agent.contrib.resource_permissions import PermissionMode
from luca.agent.core import AlreadyCancellingError
from luca.agent.core.models import (
    AudioContent,
    ContentPart,
    FileContent,
    ImageContent,
    MediaBase64,
    TextContent,
    TurnOutcome,
)

from .permissions import Cancelled, PermissionBridge
from .questions import QuestionBridge, is_questions_tool
from .replay import replay
from .stream import Translator

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = 1

# Our permission modes as ACP session modes. `auto` is not offered: it is
# documented as "same promotion as YOLO, reserved for divergence later", and a
# picker with two identical entries is a worse UI than one with two.
MODES = [
    SessionMode(id=PermissionMode.ASK.value, name="Ask", description="Ask before anything a rule does not cover."),
    SessionMode(id=PermissionMode.YOLO.value, name="Yolo", description="Allow anything no rule explicitly denies."),
]

# `stopReason` for a turn that ended. ERRORED and TIMED_OUT are absent on
# purpose: ACP has no stop reason for "the model call failed", so those raise a
# JSON-RPC error instead and the client shows a failed turn rather than a
# successful one with nothing in it.
STOP_REASONS = {
    TurnOutcome.COMPLETED: "end_turn",
    TurnOutcome.CANCELLED: "cancelled",
}

STREAMING = True
"""Drive with token-level events. The translator has to know, because text and
reasoning arrive as BOTH deltas and a completed block when it is on."""

MAX_DRIVES = 64
"""A backstop, not a policy. Every iteration of the loop must resolve something
— an approval, a question — or the runner would not still be busy; a loop that
cannot make progress is a bug, and spinning on a socket forever hides it."""


class _Block:
    """Attribute access over a content block that may have arrived as either a
    schema model or a plain dict.

    The router validates into models, but the SDK's own examples read blocks
    defensively because a dict can reach an agent through the extension methods
    and through a hand-driven connection. One shim beats a `isinstance` at
    every field."""

    __slots__ = ("_raw",)

    def __init__(self, raw) -> None:
        self._raw = raw

    def __getattr__(self, name: str):
        raw = self._raw
        if isinstance(raw, dict):
            value = raw.get(name)
            if value is None and name == "mime_type":
                value = raw.get("mimeType")
            return _Block(value) if isinstance(value, dict) else value
        return getattr(raw, name, None)


def content_parts(blocks: list) -> list[ContentPart]:
    """An ACP prompt as luca content parts.

    `resource` (embedded context, which is how a client inlines an `@`-mention)
    becomes text carrying its URI, because that is what the model needs and it
    survives every provider. `resource_link` becomes a line naming the file:
    the client is telling the agent a path is relevant, not handing over its
    bytes."""
    parts: list[ContentPart] = []
    for raw in blocks:
        block = _Block(raw)
        kind = block.type
        if kind == "text":
            parts.append(TextContent(text=block.text))
        elif kind == "image":
            parts.append(ImageContent(source=MediaBase64(data=block.data, media_type=block.mime_type)))
        elif kind == "audio":
            parts.append(AudioContent(source=MediaBase64(data=block.data, media_type=block.mime_type)))
        elif kind == "resource_link":
            parts.append(TextContent(text=f"@{block.uri}"))
        elif kind == "resource":
            resource = block.resource
            text = getattr(resource, "text", None)
            if text is not None:
                parts.append(TextContent(text=f'<file uri="{resource.uri}">\n{text}\n</file>'))
            else:
                parts.append(
                    FileContent(
                        source=MediaBase64(
                            data=resource.blob,
                            media_type=getattr(resource, "mime_type", None) or "application/octet-stream",
                        ),
                        name=str(resource.uri),
                    )
                )
    return parts


class LucaAgent(Agent):
    """One process, many sessions. Each is an `AgentApplication` keyed by its
    luca session id, which IS the ACP session id — so `session/load` is a
    lookup in the store this project's sessions already live in."""

    def __init__(
        self,
        *,
        config_path: str | None = None,
        workspace: str | None = None,
        provider=None,
        faux: bool = False,
        checkpoints: bool = True,
        subagents: bool = True,
        skills: bool = True,
        instructions: bool = True,
        log_level: str | None = None,
    ) -> None:
        self._config_path = config_path
        self._workspace_flag = workspace
        self._provider = provider
        self._faux = faux
        self._checkpoints = checkpoints
        self._subagents = subagents
        self._skills = skills
        self._instructions = instructions
        self._log_level = log_level
        self._connection: Any = None
        self._client_capabilities = ClientCapabilities()
        self._sessions: dict[str, AgentApplication] = {}
        self._bridges: dict[str, PermissionBridge] = {}
        self._cancelled: set[str] = set()

    def on_connect(self, conn) -> None:
        self._connection = conn

    # ── initialize ────────────────────────────────────────────────────────────

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        self._client_capabilities = client_capabilities or ClientCapabilities()
        return InitializeResponse(
            protocol_version=min(protocol_version, PROTOCOL_VERSION),
            agent_capabilities=AgentCapabilities(
                load_session=True,
                # `ContentPart` is text | image | audio | file, so all three.
                prompt_capabilities=PromptCapabilities(image=True, audio=True, embedded_context=True),
            ),
            # No auth methods: a credential comes from `auth.json` or the
            # provider's own environment variable, both resolved in-process,
            # so there is nothing for a client to log into.
            auth_methods=[],
            agent_info=Implementation(name="luca", version=_version()),
        )

    # ── sessions ──────────────────────────────────────────────────────────────

    async def new_session(
        self,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list | None = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        if mcp_servers:
            logger.warning("ignoring %d MCP server(s): luca does not speak MCP yet", len(mcp_servers))
        application = self._compose(cwd, additional_directories, resume=None)
        self._sessions[application.session.id] = application
        application.save()
        return NewSessionResponse(session_id=application.session.id, modes=self._mode_state(application))

    async def load_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list | None = None,
        additional_directories: list[str] | None = None,
        **kwargs: Any,
    ) -> LoadSessionResponse:
        application = self._compose(cwd, additional_directories, resume=session_id)
        self._sessions[session_id] = application
        # The whole conversation goes back BEFORE this request is answered, so
        # the client rebuilds the thread and carries on as if uninterrupted.
        for update in replay(application.session):
            await self._connection.session_update(session_id=session_id, update=update)
        return LoadSessionResponse(modes=self._mode_state(application))

    async def set_session_mode(self, session_id: str, mode_id: str, **kwargs: Any) -> SetSessionModeResponse | None:
        application = self._application(session_id)
        try:
            mode = PermissionMode(mode_id)
        except ValueError:
            raise RequestError.invalid_params(f"unknown mode {mode_id!r}") from None
        application.mode = mode.value
        application.strategy.mode = mode
        return SetSessionModeResponse()

    def _mode_state(self, application: AgentApplication) -> SessionModeState:
        return SessionModeState(current_mode_id=application.mode, available_modes=MODES)

    def _compose(self, cwd: str, additional: list[str] | None, *, resume: str | None) -> AgentApplication:
        """One configured agent for one ACP session.

        `cwd` is the client's, and ACP says it MUST be used regardless of where
        the agent process was spawned — so it wins over the `--workspace` flag
        and over `luca.json`."""
        environment = boot(workspace=self._workspace_flag or cwd, config_path=self._config_path)
        config: LucaConfig = environment.config
        session = (
            load_session(resume, environment.session_dir)
            if resume
            else build_session(
                config=config,
                session_dir=environment.session_dir,
                faux=self._faux,
                subagents=self._subagents,
            )
        )
        auth = credentials(config, session.session_config.llm_config, faux=self._faux)
        setup_logging(
            log_path(session.id, environment.session_dir, config.logging.file),
            # BOTH sources are optional and `setup_logging` takes a real level,
            # so the default has to be applied here. Without it a launch with
            # no `--log-level` and no `luca.json` dies on `None.upper()` before
            # the first session exists, which a client reports as nothing more
            # useful than "failed to launch".
            pick(self._log_level, config.logging.level, DEFAULT_LOG_LEVEL),
        )
        mode = config.permissions.mode.value if config.permissions.mode is not None else PermissionMode.ASK.value
        return AgentApplication(
            session,
            provider=self._provider,
            auth=auth,
            workspace=cwd,
            session_dir=environment.session_dir,
            mode=mode,
            context_manager=build_context_manager(
                config,
                provider=self._provider,
                api_key=None,
                enabled=None,
                threshold=None,
                keep_turns=None,
            ),
            additional_directories=[*(config.additional_directories or []), *(additional or [])] or None,
            permission_rules=build_permission_rules(config) or None,
            model_options=lambda llm_config: apply_model_options(llm_config, config=config),
            subagents=self._subagents,
            skills=self._skills,
            instructions=self._instructions,
            checkpoints=self._checkpoints,
        )

    def _application(self, session_id: str) -> AgentApplication:
        application = self._sessions.get(session_id)
        if application is None:
            raise RequestError.invalid_params(f"unknown session {session_id!r}")
        return application

    # ── the prompt turn ───────────────────────────────────────────────────────

    async def prompt(self, session_id: str, prompt: list, **kwargs: Any) -> PromptResponse:
        application = self._application(session_id)
        self._cancelled.discard(session_id)
        parts = content_parts(prompt)
        if not parts:
            raise RequestError.invalid_params("the prompt carried no content luca can read")
        if application.checkpoints.available:
            await application.checkpoints.take(application.session, label=_label(parts))
        application.runner.post_message(parts)
        return await self._drive(session_id, application)

    async def _drive(self, session_id: str, application: AgentApplication) -> PromptResponse:
        runner = application.runner
        translator = Translator(runner.main_conversation_id, streaming=STREAMING)
        bridge = self._bridges.setdefault(
            session_id,
            PermissionBridge(self._connection, session_id, application.strategy),
        )
        questions = QuestionBridge(
            self._connection,
            session_id,
            application.questions.tool,
            elicitation=self._client_capabilities.elicitation is not None,
        )
        outcome: TurnOutcome | None = None
        for _ in range(MAX_DRIVES):
            if runner.idle():
                break
            try:
                async with runner.run(streaming=True) as run:
                    async for event in run:
                        for update in translator.translate(event):
                            await self._connection.session_update(session_id=session_id, update=update)
                    result = await run
                outcome = result.outcome or outcome
            except Exception as exc:
                logger.error("the turn failed: %s", exc, exc_info=True)
                application.save()
                raise RequestError.internal_error(str(exc)) from exc
            finally:
                application.save()
            translator.end_message()
            if runner.blocked():
                try:
                    await bridge.resolve(runner.pending_approvals(), runner.main_conversation_id)
                except Cancelled:
                    self._cancel(runner)
                    continue
            if not runner.cancelling() and not await self._resolve_questions(runner, questions):
                break
        else:
            logger.error("gave up after %d drives without reaching idle", MAX_DRIVES)
        application.save()
        return PromptResponse(stop_reason=self._stop_reason(session_id, outcome))

    async def _resolve_questions(self, runner, bridge: QuestionBridge) -> bool:
        """False when a parked call is one this agent cannot answer, which means
        driving again would spin: the deferral would come straight back."""
        parked = runner.pending_deferred_tool_executions()
        if not parked:
            return True
        unresolved = [execution for execution in parked if not is_questions_tool(execution)]
        for execution in parked:
            if is_questions_tool(execution):
                await bridge.resolve(execution)
        if unresolved:
            names = ", ".join(sorted({execution.raw_tool_call.name for execution in unresolved}))
            logger.error("stopping: %s is parked on something this agent cannot answer", names)
            return False
        return True

    def _stop_reason(self, session_id: str, outcome: TurnOutcome | None) -> str:
        if session_id in self._cancelled:
            return "cancelled"
        if outcome is None:
            return "end_turn"
        reason = STOP_REASONS.get(outcome)
        if reason is None:
            raise RequestError.internal_error(f"the turn ended {outcome.value}")
        return reason

    # ── cancellation ──────────────────────────────────────────────────────────

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        """A NOTIFICATION, so there is nothing to answer. The open `prompt`
        call winds the turn down and returns `cancelled` itself."""
        application = self._sessions.get(session_id)
        if application is None:
            return
        self._cancel(application.runner)
        self._cancelled.add(session_id)

    @staticmethod
    def _cancel(runner) -> None:
        with contextlib.suppress(AlreadyCancellingError):
            runner.cancel(error="cancelled by the client")


def _label(parts: list[ContentPart]) -> str:
    for part in parts:
        if isinstance(part, TextContent) and part.text.strip():
            return part.text.strip().splitlines()[0][:80]
    return "prompt"


def _version() -> str:
    from luca import __version__

    return __version__


__all__ = ["LucaAgent", "content_parts"]
