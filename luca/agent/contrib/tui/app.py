"""AgentApp — the Textual front end of the agent loop.

One drive worker owns the runner exactly like the classic REPL loop did:
answer the approval gate (modal screens), then fall THROUGH to a run —
recording answers on the strategy does not advance the runner, so the
approval branch must always be followed by a run, never a re-prompt. A lazy
run is created per iteration (`streaming=` decides the event tier), events
render into transcript cells through one unified handler (delta events
stream into the live cell; block events finalize it — so the same handler
serves both modes), and the session persists after every run.

Escape while a run is live requests cancellation (`run.cancel()`); the
wind-down renders live and the turn closes CANCELLED. Abandoning at the
approval modal cancels the runner; the loop's next run is the flush.

The prompt stays enabled while the agent works: submitting mid-turn posts
into the open turn (rendered immediately, answered before the turn closes),
and a rejection — cancelling, subagents active — surfaces as a notice with
the draft preserved. Slash commands stay idle-only. A mid-turn post never
starts a second drive; the live loop picks it up at its next LLM call.

Construction wires the full demo agent via `wiring.build_runner` (shell +
memory plugins, math tools, one shared permission strategy); `provider=` is
the zero-logic passthrough tests use to inject a scripted `FauxProvider`.
On a resumed session the transcript replays the persisted conversation, and
a non-idle session (gated, parked cancel, retry-ready) starts driving on
mount.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import os
from pathlib import Path
from typing import ClassVar, TypeVar

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.css.query import NoMatches
from textual.widgets import Footer, Header

from luca.agent.core import AgentError, AgentRun, AlreadyCancellingError
from luca.agent.core.events import (
    ApprovalRequired,
    CompactionFinished,
    CompactionScheduled,
    CompactionStarted,
    FinishReason,
    ReasoningBlock,
    ReasoningDelta,
    ReasoningStart,
    SubagentFinished,
    SubagentPaused,
    SubagentsSpawned,
    SubagentStarted,
    TextBlock,
    TextDelta,
    TextStart,
    ToolCallReceived,
    ToolCallStart,
    ToolExecuted,
    ToolExecutionStarted,
)
from luca.agent.core.exceptions import ProjectionError
from luca.agent.core.models import (
    NONTERMINAL_STATUSES,
    AgentSession,
    AssistantMessage,
    ChildConversation,
    CompactionEntry,
    ContentPart,
    ExecutionStatus,
    ImageBase64,
    ImageContent,
    TextContent,
    ThinkingContent,
    ToolExecution,
    TurnFinish,
    TurnOutcome,
    UserMessage,
)
from luca.agent.core.projection import tool_message_text

from .approvals import build_approval_prompts
from .cells import (
    AssistantCell,
    CompactionCell,
    NoticeCell,
    ReasoningCell,
    SubagentPanel,
    ToolCallCell,
    TranscriptCell,
    UserCell,
)
from .clipboard import MEDIA_TYPE, ClipboardUnavailable, read_clipboard_image
from .commands import COMMANDS, dispatch
from .context_bar import ContextBar
from .prompt import PromptInput
from .render import (
    REDACTED_REASONING_MARKER,
    child_links,
    is_runtime_plumbing,
    reasoning_transcript_text,
    subagent_task,
    user_transcript_text,
)
from .screens import ApprovalScreen
from .sessions import save_session
from .wiring import build_runner

_CellT = TypeVar("_CellT", bound=TranscriptCell)
DEFAULT_THEME = "nord"


class AgentApp(App):
    TITLE = "luca"

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "cancel_run", "Cancel turn"),
        Binding("ctrl+v", "paste_image", "Attach image", priority=True),
        Binding("ctrl+d", "save_quit", "Quit", priority=True),
    ]

    CSS = """
    #transcript {
        padding: 1 0 0 0;
    }
    #prompt {
        margin: 0 1 1 1;
    }
    """

    def __init__(
        self,
        session: AgentSession,
        *,
        provider=None,
        theme: str = DEFAULT_THEME,
        workspace: str | os.PathLike[str] = ".",
        session_dir: str | os.PathLike[str] = ".",
        streaming: bool = True,
        mode: str = "ask",
        context_manager=None,
        additional_directories: list | None = None,
        permission_rules: list | None = None,
        recommended_models: dict | None = None,
        subagents: bool = True,
        skills: bool = True,
        extra_skill_locations: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.theme = theme
        self._session_dir = Path(session_dir)
        self._streaming = streaming
        self._workspace = workspace
        self._provider = provider
        self._mode = mode
        self._context_manager = context_manager
        self._additional_directories = additional_directories
        self._permission_rules = permission_rules
        self.recommended_models = recommended_models
        self._subagents = subagents
        self._skills = skills
        self._extra_skill_locations = extra_skill_locations
        self.runner, self.strategy = self._build_runner(session)
        self._current_run: AgentRun | None = None
        # True while the drive worker is alive. The worker group is
        # `exclusive=True`, so starting a second drive would CANCEL the live
        # run mid-turn — a mid-turn post must never start one (the live loop's
        # next LLM call picks the message up on its own).
        self._driving = False
        # KEYED BY CONVERSATION. With subagents, the main agent and several
        # children stream at once on ONE event stream; a single live-cell slot
        # would splice two conversations' text into the same cell.
        self._live_reasoning: dict[str, ReasoningCell | None] = {}
        self._live_text: dict[str, AssistantCell | None] = {}
        self._tool_cells: dict[str, ToolCallCell] = {}
        # conversation id → the panel its cells mount into. A conversation with
        # no panel here is the main one, and mounts into the transcript itself.
        self._panels: dict[str, SubagentPanel] = {}
        self._pending_images: list[ImageContent] = []

    @property
    def current_run(self) -> AgentRun | None:
        """The live run handle while the drive worker is inside one."""
        return self._current_run

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(id="transcript")
        yield ContextBar(id="context-bar")
        yield PromptInput(
            placeholder="Message the agent — Enter to send, Alt+Enter for a new line, /help for commands",
            commands=[f"/{command.name}" for command in COMMANDS],
            id="prompt",
        )
        yield Footer()

    async def on_mount(self) -> None:
        self._refresh_status()
        await self._replay_history()
        if self.runner.idle():
            self.query_one("#prompt", PromptInput).focus()
        else:  # gated / parked cancel / retry-ready — resume driving
            self._start_drive()

    # ── input ──────────────────────────────────────────────────────────────────

    async def on_prompt_input_submitted(self, event: PromptInput.Submitted) -> None:
        text = event.value.strip()
        if text.startswith("/"):
            # Commands mutate runner/session state and stay idle-only. The
            # draft is preserved — input is never silently discarded.
            if not self.runner.idle():
                self.notify("Commands are available when the agent is idle.", severity="warning")
                return
            if await dispatch(self, text):
                event.prompt_input.clear()
                return
        parts: list[ContentPart] = [*self._pending_images]
        if text:
            parts.append(TextContent(text=text))
        if not parts:
            return
        try:
            self.runner.post_message(parts)
        except AgentError as exc:
            # The transient rejection (cancelling) and every other refusal
            # alike: a brief notice, and the draft stays in the input for a
            # later resubmit.
            self.notify(str(exc), severity="warning")
            return
        event.prompt_input.clear()
        self._pending_images = []
        # Rendered immediately; mid-turn it may interleave with streaming
        # cells, which is correct — the message is answered after the current
        # response completes.
        await self._mount_cell(UserCell(user_transcript_text(parts)))
        if not self._driving:
            self._start_drive()

    # ── the drive worker ───────────────────────────────────────────────────────

    def _start_drive(self) -> None:
        self._driving = True
        self._set_busy(True)
        self.run_worker(self._drive(), group="drive", exclusive=True)

    async def _drive(self) -> None:
        """THE DRIVE COMES BEFORE THE PROMPT.

        Answering writes to the permission strategy, not to the execution, so
        `approval_status` stays PENDING until a drive re-asks `decide()` — the
        engine's single decide call site. Prompting first would therefore
        re-ask the user for whatever was just answered; driving first means
        "still `BLOCKED` after a drive" is a genuinely unanswered gate.

        Abandoning at the modal cancels instead of answering, which leaves the
        conversation CANCELLING — not idle — so the next pass drives the flush
        and the loop still terminates."""
        runner = self.runner
        try:
            while True:
                if runner.idle():
                    break
                run = runner.run(streaming=self._streaming)
                self._current_run = run
                try:
                    async with run:
                        async for event in run:
                            await self._on_agent_event(event)
                finally:
                    self._current_run = None
                    # A cancelled subagent is resolved by the wind-down with no
                    # result-tool event to close its panel, so reconcile here
                    # too — the run ending is the last chance to.
                    self._sync_panels()
                    save_session(runner.session, self._session_dir)
                    self._refresh_status()
                if runner.blocked():
                    await self._resolve_approvals()
        except Exception as exc:
            await self._notice(f"turn failed: {exc}", error=True)
            save_session(runner.session, self._session_dir)
            self._refresh_status()
        finally:
            self._driving = False
            self._set_busy(False)

    async def _resolve_approvals(self) -> None:
        """Collect verdicts for every pending execution through the modal,
        then record them all — same policy as the REPL: abandoning discards
        everything collected so far and cancels instead of answering; a DENY
        skips the execution's remaining steps (the call is dead anyway).

        `pending_approvals()` is SUBTREE-SCOPED, so this list can hold gates
        from several conversations at once. Each modal names the subagent that
        raised it; the main agent's are unlabelled, which is the ordinary
        case."""
        collected: list[tuple[ToolExecution, list]] = []
        for execution in self.runner.pending_approvals():
            answers = []
            for prompt in build_approval_prompts(
                execution,
                self.strategy,
                main_conversation_id=self.runner.main_conversation_id,
                subagent_labels={cid: panel.description for cid, panel in self._panels.items()},
            ):
                option = await self.push_screen_wait(ApprovalScreen(prompt))
                if option.is_abandon:
                    self.runner.cancel(error="abandoned at the approval prompt")
                    await self._notice("turn abandoned — flushing")
                    return
                answers.append(option.answer)
                if option.is_deny:
                    break
            collected.append((execution, answers))
        for execution, answers in collected:
            self.strategy.apply_answer(execution, answers)

    # ── event rendering ────────────────────────────────────────────────────────

    async def _on_agent_event(self, event) -> None:
        """One handler for both tiers: deltas stream into the live cell,
        the block event finalizes it (or mounts it whole when not
        streaming).

        Every event names its conversation, and BOTH the live-cell state and
        the mount target are keyed by it — with subagents the stream is the
        whole tree's, so two conversations can be mid-block at the same time
        and their cells must not land in the same column."""
        source = event.conversation_id
        match event:
            case ReasoningStart():
                self._live_reasoning[source] = None
            case ReasoningDelta(text=text):
                self._live_reasoning[source] = await self._stream_into(
                    self._live_reasoning.get(source),
                    ReasoningCell,
                    text,
                    source,
                )
            case ReasoningBlock(text=text, redacted=redacted):
                await self._settle_cell(
                    self._live_reasoning.get(source),
                    ReasoningCell,
                    REDACTED_REASONING_MARKER if redacted else text,
                    source,
                )
                self._live_reasoning[source] = None
            case TextStart():
                self._live_text[source] = None
            case TextDelta(text=text):
                self._live_text[source] = await self._stream_into(
                    self._live_text.get(source),
                    AssistantCell,
                    text,
                    source,
                )
            case TextBlock(text=text):
                await self._settle_cell(self._live_text.get(source), AssistantCell, text, source)
                self._live_text[source] = None
            case ToolCallReceived(execution=execution) if is_runtime_plumbing(execution):
                # A spawn renders as its child's panel and a private tool as
                # nothing; neither gets a cell to start, so neither gets one to
                # update later.
                pass
            case ToolCallReceived(tool_call_id=tool_call_id, execution=execution):
                cell = ToolCallCell(execution)
                self._tool_cells[tool_call_id] = cell
                await self._mount_cell(cell, source)
            case ToolExecutionStarted(tool_call_id=tool_call_id, execution=execution):
                cell = self._tool_cells.get(tool_call_id)
                if cell is not None:
                    cell.mark_running(execution)
            case ToolExecuted(execution=execution, result_text=result_text) if is_runtime_plumbing(execution):
                if execution.status is ExecutionStatus.REFUSED:
                    # A refused spawn has no panel to render as — without this
                    # the budget's refusal would be invisible to the user
                    # while the model reads it.
                    await self._mount_cell(NoticeCell(result_text, error=True), source)
                # The result tool completing is how a subagent's panel closes
                # while its siblings are still working.
                self._sync_panels()
            case ToolExecuted(
                tool_call_id=tool_call_id,
                execution=execution,
                result_text=result_text,
                is_error=is_error,
            ):
                cell = self._tool_cells.get(tool_call_id)
                if cell is None:  # e.g. an orphan recovered on resume
                    cell = ToolCallCell(execution)
                    self._tool_cells[tool_call_id] = cell
                    await self._mount_cell(cell, source)
                cell.finish(execution, result_text, is_error)
            case CompactionFinished(entry=entry, outcome=TurnOutcome.COMPLETED):
                if entry.parts:
                    await self._mount_cell(CompactionCell(entry), source)
                else:
                    await self._mount_cell(NoticeCell("nothing to compact"), source)
            case CompactionFinished(outcome=outcome, error=error):
                # A policy-initiated failure degrades silently otherwise: this
                # event is the only place it surfaces.
                await self._mount_cell(
                    NoticeCell(
                        f"compaction {outcome.value}" + (f": {error}" if error else ""),
                        error=outcome is not TurnOutcome.CANCELLED,
                    ),
                    source,
                )
            case SubagentsSpawned(conversation_ids=conversation_ids):
                # `source` is the PARENT, so a panel nests inside its spawner's
                # panel whenever that spawner has one: at depth 1 the parent is
                # the transcript, and a deeper tree nests by the same rule.
                for child_id in conversation_ids:
                    await self._open_panel(child_id, source)
            case SubagentStarted():
                # `source` is the SUBAGENT here — the panel lookup is the same
                # one the tool cells use.
                panel = self._panels.get(source)
                if panel is not None:
                    panel.set_activity("running")
            case SubagentPaused():
                panel = self._panels.get(source)
                if panel is not None:
                    panel.set_activity("waiting")
            case (
                ToolCallStart()
                | FinishReason()
                | ApprovalRequired()
                | CompactionScheduled()
                | CompactionStarted()
                | SubagentFinished()  # terminal rendering is the settle path's
            ):
                pass

    async def _stream_into(
        self,
        cell: _CellT | None,
        cell_class: type[_CellT],
        delta: str,
        conversation_id: str | None = None,
    ) -> _CellT | None:
        """Append a delta, mounting the cell on the first visible one so a
        whitespace-only block never gets a cell."""
        if cell is None:
            if not delta.strip():
                return None
            cell = cell_class()
            await self._mount_cell(cell, conversation_id)
        cell.append_text(delta)
        self._scroll_end()
        return cell

    async def _settle_cell(
        self,
        cell: _CellT | None,
        cell_class: type[_CellT],
        text: str,
        conversation_id: str | None = None,
    ) -> None:
        """Settle a streamed cell against its completed block (or mount it
        whole when not streaming). Blank text never survives: providers emit
        whitespace-only content alongside tool calls."""
        if not text.strip():
            if cell is not None:
                await cell.remove()
            return
        if cell is None:
            cell = cell_class()
            await self._mount_cell(cell, conversation_id)
        cell.set_text(text)

    # ── subagent panels ────────────────────────────────────────────────────────

    async def _open_panel(
        self,
        conversation_id: str,
        parent_id: str | None = None,
    ) -> SubagentPanel | None:
        """Mount the panel a subagent's cells go into. Idempotent: the same
        child is announced once by the event and again by a replay, and only
        the first one mounts."""
        if conversation_id in self._panels:
            return self._panels[conversation_id]
        entry = self._child_link(conversation_id)
        if entry is None:
            return None
        description, prompt = subagent_task(self.runner.session, entry)
        panel = SubagentPanel(conversation_id, description, prompt)
        self._panels[conversation_id] = panel
        await self._mount_cell(panel, parent_id)
        return panel

    def _child_link(self, conversation_id: str) -> ChildConversation | None:
        for _parent_id, entry in child_links(self.runner.session):
            if entry.conversation_id == conversation_id:
                return entry
        return None

    def _sync_panels(self) -> None:
        """Settle every panel whose subagent has resolved.

        THE LINK IS THE SOURCE OF TRUTH, not the result tool's event: a
        cancelled subagent is resolved by the parent's wind-down without that
        tool ever running, so a panel driven by the event alone would say
        "running" forever on the one path where the user most wants to know it
        stopped."""
        for _parent_id, entry in child_links(self.runner.session):
            panel = self._panels.get(entry.conversation_id)
            if panel is None or entry.execution_result is None:
                continue
            panel.settle(is_error=entry.execution_result.is_error)

    # ── history replay (resume) ────────────────────────────────────────────────

    async def _replay_history(self) -> None:
        session = self.runner.session
        await self._replay_path(session.conversations[session.main_conversation_id].nodes)
        self._sync_panels()

    async def _replay_path(
        self,
        nodes: list[str],
        conversation_id: str | None = None,
    ) -> None:
        """Replay one path's entries as cells. Recursive: a `ChildConversation`
        replays the subagent's own path inside the panel it mounts, so a
        resumed session shows the delegated work rather than a gap where it
        happened.

        `conversation_id` is None for the main conversation — the one that
        mounts into the transcript itself — and names the subagent otherwise."""
        session = self.runner.session
        entries = session.entries
        for node_id in nodes:
            entry = entries.get(node_id)
            if isinstance(entry, UserMessage):
                # A subagent's path begins with its seed prompt and never gets
                # another user message. The panel's border already carries it,
                # so replaying it here would say the same thing twice.
                if conversation_id is None:
                    await self._mount_cell(UserCell(user_transcript_text(entry.parts)))
            elif isinstance(entry, AssistantMessage):
                for part in entry.parts:
                    if isinstance(part, ThinkingContent):
                        text = reasoning_transcript_text(part)
                        if text:
                            await self._mount_cell(ReasoningCell(text), conversation_id)
                    elif isinstance(part, TextContent) and part.text.strip():
                        await self._mount_cell(AssistantCell(part.text), conversation_id)
                    # a ToolCall part renders through its ToolExecution entry
            elif isinstance(entry, ToolExecution):
                if is_runtime_plumbing(entry):
                    if entry.status is ExecutionStatus.REFUSED and entry.error is not None:
                        # a refused spawn never got a panel; replay its notice
                        await self._mount_cell(
                            NoticeCell(entry.error.error_message, error=True),
                            conversation_id,
                        )
                    continue  # the spawn and the result tool render as the panel
                cell = ToolCallCell(entry)
                self._tool_cells[entry.tool_call_id] = cell
                await self._mount_cell(cell, conversation_id)
                if entry.status not in NONTERMINAL_STATUSES:
                    try:
                        message = self.runner.conversation_projector.project_tool_execution(
                            entry,
                            entries,
                        )
                    except ProjectionError:
                        continue
                    cell.finish(entry, tool_message_text(message), message.is_error)
            elif isinstance(entry, ChildConversation):
                panel = await self._open_panel(entry.conversation_id, conversation_id)
                child = session.conversations.get(entry.conversation_id)
                if panel is not None and child is not None:
                    await self._replay_path(child.nodes, entry.conversation_id)
            elif isinstance(entry, CompactionEntry):
                # Only a COMMITTED compaction carries content, and only on the
                # path where the history it replaced is gone — the same rule
                # the wire projection follows.
                if entry.parts:
                    await self._mount_cell(CompactionCell(entry), conversation_id)
            elif isinstance(entry, TurnFinish) and entry.outcome is TurnOutcome.CANCELLED:
                await self._mount_cell(NoticeCell("turn cancelled"), conversation_id)

    # ── actions ────────────────────────────────────────────────────────────────

    async def action_paste_image(self) -> None:
        """Attach the clipboard's image to the next message. A terminal never
        transmits image bytes on paste, so the clipboard is read directly —
        blocking work, kept off the UI thread."""
        try:
            data = await asyncio.to_thread(read_clipboard_image)
        except ClipboardUnavailable as exc:
            self.notify(str(exc), severity="error")
            return
        if data is None:
            self.notify("No image in the clipboard.")
            return
        self._pending_images.append(
            ImageContent(
                source=ImageBase64(
                    data=base64.b64encode(data).decode("ascii"),
                    media_type=MEDIA_TYPE,
                ),
                metadata={
                    "name": f"pasted-{len(self._pending_images) + 1}.png",
                    "size_bytes": len(data),
                    "origin": "clipboard",
                },
            ),
        )
        self._refresh_status()
        self.notify("Image attached — Enter to send, Esc to clear.")

    async def action_cancel_run(self) -> None:
        run = self._current_run
        if run is None:
            if self._pending_images:
                self._pending_images = []
                self._refresh_status()
                self.notify("Attachments cleared.")
            return
        try:
            run.cancel(error="cancelled by user")
        except AlreadyCancellingError:
            return
        await self._notice("cancelling — winding down the turn")

    async def action_save_quit(self) -> None:
        await self._quit()

    async def _quit(self) -> None:
        save_session(self.runner.session, self._session_dir)
        self.exit()

    def _build_runner(self, session: AgentSession):
        """One build_runner call site, so `__init__` and the session-swap paths
        carry the same config-derived workspace / mode / rules and the
        context manager."""
        return build_runner(
            session,
            workspace=self._workspace,
            provider=self._provider,
            mode=self._mode,
            context_manager=self._context_manager,
            additional_directories=self._additional_directories,
            extra_rules=self._permission_rules,
            subagents=self._subagents,
            skills=self._skills,
            extra_skill_locations=self._extra_skill_locations,
        )

    async def _reset_session(self, session: AgentSession) -> None:
        """Swap in a fresh session and wipe the transcript. `/new` rebuilds the
        runner so the new conversation drives cleanly; under `--faux` the
        scripted provider is stateful and already spent, which is a demo-only
        edge (real runs pass provider=None and build fresh clients per turn)."""
        self.runner, self.strategy = self._build_runner(session)
        await self.query_one("#transcript", VerticalScroll).remove_children()
        self._live_reasoning.clear()
        self._live_text.clear()
        self._tool_cells.clear()
        self._panels.clear()
        self._pending_images.clear()
        self._refresh_status()
        self.query_one("#prompt", PromptInput).focus()

    # ── plumbing ───────────────────────────────────────────────────────────────

    async def _mount_cell(
        self,
        cell: TranscriptCell | SubagentPanel,
        conversation_id: str | None = None,
    ) -> None:
        """Mount a cell where its conversation lives: inside that subagent's
        panel, or in the transcript for the main agent (and for any
        conversation whose panel is not up yet)."""
        panel = self._panels.get(conversation_id) if conversation_id else None
        target = panel if panel is not None else self.query_one("#transcript", VerticalScroll)
        await target.mount(cell)
        self._scroll_end()

    def _scroll_end(self) -> None:
        self.query_one("#transcript", VerticalScroll).scroll_end(animate=False)

    async def _notice(self, text: str, *, error: bool = False) -> None:
        await self._mount_cell(NoticeCell(text, error=error))

    def _set_busy(self, busy: bool) -> None:
        # The prompt stays ENABLED while the agent works — posting mid-turn is
        # the feature, and a rejection surfaces as a notice with the draft
        # preserved. Busy only refreshes the status line; going idle refocuses
        # the input.
        if not busy:
            self.query_one("#prompt", PromptInput).focus()
        self._refresh_status()

    def _refresh_status(self) -> None:
        session = self.runner.session
        cfg = session.session_config.llm_config
        status = f"session {session.id} · {cfg.provider}:{cfg.model} · {self.runner.status.value}"
        if cfg.reasoning:
            status += f" · reasoning {cfg.reasoning}"
        if self._pending_images:
            count = len(self._pending_images)
            status += f" · {count} image{'s' if count > 1 else ''} attached"
        self.sub_title = status
        threshold = getattr(self._context_manager, "threshold", 0.8)
        # the bar is absent during the early __init__-time refresh, before compose()
        with contextlib.suppress(NoMatches):
            self.query_one("#context-bar", ContextBar).update_from(session, threshold=threshold)
