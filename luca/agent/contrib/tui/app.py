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

Construction wires the full demo agent via `wiring.build_runner` (shell +
memory plugins, math tools, one shared permission strategy); `provider=` is
the zero-logic passthrough tests use to inject a scripted `FauxProvider`.
On a resumed session the transcript replays the persisted conversation, and
a non-idle session (gated, parked cancel, retry-ready) starts driving on
mount.
"""

from __future__ import annotations

import os
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Footer, Header, Input

from luca.agent.core import AgentRun, AlreadyCancellingError
from luca.agent.core.events import (
    ApprovalRequired,
    FinishReason,
    ReasoningBlock,
    ReasoningDelta,
    ReasoningStart,
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
    AgentSession,
    AssistantMessage,
    ExecutionStatus,
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
    NoticeCell,
    ReasoningCell,
    ToolCallCell,
    TranscriptCell,
    UserCell,
)
from .screens import ApprovalScreen
from .sessions import save_session
from .wiring import build_runner


class AgentApp(App):
    TITLE = "luca"

    BINDINGS = [
        Binding("escape", "cancel_run", "Cancel turn"),
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
        workspace: str | os.PathLike[str] = ".",
        session_dir: str | os.PathLike[str] = ".",
        streaming: bool = True,
        mode: str = "ask",
    ) -> None:
        super().__init__()
        self._session_dir = Path(session_dir)
        self._streaming = streaming
        self.runner, self.strategy = build_runner(
            session, workspace=workspace, provider=provider, mode=mode,
        )
        self._current_run: AgentRun | None = None
        self._live_reasoning: ReasoningCell | None = None
        self._live_text: AssistantCell | None = None
        self._tool_cells: dict[str, ToolCallCell] = {}

    @property
    def current_run(self) -> AgentRun | None:
        """The live run handle while the drive worker is inside one."""
        return self._current_run

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(id="transcript")
        yield Input(placeholder="Message the agent — Enter to send", id="prompt")
        yield Footer()

    async def on_mount(self) -> None:
        self._refresh_status()
        await self._replay_history()
        if self.runner.idle():
            self.query_one("#prompt", Input).focus()
        else:  # gated / parked cancel / retry-ready — resume driving
            self._start_drive()

    # ── input ──────────────────────────────────────────────────────────────────

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text or not self.runner.idle():
            return
        event.input.value = ""
        self.runner.post_message(text)
        await self._mount_cell(UserCell(text))
        self._start_drive()

    # ── the drive worker ───────────────────────────────────────────────────────

    def _start_drive(self) -> None:
        self._set_busy(True)
        self.run_worker(self._drive(), group="drive", exclusive=True)

    async def _drive(self) -> None:
        runner = self.runner
        try:
            while True:
                if runner.awaiting_approval():
                    await self._resolve_approvals()
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
                    save_session(runner.session, self._session_dir)
                    self._refresh_status()
                if runner.idle():
                    break
        except Exception as exc:
            await self._notice(f"turn failed: {exc}", error=True)
            save_session(runner.session, self._session_dir)
            self._refresh_status()
        finally:
            self._set_busy(False)

    async def _resolve_approvals(self) -> None:
        """Collect verdicts for every pending execution through the modal,
        then record them all — same policy as the REPL: abandoning discards
        everything collected so far and cancels instead of answering; a DENY
        skips the execution's remaining steps (the call is dead anyway)."""
        collected: list[tuple[ToolExecution, list]] = []
        for execution in self.runner.pending_approvals():
            answers = []
            for prompt in build_approval_prompts(execution, self.strategy):
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
        streaming)."""
        match event:
            case ReasoningStart():
                self._live_reasoning = ReasoningCell()
                await self._mount_cell(self._live_reasoning)
            case ReasoningDelta(text=text):
                if self._live_reasoning is None:
                    self._live_reasoning = ReasoningCell()
                    await self._mount_cell(self._live_reasoning)
                self._live_reasoning.append_text(text)
                self._scroll_end()
            case ReasoningBlock(text=text):
                cell = self._live_reasoning
                if cell is None:
                    cell = ReasoningCell()
                    await self._mount_cell(cell)
                cell.set_text(text)
                self._live_reasoning = None
            case TextStart():
                self._live_text = AssistantCell()
                await self._mount_cell(self._live_text)
            case TextDelta(text=text):
                if self._live_text is None:
                    self._live_text = AssistantCell()
                    await self._mount_cell(self._live_text)
                self._live_text.append_text(text)
                self._scroll_end()
            case TextBlock(text=text):
                cell = self._live_text
                if cell is None:
                    cell = AssistantCell()
                    await self._mount_cell(cell)
                cell.set_text(text)
                self._live_text = None
            case ToolCallReceived(tool_call_id=tool_call_id, execution=execution):
                cell = ToolCallCell(execution)
                self._tool_cells[tool_call_id] = cell
                await self._mount_cell(cell)
            case ToolExecutionStarted(tool_call_id=tool_call_id, execution=execution):
                cell = self._tool_cells.get(tool_call_id)
                if cell is not None:
                    cell.mark_running(execution)
            case ToolExecuted(
                tool_call_id=tool_call_id, execution=execution,
                result_text=result_text, is_error=is_error,
            ):
                cell = self._tool_cells.get(tool_call_id)
                if cell is None:  # e.g. an orphan recovered on resume
                    cell = ToolCallCell(execution)
                    self._tool_cells[tool_call_id] = cell
                    await self._mount_cell(cell)
                cell.finish(execution, result_text, is_error)
            case ToolCallStart() | FinishReason() | ApprovalRequired():
                pass

    # ── history replay (resume) ────────────────────────────────────────────────

    async def _replay_history(self) -> None:
        session = self.runner.session
        entries = session.entries
        for node_id in session.active_conversation.nodes:
            entry = entries.get(node_id)
            if isinstance(entry, UserMessage):
                text = "".join(
                    part.text for part in entry.parts
                    if isinstance(part, TextContent)
                )
                await self._mount_cell(UserCell(text))
            elif isinstance(entry, AssistantMessage):
                for part in entry.parts:
                    if isinstance(part, ThinkingContent):
                        await self._mount_cell(ReasoningCell(part.thinking))
                    elif isinstance(part, TextContent):
                        await self._mount_cell(AssistantCell(part.text))
                    # a ToolCall part renders through its ToolExecution entry
            elif isinstance(entry, ToolExecution):
                cell = ToolCallCell(entry)
                self._tool_cells[entry.tool_call_id] = cell
                await self._mount_cell(cell)
                if entry.status not in (
                    ExecutionStatus.PENDING, ExecutionStatus.RUNNING,
                ):
                    try:
                        message = self.runner.conversation_projector.project_tool_execution(
                            entry, entries,
                        )
                    except ProjectionError:
                        continue
                    cell.finish(entry, tool_message_text(message), message.is_error)
            elif isinstance(entry, TurnFinish):
                if entry.outcome is TurnOutcome.CANCELLED:
                    await self._mount_cell(NoticeCell("turn cancelled"))

    # ── actions ────────────────────────────────────────────────────────────────

    async def action_cancel_run(self) -> None:
        run = self._current_run
        if run is None:
            return
        try:
            run.cancel(error="cancelled by user")
        except AlreadyCancellingError:
            return
        await self._notice("cancelling — winding down the turn")

    async def action_save_quit(self) -> None:
        save_session(self.runner.session, self._session_dir)
        self.exit()

    # ── plumbing ───────────────────────────────────────────────────────────────

    async def _mount_cell(self, cell: TranscriptCell) -> None:
        transcript = self.query_one("#transcript", VerticalScroll)
        await transcript.mount(cell)
        self._scroll_end()

    def _scroll_end(self) -> None:
        self.query_one("#transcript", VerticalScroll).scroll_end(animate=False)

    async def _notice(self, text: str, *, error: bool = False) -> None:
        await self._mount_cell(NoticeCell(text, error=error))

    def _set_busy(self, busy: bool) -> None:
        prompt = self.query_one("#prompt", Input)
        prompt.disabled = busy
        if not busy:
            prompt.focus()
        self._refresh_status()

    def _refresh_status(self) -> None:
        session = self.runner.session
        self.sub_title = f"session {session.id} · {self.runner.status.value}"
