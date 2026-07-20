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

import asyncio
import base64
import os
from pathlib import Path
from typing import TypeVar

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
    ImageBase64,
    ImageContent,
    TextContent,
    ThinkingContent,
    ToolExecution,
    TurnFinish,
    TurnOutcome,
    UserMessage,
    UserPart,
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
from .clipboard import MEDIA_TYPE, ClipboardUnavailable, read_clipboard_image
from .render import user_transcript_text
from .screens import ApprovalScreen
from .sessions import save_session
from .wiring import build_runner

_CellT = TypeVar("_CellT", bound=TranscriptCell)


class AgentApp(App):
    TITLE = "luca"

    BINDINGS = [
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
        self._pending_images: list[ImageContent] = []

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
        if not self.runner.idle():
            return
        text = event.value.strip()
        # Attachments lead the message, so an image-only post is legal and a
        # bare Enter with nothing pending still does nothing.
        parts: list[UserPart] = [*self._pending_images]
        if text:
            parts.append(TextContent(text=text))
        if not parts:
            return
        event.input.value = ""
        self._pending_images = []
        self.runner.post_message(parts)
        await self._mount_cell(UserCell(user_transcript_text(parts)))
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
                self._live_reasoning = None
            case ReasoningDelta(text=text):
                self._live_reasoning = await self._stream_into(
                    self._live_reasoning, ReasoningCell, text,
                )
            case ReasoningBlock(text=text):
                await self._settle_cell(self._live_reasoning, ReasoningCell, text)
                self._live_reasoning = None
            case TextStart():
                self._live_text = None
            case TextDelta(text=text):
                self._live_text = await self._stream_into(
                    self._live_text, AssistantCell, text,
                )
            case TextBlock(text=text):
                await self._settle_cell(self._live_text, AssistantCell, text)
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

    async def _stream_into(
        self, cell: _CellT | None, cell_class: type[_CellT], delta: str,
    ) -> _CellT | None:
        """Append a delta, mounting the cell on the first visible one so a
        whitespace-only block never gets a cell."""
        if cell is None:
            if not delta.strip():
                return None
            cell = cell_class()
            await self._mount_cell(cell)
        cell.append_text(delta)
        self._scroll_end()
        return cell

    async def _settle_cell(
        self, cell: _CellT | None, cell_class: type[_CellT], text: str,
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
            await self._mount_cell(cell)
        cell.set_text(text)

    # ── history replay (resume) ────────────────────────────────────────────────

    async def _replay_history(self) -> None:
        session = self.runner.session
        entries = session.entries
        for node_id in session.active_conversation.nodes:
            entry = entries.get(node_id)
            if isinstance(entry, UserMessage):
                await self._mount_cell(
                    UserCell(user_transcript_text(entry.parts)),
                )
            elif isinstance(entry, AssistantMessage):
                for part in entry.parts:
                    # blank parts are skipped exactly as a live run drops them
                    if isinstance(part, ThinkingContent) and part.thinking.strip():
                        await self._mount_cell(ReasoningCell(part.thinking))
                    elif isinstance(part, TextContent) and part.text.strip():
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
                name=f"pasted-{len(self._pending_images) + 1}.png",
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
        status = f"session {session.id} · {self.runner.status.value}"
        if self._pending_images:
            count = len(self._pending_images)
            status += f" · {count} image{'s' if count > 1 else ''} attached"
        self.sub_title = status
