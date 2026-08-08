"""`AgentApp` — the live agent wired onto the Luca frame.

One drive worker owns the runner exactly like the classic REPL loop did:
answer the approval gate (the inline prompt that replaces the composer), then
fall THROUGH to a run — recording answers on the strategy does not advance
the runner, so the approval branch must always be followed by a run, never a
re-prompt. A lazy run is created per iteration (`streaming=` decides the
event tier), events render into transcript blocks through one unified
handler, and the session persists after every run.

Escape while a run is live requests cancellation; the wind-down renders live
and the turn closes CANCELLED. Choosing "Cancel turn" at the approval prompt
does the same. A drive failure renders as an error block plus the recovery
prompt (keep retrying / switch model / cancel turn).

The composer stays enabled while the agent works, with one exception:
submitting mid-turn posts into the open turn, but a submit while the main
conversation is BLOCKED is refused with a notice — the framework would carry
the message past the gate (0008), and in this UI the user's answer belongs to
the approval prompt instead. Slash commands stay idle-only. Typing `/` in an
empty composer opens the palette; `@` opens the context picker, which writes
the paths it commits back into the composer as text.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import time
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

from textual.binding import Binding
from textual.widgets import TextArea

from luca.agent.contrib.memory import changed_of, is_todo_tool, is_todo_update
from luca.agent.contrib.simple_context_manager import get_context_window_size
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
    AgentSession,
    ChildConversation,
    ContentPart,
    ExecutionStatus,
    ImageBase64,
    ImageContent,
    LLMConfig,
    ToolExecution,
    TurnOutcome,
)
from luca.agent.core.projection import tool_message_text

from . import state as vm
from .approvals import build_approval_prompts
from .blocks import (
    AssistantText,
    TaskBlockView,
    ThinkingLine,
    ToolBlockView,
)
from .clipboard import MEDIA_TYPE, ClipboardUnavailable, read_clipboard_image
from .format import HINTS, home_path, inline_paths, short_model
from .frame import DEFAULT_THEME, LucaApp
from .gitinfo import GitInfo, read_git_info
from .modals import CostScreen, SessionsScreen, SettingsScreen
from .prompt import PromptInput
from .prompt_files import ReadLimits, parse_prompt
from .render import (
    SCRATCHPAD_STORE_KEY,
    TODO_STORE_KEY,
    child_links,
    entry_blocks,
    filter_rows,
    is_runtime_plumbing,
    mention_blocks,
    picker_rows,
    plan_block,
    plan_dismissed,
    session_todos,
    subagent_task,
    tool_block,
    user_transcript_text,
    was_auto_approved,
)
from .sessions import save_session
from .shells import ApprovalPromptView, OverlayListView
from .usage import status_counter
from .wiring import build_runner

__all__ = ["AgentApp", "DEFAULT_THEME"]


class AgentApp(LucaApp):
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "cancel_run", show=False),
        Binding("ctrl+p", "palette", show=False, priority=True),
        Binding("ctrl+s", "show_skills", show=False, priority=True),
        Binding("ctrl+o", "expand_output", show=False, priority=True),
        Binding("ctrl+v", "paste_image", show=False, priority=True),
    ]

    def __init__(
        self,
        session: AgentSession,
        *,
        provider=None,
        theme: str = DEFAULT_THEME,
        workspace: str | Path = ".",
        session_dir: str | Path = ".",
        streaming: bool = True,
        mode: str = "ask",
        context_manager=None,
        additional_directories: list | None = None,
        permission_rules: list | None = None,
        recommended_models: dict | None = None,
        model_options: Callable[[LLMConfig], LLMConfig] | None = None,
        subagents: bool = True,
        skills: bool = True,
        extra_skill_locations: list[str] | None = None,
        instructions: bool = True,
        extra_instructions: list[str] | None = None,
        resume: bool = False,
        read_limits: ReadLimits | None = None,
    ) -> None:
        super().__init__(theme=theme)
        self._read_limits = read_limits or ReadLimits()
        self._session_dir = Path(session_dir)
        self._streaming = streaming
        self._workspace = workspace
        self._provider = provider
        self._mode = mode
        self._context_manager = context_manager
        self._additional_directories = additional_directories
        self._permission_rules = permission_rules
        self.recommended_models = recommended_models
        # A callable, not the LucaConfig: the app needs "re-resolve options for
        # this pair", not the file the answer came from. Identity when nothing
        # configured any, so a plain app carries no options at all.
        self._model_options = model_options or (lambda llm_config: llm_config)
        self._subagents = subagents
        self._skills = skills
        self._extra_skill_locations = extra_skill_locations
        self._instructions = instructions
        self._extra_instructions = extra_instructions
        self._resume = resume
        self.runner, self.strategy = self._build_runner(session)
        self._current_run: AgentRun | None = None
        self._driving = False
        # KEYED BY CONVERSATION: with subagents several conversations stream
        # onto one event stream at once.
        self._live_thinking: dict[str, ThinkingLine | None] = {}
        self._thinking_started: dict[str, float] = {}
        self._live_text: dict[str, AssistantText | None] = {}
        self._tool_views: dict[str, ToolBlockView] = {}
        self._tasks: dict[str, TaskBlockView] = {}
        # The ids the running turn's last todo write moved. Presentation only —
        # the list itself is read straight off the memory plugin's store, so
        # there is no second copy of it to keep in step.
        self._plan_changed: list[int] = []
        self._pending_images: list[ImageContent] = []
        # Which calls the user answered at a prompt — everything else that ran
        # was decided by rule, and the transcript says so.
        self._answered: set[str] = set()
        self._denied_by_user: set[str] = set()
        self._approval_future: asyncio.Future[int] | None = None
        self._retry_prompt_active = False
        self._git = GitInfo()
        self._show_counter = True
        # What the composer held when an overlay replaced it. Opening one wipes
        # the dock, so the half-typed message has to be carried across and put
        # back — with the `@` picker's paths inlined, or unchanged on esc.
        self._composer_prefix = ""
        self._menu_handler = None
        self._menu_rows: list[vm.OverlayRow] = []
        self._menu_all_rows: list[vm.OverlayRow] = []
        self._picker_selected: set[str] = set()
        self._picker_files: list[str] = []

    @property
    def current_run(self) -> AgentRun | None:
        return self._current_run

    def _composer_commands(self) -> list[str]:
        from .commands import COMMANDS

        return [f"/{command.name}" for command in COMMANDS]

    # ── mount ─────────────────────────────────────────────────────────────────

    async def on_mount(self) -> None:
        super().on_mount()
        await self.show_composer(vm.ComposerState())
        self.set_hints(HINTS["idle"])
        self._refresh_status()
        self.run_worker(self._load_git_info(), group="git", exclusive=True)
        await self._replay_history()
        if self._resume:
            from .commands import dispatch

            await dispatch(self, "/resume")
            return
        self._settle()

    async def _load_git_info(self) -> None:
        self._git = await asyncio.to_thread(read_git_info, self._workspace)
        self._refresh_status()

    # ── input ─────────────────────────────────────────────────────────────────

    async def on_prompt_input_submitted(self, event: PromptInput.Submitted) -> None:
        text = event.value.strip()
        if text.startswith("/"):
            from .commands import dispatch

            if not self.runner.idle():
                await self._notice("commands are available when the agent is idle", error=True)
                return
            if await dispatch(self, text):
                event.prompt_input.clear()
                return
        parts: list[ContentPart] = [*self._pending_images, *self._expand_mentions(text)]
        if not parts:
            return
        # THE TUI OPTS OUT OF POSTING PAST A GATE (0008). The framework will
        # carry this message to the model with the gated call projected as a
        # placeholder; in this UI that is never what the user meant — the
        # answer they want is the approval prompt two lines below. The composer
        # is normally not even mounted here (the prompt replaces it), but the
        # swap is driven by the drive worker, so there are windows where it is:
        # before the parked run has returned, and after `_restore_composer()`
        # when a partially-answered gate re-arms.
        #
        # `blocked()` and not `pending_approvals()`: a SUBAGENT gate with
        # siblings still working leaves this conversation BUSY, and steering
        # posts into a live orchestration stay supported. Checked BEFORE the
        # post, so 0008's own BLOCKED → BUSY transition cannot bypass it.
        if self.runner.blocked():
            await self._notice("answer the approval prompt first", error=True)
            return
        try:
            self.runner.post_message(parts)
        except AgentError as exc:
            await self._notice(str(exc), error=True)
            return
        # A new turn is not a new plan: the list stays exactly as the agent
        # left it. What is dropped is the HIGHLIGHT — `changed` belongs to the
        # write that produced it, and carrying it into the next turn would
        # lead the panel with items nothing has touched since.
        self._plan_changed = []
        self._render_plan()
        event.prompt_input.clear()
        self._pending_images = []
        await self.mount_block(vm.UserBlock(text=user_transcript_text(parts)))
        for block in mention_blocks(parts):
            await self.mount_block(block)
        if not self._driving:
            self._start_drive()

    def _expand_mentions(self, text: str) -> list[ContentPart]:
        """The typed text plus one part per resolvable `@path`. A file that
        cannot be read still yields a part — one that tells the agent to reach
        for its own tools — so a mention is never silently dropped."""
        if not text:
            return []
        return parse_prompt(
            text,
            workspace=self._workspace,
            limits=self._read_limits,
            context_window=get_context_window_size(self.runner.session),
        )

    async def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """`/` in an empty composer opens the palette; `@` opens the context
        picker. Both consume the trigger character."""
        area = event.text_area
        if not isinstance(area, PromptInput):
            return
        text = area.text
        if text == "/":
            area.clear()
            await self.open_palette()
        elif text.endswith("@") and (len(text) == 1 or text[-2].isspace()):
            area.load_text(text[:-1])
            area.move_cursor(area.document.end)
            await self.open_context_picker()
        else:
            area.update_suggestion()

    # ── the drive worker ──────────────────────────────────────────────────────

    def _start_drive(self) -> None:
        self._driving = True
        self._set_busy(True)
        self.run_worker(self._drive(), group="drive", exclusive=True)

    async def _drive(self) -> None:
        """THE DRIVE COMES BEFORE THE PROMPT: answering writes to the
        permission strategy, so only a drive can consume it — still-BLOCKED
        after a drive is a genuinely unanswered gate."""
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
                except Exception as exc:
                    self._current_run = None
                    save_session(runner.session, self._session_dir)
                    self._refresh_status()
                    if not await self._recover_from(exc):
                        break
                    continue
                finally:
                    self._current_run = None
                    self._sync_tasks()
                    save_session(runner.session, self._session_dir)
                    self._refresh_status()
                if runner.blocked():
                    await self._resolve_approvals()
        finally:
            self._driving = False
            self._set_busy(False)
            self.run_worker(self._load_git_info(), group="git", exclusive=True)

    async def _recover_from(self, exc: Exception) -> bool:
        """The 1h recovery prompt. True = keep driving, False = stop."""
        await self.mount_block(
            vm.ToolBlock(
                tool="model",
                arg=short_model(self.runner.session.session_config.llm_config.model),
                status="error",
                output=vm.ToolOutput(lines=[f"[error]{type(exc).__name__}[/]", str(exc)]),
            )
        )
        alternate = self._alternate_model()
        options = [vm.ApprovalOption(label="Retry now")]
        if alternate is not None:
            options.append(vm.ApprovalOption(label=f"Switch to {short_model(alternate[1])} and continue"))
        options.append(vm.ApprovalOption(label="Cancel turn", key_hint="esc"))
        self._retry_prompt_active = True
        try:
            choice = await self._ask(
                vm.ApprovalState(question="Keep going, or stop here?", options=options),
                hints=HINTS["retry"],
            )
        finally:
            self._retry_prompt_active = False
        await self.show_composer(vm.ComposerState(placeholder="working…"))
        if alternate is not None and choice == 1:
            from .commands import _apply

            provider, model = alternate
            _apply(self, provider=provider, model=model)
            return True
        if choice == len(options) - 1:
            with contextlib.suppress(AlreadyCancellingError):
                self.runner.cancel(error="cancelled after a failure")
            return not self.runner.idle()
        return True

    def resolve_model_options(self, llm_config: LLMConfig) -> LLMConfig:
        """Re-resolve the configured `options` for this pair. Called by `_apply`
        on every model switch, so a session never keeps the previous model's
        settings."""
        return self._model_options(llm_config)

    def _alternate_model(self) -> tuple[str, str] | None:
        """A sibling to offer after a turn fails — the newest model from the
        same provider that is not the one that just failed."""
        from .commands import recent_models

        config = self.runner.session.session_config.llm_config
        for model in recent_models(config.provider, configured=self.recommended_models):
            if model != config.model:
                return config.provider, model
        return None

    # ── approvals (inline, replacing the composer) ────────────────────────────

    async def _ask(self, state: vm.ApprovalState, *, hints: list[str] | None = None) -> int:
        prompt = await self.show_approval(state)
        self.set_hints(hints or ["↑↓ move", "enter confirm", f"1–{len(state.options)} pick"])
        self._approval_future = asyncio.get_running_loop().create_future()
        try:
            return await self._approval_future
        finally:
            self._approval_future = None
            if prompt.is_attached:
                await prompt.remove()

    def on_approval_prompt_view_decided(self, message: ApprovalPromptView.Decided) -> None:
        if self._approval_future is not None and not self._approval_future.done():
            self._approval_future.set_result(message.index)

    async def _resolve_approvals(self) -> None:
        """Collect verdicts for every pending execution through the inline
        prompt, then record them all. Cancel-turn discards everything
        collected so far; a deny skips the execution's remaining steps and
        hands the composer back empty — tell Luca what to do instead."""
        collected: list[tuple[ToolExecution, list]] = []
        for execution in self.runner.pending_approvals():
            if execution.tool_call_id in self._answered:
                # Coverage is emergent: an answer that does not cover every
                # required pair leaves the call pending and the gate re-arms.
                # Say so — a silently repeating prompt reads as a frozen UI.
                await self._notice("that answer did not cover the request — asking again")
            answers = []
            for prompt in build_approval_prompts(
                execution,
                self.strategy,
                main_conversation_id=self.runner.main_conversation_id,
                subagent_labels={cid: task.status_model.description for cid, task in self._tasks.items()},
            ):
                choice = await self._ask(prompt.to_state())
                option = prompt.options[choice]
                self._answered.add(execution.tool_call_id)
                if option.is_cancel:
                    with contextlib.suppress(AlreadyCancellingError):
                        self.runner.cancel(error="cancelled at the approval prompt")
                    await self._notice("turn cancelled — winding down")
                    await self._restore_composer()
                    return
                answers.append(option.answer)
                if option.is_deny:
                    self._denied_by_user.add(execution.tool_call_id)
                    break
            collected.append((execution, answers))
        for execution, answers in collected:
            self.strategy.apply_answer(execution, answers)
        await self._restore_composer()

    async def _restore_composer(self, text: str = "") -> None:
        placeholder = "working…" if self._driving else "ask, or / for commands"
        composer = await self.show_composer(vm.ComposerState(placeholder=placeholder, text=text))
        if text:
            composer.input.move_cursor(composer.input.document.end)
        self._set_busy(self._driving)

    # ── event rendering ───────────────────────────────────────────────────────

    async def _on_agent_event(self, event) -> None:
        source = event.conversation_id
        match event:
            case ReasoningStart():
                self._thinking_started[source] = time.monotonic()
                line = ThinkingLine(vm.ThinkingBlock(activity="thinking"))
                line.add_class("block")
                await self._mount_widget(line, source)
                self._live_thinking[source] = line
            case ReasoningDelta():
                pass  # the design collapses thinking to its one-line label
            case ReasoningBlock():
                started = self._thinking_started.pop(source, None)
                duration = max(time.monotonic() - started, 1.0) if started else None
                line = self._live_thinking.pop(source, None)
                if line is None:
                    line = ThinkingLine(vm.ThinkingBlock(duration=duration))
                    await self._mount_widget(line, source)
                else:
                    line.settle(duration) if duration is not None else line.set_activity("thought")
            case TextStart():
                self._live_text[source] = None
            case TextDelta(text=text):
                view = self._live_text.get(source)
                if view is None:
                    if not text.strip():
                        return
                    view = AssistantText(vm.TextBlock(text="", streaming=True))
                    await self._mount_widget(view, source)
                    self._live_text[source] = view
                view.append_text(text)
                self.scroll_transcript_end()
            case TextBlock(text=text):
                view = self._live_text.get(source)
                if not text.strip():
                    if view is not None:
                        await view.remove()
                elif view is None:
                    await self.mount_block(vm.TextBlock(text=text), self._tasks.get(source))
                else:
                    view.settle(text)
                self._live_text[source] = None
            case ToolCallReceived(execution=execution) if is_runtime_plumbing(execution):
                pass  # a spawn renders as its task block; private tools render nothing
            case ToolCallReceived(execution=execution) if is_todo_tool(execution):
                pass  # a todo call is the sticky panel, not a transcript row
            case ToolCallReceived(tool_call_id=tool_call_id, execution=execution):
                view = ToolBlockView(tool_block(execution))
                self._tool_views[tool_call_id] = view
                await self._mount_widget(view, source)
            case ToolExecutionStarted():
                pass  # no running affordance in the design — no spinners
            case ToolExecuted(execution=execution, result_text=result_text) if is_runtime_plumbing(execution):
                if execution.status is ExecutionStatus.REFUSED:
                    await self._mount_widget_block(vm.NoticeBlock(text=result_text, error=True), source)
                self._sync_tasks()
            case ToolExecuted(execution=execution, result_text=result_text) if is_todo_tool(execution):
                if execution.status is ExecutionStatus.REFUSED:
                    await self._mount_widget_block(vm.NoticeBlock(text=result_text, error=True), source)
                # Only the main agent's list is sticky: one panel cannot speak
                # for a parent and three subagents at once.
                elif source == self.runner.main_conversation_id and is_todo_update(execution):
                    self._plan_changed = changed_of(execution)
                    self._render_plan(running=True)
            case ToolExecuted(
                tool_call_id=tool_call_id,
                execution=execution,
                result_text=result_text,
                is_error=is_error,
            ):
                view = self._tool_views.get(tool_call_id)
                block = tool_block(
                    execution,
                    result_text,
                    is_error=is_error,
                    auto_approved=self._was_auto_approved(execution),
                    denied_by_user=tool_call_id in self._denied_by_user,
                )
                if view is None:
                    view = ToolBlockView(block)
                    self._tool_views[tool_call_id] = view
                    await self._mount_widget(view, source)
                else:
                    await view.apply(block)
                self.scroll_transcript_end()
            case CompactionFinished(entry=entry, outcome=TurnOutcome.COMPLETED, new_conversation_id=new_id):
                if new_id is not None:
                    self._move_memory_stores(source, new_id)
                replaced = len(entry.compacted_nodes or [])
                text = f"context compacted · {replaced} entries summarized" if entry.parts else "nothing to compact"
                await self._mount_widget_block(vm.NoticeBlock(text=text), source)
                self._refresh_status()
            case CompactionFinished(outcome=outcome, error=error):
                await self._mount_widget_block(
                    vm.NoticeBlock(
                        text=f"compaction {outcome.value}" + (f": {error}" if error else ""),
                        error=outcome is not TurnOutcome.CANCELLED,
                    ),
                    source,
                )
            case SubagentsSpawned(conversation_ids=conversation_ids):
                for child_id in conversation_ids:
                    await self._open_task(child_id, source)
            case SubagentStarted():
                task = self._tasks.get(source)
                if task is not None:
                    task.set_status("running")
            case SubagentPaused():
                task = self._tasks.get(source)
                if task is not None:
                    task.set_status("waiting")
            case (
                ToolCallStart()
                | FinishReason()
                | ApprovalRequired()
                | CompactionScheduled()
                | CompactionStarted()
                | SubagentFinished()
            ):
                pass

    def _was_auto_approved(self, execution: ToolExecution) -> bool:
        """State every automatic permission decision: the call went through a
        gate and the user never answered. Narrows the stored-session rule with
        the answers collected in this process."""
        if execution.tool_call_id in self._answered:
            return False
        return was_auto_approved(execution)

    # ── the sticky plan panel ─────────────────────────────────────────────────

    def _todos(self) -> list[dict]:
        """The main conversation's todo list. THE STORE IS THE TRUTH: `wiring`
        handed the plugin a dict that lives on the session, so reading it back
        off the session reads exactly what the agent answers from."""
        return session_todos(self.runner.session)

    def _move_memory_stores(self, outgoing: str, incoming: str) -> None:
        """Compaction installs a NEW conversation id, and both memory stores
        are keyed by the old one. Nothing in the agent moves them: the stores
        are the app's, handed to the plugin at construction, so keeping them
        addressable is the app's job. Mutated IN PLACE — the plugin's tools
        hold these same dicts by reference."""
        for key in (TODO_STORE_KEY, SCRATCHPAD_STORE_KEY):
            store = self.runner.session.extras.get(key)
            if isinstance(store, dict) and outgoing in store:
                store[incoming] = store.pop(outgoing)

    def _render_plan(self, *, running: bool = False) -> None:
        if plan_dismissed(self.runner.session):
            self.set_plan(None)
            return
        self.set_plan(plan_block(self._todos(), changed=self._plan_changed, running=running))

    async def _mount_widget(self, widget, conversation_id: str | None) -> None:
        task = self._tasks.get(conversation_id) if conversation_id else None
        target = task.body if task is not None else self.transcript
        await target.mount(widget)
        self.scroll_transcript_end()

    async def _mount_widget_block(self, block: vm.Block, conversation_id: str | None):
        return await self.mount_block(block, self._tasks.get(conversation_id) if conversation_id else None)

    # ── subagent tasks ────────────────────────────────────────────────────────

    async def _open_task(self, conversation_id: str, parent_id: str | None = None) -> TaskBlockView | None:
        if conversation_id in self._tasks:
            return self._tasks[conversation_id]
        entry = self._child_link(conversation_id)
        if entry is None:
            return None
        description, _prompt = subagent_task(self.runner.session, entry)
        task = TaskBlockView(vm.TaskBlock(description=description, status="waiting"))
        self._tasks[conversation_id] = task
        await self._mount_widget(task, parent_id)
        return task

    def _child_link(self, conversation_id: str) -> ChildConversation | None:
        for _parent_id, entry in child_links(self.runner.session):
            if entry.conversation_id == conversation_id:
                return entry
        return None

    def _sync_tasks(self) -> None:
        """THE LINK IS THE SOURCE OF TRUTH: a cancelled subagent resolves
        without its result tool ever running, and its task block still has to
        stop saying running."""
        for _parent_id, entry in child_links(self.runner.session):
            task = self._tasks.get(entry.conversation_id)
            if task is None or entry.execution_result is None:
                continue
            task.set_status("failed" if entry.execution_result.is_error else "done")

    # ── history replay (resume) ───────────────────────────────────────────────

    async def _replay_history(self) -> None:
        session = self.runner.session
        await self._replay_path(session.conversations[session.main_conversation_id].nodes)
        self._sync_tasks()
        # The plugin rebuilt its list from this same session when the runner
        # was composed, so the panel just reads it back out.
        self._render_plan()

    async def _replay_path(self, nodes: list[str], conversation_id: str | None = None) -> None:
        """Mount what `render.entry_blocks` derives, and keep the widgets a
        later event still has to mutate reachable — a call replayed while
        pending settles through `_tool_views` when its result finally lands."""
        session = self.runner.session
        for node_id in nodes:
            entry = session.entries.get(node_id)
            if isinstance(entry, ChildConversation):
                task = await self._open_task(entry.conversation_id, conversation_id)
                child = session.conversations.get(entry.conversation_id)
                if task is not None and child is not None:
                    await self._replay_path(child.nodes, entry.conversation_id)
                continue
            blocks = entry_blocks(
                entry,
                resolve_result=self._resolve_result,
                subagent=conversation_id is not None,
            )
            for block in blocks:
                widget = await self._mount_widget_block(block, conversation_id)
                if isinstance(entry, ToolExecution) and isinstance(block, vm.ToolBlock):
                    self._tool_views[entry.tool_call_id] = widget

    def _resolve_result(self, execution: ToolExecution) -> tuple[str | None, bool]:
        """A terminal execution's model-facing result text, through the
        runner's OWN projector — the transcript then shows what the model was
        actually told, not a second rendering of the raw result."""
        try:
            message = self.runner.conversation_projector.project_tool_execution(
                execution,
                self.runner.session.entries,
            )
        except ProjectionError:
            return None, False
        return tool_message_text(message), message.is_error

    # ── overlays: palette + context picker + menus ────────────────────────────

    async def open_palette(self, query: str = "") -> None:
        from .commands import palette_rows

        self._composer_prefix = ""  # the palette only opens on a lone `/`
        self._menu_all_rows = palette_rows()
        self._menu_handler = self._run_palette_choice
        await self._refresh_overlay("palette", "/", query)
        self.set_hints(HINTS["palette"])

    async def open_context_picker(self, query: str = "") -> None:
        """`@` picker. Each open starts unchecked: the paths it commits go into
        the composer as text, so there is no standing set to reopen onto."""
        from .files import list_workspace_files

        composer = self.composer()
        self._composer_prefix = composer.input.text if composer is not None else ""
        self._picker_selected = set()
        # Listed ONCE per open. This shells out to `git ls-files`; doing it per
        # keystroke costs ~9ms on a small repo and far more on a large one.
        self._picker_files = list_workspace_files(self._workspace)
        self._menu_all_rows = self._context_rows(query)
        self._menu_handler = self._commit_context
        await self._refresh_overlay("picker", "@", query, filtered=False)
        self.set_hints(HINTS["picker"])

    async def open_menu(
        self, rows: list[vm.OverlayRow], handler, *, sigil: str = "/", column: int | None = None
    ) -> None:
        """A generic single-pick overlay (model / reasoning / theme lists)."""
        self._composer_prefix = ""
        self._menu_all_rows = rows
        self._menu_handler = handler
        await self._refresh_overlay("menu", sigil, "", column=column)
        self.set_hints(HINTS["menu"])

    def _filter_rows(self, query: str) -> list[vm.OverlayRow]:
        """Every row matching the query. The overlay list scrolls, so a long
        result set is bounded by the stylesheet rather than trimmed here."""
        return filter_rows(self._menu_all_rows, query)

    async def _refresh_overlay(
        self,
        mode: str,
        sigil: str,
        query: str,
        *,
        filtered: bool = True,
        column: int | None = None,
        selected: int = 0,
    ) -> None:
        rows = self._filter_rows(query) if filtered else self._menu_all_rows
        counter = f"{len(rows)} of {len(self._menu_all_rows)}" if filtered else f"{len(rows)} files"
        state = vm.OverlayState(
            mode=mode,
            rows=rows,
            query=query,
            sigil=sigil,
            counter=counter,
            selected=min(selected, max(len(rows) - 1, 0)),
            column=column,
        )
        self._menu_rows = rows
        view = self.query(OverlayListView)
        if view:
            await view.first().set_state(state)
        else:
            await self.show_overlay(state)

    async def on_overlay_list_view_query_changed(self, message: OverlayListView.QueryChanged) -> None:
        if message.view.model.mode == "picker":
            self._menu_all_rows = self._context_rows(message.value)
            await self._refresh_overlay("picker", "@", message.value, filtered=False)
        else:
            mode = message.view.model.mode
            await self._refresh_overlay(mode, message.view.model.sigil, message.value)

    async def on_overlay_list_view_toggled(self, message: OverlayListView.Toggled) -> None:
        if message.view.model.mode != "picker":
            return
        row = self._menu_rows[message.index]
        path = _strip_spans(row.primary)
        if path in self._picker_selected:
            self._picker_selected.discard(path)
        else:
            self._picker_selected.add(path)
        query = message.view.model.query
        self._menu_all_rows = self._context_rows(query)
        # Keep the highlight where it was: checking a box must not bounce the
        # cursor back to the top row, or multi-select is unusable.
        await self._refresh_overlay("picker", "@", query, filtered=False, selected=message.index)

    async def on_overlay_list_view_committed(self, message: OverlayListView.Committed) -> None:
        handler = self._menu_handler
        self._menu_handler = None
        if handler is not None:
            await handler(message.index)

    async def on_overlay_list_view_dismissed(self, message: OverlayListView.Dismissed) -> None:
        self._menu_handler = None
        # esc gives the half-typed message back untouched — dismissing a picker
        # you opened by mistake must not cost you what you had written.
        await self._restore_composer(self._composer_prefix)

    async def _run_palette_choice(self, index: int) -> None:
        from .commands import run_palette_choice

        if not self._menu_rows:
            await self._restore_composer()
            return
        row = self._menu_rows[index]
        await self._restore_composer()
        await run_palette_choice(self, row.primary)

    def _context_rows(self, query: str) -> list[vm.OverlayRow]:
        from .files import match_files

        return picker_rows(match_files(self._picker_files, query, self._workspace), self._picker_selected)

    async def _commit_context(self, index: int) -> None:
        """Write the picked paths into the composer. Nothing checked means the
        highlighted row — picking one file should not need `space` first."""
        paths = sorted(self._picker_selected)
        if not paths and self._menu_rows:
            paths = [_strip_spans(self._menu_rows[index].primary)]
        await self._restore_composer(inline_paths(self._composer_prefix, paths))

    # ── actions ───────────────────────────────────────────────────────────────

    async def action_palette(self) -> None:
        if self.query(OverlayListView):
            return
        await self.open_palette()

    async def action_show_skills(self) -> None:
        from .commands import skills_block

        block = skills_block(self)
        await self.mount_block(block)

    def action_expand_output(self) -> None:
        for view in reversed(list(self.query(ToolBlockView))):
            output = view.model.output
            if output is not None and output.hidden_lines:
                view.toggle_expanded()
                return

    async def action_paste_image(self) -> None:
        try:
            data = await asyncio.to_thread(read_clipboard_image)
        except ClipboardUnavailable as exc:
            await self._notice(str(exc), error=True)
            return
        if data is None:
            await self._notice("no image in the clipboard")
            return
        self._pending_images.append(
            ImageContent(
                source=ImageBase64(data=base64.b64encode(data).decode("ascii"), media_type=MEDIA_TYPE),
                metadata={
                    "name": f"pasted-{len(self._pending_images) + 1}.png",
                    "size_bytes": len(data),
                    "origin": "clipboard",
                },
            ),
        )
        await self._notice(f"image attached ({len(self._pending_images)}) — enter sends it")

    async def action_cancel_run(self) -> None:
        run = self._current_run
        if run is None:
            if self._pending_images:
                self._pending_images = []
                await self._notice("attachments cleared")
            return
        try:
            run.cancel(error="cancelled by user")
        except AlreadyCancellingError:
            return
        await self._notice("cancelling — winding down the turn")

    async def action_quit(self) -> None:
        await self._quit()

    async def _quit(self) -> None:
        save_session(self.runner.session, self._session_dir)
        self.exit()

    # ── session plumbing ──────────────────────────────────────────────────────

    def _build_runner(self, session: AgentSession):
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
            instructions=self._instructions,
            extra_instructions=self._extra_instructions,
        )

    def _settle(self) -> None:
        if self.runner.idle():
            composer = self.composer()
            if composer is not None:
                composer.input.focus()
        else:  # gated / parked cancel / retry-ready — resume driving
            self._start_drive()

    async def _reset_session(self, session: AgentSession) -> None:
        self.runner, self.strategy = self._build_runner(session)
        await self.clear_transcript()
        self._live_thinking.clear()
        self._live_text.clear()
        self._tool_views.clear()
        self._tasks.clear()
        self._plan_changed = []
        self._pending_images.clear()
        self._answered.clear()
        self._denied_by_user.clear()
        await self._replay_history()
        self._refresh_status()
        self._settle()

    async def _notice(self, text: str, *, error: bool = False) -> None:
        await self.mount_block(vm.NoticeBlock(text=text, error=error))

    def _set_busy(self, busy: bool) -> None:
        composer = self.composer()
        if composer is not None:
            composer.set_placeholder("working…" if busy else "ask, or / for commands")
            if not busy:
                composer.input.focus()
        self.set_hints(HINTS["running"] if busy else HINTS["idle"])
        # The panel follows the work while a turn runs and goes back to being
        # a to-do list the moment it stops.
        self._render_plan(running=busy)
        self._refresh_status()

    def _refresh_status(self) -> None:
        session = self.runner.session
        config = session.session_config.llm_config
        tokens, cost = status_counter(session) if self._show_counter else (None, None)
        self.set_status(
            vm.StatusState(
                cwd=home_path(self._workspace),
                model=short_model(config.model),
                branch=self._git.branch,
                dirty=self._git.dirty,
                tokens=tokens,
                cost=cost,
            )
        )

    # ── modal screens (live) ──────────────────────────────────────────────────

    async def open_sessions_screen(self) -> None:
        from .commands import build_sessions_state
        from .sessions import list_sessions

        summaries = list_sessions(self._session_dir)
        state = build_sessions_state(summaries, directory_name=Path(self._session_dir).name)
        if state is None:
            await self._notice("no saved sessions for this project yet")
            return
        screen = SessionsScreen(state, self._modal_status("sessions"), HINTS["sessions"])
        screen._summaries = summaries
        await self.push_screen(screen)

    def settings_state(self, selected: int = 0) -> vm.SettingsState:
        """This app's ambient state as the settings screen's view-model."""
        from .commands import build_settings_state

        return build_settings_state(
            self.runner.session.session_config.llm_config,
            theme=self.theme,
            streaming=self._streaming,
            mode=self._mode,
            show_counter=self._show_counter,
            selected=selected,
        )

    async def open_settings_screen(self) -> None:
        screen = SettingsScreen(
            self.settings_state(),
            self._modal_status("settings · luca.json"),
            HINTS["settings"],
        )
        await self.push_screen(screen)

    async def open_cost_screen(self) -> None:
        from .usage import cost_state

        screen = CostScreen(
            cost_state(self.runner.session),
            self._modal_status("cost · this session"),
            HINTS["cost"],
        )
        await self.push_screen(screen)

    def _modal_status(self, label: str) -> vm.StatusState:
        return vm.StatusState(cwd=home_path(self._workspace), label=label)

    async def on_sessions_screen_highlighted(self, message: SessionsScreen.Highlighted) -> None:
        from .commands import session_preview

        summaries = getattr(message.screen, "_summaries", None)
        if summaries and 0 <= message.index < len(summaries):
            message.screen.update_preview(session_preview(summaries[message.index]))

    async def on_sessions_screen_resume(self, message: SessionsScreen.Resume) -> None:
        from .commands import resume_session

        await resume_session(self, message.screen, message.row)

    async def on_sessions_screen_fork(self, message: SessionsScreen.Fork) -> None:
        from .commands import fork_session_row

        await fork_session_row(self, message.screen, message.row)

    async def on_sessions_screen_delete(self, message: SessionsScreen.Delete) -> None:
        from .commands import delete_session_row

        await delete_session_row(self, message.screen, message.row)

    async def on_settings_screen_adjusted(self, message: SettingsScreen.Adjusted) -> None:
        from .commands import adjust_setting

        adjust_setting(self, message.screen, message.row, message.delta)

    async def on_settings_screen_closed(self, message: SettingsScreen.Closed) -> None:
        save_session(self.runner.session, self._session_dir)
        self._refresh_status()

    async def on_cost_screen_compact_requested(self, message: CostScreen.CompactRequested) -> None:
        message.screen.dismiss(None)
        self.runner.schedule_compaction()
        await self._notice("compacting the conversation…")
        if not self._driving:
            self._start_drive()

    async def on_key(self, event) -> None:
        # ^r at the retry prompt = "Retry now" (option 1).
        if event.key == "ctrl+r" and self._retry_prompt_active and self._approval_future is not None:
            if not self._approval_future.done():
                self._approval_future.set_result(0)
            event.stop()


def _strip_spans(text: str) -> str:
    import re

    return re.sub(r"\[/?[a-z]*\]", "", text)
