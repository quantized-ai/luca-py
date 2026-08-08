"""`ShellNativeMiddleware` — the shell plugin's entire provider knowledge.

The core stores and projects every tool call provider-blind: under its
internal `ToolSpec.name`, with no extras, as an ordinary function call —
a form both providers accept even when the name is not in the request's
tools. This middleware is the last-mile UPGRADE of that default, applied by
the plugin that owns the native tools, over four slots:

    build_tool_list          which of my tools this request advertises (drop)
    adapt_tool_declarations  my active natives -> the client's native items
    before_llm_call          my active calls/results -> their native payloads
    after_llm_response       inbound native calls -> my internal names

Nothing else in the framework learns that a native tool exists. Uninstall the
plugin and every session still projects — verbatim is the default, not a
branch.

All per-call state is read from the session, which the runner stamps at the
top of every drive iteration (`session.llm_config`, `session.use_native_tools`)
— which is also what makes a mid-session model switch or native flip take
effect on the very next call.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import ClassVar

from luca.agent.core import AgentSession, ToolSpec
from luca.client.providers.anthropic import BashTool, TextEditorTool
from luca.client.providers.openai import ApplyPatchTool, LocalShellTool
from luca.client.types import (
    AssistantMessage as ClientAssistantMessage,
    TextBlock as ClientTextBlock,
    ToolCall as ClientToolCall,
    ToolMessage as ClientToolMessage,
)

from .support import supported_native_tools


def active_natives(session: AgentSession) -> frozenset[str]:
    """The natives the NEXT call may use: supported by the ACTIVE (provider,
    model) pair, and switched on. The one place the capability table and the
    on/off flag meet."""
    if not session.use_native_tools:
        return frozenset()
    return supported_native_tools(session.llm_config)


class ShellNativeMiddleware:
    WIRE_NAMES: ClassVar[dict[str, str]] = {  # internal -> provider-facing, OUT
        "openai_apply_patch": "apply_patch",
        "openai_shell": "shell",
        "anthropic_text_editor_20250728": "str_replace_based_edit_tool",
        "anthropic_bash_20250124": "bash",
    }
    ADOPT_BY_CUSTOM_TYPE: ClassVar[dict[str, str]] = {  # inbound OpenAI: typed calls
        "apply_patch_call": "openai_apply_patch",
        "shell_call": "openai_shell",
    }
    ADOPT_BY_WIRE_NAME: ClassVar[dict[str, str]] = {  # inbound Anthropic: plain calls
        "str_replace_based_edit_tool": "anthropic_text_editor_20250728",
        "bash": "anthropic_bash_20250124",
    }
    DECLARATIONS: ClassVar[dict[str, type]] = {
        "openai_apply_patch": ApplyPatchTool,
        "openai_shell": LocalShellTool,
        "anthropic_text_editor_20250728": TextEditorTool,
        "anthropic_bash_20250124": BashTool,
    }
    REPLACES: ClassVar[dict[str, tuple[str, ...]]] = {
        # Which generics each native makes redundant. Per NATIVE, not per
        # provider, because support is per MODEL: gpt-5.5-pro has the native
        # shell and no native apply_patch, so it keeps the generic editors.
        "openai_apply_patch": ("edit", "write", "apply_patch", "delete_file"),
        "openai_shell": ("bash",),
        "anthropic_text_editor_20250728": ("edit", "write", "apply_patch"),
        "anthropic_bash_20250124": ("bash",),
    }

    def __init__(self, session: AgentSession) -> None:
        self.session = session  # the ONLY state — everything per-call reads it

    # ── the drop rule, shared with the plugin's system prompt ───────────────

    @classmethod
    def visible_names(cls, session: AgentSession, names: Iterable[str]) -> list[str]:
        """`names` minus the generics the active natives replace, minus the
        natives that are not active. Order preserved; a name this middleware
        does not own rides through untouched."""
        active = active_natives(session)
        replaced = {generic for name in active for generic in cls.REPLACES[name]}
        return [name for name in names if name not in replaced and (name not in cls.REPLACES or name in active)]

    # ── OUT: the advertised tools — two slots, two vocabularies ─────────────

    def build_tool_list(
        self,
        session: AgentSession,
        conversation_id: str,
        specs: list[ToolSpec],
    ) -> list[ToolSpec]:
        """Spec vocabulary: decides WHICH of my tools this request advertises.
        Drop only, never reshape."""
        keep = set(self.visible_names(self.session, [spec.name for spec in specs]))
        return [spec for spec in specs if spec.name in keep]

    def adapt_tool_declarations(
        self,
        session: AgentSession,
        conversation_id: str,
        tools: list,
    ) -> list:
        """Client vocabulary: swaps my surviving native function declarations
        for the client's native declaration items. Only actives survive
        `build_tool_list`, so a name match is enough."""
        return [self.DECLARATIONS[tool.name]() if tool.name in self.DECLARATIONS else tool for tool in tools]

    # ── OUT: the projected messages ─────────────────────────────────────────

    def before_llm_call(
        self,
        session: AgentSession,
        conversation_id: str,
        messages: list,
        system_message: str | None,
    ) -> tuple[list, str | None]:
        """The projector already produced the DEFAULT form. This upgrades MY
        active calls to native. Not-mine, foreign-family and native-off calls
        are untouched: verbatim is the default, not a branch."""
        active = active_natives(self.session)
        if not active:
            return messages, system_message
        # The ids actually on the projected path. A tool message whose
        # execution is NOT among them is a SUBSTITUTE the projector emitted in
        # its place — a pruned entry — and re-attaching the stored payload
        # would put the very output that was pruned back on the wire.
        on_path = set(self.session.conversations[conversation_id].nodes)
        out = []
        for message in messages:
            if isinstance(message, ClientAssistantMessage):
                message = self._upgrade_calls(message, active)
            elif isinstance(message, ClientToolMessage):
                message = self._upgrade_result(message, active, on_path)
            out.append(message)
        return out, system_message

    def _upgrade_calls(
        self,
        message: ClientAssistantMessage,
        active: frozenset[str],
    ) -> ClientAssistantMessage:
        blocks = []
        for block in message.content:
            if isinstance(block, ClientToolCall) and block.name in active:
                execution = self.session.get_tool_execution(block.id)
                block = block.model_copy(
                    update={
                        "name": self.WIRE_NAMES[block.name],
                        "extras": copy.deepcopy(execution.raw_tool_call.extras),  # stored at adoption
                    },
                )
            blocks.append(block)
        return message.model_copy(update={"content": blocks})

    def _upgrade_result(
        self,
        message: ClientToolMessage,
        active: frozenset[str],
        on_path: set[str],
    ) -> ClientToolMessage:
        execution = self.session.get_tool_execution(message.tool_call_id)
        if execution is None or execution.raw_tool_call.name not in active:
            return message
        result = execution.result
        if execution.id not in on_path:
            # PRUNED: the projector put a placeholder where this execution's
            # output used to be, so the stored payload must not go back on the
            # wire — that would re-send the very output that was pruned. The
            # placeholder still has to be STRUCTURED for shell, and here the
            # real outcome is known, so it is kept.
            failed = result is None or result.is_error
            return self._structured(message, execution.raw_tool_call.name, 1 if failed else 0)
        if result is not None and result.extras:
            return message.model_copy(update={"extras": copy.deepcopy(result.extras)})  # stored at birth
        # DERIVED outcome (rejected / failed / awaiting approval): no
        # ExecutionResult was ever stored, but shell's native result MUST be
        # structured — synthesize it from the text the projector derived. This
        # is the whole "denied native call" problem, solved where the knowledge
        # lives. Everything else's plain text IS its native form.
        return self._structured(message, execution.raw_tool_call.name, 1)

    def _structured(self, message: ClientToolMessage, name: str, exit_code: int) -> ClientToolMessage:
        """A shell result built from the text the projector derived. A no-op
        for every other native — only `shell` requires a structured wire
        result, and its structure is the one thing prose cannot carry."""
        if name != "openai_shell":
            return message
        text = "".join(block.text for block in message.content if isinstance(block, ClientTextBlock))
        return message.model_copy(
            update={
                "extras": {
                    "custom_type": "shell_call_output",
                    "results": [
                        {
                            "stdout": "" if exit_code else text,
                            "stderr": text if exit_code else "",
                            "outcome": {"type": "exit", "exit_code": exit_code},
                        },
                    ],
                },
            },
        )

    # ── IN: adoption ────────────────────────────────────────────────────────

    def after_llm_response(
        self,
        session: AgentSession,
        conversation_id: str,
        message: ClientAssistantMessage,
    ) -> ClientAssistantMessage:
        """Rewrites MY native call blocks to the canonical form under the
        INTERNAL name, generic extras attached — before the runner records
        anything. Everything downstream sees an ordinary `ToolCall`."""
        blocks = []
        for block in message.content:
            if isinstance(block, ClientToolCall):
                block = self._adopt(block) or block
            blocks.append(block)
        return message.model_copy(update={"content": blocks})

    def _adopt(self, block: ClientToolCall) -> ClientToolCall | None:
        generic = block.as_generic()  # typed subclass -> base + extras (lossless)
        name = self.ADOPT_BY_CUSTOM_TYPE.get(generic.extras.get("custom_type"))
        if name is None:
            # A plain wire name is only mine while that native is ACTIVE —
            # `bash` is also a generic tool of ours, and on a provider without
            # natives it is exactly that.
            active = active_natives(self.session)
            if generic.name in {self.WIRE_NAMES[internal] for internal in active}:
                name = self.ADOPT_BY_WIRE_NAME.get(generic.name)
        if name is None:
            return None
        return generic.model_copy(update={"name": name})
