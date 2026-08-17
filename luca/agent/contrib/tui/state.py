"""The declarative view-model of the TUI — what any screen state IS, as data.

Every visual state the app can show is expressible as one `ScreenState`:
the status line, a list of transcript blocks, exactly one bottom dock
(composer, approval prompt, or overlay list), an optional modal, and the hint
legend. Widgets render these models; the live agent wiring produces them from
events; gallery fixtures load them from YAML. One vocabulary, three producers
— which is what makes any state renderable without driving a real agent.

The block vocabulary mirrors the design handoff: user turn, thinking,
assistant text, tool call (with its result / verbatim output riding along),
list block, diff block — plus `notice` (turn-level failures the product needs
a home for) and `task` (a subagent's nested conversation, rendered with the
handoff's gutter idiom).

Inline color spans use theme-token tags, never hex: `[accent]…[/]`,
`[error]…[/]`, `[success]…[/]`, `[muted]…[/]`, `[faint]…[/]`, `[fg]…[/]`.
Assistant text additionally renders `` `code spans` `` in `$accent`, per the
handoff. No Textual imports here — this module is pure data.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

_STRICT = ConfigDict(extra="forbid")


# ── transcript blocks ─────────────────────────────────────────────────────────


class DiffStat(BaseModel):
    """`+N −M`, on a tool header or a result row. Either side may be absent."""

    add: int | None = None
    remove: int | None = Field(default=None, alias="del")
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class UserBlock(BaseModel):
    kind: Literal["user"] = "user"
    text: str
    model_config = _STRICT


class ThinkingBlock(BaseModel):
    """Collapsed one-liner. `activity` set = in progress (`∴ reading …`);
    otherwise `∴ thought for {duration}s`."""

    kind: Literal["thinking"] = "thinking"
    duration: float | None = None
    activity: str | None = None
    model_config = _STRICT


class TextBlock(BaseModel):
    """Assistant text, wrapped at the 84-column measure. A streaming block
    ends with the 1-cell block cursor."""

    kind: Literal["text"] = "text"
    text: str
    streaming: bool = False
    model_config = _STRICT


class ToolOutput(BaseModel):
    """Verbatim multi-line output inside the `│` gutter, then an optional
    summary row (`exit 1 · 1.9s · 11 lines hidden`) with the `^o expand`
    affordance. Lines accept theme-token spans."""

    lines: list[str] = Field(default_factory=list)
    summary: str | None = None
    expand_hint: bool = False
    hidden_lines: list[str] = Field(default_factory=list)
    model_config = _STRICT


class ToolResult(BaseModel):
    """The single-row result under a call: a faint summary, a diff stat, a
    trailing note (`approved by rule · edits under luca/`) — any combination."""

    summary: str | None = None
    stat: DiffStat | None = None
    note: str | None = None
    model_config = _STRICT


ToolStatus = Literal["pending", "running", "ok", "denied", "error"]


class ToolBlock(BaseModel):
    """One tool call: the `▸ name arg` header, optionally a right-aligned
    header annotation (a diff stat while pending, `loaded`, …), and its result
    — a summary row, or a verbatim output gutter."""

    kind: Literal["tool"] = "tool"
    tool: str
    arg: str = ""
    status: ToolStatus = "ok"
    stat: DiffStat | None = None  # right-aligned on the header row
    note_right: str | None = None  # right-aligned text on the header row
    result: ToolResult | None = None
    output: ToolOutput | None = None
    model_config = _STRICT


ListGlyph = Literal["done", "active", "pending", "included", "none"]


class ListRow(BaseModel):
    glyph: ListGlyph = "none"
    text: str
    annotation: str | None = None
    # Struck through — a settled todo. Said here rather than derived from the
    # glyph: the other list users (skills, context set, consumers) share the
    # vocabulary and none of them wants the treatment.
    strike: bool = False
    model_config = _STRICT


class ListBlock(BaseModel):
    """Plan / active skills / context set / biggest consumers: a faint label
    row, then rows with a 1-cell status-glyph column. `column` fixes the text
    column's width in cells when rows carry annotations."""

    kind: Literal["list"] = "list"
    label: str | None = None
    rows: list[ListRow] = Field(default_factory=list)
    column: int | None = None
    model_config = _STRICT


class DiffLine(BaseModel):
    num: int | None = None  # `num`, not `no` — bare `no` is a YAML boolean
    sign: Literal["+", "-"] | None = None
    text: str = ""
    model_config = _STRICT


class DiffBlock(BaseModel):
    """A unified diff in the gutter: `[5 line-no][3 sign][code]`, removed rows
    first, added rows renumbered, row backgrounds spanning the full width."""

    kind: Literal["diff"] = "diff"
    lines: list[DiffLine] = Field(default_factory=list)
    model_config = _STRICT


class NoticeBlock(BaseModel):
    """A turn-level notice (cancellation, turn failure). Not part of the
    handoff's seven-block vocabulary — rendered as a single faint (or error)
    row so it stays inside the design's idiom."""

    kind: Literal["notice"] = "notice"
    text: str
    error: bool = False
    model_config = _STRICT


TaskStatus = Literal["waiting", "running", "done", "failed"]


class TaskBlock(BaseModel):
    """A subagent's conversation: a `task · description` label row with its
    status, then the child's own blocks indented in a `│` gutter. An extension
    of the handoff vocabulary (the design predates subagents), built from its
    existing idioms — the faint label row and the gutter."""

    kind: Literal["task"] = "task"
    description: str
    status: TaskStatus = "running"
    blocks: list[Block] = Field(default_factory=list)
    model_config = _STRICT


Block = Annotated[
    UserBlock | ThinkingBlock | TextBlock | ToolBlock | ListBlock | DiffBlock | NoticeBlock | TaskBlock,
    Field(discriminator="kind"),
]


# ── frame state ───────────────────────────────────────────────────────────────


class StatusState(BaseModel):
    """The one-row status bar. On modal screens `label` replaces the
    model/branch/counter run (`sessions`, `settings · .luca/config.toml`)."""

    cwd: str = "~"
    model: str | None = None
    branch: str | None = None
    dirty: bool = False
    tokens: str | None = None
    cost: str | None = None
    label: str | None = None
    model_config = _STRICT


class ComposerState(BaseModel):
    placeholder: str = "ask, or / for commands"
    text: str = ""
    disabled: bool = False
    model_config = _STRICT


class ApprovalOption(BaseModel):
    label: str
    key_hint: str | None = None  # right-aligned (`esc` on Cancel turn)
    model_config = _STRICT


class ApprovalState(BaseModel):
    """The bordered prompt that replaces the composer. `question` accepts
    theme-token spans (`Apply this edit to [accent]luca/events.py[/]?`)."""

    question: str
    options: list[ApprovalOption]
    selected: int = 0
    model_config = _STRICT


class OverlayRow(BaseModel):
    """One row of the palette / context picker. Palette rows: `primary` is
    the command, `secondary` the description. Picker rows: `checked` drives
    the checkbox, `primary` the path (with `[accent]` match spans),
    `annotation` the token cost."""

    primary: str
    secondary: str | None = None
    annotation: str | None = None
    checked: bool | None = None
    model_config = _STRICT


class OverlayState(BaseModel):
    """The bordered list that replaces the composer in place and grows
    upward, dimming the transcript behind it."""

    mode: Literal["palette", "picker", "menu"] = "palette"
    rows: list[OverlayRow] = Field(default_factory=list)
    query: str = ""
    sigil: str = "/"
    counter: str | None = None  # `6 of 14`
    selected: int = 0
    column: int | None = None  # primary column width in cells
    model_config = _STRICT


# ── the question set (the `ask_user` dock) ────────────────────────────────────


class QuestionOption(BaseModel):
    """One row of a question. `kind` is what makes the row's column treatment
    differ:

    - `"option"` — a normal answer. Tick column in multi mode.
    - `"custom"` — the free-text row. Tick column in multi mode; `checked` is
      derived, and is True exactly when `text` is non-empty.
    - `"chat"`   — Chat about this. NO tick column, ever: the caret column and
      the label only, so it cannot read as a tickable answer, and `space` is
      inert on it. It is the way out, not an answer.
    """

    label: str
    kind: Literal["option", "custom", "chat"] = "option"
    checked: bool = False  # the multi-select tick; ignored when mode == "single"
    text: str | None = None  # kind == "custom" only: what the user typed
    key_hint: str | None = None  # right-aligned — `enter` on the chat row
    model_config = _STRICT


class Question(BaseModel):
    """One tab of the set.

    `body` is PRE-WRAPPED at the 99-column measure by the producer, the way
    `ToolOutput.lines` are — the widget does not reflow prose.

    `options` is the agent's list; the producer appends the custom and chat
    rows, so a `Question` is never authored with them and they are always
    last."""

    tab: str  # the tab label, 1–2 words, ≤ 12 cells
    title: str  # the question; code spans allowed
    body: list[str] = Field(default_factory=list)
    mode: Literal["single", "multi"] = "single"
    options: list[QuestionOption] = Field(default_factory=list)
    # THE CARET, and only the caret. `↑↓` move it and nothing else — moving it
    # over an answered radio question must not silently re-answer, which is
    # what conflating it with the pick would do.
    selected: int = 0
    # THE PICK, in single-select mode: the index of the option the user
    # committed, or None while the question is open. Multi-select carries its
    # answer on each option's `checked` instead, so this stays None there.
    answer: int | None = None
    answered: bool = False
    model_config = _STRICT


class QuestionSetState(BaseModel):
    """The fourth dock: the agent's questions, replacing the composer exactly
    as the approval prompt does.

    `phase` is what the dock shows — the question panel, or the confirmation.
    It is the ONLY thing that decides which, so the two can never both render.
    `extra` is the confirmation's optional free-text field, which replaced
    per-question notes once the custom-answer row made them redundant.

    Deliberately absent, in the spirit of `ExecutionDeferred` being an empty
    marker: no `skipped` state (every question is answered, and the custom row
    means every question CAN be), no per-question `required` or `recommended`
    flag (the body says which one the agent prefers, in prose), no per-question
    notes, and no progress/percent field — the tabs are the progress
    indicator."""

    questions: list[Question] = Field(default_factory=list)
    active: int = 0
    phase: Literal["asking", "confirming"] = "asking"
    editing_custom: bool = False  # the active question's custom field has the keyboard
    extra: str | None = None
    model_config = _STRICT

    @property
    def settled(self) -> bool:
        """Every question answered — the predicate behind BOTH `enter`'s
        meaning and the hint legend. One source, so they cannot disagree."""
        return bool(self.questions) and all(question.answered for question in self.questions)


# ── modal screens ─────────────────────────────────────────────────────────────


class SessionRow(BaseModel):
    when: str
    first_message: str
    turns: str
    tokens: str
    cost: str
    id: str | None = None  # live use only; fixtures omit it
    model_config = _STRICT


class SessionsState(BaseModel):
    count_line: str
    rows: list[SessionRow] = Field(default_factory=list)
    selected: int = 0
    preview: list[str] = Field(default_factory=list)  # last-turn rows, spans allowed
    model_config = _STRICT


SettingColor = Literal["accent", "error", "muted", "foreground"]


class SettingRow(BaseModel):
    name: str
    value: str
    color: SettingColor | None = None  # value color override (permissions)
    model_config = _STRICT


class SettingsGroup(BaseModel):
    label: str
    rows: list[SettingRow] = Field(default_factory=list)
    model_config = _STRICT


class SettingsState(BaseModel):
    groups: list[SettingsGroup] = Field(default_factory=list)
    selected: int = 0  # flat index across groups
    swatch_label: str | None = None
    model_config = _STRICT


MeterColor = Literal["accent", "foreground", "faint", "rule", "hairline"]


class CostItem(BaseModel):
    label: str
    tokens: str
    cost: str
    fraction: float  # of the widest meter — proportional to COST, not tokens
    color: MeterColor = "accent"
    model_config = _STRICT


class ContextWindowState(BaseModel):
    used: str  # `31.7k / 200k`
    percent: str  # `16%`
    context_fraction: float
    reply_fraction: float
    legend: list[str] = Field(default_factory=list)  # spans allowed
    model_config = _STRICT


class ConsumerRow(BaseModel):
    label: str
    tokens: str
    model_config = _STRICT


class CostState(BaseModel):
    headline: str  # `$0.21`
    subline: str  # `14 turns · 22m · sonnet-4.5`
    items: list[CostItem] = Field(default_factory=list)
    context: ContextWindowState | None = None
    consumers: list[ConsumerRow] = Field(default_factory=list)
    model_config = _STRICT


McpRowState = Literal["connected", "needs_auth", "failed", "disabled"]


class McpRow(BaseModel):
    label: str
    state: McpRowState
    detail: str  # tool count and protocol, the error, or why there is neither
    action: str  # what enter does to THIS row
    model_config = _STRICT


class McpState(BaseModel):
    count_line: str
    rows: list[McpRow] = Field(default_factory=list)
    selected: int = 0
    notes: list[str] = Field(default_factory=list)  # excluded tools, with the reason
    model_config = _STRICT


class ModalState(BaseModel):
    """Exactly one of the full-screen modals."""

    sessions: SessionsState | None = None
    settings: SettingsState | None = None
    cost: CostState | None = None
    mcp: McpState | None = None
    model_config = _STRICT


# ── the whole screen ──────────────────────────────────────────────────────────


class ScreenState(BaseModel):
    """One renderable state of the app — the fixture schema.

    The dock is exactly one of `composer` / `approval` / `questions` /
    `overlay` (an overlay dims the transcript). A `modal` covers the frame with
    its own screen; the transcript and dock still describe what sits underneath
    it.

    `plan` is the sticky todo panel, which sits BETWEEN the transcript and the
    dock and belongs to neither: it outlives the turn that wrote it and stays
    put while an approval prompt or an overlay takes the dock."""

    name: str = "state"
    title: str | None = None
    status: StatusState = Field(default_factory=StatusState)
    transcript: list[Block] = Field(default_factory=list)
    plan: ListBlock | None = None
    composer: ComposerState | None = None
    approval: ApprovalState | None = None
    questions: QuestionSetState | None = None
    overlay: OverlayState | None = None
    modal: ModalState | None = None
    hints: list[str] = Field(default_factory=list)
    model_config = _STRICT

    def dock(self) -> str:
        # A question set and an approval prompt cannot both hold the dock,
        # because a deferred `ask_user` parks the turn before any other tool
        # dispatches — but the order is fixed anyway rather than left to
        # chance, so a hand-authored fixture cannot render an ambiguous state.
        if self.approval is not None:
            return "approval"
        if self.questions is not None:
            return "questions"
        if self.overlay is not None:
            return "overlay"
        return "composer"
