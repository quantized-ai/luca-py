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


class ModalState(BaseModel):
    """Exactly one of the three full-screen modals."""

    sessions: SessionsState | None = None
    settings: SettingsState | None = None
    cost: CostState | None = None
    model_config = _STRICT


# ── the whole screen ──────────────────────────────────────────────────────────


class ScreenState(BaseModel):
    """One renderable state of the app — the fixture schema.

    The dock is exactly one of `composer` / `approval` / `overlay` (an overlay
    dims the transcript). A `modal` covers the frame with its own screen; the
    transcript and dock still describe what sits underneath it."""

    name: str = "state"
    title: str | None = None
    status: StatusState = Field(default_factory=StatusState)
    transcript: list[Block] = Field(default_factory=list)
    composer: ComposerState | None = None
    approval: ApprovalState | None = None
    overlay: OverlayState | None = None
    modal: ModalState | None = None
    hints: list[str] = Field(default_factory=list)
    model_config = _STRICT

    def dock(self) -> str:
        if self.approval is not None:
            return "approval"
        if self.overlay is not None:
            return "overlay"
        return "composer"
