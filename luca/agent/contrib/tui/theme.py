"""The Luca palette as a Textual theme — the ONLY place a hex value may appear.

Every other module references colors as theme tokens (`$accent`,
`$text-faint`, `$rule`, …) in TCSS, or resolves them at render time through
`resolve_tokens(app)`. The design handoff fixes the palette; the contrast
floor is `text-faint` (#7E7E7E, 4.2:1 on #171717) — nothing functional may be
dimmer. If a new element seems to need a dimmer grey, remove the element.

`luca-dark` is the shipped theme. Registering rather than hardcoding is what
gives the settings screen's `appearance.theme` row something to switch.
"""

from __future__ import annotations

from dataclasses import dataclass

from textual.theme import Theme

LUCA_DARK = Theme(
    name="luca-dark",
    dark=True,
    background="#171717",
    surface="#171717",
    panel="#2D2D2D",
    foreground="#E8E8E8",
    primary="#7AA8A6",  # terminal-tuned lift of the brand teal #5A9492
    accent="#7AA8A6",
    secondary="#999999",
    success="#7EA87E",
    error="#C47C7C",
    warning="#C4A87C",
    variables={
        "text-muted": "#999999",  # 6.3:1 — secondary text, tool args
        "text-faint": "#7E7E7E",  # 4.2:1 — dimmest legible tier, hard floor
        "rule": "#2D2D2D",  # rules, gutters, meter troughs
        "border-hairline": "#3A3A3A",  # composer + approval prompt borders
        "selection-bg": "#212625",  # accent at 10% over background
        "diff-add-bg": "#1F241F",
        "diff-del-bg": "#2A1E1E",
        # Textual derives block-cursor/input colors from primary by default;
        # pin the ones the frame relies on to palette values.
        "input-cursor-background": "#E8E8E8",
        "input-cursor-foreground": "#171717",
        "input-selection-background": "#212625",
    },
)

# Light-background contexts only (logo lockups, docs, README imagery) — never
# used inside the TUI.
BRAND_TEAL = "#5A9492"
BRAND_INK = "#232323"
BRAND_PAPER = "#F5F5F5"


@dataclass(frozen=True)
class Tokens:
    """The palette resolved from the app's CURRENT theme, for the few places
    Rich renderables need a concrete color (inline spans, meters, diff rows).
    Always resolved through the registered theme — never construct one from
    literals."""

    foreground: str
    muted: str
    faint: str
    accent: str
    success: str
    error: str
    rule: str
    hairline: str
    selection_bg: str
    diff_add_bg: str
    diff_del_bg: str
    surface: str


def resolve_tokens(app) -> Tokens:
    """Read the current theme's colors off a running app. Stock themes lack
    the Luca variables, so those fall back to the luca-dark values — a
    non-luca theme renders off-palette but never crashes."""
    theme = app.current_theme
    variables = {**LUCA_DARK.variables, **theme.variables}
    return Tokens(
        foreground=str(theme.foreground),
        muted=variables["text-muted"],
        faint=variables["text-faint"],
        accent=str(theme.accent),
        success=str(theme.success),
        error=str(theme.error),
        rule=variables["rule"],
        hairline=variables["border-hairline"],
        selection_bg=variables["selection-bg"],
        diff_add_bg=variables["diff-add-bg"],
        diff_del_bg=variables["diff-del-bg"],
        surface=str(theme.surface),
    )
