"""Transcript cell rendering: `.text` stays the raw string; assistant and
reasoning cells render it as markdown, other cells stay plain."""

from rich.markdown import Markdown

from luca.agent.contrib.tui.cells import (
    AssistantCell,
    CompactionCell,
    ReasoningCell,
    TranscriptCell,
    UserCell,
)
from luca.agent.core.models import (
    CompactionEntry,
    CompactionSource,
    TextContent,
)


def test_assistant_cell_keeps_raw_text_but_renders_markdown():
    cell = AssistantCell("**hi**")
    assert cell.text == "**hi**"
    assert isinstance(cell._renderable("**hi**"), Markdown)


def test_reasoning_cell_renders_markdown():
    assert isinstance(ReasoningCell("x")._renderable("**thinking**"), Markdown)


def test_empty_text_is_not_rendered_as_markdown():
    assert AssistantCell("")._renderable("") == ""
    assert AssistantCell("x")._renderable("   ") == ""


def test_base_and_user_cells_stay_plain():
    assert TranscriptCell("x")._renderable("**x**") == "**x**"
    assert UserCell("x")._renderable("**x**") == "**x**"


def test_compaction_cell_renders_the_summary_and_counts_what_it_replaced():
    entry = CompactionEntry(
        id="cmp",
        created_at=1000,
        source=CompactionSource.POLICY,
        parts=[TextContent(text="**the story so far**")],
        compacted_nodes=["u1", "a1", "tf1"],
    )

    cell = CompactionCell(entry)

    assert cell.text == "**the story so far**"
    assert cell.border_subtitle == "replaced 3 entries"
    assert isinstance(cell._renderable(cell.text), Markdown)


def test_a_compaction_cell_for_a_single_replaced_entry_is_singular():
    entry = CompactionEntry(
        id="cmp",
        created_at=1000,
        source=CompactionSource.USER,
        parts=[TextContent(text="brief")],
        compacted_nodes=["u1"],
    )

    assert CompactionCell(entry).border_subtitle == "replaced 1 entry"
