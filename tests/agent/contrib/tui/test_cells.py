"""Transcript cell rendering: `.text` stays the raw string; assistant and
reasoning cells render it as markdown, other cells stay plain."""

from rich.markdown import Markdown

from luca.agent.contrib.tui.cells import (
    AssistantCell,
    ReasoningCell,
    TranscriptCell,
    UserCell,
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
