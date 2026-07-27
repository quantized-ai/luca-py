"""The context-utilization bar renders a filled block gauge colored by fill
level, followed by the percentage and token readout, left-aligned to the input.
"""

from textual.widgets import Input

from luca.agent.contrib.tui import AgentApp
from luca.agent.contrib.tui.context_bar import ContextBar, render_context_bar
from luca.client.testing import FauxProvider, faux_assistant_message, faux_text

from .helpers import fresh_session, idle_again, submit, wait_until


def scripted(*responses) -> FauxProvider:
    provider = FauxProvider()
    provider.set_responses(list(responses))
    return provider


def test_empty_context_is_a_green_zero_bar():
    assert render_context_bar(0, 200_000) == "[green]▐░░░░░░░░░░░░░░░░░░░░▌[/] 0%  0/200k"


def test_mid_fill_below_threshold_is_yellow():
    assert render_context_bar(124_000, 200_000, threshold=0.8) == ("[yellow]▐████████████░░░░░░░░▌[/] 62%  124k/200k")


def test_at_or_above_threshold_is_red():
    assert render_context_bar(180_000, 200_000, threshold=0.8) == ("[red]▐██████████████████░░▌[/] 90%  180k/200k")


def test_over_full_is_clamped_to_a_hundred_percent():
    assert render_context_bar(500_000, 200_000) == "[red]▐████████████████████▌[/] 100%  500k/200k"


async def test_the_context_bar_renders_after_a_turn(tmp_path):
    app = AgentApp(
        fresh_session(),
        provider=scripted(faux_assistant_message([faux_text("Hello.")])),
        workspace=tmp_path,
        session_dir=tmp_path,
    )

    async with app.run_test() as pilot:
        await submit(pilot, "hi")
        await wait_until(pilot, lambda: idle_again(app))
        bar = app.query_one("#context-bar", ContextBar)
        assert "%" in bar.text  # a percentage
        assert "/" in bar.text  # a token readout


async def test_the_context_bar_sits_just_left_of_the_input_text(tmp_path):
    app = AgentApp(
        fresh_session(),
        provider=scripted(faux_assistant_message([faux_text("hi")])),
        workspace=tmp_path,
        session_dir=tmp_path,
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        bar = app.query_one("#context-bar", ContextBar)
        assert bar.content_region.x < app.query_one("#prompt", Input).content_region.x
