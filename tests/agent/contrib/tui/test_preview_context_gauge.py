"""The context-gauge preview is made from the real widget and is interactive."""

from luca.agent.contrib.tui.context_bar import ContextBar

from .previews.context_gauge import ContextGaugePreview, SemanticContextBar


async def test_context_gauge_preview_shows_current_and_semantic_rows():
    app = ContextGaugePreview()

    async with app.run_test() as pilot:
        await pilot.pause()

        assert ([bar.text for bar in app.query(ContextBar)], len(app.query(SemanticContextBar))) == (
            [
                "[green]▐█████░░░░░░░░░░░░░░░▌[/] 25%  50k/200k",
                "[yellow]▐████████████░░░░░░░░▌[/] 62%  124k/200k",
                "[red]▐██████████████████░░▌[/] 90%  180k/200k",
            ],
            3,
        )


async def test_context_gauge_preview_can_toggle_to_a_light_theme():
    app = ContextGaugePreview()

    async with app.run_test() as pilot:
        await pilot.press("t")

        assert (app.theme, app.sub_title) == ("atom-one-light", "atom-one-light")
