"""The context-utilization bar renders a filled block gauge colored by fill
level, followed by the percentage and token readout. No 'context' label."""

from luca.agent.contrib.tui.context_bar import render_context_bar


def test_empty_context_is_a_green_zero_bar():
    assert render_context_bar(0, 200_000) == (
        "[green]▐░░░░░░░░░░░░░░░░░░░░▌[/] 0%  0/200k"
    )


def test_mid_fill_below_threshold_is_yellow():
    assert render_context_bar(124_000, 200_000, threshold=0.8) == (
        "[yellow]▐████████████░░░░░░░░▌[/] 62%  124k/200k"
    )


def test_at_or_above_threshold_is_red():
    assert render_context_bar(180_000, 200_000, threshold=0.8) == (
        "[red]▐██████████████████░░▌[/] 90%  180k/200k"
    )


def test_over_full_is_clamped_to_a_hundred_percent():
    assert render_context_bar(500_000, 200_000) == (
        "[red]▐████████████████████▌[/] 100%  500k/200k"
    )
