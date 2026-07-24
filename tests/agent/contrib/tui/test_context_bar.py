"""The context gauge maps usage to a percent, a token readout, and a color
state (green below 60%, yellow up to the threshold, red at or above it)."""

from luca.agent.contrib.tui.context_bar import context_stats


def test_empty_context_is_ok_and_zero():
    assert context_stats(0, 200_000, 0.8) == (0, "0/200k", "ok")


def test_a_low_fill_stays_ok():
    assert context_stats(2_048, 200_000, 0.8) == (1, "2k/200k", "ok")


def test_past_sixty_percent_warns():
    assert context_stats(124_000, 200_000, 0.8) == (62, "124k/200k", "warn")


def test_at_or_above_the_threshold_is_danger():
    assert context_stats(160_000, 200_000, 0.8) == (80, "160k/200k", "danger")


def test_over_full_clamps_to_a_hundred():
    assert context_stats(500_000, 200_000, 0.8) == (100, "500k/200k", "danger")


def test_a_zero_window_never_divides():
    assert context_stats(10, 0, 0.8) == (0, "10/0", "ok")
