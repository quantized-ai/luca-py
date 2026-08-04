"""Snapshot coverage: every bundled fixture rendered through `GalleryApp` at
the 105×35 design size and compared against the committed SVGs
(`pytest --snapshot-update` regenerates them), plus a responsive render at
80 columns and the sub-60-column guard."""

import pytest

from luca.agent.contrib.tui.gallery import GalleryApp, fixture_name, list_fixtures, resolve_fixture

FIXTURES = list_fixtures()


@pytest.mark.parametrize("path", FIXTURES, ids=[fixture_name(path).replace("/", "-") for path in FIXTURES])
def test_fixture(path, snap_compare):
    assert snap_compare(GalleryApp([path]), terminal_size=(105, 35))


def test_1a_agent_loop_at_80_columns(snap_compare):
    assert snap_compare(GalleryApp([resolve_fixture("1a_agent_loop")]), terminal_size=(80, 30))


def test_below_60_columns_the_guard_renders(snap_compare):
    assert snap_compare(GalleryApp([resolve_fixture("1a_agent_loop")]), terminal_size=(59, 20))
