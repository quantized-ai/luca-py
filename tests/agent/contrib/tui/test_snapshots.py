"""Snapshot coverage for both catalog tiers, rendered through `GalleryApp` at
the 105×35 design size and compared against the committed SVGs
(`pytest --snapshot-update` regenerates them).

`test_catalog_entry` covers the DERIVED tier — screen × world through the
product's own projections — and `test_fixture` the hand-authored one. Plus a
responsive render at 80 columns and the sub-60-column guard.
"""

import pytest

from luca.agent.contrib.tui.gallery import GalleryApp, catalog_items, fixture_items, resolve_item

CATALOG = catalog_items()
FIXTURES = fixture_items()


@pytest.mark.parametrize("item", CATALOG, ids=[item.name.replace("/", "-") for item in CATALOG])
def test_catalog_entry(item, snap_compare):
    assert snap_compare(GalleryApp([item]), terminal_size=(105, 35))


@pytest.mark.parametrize("item", FIXTURES, ids=[item.name.replace("/", "-") for item in FIXTURES])
def test_fixture(item, snap_compare):
    assert snap_compare(GalleryApp([item]), terminal_size=(105, 35))


def test_1a_agent_loop_at_80_columns(snap_compare):
    assert snap_compare(GalleryApp([resolve_item("1a_agent_loop")]), terminal_size=(80, 30))


def test_below_60_columns_the_guard_renders(snap_compare):
    assert snap_compare(GalleryApp([resolve_item("1a_agent_loop")]), terminal_size=(59, 20))
