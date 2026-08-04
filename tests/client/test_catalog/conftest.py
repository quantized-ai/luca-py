"""Keeps the catalog tests off the contributor's real refresh cache.

`cached_records()` reads `$XDG_CACHE_HOME/luca/models.json`, so a machine where
someone has run a refresh would otherwise see different records from a machine
where nobody has.
"""

import pytest

from luca.client.catalog import _store


@pytest.fixture(autouse=True)
def _isolated_catalog(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    _store._clear_for_tests()
    yield
    _store._clear_for_tests()
