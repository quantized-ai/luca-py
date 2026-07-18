"""Catalog get / list / register."""

import pytest

from luca.client import catalog
from luca.client.catalog import _store
from luca.client.types import ModelCost, ModelInfo


@pytest.fixture(autouse=True)
def _restore_catalog():
    yield
    _store._clear_for_tests()


def test_get_returns_known_record():
    info = catalog.get("openai", "gpt-4o")
    assert info is not None
    assert info.provider == "openai"
    assert info.context_window == 128000


def test_get_returns_none_for_unknown():
    assert catalog.get("openai", "definitely-not-a-real-model") is None


def test_list_all_returns_records():
    records = catalog.list()
    assert len(records) > 0


def test_list_by_provider_filters():
    openai_only = catalog.list(provider="openai")
    assert openai_only
    assert all(r.provider == "openai" for r in openai_only)


def test_list_by_supports_filters():
    vision = catalog.list(supports="vision")
    assert all(r.supports_image_input for r in vision)


def test_register_adds_record():
    catalog.register(
        model="x-test",
        provider="custom",
        info=ModelInfo(context_window=1000),
    )
    info = catalog.get("custom", "x-test")
    assert info is not None
    assert info.provider == "custom"
    assert info.model == "x-test"
    assert info.context_window == 1000
