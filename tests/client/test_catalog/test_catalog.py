"""Catalog get / list / register, and the vendored-plus-cache layering."""

import json

from luca.client import catalog
from luca.client.catalog._data import cache_path, dump_records
from luca.client.types import ModelInfo


def test_get_returns_a_vendored_record():
    info = catalog.get("openai", "gpt-4o")

    assert info is not None
    assert info.provider == "openai"
    assert info.context_window == 128000


def test_get_returns_none_for_unknown():
    assert catalog.get("openai", "definitely-not-a-real-model") is None


def test_a_host_models_dev_does_not_know_is_absent():
    # `quantized` is a custom OpenAI-compatible host in the provider registry.
    # models.dev has no such provider, so its models come from the `models` key
    # in luca.json rather than from here, and the catalog says so honestly.
    assert catalog.get("quantized", "anthropic/claude-sonnet-4.6") is None


def test_a_model_models_dev_omits_is_absent_but_still_routable():
    # openrouter really serves this; models.dev just does not list it. The
    # catalog is metadata, never a gate — `/model` must still switch to it.
    assert catalog.get("openrouter", "anthropic/claude-opus-4-8") is None


def test_list_all_returns_records():
    assert len(catalog.list()) > 0


def test_list_by_provider_filters():
    openai_only = catalog.list(provider="openai")

    assert openai_only
    assert all(record.provider == "openai" for record in openai_only)


def test_list_by_supports_filters():
    vision = catalog.list(supports="vision")

    assert all(record.supports_image_input for record in vision)


def test_register_adds_record():
    catalog.register(model="x-test", provider="custom", info=ModelInfo(context_window=1000))

    assert catalog.get("custom", "x-test") == ModelInfo(
        provider="custom",
        model="x-test",
        context_window=1000,
    )


# ── the vendored floor and the refresh cache ─────────────────────────────────


def test_the_cache_adds_models_the_vendored_file_does_not_have():
    dump_records(
        cache_path(),
        [ModelInfo(provider="openai", model="gpt-6-preview", context_window=2000000)],
    )

    assert catalog.get("openai", "gpt-6-preview") == ModelInfo(
        provider="openai",
        model="gpt-6-preview",
        context_window=2000000,
    )


def test_the_cache_wins_where_the_two_overlap():
    dump_records(
        cache_path(),
        [ModelInfo(provider="openai", model="gpt-4o", context_window=999)],
    )

    assert catalog.get("openai", "gpt-4o").context_window == 999


def test_the_cache_never_subtracts_from_the_vendored_floor():
    dump_records(cache_path(), [ModelInfo(provider="openai", model="gpt-6-preview")])

    assert catalog.get("anthropic", "claude-fable-5") is not None


def test_a_corrupt_cache_leaves_the_vendored_floor_standing():
    # the compaction gauge reads this on every check; a half-written file must
    # not take every context window with it
    path = cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"source": "models.dev", "records": [{"provider"')

    assert catalog.get("openai", "gpt-4o") is not None


def test_one_unreadable_record_does_not_discard_its_neighbours():
    path = cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "source": "models.dev",
                "records": [
                    {"provider": "openai", "model": "good", "context_window": 10},
                    {"provider": "openai", "model": "bad", "not_a_field": True},
                ],
            }
        )
    )

    assert catalog.get("openai", "good") is not None
    assert catalog.get("openai", "bad") is None


def test_a_missing_cache_costs_nothing():
    assert not cache_path().exists()
    assert catalog.get("openai", "gpt-4o") is not None
