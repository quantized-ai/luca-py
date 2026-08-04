"""models.dev -> ModelInfo: the filter, the mapping, and the provider aliasing.

Against a fixture rather than the live 3.45 MB payload, so the suite neither
needs a network nor goes red when models.dev ships a model.
"""

import httpx
import pytest

from luca.client.catalog._source import (
    CatalogSourceError,
    build_records,
    fetch_payload,
    is_agent_usable,
    to_model_info,
)
from luca.client.types import ModelCost, ModelInfo

REASONING_MODEL = {
    "id": "claude-fable-5",
    "name": "Claude Fable 5",
    "family": "claude-fable",
    "attachment": True,
    "reasoning": True,
    "reasoning_options": [{"type": "effort", "values": ["low", "high"]}],
    "tool_call": True,
    "structured_output": True,
    "release_date": "2026-06-07",
    "modalities": {"input": ["text", "image", "pdf"], "output": ["text"]},
    "limit": {"context": 1000000, "output": 128000},
    "cost": {"input": 10, "output": 50, "cache_read": 1, "cache_write": 12.5},
}

PLAIN_MODEL = {
    "id": "plain-1",
    "name": "Plain",
    "tool_call": True,
    "modalities": {"input": ["text"], "output": ["text"]},
    "limit": {"context": 8000, "output": 4000},
}

IMAGE_MODEL = {
    "id": "paint-1",
    "name": "Paint",
    "tool_call": False,
    "modalities": {"input": ["text"], "output": ["image"]},
}

EMBEDDING_MODEL = {
    "id": "embed-1",
    "name": "Embed",
    "modalities": {"input": ["text"], "output": ["text"]},
}


REQUEST = httpx.Request("GET", "https://models.dev/api.json")


def payload(**providers):
    return {source: {"id": source, "models": models} for source, models in providers.items()}


# ── the agent-usability filter ───────────────────────────────────────────────


@pytest.mark.parametrize(
    ("entry", "usable"),
    [
        (REASONING_MODEL, True),
        (PLAIN_MODEL, True),
        (IMAGE_MODEL, False),  # cannot call a tool, and does not answer in text
        (EMBEDDING_MODEL, False),  # no tool_call at all
        ({"id": "x", "tool_call": True, "modalities": {"input": ["audio"], "output": ["text"]}}, False),
        ({"id": "x", "tool_call": True}, False),  # no modalities block
    ],
    ids=["reasoning", "plain", "image", "embedding", "no-text-input", "no-modalities"],
)
def test_only_models_an_agent_can_drive_survive(entry, usable):
    assert is_agent_usable(entry) is usable


# ── the mapping ──────────────────────────────────────────────────────────────


def test_a_full_record_maps_field_for_field():
    assert to_model_info("anthropic", REASONING_MODEL) == ModelInfo(
        provider="anthropic",
        model="claude-fable-5",
        display_name="Claude Fable 5",
        release_date="2026-06-07",
        family="claude-fable",
        context_window=1000000,
        max_tokens=128000,
        supports_image_input=True,
        supports_pdf_input=True,
        supports_tools=True,
        supports_structured_output="strict",
        supports_reasoning=True,
        supports_prompt_caching=True,
        cost=ModelCost(
            input_per_million_tokens=10,
            output_per_million_tokens=50,
            cached_input_per_million_tokens=1,
            cache_write_per_million_tokens=12.5,
        ),
    )


def test_a_model_with_no_cost_block_has_no_cost():
    assert to_model_info("groq", PLAIN_MODEL) == ModelInfo(
        provider="groq",
        model="plain-1",
        display_name="Plain",
        context_window=8000,
        max_tokens=4000,
        supports_tools=True,
    )


def test_prompt_caching_is_derived_from_a_cache_read_price():
    # models.dev has no caching flag; a cache_read price is only quoted for
    # models that support it
    priced = {**PLAIN_MODEL, "cost": {"input": 1, "output": 2, "cache_read": 0.1}}
    unpriced = {**PLAIN_MODEL, "cost": {"input": 1, "output": 2}}

    assert to_model_info("openai", priced).supports_prompt_caching is True
    assert to_model_info("openai", unpriced).supports_prompt_caching is False


def test_fields_models_dev_does_not_carry_keep_their_defaults():
    # nothing in the wire path reads these — model_info reaches a transport
    # only through `cost` — so leaving them at their defaults is safe
    info = to_model_info("anthropic", REASONING_MODEL)

    assert (info.supports_parallel_tool_calls, info.reasoning_signature_format, info.supports_streaming) == (
        False,
        "none",
        True,
    )


# ── building the record set ──────────────────────────────────────────────────


def test_records_are_stored_under_lucas_provider_name_not_models_devs():
    records = build_records(payload(**{"amazon-bedrock": {"m": PLAIN_MODEL}}))

    assert [record.provider for record in records] == ["bedrock"]


def test_providers_luca_cannot_route_to_are_skipped():
    records = build_records(payload(google={"m": PLAIN_MODEL}, anthropic={"m": PLAIN_MODEL}))

    assert [record.provider for record in records] == ["anthropic"]


def test_unusable_models_are_dropped_while_their_neighbours_survive():
    records = build_records(payload(openai={"a": PLAIN_MODEL, "b": IMAGE_MODEL}))

    assert [record.model for record in records] == ["plain-1"]


def test_records_come_out_sorted_so_the_vendored_file_has_a_stable_diff():
    records = build_records(
        payload(
            openai={"b": {**PLAIN_MODEL, "id": "b"}, "a": {**PLAIN_MODEL, "id": "a"}},
            anthropic={"z": {**PLAIN_MODEL, "id": "z"}},
        )
    )

    assert [(record.provider, record.model) for record in records] == [
        ("anthropic", "z"),
        ("openai", "a"),
        ("openai", "b"),
    ]


@pytest.mark.parametrize(
    "broken",
    [
        {"openai": {"models": "not a mapping"}},
        {"openai": {"models": {"m": "not a mapping"}}},
        {"openai": {"models": {"m": {"tool_call": True}}}},  # no id
        {"openai": {}},
        {},
    ],
    ids=["models-not-a-map", "entry-not-a-map", "no-id", "no-models-key", "empty"],
)
def test_a_malformed_payload_yields_nothing_rather_than_raising(broken):
    assert build_records(broken) == []


# ── fetching ─────────────────────────────────────────────────────────────────


def test_a_network_failure_becomes_one_catchable_error(monkeypatch):
    def boom(*args, **kwargs):
        raise httpx.ConnectError("no route to host")

    monkeypatch.setattr(httpx, "get", boom)

    with pytest.raises(CatalogSourceError, match="could not reach"):
        fetch_payload()


def test_a_non_json_body_becomes_one_catchable_error(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: httpx.Response(200, text="<html>nope</html>", request=REQUEST))

    with pytest.raises(CatalogSourceError, match="did not return JSON"):
        fetch_payload()


def test_an_http_error_becomes_one_catchable_error(monkeypatch):
    monkeypatch.setattr(
        httpx,
        "get",
        lambda *a, **k: httpx.Response(503, request=REQUEST),
    )

    with pytest.raises(CatalogSourceError, match="could not reach"):
        fetch_payload()


def test_a_json_array_is_not_a_provider_map(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: httpx.Response(200, json=[1, 2, 3], request=REQUEST))

    with pytest.raises(CatalogSourceError, match="expected a JSON object"):
        fetch_payload()


def test_a_model_with_no_family_or_date_still_maps():
    # 9 of 450 real records carry no family; nothing may depend on it existing
    bare = {"id": "x", "tool_call": True, "modalities": {"input": ["text"], "output": ["text"]}}

    info = to_model_info("groq", bare)

    assert (info.family, info.release_date) == (None, None)
