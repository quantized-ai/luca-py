"""models.dev as the catalog's source of truth.

One endpoint, `https://models.dev/api.json`: a single object keyed by provider
id, each holding a `models` map. It carries ~180 providers and ~6000 models,
most of which luca cannot route to and many of which an agent cannot use, so
everything here is about narrowing that down and mapping what survives onto
`ModelInfo`.

Shared by the offline generator and the runtime refresh, so the mapping exists
once and the vendored snapshot and a user's cache cannot disagree.

The payload is read as plain dicts with `.get`. models.dev grows fields over
time and a new one must never fail a refresh — `ModelInfo` keeps `extra="forbid"`
for luca's own shape, but nothing here validates theirs.
"""

from __future__ import annotations

import httpx

from ..types.catalog import ModelCost, ModelInfo

MODELS_DEV_URL = "https://models.dev/api.json"
DEFAULT_TIMEOUT = 30.0

# luca's provider name → the models.dev id. Explicit rather than derived:
# `bedrock` is `amazon-bedrock` there and no rule produces that.
#
# Deliberately absent: `ollama` (local — its models are whatever you pulled)
# and `quantized` (a custom host). Neither is in models.dev, and both are
# served by the `models` key in luca.json instead.
PROVIDER_IDS: dict[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "openrouter": "openrouter",
    "bedrock": "amazon-bedrock",
    "groq": "groq",
    "deepseek": "deepseek",
}


class CatalogSourceError(Exception):
    """models.dev could not be fetched or made sense of."""


def is_agent_usable(entry: dict) -> bool:
    """Whether an agent can drive this model at all.

    Image, embedding and speech models are a quarter of the payload and none of
    them can call a tool, so offering one is a dead end."""
    modalities = entry.get("modalities") or {}
    return bool(
        entry.get("tool_call")
        and "text" in (modalities.get("input") or [])
        and "text" in (modalities.get("output") or [])
    )


def to_model_cost(entry: dict) -> ModelCost | None:
    """models.dev quotes dollars per million tokens, the same unit `ModelCost`
    uses, so the numbers carry across unchanged. `reasoning_per_million_tokens`
    has no source and stays unset."""
    cost = entry.get("cost") or {}
    if not cost:
        return None
    return ModelCost(
        input_per_million_tokens=cost.get("input"),
        output_per_million_tokens=cost.get("output"),
        cached_input_per_million_tokens=cost.get("cache_read"),
        cache_write_per_million_tokens=cost.get("cache_write"),
    )


def to_model_info(provider: str, entry: dict) -> ModelInfo:
    """One models.dev model as a `ModelInfo` under LUCA's provider name.

    `supports_parallel_tool_calls`, `reasoning_signature_format` and
    `supports_streaming` have no source here and keep their field defaults.
    That is safe because no transport reads them — `model_info` reaches the
    wire only through `cost`."""
    modalities = entry.get("modalities") or {}
    inputs = modalities.get("input") or []
    limit = entry.get("limit") or {}
    cost = entry.get("cost") or {}
    return ModelInfo(
        provider=provider,
        model=entry["id"],
        display_name=entry.get("name"),
        release_date=entry.get("release_date"),
        family=entry.get("family"),
        context_window=limit.get("context"),
        max_tokens=limit.get("output"),
        supports_image_input="image" in inputs,
        supports_audio_input="audio" in inputs,
        supports_pdf_input="pdf" in inputs,
        supports_video_input="video" in inputs,
        supports_tools=bool(entry.get("tool_call")),
        # models.dev has one boolean where ModelInfo distinguishes strict from
        # loose, so a true maps to the stronger claim it can actually make.
        supports_structured_output="strict" if entry.get("structured_output") else "none",
        supports_reasoning=bool(entry.get("reasoning")),
        # Derived: models.dev has no caching flag, but a cache-read price is
        # only quoted for models that support it.
        supports_prompt_caching=cost.get("cache_read") is not None,
        cost=to_model_cost(entry),
    )


def build_records(payload: dict) -> list[ModelInfo]:
    """Every agent-usable model from the providers luca can route to.

    A provider or model entry that is not shaped the way we expect is skipped
    rather than fatal: one bad record upstream must not cost the whole
    catalog."""
    records: list[ModelInfo] = []
    for provider, source_id in PROVIDER_IDS.items():
        models = (payload.get(source_id) or {}).get("models") or {}
        if not isinstance(models, dict):
            continue
        for entry in models.values():
            if not isinstance(entry, dict) or not entry.get("id"):
                continue
            if is_agent_usable(entry):
                records.append(to_model_info(provider, entry))
    return sorted(records, key=lambda info: (info.provider or "", info.model or ""))


def fetch_payload(*, timeout: float = DEFAULT_TIMEOUT, url: str = MODELS_DEV_URL) -> dict:
    """The raw models.dev payload. Raises `CatalogSourceError` for anything
    that goes wrong, so a caller has one thing to catch."""
    try:
        response = httpx.get(url, timeout=timeout, follow_redirects=True)
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPError as exc:
        raise CatalogSourceError(f"could not reach {url}: {exc}") from exc
    except ValueError as exc:
        raise CatalogSourceError(f"{url} did not return JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise CatalogSourceError(f"{url}: expected a JSON object of providers")
    return payload


def fetch_records(*, timeout: float = DEFAULT_TIMEOUT, url: str = MODELS_DEV_URL) -> list[ModelInfo]:
    return build_records(fetch_payload(timeout=timeout, url=url))
