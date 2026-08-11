"""What models a local Ollama has, and what they can do.

models.dev cannot know what you pulled, so `catalog.get("ollama", …)` is
always `None` and every consumer falls back to a default window that is wrong
by an order of magnitude. Ollama knows the real answer and will tell you:
`/api/tags` lists what is installed, `/api/show` gives per-model capabilities
and the architectural context length.

This module returns `ModelInfo` and registers nothing. Transports do not
import `luca.client.catalog` — the caller does the registering, which keeps
the layering arrows pointing inward.
"""

from __future__ import annotations

import httpx

from ...exceptions import ConnectionError as ClientConnectionError, ProviderAPIError
from ...types.catalog import ModelInfo

DEFAULT_NUM_CTX_CEILING = 32_768
"""How large a window to ask for when the model would allow more.

`llama3.2` advertises 131072; allocating that on a laptop either fails to
load or spills to CPU and crawls."""

UNKNOWN_CONTEXT_WINDOW = 8_192
"""When `/api/show` reports no context length at all — which happens for
models built from a bare Modelfile. Conservative on purpose: luca SETS the
window, so whatever it picks is true."""

CHAT_CAPABILITY = "completion"


def _architectural_context_length(payload: dict) -> int | None:
    """The `<family>.context_length` key, whatever the family is called.

    The prefix is the architecture (`llama.`, `qwen2.`, `nomic-bert.`), so the
    key cannot be looked up by name."""
    for key, value in (payload.get("model_info") or {}).items():
        if key.endswith(".context_length") and isinstance(value, int):
            return value
    return None


def model_info_from_show(
    name: str,
    payload: dict,
    *,
    ceiling: int = DEFAULT_NUM_CTX_CEILING,
) -> ModelInfo | None:
    """One `/api/show` response → a catalog record, or None if it is not a
    chat model.

    Pure. `None` for anything without the `completion` capability: an
    embedding model carries a plausible-looking context length and would
    otherwise sit in the model picker."""
    capabilities = payload.get("capabilities") or []
    if CHAT_CAPABILITY not in capabilities:
        return None

    architectural = _architectural_context_length(payload)
    window = min(architectural, ceiling) if architectural else UNKNOWN_CONTEXT_WINDOW

    return ModelInfo(
        model=name,
        provider="ollama",
        display_name=name,
        family=(payload.get("details") or {}).get("family") or None,
        context_window=window,
        supports_tools="tools" in capabilities,
        supports_reasoning="thinking" in capabilities,
        supports_image_input="vision" in capabilities,
        # No cost: local inference is free, and a zeroed ModelCost would put a
        # "$0.00" in the usage screen where "—" is the honest answer.
        cost=None,
    )


def discover(
    base_url: str,
    *,
    ceiling: int = DEFAULT_NUM_CTX_CEILING,
    timeout: float = 10.0,
    client: httpx.Client | None = None,
) -> list[ModelInfo]:
    """Every chat model the daemon at `base_url` has, newest first.

    Raises `ConnectionError` when the daemon is not reachable; whether that is
    fatal is the caller's decision."""
    owned = client is None
    http = client or httpx.Client(timeout=timeout)
    try:
        tags = _get_json(http, f"{base_url.rstrip('/')}/api/tags", base_url)
        records = []
        for entry in tags.get("models") or []:
            name = entry.get("model") or entry.get("name")
            if not name:
                continue
            show = _post_json(http, f"{base_url.rstrip('/')}/api/show", {"model": name}, base_url)
            info = model_info_from_show(name, show, ceiling=ceiling)
            if info is not None:
                records.append(info)
        return records
    finally:
        if owned:
            http.close()


def _get_json(client: httpx.Client, url: str, base_url: str) -> dict:
    try:
        response = client.get(url)
        response.raise_for_status()
        return response.json()
    except httpx.NetworkError as exc:
        raise _not_running(base_url, exc) from exc
    except httpx.HTTPError as exc:
        raise ProviderAPIError(f"Ollama: {url} failed ({exc})", provider="ollama") from exc


def _post_json(client: httpx.Client, url: str, payload: dict, base_url: str) -> dict:
    try:
        response = client.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except httpx.NetworkError as exc:
        raise _not_running(base_url, exc) from exc
    except httpx.HTTPError as exc:
        raise ProviderAPIError(f"Ollama: {url} failed ({exc})", provider="ollama") from exc


def _not_running(base_url: str, exc: Exception) -> ClientConnectionError:
    return ClientConnectionError(
        f"Cannot reach Ollama at {base_url} ({exc}). Is the daemon running? Start it with `ollama serve`.",
        provider="ollama",
        original_exception=exc,
    )
