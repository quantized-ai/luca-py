"""`/api/show` → `ModelInfo`, and the daemon walk that produces them.

The payloads below are captured verbatim from a live Ollama (v0.32.6), one
per interesting shape: a model whose window needs capping, one already under
the cap, an embedding model that is not a chat model at all, and one whose
`model_info` is entirely empty.
"""

import httpx
import pytest

from luca.client.exceptions import ConnectionError as ClientConnectionError
from luca.client.transports.ollama.discovery import (
    UNKNOWN_CONTEXT_WINDOW,
    discover,
    model_info_from_show,
)
from luca.client.types import ModelInfo

from ..._helpers.httpx_mocks import make_sync_client

# ── captured payloads ────────────────────────────────────────────────────────

LLAMA = {
    "capabilities": ["completion", "tools"],
    "details": {"family": "llama", "parameter_size": "3.2B"},
    "model_info": {"general.parameter_count": 3212749888, "llama.context_length": 131072},
}

QWEN = {
    "capabilities": ["completion", "tools"],
    "details": {"family": "qwen2", "parameter_size": "14.8B"},
    "model_info": {"general.parameter_count": 14770033664, "qwen2.context_length": 32768},
}

EMBEDDING = {
    "capabilities": ["embedding"],
    "details": {"family": "nomic-bert", "parameter_size": "137M"},
    "model_info": {"nomic-bert.context_length": 2048},
}

# A Modelfile-built model: advertises thinking, reports no architecture at all.
BARE = {
    "capabilities": ["completion", "tools", "thinking"],
    "details": {"family": "", "families": None, "parameter_size": ""},
    "model_info": {},
}


# ── the mapper ───────────────────────────────────────────────────────────────


def test_a_window_larger_than_the_ceiling_is_capped():
    # 131072 would either fail to load or spill to CPU on an ordinary machine.
    assert model_info_from_show("llama3.2:latest", LLAMA, ceiling=32_768) == ModelInfo(
        model="llama3.2:latest",
        provider="ollama",
        display_name="llama3.2:latest",
        family="llama",
        context_window=32_768,
        supports_tools=True,
        supports_reasoning=False,
        supports_image_input=False,
        cost=None,
    )


def test_a_window_under_the_ceiling_is_taken_as_is():
    assert model_info_from_show("qwen2.5:14b", QWEN, ceiling=32_768) == ModelInfo(
        model="qwen2.5:14b",
        provider="ollama",
        display_name="qwen2.5:14b",
        family="qwen2",
        context_window=32_768,
        supports_tools=True,
        supports_reasoning=False,
        supports_image_input=False,
        cost=None,
    )


def test_the_thinking_capability_becomes_supports_reasoning():
    info = model_info_from_show("bjoernb/claude-opus-4-5:latest", BARE)

    assert (info.supports_reasoning, info.supports_tools) == (True, True)


def test_a_model_with_no_architecture_gets_the_conservative_window():
    # `model_info` is empty, so there is nothing to read. luca SETS num_ctx,
    # so whatever it picks here becomes true.
    info = model_info_from_show("bjoernb/claude-opus-4-5:latest", BARE)

    assert (info.context_window, info.family) == (UNKNOWN_CONTEXT_WINDOW, None)


def test_an_embedding_model_is_not_a_chat_model():
    # It carries a plausible context_length, so skipping has to key on the
    # capability rather than on a missing field.
    assert model_info_from_show("nomic-embed-text:latest", EMBEDDING) is None


def test_no_cost_is_attached():
    # Local inference is free; a zeroed ModelCost would render "$0.00" where
    # the usage screen's "—" is the honest answer.
    assert model_info_from_show("llama3.2:latest", LLAMA).cost is None


# ── the daemon walk ──────────────────────────────────────────────────────────

TAGS = {
    "models": [
        {"model": "nomic-embed-text:latest", "name": "nomic-embed-text:latest"},
        {"model": "llama3.2:latest", "name": "llama3.2:latest"},
        {"model": "qwen2.5:14b", "name": "qwen2.5:14b"},
    ]
}

SHOW_BY_MODEL = {
    "nomic-embed-text:latest": EMBEDDING,
    "llama3.2:latest": LLAMA,
    "qwen2.5:14b": QWEN,
}


def _daemon(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/api/tags":
        return httpx.Response(200, json=TAGS)
    import json as _json

    model = _json.loads(request.content)["model"]
    return httpx.Response(200, json=SHOW_BY_MODEL[model])


def test_discover_returns_only_the_chat_models():
    models = discover("http://localhost:11434", client=make_sync_client(_daemon))

    assert [m.model for m in models] == ["llama3.2:latest", "qwen2.5:14b"]


def test_a_daemon_that_is_not_running_says_so_and_says_what_to_do():
    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("[Errno 61] Connection refused")

    with pytest.raises(ClientConnectionError, match="Is the daemon running"):
        discover("http://localhost:11434", client=make_sync_client(refused))
