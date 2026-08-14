"""Against a real Ollama daemon. `uv run py.test -m live`.

Unlike the Bedrock live tests these are free and need no credentials, so
there is no excuse for not running them. Skipped when nothing answers on the
port.

The first test is the reason this transport exists: everything else in the
suite would pass just as happily against a wire that silently ignores the
window it was asked for.
"""

import json

import httpx
import pytest

from luca.client import catalog, completion, completion_stream
from luca.client.transports.ollama import discover
from luca.client.types import Tool, ToolMessage, UserMessage

BASE_URL = "http://localhost:11434"
MODEL = "llama3.2:latest"

TOOL = Tool(
    name="get_weather",
    description="Weather for a city",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
)


def _daemon_is_up() -> bool:
    try:
        return httpx.get(f"{BASE_URL}/api/version", timeout=2.0).status_code == 200
    except httpx.HTTPError:
        return False


pytestmark = [pytest.mark.live, pytest.mark.skipif(not _daemon_is_up(), reason="no local Ollama daemon")]


def _loaded_window(model: str) -> int | None:
    running = httpx.get(f"{BASE_URL}/api/ps", timeout=5.0).json().get("models") or []
    return next((m.get("context_length") for m in running if m["name"] == model), None)


@pytest.fixture
def registered():
    """Discovery, exactly as the TUI runs it at boot."""
    for info in discover(BASE_URL):
        catalog.register(provider="ollama", model=info.model, info=info)
    return catalog.get("ollama", MODEL)


def test_the_window_asked_for_is_the_window_ollama_runs(registered):
    # THE regression test. Ollama's default is 4096 and the OpenAI-compatible
    # endpoint silently ignores any attempt to change it, so a passing
    # hermetic suite proves nothing about this.
    completion(
        model=f"ollama:{MODEL}",
        messages=[UserMessage(content="Say OK.")],
        max_tokens=5,
        provider_options={"ollama": {"options": {"num_ctx": 16_384}}},
    )

    assert _loaded_window(MODEL) == 16_384


def test_the_registered_window_is_what_gets_requested(registered):
    completion(model=f"ollama:{MODEL}", messages=[UserMessage(content="Say OK.")], max_tokens=5)

    # No override: the number discovery put in the catalog is the number the
    # daemon reports, which is the whole point of the loop.
    assert _loaded_window(MODEL) == registered.context_window


def test_discovery_finds_the_local_models():
    models = discover(BASE_URL)

    assert models, "no chat models pulled; `ollama pull llama3.2` first"
    assert all(m.provider == "ollama" and m.context_window for m in models)
    # An embedding model is not a chat model and must not be offered.
    assert all("embed" not in m.model for m in models)


def test_a_completion_round_trip(registered):
    response = completion(
        model=f"ollama:{MODEL}",
        messages=[UserMessage(content="Reply with the single word OK.")],
        max_tokens=10,
    )

    assert response.message.content[0].text.strip()
    assert response.usage.input_tokens > 0
    # Local inference is free; a cost here would be invented.
    assert response.usage.cost is None


def test_a_streamed_completion_round_trip(registered):
    with completion_stream(
        model=f"ollama:{MODEL}",
        messages=[UserMessage(content="Count to three.")],
        max_tokens=30,
    ) as stream:
        events = list(stream)

    assert type(events[-1]).__name__ == "FinishEvent"
    assert events[-1].message.content[0].text.strip()


def test_a_tool_call_round_trip(registered):
    first = completion(
        model=f"ollama:{MODEL}",
        messages=[UserMessage(content="What is the weather in Paris?")],
        tools=[TOOL],
    )
    call = next(b for b in first.message.content if getattr(b, "type", "") == "tool_call")

    # Ollama reports done_reason "stop" even when it emitted a call, so the
    # canonical reason has to be derived from the content.
    assert (call.name, call.arguments, first.message.finish_reason) == (
        "get_weather",
        {"city": "Paris"},
        "tool_use",
    )

    second = completion(
        model=f"ollama:{MODEL}",
        messages=[
            UserMessage(content="What is the weather in Paris?"),
            first.message,
            ToolMessage(tool_call_id=call.id, name=call.name, content="18C and sunny"),
        ],
        tools=[TOOL],
        max_tokens=60,
    )

    # The result reached the model: correlation by `tool_name` works.
    assert "18" in second.message.content[0].text


def test_a_model_that_was_never_pulled_says_how_to_pull_it():
    from luca.client.exceptions import ModelNotFoundError

    with pytest.raises(ModelNotFoundError, match="ollama pull"):
        completion(model="ollama:no-such-model:v9", messages=[UserMessage(content="Hi")])


def test_a_prompt_over_the_window_is_not_silently_dropped(registered):
    # Ollama truncates past num_ctx with no error and no flag. This asserts
    # the shape of that so a future change to the window cannot go unnoticed.
    filler = "word " * 4000
    response = completion(
        model=f"ollama:{MODEL}",
        messages=[UserMessage(content=filler + "Say OK.")],
        max_tokens=5,
        provider_options={"ollama": {"options": {"num_ctx": 2048}}},
    )

    raw = json.loads(json.dumps(response.raw))
    assert raw["prompt_eval_count"] <= 2048
