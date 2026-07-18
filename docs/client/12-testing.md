# Testing

The SDK ships with a fully scripted "faux" provider/transport pair so you
can write tests of your own apps without hitting any real LLM. The same
machinery is what the SDK's own test suite uses.

## What's exported

```python
from luca.client.testing import (
    FauxProvider,
    FauxTransport,
    faux_assistant_message,
    faux_text,
    faux_thinking,
    faux_tool_call,
    faux_refusal,
    faux_error,
)
```

The builders return small dataclasses describing **intended** scripted
output. `FauxTransport` plays them back as proper
`ChatCompletionResponse` / stream events that exercise the full canonical
pipeline (the same accumulator, the same finish-reason classification, the
same `tool_calls` filter view).

## Quick example — non-streaming

```python
from luca.client.testing import (
    FauxProvider, faux_assistant_message, faux_text, faux_tool_call,
)
from luca.client.types import ChatCompletionRequest, UserMessage

prov = FauxProvider()
prov.set_responses([
    faux_assistant_message(
        blocks=[
            faux_text("I'll add those for you."),
            faux_tool_call(name="add", arguments={"a": 1, "b": 2}, id="tc_1"),
        ],
        finish_reason="tool_use",
    ),
    faux_assistant_message(
        blocks=[faux_text("The answer is 3.")],
        finish_reason="stop",
    ),
])

request = ChatCompletionRequest(
    model="faux",
    messages=[UserMessage(content="What is 1 + 2?")],
)
response = prov.completion(request)
assert response.finish_reason == "tool_use"
assert response.tool_calls[0].arguments == {"a": 1, "b": 2}

# Second call pops the next scripted response.
response2 = prov.completion(request)
assert response2.finish_reason == "stop"
```

## Streaming

The same builders work for streaming — `FauxTransport.completion_stream`
returns a `FauxChatCompletionStream` that yields the proper
`Raw…` → public-event sequence.

```python
from luca.client.testing import (
    FauxProvider, faux_assistant_message, faux_text,
)

prov = FauxProvider()
prov.set_responses([
    faux_assistant_message(
        blocks=[faux_text("Hello, ")],
        finish_reason="stop",
    ),
])

request = ChatCompletionRequest(model="faux", messages=[UserMessage(content="hi")])

with prov.completion_stream(request) as s:
    for event in s:
        print(event.type, getattr(event, "delta", ""))
```

## Errors

`faux_error(...)` lets you script transport-level failures:

```python
from luca.client.exceptions import RateLimitError

prov.set_responses([
    faux_assistant_message(
        blocks=[],
        finish_reason="error",
        error=faux_error("rate limited", error_class=RateLimitError),
    ),
])

# completion() raises RateLimitError
# stream surfaces a terminal ErrorEvent
```

To simulate an **LLM-side refusal** (which is *not* an exception in this
SDK) use `faux_refusal(...)`:

```python
faux_assistant_message(
    blocks=[faux_refusal("I can't help with that.")],
    finish_reason="error",      # canonical "error" with RefusalBlock present
)
```

## Token pacing

`FauxProvider(tokens_per_second=...)` is available for tests that want
realistic streaming throughput. V1 emits one delta per block; pacing only
matters in tests that read the throughput value directly.

## Using the faux in production-like setups

If a test exercises code that calls `completion(model="openai:gpt-4o", ...)`
through the helper, you can either:

1. Register a one-off provider: `register_provider("openai",
   FauxProvider)` — but this is global, so it leaks across tests unless
   you undo it in teardown.
2. Pass `provider=` directly: `completion(model="...", provider=faux_inst)`.
   No registry mutation, no leakage. Preferred.

## The SDK's own test suite

`tests/` mirrors the package layout — `tests/client/test_types/`,
`tests/client/test_providers/`, etc. Run with:

```bash
uv run py.test tests/
```

`pyproject.toml` configures `pytest` to fail on any warnings (including
`ResourceWarning`), so unclosed streams or connections show up as test
failures. The full test design lives in
[`testing_architecture.md`](../../testing_architecture.md).
