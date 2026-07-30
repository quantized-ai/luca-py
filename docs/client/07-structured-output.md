# Structured Output

Pass `response_format=` to constrain the model's reply to a schema. The same
three input styles work as for tool parameters:

1. A raw **JSON Schema** `dict`.
2. A **Pydantic `BaseModel`** subclass.
3. A `TypeAdapter[...]` wrapping any type.

A `dict` is **always a JSON Schema**, never a provider's wire-level
`response_format` payload — each provider spells that differently and the SDK
projects it for you. To hand-write the wire shape, use `provider_options`.

## Returning a Pydantic instance

```python
from pydantic import BaseModel
from luca.client import completion

class CityFact(BaseModel):
    city: str
    country: str
    population: int

response = completion(
    model="openai:gpt-4o",
    messages=[{"role": "user", "content": "Give me a fact about Tokyo."}],
    response_format=CityFact,
)

fact = response.parse()        # → CityFact instance
print(fact.city, fact.population)
```

`response.parse()` concatenates the text blocks in `response.message.content`,
parses the result as JSON, and validates it against the `response_format`
that was on the originating request. The schema is stored as a private
attribute on the response so you don't pass it twice.

| `response_format=` type | `response.parse()` returns |
|---|---|
| `dict` (raw JSON Schema) | a `dict` (just `json.loads`) |
| `type[BaseModel]` | an instance of that model |
| `TypeAdapter` | the `validate_python` result |

## Error paths

`response.parse()` raises:

- `ValueError` — if `response_format` was not set on the originating request.
- `StructuredOutputError` (a `ClientError`) — if:
  - the text is not valid JSON, or
  - the data doesn't validate against the schema.

`StructuredOutputError.original_exception` carries the underlying
`json.JSONDecodeError` or `pydantic.ValidationError`.

## Streaming + structured output

`FinishEvent` carries the same `parse()` method (the same
`_response_format` is propagated from the request). So after collecting the
final event:

```python
with completion_stream(
    model="openai:gpt-4o",
    messages=[{"role": "user", "content": "Give me a fact about Tokyo."}],
    response_format=CityFact,
) as s:
    for event in s:
        if event.type == "finish":
            fact = event.parse()
```

Or just use `stream.collect()` to skip the loop and get a regular
`ChatCompletionResponse`:

```python
with completion_stream(...) as s:
    response = s.collect()

fact = response.parse()
```

## What each provider gets

| Provider | Wire field | Notes |
|---|---|---|
| `openai` (Responses) | `text.format` | flat `{"type": "json_schema", "name", "schema", "strict": true}` |
| OpenAI-compatible hosts (chat completions: `groq`, `deepseek`, `ollama`, `openrouter`) | `response_format.json_schema` | sent as-is; a host that doesn't support it answers with its own `BadRequestError` |
| `anthropic` | `output_config.format` | merged into `output_config`, so it coexists with an adaptive-thinking `effort`. Refused on models known to predate structured outputs (pre-4.5); an unrecognized model id is sent through rather than blocked |
| `bedrock` | — | `UnsupportedParameterError`. Converse has no structured-output field, and accepting the parameter while ignoring it would hand you prose and an unexplained `StructuredOutputError` |

## Strict mode rewrites your schema

Providers that enforce a schema require every object to set
`additionalProperties: false` and to list **every** property in `required`.
`model_json_schema()` produces neither, so the SDK rewrites the schema before
sending it (`strictify_json_schema`, recursive through `$defs`, arrays and
unions).

The visible consequence: **a field with a default becomes required**, so the
model must emit it. That is the price of strict mode working at all — the
alternative is a `strict: true` request the provider rejects on every call. If
you need the untouched schema, pass the provider's raw payload through
`provider_options`.

## Caveats

- The SDK does **not** downgrade a strict schema to a "loose" JSON mode when
  the (model, provider) pair only supports loose JSON. The upstream rejection
  comes back as `BadRequestError`.
- The catalog records `supports_structured_output: "strict" | "loose" |
  "none"` per model — useful for pre-flight decisions, but the SDK does not
  consult it on the request path. Anthropic's own capability table is the one
  exception, and it only refuses models known to be too old.
