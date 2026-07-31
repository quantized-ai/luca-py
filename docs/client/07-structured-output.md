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
| `anthropic` | `output_config.format` | merged into `output_config`, so it coexists with an adaptive-thinking `effort` and with a hand-written `provider_options` entry. Refused only on models known to predate structured outputs (Claude 3.x); an unrecognized id is sent through rather than blocked |
| `bedrock` | — | `UnsupportedParameterError`. Converse has no structured-output field, and accepting the parameter while ignoring it would hand you prose and an unexplained `StructuredOutputError` |

## Per-provider schema limits

The rewrite makes a schema *sendable*, not universal. Two providers, two
grammars:

- **Anthropic** rejects `minimum`, `maximum`, `multipleOf`, `minLength`,
  `maxLength`, `pattern`, `maxItems`, `uniqueItems`, a `minItems` other than
  0 or 1, and any `format` outside its own list. `Field(ge=…)` and
  `Field(min_length=…)` produce exactly these, so the SDK strips them and
  appends each to the field's `description` — the same thing Anthropic's own
  SDKs do. The model still sees the intent, and `parse()` validates the reply
  against your original schema, constraints included. Recursive models are
  refused outright by their grammar and cannot be rescued this way.
- **OpenAI** requires the schema root to be `type: "object"`, so a
  `TypeAdapter(list[int])` is rejected there while Anthropic accepts it. It
  also rejects `allOf`, and open objects (`dict[str, X]`, which produces an
  `additionalProperties` schema) are rejected by both.

## Strict mode rewrites your schema

Providers that enforce a schema require every object to set
`additionalProperties: false` and to list **every** property in `required`.
`model_json_schema()` produces neither, so the SDK rewrites the schema before
sending it (`strictify_json_schema`, recursive through `$defs`, arrays and
unions). A `$ref` carrying sibling keys is inlined at the same time —
`Field(description=…)` on a nested model produces one, and OpenAI rejects
`$ref` beside any other keyword.

The schema's wire *name* is sanitized too. A Pydantic generic keeps its
parameters in `__name__` (`Box[int]`), which fails the provider's
`^[a-zA-Z0-9_-]+$` / 64-character rule; illegal characters become `_`.

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
