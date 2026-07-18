# Structured Output

Pass `response_format=` to constrain the model's reply to a schema. The same
three input styles work as for tool parameters:

1. A raw **JSON Schema** `dict`.
2. A **Pydantic `BaseModel`** subclass.
3. A `TypeAdapter[...]` wrapping any type.

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

## Caveats

- The SDK does **not** automatically downgrade strict JSON Schema to "loose"
  modes if the (model, provider) pair only supports loose JSON. If you ask
  for a strict schema and the upstream rejects it, you get the rejection
  back as `BadRequestError`.
- The catalog records `supports_structured_output: "strict" | "loose" |
  "none"` per model — useful for pre-flight decisions, but the SDK does not
  consult it on the request path.
