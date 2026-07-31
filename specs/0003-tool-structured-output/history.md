# 2026-07-31

The user's initial intent was: add `output_schema` to `ToolSpec` and
`structured_content` to `ExecutionResult` so a tool can "advertise that it can
produce structured output", with a Pydantic-derived helper on
`contrib.tools.Tool` mirroring `Args`. The user's framing was that
`luca.client` already implements structured output and the open question was
whether it was "compatible and ready".

By inspecting the codebase two premises were corrected:

1. **The client's structured output is an unrelated mechanism.**
   `ChatCompletionRequest.response_format` constrains the ASSISTANT's reply text
   to a JSON Schema. A tool declaring its result shape shares nothing with it.
   There was no compatibility question to answer.
2. **No provider luca talks to accepts a tool output schema.** All four
   transports project a tool as name + description + input schema only —
   OpenAI (chat and Responses), Anthropic and Bedrock Converse have no such
   field. So `output_schema` cannot be advertised to the MODEL; it is advertised
   to the APPLICATION. `luca.client` needs zero changes and
   `adapter.tool_spec_to_luca_tool` stays as it is.

Four decisions were then settled with the user:

- `structured_content` never reaches the wire; `content` stays the sole
  model-facing channel (so the projector and `calculate_context` are untouched).
- `structured_content` and `ExecutionResult.metadata` coexist; the shell tools
  migrate nothing.
- `Tool.output_schema` is a declared `ClassVar` defaulting to `None`, not a
  `hasattr` probe; tools populate the payload by overriding `execute()` — no new
  override point on the base class.
- `tool()` / `tool_class()` gain `output=`, mirroring `arguments=`.

The ClassVar was then renamed `OutputSchema` → `output_schema` after working
through the example: the preferred declaration is binding a module-level model
(`output_schema = WeatherReport`), where CapWords reads as a class alias rather
than an attribute binding. `Args` keeps CapWords because it is almost always
written as a nested `class Args(BaseModel)`. The cost — `Tool.output_schema` is
a model class while `ToolSpec.output_schema` is a JSON Schema dict, unlike the
deliberately-distinct `Args → input_schema` — was accepted and recorded in the
PRD so it is not "fixed" back later.

The PRD's SCOPE did not change — it is still two optional fields plus contrib
ergonomics. What changed is its FRAMING and its stated consequences: the feature
is a data-model + contrib change with no wire, runner, projector, event or
client impact, and V1 deliberately ships two fields that nothing in the repo
reads yet (the landing place for a future MCP-backed registry). The spec-hash
consequence, the two mechanical test fixtures, and the
`process_tool_output`-rebuild footgun were added as recorded consequences.
