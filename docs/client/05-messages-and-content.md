# Messages and Content Blocks

The SDK's conversation model is **Pi-aligned, not OpenAI-aligned**: every
turn carries a list of typed `ContentBlock` instances. The OpenAI-style flat
`{role, content, tool_calls}` shape is the wire format for OpenAI
specifically — the transport projects to/from it on the way in and out.

## Roles

There are **three** message roles. There is intentionally no
`SystemMessage` class and no `"system"` role.

```python
class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str | list[TextBlock | ImageBlock | AudioBlock | FileBlock]
    name: str | None = None
    timestamp: int | None = None

class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: list[TextBlock | ThinkingBlock | ToolCall | RefusalBlock]

    finish_reason: str | None = None
    provider_finish_reason: str | None = None
    cancelled: bool = False
    error_message: str | None = None
    provider: str | None = None
    model: str | None = None
    response_model: str | None = None
    response_id: str | None = None
    usage: Usage | None = None
    timestamp: int | None = None

    @property
    def tool_calls(self) -> list[ToolCall]: ...  # filter view, same instances

class ToolMessage(BaseModel):
    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: str | list[TextBlock | ImageBlock]
    name: str | None = None
    is_error: bool = False
    timestamp: int | None = None
```

All three are `extra="forbid"`.

`AssistantMessage` is **self-describing**: it carries its own finish state,
provider, model, usage, and timestamp. A serialized conversation reloads
with full context.

The `Message` annotated union (`UserMessage | AssistantMessage |
ToolMessage`) is discriminated on `role`.

## Content blocks

`ContentBlock` is a discriminated union on `type`. Every block has
`extra="forbid"`.

| Block | `type` | Used in |
|---|---|---|
| `TextBlock(text, signature=None)` | `"text"` | User, assistant, tool |
| `ImageBlock(source)` | `"image"` | User, tool |
| `AudioBlock(source)` | `"audio"` | User |
| `FileBlock(source, name=None)` | `"file"` | User |
| `ThinkingBlock(text, signature=None, redacted=False)` | `"thinking"` | Assistant |
| `ToolCall(id, name, arguments, partial_arguments, complete, thought_signature=None)` | `"tool_call"` | Assistant |
| `ToolResultBlock(tool_call_id, content, is_error=False)` | `"tool_result"` | (Anthropic-style inline; prefer `ToolMessage`) |
| `RefusalBlock(text)` | `"refusal"` | Assistant |

### Media sources

`ImageBlock` / `AudioBlock` / `FileBlock` carry a `source` field that is
itself a discriminated union on `kind`:

| Source | `kind` | Fields |
|---|---|---|
| `MediaURL` | `"url"` | `url`, `media_type=None` |
| `MediaBase64` | `"base64"` | `data`, `media_type` (required) |
| `MediaFileId` | `"file"` | `file_id`, `media_type=None` |

```python
from luca.client.types import (
    UserMessage, TextBlock, ImageBlock, MediaURL,
)

UserMessage(content=[
    TextBlock(text="What's in this image?"),
    ImageBlock(source=MediaURL(url="https://example.com/cat.png")),
])
```

### Tool calls — one class, two views

`ToolCall` lives both inside `AssistantMessage.content` **and** surfaces via
`message.tool_calls` / `response.tool_calls` / `stream.tool_calls`. These
are the **same** instances, filtered out of `content` — never copied.
Mutating one view mutates the other.

```python
class ToolCall(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: dict             # parsed; {} while streaming, populated at end
    partial_arguments: str = "" # raw JSON fragments during streaming
    complete: bool = True       # False while args still streaming
    thought_signature: str | None = None

    def parse_arguments(self, schema) -> Any:
        """Validate self.arguments against a Pydantic model or TypeAdapter."""
```

For non-streamed responses `complete=True`, `arguments` is parsed,
`partial_arguments=""`. During streaming the buffer accumulates and resolves
at `tool_call_end`.

## Coercion

The helpers accept dict-shape messages and coerce them on the way in:

```python
completion(
    model="openai:gpt-4o",
    messages=[
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": [{"type": "text", "text": "Hi!"}]},
        {"role": "tool", "tool_call_id": "tc_1", "content": "42"},
    ],
)
```

A dict with `role="system"` raises `BadRequestError` with a hint to move it
to `system_message=`. Unknown roles raise `BadRequestError`.

## System prompts (request-scoped)

`ChatCompletionRequest.system_message` is `str | list[TextBlock] | None` and
is **request-scoped** — it never enters `messages`. Each transport projects
it into the host's expected shape:

- OpenAI / OpenAI-compatible — prepends a wire-level
  `{role: "system", content: ...}` entry to the wire `messages`.
- Anthropic — populates the top-level `system` field.
- (Future) Gemini / Vertex — populates `systemInstruction`. Bedrock — feeds
  the Converse API's `system` shape.

None of the wire shape leaks into the SDK's `messages` list.

## Putting it together

```python
from luca.client import completion
from luca.client.types import (
    UserMessage, AssistantMessage, ToolMessage,
    TextBlock, ImageBlock, MediaURL,
)

response = completion(
    model="anthropic:claude-3-5-sonnet-latest",
    messages=[
        UserMessage(content=[
            TextBlock(text="Caption this image in one sentence."),
            ImageBlock(source=MediaURL(url="https://example.com/cat.png")),
        ]),
    ],
    system_message="You are concise.",
)

for block in response.message.content:
    if isinstance(block, TextBlock):
        print(block.text)
```
