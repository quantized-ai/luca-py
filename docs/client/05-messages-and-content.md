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
| `ThinkingBlock(text, id=None, signature=None, redacted=False)` | `"thinking"` | Assistant |
| `ToolCall(id, name, arguments, partial_arguments, complete, thought_signature=None)` | `"tool_call"` | Assistant |
| `ToolResultBlock(tool_call_id, content, is_error=False)` | `"tool_result"` | (Anthropic-style inline; prefer `ToolMessage`) |
| `RefusalBlock(text)` | `"refusal"` | Assistant |

### Reasoning is provider-owned

`ThinkingBlock.id` and `ThinkingBlock.signature` are opaque: the id is how the
provider names the reasoning item (`rs_…` on OpenAI), the signature is its
attestation over the content (Anthropic's `signature`, OpenAI's
`encrypted_content`). Both round-trip verbatim or not at all — editing `text`
invalidates the signature.

They are also **pair-scoped**. A signature minted by one (provider, model) is
rejected by every other, so a transport replays a thinking block only when the
assistant message carrying it names the provider and model being called:

```python
AssistantMessage(
    content=[ThinkingBlock(text="…", id="rs_1", signature="enc-1")],
    provider="openai",         # ← the pair that minted it
    model="gpt-5.4",
)
```

Send that to `anthropic:claude-sonnet-5` and the reasoning is dropped from the
wire (the text stays in your message object). A message with no `provider` /
`model` is taken as caller-driven and replayed as-is. This is what keeps a
conversation valid across a mid-session model switch.

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

Which sources a given transport takes differs per block, because the wires
differ. Audio is the narrow one: chat completions is the only API with an
audio input part, and it reads inline bytes only.

| Block | Anthropic | OpenAI chat | OpenAI Responses | Bedrock |
|---|---|---|---|---|
| `ImageBlock` | all three | url, base64 | all three | base64 |
| `AudioBlock` | raises | base64 | raises | raises |
| `FileBlock` | all three | base64, file id | all three | base64 |

An `AudioBlock`'s `media_type` picks the wire's `format` token, so it must be
one `OpenAITransport.AUDIO_FORMATS` maps — subclass and widen that dict for a
host that reads more. Anything else raises rather than guessing a format.

> ⚠️ **`openai:gpt-audio` raises out of the box.** `OpenAIProvider` defaults to
> the Responses transport, which has no audio part. Ask for the other wire to
> send audio to OpenAI directly:
>
> ```python
> from luca.client.providers.openai import OpenAIProvider
> from luca.client.transports import OpenAITransport
>
> provider = OpenAIProvider(transport_class=OpenAITransport)
> ```
>
> OpenRouter has no such split — it is a chat-completions host already, so
> `openrouter:openai/gpt-audio` and the Gemini models work as they are.

To ask this before building a request, read `SUPPORTS_AUDIO_INPUT` off the
transport a host routes to. A model's audio capability does not imply its
host can carry audio, and the model catalog cannot see the wire:

```python
from luca.client.providers import default_transport_class

default_transport_class("openrouter").SUPPORTS_AUDIO_INPUT   # True
default_transport_class("openai").SUPPORTS_AUDIO_INPUT       # False — Responses
default_transport_class("bedrock").SUPPORTS_AUDIO_INPUT      # False — Converse
```

### Tool calls — one class, two views

`ToolCall` lives both inside `AssistantMessage.content` **and** surfaces via
`message.tool_calls` / `stream.tool_calls`. These
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

for block in response.messages[-1].content:
    if isinstance(block, TextBlock):
        print(block.text)
```
