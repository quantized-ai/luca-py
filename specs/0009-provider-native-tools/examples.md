# 0009 — implementation pseudocode + usage

Part 1 is the implementation: every new class, every new method, every
transport interaction. Part 2 is what the user of the client writes. Wire
shapes in comments are the ones observed live on 2026-08-06.

---

## Part 1 — implementation

### `luca/client/types/tools.py`

```python
class BaseTool(BaseModel):
    """Everything accepted through `tools=`."""

    model_config = ConfigDict(extra="forbid")

    def get_projector(self) -> ToolProjector | None:
        return None                     # None -> transport's default projector


class Tool(BaseTool):                   # unchanged shape
    name: str
    description: str
    parameters: Any

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class ToolProjector:
    """All four wire touchpoints of one tool kind. Concrete class,
    NotImplementedError hooks — house style, no ABC."""

    def project_tool_to_llm(self, tool: BaseTool) -> dict:
        raise NotImplementedError       # declaration -> `tools` entry

    def build_tool_call(self, item: dict) -> ToolCall:
        raise NotImplementedError       # wire call item -> canonical ToolCall

    def project_tool_call_to_llm(self, tc: ToolCall) -> dict:
        raise NotImplementedError       # ToolCall -> wire item (replay in input)

    def project_tool_message_to_llm(self, msg: ToolMessage, call: ToolCall | None) -> dict:
        raise NotImplementedError       # result -> wire item. `call` is the
                                        # originating ToolCall when the transport
                                        # found it in history (shell needs its
                                        # action.max_output_length), else None.

    def project_tool_choice_to_llm(self, tool: BaseTool) -> dict:
        raise NotImplementedError       # the shape that FORCES this one tool
                                        # (tool_choice={"name": ...} resolution)
```

### `luca/client/types/content.py`

```python
NATIVE_TOOL_CALL_TYPES: dict[str, type[ToolCall]] = {}


class ToolCall(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: dict = Field(default_factory=dict)
    partial_arguments: str = ""
    complete: bool = True
    thought_signature: str | None = None

    # Bound by native subclasses to their ToolProjector subclass. None on the
    # base and on any non-native subclass -> transport default projector.
    # ClassVar[Any], NOT ClassVar[type | None]: the sibling `type` FIELD
    # shadows the builtin in the class namespace during annotation evaluation
    # -> TypeError at class construction (spiked, pydantic 2.13.4 / py3.14).
    projector_class: ClassVar[Any] = None

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs):
        super().__pydantic_init_subclass__(**kwargs)
        wire_type = cls.model_fields["type"].default
        if wire_type == "tool_call":
            return                                  # not a native subclass
        existing = NATIVE_TOOL_CALL_TYPES.get(wire_type)
        if existing is not None and existing is not cls:
            raise TypeError(f"{wire_type!r} already registered by {existing.__name__}")
        NATIVE_TOOL_CALL_TYPES[wire_type] = cls

    @model_validator(mode="wrap")
    @classmethod
    def _dispatch_native(cls, value, handler):
        # Deserialization: validating a native payload against the BASE class
        # dispatches to the registered subclass, so AssistantMessage.content
        # and dict-shaped messages= revalidate conversations that contain
        # native calls. Requires the subclass's module to be imported —
        # importing luca.client imports all first-party ones.
        if cls is ToolCall and isinstance(value, dict):
            target = NATIVE_TOOL_CALL_TYPES.get(value.get("type", "tool_call"))
            if target is not None:
                return target.model_validate(value)
        return handler(value)
```

### `luca/client/types/messages.py`

```python
NATIVE_TOOL_MESSAGE_TYPES: dict[str, type[ToolMessage]] = {}


class ToolMessage(BaseModel):
    # NO new field on the base: payloads without "type" validate as base,
    # existing serialized data untouched. Native subclasses declare
    # `type: Literal["..."]` and register.
    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: str | list[TextBlock | ImageBlock]
    name: str | None = None
    is_error: bool = False
    timestamp: int | None = None

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs):
        super().__pydantic_init_subclass__(**kwargs)
        field = cls.model_fields.get("type")
        if field is None:
            return
        NATIVE_TOOL_MESSAGE_TYPES[field.default] = cls    # same duplicate guard as ToolCall

    @model_validator(mode="wrap")
    @classmethod
    def _dispatch_native(cls, value, handler):
        if cls is ToolMessage and isinstance(value, dict) and "type" in value:
            target = NATIVE_TOOL_MESSAGE_TYPES.get(value["type"])
            if target is not None:
                return target.model_validate(value)
        return handler(value)                             # unknown "type" -> extra="forbid" error


class AssistantMessage(BaseModel):
    # SerializeAsAny: model_dump() must keep subclass fields (item_id, status,
    # …). Without it pydantic dumps per the declared base and DROPS them —
    # and warns, which fails the suite's filterwarnings=error.
    content: list[TextBlock | ThinkingBlock | SerializeAsAny[ToolCall] | RefusalBlock] = ...
    # everything else unchanged. ToolMessage needs no container change:
    # instances are dumped directly, and a direct dump uses the real class.
```

### `luca/client/_client.py`

```python
def _coerce_tools(tools: list | None) -> list[BaseTool] | None:
    if tools is None:
        return None
    out: list[BaseTool] = []
    for t in tools:
        if isinstance(t, BaseTool):                  # Tool AND native tools
            out.append(t)
        elif isinstance(t, dict):                    # dicts stay standard-only
            try:
                out.append(Tool.model_validate(t))
            except ValidationError as e:
                raise BadRequestError(
                    f"tool dict is not a standard function tool: {e}. "
                    "Native tools are passed as instances (e.g. ApplyPatchTool())."
                ) from e
        else:
            raise BadRequestError(f"tool entry is {type(t).__name__}; expected dict or BaseTool.")
    return out

# ChatCompletionRequest.tools: list[BaseTool] | None
# (verified: pydantic passes subclass instances through the DTO untouched)
```

### `luca/client/transports/openai_responses/transport.py`

The default projector is today's standard logic, moved verbatim out of the
transport methods:

```python
class OpenAIResponsesToolProjector(ToolProjector):

    def project_tool_to_llm(self, tool: Tool) -> dict:
        return {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool_parameters_to_json_schema(tool.parameters),
        }

    def build_tool_call(self, item: dict) -> ToolCall:
        return ToolCall(
            id=item["call_id"],
            name=item["name"],
            arguments=_parse_arguments_json(item.get("arguments")),   # ex-transport staticmethod
            complete=True,
        )

    def project_tool_call_to_llm(self, tc: ToolCall) -> dict:
        return {
            "type": "function_call",
            "call_id": tc.id,
            "name": tc.name,
            "arguments": json.dumps(tc.arguments) if tc.arguments else "{}",
        }

    def project_tool_message_to_llm(self, msg: ToolMessage, call: ToolCall | None) -> dict:
        return {
            "type": "function_call_output",
            "call_id": msg.tool_call_id,
            "output": self._output_text(msg),
        }

    def project_tool_choice_to_llm(self, tool: Tool) -> dict:
        return {"type": "function", "name": tool.name}

    def _output_text(self, msg: ToolMessage) -> str:
        # today's body: str passes through; TextBlocks join; images ->
        # BadRequestError ("refusing beats dropping"). Shared with native
        # projectors below.
        ...
```

The transport keeps zero per-tool knowledge:

```python
class OpenAIResponsesTransport(BaseTransport, OpenAIErrorMappingMixin, ChatCompletionTransportMixin):

    TOOL_PROJECTOR_BASE: ClassVar[type] = OpenAIResponsesToolProjector

    def _default_tool_projector(self) -> ToolProjector:
        return OpenAIResponsesToolProjector()

    # -- declaration ------------------------------------------------------

    def _resolve_projector(self, tool: BaseTool) -> ToolProjector:
        projector = tool.get_projector() or self._default_tool_projector()
        if not isinstance(projector, self.TOOL_PROJECTOR_BASE):
            # class-based, not transport_id-based: OpenRouter subclasses
            # OpenAITransport and must inherit its acceptance set.
            raise BadRequestError(
                f"{type(tool).__name__}'s projector targets another transport; "
                f"this request uses {type(self).__name__}.",
                provider=self._provider,
            )
        return projector

    def _project_tools(self, tools: list[BaseTool]) -> list[dict]:
        return [self._resolve_projector(t).project_tool_to_llm(t) for t in tools]

    def _project_tool_choice(self, choice, tools: list[BaseTool] | None) -> Any:
        # Strings ("auto"/"required"/"none") and raw dicts stay transport-level,
        # unchanged. Only the {"name": ...} forcing shape is per-tool.
        if isinstance(choice, dict) and set(choice.keys()) == {"name"}:
            tool = next(
                (t for t in tools or [] if getattr(t, "name", None) == choice["name"]),
                None,   # first match wins; name collisions documented, not policed
            )
            if tool is not None:
                return self._resolve_projector(tool).project_tool_choice_to_llm(tool)
            return {"type": "function", "name": choice["name"]}   # today's blind shape
        return choice

    # -- parse ------------------------------------------------------------

    def _parse_assistant_message(self, data, request) -> AssistantMessage:
        content: list = []
        default = self._default_tool_projector()
        for item in data.get("output") or []:
            item_type = item.get("type")
            if item_type == "reasoning":
                content.append(self._parse_reasoning_item(item))
            elif item_type == "message":
                content.extend(self._parse_message_item(item))
            elif item_type == "function_call":
                content.append(default.build_tool_call(item))
            elif (native := NATIVE_TOOL_CALL_TYPES.get(item_type)) is not None:
                # registry replaces the per-request tools scan: an item type
                # can only appear if its tool was declared, and the registry
                # is filled at import by the same module that defines the tool.
                projector = native.projector_class()
                if isinstance(projector, self.TOOL_PROJECTOR_BASE):
                    # same check as declarations: a foreign family's
                    # registration can never mint calls on this wire.
                    content.append(projector.build_tool_call(item))
            # else: hosted-tool items — ignored, unchanged.
        return AssistantMessage(content=content, ...)

    # -- replay + results: one walk, call lineage carried forward ---------

    def _project_messages(self, messages, request) -> list[dict]:
        out: list[dict] = []
        # call_id -> (projector, call) | None. None = the call was dropped
        # (foreign native call), so its result must be dropped too.
        lineage: dict[str, tuple[ToolProjector, ToolCall] | None] = {}
        for msg in messages:
            if isinstance(msg, UserMessage):
                out.append(self._project_user_message(msg))
            elif isinstance(msg, AssistantMessage):
                out.extend(self._project_assistant_message(msg, request, lineage))
            elif isinstance(msg, ToolMessage):
                item = self._project_tool_message(msg, lineage)
                if item is not None:
                    out.append(item)
            else:
                raise BadRequestError(...)
        return out

    def _projector_for_call(self, tc: ToolCall) -> ToolProjector | None:
        if type(tc).projector_class is None:
            return self._default_tool_projector()       # plain ToolCall -> function_call
        projector = type(tc).projector_class()
        if isinstance(projector, self.TOOL_PROJECTOR_BASE):
            return projector
        return None   # a native call minted by ANOTHER transport (e.g. /model
                      # switched providers): dropped, same policy as a foreign
                      # thinking-block attestation — one lost exchange beats a 400
                      # that kills the conversation.

    def _project_assistant_message(self, msg, request, lineage) -> list[dict]:
        items: list[dict] = []
        for block in msg.content:
            if isinstance(block, ToolCall):
                projector = self._projector_for_call(block)
                lineage[block.id] = (projector, block) if projector else None
                if projector is not None:
                    items.append(projector.project_tool_call_to_llm(block))
            elif isinstance(block, ThinkingBlock):
                ...   # unchanged (attestation rules)
            elif isinstance(block, TextBlock):
                ...   # unchanged
            # RefusalBlock: skipped, unchanged
        return items

    def _project_tool_message(self, msg: ToolMessage, lineage) -> dict | None:
        if msg.tool_call_id in lineage:
            entry = lineage[msg.tool_call_id]
            if entry is None:
                return None                              # its call was dropped
            projector, call = entry
            return projector.project_tool_message_to_llm(msg, call)
        # no lineage (hand-built history): standard shape, call unknown
        return self._default_tool_projector().project_tool_message_to_llm(msg, None)
```

### `luca/client/transports/openai_responses/native_tools.py`

```python
# ---- Apply Patch ----------------------------------------------------------

class ApplyPatchToolCall(ToolCall):
    type: Literal["apply_patch_call"] = "apply_patch_call"
    item_id: str | None = None            # wire item "id" (apc_…) — replayed
    status: str = "completed"             # "in_progress" while streaming

    @property
    def operation(self) -> dict:          # typed accessor; storage is arguments
        return self.arguments


class ApplyPatchProjector(OpenAIResponsesToolProjector):

    def project_tool_to_llm(self, tool) -> dict:
        return {"type": "apply_patch"}

    def build_tool_call(self, item: dict) -> ApplyPatchToolCall:
        # wire: {"type": "apply_patch_call", "id": "apc_…", "call_id": "call_…",
        #        "status": "completed",
        #        "operation": {"type": "create_file", "path": "…", "diff": "+…"}}
        return ApplyPatchToolCall(
            id=item["call_id"],
            name=ApplyPatchTool.name,                 # synthesized: wire has no name
            arguments=dict(item.get("operation") or {}),
            item_id=item.get("id"),
            status=item.get("status", "completed"),
            complete=True,
        )

    def project_tool_call_to_llm(self, tc: ApplyPatchToolCall) -> dict:
        # replay verified live: full item, id + status included, accepted on
        # every later turn under store:false
        return {
            "type": "apply_patch_call",
            "id": tc.item_id,
            "call_id": tc.id,
            "status": tc.status,
            "operation": tc.arguments,
        }

    def project_tool_message_to_llm(self, msg: ToolMessage, call) -> dict:
        # plain ToolMessage suffices: is_error IS the status. No
        # ApplyPatchToolMessage — two fields saying "failed" would drift.
        return {
            "type": "apply_patch_call_output",
            "call_id": msg.tool_call_id,
            "status": "failed" if msg.is_error else "completed",
            "output": self._output_text(msg),
        }

    def project_tool_choice_to_llm(self, tool) -> dict:
        return {"type": "apply_patch"}    # native tools force by TYPE, not name
                                          # (validated live 2026-08-06, streamed)


ApplyPatchToolCall.projector_class = ApplyPatchProjector


class ApplyPatchTool(BaseTool):
    name: ClassVar[str] = "apply_patch"

    def get_projector(self) -> ToolProjector:
        return ApplyPatchProjector()


# ---- Local Shell ----------------------------------------------------------

class ShellExitOutcome(BaseModel):
    type: Literal["exit"] = "exit"
    exit_code: int

class ShellTimeoutOutcome(BaseModel):
    type: Literal["timeout"] = "timeout"

class ShellCommandResult(BaseModel):
    stdout: str
    stderr: str
    outcome: Annotated[ShellExitOutcome | ShellTimeoutOutcome, Field(discriminator="type")]


class ShellToolCall(ToolCall):
    type: Literal["shell_call"] = "shell_call"
    item_id: str | None = None
    status: str = "completed"

    @property
    def action(self) -> dict:             # {"commands": [...], "timeout_ms"?, "max_output_length"?}
        return self.arguments


class ShellToolMessage(ToolMessage):
    # structured results CAN'T ride prose — this is the one native result type
    type: Literal["shell_call_output"] = "shell_call_output"
    results: list[ShellCommandResult]
    content: str | list[TextBlock | ImageBlock] = ""    # optional human summary


class ShellProjector(OpenAIResponsesToolProjector):

    def project_tool_to_llm(self, tool) -> dict:
        return {"type": "shell", "environment": {"type": "local"}}

    def build_tool_call(self, item: dict) -> ShellToolCall:
        # wire: {"type": "shell_call", "id": "…", "call_id": "call_…",
        #        "status": "…", "action": {"commands": ["wc -c hello.txt"],
        #        "timeout_ms": 10000, "max_output_length": 10240}}
        return ShellToolCall(
            id=item["call_id"],
            name=LocalShellTool.name,
            arguments=dict(item.get("action") or {}),
            item_id=item.get("id"),
            status=item.get("status", "completed"),
            complete=True,
        )

    def project_tool_call_to_llm(self, tc: ShellToolCall) -> dict:
        return {
            "type": "shell_call",
            "id": tc.item_id,
            "call_id": tc.id,
            "status": tc.status,
            "action": tc.arguments,
        }

    def project_tool_message_to_llm(self, msg: ToolMessage, call: ToolCall | None) -> dict:
        if not isinstance(msg, ShellToolMessage):
            raise BadRequestError(
                f"shell call {msg.tool_call_id!r} needs a ShellToolMessage "
                "(per-command stdout/stderr/exit codes); got a plain ToolMessage.",
            )
        out = {
            "type": "shell_call_output",
            "call_id": msg.tool_call_id,
            "output": [r.model_dump() for r in msg.results],
        }
        # observed: echo max_output_length when the CALL carried it — this is
        # why project_tool_message_to_llm receives the originating call.
        if call is not None and call.arguments.get("max_output_length") is not None:
            out["max_output_length"] = call.arguments["max_output_length"]
        return out

    def project_tool_choice_to_llm(self, tool) -> dict:
        return {"type": "shell"}          # native tools force by TYPE, not name


ShellToolCall.projector_class = ShellProjector


class LocalShellTool(BaseTool):
    name: ClassVar[str] = "shell"

    def get_projector(self) -> ToolProjector:
        return ShellProjector()
```

### `luca/client/transports/anthropic/transport.py`

```python
class AnthropicToolProjector(ToolProjector):

    def project_tool_to_llm(self, tool: Tool) -> dict:
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool_parameters_to_json_schema(tool.parameters),
        }

    def build_tool_call(self, block: dict) -> ToolCall:      # a tool_use block
        return ToolCall(id=block["id"], name=block["name"],
                        arguments=block.get("input", {}) or {}, complete=True)

    def project_tool_call_to_llm(self, tc: ToolCall) -> dict:
        return {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}

    def project_tool_message_to_llm(self, msg: ToolMessage, call) -> dict:
        # returns the tool_result BLOCK; the transport wraps it in the
        # {"role": "user", "content": [block]} envelope it owns.
        return {
            "type": "tool_result",
            "tool_use_id": msg.tool_call_id,
            "content": self._result_content(msg),            # today's str/blocks logic
            "is_error": msg.is_error,
        }

    def project_tool_choice_to_llm(self, tool: Tool) -> dict:
        # {"type": "tool", "name": ...} forces native tools too (they are
        # ordinary named tools) — the native projectors need no override.
        return {"type": "tool", "name": tool.name}


class AnthropicTransport(BaseTransport, ChatCompletionTransportMixin):
    TOOL_PROJECTOR_BASE: ClassVar[type] = AnthropicToolProjector
    # _default_tool_projector / _resolve_projector / _project_tools: same
    # shapes as the Responses transport above.
    #
    # _parse_assistant_message: the tool_use branch calls
    # default.build_tool_call(block) — native calls arrive through the SAME
    # branch (they are ordinary tool_use blocks; validated live), so there is
    # no registry consult on this wire.
    #
    # _project_messages: same lineage walk as Responses (records
    # {call_id: (projector, call) | None}); its only job here is dropping
    # foreign native calls/results after a provider switch — Anthropic-native
    # calls are plain ToolCalls and take the default path.
    #
    # An assistant message that projects to ZERO wire blocks (foreign
    # thinking + foreign native calls, the typical post-switch OpenAI turn)
    # is omitted entirely — this wire rejects empty content. Same rule on
    # chat completions and Bedrock.
```

### `luca/client/transports/anthropic/native_tools.py`

```python
class TextEditorProjector(AnthropicToolProjector):
    # ONLY the declaration differs; build/replay/result inherited (native
    # calls are ordinary tool_use / tool_result — validated live).
    def project_tool_to_llm(self, tool: TextEditorTool) -> dict:
        decl = {"type": "text_editor_20250728", "name": tool.name}
        if tool.max_characters is not None:
            decl["max_characters"] = tool.max_characters
        return decl


class TextEditorTool(BaseTool):
    name: ClassVar[str] = "str_replace_based_edit_tool"
    max_characters: int | None = None

    def get_projector(self) -> ToolProjector:
        return TextEditorProjector()


class BashProjector(AnthropicToolProjector):
    def project_tool_to_llm(self, tool) -> dict:
        return {"type": "bash_20250124", "name": tool.name}


class BashTool(BaseTool):
    name: ClassVar[str] = "bash"

    def get_projector(self) -> ToolProjector:
        return BashProjector()
```

### `luca/client/types/streaming.py` + `openai_responses/stream.py`

```python
@dataclass
class RawBlockStart:
    ...
    prebuilt: ToolCall | None = None      # transport-built typed subclass
                                          # (accumulator stays projector-blind)

@dataclass
class RawBlockStop:
    index: int
    replacement: ToolCall | None = None   # the complete native call


# Same dump rule as AssistantMessage.content — these are the other two
# containers declared as base ToolCall, and dumping a stream event must not
# drop native subclass fields (the pydantic warning would fail the suite):
class ToolCallEndEvent(BaseModel):
    tool_call: SerializeAsAny[ToolCall]

class FinishEvent(BaseModel):
    tool_calls: list[SerializeAsAny[ToolCall]] = Field(default_factory=list)


class _ChatCompletionAccumulator:
    def handle_raw(self, raw):
        if isinstance(raw, RawBlockStart):
            ...
            elif raw.block_type == "tool_call":
                if raw.prebuilt is not None:
                    self._message.content.append(raw.prebuilt)        # complete=False
                    yield ToolCallStartEvent(index=raw.index, id=raw.prebuilt.id,
                                             name=raw.prebuilt.name, partial=...)
                else:
                    ...                                               # today's path
        elif isinstance(raw, RawBlockStop):
            ...
            elif isinstance(block, ToolCall):
                if raw.replacement is not None:
                    self._message.content[raw.index] = raw.replacement  # final args
                    yield ToolCallEndEvent(index=raw.index, tool_call=raw.replacement, partial=...)
                else:
                    ...   # today's path: json.loads(partial_arguments)
```

```python
# openai_responses/stream.py
_NATIVE_CALL_SLOT = -3

def _item_added(state, event):
    ...
    elif (native := NATIVE_TOOL_CALL_TYPES.get(item_type)) is not None:
        # observed SSE: output_item.added carries the skeleton —
        # status "in_progress", operation {"type", "path", "diff": ""}
        call = native.projector_class().build_tool_call(item)
        call.complete = False
        index = state.allocate((output_index, _NATIVE_CALL_SLOT))
        yield RawBlockStart(index=index, block_type="tool_call",
                            tool_id=call.id, tool_name=call.name, prebuilt=call)

def _item_done(state, event):
    ...
    elif (native := NATIVE_TOOL_CALL_TYPES.get(item_type)) is not None:
        index = state.resolve((output_index, _NATIVE_CALL_SLOT))
        if index is not None and state.close(index):
            final = native.projector_class().build_tool_call(item)   # complete item
            yield RawBlockStop(index=index, replacement=final)

# response.apply_patch_call_operation_diff.delta — RAW TEXT, not JSON; falls
# through the event dispatch unhandled in V1 (no public tool_call_delta).
# A terminal closing an open native block emits RawBlockStop with NO
# replacement -> same degenerate empty-arguments close as a truncated
# function call today.
```

### `luca/client/providers/*.py` — re-exports

```python
# providers/openai.py
from ..transports.openai_responses.native_tools import (
    ApplyPatchTool, ApplyPatchToolCall,
    LocalShellTool, ShellToolCall, ShellToolMessage,
    ShellCommandResult, ShellExitOutcome, ShellTimeoutOutcome,
)

# providers/anthropic.py
from ..transports.anthropic.native_tools import BashTool, TextEditorTool

# transports/openai_responses/__init__.py and transports/anthropic/__init__.py
# import their native_tools module -> registration rides the existing eager
# import chain: importing luca.client registers every first-party native type.
```

---

## Part 2 — usage

### One task, two providers, one loop

The design's point in one example: the loop and the registry pattern are
provider-agnostic — only the tool set and its handlers change. Same task on
both providers; the custom tool is identical everywhere.

```python
from pydantic import BaseModel

from luca.client import completion
from luca.client.types import TextBlock, Tool, ToolCall, ToolMessage, UserMessage
from luca.client.providers.anthropic import BashTool, TextEditorTool
from luca.client.providers.openai import (
    ApplyPatchTool, ApplyPatchToolCall,
    LocalShellTool, ShellToolCall, ShellToolMessage,
    ShellCommandResult, ShellExitOutcome, ShellTimeoutOutcome,
)


# --- the user's custom tool: identical on every provider --------------------

class BinaryOp(BaseModel):
    a: float
    b: float

add = Tool(name="add", description="Add two numbers.", parameters=BinaryOp)

def handle_add(tc: ToolCall) -> ToolMessage:
    op = tc.parse_arguments(BinaryOp)            # typed validation of arguments
    return ToolMessage(tool_call_id=tc.id, name=tc.name,
                       content=[TextBlock(text=str(op.a + op.b))])


# --- the loop: provider-agnostic --------------------------------------------

TASK = "Create hello.txt with 'hi luca', count its characters, then add 5 to the count."

def run_task(model: str, tools: list, registry: dict):
    messages = [UserMessage(content=[TextBlock(text=TASK)])]
    while True:
        response = completion(model, messages, tools=tools)
        messages.append(response.message)        # replay bookkeeping ends here
        if response.finish_reason != "tool_use":
            return response
        for tc in response.tool_calls:
            messages.append(registry[tc.name](tc))   # handlers return ToolMessages

# tool sets + registries defined in the next two sections
run_task("openai:gpt-5.1", OPENAI_TOOLS, OPENAI_REGISTRY)
run_task("anthropic:claude-sonnet-4-5", ANTHROPIC_TOOLS, ANTHROPIC_REGISTRY)
```

### OpenAI (Responses): apply_patch + shell

```python
OPENAI_TOOLS = [add, ApplyPatchTool(), LocalShellTool()]

OPENAI_REGISTRY = {
    "add": handle_add,
    ApplyPatchTool.name: handle_apply_patch,     # "apply_patch"  (synthesized)
    LocalShellTool.name: handle_shell,           # "shell"        (synthesized)
}

# What arrives — TYPED subclasses; the REPL tells you what you are holding:
#
# >>> response.tool_calls
# [ApplyPatchToolCall(id='call_…', name='apply_patch', item_id='apc_…',
#      status='completed',
#      arguments={'type': 'create_file', 'path': 'hello.txt', 'diff': '+hi luca\n'})]
#
# ...next turn...
# [ShellToolCall(id='call_…', name='shell', item_id='sh_…', status='completed',
#      arguments={'commands': ['wc -c hello.txt'], 'timeout_ms': 10000,
#                 'max_output_length': 10240})]


def handle_apply_patch(tc: ApplyPatchToolCall) -> ToolMessage:
    op = tc.operation        # typed accessor over .arguments:
                             # {"type": "create_file"|"update_file"|"delete_file", "path", "diff"?}
    ok, detail = apply_v4a(op)
    return ToolMessage(
        tool_call_id=tc.id, name=tc.name,
        content=[TextBlock(text=detail)],        # -> "output"; make failures useful
        is_error=not ok,                         # -> status "completed" | "failed"
    )


def handle_shell(tc: ShellToolCall) -> ShellToolMessage:
    action = tc.action       # {"commands": [...], "timeout_ms": 10000, "max_output_length": 10240}
    results = []
    for cmd in action["commands"]:               # in order, one result each
        r = run(cmd, timeout_ms=action.get("timeout_ms"))
        results.append(ShellCommandResult(
            stdout=r.stdout, stderr=r.stderr,
            outcome=ShellExitOutcome(exit_code=r.code) if not r.timed_out
                    else ShellTimeoutOutcome(),
        ))
    return ShellToolMessage(tool_call_id=tc.id, name=tc.name, results=results)
```

### Anthropic: text editor + bash

```python
ANTHROPIC_TOOLS = [add, TextEditorTool(max_characters=10_000), BashTool()]

ANTHROPIC_REGISTRY = {
    "add": handle_add,
    TextEditorTool.name: handle_text_editor,     # "str_replace_based_edit_tool" (wire name)
    BashTool.name: handle_bash,                  # "bash"                        (wire name)
}

# What arrives — BASE ToolCalls (ordinary tool_use blocks, wire names kept);
# .name points at the tool, the payload is .arguments:
#
# >>> response.tool_calls
# [ToolCall(id='toolu_…', name='str_replace_based_edit_tool',
#      arguments={'command': 'create', 'path': 'hello.txt', 'file_text': 'hi luca'})]
#
# ...next turn...
# [ToolCall(id='toolu_…', name='bash', arguments={'command': 'wc -c hello.txt'})]


def handle_text_editor(tc: ToolCall) -> ToolMessage:
    args = tc.arguments      # the text_editor_20250728 command set
    match args["command"]:
        case "view":         # {"command", "path", "view_range"?: [start, end]}
            detail = view_numbered(args["path"], args.get("view_range"))   # cat -n style
        case "create":       # {"command", "path", "file_text"}
            detail = create_file(args["path"], args["file_text"])
        case "str_replace":  # {"command", "path", "old_str", "new_str"}
            detail = str_replace(args["path"], args["old_str"], args["new_str"])
        case "insert":       # {"command", "path", "insert_line", "insert_text"}
            detail = insert_at(args["path"], args["insert_line"], args["insert_text"])
        case other:
            return ToolMessage(tool_call_id=tc.id, name=tc.name, is_error=True,
                               content=[TextBlock(text=f"unknown command {other!r}")])
    return ToolMessage(tool_call_id=tc.id, name=tc.name, content=[TextBlock(text=detail)])


def handle_bash(tc: ToolCall) -> ToolMessage:
    if tc.arguments.get("restart"):              # {"restart": true} — no command key
        SHELL.restart()                          # caller owns the persistent session
        return ToolMessage(tool_call_id=tc.id, name=tc.name,
                           content=[TextBlock(text="shell session restarted")])
    r = SHELL.run(tc.arguments["command"])       # persistent across calls
    return ToolMessage(
        tool_call_id=tc.id, name=tc.name,
        content=[TextBlock(text=r.stdout + r.stderr)],
        is_error=r.exit_code != 0,
    )
```

### Side by side

| | OpenAI Responses | Anthropic |
|---|---|---|
| Declare | `ApplyPatchTool()`, `LocalShellTool()` | `TextEditorTool(max_characters=…)`, `BashTool()` |
| Wire declaration | `{"type": "apply_patch"}` / `{"type": "shell", "environment": {"type": "local"}}` | `{"type": "text_editor_20250728", "name": …}` / `{"type": "bash_20250124", "name": "bash"}` |
| Call arrives as | `ApplyPatchToolCall` / `ShellToolCall` — typed, names synthesized (`"apply_patch"`, `"shell"`) | base `ToolCall` — wire names kept (`"str_replace_based_edit_tool"`, `"bash"`) |
| Payload | `tc.operation` / `tc.action` (properties over `arguments`) | `tc.arguments` |
| Result you return | plain `ToolMessage` / `ShellToolMessage` (structured) | plain `ToolMessage` |
| Registry keys | `ApplyPatchTool.name` / `LocalShellTool.name` | `TextEditorTool.name` / `BashTool.name` |
| Streaming | start → end, arguments complete at end | full JSON argument deltas |
| Forcing | `{"name": "apply_patch"}` → `{"type": "apply_patch"}` | `{"name": "bash"}` → `{"type": "tool", "name": "bash"}` |

The custom `add` tool is untouched by any of this: same declaration, same
handler, same registry entry on both providers. Subclassing happens at both
extension points — `ToolCall` and `ToolMessage` — and which concrete classes
materialize per tool is dictated by what the wire carries, not by branding.

### Failure modes

```python
completion("openrouter:openai/gpt-4o", messages, tools=[BashTool()])
# BadRequestError before any HTTP: BashTool's projector targets another transport.

completion("openai:gpt-5.1", messages, tools=[{"type": "apply_patch"}])
# BadRequestError: dicts mean standard function tools; native tools are instances.

# plain ToolMessage answering a shell call
# BadRequestError at projection, naming ShellToolMessage.
```

### Forcing a tool (`tool_choice`)

```python
completion("openai:gpt-5.1", messages, tools=OPENAI_TOOLS, tool_choice={"name": "apply_patch"})
# wire: "tool_choice": {"type": "apply_patch"} — the name resolved against the
# declared tools; the matched tool's projector emits the forcing shape.

completion("openai:gpt-5.1", messages, tools=OPENAI_TOOLS, tool_choice={"name": "add"})
# wire: {"type": "function", "name": "add"} — default projector, today's shape.

completion("anthropic:claude-sonnet-4-5", messages, tools=[BashTool()], tool_choice={"name": "bash"})
# wire: {"type": "tool", "name": "bash"} — inherited default; no override needed.
```

### Streaming

```python
with completion_stream("openai:gpt-5.1", messages, tools=OPENAI_TOOLS) as stream:
    for event in stream:
        if event.type == "tool_call_start":
            print("starting:", event.name)                  # "apply_patch"
        elif event.type == "tool_call_delta":
            ...                                             # function tools only (V1)
        elif event.type == "tool_call_end":
            print("final:", event.tool_call.arguments)      # always complete
```

Anthropic native tools stream with full argument deltas (ordinary `tool_use`
JSON — validated live). OpenAI native calls stream start → end, arguments
complete at end.

### Persistence

```python
payload = [m.model_dump() for m in messages]           # subclass fields survive
restored = [AssistantMessage.model_validate(p) if p["role"] == "assistant" else ...
            for p in payload]
# native payloads revalidate into ApplyPatchToolCall / ShellToolMessage —
# provided luca.client is imported (registration happens at import).
```
