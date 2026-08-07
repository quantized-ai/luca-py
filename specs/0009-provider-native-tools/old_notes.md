I want you to implement the changes to the client defined in @specs/0009-provider-native-tools. Read the following files:

prd.md - main driver of the implementation
plan.md - implementation plan
examples.md - some "pseudocode-style" examples that define the high level API for users




```python
# luca.client.types
class ToolProjector:
    def project_tool_to_llm(self, tool):
        raise NotImplementedError()

    def project_tool_message_to_llm(self, msg: ToolMessage):
        raise NotImplementedError()


class Tool:
    def get_projector(self):
        return None


# luca.client.transports.openai_responses.transport
class DefaultOpenAIResponsesToolProjector:
    def project_tool_to_llm(self, tool):
        return {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool_parameters_to_json_schema(tool.parameters),
        }

    def project_tool_message_to_llm(self, msg: ToolMessage) -> dict:
        if isinstance(msg.content, str):
            output = msg.content
        else:
            # Refusing beats dropping: the model would be told the tool call
            # succeeded and handed a result with the image missing.
            unsupported = {type(b).__name__ for b in msg.content if not isinstance(b, TextBlock)}
            if unsupported:
                raise BadRequestError(
                    "The responses API allows only text in a function call "
                    f"output; cannot send {', '.join(sorted(unsupported))} "
                    f"for call_id {msg.tool_call_id!r}.",
                    provider=self._provider,
                )
            output = "".join(b.text for b in msg.content)
        return {
            "type": "function_call_output",
            "call_id": msg.tool_call_id,
            "output": output,
        }

    def get_tool_message_matching_values(self, tool):
        # This method doesn't exist in the base class
        # it's an OpenAI only problem
        return {"function_call"}

    def build_tool_call(self, item: dict) -> ToolCall:
        return ToolCall(
            id=item["call_id"],
            name=item["name"],
            arguments=self._parse_arguments(item.get("arguments")),
            complete=True,
        )


class OpenAIResponsesTransport(BaseTransport, OpenAIErrorMappingMixin, ChatCompletionTransportMixin):
    ...
    def get_projector(self, tool):
        return tool.get_projector() or DefaultOpenAIResponsesToolProjector()

    def _project_tools(self, tools: list) -> list[dict]:
        tool_projections = []
        for tool in tools:
            projector = self.get_projector(tool)
            tool_projections.append(projector.project_tool_to_llm(tool))

    def _project_tool_message(self, msg: ToolMessage) -> dict:
        projector = self.get_projector(tool)
        return projector.project_tool_message_to_llm(msg)

    def _parse_assistant_message(
        self,
        data: dict,
        request: ChatCompletionRequest,
    ) -> AssistantMessage:
        content: list = []
        for item in data.get("output") or []:
            item_type = item.get("type")
            if item_type == "reasoning":
                content.append(self._parse_reasoning_item(item))
                continue
            if item_type == "message":
                content.extend(self._parse_message_item(item))
                continue
            if item_type == "function_call":
                projector = DefaultOpenAIResponsesToolProjector()
                content.append(projector.build_tool_call(item))

            # None of the previous types types matched, might be a "custom" tool
            for tool in self.request.tools:
                projector = self.get_projector(tool)
                if item_type in projector.get_tool_message_matching_values(tool)
                    content.append(projector.build_tool_call(item))
                    continue
            # Hosted-tool items (web_search_call, file_search_call, …) have no
            # canonical block; ignored rather than guessed at.
        return AssistantMessage(
            content=content,
            provider=self._provider,
            model=request.model,
            response_model=data.get("model"),
            response_id=data.get("id"),
        )

# luca.client.transports.openai_responses.native_tools

class ApplyPatchToolCall(ToolCall):
    status: str # one of completed|failed
    type: str = "apply_patch_call"
    operation: dict


class ApplyPatchTool:
    class Projector:
        def project_tool_to_llm(self, tool):
            return {"type": "apply_patch"}

        def project_tool_message_to_llm(self, msg: ToolMessage) -> dict:
            if isinstance(msg.content, str):
                output = msg.content
            else:
                # Refusing beats dropping: the model would be told the tool call
                # succeeded and handed a result with the image missing.
                unsupported = {type(b).__name__ for b in msg.content if not isinstance(b, TextBlock)}
                if unsupported:
                    raise BadRequestError(
                        "The responses API allows only text in a function call "
                        f"output; cannot send {', '.join(sorted(unsupported))} "
                        f"for call_id {msg.tool_call_id!r}.",
                        provider=self._provider,
                    )
                output = "".join(b.text for b in msg.content)
            return {
                "type": "apply_patch_call_output",
                "call_id": msg.tool_call_id,
                "status": "completed",
                "output": output,
            }

        def get_tool_message_matching_values(self, tool):
            return {"apply_patch_call"}

        def build_tool_call(self, item: dict) -> ApplyPatchToolCall:
            return ApplyPatchToolCall(
                id=item["call_id"],
                type="apply_patch_call",
                status="completed",
                operation=item['operation']

            )

{"type": "apply_patch_call",
     "id": "apc_071c…", "call_id": "call_fc6U…", "status": "completed",
     "operation": {"type": "create_file", "path": "hello.txt", "diff": "+hi luca\n"}},

    def get_projector(self):
        return self.Projector()

class ApplyPatchToolMessage(ToolMessage):
    status: str
    type: str = "apply_patch_call_output"

def apply_patch_tool_handler(tool_call: ApplyPatchToolCall):
    # ... whatever all the logic ...
    return ApplyPatchToolMessage(
        status=completed,
        output="created hello.txt"
    )

# main usage of the user


class BinaryOp(BaseModel):
    a: float = Field(description="First operand.")
    b: float = Field(description="Second operand.")

def add(a, b): return a + b
def multiply(a, b): return a * b

TOOLS = [
    Tool(name="add", description="Add two numbers.", parameters=BinaryOp),
    Tool(name="multiply", description="Multiply two numbers.", parameters=BinaryOp),
]

def partial(fn):
    def execute(tc: ToolCall):
        result = str(fn(**tc.arguments))
        return ToolMessage(
            tool_call_id=tc.id,
            content=[TextBlock(text=result)],
        )
    return execute

REGISTRY: dict[str, Callable] = {"add": partial(add), "multiply": partial(multiply)}

TOOLS += [ApplyPatchTool]
REGISTRY[ApplyPatchTool.name] = apply_patch_tool_handler

while True:
    response = completion(
        model="openai:gpt-5.4",
        messages=messages,
        system_message="Use the tools for any arithmetic.",
        tools=TOOLS,
    )
    messages.append(response.message)

    if response.finish_reason != "tool_use":
        break

    for tc in response.tool_calls:
        handler = REGISTRY[tc.name]
        messages.append(handler(tc))
```

