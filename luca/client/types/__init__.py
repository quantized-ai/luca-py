"""Public DTOs for the SDK. Re-exports everything from the submodules."""

from .catalog import ModelCost, ModelInfo
from .completion import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Usage,
    UsageCost,
)
from .content import (
    AudioBlock,
    ContentBlock,
    FileBlock,
    ImageBlock,
    RefusalBlock,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolResultBlock,
)
from .media import MediaBase64, MediaFileId, MediaSource, MediaURL
from .messages import AssistantMessage, Message, ToolMessage, UserMessage
from .reasoning import ReasoningEffort
from .streaming import (
    AsyncBaseStream,
    AsyncChatCompletionStream,
    BaseStream,
    ChatCompletionStream,
    ErrorEvent,
    FinishEvent,
    RawBlockStart,
    RawBlockStop,
    RawFinish,
    RawRefusalDelta,
    RawStreamEvent,
    RawTextDelta,
    RawThinkingDelta,
    RawToolArgumentsDelta,
    RawUsage,
    RefusalDeltaEvent,
    RefusalEndEvent,
    RefusalStartEvent,
    StartEvent,
    StreamEvent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    UsageEvent,
)
from .structured import ResponseFormat, parse_structured_output
from .tools import Tool, ToolChoice, ToolParameters, tool_parameters_to_json_schema

# Resolve forward references. AssistantMessage references Usage as a string;
# completion.py is now imported, so we can rebuild.
AssistantMessage.model_rebuild()
ChatCompletionResponse.model_rebuild()


__all__ = [
    # catalog
    "ModelCost", "ModelInfo",
    # completion
    "ChatCompletionRequest", "ChatCompletionResponse", "Usage", "UsageCost",
    # content
    "AudioBlock", "ContentBlock", "FileBlock", "ImageBlock",
    "RefusalBlock", "TextBlock", "ThinkingBlock", "ToolCall", "ToolResultBlock",
    # media
    "MediaBase64", "MediaFileId", "MediaSource", "MediaURL",
    # messages
    "AssistantMessage", "Message", "ToolMessage", "UserMessage",
    # reasoning
    "ReasoningEffort",
    # streaming
    "AsyncBaseStream", "AsyncChatCompletionStream", "BaseStream",
    "ChatCompletionStream", "ErrorEvent", "FinishEvent",
    "RawBlockStart", "RawBlockStop", "RawFinish", "RawRefusalDelta",
    "RawStreamEvent", "RawTextDelta", "RawThinkingDelta",
    "RawToolArgumentsDelta", "RawUsage",
    "RefusalDeltaEvent", "RefusalEndEvent", "RefusalStartEvent",
    "StartEvent", "StreamEvent",
    "TextDeltaEvent", "TextEndEvent", "TextStartEvent",
    "ThinkingDeltaEvent", "ThinkingEndEvent", "ThinkingStartEvent",
    "ToolCallDeltaEvent", "ToolCallEndEvent", "ToolCallStartEvent",
    "UsageEvent",
    # structured
    "ResponseFormat", "parse_structured_output",
    # tools
    "Tool", "ToolChoice", "ToolParameters", "tool_parameters_to_json_schema",
]
