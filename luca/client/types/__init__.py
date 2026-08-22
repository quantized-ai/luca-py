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
    PrivateProviderBlock,
    RefusalBlock,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolResultBlock,
    URLCitationAnnotation,
    WebFetchBlock,
    WebPagePart,
    WebSearchBlock,
)
from .media import MediaBase64, MediaFileId, MediaSource, MediaURL
from .messages import AssistantMessage, Message, ToolMessage, UserMessage
from .reasoning import Reasoning
from .streaming import (
    ErrorEvent,
    FinishEvent,
    RefusalDeltaEvent,
    RefusalEndEvent,
    RefusalStartEvent,
    StartEvent,
    StreamEvent,
    TextAnnotationEvent,
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
    WebEndEvent,
    WebFetchEvent,
    WebFindEvent,
    WebSearchEvent,
    WebSearchResultEvent,
    WebStartEvent,
)
from .structured import (
    ResponseFormat,
    parse_structured_output,
    response_format_to_json_schema,
    strictify_json_schema,
)
from .tools import (
    ApproximateLocation,
    BaseTool,
    Tool,
    ToolChoice,
    ToolParameters,
    ToolProjector,
    tool_parameters_to_json_schema,
)

# Resolve forward references. AssistantMessage references Usage as a string;
# completion.py is now imported, so we can rebuild.
AssistantMessage.model_rebuild()
ChatCompletionResponse.model_rebuild()


__all__ = [
    # catalog
    "ModelCost",
    "ModelInfo",
    # completion
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "Usage",
    "UsageCost",
    # content
    "AudioBlock",
    "ContentBlock",
    "FileBlock",
    "ImageBlock",
    "PrivateProviderBlock",
    "RefusalBlock",
    "TextBlock",
    "ThinkingBlock",
    "ToolCall",
    "ToolResultBlock",
    "URLCitationAnnotation",
    "WebFetchBlock",
    "WebPagePart",
    "WebSearchBlock",
    # media
    "MediaBase64",
    "MediaFileId",
    "MediaSource",
    "MediaURL",
    # messages
    "AssistantMessage",
    "Message",
    "ToolMessage",
    "UserMessage",
    # reasoning
    "Reasoning",
    # streaming
    "ErrorEvent",
    "FinishEvent",
    "RefusalDeltaEvent",
    "RefusalEndEvent",
    "RefusalStartEvent",
    "StartEvent",
    "StreamEvent",
    "TextAnnotationEvent",
    "TextDeltaEvent",
    "TextEndEvent",
    "TextStartEvent",
    "ThinkingDeltaEvent",
    "ThinkingEndEvent",
    "ThinkingStartEvent",
    "ToolCallDeltaEvent",
    "ToolCallEndEvent",
    "ToolCallStartEvent",
    "UsageEvent",
    "WebEndEvent",
    "WebFetchEvent",
    "WebFindEvent",
    "WebSearchEvent",
    "WebSearchResultEvent",
    "WebStartEvent",
    # structured
    "ResponseFormat",
    "parse_structured_output",
    "response_format_to_json_schema",
    "strictify_json_schema",
    # tools
    "ApproximateLocation",
    "BaseTool",
    "Tool",
    "ToolChoice",
    "ToolParameters",
    "ToolProjector",
    "tool_parameters_to_json_schema",
]
