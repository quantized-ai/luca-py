"""luca.agent.core — the agent framework core.

The conversation data model (`models`) plus the runtime surface: the
`AgentSessionRunner` loop, the `ToolRegistry` contract, the
`ConversationProjector` LLM-projection strategy, the `ContextManager`
accounting strategy, the `CompactionPolicy` contract, the system-prompt
strategy, and the agent exceptions. Event classes live in
`luca.agent.core.events`; the inbound response / tool-definition translations
in `luca.agent.core.adapter`; the read-only `pretty_print` session transcript
in `luca.agent.core.utils`.

The core's only tool type is `ToolSpec` — plain, language-neutral data. The
ergonomic Python `Tool` base class lives in `luca.agent.contrib.tools`;
nothing in the core depends on it.
"""

from .compaction import (
    CompactionPlan,
    CompactionPolicy,
    UsageCounters,
)
from .context import CancellationToken
from .context_manager import PRUNED_TOOL_OUTPUT_MARKER, ContextManager
from .exceptions import (
    AgentError,
    AlreadyCancellingError,
    CancelledError,
    CompactionPlanError,
    InvalidToolArguments,
    ProjectionError,
    ToolNotFound,
)
from .middleware import AgentMiddlewareMixin
from .models import (
    AgentSession,
    AnyEntry,
    ApprovalDecision,
    ApprovalOption,
    ApprovalStatus,
    AssistantContentPart,
    AssistantMessage,
    BaseConfigModel,
    CancelRequested,
    CompactionEntry,
    CompactionSource,
    ContentPart,
    Conversation,
    ConversationStatus,
    Entry,
    ExecutionResult,
    ExecutionStatus,
    ImageBase64,
    ImageContent,
    ImageFileId,
    ImageSource,
    ImageURL,
    Inf,
    LLMConfig,
    MilliSeconds,
    PrunedEntry,
    RuntimeConfig,
    Seconds,
    SessionConfig,
    SessionRuntimeStatus,
    SystemPromptPart,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolExecution,
    ToolExecutionError,
    ToolKind,
    ToolSpec,
    TurnFinish,
    TurnOutcome,
    TurnStart,
    Usage,
    UserMessage,
)
from .projection import (
    CANCELLED_TURN_MARKER,
    IMAGE_BLOCK_MARKER,
    ConversationProjector,
)
from .runner import AgentRun, AgentSessionRunner, RunResult
from .system_prompt import DefaultSystemPromptAssembler, SystemPromptAssembler
from .tool_registry import PreparedTool, ToolRegistry
from .utils import pretty_print

__all__ = [
    "AgentError",
    "AgentMiddlewareMixin",
    "AgentRun",
    "AgentSession",
    "AgentSessionRunner",
    "AlreadyCancellingError",
    "AnyEntry",
    "ApprovalDecision",
    "ApprovalOption",
    "ApprovalStatus",
    "AssistantContentPart",
    "AssistantMessage",
    "BaseConfigModel",
    "CANCELLED_TURN_MARKER",
    "CancelRequested",
    "CancellationToken",
    "CancelledError",
    "CompactionEntry",
    "CompactionPlan",
    "CompactionPlanError",
    "CompactionPolicy",
    "CompactionSource",
    "ContentPart",
    "ContextManager",
    "Conversation",
    "ConversationProjector",
    "ConversationStatus",
    "DefaultSystemPromptAssembler",
    "Entry",
    "ExecutionResult",
    "ExecutionStatus",
    "ImageBase64",
    "ImageContent",
    "ImageFileId",
    "ImageSource",
    "IMAGE_BLOCK_MARKER",
    "ImageURL",
    "Inf",
    "InvalidToolArguments",
    "LLMConfig",
    "MilliSeconds",
    "PRUNED_TOOL_OUTPUT_MARKER",
    "ProjectionError",
    "PreparedTool",
    "PrunedEntry",
    "RunResult",
    "RuntimeConfig",
    "Seconds",
    "SessionConfig",
    "SessionRuntimeStatus",
    "SystemPromptAssembler",
    "SystemPromptPart",
    "TextContent",
    "ThinkingContent",
    "ToolCall",
    "ToolExecution",
    "ToolExecutionError",
    "ToolKind",
    "ToolNotFound",
    "ToolRegistry",
    "ToolSpec",
    "TurnFinish",
    "TurnOutcome",
    "TurnStart",
    "Usage",
    "UsageCounters",
    "UserMessage",
    "pretty_print",
]
