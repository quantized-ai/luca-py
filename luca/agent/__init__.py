"""luca.agent — agent framework.

The framework lives in `luca.agent.core`: the data model (`core.models`), the
`ToolRegistry` contract (`core.tool_registry`), the `Tool` base class, the
wire-projection `adapter`, the runtime `ToolContext`, the informational event
union (`core.events`), the system-prompt strategy, and the resumable
`AgentSessionRunner`. Optional packages built on that surface live in
`luca.agent.contrib`. Import from `luca.agent.core` (or a specific contrib
package); this package level intentionally exports nothing.
"""
