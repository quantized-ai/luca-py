"""The two D9 tool specs the synthetic web executions resolve to.

One spec per OPERATION KIND, provider-independent: the definition ("a hosted
web search: queries → results") does not depend on which provider ran it —
that is provenance, call-scoped, living on the execution and the linked
entry's `llm_config`. Exactly the split `spec_id()`'s purity rule wants;
`metadata` stays empty for the same reason.

`is_private=True` is required by the minting door (no `ToolCall` block for a
hosted operation exists on the path, so private is the only wire-legal
shape); `is_synthetic=True` keeps the executions out of the unseen-material
and doom-loop predicates — they record work the model has already seen.
"""

from __future__ import annotations

from luca.agent.core import ToolKind, ToolSpec, WebFetchContent, WebSearchContent

WEB_SEARCH_SPEC = ToolSpec(
    name="web_search",
    namespace="web",
    version="1",
    tool_kind=ToolKind.WEB,
    is_private=True,
    is_synthetic=True,
    timeout_in_ms=None,  # never dispatched
    description=(
        "Provider-hosted web search, recorded for posterity by the websearch "
        "plugin. Never advertised and never dispatched: executions are "
        "synthesized from response content; the exact wire items live in the "
        "adjacent assistant message's private blocks."
    ),
    # `queries` is plural on both providers (Anthropic = a list of one), so
    # the synthetic call's argument shape is uniform.
    input_schema={
        "type": "object",
        "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
        "required": ["queries"],
    },
    output_schema=WebSearchContent.model_json_schema(),  # = structured_content
    metadata={},  # stays empty — definition-pure
)

WEB_FETCH_SPEC = ToolSpec(
    name="web_fetch",
    namespace="web",
    version="1",
    tool_kind=ToolKind.WEB,
    is_private=True,
    is_synthetic=True,
    timeout_in_ms=None,
    description=(
        "Provider-hosted page fetch (OpenAI open_page / Anthropic web_fetch), "
        "recorded for posterity by the websearch plugin. Never advertised and "
        "never dispatched: executions are synthesized from response content; "
        "the exact wire items live in the adjacent assistant message's "
        "private blocks."
    ),
    input_schema={
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
    },
    output_schema=WebFetchContent.model_json_schema(),
    metadata={},
)
