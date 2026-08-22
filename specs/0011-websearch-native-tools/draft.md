# Native web tools — design checkpoint

## Context

We want to add provider-native web search and web fetch tools to `luca.client`, initially for OpenAI and Anthropic.

The user-facing web blocks should remain provider-agnostic:

```python
web_blocks = [
    block
    for block in response.messages[-1].content
    if isinstance(block, (WebSearchBlock, WebFetchBlock))
]

web_blocks == [
    WebSearchBlock(...),
    WebFetchBlock(...),
]
```

The same web operation has a different wire shape on each provider. An OpenAI search is one `web_search_call` containing its action and any requested results. An Anthropic search is at least two blocks: a `server_tool_use` call followed by a `web_search_tool_result` linked through `tool_use_id`. The examples below show those structures and their exact Luca projection.

Luca conversations must also remain portable. A conversation may move from Anthropic to OpenAI, or between OpenAI Responses and Chat Completions, and later switch back. We therefore need both:

1. A small canonical representation that users and higher-level consumers can understand independently of the provider.
2. The exact provider blocks required to replay the conversation through the wire format that produced them.

The chosen design stores both views in the ordered assistant content:

```python
AssistantMessage.content = [
    PrivateProviderBlock(...),  # exact wire data
    WebSearchBlock(...),        # canonical user-facing projection
    TextBlock(...),
]
```

The private blocks preserve exact replay and are ignored by other transports. The synthetic blocks provide the portable meaning and are ignored by client transports on replay. Higher-level consumers such as `luca.agent` may use the synthetic blocks to adapt history when switching providers without changing the client's deliberately simple behavior.

Streaming follows the same principle: providers translate their detailed wire events into a small set of direct, user-facing events suitable for status UIs. Events describe facts such as “searching,” “found these results,” or “opening these URLs”; callers never inspect a nested provider-style action. Results and URLs remain batched because both providers commonly deliver them together and callers should not need to process one event per item.

## Response data model

Every web operation is stored as two adjacent views:

1. `PrivateProviderBlock` contains each original provider item, unchanged and in wire order.
2. A synthetic `WebSearchBlock` or `WebFetchBlock` contains the small, portable meaning Luca can expose everywhere.

```python
class PrivateProviderBlock(BaseModel):
    type: Literal["private_provider"] = "private_provider"
    format: str  # "openai.responses" or "anthropic.messages"
    data: dict[str, Any]

    model_config = ConfigDict(extra="forbid")


class WebPagePart(BaseModel):
    type: Literal["web_page"] = "web_page"
    url: str
    title: str | None = None
    content: str | None = None
    extras: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class WebSearchBlock(BaseModel):
    type: Literal["web_search"] = "web_search"
    queries: list[str]
    results: list[WebPagePart] | None = None
    extras: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class WebFetchBlock(BaseModel):
    type: Literal["web_fetch"] = "web_fetch"
    web_page: WebPagePart
    extras: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class URLCitationAnnotation(BaseModel):
    type: Literal["url_citation"] = "url_citation"
    url: str
    title: str
    start_index: int | None = None
    end_index: int | None = None

    model_config = ConfigDict(extra="forbid")


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str
    signature: str | None = None
    annotations: list[URLCitationAnnotation] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")
```

`WebPagePart.content` is the readable content the provider exposed: normally a search snippet for a result and page text for a fetch. Encrypted payloads, provider timestamps and other non-portable fields remain in the private block.

### Citations on answer text

OpenAI returns citations as character ranges over one text value:

```json
{
  "type": "output_text",
  "text": "Market update: Apple rose 2.8% today. NVIDIA was flat.",
  "annotations": [
    {
      "type": "url_citation",
      "url": "https://example.com/apple",
      "title": "Apple shares rise",
      "start_index": 15,
      "end_index": 37
    }
  ]
}
```

The OpenAI transport copies that range directly:

```python
TextBlock(
    text="Market update: Apple rose 2.8% today. NVIDIA was flat.",
    annotations=[
        URLCitationAnnotation(
            url="https://example.com/apple",
            title="Apple shares rise",
            start_index=15,
            end_index=37,
        )
    ],
)
```

Anthropic instead expresses the range by splitting the answer into text blocks and attaching citations to the supported block:

```json
[
  {"type": "text", "text": "Market update: "},
  {
    "type": "text",
    "text": "Apple rose 2.8% today.",
    "citations": [
      {
        "type": "web_search_result_location",
        "url": "https://example.com/apple",
        "title": "Apple shares rise",
        "cited_text": "Apple shares rose 2.8%...",
        "encrypted_index": "..."
      }
    ]
  },
  {"type": "text", "text": " NVIDIA was flat."}
]
```

The Anthropic transport concatenates adjacent text blocks. For each citation, its start is the cumulative text length before the containing block and its end is that start plus the block's length. The example therefore produces the same generic block as OpenAI: `start_index=15` and `end_index=37`. Multiple citations attached to one Anthropic block receive the same range. Provider-only fields such as `cited_text` and `encrypted_index` remain in the corresponding `PrivateProviderBlock`.

Indexes are nullable because not every provider or transport can supply or safely derive a range.

### Example 1: searching for a page

Suppose the model searches for Apple's latest quarterly results and finds the Apple Newsroom page.

#### OpenAI

OpenAI returns one item containing the search action and, when requested, its sources and results. Luca retains that whole item and adds one canonical search block:

```python
AssistantMessage(
    content=[
        PrivateProviderBlock(
            format="openai.responses",
            data={
                "id": "ws_01",
                "type": "web_search_call",
                "status": "completed",
                "action": {
                    "type": "search",
                    "query": "Apple latest quarterly results",
                    "queries": ["Apple latest quarterly results"],
                    "sources": [
                        {
                            "type": "url",
                            "url": "https://www.apple.com/newsroom/2026/07/apple-reports-third-quarter-results/",
                        }
                    ],
                },
                "results": [
                    {
                        "type": "text_result",
                        "title": "Apple reports third quarter results",
                        "url": "https://www.apple.com/newsroom/2026/07/apple-reports-third-quarter-results/",
                        "snippet": "Apple today announced financial results for its fiscal 2026 third quarter...",
                    }
                ],
            },
        ),
        WebSearchBlock(
            queries=["Apple latest quarterly results"],
            results=[
                WebPagePart(
                    title="Apple reports third quarter results",
                    url="https://www.apple.com/newsroom/2026/07/apple-reports-third-quarter-results/",
                    content="Apple today announced financial results for its fiscal 2026 third quarter...",
                )
            ],
        ),
        TextBlock(text="Apple's latest quarterly report says..."),
    ],
)
```

OpenAI's `sources` and `results` are different provider concepts. Luca merges URL sources and result metadata by URL into the generic `results` list. If only sources were requested, the same page would still be a `WebPagePart`, but `title` and `content` may be `None`. The untouched private block preserves whether the page came from `sources`, `results`, or both.

#### Anthropic

With direct invocation, Anthropic returns a call block and a separate result block linked by ID. Luca retains both and produces the same synthetic structure:

```python
AssistantMessage(
    content=[
        PrivateProviderBlock(
            format="anthropic.messages",
            data={
                "type": "server_tool_use",
                "id": "srvtoolu_search_01",
                "name": "web_search",
                "input": {"query": "Apple latest quarterly results"},
            },
        ),
        PrivateProviderBlock(
            format="anthropic.messages",
            data={
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_search_01",
                "content": [
                    {
                        "type": "web_search_result",
                        "title": "Apple reports third quarter results",
                        "url": "https://www.apple.com/newsroom/2026/07/apple-reports-third-quarter-results/",
                        "encrypted_content": "...",
                        "page_age": "2 weeks ago",
                    }
                ],
            },
        ),
        WebSearchBlock(
            queries=["Apple latest quarterly results"],
            results=[
                WebPagePart(
                    title="Apple reports third quarter results",
                    url="https://www.apple.com/newsroom/2026/07/apple-reports-third-quarter-results/",
                )
            ],
        ),
        TextBlock(text="Apple's latest quarterly report says..."),
    ],
)
```

Anthropic's dynamic-filtering mode may add a parent `code_execution` block and `caller` references to these two blocks. Those fields become additional private data; the `WebSearchBlock` remains identical.

### Example 2: opening the selected page

OpenAI calls this action `open_page`. Anthropic exposes a separate `web_fetch` tool. Both become `WebFetchBlock`.

#### OpenAI

```python
AssistantMessage(
    content=[
        PrivateProviderBlock(
            format="openai.responses",
            data={
                "id": "ws_02",
                "type": "web_search_call",
                "status": "completed",
                "action": {
                    "type": "open_page",
                    "url": "https://www.apple.com/newsroom/2026/07/apple-reports-third-quarter-results/",
                },
                "results": [
                    {
                        "type": "text_result",
                        "title": "Apple reports third quarter results",
                        "url": "https://www.apple.com/newsroom/2026/07/apple-reports-third-quarter-results/",
                        "snippet": "Apple today announced financial results for its fiscal 2026 third quarter...",
                    }
                ],
            },
        ),
        WebFetchBlock(
            web_page=WebPagePart(
                title="Apple reports third quarter results",
                url="https://www.apple.com/newsroom/2026/07/apple-reports-third-quarter-results/",
                content="Apple today announced financial results for its fiscal 2026 third quarter...",
            )
        ),
    ],
)
```

#### Anthropic

```python
AssistantMessage(
    content=[
        PrivateProviderBlock(
            format="anthropic.messages",
            data={
                "type": "server_tool_use",
                "id": "srvtoolu_fetch_01",
                "name": "web_fetch",
                "input": {
                    "url": "https://www.apple.com/newsroom/2026/07/apple-reports-third-quarter-results/"
                },
            },
        ),
        PrivateProviderBlock(
            format="anthropic.messages",
            data={
                "type": "web_fetch_tool_result",
                "tool_use_id": "srvtoolu_fetch_01",
                "content": {
                    "type": "web_fetch_result",
                    "url": "https://www.apple.com/newsroom/2026/07/apple-reports-third-quarter-results/",
                    "retrieved_at": "2026-08-15T03:41:54.963000+00:00",
                    "content": {
                        "type": "document",
                        "source": {
                            "type": "text",
                            "media_type": "text/plain",
                            "data": "Apple today announced financial results for its fiscal 2026 third quarter...",
                        },
                        "title": "Apple reports third quarter results",
                    },
                },
            },
        ),
        WebFetchBlock(
            web_page=WebPagePart(
                title="Apple reports third quarter results",
                url="https://www.apple.com/newsroom/2026/07/apple-reports-third-quarter-results/",
                content="Apple today announced financial results for its fiscal 2026 third quarter...",
            )
        ),
    ],
)
```

The examples shorten long snippets and opaque encrypted values only for readability. `PrivateProviderBlock.data` stores the complete provider object without truncation.

Rules:

- A synthetic block immediately follows the private block or contiguous private-block sequence it summarizes.
- Private blocks are authoritative for replay by their originating wire format. Other transports omit formats they do not understand.
- Synthetic blocks are portable observations and are never replayed by the client as provider messages.
- `results=None` means result metadata was not returned, usually because it was not requested. `results=[]` means the provider explicitly returned an empty result set.
- Switching transports never removes private blocks, so switching back preserves exact replay.
- Higher layers such as the Agent may adapt synthetic blocks during projection, for example into a user message when switching providers.

## Tool definitions

Tool declarations remain provider-specific because the available controls and behavior differ. They share only the location value object:

```python
class ApproximateLocation(BaseModel):
    # At least one field must be provided.
    city: str | None = None
    region: str | None = None
    country: str | None = None   # ISO 3166-1 alpha-2
    timezone: str | None = None  # IANA timezone
```

### OpenAI

OpenAI exposes one hosted tool covering search, page opening, find-in-page and image search:

```python
class WebSearchFilters(BaseModel):
    allowed_domains: list[str] | None = None  # Maximum 100
    blocked_domains: list[str] | None = None  # Maximum 100


class WebSearchImageSettings(BaseModel):
    max_results: PositiveInt | None = None
    caption: bool | None = None


# luca.client.providers.openai.WebSearchTool
class OpenAIWebSearchTool(BaseTool):
    name: ClassVar[str] = "web_search"

    search_context_size: Literal["low", "medium", "high"] | None = None
    filters: WebSearchFilters | None = None
    user_location: ApproximateLocation | None = None

    external_web_access: bool | None = None
    return_token_budget: Literal["default", "unlimited"] | None = None

    search_content_types: list[Literal["text", "image"]] | None = None
    image_settings: WebSearchImageSettings | None = None

    include_sources: bool = False
    include_results: bool = False
```

The transport injects `{"type": "web_search"}`. The two inclusion flags are logically tool options even though OpenAI projects them into the request-level `include` list:

```python
include_sources -> "web_search_call.action.sources"
include_results -> "web_search_call.results"
```

The OpenAI Responses projector exposes a generic request-parameter contribution. For now, web search is the only native tool that uses it:

```python
class OpenAIResponsesToolProjector(ToolProjector):
    def project_request_params(
        self,
        tool: BaseTool,
    ) -> dict[str, Any]:
        return {}


class WebSearchProjector(OpenAIResponsesToolProjector):
    def project_request_params(
        self,
        tool: OpenAIWebSearchTool,
    ) -> dict[str, Any]:
        includes = []

        if tool.include_sources:
            includes.append("web_search_call.action.sources")
        if tool.include_results:
            includes.append("web_search_call.results")

        return {"include": includes} if includes else {}
```

`OpenAIResponsesWireMixin._build_chat_completion_payload()` resolves each tool projector once, projects its declaration, collects its request parameters and merges them into the final payload. Lists are combined and deduplicated, identical values are accepted, and conflicting scalar values raise `BadRequestError`; a tool cannot silently replace core fields such as `model` or `input`. This also merges the web-search includes with `reasoning.encrypted_content` when reasoning is enabled.

Nothing is included automatically. This hook remains specific to `OpenAIResponsesToolProjector` until another transport has a concrete need for request-level tool parameters.

### Anthropic

Anthropic exposes search and fetch as separate server tools:

```python
AnthropicWebCaller = Literal[
    "direct",
    "code_execution_20260120",
]

ResponseInclusion = Literal["full", "excluded"]


# luca.client.providers.anthropic.WebSearchTool
class AnthropicWebSearchTool(BaseTool):
    name: ClassVar[str] = "web_search"

    max_uses: PositiveInt | None = None

    # Mutually exclusive.
    allowed_domains: list[str] | None = None
    blocked_domains: list[str] | None = None

    user_location: ApproximateLocation | None = None
    allowed_callers: list[AnthropicWebCaller] | None = None
    response_inclusion: ResponseInclusion | None = None


# luca.client.providers.anthropic.WebFetchTool
class AnthropicWebFetchTool(BaseTool):
    name: ClassVar[str] = "web_fetch"

    max_uses: PositiveInt | None = None

    # Mutually exclusive.
    allowed_domains: list[str] | None = None
    blocked_domains: list[str] | None = None

    max_content_tokens: PositiveInt | None = None
    citations: bool | None = None
    use_cache: bool | None = None

    allowed_callers: list[AnthropicWebCaller] | None = None
    response_inclusion: ResponseInclusion | None = None
```

The transport supplies the fixed wire fields and maps the simple `citations` boolean to Anthropic's nested representation:

```python
AnthropicWebSearchTool -> {"type": "web_search_20260318", "name": "web_search"}
AnthropicWebFetchTool  -> {"type": "web_fetch_20260318", "name": "web_fetch"}

citations=True -> {"citations": {"enabled": True}}
```

All optional parameters are omitted from the wire when unset, preserving provider defaults. Inclusion settings only determine how much provider data is available: they never select a different canonical block or event model. Missing search results remain `results=None`, and no result event is emitted when the provider sends no results.

## Usage

Both providers report the number of requests made by hosted web tools. Normalize that common fact directly on `Usage`, while retaining the exact provider payload for inspection:

```python
class Usage(BaseModel):
    # Existing token and cost fields...

    tool_requests: dict[str, int] = Field(default_factory=dict)
    provider_tool_usage: dict[str, Any] = Field(default_factory=dict)
```

```python
# OpenAI
Usage(
    tool_requests={"web_search": 3},
    provider_tool_usage=raw["tool_usage"],
)

# Anthropic
Usage(
    tool_requests={"web_search": 2, "web_fetch": 0},
    provider_tool_usage=raw["usage"]["server_tool_use"],
)
```

```python
searches = usage.tool_requests.get("web_search", 0)
provider_details = usage.provider_tool_usage
```

Usage is metadata, not conversation content, so it does not use `PrivateProviderBlock`. The same normalized `Usage` object is exposed on the final assistant message and streaming `UsageEvent`.

## Streaming events

Events are direct user-facing facts. Consumers dispatch only on the event type; there is no nested action union. Text citations use one generic event alongside the web-operation events:

```python
class TextAnnotationEvent(BaseModel):
    type: Literal["text_annotation"] = "text_annotation"
    index: int
    annotation: URLCitationAnnotation


class WebStartEvent(BaseModel):
    type: Literal["web_start"] = "web_start"
    id: str


class WebSearchEvent(BaseModel):
    type: Literal["web_search"] = "web_search"
    id: str
    queries: list[str]


class WebSearchResultEvent(BaseModel):
    type: Literal["web_search_result"] = "web_search_result"
    id: str
    results: list[WebPagePart]


class WebFetchEvent(BaseModel):
    type: Literal["web_fetch"] = "web_fetch"
    id: str
    urls: list[str]


class WebFindEvent(BaseModel):
    type: Literal["web_find"] = "web_find"
    id: str
    url: str
    pattern: str


class WebEndEvent(BaseModel):
    type: Literal["web_end"] = "web_end"
    id: str
```

OpenAI streams one text value and then sends the complete annotation with its exact range:

```python
response.output_text.delta(
    delta="Market update: Apple rose 2.8% today."
)
response.output_text.annotation.added(
    annotation={
        "type": "url_citation",
        "url": "https://example.com/apple",
        "title": "Apple shares rise",
        "start_index": 15,
        "end_index": 37,
    }
)
```

The transport immediately emits:

```python
TextAnnotationEvent(
    index=text_index,
    annotation=URLCitationAnnotation(
        url="https://example.com/apple",
        title="Apple shares rise",
        start_index=15,
        end_index=37,
    ),
)
```

Anthropic sends a complete citation before streaming the text block it supports. This is an actual sequence from `anthropic_basic_streaming_lines_2.txt` (long strings were shortened by the capture script):

```text
event: content_block_start
{
  "type": "content_block_start",
  "index": 14,
  "content_block": {"citations": [], "type": "text", "text": ""}
}

event: content_block_delta
{
  "type": "content_block_delta",
  "index": 14,
  "delta": {
    "type": "citations_delta",
    "citation": {
      "type": "web_search_result_location",
      "cited_text": "Apple (NASDAQ:AAPL), the iPhone giant sitting near",
      "url": "https://finance.yahoo.com/markets/stocks/articles/",
      "title": "Apple Stock Jumps 2.8% as Investors Rebuild Mega-C",
      "encrypted_index": "EpABCioIEhgCIiRmMDQ1ZTA1Yy02MjA3LTQ0YTEtOTcxMi1lY2"
    }
  }
}

event: content_block_delta
{
  "type": "content_block_delta",
  "index": 14,
  "delta": {
    "type": "text_delta",
    "text": "jumped approximately 2.8% to $318.60 Wednesday mor"
  }
}

event: content_block_delta
{
  "type": "content_block_delta",
  "index": 14,
  "delta": {"type": "text_delta", "text": " Tuesday's selloff"}
}

event: content_block_stop
{"type": "content_block_stop", "index": 14}
```

The projection follows those exact boundaries:

```python
start = len(text_accumulated_before_block_14)

cited_text = (
    "jumped approximately 2.8% to $318.60 Wednesday mor"
    " Tuesday's selloff"
)
end = start + len(cited_text)

TextAnnotationEvent(
    index=text_index,
    annotation=URLCitationAnnotation(
        url="https://finance.yahoo.com/markets/stocks/articles/",
        title="Apple Stock Jumps 2.8% as Investors Rebuild Mega-C",
        start_index=start,
        end_index=end,
    ),
)
```

The citation itself is complete when `citations_delta` arrives. Luca holds it only until `content_block_stop`, when the cited text and therefore `end_index` are known. Several citations received before the same block's text share that derived range.

Typical UI handling:

```python
match event:
    case WebStartEvent():
        show_status("Using the web…")
    case WebSearchEvent(queries=queries):
        show_status(f"Searching: {', '.join(queries)}")
    case WebSearchResultEvent(results=results):
        show_found_results(results)
    case WebFetchEvent(urls=urls):
        show_status(f"Opening: {', '.join(urls)}")
    case WebFindEvent(url=url, pattern=pattern):
        show_status(f"Searching {url} for {pattern!r}")
    case WebEndEvent():
        clear_status()
```

Provider batches stay batched: one `WebSearchResultEvent` carries all results received together, and one `WebFetchEvent` may carry several URLs.

The completed canonical blocks are available on the final `AssistantMessage`; `WebEndEvent` does not duplicate them.
