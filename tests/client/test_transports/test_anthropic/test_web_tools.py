"""Anthropic's web_search / web_fetch server tools: declarations, response
parsing into private + synthetic blocks, adjacent-text merging with citation
ranges, replay, and usage. Wire shapes mirror the live captures from
2026-08-15/19 (specs/0011-websearch-native-tools/notebooks/)."""

import pytest
from pydantic import ValidationError

from luca.client.transports.anthropic.native_tools import WebFetchTool, WebSearchTool
from luca.client.types import (
    ApproximateLocation,
    AssistantMessage,
    ChatCompletionRequest,
    PrivateProviderBlock,
    TextBlock,
    ThinkingBlock,
    URLCitationAnnotation,
    Usage,
    UserMessage,
    WebFetchBlock,
    WebPagePart,
    WebSearchBlock,
)

APPLE_URL = "https://www.apple.com/newsroom/2026/07/apple-reports-third-quarter-results/"


def _request(**kwargs) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="claude-sonnet-5",
        provider="anthropic",
        messages=[UserMessage(content="hi")],
        **kwargs,
    )


# --- declarations ----------------------------------------------------------


def test_the_minimal_search_declaration(anthropic_transport_factory):
    transport = anthropic_transport_factory()

    payload = transport._build_chat_completion_payload(_request(tools=[WebSearchTool()]))

    assert payload["tools"] == [{"type": "web_search_20260318", "name": "web_search"}]


def test_every_search_option_projects_under_its_wire_name(anthropic_transport_factory):
    transport = anthropic_transport_factory()
    tool = WebSearchTool(
        max_uses=5,
        allowed_domains=["apple.com"],
        user_location=ApproximateLocation(city="Cordoba", country="AR"),
        allowed_callers=["direct", "code_execution_20260120"],
        response_inclusion="excluded",
    )

    payload = transport._build_chat_completion_payload(_request(tools=[tool]))

    assert payload["tools"] == [
        {
            "type": "web_search_20260318",
            "name": "web_search",
            "max_uses": 5,
            "allowed_domains": ["apple.com"],
            "allowed_callers": ["direct", "code_execution_20260120"],
            "response_inclusion": "excluded",
            "user_location": {"type": "approximate", "city": "Cordoba", "country": "AR"},
        }
    ]


def test_the_fetch_declaration_nests_the_citations_flag(anthropic_transport_factory):
    transport = anthropic_transport_factory()
    tool = WebFetchTool(max_uses=3, max_content_tokens=20000, citations=True, use_cache=False)

    payload = transport._build_chat_completion_payload(_request(tools=[tool]))

    assert payload["tools"] == [
        {
            "type": "web_fetch_20260318",
            "name": "web_fetch",
            "max_uses": 3,
            "max_content_tokens": 20000,
            "citations": {"enabled": True},
            "use_cache": False,
        }
    ]


def test_allowed_and_blocked_domains_are_mutually_exclusive():
    with pytest.raises(ValidationError, match="mutually exclusive"):
        WebSearchTool(allowed_domains=["a.com"], blocked_domains=["b.com"])
    with pytest.raises(ValidationError, match="mutually exclusive"):
        WebFetchTool(allowed_domains=["a.com"], blocked_domains=["b.com"])


# --- parsing: server tool blocks -------------------------------------------

SEARCH_CALL = {
    "type": "server_tool_use",
    "id": "srvtoolu_search_01",
    "name": "web_search",
    "input": {"query": "Apple latest quarterly results"},
}

SEARCH_RESULT = {
    "type": "web_search_tool_result",
    "tool_use_id": "srvtoolu_search_01",
    "content": [
        {
            "type": "web_search_result",
            "title": "Apple reports third quarter results",
            "url": APPLE_URL,
            "encrypted_content": "EncryptedOpaque...",
            "page_age": "2 weeks ago",
        }
    ],
    "caller": {"type": "direct"},
}


def test_a_search_becomes_two_private_blocks_and_one_projection(anthropic_transport_factory):
    transport = anthropic_transport_factory()
    data = {
        "id": "msg_1",
        "content": [SEARCH_CALL, SEARCH_RESULT, {"type": "text", "text": "Apple's latest quarterly report says..."}],
    }

    message = transport._parse_assistant_message(data, _request())

    assert message.content == [
        PrivateProviderBlock(format="anthropic.messages", data=SEARCH_CALL),
        PrivateProviderBlock(format="anthropic.messages", data=SEARCH_RESULT),
        WebSearchBlock(
            queries=["Apple latest quarterly results"],
            results=[WebPagePart(url=APPLE_URL, title="Apple reports third quarter results")],
            extras={"id": "srvtoolu_search_01"},
        ),
        TextBlock(text="Apple's latest quarterly report says..."),
    ]


def test_parallel_searches_pair_results_by_tool_use_id(anthropic_transport_factory):
    # Batched calls stream call-call-result-result; each result pairs with
    # ITS call, not the adjacent block.
    transport = anthropic_transport_factory()
    call_a = {"type": "server_tool_use", "id": "s_a", "name": "web_search", "input": {"query": "apple"}}
    call_b = {"type": "server_tool_use", "id": "s_b", "name": "web_search", "input": {"query": "nvidia"}}
    result_a = {"type": "web_search_tool_result", "tool_use_id": "s_a", "content": [], "caller": {"type": "direct"}}
    result_b = {"type": "web_search_tool_result", "tool_use_id": "s_b", "content": [], "caller": {"type": "direct"}}

    message = transport._parse_assistant_message(
        {"id": "msg_1", "content": [call_a, call_b, result_a, result_b]},
        _request(),
    )

    assert message.content == [
        PrivateProviderBlock(format="anthropic.messages", data=call_a),
        PrivateProviderBlock(format="anthropic.messages", data=call_b),
        PrivateProviderBlock(format="anthropic.messages", data=result_a),
        WebSearchBlock(queries=["apple"], results=[], extras={"id": "s_a"}),
        PrivateProviderBlock(format="anthropic.messages", data=result_b),
        WebSearchBlock(queries=["nvidia"], results=[], extras={"id": "s_b"}),
    ]


def test_an_errored_search_keeps_queries_and_no_results(anthropic_transport_factory):
    transport = anthropic_transport_factory()
    result = {
        "type": "web_search_tool_result",
        "tool_use_id": "srvtoolu_search_01",
        "content": {"type": "web_search_tool_result_error", "error_code": "max_uses_exceeded"},
    }

    message = transport._parse_assistant_message({"id": "m", "content": [SEARCH_CALL, result]}, _request())

    assert message.content == [
        PrivateProviderBlock(format="anthropic.messages", data=SEARCH_CALL),
        PrivateProviderBlock(format="anthropic.messages", data=result),
        WebSearchBlock(
            queries=["Apple latest quarterly results"],
            results=None,
            extras={
                "id": "srvtoolu_search_01",
                "error": {"type": "web_search_tool_result_error", "error_code": "max_uses_exceeded"},
            },
        ),
    ]


FETCH_CALL = {
    "type": "server_tool_use",
    "id": "srvtoolu_fetch_01",
    "name": "web_fetch",
    "input": {"url": APPLE_URL},
}

FETCH_RESULT = {
    "type": "web_fetch_tool_result",
    "tool_use_id": "srvtoolu_fetch_01",
    "content": {
        "type": "web_fetch_result",
        "url": APPLE_URL,
        "retrieved_at": "2026-08-15T03:41:54.963000+00:00",
        "content": {
            "type": "document",
            "source": {
                "type": "text",
                "media_type": "text/plain",
                "data": "Apple today announced financial results...",
            },
            "title": "Apple reports third quarter results",
        },
    },
}


def test_a_fetch_carries_the_documents_readable_text(anthropic_transport_factory):
    transport = anthropic_transport_factory()

    message = transport._parse_assistant_message({"id": "m", "content": [FETCH_CALL, FETCH_RESULT]}, _request())

    assert message.content == [
        PrivateProviderBlock(format="anthropic.messages", data=FETCH_CALL),
        PrivateProviderBlock(format="anthropic.messages", data=FETCH_RESULT),
        WebFetchBlock(
            web_page=WebPagePart(
                url=APPLE_URL,
                title="Apple reports third quarter results",
                content="Apple today announced financial results...",
            ),
            extras={"id": "srvtoolu_fetch_01"},
        ),
    ]


def test_an_errored_fetch_keeps_only_the_requested_url(anthropic_transport_factory):
    transport = anthropic_transport_factory()
    result = {
        "type": "web_fetch_tool_result",
        "tool_use_id": "srvtoolu_fetch_01",
        "content": {"type": "web_fetch_tool_result_error", "error_code": "url_not_in_prior_context"},
    }

    message = transport._parse_assistant_message({"id": "m", "content": [FETCH_CALL, result]}, _request())

    assert message.content == [
        PrivateProviderBlock(format="anthropic.messages", data=FETCH_CALL),
        PrivateProviderBlock(format="anthropic.messages", data=result),
        WebFetchBlock(
            web_page=WebPagePart(url=APPLE_URL),
            extras={
                "id": "srvtoolu_fetch_01",
                "error": {"type": "web_fetch_tool_result_error", "error_code": "url_not_in_prior_context"},
            },
        ),
    ]


def test_a_code_execution_caller_adds_private_blocks_only(anthropic_transport_factory):
    # Dynamic filtering wraps the search in code_execution: the parent call
    # and its result are kept for exact replay, with no synthetic block; the
    # nested search projects exactly as a direct one.
    transport = anthropic_transport_factory()
    code_call = {
        "type": "server_tool_use",
        "id": "srvtoolu_code_01",
        "name": "code_execution",
        "input": {"code": "await web_search(...)"},
    }
    caller = {"type": "code_execution_20260120", "tool_id": "srvtoolu_code_01"}
    nested_call = {**SEARCH_CALL, "caller": caller}
    nested_result = {**SEARCH_RESULT, "caller": caller}
    code_result = {
        "type": "code_execution_tool_result",
        "tool_use_id": "srvtoolu_code_01",
        "content": {"type": "code_execution_result", "stdout": "ok", "stderr": "", "return_code": 0, "content": []},
    }

    message = transport._parse_assistant_message(
        {"id": "m", "content": [code_call, nested_call, nested_result, code_result]},
        _request(),
    )

    assert message.content == [
        PrivateProviderBlock(format="anthropic.messages", data=code_call),
        PrivateProviderBlock(format="anthropic.messages", data=nested_call),
        PrivateProviderBlock(format="anthropic.messages", data=nested_result),
        WebSearchBlock(
            queries=["Apple latest quarterly results"],
            results=[WebPagePart(url=APPLE_URL, title="Apple reports third quarter results")],
            extras={"id": "srvtoolu_search_01"},
        ),
        PrivateProviderBlock(format="anthropic.messages", data=code_result),
    ]


# --- parsing: adjacent text blocks merge, citations become ranges ----------


CITED_RUN = [
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
                "encrypted_index": "opaque",
            }
        ],
    },
    {"type": "text", "text": " NVIDIA was flat."},
]


def test_adjacent_text_blocks_merge_with_citation_ranges(anthropic_transport_factory):
    # The merged block is the portable view; the run's wire blocks are kept
    # verbatim after it because their citations carry provider-only fields
    # (cited_text, encrypted_index) Anthropic wants back on later turns.
    transport = anthropic_transport_factory()

    message = transport._parse_assistant_message({"id": "msg_1", "content": CITED_RUN}, _request())

    assert message.content == [
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
        ),
        PrivateProviderBlock(format="anthropic.messages", data=CITED_RUN[0]),
        PrivateProviderBlock(format="anthropic.messages", data=CITED_RUN[1]),
        PrivateProviderBlock(format="anthropic.messages", data=CITED_RUN[2]),
    ]


def test_multiple_citations_on_one_block_share_the_derived_range(anthropic_transport_factory):
    transport = anthropic_transport_factory()
    data = {
        "id": "msg_1",
        "content": [
            {
                "type": "text",
                "text": "Apple rose.",
                "citations": [
                    {"type": "web_search_result_location", "url": "https://a", "title": "A", "cited_text": "x"},
                    {"type": "web_search_result_location", "url": "https://b", "title": "B", "cited_text": "y"},
                ],
            },
        ],
    }

    message = transport._parse_assistant_message(data, _request())

    assert message.content[0] == TextBlock(
        text="Apple rose.",
        annotations=[
            URLCitationAnnotation(url="https://a", title="A", start_index=0, end_index=11),
            URLCitationAnnotation(url="https://b", title="B", start_index=0, end_index=11),
        ],
    )


def test_a_cited_answer_replays_exactly_as_it_arrived(anthropic_transport_factory):
    # Parse then project: the wire content round-trips verbatim — the merged
    # TextBlock is skipped because its privates replay the split spans,
    # citations and all.
    transport = anthropic_transport_factory()

    message = transport._parse_assistant_message({"id": "msg_1", "content": CITED_RUN}, _request())
    message.provider, message.model = "anthropic", "claude-sonnet-5"
    projected = transport._project_assistant_message(message, _request())

    assert projected == {"role": "assistant", "content": CITED_RUN}


def test_an_uncited_run_keeps_no_private_blocks(anthropic_transport_factory):
    # Merging plain text is lossless, so there is nothing to preserve.
    transport = anthropic_transport_factory()
    data = {"id": "msg_1", "content": [{"type": "text", "text": "Hello "}, {"type": "text", "text": "there."}]}

    message = transport._parse_assistant_message(data, _request())

    assert message.content == [TextBlock(text="Hello there.")]


def test_a_non_text_block_breaks_the_merge(anthropic_transport_factory):
    transport = anthropic_transport_factory()
    data = {
        "id": "msg_1",
        "content": [
            {"type": "text", "text": "Searching."},
            SEARCH_CALL,
            SEARCH_RESULT,
            {"type": "text", "text": "Found it."},
        ],
    }

    message = transport._parse_assistant_message(data, _request())

    assert message.content == [
        TextBlock(text="Searching."),
        PrivateProviderBlock(format="anthropic.messages", data=SEARCH_CALL),
        PrivateProviderBlock(format="anthropic.messages", data=SEARCH_RESULT),
        WebSearchBlock(
            queries=["Apple latest quarterly results"],
            results=[WebPagePart(url=APPLE_URL, title="Apple reports third quarter results")],
            extras={"id": "srvtoolu_search_01"},
        ),
        TextBlock(text="Found it."),
    ]


def test_a_urlless_citation_gets_no_annotation_but_stays_private(anthropic_transport_factory):
    # A document citation has no canonical (URL) shape, so no annotation —
    # but the cited wire block is still preserved for exact replay.
    transport = anthropic_transport_factory()
    cited = {
        "type": "text",
        "text": "From the document.",
        "citations": [{"type": "char_location", "cited_text": "x", "document_index": 0}],
    }

    message = transport._parse_assistant_message({"id": "msg_1", "content": [cited]}, _request())

    assert message.content == [
        TextBlock(text="From the document."),
        PrivateProviderBlock(format="anthropic.messages", data=cited),
    ]


# --- replay ----------------------------------------------------------------


def test_replay_sends_own_private_blocks_verbatim_and_nothing_else(anthropic_transport_factory):
    transport = anthropic_transport_factory()
    message = AssistantMessage(
        content=[
            ThinkingBlock(text="hm", signature="sig-1"),
            PrivateProviderBlock(format="anthropic.messages", data=SEARCH_CALL),
            PrivateProviderBlock(format="anthropic.messages", data=SEARCH_RESULT),
            WebSearchBlock(queries=["Apple latest quarterly results"]),
            PrivateProviderBlock(format="openai.responses", data={"type": "web_search_call", "id": "ws_1"}),
            TextBlock(
                text="Cited answer.",
                annotations=[URLCitationAnnotation(url="https://x", title="X", start_index=0, end_index=5)],
            ),
        ],
        provider="anthropic",
        model="claude-sonnet-5",
    )

    projected = transport._project_assistant_message(message, _request())

    # The foreign-format private block and the synthetic block are omitted;
    # the merged text replays plain (annotations are canonical-only).
    assert projected == {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "hm", "signature": "sig-1"},
            SEARCH_CALL,
            SEARCH_RESULT,
            {"type": "text", "text": "Cited answer."},
        ],
    }


# --- usage -----------------------------------------------------------------


def test_usage_normalizes_the_server_tool_counters(anthropic_transport_factory):
    transport = anthropic_transport_factory()

    usage = transport._parse_usage(
        {
            "input_tokens": 100,
            "output_tokens": 50,
            "output_tokens_details": {"thinking_tokens": 11},
            "server_tool_use": {"web_search_requests": 2, "web_fetch_requests": 0},
        },
        None,
    )

    assert usage == Usage(
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        reasoning_tokens=11,
        tool_requests={"web_search": 2, "web_fetch": 0},
        provider_tool_usage={"web_search_requests": 2, "web_fetch_requests": 0},
    )


# --- extras stamping: the adjacency-free facts ------------------------------


def test_a_search_result_stamps_the_operation_id(anthropic_transport_factory):
    # the op id rides the portable block itself, so a consumer never needs the
    # adjacent private block; WebPagePart.extras stay untouched
    transport = anthropic_transport_factory()

    message = transport._parse_assistant_message({"id": "m", "content": [SEARCH_CALL, SEARCH_RESULT]}, _request())

    assert message.content[2] == WebSearchBlock(
        queries=["Apple latest quarterly results"],
        results=[WebPagePart(url=APPLE_URL, title="Apple reports third quarter results")],
        extras={"id": "srvtoolu_search_01"},
    )


def test_a_failed_search_stamps_the_provider_error(anthropic_transport_factory):
    # the wire error dict verbatim — `results=None` plus `extras["error"]`
    # present now means "failed", vs "metadata not returned"
    transport = anthropic_transport_factory()
    result = {
        "type": "web_search_tool_result",
        "tool_use_id": "srvtoolu_search_01",
        "content": {"type": "web_search_tool_result_error", "error_code": "max_uses_exceeded"},
    }

    message = transport._parse_assistant_message({"id": "m", "content": [SEARCH_CALL, result]}, _request())

    assert message.content[2] == WebSearchBlock(
        queries=["Apple latest quarterly results"],
        results=None,
        extras={
            "id": "srvtoolu_search_01",
            "error": {"type": "web_search_tool_result_error", "error_code": "max_uses_exceeded"},
        },
    )


def test_a_failed_fetch_stamps_the_provider_error(anthropic_transport_factory):
    transport = anthropic_transport_factory()
    result = {
        "type": "web_fetch_tool_result",
        "tool_use_id": "srvtoolu_fetch_01",
        "content": {"type": "web_fetch_tool_result_error", "error_code": "url_not_in_prior_context"},
    }

    message = transport._parse_assistant_message({"id": "m", "content": [FETCH_CALL, result]}, _request())

    assert message.content[2] == WebFetchBlock(
        web_page=WebPagePart(url=APPLE_URL),
        extras={
            "id": "srvtoolu_fetch_01",
            "error": {"type": "web_fetch_tool_result_error", "error_code": "url_not_in_prior_context"},
        },
    )
