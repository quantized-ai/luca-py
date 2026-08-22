"""OpenAI's hosted web_search tool on the Responses wire: declaration,
request-level include merging, response parsing into private + synthetic
blocks, citation annotations, replay, and usage. Wire shapes mirror the live
captures from 2026-08-15/19 (specs/0011-websearch-native-tools/notebooks/)."""

from typing import ClassVar

import pytest

from luca.client.exceptions import BadRequestError
from luca.client.transports.openai_responses.native_tools import (
    WebSearchFilters,
    WebSearchImageSettings,
    WebSearchTool,
)
from luca.client.transports.openai_responses.transport import OpenAIResponsesToolProjector
from luca.client.types import (
    ApproximateLocation,
    AssistantMessage,
    BaseTool,
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
        model="gpt-5.1",
        provider="openai",
        messages=[UserMessage(content="hi")],
        **kwargs,
    )


# --- declaration -----------------------------------------------------------


def test_the_minimal_declaration_is_the_bare_type(responses_transport_factory):
    transport = responses_transport_factory()

    payload = transport._build_chat_completion_payload(_request(tools=[WebSearchTool()]))

    assert payload["tools"] == [{"type": "web_search"}]
    assert "include" not in payload


def test_every_option_projects_under_its_wire_name(responses_transport_factory):
    transport = responses_transport_factory()
    tool = WebSearchTool(
        search_context_size="high",
        filters=WebSearchFilters(allowed_domains=["apple.com"]),
        user_location=ApproximateLocation(city="Cordoba", country="AR"),
        external_web_access=True,
        return_token_budget="unlimited",
        search_content_types=["text", "image"],
        image_settings=WebSearchImageSettings(max_results=3, caption=True),
    )

    payload = transport._build_chat_completion_payload(_request(tools=[tool]))

    assert payload["tools"] == [
        {
            "type": "web_search",
            "search_context_size": "high",
            "filters": {"allowed_domains": ["apple.com"]},
            "user_location": {"type": "approximate", "city": "Cordoba", "country": "AR"},
            "external_web_access": True,
            "return_token_budget": "unlimited",
            "search_content_types": ["text", "image"],
            "image_settings": {"max_results": 3, "caption": True},
        }
    ]


def test_the_inclusion_flags_land_on_the_request_include_list(responses_transport_factory):
    transport = responses_transport_factory()
    tool = WebSearchTool(include_sources=True, include_results=True)

    payload = transport._build_chat_completion_payload(_request(tools=[tool]))

    assert payload["include"] == [
        "web_search_call.action.sources",
        "web_search_call.results",
    ]


def test_the_includes_merge_with_the_reasoning_include(responses_transport_factory):
    transport = responses_transport_factory()
    tool = WebSearchTool(include_results=True)

    payload = transport._build_chat_completion_payload(_request(tools=[tool], reasoning="low"))

    assert payload["include"] == [
        "reasoning.encrypted_content",
        "web_search_call.results",
    ]


def test_two_web_search_tools_deduplicate_their_includes(responses_transport_factory):
    transport = responses_transport_factory()
    tools = [WebSearchTool(include_sources=True), WebSearchTool(include_sources=True, include_results=True)]

    payload = transport._build_chat_completion_payload(_request(tools=tools))

    assert payload["include"] == [
        "web_search_call.action.sources",
        "web_search_call.results",
    ]


class _ClobberProjector(OpenAIResponsesToolProjector):
    def project_tool_to_llm(self, tool: BaseTool) -> dict:
        return {"type": "clobber"}

    def project_request_params(self, tool: BaseTool) -> dict:
        return {"store": True}


class _ClobberTool(BaseTool):
    name: ClassVar[str] = "clobber"

    def get_projector(self) -> _ClobberProjector:
        return _ClobberProjector()


def test_a_conflicting_request_param_is_refused(responses_transport_factory):
    transport = responses_transport_factory()

    # The payload always carries store=False; a tool cannot silently flip it.
    with pytest.raises(BadRequestError, match="'store'"):
        transport._build_chat_completion_payload(_request(tools=[_ClobberTool()]))


def test_tool_choice_forces_by_type(responses_transport_factory):
    transport = responses_transport_factory()

    payload = transport._build_chat_completion_payload(
        _request(tools=[WebSearchTool()], tool_choice={"name": "web_search"}),
    )

    assert payload["tool_choice"] == {"type": "web_search"}


# --- parsing: web_search_call items ----------------------------------------

SEARCH_ITEM = {
    "id": "ws_01",
    "type": "web_search_call",
    "status": "completed",
    "action": {
        "type": "search",
        "queries": ["Apple latest quarterly results"],
        "query": "Apple latest quarterly results",
        "sources": [
            {"type": "url", "url": APPLE_URL},
            {"type": "url", "url": "https://ph.investing.com/equities/apple"},
            {"type": "api", "name": "oai-time"},
        ],
    },
    "results": [
        {
            "type": "text_result",
            "title": "Apple reports third quarter results",
            "url": APPLE_URL,
            "snippet": "Apple today announced financial results...",
        }
    ],
}


def test_a_search_becomes_the_private_block_and_its_projection(responses_transport_factory):
    transport = responses_transport_factory()
    data = {"id": "resp_1", "output": [SEARCH_ITEM]}

    message = transport._parse_assistant_message(data, _request())

    # Results come first (they carry metadata); the uncovered URL source is
    # appended bare, and the non-url source is dropped.
    assert message.content == [
        PrivateProviderBlock(format="openai.responses", data=SEARCH_ITEM),
        WebSearchBlock(
            queries=["Apple latest quarterly results"],
            results=[
                WebPagePart(
                    url=APPLE_URL,
                    title="Apple reports third quarter results",
                    content="Apple today announced financial results...",
                ),
                WebPagePart(url="https://ph.investing.com/equities/apple"),
            ],
            extras={"id": "ws_01"},
        ),
    ]


def test_a_search_without_includes_has_no_result_metadata(responses_transport_factory):
    transport = responses_transport_factory()
    item = {
        "id": "ws_02",
        "type": "web_search_call",
        "status": "completed",
        "action": {"type": "search", "queries": ["q1", "q2"], "query": "q1"},
    }

    message = transport._parse_assistant_message({"id": "r", "output": [item]}, _request())

    assert message.content == [
        PrivateProviderBlock(format="openai.responses", data=item),
        WebSearchBlock(queries=["q1", "q2"], results=None, extras={"id": "ws_02"}),
    ]


def test_a_sources_only_search_keeps_the_bare_urls(responses_transport_factory):
    # include_sources without include_results: no `results` key at all, URL
    # sources inside the action.
    transport = responses_transport_factory()
    item = {
        "id": "ws_07",
        "type": "web_search_call",
        "status": "completed",
        "action": {
            "type": "search",
            "queries": ["apple"],
            "query": "apple",
            "sources": [{"type": "url", "url": "https://a"}, {"type": "url", "url": "https://b"}],
        },
    }

    message = transport._parse_assistant_message({"id": "r", "output": [item]}, _request())

    assert message.content == [
        PrivateProviderBlock(format="openai.responses", data=item),
        WebSearchBlock(
            queries=["apple"],
            results=[WebPagePart(url="https://a"), WebPagePart(url="https://b")],
            extras={"id": "ws_07"},
        ),
    ]


def test_an_explicitly_empty_result_set_stays_empty(responses_transport_factory):
    transport = responses_transport_factory()
    item = {
        "id": "ws_03",
        "type": "web_search_call",
        "status": "completed",
        "action": {"type": "search", "queries": ["nothing"], "query": "nothing"},
        "results": [],
    }

    message = transport._parse_assistant_message({"id": "r", "output": [item]}, _request())

    assert message.content == [
        PrivateProviderBlock(format="openai.responses", data=item),
        WebSearchBlock(queries=["nothing"], results=[], extras={"id": "ws_03"}),
    ]


def test_an_open_page_becomes_a_fetch_with_the_result_page(responses_transport_factory):
    transport = responses_transport_factory()
    item = {
        "id": "ws_04",
        "type": "web_search_call",
        "status": "completed",
        "action": {"type": "open_page", "url": "https://investor.apple.com/investor-relations/"},
        "results": [
            {
                "type": "text_result",
                # A redirect: the fetched page differs from the action URL.
                "title": "Contact - Apple",
                "url": "https://investor.apple.com/contact/default.aspx",
                "snippet": "Crawled: today",
            }
        ],
    }

    message = transport._parse_assistant_message({"id": "r", "output": [item]}, _request())

    assert message.content == [
        PrivateProviderBlock(format="openai.responses", data=item),
        WebFetchBlock(
            web_page=WebPagePart(
                url="https://investor.apple.com/contact/default.aspx",
                title="Contact - Apple",
                content="Crawled: today",
            ),
            extras={"id": "ws_04"},
        ),
    ]


def test_an_open_page_without_results_keeps_the_action_url(responses_transport_factory):
    transport = responses_transport_factory()
    item = {
        "id": "ws_05",
        "type": "web_search_call",
        "status": "completed",
        "action": {"type": "open_page", "url": APPLE_URL},
    }

    message = transport._parse_assistant_message({"id": "r", "output": [item]}, _request())

    assert message.content == [
        PrivateProviderBlock(format="openai.responses", data=item),
        WebFetchBlock(web_page=WebPagePart(url=APPLE_URL), extras={"id": "ws_05"}),
    ]


def test_a_find_in_page_keeps_only_the_private_block(responses_transport_factory):
    transport = responses_transport_factory()
    item = {
        "id": "ws_06",
        "type": "web_search_call",
        "status": "completed",
        "action": {"type": "find_in_page", "pattern": "research and development", "url": APPLE_URL},
    }

    message = transport._parse_assistant_message({"id": "r", "output": [item]}, _request())

    assert message.content == [PrivateProviderBlock(format="openai.responses", data=item)]


def test_text_annotations_copy_their_ranges_verbatim(responses_transport_factory):
    transport = responses_transport_factory()
    data = {
        "id": "resp_2",
        "output": [
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Market update: Apple rose 2.8% today. NVIDIA was flat.",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url": "https://example.com/apple",
                                "title": "Apple shares rise",
                                "start_index": 15,
                                "end_index": 37,
                            },
                            {"type": "file_citation", "file_id": "file_1"},
                        ],
                    }
                ],
            }
        ],
    }

    message = transport._parse_assistant_message(data, _request())

    # The file citation has no canonical shape and drops.
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
        )
    ]


# --- replay ----------------------------------------------------------------

HISTORY = AssistantMessage(
    content=[
        PrivateProviderBlock(format="openai.responses", data=SEARCH_ITEM),
        PrivateProviderBlock(format="anthropic.messages", data={"type": "server_tool_use", "id": "s1"}),
        WebSearchBlock(queries=["Apple latest quarterly results"]),
        WebFetchBlock(web_page=WebPagePart(url=APPLE_URL)),
        TextBlock(text="Apple's latest quarterly report says..."),
    ],
    provider="openai",
    model="gpt-5.1",
)


def test_replay_sends_own_private_blocks_verbatim_and_nothing_else(responses_transport_factory):
    transport = responses_transport_factory()

    items = transport._project_assistant_message(HISTORY, _request())

    # The foreign-format private block and both synthetic blocks are omitted.
    assert items == [
        SEARCH_ITEM,
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Apple's latest quarterly report says..."}],
        },
    ]


def test_an_in_progress_stub_from_a_cancelled_stream_is_not_replayed(responses_transport_factory):
    # A stream cancelled between item-added and item-done leaves the stub
    # private block behind; replaying an incomplete item would 400.
    transport = responses_transport_factory()
    message = AssistantMessage(
        content=[
            PrivateProviderBlock(
                format="openai.responses",
                data={"id": "ws_1", "type": "web_search_call", "status": "in_progress"},
            ),
            TextBlock(text="partial"),
        ],
        provider="openai",
        model="gpt-5.1",
    )

    items = transport._project_assistant_message(message, _request())

    assert items == [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "partial"}],
        }
    ]


# --- usage -----------------------------------------------------------------


def test_a_full_completion_round_trip_carries_blocks_and_usage(responses_transport_factory):
    from tests.client._helpers.httpx_mocks import json_response, make_sync_client

    search = {
        "id": "ws_1",
        "type": "web_search_call",
        "status": "completed",
        "action": {"type": "search", "queries": ["apple"], "query": "apple"},
    }
    payload = {
        "id": "resp_9",
        "model": "gpt-5.1",
        "status": "completed",
        "output": [
            search,
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Apple is up.",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url": "https://apple.com",
                                "title": "Apple",
                                "start_index": 0,
                                "end_index": 12,
                            }
                        ],
                    }
                ],
            },
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        "tool_usage": {"web_search": {"num_requests": 1}},
    }
    transport = responses_transport_factory(http_client=make_sync_client(json_response(payload)))

    response = transport.completion(_request())

    assert response.messages == [
        AssistantMessage(
            content=[
                PrivateProviderBlock(format="openai.responses", data=search),
                WebSearchBlock(queries=["apple"], results=None, extras={"id": "ws_1"}),
                TextBlock(
                    text="Apple is up.",
                    annotations=[
                        URLCitationAnnotation(url="https://apple.com", title="Apple", start_index=0, end_index=12)
                    ],
                ),
            ],
            finish_reason="stop",
            provider_finish_reason="completed",
            provider="openai",
            model="gpt-5.1",
            response_model="gpt-5.1",
            response_id="resp_9",
            usage=Usage(
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
                tool_requests={"web_search": 1},
                provider_tool_usage={"web_search": {"num_requests": 1}},
            ),
        )
    ]


def test_usage_normalizes_the_tool_request_counters(responses_transport_factory):
    transport = responses_transport_factory()
    tool_usage = {
        "image_gen": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "web_search": {"num_requests": 3},
    }

    usage = transport._parse_usage(
        {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        None,
        tool_usage,
    )

    assert usage == Usage(
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        tool_requests={"web_search": 3},
        provider_tool_usage=tool_usage,
    )


def test_reasoning_survives_beside_a_search(responses_transport_factory):
    # The full parse: reasoning item, search call, cited answer — one pass.
    transport = responses_transport_factory()
    search = {
        "id": "ws_1",
        "type": "web_search_call",
        "status": "completed",
        "action": {"type": "search", "queries": ["apple"], "query": "apple"},
    }
    data = {
        "id": "resp_3",
        "output": [
            {"id": "rs_1", "type": "reasoning", "summary": [], "encrypted_content": "enc-1"},
            search,
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Done.", "annotations": []}],
            },
        ],
    }

    message = transport._parse_assistant_message(data, _request())

    assert message.content == [
        ThinkingBlock(text="", id="rs_1", signature="enc-1"),
        PrivateProviderBlock(format="openai.responses", data=search),
        WebSearchBlock(queries=["apple"], results=None, extras={"id": "ws_1"}),
        TextBlock(text="Done."),
    ]


def test_a_failed_search_stamps_the_status(responses_transport_factory):
    # OpenAI has NO structured per-operation error on the wire; a failed item
    # carries only `status: "failed"`, stamped as the whole error signal.
    # Fixture invented (minimal shape), not captured live — no failed
    # web_search_call has been observed yet.
    transport = responses_transport_factory()
    item = {
        "id": "ws_08",
        "type": "web_search_call",
        "status": "failed",
        "action": {"type": "search", "queries": ["apple"], "query": "apple"},
    }

    message = transport._parse_assistant_message({"id": "r", "output": [item]}, _request())

    assert message.content == [
        PrivateProviderBlock(format="openai.responses", data=item),
        WebSearchBlock(
            queries=["apple"],
            results=None,
            extras={"id": "ws_08", "error": {"status": "failed"}},
        ),
    ]
