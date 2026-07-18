from luca.client import catalog, completion
from luca.client.types import (
    AssistantMessage,
    ChatCompletionResponse,
    ModelInfo,
    TextBlock,
    UserMessage,
)


RESP = ChatCompletionResponse(
    message=AssistantMessage(
        content=[TextBlock(text="ok")],
        finish_reason="stop", provider_finish_reason="stop",
        provider="stub", model="m",
    ),
)


def test_catalog_hit_attaches_model_info(stub_provider, monkeypatch):
    info = ModelInfo(provider="stub", model="m", context_window=128000)
    monkeypatch.setattr(
        catalog, "get",
        lambda provider, model: info if (provider, model) == ("stub", "m") else None,
    )
    stub_provider.configure(responses=[RESP])
    completion(model="stub:m", messages=[UserMessage(content="hi")])
    assert stub_provider.instances[0].calls[0].request.model_info == info


def test_catalog_miss_leaves_model_info_none(stub_provider, monkeypatch):
    monkeypatch.setattr(catalog, "get", lambda provider, model: None)
    stub_provider.configure(responses=[RESP])
    completion(model="stub:unknown", messages=[UserMessage(content="hi")])
    assert stub_provider.instances[0].calls[0].request.model_info is None


def test_model_info_kwarg_overrides_catalog(stub_provider, monkeypatch):
    catalog_info = ModelInfo(provider="stub", model="m", context_window=999)
    override = ModelInfo(provider="stub", model="m", context_window=42)
    monkeypatch.setattr(catalog, "get", lambda provider, model: catalog_info)
    stub_provider.configure(responses=[RESP])
    completion(
        model="stub:m", model_info=override,
        messages=[UserMessage(content="hi")],
    )
    assert stub_provider.instances[0].calls[0].request.model_info == override
