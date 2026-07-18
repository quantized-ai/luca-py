"""StubProvider — installed into PROVIDERS under name 'stub'."""

from dataclasses import dataclass, field

import pytest

from luca.client.providers import (
    PROVIDERS,
    BaseProvider,
    ChatCompletionMixin,
)
from luca.client.types import ChatCompletionRequest


@dataclass
class StubProviderCall:
    method: str
    request: ChatCompletionRequest


class StubProvider(BaseProvider, ChatCompletionMixin):
    name = "stub"
    default_base_url = "https://stub.test/v1"
    default_api_key_env_var = "STUB_API_KEY"

    instantiations: list[dict] = []
    instances: list = []
    _scripted_responses: list = []

    def __init__(self, **kwargs):
        StubProvider.instantiations.append(dict(kwargs))
        StubProvider.instances.append(self)
        self.calls: list[StubProviderCall] = []
        self._scripted = list(StubProvider._scripted_responses)
        self._transport = None

    @classmethod
    def configure(cls, *, responses):
        cls._scripted_responses = list(responses)

    @classmethod
    def reset(cls):
        cls.instantiations = []
        cls.instances = []
        cls._scripted_responses = []

    def _pop(self):
        if not self._scripted:
            raise RuntimeError("StubProvider: no more scripted responses")
        return self._scripted.pop(0)

    def completion(self, request):
        self.calls.append(StubProviderCall("completion", request))
        return self._pop()

    async def acompletion(self, request):
        self.calls.append(StubProviderCall("acompletion", request))
        return self._pop()

    def completion_stream(self, request):
        self.calls.append(StubProviderCall("completion_stream", request))
        return self._pop()

    def acompletion_stream(self, request):
        self.calls.append(StubProviderCall("acompletion_stream", request))
        return self._pop()

    # no-ops
    def close(self): pass
    async def aclose(self): pass


@pytest.fixture
def stub_provider(monkeypatch):
    StubProvider.reset()
    monkeypatch.setitem(PROVIDERS, "stub", StubProvider)
    yield StubProvider
    StubProvider.reset()
