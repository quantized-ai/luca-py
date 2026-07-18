"""StubTransport — records calls, returns scripted responses. Used by
provider/helper tests that want a real BaseTransport-shaped object without
httpx involvement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from luca.client.types import ChatCompletionRequest


@dataclass
class TransportCall:
    method: str
    request: ChatCompletionRequest


@dataclass
class StubTransport:
    responses: list[Any] = field(default_factory=list)
    calls: list[TransportCall] = field(default_factory=list)
    _provider: str = "stub"
    _base_url: str = ""
    _api_key: str | None = None

    def _pop(self):
        if not self.responses:
            raise RuntimeError("StubTransport: no more scripted responses")
        return self.responses.pop(0)

    def completion(self, request):
        self.calls.append(TransportCall("completion", request))
        return self._pop()

    async def acompletion(self, request):
        self.calls.append(TransportCall("acompletion", request))
        return self._pop()

    def completion_stream(self, request):
        self.calls.append(TransportCall("completion_stream", request))
        return self._pop()

    def acompletion_stream(self, request):
        self.calls.append(TransportCall("acompletion_stream", request))
        return self._pop()

    def close(self):
        pass

    async def aclose(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass
