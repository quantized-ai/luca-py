"""total_timeout= on the async helpers: a wall-clock deadline over the whole
call (beside the per-phase httpx timeout=). Async-only — the sync helpers have
no loop to enforce a total deadline on. Expiry raises the SDK TimeoutError
(non-streaming) or yields exactly one terminal ErrorEvent carrying it
(streaming), per the streaming contract. The faux hang (`faux_hang()`) stands
in for a provider that stops sending."""

import pytest

from luca.client import acompletion, acompletion_stream
from luca.client.exceptions import TimeoutError as SDKTimeoutError
from luca.client.testing import (
    FauxProvider,
    faux_assistant_message,
    faux_hang,
    faux_text,
)
from luca.client.types import TextBlock, UserMessage


async def test_acompletion_total_timeout_raises_on_a_hung_call():
    faux = FauxProvider()
    faux.set_responses([faux_assistant_message([faux_hang()])])

    with pytest.raises(SDKTimeoutError):
        await acompletion(
            "faux:test-model", [UserMessage(content="hi")],
            provider=faux, total_timeout=0.05,
        )


async def test_acompletion_total_timeout_is_inert_on_an_instant_response():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message([faux_text("ok")], finish_reason="stop"),
    ])

    response = await acompletion(
        "faux:test-model", [UserMessage(content="hi")],
        provider=faux, total_timeout=60.0,
    )

    assert response.message.content == [TextBlock(text="ok")]
    assert response.finish_reason == "stop"


async def test_stream_total_timeout_emits_one_terminal_error_event():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message(
            [faux_text("Hel"), faux_text("lo"), faux_hang()],
            finish_reason="stop",
        ),
    ])
    stream = acompletion_stream(
        "faux:test-model", [UserMessage(content="hi")],
        provider=faux, total_timeout=0.05,
    )

    events = []
    async with stream as s:
        async for event in s:
            events.append(event)

    assert [e.delta for e in events if e.type == "text_delta"] == ["Hel", "lo"]
    terminals = [e for e in events if e.type in ("finish", "error")]
    assert len(terminals) == 1
    assert terminals[0].type == "error"
    assert isinstance(terminals[0].error, SDKTimeoutError)
    assert s._http_response is None  # closed, not leaked


async def test_stream_total_timeout_is_inert_on_an_instant_stream():
    faux = FauxProvider()
    faux.set_responses([
        faux_assistant_message([faux_text("Hello")], finish_reason="stop"),
    ])
    stream = acompletion_stream(
        "faux:test-model", [UserMessage(content="hi")],
        provider=faux, total_timeout=60.0,
    )

    events = []
    async with stream as s:
        async for event in s:
            events.append(event)

    assert events[-1].type == "finish"
    assert events[-1].finish_reason == "stop"
    assert not any(e.type == "error" for e in events)
