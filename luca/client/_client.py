"""High-level helper functions: completion, acompletion, completion_stream,
acompletion_stream, get_provider.

Owns model-string parsing, catalog lookup, request DTO construction, provider
caching, and dispatch.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Literal

from . import catalog as _catalog_module
from .exceptions import BadRequestError
from .exceptions import TimeoutError as SDKTimeoutError
from .providers import BaseProvider, resolve_provider
from .types.completion import ChatCompletionRequest
from .types.messages import AssistantMessage, ToolMessage, UserMessage
from .types.tools import Tool

if TYPE_CHECKING:
    from .types.completion import ChatCompletionResponse
    from .types.streaming import AsyncChatCompletionStream, ChatCompletionStream


# Process-local provider cache: keyed by (name, api_key, base_url, transport_class, timeout).
# Never evicts. close_all() (not exposed in V1) can clear it for tests.
_provider_cache: dict[tuple, BaseProvider] = {}


def _parse_model_string(model: str, provider: str | None) -> tuple[str, str]:
    """Split `provider:author/model` at the first colon. `provider=` always wins."""
    if provider is not None:
        model_id = model.split(":", 1)[1] if ":" in model else model
        return provider, model_id
    if ":" in model:
        host, model_id = model.split(":", 1)
        return host, model_id
    raise ValueError(
        f"No provider specified. Use `provider=...` or prefixed model "
        f"(e.g. 'openai:{model}')."
    )


def _coerce_messages(messages: list) -> list:
    """Coerce dict-shape messages to typed Message subclasses.

    Raises BadRequestError on `role: "system"` — the system prompt is
    request-scoped and lives on `system_message=`.
    """
    out: list = []
    for i, m in enumerate(messages):
        if isinstance(m, (UserMessage, AssistantMessage, ToolMessage)):
            out.append(m)
            continue
        if not isinstance(m, dict):
            raise BadRequestError(
                f"messages[{i}] is {type(m).__name__}; expected dict or typed Message."
            )
        role = m.get("role")
        if role == "system":
            raise BadRequestError(
                "Found a message with role='system'. The system prompt is request-scoped: "
                "pass it as `system_message=` instead of including it in `messages`."
            )
        if role == "user":
            out.append(UserMessage.model_validate(m))
        elif role == "assistant":
            out.append(AssistantMessage.model_validate(m))
        elif role == "tool":
            out.append(ToolMessage.model_validate(m))
        else:
            raise BadRequestError(
                f"messages[{i}] has unknown role={role!r}; expected user / assistant / tool."
            )
    return out


def _coerce_tools(tools: list | None) -> list[Tool] | None:
    if tools is None:
        return None
    out: list[Tool] = []
    for t in tools:
        if isinstance(t, Tool):
            out.append(t)
        elif isinstance(t, dict):
            out.append(Tool.model_validate(t))
        else:
            raise BadRequestError(
                f"tool entry is {type(t).__name__}; expected dict or Tool."
            )
    return out


def _build_request(
    *,
    model: str,
    provider_name: str,
    messages: list,
    system_message: Any,
    tools: Any,
    tool_choice: Any,
    response_format: Any,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    max_tokens: int | None,
    stop: Any,
    seed: int | None,
    presence_penalty: float | None,
    frequency_penalty: float | None,
    logprobs: bool | None,
    top_logprobs: int | None,
    reasoning_effort: Any,
    thinking_budgets: dict | None,
    cache_retention: Any,
    session_id: str | None,
    parallel_tool_calls: bool | None,
    user: str | None,
    model_info: Any,
    metadata: dict | None,
    extra_args: dict | None,
) -> ChatCompletionRequest:
    coerced_messages = _coerce_messages(messages)
    coerced_tools = _coerce_tools(tools)

    if model_info is None:
        try:
            model_info = _catalog_module.get(provider_name, model)
        except Exception:
            model_info = None

    return ChatCompletionRequest(
        model=model,
        provider=provider_name,
        messages=coerced_messages,
        system_message=system_message,
        tools=coerced_tools,
        tool_choice=tool_choice,
        response_format=response_format,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        stop=stop,
        seed=seed,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        reasoning_effort=reasoning_effort,
        thinking_budgets=thinking_budgets,
        cache_retention=cache_retention,
        session_id=session_id,
        parallel_tool_calls=parallel_tool_calls,
        user=user,
        model_info=model_info,
        metadata=metadata,
        extra_args=extra_args,
    )


def _get_cached_provider(
    name: str,
    *,
    api_key: str | None,
    base_url: str | None,
    transport_class: type | None,
    timeout: float | None,
) -> BaseProvider:
    key = (name, api_key, base_url, transport_class, timeout)
    inst = _provider_cache.get(key)
    if inst is None:
        inst = resolve_provider(
            name,
            api_key=api_key, base_url=base_url,
            transport_class=transport_class, timeout=timeout,
        )
        _provider_cache[key] = inst
    return inst


def _resolve_for_call(
    *,
    model: str,
    provider: Any,
    transport: Any,
    transport_class: type | None,
    api_key: str | None,
    base_url: str | None,
    timeout: float | None,
) -> tuple[BaseProvider, str, str]:
    """Returns (provider_instance, provider_name, model_id)."""
    # Pre-built provider — caller drives the lifecycle entirely.
    if isinstance(provider, BaseProvider):
        provider_name = provider.name
        model_id = model.split(":", 1)[1] if ":" in model else model
        return provider, provider_name, model_id

    provider_name_or_none = provider if isinstance(provider, str) else None
    provider_name, model_id = _parse_model_string(model, provider_name_or_none)

    if transport is not None:
        # Caller passed a pre-built transport — wrap it in a generic provider.
        prov = resolve_provider(
            provider_name,
            transport=transport,
        )
        return prov, provider_name, model_id

    prov = _get_cached_provider(
        provider_name,
        api_key=api_key, base_url=base_url,
        transport_class=transport_class, timeout=timeout,
    )
    return prov, provider_name, model_id


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def completion(
    model: str,
    messages: list,
    *,
    provider: str | BaseProvider | None = None,
    transport: Any = None,
    transport_class: type | None = None,
    system_message: Any = None,
    tools: list | None = None,
    tool_choice: Any = None,
    response_format: Any = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    max_tokens: int | None = None,
    stop: Any = None,
    seed: int | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    logprobs: bool | None = None,
    top_logprobs: int | None = None,
    reasoning_effort: Any = None,
    thinking_budgets: dict | None = None,
    cache_retention: Any = None,
    session_id: str | None = None,
    parallel_tool_calls: bool | None = None,
    user: str | None = None,
    model_info: Any = None,
    metadata: dict | None = None,
    extra_args: dict | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
) -> "ChatCompletionResponse":
    prov, provider_name, model_id = _resolve_for_call(
        model=model, provider=provider, transport=transport,
        transport_class=transport_class, api_key=api_key,
        base_url=base_url, timeout=timeout,
    )
    request = _build_request(
        model=model_id, provider_name=provider_name,
        messages=messages, system_message=system_message,
        tools=tools, tool_choice=tool_choice, response_format=response_format,
        temperature=temperature, top_p=top_p, top_k=top_k,
        max_tokens=max_tokens, stop=stop, seed=seed,
        presence_penalty=presence_penalty, frequency_penalty=frequency_penalty,
        logprobs=logprobs, top_logprobs=top_logprobs,
        reasoning_effort=reasoning_effort, thinking_budgets=thinking_budgets,
        cache_retention=cache_retention, session_id=session_id,
        parallel_tool_calls=parallel_tool_calls, user=user,
        model_info=model_info, metadata=metadata, extra_args=extra_args,
    )
    return prov.completion(request)


async def acompletion(
    model: str,
    messages: list,
    *,
    provider: str | BaseProvider | None = None,
    transport: Any = None,
    transport_class: type | None = None,
    system_message: Any = None,
    tools: list | None = None,
    tool_choice: Any = None,
    response_format: Any = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    max_tokens: int | None = None,
    stop: Any = None,
    seed: int | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    logprobs: bool | None = None,
    top_logprobs: int | None = None,
    reasoning_effort: Any = None,
    thinking_budgets: dict | None = None,
    cache_retention: Any = None,
    session_id: str | None = None,
    parallel_tool_calls: bool | None = None,
    user: str | None = None,
    model_info: Any = None,
    metadata: dict | None = None,
    extra_args: dict | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    total_timeout: float | None = None,
) -> "ChatCompletionResponse":
    """Async completion. `timeout=` is the per-phase httpx timeout;
    `total_timeout=` is a wall-clock deadline over the whole call — expiry
    raises the SDK `TimeoutError`. Async-only: the sync `completion` has no
    loop to enforce a total deadline on."""
    prov, provider_name, model_id = _resolve_for_call(
        model=model, provider=provider, transport=transport,
        transport_class=transport_class, api_key=api_key,
        base_url=base_url, timeout=timeout,
    )
    request = _build_request(
        model=model_id, provider_name=provider_name,
        messages=messages, system_message=system_message,
        tools=tools, tool_choice=tool_choice, response_format=response_format,
        temperature=temperature, top_p=top_p, top_k=top_k,
        max_tokens=max_tokens, stop=stop, seed=seed,
        presence_penalty=presence_penalty, frequency_penalty=frequency_penalty,
        logprobs=logprobs, top_logprobs=top_logprobs,
        reasoning_effort=reasoning_effort, thinking_budgets=thinking_budgets,
        cache_retention=cache_retention, session_id=session_id,
        parallel_tool_calls=parallel_tool_calls, user=user,
        model_info=model_info, metadata=metadata, extra_args=extra_args,
    )
    if total_timeout is None:
        return await prov.acompletion(request)
    try:
        async with asyncio.timeout(total_timeout):
            return await prov.acompletion(request)
    except TimeoutError as exc:  # the builtin, raised by asyncio.timeout
        raise SDKTimeoutError(
            f"completion exceeded total_timeout={total_timeout}s",
            provider=provider_name,
            original_exception=exc,
        ) from exc


def completion_stream(
    model: str,
    messages: list,
    *,
    provider: str | BaseProvider | None = None,
    transport: Any = None,
    transport_class: type | None = None,
    system_message: Any = None,
    tools: list | None = None,
    tool_choice: Any = None,
    response_format: Any = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    max_tokens: int | None = None,
    stop: Any = None,
    seed: int | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    logprobs: bool | None = None,
    top_logprobs: int | None = None,
    reasoning_effort: Any = None,
    thinking_budgets: dict | None = None,
    cache_retention: Any = None,
    session_id: str | None = None,
    parallel_tool_calls: bool | None = None,
    user: str | None = None,
    model_info: Any = None,
    metadata: dict | None = None,
    extra_args: dict | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
) -> "ChatCompletionStream":
    prov, provider_name, model_id = _resolve_for_call(
        model=model, provider=provider, transport=transport,
        transport_class=transport_class, api_key=api_key,
        base_url=base_url, timeout=timeout,
    )
    request = _build_request(
        model=model_id, provider_name=provider_name,
        messages=messages, system_message=system_message,
        tools=tools, tool_choice=tool_choice, response_format=response_format,
        temperature=temperature, top_p=top_p, top_k=top_k,
        max_tokens=max_tokens, stop=stop, seed=seed,
        presence_penalty=presence_penalty, frequency_penalty=frequency_penalty,
        logprobs=logprobs, top_logprobs=top_logprobs,
        reasoning_effort=reasoning_effort, thinking_budgets=thinking_budgets,
        cache_retention=cache_retention, session_id=session_id,
        parallel_tool_calls=parallel_tool_calls, user=user,
        model_info=model_info, metadata=metadata, extra_args=extra_args,
    )
    return prov.completion_stream(request)


def acompletion_stream(
    model: str,
    messages: list,
    *,
    provider: str | BaseProvider | None = None,
    transport: Any = None,
    transport_class: type | None = None,
    system_message: Any = None,
    tools: list | None = None,
    tool_choice: Any = None,
    response_format: Any = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    max_tokens: int | None = None,
    stop: Any = None,
    seed: int | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    logprobs: bool | None = None,
    top_logprobs: int | None = None,
    reasoning_effort: Any = None,
    thinking_budgets: dict | None = None,
    cache_retention: Any = None,
    session_id: str | None = None,
    parallel_tool_calls: bool | None = None,
    user: str | None = None,
    model_info: Any = None,
    metadata: dict | None = None,
    extra_args: dict | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    total_timeout: float | None = None,
) -> "AsyncChatCompletionStream":
    # NOTE: this is a regular `def`, not `async def`. The function returns
    # AsyncChatCompletionStream synchronously; HTTP fires on first iteration.
    # `total_timeout=` arms a wall-clock deadline on the stream object
    # (recorded at open, enforced on every chunk pull): expiry follows the
    # streaming contract — exactly one terminal ErrorEvent carrying the SDK
    # TimeoutError, then close. Async-only; `completion_stream` has no loop
    # to enforce a total deadline on.
    prov, provider_name, model_id = _resolve_for_call(
        model=model, provider=provider, transport=transport,
        transport_class=transport_class, api_key=api_key,
        base_url=base_url, timeout=timeout,
    )
    request = _build_request(
        model=model_id, provider_name=provider_name,
        messages=messages, system_message=system_message,
        tools=tools, tool_choice=tool_choice, response_format=response_format,
        temperature=temperature, top_p=top_p, top_k=top_k,
        max_tokens=max_tokens, stop=stop, seed=seed,
        presence_penalty=presence_penalty, frequency_penalty=frequency_penalty,
        logprobs=logprobs, top_logprobs=top_logprobs,
        reasoning_effort=reasoning_effort, thinking_budgets=thinking_budgets,
        cache_retention=cache_retention, session_id=session_id,
        parallel_tool_calls=parallel_tool_calls, user=user,
        model_info=model_info, metadata=metadata, extra_args=extra_args,
    )
    stream = prov.acompletion_stream(request)
    if total_timeout is not None:
        stream._set_total_timeout(total_timeout)
    return stream


def get_provider(model_or_pair: str) -> BaseProvider:
    """Returns the cached provider instance configured for the given model
    string (`provider:model`)."""
    provider_name, _ = _parse_model_string(model_or_pair, None)
    return _get_cached_provider(
        provider_name,
        api_key=None, base_url=None, transport_class=None, timeout=None,
    )
