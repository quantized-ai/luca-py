"""Per-model Anthropic facts, and the reasoning resolver built on them.

Which thinking shape a model takes is not discoverable from the wire and the
two shapes are mutually exclusive: adaptive models reject `budget_tokens`,
manual models reject `adaptive`, and the older models reject both. Sampling
support is a separate axis again — the newest models refuse `temperature`
outright, thinking or not.

That is three facts per model with no runtime probe available, so they live in
a table. Unknown ids resolve to a conservative all-false record rather than a
guess, and the caller decides what to do with `is_known_model`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ...exceptions import UnsupportedParameterError
from ...types.reasoning import Reasoning

# Fraction of a model's output budget spent on reasoning, per level.
BUDGET_PERCENTAGES: dict[str, float] = {
    "minimal": 0.02, "low": 0.1, "medium": 0.3, "high": 0.6, "xhigh": 0.9,
}
# Adaptive models take a word. `xhigh` degrades to `max` where unsupported.
ADAPTIVE_EFFORTS: dict[str, str] = {
    "minimal": "low", "low": "low", "medium": "medium",
    "high": "high", "xhigh": "xhigh",
}
MIN_THINKING_BUDGET = 1024
MIN_COMPLETION_TOKENS = 1024
UNKNOWN_MAX_OUTPUT_TOKENS = 4096


class ModelCapabilities(BaseModel):
    """What one Anthropic model can be asked to do.

    `extras` is deliberately open: the next capability we meet does not need a
    schema change to be carried."""

    max_output_tokens: int = UNKNOWN_MAX_OUTPUT_TOKENS
    supports_thinking: bool = False
    supports_adaptive_thinking: bool = False
    supports_xhigh_effort: bool = False
    rejects_sampling_parameters: bool = False
    is_known_model: bool = False
    extras: dict = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


# Ordered most specific first — `claude-sonnet-4-5` must be tested before any
# broader `claude-sonnet-4` style prefix would swallow it.
_CAPABILITY_TABLE: tuple[tuple[tuple[str, ...], ModelCapabilities], ...] = (
    (
        ("claude-opus-4-8", "claude-opus-4-7", "claude-fable-5", "claude-sonnet-5"),
        ModelCapabilities(
            max_output_tokens=128_000, supports_thinking=True,
            supports_adaptive_thinking=True, supports_xhigh_effort=True,
            rejects_sampling_parameters=True, is_known_model=True,
        ),
    ),
    (
        ("claude-sonnet-4-6", "claude-opus-4-6"),
        ModelCapabilities(
            max_output_tokens=128_000, supports_thinking=True,
            supports_adaptive_thinking=True, is_known_model=True,
        ),
    ),
    (
        ("claude-sonnet-4-5", "claude-opus-4-5", "claude-haiku-4-5"),
        ModelCapabilities(
            max_output_tokens=64_000, supports_thinking=True, is_known_model=True,
        ),
    ),
    (
        ("claude-opus-4-1", "claude-opus-4-0", "claude-sonnet-4-0"),
        ModelCapabilities(
            max_output_tokens=32_000, supports_thinking=True, is_known_model=True,
        ),
    ),
    (
        ("claude-3-5-sonnet", "claude-3-5-haiku", "claude-3-opus",
         "claude-3-sonnet", "claude-3-haiku"),
        ModelCapabilities(max_output_tokens=4_096, is_known_model=True),
    ),
)


def normalize_model_id(model: str) -> str:
    """A wire model id reduced to something the table can match.

    Handles a gateway prefix (`us.anthropic.claude-…`), a namespace
    (`anthropic/claude-…`) and a dotted version (`claude-sonnet-4.5`, the same
    model as `claude-sonnet-4-5`). Dated suffixes and `-latest` need no work
    because matching is by prefix."""
    normalized = model.rsplit("/", 1)[-1]
    anchor = normalized.find("claude")
    if anchor > 0:
        normalized = normalized[anchor:]
    return normalized.replace(".", "-")


def get_model_capabilities(model: str) -> ModelCapabilities:
    """Look up a model, falling back to a conservative unknown record.

    An unknown id gets no thinking and the smallest output budget rather than
    a guess: sending the wrong thinking shape is a hard 400, and a model we
    have never seen is as likely to be old as new."""
    normalized = normalize_model_id(model)
    for prefixes, capabilities in _CAPABILITY_TABLE:
        if normalized.startswith(prefixes):
            return capabilities
    return ModelCapabilities()


def resolve_reasoning(
    reasoning: Reasoning | None,
    capabilities: ModelCapabilities,
    max_tokens: int | None,
    *,
    display: str | None = None,
    model: str = "",
) -> tuple[dict, int]:
    """A reasoning level plus model facts → the thinking payload keys and the
    `max_tokens` to send.

    Returns `({}, max_tokens)` when no thinking should be requested at all, so
    a caller who never asked for reasoning keeps the plain wire shape."""
    resolved_max = max_tokens or (
        capabilities.max_output_tokens
        if capabilities.is_known_model
        else UNKNOWN_MAX_OUTPUT_TOKENS
    )

    if reasoning is None or reasoning == "provider-default":
        return {}, resolved_max
    if reasoning == "none" or not capabilities.supports_thinking:
        return {"thinking": {"type": "disabled"}}, resolved_max

    if capabilities.supports_adaptive_thinking:
        config: dict = {"type": "adaptive"}
        if display is not None:
            config["display"] = display
        effort = ADAPTIVE_EFFORTS[reasoning]
        if effort == "xhigh" and not capabilities.supports_xhigh_effort:
            effort = "max"
        return {"thinking": config, "output_config": {"effort": effort}}, resolved_max

    budget = round(capabilities.max_output_tokens * BUDGET_PERCENTAGES[reasoning])
    budget = min(max(budget, MIN_THINKING_BUDGET), capabilities.max_output_tokens)
    if max_tokens is None:
        resolved_max = min(
            budget + MIN_COMPLETION_TOKENS, capabilities.max_output_tokens,
        )
    else:
        # The caller's cap is a billing contract: shrink the budget to fit
        # inside it rather than quietly raising it.
        budget = min(budget, max_tokens - MIN_COMPLETION_TOKENS)
    if budget < MIN_THINKING_BUDGET:
        raise UnsupportedParameterError(
            f"max_tokens={max_tokens} leaves no room for extended thinking on "
            f"{model!r}: Anthropic requires a budget of at least "
            f"{MIN_THINKING_BUDGET} tokens below it.",
        )
    return {"thinking": {"type": "enabled", "budget_tokens": budget}}, resolved_max


def check_sampling(
    capabilities: ModelCapabilities,
    thinking: dict,
    *,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    model: str = "",
) -> None:
    """Refuse a sampling control the request cannot carry.

    Two independent reasons, and only the second depends on this request: the
    newest models reject `temperature` outright, and *any* model rejects it
    while thinking is active. Refused rather than stripped — a silently
    dropped temperature changes the output with nothing to notice."""
    conflicting = [
        name for name, value in (
            ("temperature", temperature), ("top_p", top_p), ("top_k", top_k),
        )
        if value is not None
    ]
    if not conflicting:
        return
    names = ", ".join(conflicting)
    if capabilities.rejects_sampling_parameters:
        raise UnsupportedParameterError(
            f"{names} cannot be set on {model!r}; the model does not accept "
            "sampling controls.",
        )
    if thinking.get("thinking", {}).get("type") not in (None, "disabled"):
        raise UnsupportedParameterError(
            f"{names} cannot be set while extended thinking is active on "
            f"{model!r}.",
        )
