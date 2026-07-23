"""Per-model Bedrock facts, and the reasoning resolver built on them.

This mirrors `transports/anthropic/capabilities.py`. It is kept separate rather
than shared so each transport stays self-contained; the two differ in the model
table and, more importantly, in `normalize_model_id` — Bedrock ids carry a
region prefix and a version suffix that Anthropic's do not.

Nova and Llama get `supports_thinking=False`: that flag is why a model with no
reasoning mode never has a thinking block forced onto it.

The Anthropic-on-Bedrock rows are written from the docs and the Anthropic
capability tiers. They are UNVERIFIED: every Anthropic model on the test
account is gated behind a use-case submission, so the thinking and sampling
paths for those rows have not been exercised live. Nova and Llama are verified.
"""

from __future__ import annotations

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

# Inference-profile region prefixes (`us.anthropic.…`, `eu.amazon.…`). Stripped
# before matching so a profile id and a bare model id resolve the same.
_REGION_PREFIXES = ("us.", "eu.", "apac.", "us-gov.", "ap.", "ca.", "sa.")


class ModelCapabilities(BaseModel):
    """What one Bedrock model can be asked to do.

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


# Ordered most specific first — `anthropic.claude-sonnet-4-5` must be tested
# before any broader `anthropic.claude-sonnet-4` style prefix would swallow it.
# Anthropic rows are unverified (see module docstring); Nova and Llama are live.
_CAPABILITY_TABLE: tuple[tuple[tuple[str, ...], ModelCapabilities], ...] = (
    (
        ("anthropic.claude-opus-4-8", "anthropic.claude-opus-4-7",
         "anthropic.claude-sonnet-5"),
        ModelCapabilities(
            max_output_tokens=128_000, supports_thinking=True,
            supports_adaptive_thinking=True, supports_xhigh_effort=True,
            rejects_sampling_parameters=True, is_known_model=True,
        ),
    ),
    (
        ("anthropic.claude-sonnet-4-6", "anthropic.claude-opus-4-6"),
        ModelCapabilities(
            max_output_tokens=128_000, supports_thinking=True,
            supports_adaptive_thinking=True, is_known_model=True,
        ),
    ),
    (
        ("anthropic.claude-sonnet-4-5", "anthropic.claude-opus-4-5",
         "anthropic.claude-haiku-4-5"),
        ModelCapabilities(
            max_output_tokens=64_000, supports_thinking=True, is_known_model=True,
        ),
    ),
    (
        ("anthropic.claude-opus-4-1", "anthropic.claude-opus-4-0",
         "anthropic.claude-sonnet-4-0"),
        ModelCapabilities(
            max_output_tokens=32_000, supports_thinking=True, is_known_model=True,
        ),
    ),
    (
        ("anthropic.claude-3-7-sonnet",),
        ModelCapabilities(
            max_output_tokens=64_000, supports_thinking=True, is_known_model=True,
        ),
    ),
    (
        ("anthropic.claude-3-5-sonnet", "anthropic.claude-3-5-haiku",
         "anthropic.claude-3-opus", "anthropic.claude-3-sonnet",
         "anthropic.claude-3-haiku"),
        ModelCapabilities(max_output_tokens=4_096, is_known_model=True),
    ),
    # Amazon Nova — verified live: 10000-token ceiling, no reasoning mode.
    (
        ("amazon.nova-premier", "amazon.nova-pro", "amazon.nova-lite",
         "amazon.nova-micro"),
        ModelCapabilities(max_output_tokens=10_000, is_known_model=True),
    ),
    # Meta Llama — verified live: 8192-token ceiling, no reasoning mode.
    (
        ("meta.llama",),
        ModelCapabilities(max_output_tokens=8_192, is_known_model=True),
    ),
)


def normalize_model_id(model: str) -> str:
    """A Bedrock model id reduced to something the table can match.

    Strips a gateway namespace (`bedrock/us.anthropic.claude-…`) and the
    inference-profile region prefix (`us.`, `eu.`, …), leaving the vendor id
    (`anthropic.claude-sonnet-4-5-20250929-v1:0`). The dated suffix and the
    `-v1:0` version need no work because matching is by prefix."""
    normalized = model.rsplit("/", 1)[-1]
    for prefix in _REGION_PREFIXES:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break
    return normalized


def get_model_capabilities(model: str) -> ModelCapabilities:
    """Look up a model, falling back to a conservative unknown record.

    An unknown id gets no thinking and the smallest output budget rather than a
    guess. An unknown model still works for plain text, tools and streaming;
    it simply carries no reasoning."""
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
    """A reasoning level plus model facts → the thinking fields and the
    `max_tokens` to send.

    On Bedrock the returned `thinking` / `output_config` keys go inside
    `additionalModelRequestFields`, not at the top level. Returns
    `({}, max_tokens)` when no thinking should be requested at all."""
    resolved_max = max_tokens or (
        capabilities.max_output_tokens
        if capabilities.is_known_model
        else UNKNOWN_MAX_OUTPUT_TOKENS
    )

    if reasoning is None or reasoning == "provider-default":
        return {}, resolved_max
    if reasoning == "none" or not capabilities.supports_thinking:
        return {}, resolved_max

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
            f"{model!r}: a budget of at least {MIN_THINKING_BUDGET} tokens "
            "below it is required.",
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

    Two independent reasons: the newest Anthropic models reject `temperature`
    outright, and any model rejects it while thinking is active. Refused rather
    than stripped — a silently dropped temperature changes the output with
    nothing to notice."""
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
