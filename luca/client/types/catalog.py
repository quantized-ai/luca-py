"""ModelInfo / ModelCost records. The catalog (a separate module) is the
home for these instances; this file just defines their shape."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelCost(BaseModel):
    input_per_million_tokens: float | None = None
    output_per_million_tokens: float | None = None
    cached_input_per_million_tokens: float | None = None
    cache_write_per_million_tokens: float | None = None
    reasoning_per_million_tokens: float | None = None

    model_config = ConfigDict(extra="forbid")


class ModelInfo(BaseModel):
    """One record per (model_id, provider) pair. Every field is optional —
    the SDK uses what it has and falls back to provider defaults for the rest."""

    model: str | None = None
    provider: str | None = None
    display_name: str | None = None
    aliases: list[str] = Field(default_factory=list)

    context_window: int | None = None
    max_tokens: int | None = None

    supports_text_input: bool = True
    supports_image_input: bool = False
    supports_audio_input: bool = False
    supports_pdf_input: bool = False
    supports_video_input: bool = False

    supports_tools: bool = False
    supports_parallel_tool_calls: bool = False
    supports_structured_output: Literal["strict", "loose", "none"] = "none"
    supports_reasoning: bool = False
    reasoning_signature_format: Literal["anthropic", "gemini", "openai", "none"] = "none"
    supports_prompt_caching: bool = False
    supports_streaming: bool = True

    cost: ModelCost | None = None

    compat: dict = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")
