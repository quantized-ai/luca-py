"""Curated catalog data. ~50-100 records minimum (V1 ships a dozen well-known
models from OpenAI / Anthropic / OpenRouter)."""

from __future__ import annotations

from ...types.catalog import ModelCost, ModelInfo


def default_records() -> list[ModelInfo]:
    return [
        # --- OpenAI ---
        ModelInfo(
            provider="openai", model="gpt-4o",
            display_name="GPT-4o",
            context_window=128000, max_tokens=16384,
            supports_image_input=True, supports_audio_input=True,
            supports_tools=True, supports_parallel_tool_calls=True,
            supports_structured_output="strict",
            supports_prompt_caching=True,
            cost=ModelCost(
                input_per_million_tokens=2.50,
                output_per_million_tokens=10.00,
                cached_input_per_million_tokens=1.25,
            ),
        ),
        ModelInfo(
            provider="openai", model="gpt-4o-mini",
            display_name="GPT-4o mini",
            context_window=128000, max_tokens=16384,
            supports_image_input=True,
            supports_tools=True, supports_parallel_tool_calls=True,
            supports_structured_output="strict",
            supports_prompt_caching=True,
            cost=ModelCost(
                input_per_million_tokens=0.15,
                output_per_million_tokens=0.60,
                cached_input_per_million_tokens=0.075,
            ),
        ),
        ModelInfo(
            provider="openai", model="o1",
            display_name="o1",
            context_window=200000, max_tokens=100000,
            supports_tools=True,
            supports_reasoning=True, reasoning_signature_format="openai",
            cost=ModelCost(
                input_per_million_tokens=15.00,
                output_per_million_tokens=60.00,
                reasoning_per_million_tokens=60.00,
            ),
        ),
        ModelInfo(
            provider="openai", model="gpt-4-turbo",
            display_name="GPT-4 Turbo",
            context_window=128000, max_tokens=4096,
            supports_image_input=True,
            supports_tools=True, supports_parallel_tool_calls=True,
            cost=ModelCost(
                input_per_million_tokens=10.00,
                output_per_million_tokens=30.00,
            ),
        ),

        # --- Anthropic ---
        ModelInfo(
            provider="anthropic", model="claude-3-5-sonnet-latest",
            display_name="Claude 3.5 Sonnet",
            context_window=200000, max_tokens=8192,
            supports_image_input=True, supports_pdf_input=True,
            supports_tools=True, supports_parallel_tool_calls=True,
            supports_prompt_caching=True,
            cost=ModelCost(
                input_per_million_tokens=3.00,
                output_per_million_tokens=15.00,
                cached_input_per_million_tokens=0.30,
                cache_write_per_million_tokens=3.75,
            ),
        ),
        ModelInfo(
            provider="anthropic", model="claude-3-5-haiku-latest",
            display_name="Claude 3.5 Haiku",
            context_window=200000, max_tokens=8192,
            supports_image_input=True,
            supports_tools=True, supports_parallel_tool_calls=True,
            supports_prompt_caching=True,
            cost=ModelCost(
                input_per_million_tokens=0.80,
                output_per_million_tokens=4.00,
                cached_input_per_million_tokens=0.08,
                cache_write_per_million_tokens=1.00,
            ),
        ),
        ModelInfo(
            provider="anthropic", model="claude-3-opus-latest",
            display_name="Claude 3 Opus",
            context_window=200000, max_tokens=4096,
            supports_image_input=True, supports_pdf_input=True,
            supports_tools=True, supports_parallel_tool_calls=True,
            supports_prompt_caching=True,
            cost=ModelCost(
                input_per_million_tokens=15.00,
                output_per_million_tokens=75.00,
            ),
        ),

        # --- OpenRouter pass-throughs ---
        ModelInfo(
            provider="openrouter", model="openai/gpt-4o",
            display_name="GPT-4o (via OpenRouter)",
            context_window=128000, max_tokens=16384,
            supports_image_input=True,
            supports_tools=True, supports_parallel_tool_calls=True,
            cost=ModelCost(
                input_per_million_tokens=2.50,
                output_per_million_tokens=10.00,
            ),
        ),
        ModelInfo(
            provider="openrouter", model="anthropic/claude-3.5-sonnet",
            display_name="Claude 3.5 Sonnet (via OpenRouter)",
            context_window=200000, max_tokens=8192,
            supports_image_input=True,
            supports_tools=True, supports_parallel_tool_calls=True,
            cost=ModelCost(
                input_per_million_tokens=3.00,
                output_per_million_tokens=15.00,
            ),
        ),
        ModelInfo(
            provider="openrouter", model="meta-llama/llama-3.1-70b-instruct",
            display_name="Llama 3.1 70B Instruct (via OpenRouter)",
            context_window=128000, max_tokens=4096,
            supports_tools=True,
            cost=ModelCost(
                input_per_million_tokens=0.40,
                output_per_million_tokens=0.40,
            ),
        ),

        # --- Groq ---
        ModelInfo(
            provider="groq", model="llama-3.1-8b-instant",
            display_name="Llama 3.1 8B Instant",
            context_window=128000, max_tokens=8000,
            supports_tools=True,
            cost=ModelCost(
                input_per_million_tokens=0.05,
                output_per_million_tokens=0.08,
            ),
        ),
        ModelInfo(
            provider="groq", model="llama-3.1-70b-versatile",
            display_name="Llama 3.1 70B Versatile",
            context_window=128000, max_tokens=8000,
            supports_tools=True,
            cost=ModelCost(
                input_per_million_tokens=0.59,
                output_per_million_tokens=0.79,
            ),
        ),
    ]
