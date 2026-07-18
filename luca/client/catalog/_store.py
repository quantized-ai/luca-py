"""In-memory dict-backed catalog store. Loaded lazily on first access."""

from __future__ import annotations

import threading

from ..types.catalog import ModelInfo

_lock = threading.Lock()
_store: dict[tuple[str, str], ModelInfo] = {}
_loaded: bool = False


def _ensure_loaded() -> None:
    global _loaded
    if _loaded:
        return
    with _lock:
        if _loaded:
            return
        from ._data import default_records

        for info in default_records():
            assert info.provider is not None and info.model is not None
            _store[(info.provider, info.model)] = info
        _loaded = True


def get(provider: str, model: str) -> ModelInfo | None:
    _ensure_loaded()
    return _store.get((provider, model))


def list_records(
    *, provider: str | None = None, supports: str | None = None,
) -> list[ModelInfo]:
    _ensure_loaded()
    records = list(_store.values())
    if provider is not None:
        records = [r for r in records if r.provider == provider]
    if supports is not None:
        records = [r for r in records if _matches_supports(r, supports)]
    return records


def register(*, model: str, provider: str, info: ModelInfo) -> None:
    _ensure_loaded()
    # Ensure the record's provider/model match what's registered.
    if info.provider is None:
        info = info.model_copy(update={"provider": provider})
    if info.model is None:
        info = info.model_copy(update={"model": model})
    with _lock:
        _store[(provider, model)] = info


def _matches_supports(info: ModelInfo, supports: str) -> bool:
    mapping = {
        "vision": info.supports_image_input,
        "audio": info.supports_audio_input,
        "pdf": info.supports_pdf_input,
        "video": info.supports_video_input,
        "tools": info.supports_tools,
        "parallel_tool_calls": info.supports_parallel_tool_calls,
        "structured_output_strict": info.supports_structured_output == "strict",
        "structured_output_loose": info.supports_structured_output == "loose",
        "reasoning": info.supports_reasoning,
        "prompt_caching": info.supports_prompt_caching,
        "streaming": info.supports_streaming,
    }
    return bool(mapping.get(supports, False))


def _clear_for_tests() -> None:
    """Test-only hook — reset the store."""
    global _loaded
    with _lock:
        _store.clear()
        _loaded = False
