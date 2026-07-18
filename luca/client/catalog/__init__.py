"""Public catalog facade: catalog.get / catalog.list / catalog.register."""

from __future__ import annotations

from ..types.catalog import ModelInfo
from . import _store


def get(provider: str, model: str) -> ModelInfo | None:
    """Return the ModelInfo for `(provider, model)`, or None if not in the catalog."""
    return _store.get(provider, model)


def list(
    *, provider: str | None = None, supports: str | None = None,
) -> list[ModelInfo]:
    """List ModelInfo records, optionally filtered."""
    return _store.list_records(provider=provider, supports=supports)


def register(*, model: str, provider: str, info: ModelInfo) -> None:
    """Register a ModelInfo record."""
    _store.register(model=model, provider=provider, info=info)


__all__ = ["get", "list", "register", "ModelInfo"]
