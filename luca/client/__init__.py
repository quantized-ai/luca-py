"""luca.client — unified LLM SDK.

Public surface: completion, acompletion, completion_stream, acompletion_stream,
get_provider, catalog.
"""

from . import catalog
from ._client import (
    acompletion,
    acompletion_stream,
    completion,
    completion_stream,
    get_provider,
)

__all__ = [
    "completion",
    "acompletion",
    "completion_stream",
    "acompletion_stream",
    "get_provider",
    "catalog",
]
