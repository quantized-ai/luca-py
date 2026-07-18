"""Public testing facade. Re-exports FauxProvider, FauxTransport, and faux_*
builders for SDK users writing tests of their own apps."""

from .providers.faux import FauxProvider
from .transports.faux import (
    FauxTransport,
    faux_assistant_message,
    faux_error,
    faux_hang,
    faux_text,
    faux_thinking,
    faux_tool_call,
)
from .transports.faux.transport import faux_refusal

__all__ = [
    "FauxProvider",
    "FauxTransport",
    "faux_assistant_message",
    "faux_text",
    "faux_thinking",
    "faux_tool_call",
    "faux_refusal",
    "faux_error",
    "faux_hang",
]
