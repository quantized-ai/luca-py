"""Reasoning level vocabulary (used as a Literal on the request DTO)."""

from typing import Literal

ReasoningEffort = Literal[
    "none", "minimal", "low", "medium", "high", "xhigh", "auto",
]
