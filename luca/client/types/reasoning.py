"""Reasoning level vocabulary (used as a Literal on the request DTO).

`provider-default` sends nothing and lets the provider decide, which is a
different thing from `none` — an explicit request for no reasoning at all.
"""

from typing import Literal

Reasoning = Literal[
    "provider-default", "none", "minimal", "low", "medium", "high", "xhigh",
]
