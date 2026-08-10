"""luca_tb — running `luca` against Terminal-Bench through the Harbor harness.

Three pieces, split by WHERE they run:

- `runner`  — the headless driver. Uploaded into the task's container and
              invoked there. Imports luca, nothing from harbor.
- `agent`   — `LucaAgent`, a harbor `BaseInstalledAgent`. Runs on the HOST:
              installs the driver into the container, invokes it, then reads
              the trajectory back.
- `mapping` — the pure functions both sides need (model-string parsing, usage
              rollup). No harbor import, so it is testable on its own.

Nothing here is part of the `luca` package, and nothing in `luca` knows this
exists. The harness is ordinary application code over core + contrib's public
surface, which is the point: if it needs a private import to work, that is a
finding about the framework boundary, not something to paper over here.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
