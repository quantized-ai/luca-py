"""The durable tool catalog, and the reason it is durable.

Contract rule 3 forbids listing inside `get_tools`. An in-memory cache would
obey that and still leave the model silently toolless on the first turn of every
process, so the cache outlives the process instead: the only cold moment left is
the first run after a server is configured.

WHAT INVALIDATES A SLICE:

- `definition_hash` differs: the command line or URL changed, so the tool list
  belongs to a different server.
- `credential_fingerprint` differs and the listing was `cacheScope: "private"`:
  those tools were visible to somebody else's token. Absent scope counts as
  private.
- the TTL lapsed: refresh, but keep serving. Losing tools mid-conversation over
  a lapsed freshness hint is worse than a stale description.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Final

import mcp_types
from pydantic import ValidationError

from luca.agent.core import ToolSpec

from .headers import validate_header_params
from .mapping import to_tool_spec
from .servers import Server

logger = logging.getLogger(__name__)

# What a listing's freshness is worth when the server states nothing useful.
# Handshake-era servers have no way to say, and a modern one may send 0.
DEFAULT_TTL_MS: Final = 300_000
# Never re-list faster than this, whatever a server asks for. A server that
# answers `ttlMs: 1` must not turn into a spin loop.
MIN_TTL_MS: Final = 5_000

CATALOG_VERSION: Final = 1


def _now_ms() -> int:
    return int(time.time() * 1000)


class ToolCatalog:
    """Every server's tool list, in memory and on disk.

    One instance per service, shared across every conversation. Mutation is
    whole-slice replacement under a lock; the listing I/O that produced the
    slice happens outside it, per contract rule 13c.
    """

    def __init__(self, path: Path | None = None, *, now_ms: Callable[[], int] = _now_ms) -> None:
        self.path = path
        self._now_ms = now_ms
        self._slices: dict[str, dict[str, Any]] = {}
        self._specs: dict[str, list[ToolSpec]] = {}  # label -> specs, rebuilt on write
        self._rejected: dict[str, dict[str, str]] = {}  # label -> tool name -> why
        self._identities: dict[str, str] = {}  # label -> server identity, for the write path
        self._lock = asyncio.Lock()

    def load(self, servers: dict[str, Server]) -> None:
        """Read the file and adopt whatever still applies.

        Synchronous, once, at construction, like `auth.json`. A missing or
        corrupt file just means everything is cold.
        """
        stored = self._read()
        for label, server in servers.items():
            entry = stored.get(server.identity())
            if entry is None or not self._applies(entry, server):
                continue
            self._slices[label] = entry
            self._rebuild(label, server)

    def _read(self) -> dict[str, dict[str, Any]]:
        if self.path is None or not self.path.exists():
            return {}
        try:
            document = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("mcp catalog at %s is unreadable; starting cold", self.path, exc_info=exc)
            return {}
        if not isinstance(document, dict) or document.get("version") != CATALOG_VERSION:
            return {}
        servers = document.get("servers")
        return servers if isinstance(servers, dict) else {}

    def _applies(self, entry: dict[str, Any], server: Server) -> bool:
        if entry.get("definition_hash") != server.definition_hash():
            return False
        if entry.get("cache_scope") == "public":
            return True
        return entry.get("credential_fingerprint") == server.credential_fingerprint()

    async def put(
        self,
        label: str,
        server: Server,
        tools: Iterable[mcp_types.Tool],
        *,
        ttl_ms: int,
        cache_scope: str,
    ) -> None:
        """Replace one server's slice, atomically, and write the file.

        The lock covers the swap only; the listing and the disk write are
        outside it.
        """
        entry = {
            "definition_hash": server.definition_hash(),
            "credential_fingerprint": server.credential_fingerprint(),
            "cache_scope": cache_scope,
            "ttl_ms": ttl_ms,
            "fetched_at": self._now_ms(),
            "tools": [tool.model_dump(mode="json", by_alias=True, exclude_none=True) for tool in tools],
        }
        async with self._lock:
            self._slices[label] = entry
            self._rebuild(label, server)
        await self._write()

    def _rebuild(self, label: str, server: Server) -> None:
        """Derive the specs for one slice from its stored tool definitions.

        Re-derived rather than stored, so specs stay byte-identical across runs
        and `ToolSpec.spec_id()` normalization holds. A tool with invalid
        `x-mcp-header` annotations is excluded, as the transport requires, and
        the reason kept so `/mcp` can explain the absence.
        """
        specs: list[ToolSpec] = []
        rejected: dict[str, str] = {}
        for raw in self._slices[label]["tools"]:
            try:
                tool = mcp_types.Tool.model_validate(raw)
            except ValidationError as exc:
                rejected[str(raw.get("name", "?"))] = f"not a valid tool definition: {exc.error_count()} problems"
                continue
            violations = validate_header_params(tool.input_schema or {})
            if violations:
                rejected[tool.name] = violations[0]
                logger.warning("mcp server=%s tool=%s excluded: %s", label, tool.name, violations[0])
                continue
            specs.append(to_tool_spec(label, tool, timeout_in_ms=server.call_timeout_in_ms))
        self._specs[label] = specs
        self._rejected[label] = rejected

    def specs(self) -> list[ToolSpec]:
        """Every cached tool. No awaits and no I/O, which is what makes it legal
        inside `get_tools` and `create_execution`."""
        return [spec for label in sorted(self._specs) for spec in self._specs[label]]

    def specs_for(self, label: str) -> list[ToolSpec]:
        return list(self._specs.get(label, ()))

    def spec(self, name: str) -> ToolSpec | None:
        for specs in self._specs.values():
            for spec in specs:
                if spec.name == name:
                    return spec
        return None

    def tool_count(self, label: str) -> int:
        return len(self._specs.get(label, ()))

    def rejected(self, label: str) -> dict[str, str]:
        return dict(self._rejected.get(label, {}))

    def has(self, label: str) -> bool:
        return label in self._slices

    @property
    def cold(self) -> bool:
        """True when nothing has ever been listed."""
        return not self._slices

    def expires_at(self, label: str) -> int:
        entry = self._slices.get(label)
        if entry is None:
            return 0
        ttl = entry.get("ttl_ms") or DEFAULT_TTL_MS
        return int(entry["fetched_at"]) + max(int(ttl), MIN_TTL_MS)

    def stale(self, labels: Iterable[str]) -> list[str]:
        """Which servers are due a refresh. A never-listed one is always due."""
        now = self._now_ms()
        return [label for label in labels if not self.has(label) or self.expires_at(label) <= now]

    def next_refresh_at(self, labels: Iterable[str]) -> int | None:
        """When the earliest slice goes stale, or None if there is nothing
        cached to go stale."""
        deadlines = [self.expires_at(label) for label in labels if self.has(label)]
        return min(deadlines) if deadlines else None

    def invalidate(self, label: str) -> None:
        """Mark one server due immediately, on a `toolsListChanged`."""
        entry = self._slices.get(label)
        if entry is not None:
            entry["fetched_at"] = 0

    async def _write(self) -> None:
        """Persist every slice, keyed by server IDENTITY rather than label, so
        renaming a server keeps its cached listing. Off the loop (rule 8)."""
        if self.path is None:
            return
        servers = {self._identities[label]: entry for label, entry in self._slices.items() if label in self._identities}
        await asyncio.to_thread(self._write_sync, {"version": CATALOG_VERSION, "servers": servers})

    def _write_sync(self, document: dict) -> None:
        # Named for THIS process: the path is shared by every luca on the
        # machine, and one fixed name lets two of them interleave into it.
        temporary = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            temporary.write_text(json.dumps(document, indent=2))
            temporary.chmod(0o600)  # before it has the real name, never briefly world-readable
            temporary.replace(self.path)  # atomic, so a crash never leaves a half file
        except OSError as exc:
            # An unwritable cache is a slower agent, not a broken one.
            logger.warning("could not write the mcp catalog to %s", self.path, exc_info=exc)
            temporary.unlink(missing_ok=True)

    def track(self, servers: dict[str, Server]) -> None:
        """Remember which identity each label maps to, for the write path."""
        self._identities = {label: server.identity() for label, server in servers.items()}
