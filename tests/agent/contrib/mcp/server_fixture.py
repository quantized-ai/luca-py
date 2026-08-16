"""A scriptable MCP server over stdio, for testing the client against.

Written by hand rather than pulled from a third-party package: it is a hundred
lines of newline-delimited JSON-RPC, and owning it means it can be TOLD how to
misbehave (`--era legacy`, `--crash-marker`, `--delay-tool`) and ASKED what it
saw (calling the `__received__` tool returns the notifications it got). Both of
those are what let the connection tests assert a fact instead of racing a clock.

The introspection channel is an ordinary `tools/call` rather than a method of
its own, because the client validates every result against the surface map for
the negotiated version and a made-up method has no entry there.

It answers concurrently, one task per request, so a delayed call and a fast one
overlap and come back out of order. That is the property the client's id
multiplexing exists for, and a sequential fixture could not test it.

Readiness needs no signal. The client's first request blocks until this process
answers it, so nothing anywhere has to sleep waiting for a spawn.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

RECEIVED_TOOL = "__received__"

MODERN = "2026-07-28"
LEGACY = "2025-06-18"


def tool(name: str, **over) -> dict:
    return {"name": name, "description": f"the {name} tool", "inputSchema": {"type": "object"}, **over}


class Fixture:
    def __init__(self, options: argparse.Namespace) -> None:
        self.options = options
        self.received: list[dict] = []
        self.calls = 0
        self.demanded = False
        self.lock = asyncio.Lock()
        self.out = sys.stdout

    def write(self, frame: dict) -> None:
        self.out.write(json.dumps(frame) + "\n")
        self.out.flush()

    def reply(self, request_id, result: dict) -> None:
        self.write({"jsonrpc": "2.0", "id": request_id, "result": result})

    def fail(self, request_id, code: int, message: str, data=None) -> None:
        error = {"code": code, "message": message}
        if data is not None:
            error["data"] = data
        self.write({"jsonrpc": "2.0", "id": request_id, "error": error})

    async def handle(self, message: dict) -> None:
        method, request_id = message.get("method"), message.get("id")
        if request_id is None:
            self.received.append(message)
            return

        match method:
            case "server/discover":
                await self.discover(message, request_id)
            case "initialize":
                self.reply(
                    request_id,
                    {
                        "protocolVersion": LEGACY,
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "fixture", "version": "1.0"},
                    },
                )
            case "tools/list":
                self.list_tools(message, request_id)
            case "tools/call":
                await self.call_tool(message, request_id)
            case _:
                self.fail(request_id, -32601, f"Method not found: {method}")

    async def discover(self, message: dict, request_id) -> None:
        if self.options.era == "legacy":
            # What a pre-2026 server does with an unknown pre-initialize
            # request. Deliberately NOT one of the spec-reserved codes.
            self.fail(request_id, -32601, "Method not found: server/discover")
            return
        if self.options.era == "silent":
            return  # never answers, so the probe times out into the fallback
        if self.options.demand_version and not self.demanded:
            self.demanded = True
            self.fail(
                request_id,
                -32022,
                "Unsupported protocol version",
                {"supported": [self.options.demand_version]},
            )
            return
        self.reply(
            request_id,
            {
                "supportedVersions": [MODERN],
                "capabilities": {"tools": {"listChanged": True}},
                "resultType": "complete",
                "ttlMs": self.options.ttl_ms,
                "cacheScope": "public",
            },
        )

    def list_tools(self, message: dict, request_id) -> None:
        cursor = (message.get("params") or {}).get("cursor")
        page = int(cursor) if cursor else 0
        last = page >= self.options.pages - 1
        result = {"tools": [tool(f"tool_{page}")], "resultType": "complete", "ttlMs": self.options.ttl_ms}
        if self.options.era == "modern":
            result["cacheScope"] = "public"
        if not last:
            result["nextCursor"] = str(page + 1)
        self.reply(request_id, result)

    async def call_tool(self, message: dict, request_id) -> None:
        async with self.lock:
            self.calls += 1
            mine = self.calls
        name = (message.get("params") or {}).get("name")

        if name == RECEIVED_TOOL:
            # The introspection channel, deliberately shaped as an ordinary
            # tool call: every other method name would fail the client's own
            # per-version surface validation, and rightly so.
            self.reply(
                request_id,
                {
                    "content": [{"type": "text", "text": f"{len(self.received)} received"}],
                    "structuredContent": {"received": self.received, "pid": os.getpid()},
                    "resultType": "complete",
                },
            )
            return

        if self.options.crash_marker and not os.path.exists(self.options.crash_marker):
            # Crash once, ever, across restarts. A counter would kill every
            # fresh process too and the client could never recover.
            Path(self.options.crash_marker).write_text("crashed")
            self.out.flush()
            os._exit(1)  # die with the request in flight

        if name == self.options.delay_tool:
            await asyncio.sleep(self.options.delay_seconds)
        self.reply(
            request_id,
            {
                "content": [{"type": "text", "text": f"{name} #{mine}"}],
                "resultType": "complete",
            },
        )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--era", choices=["modern", "legacy", "silent"], default="modern")
    parser.add_argument("--pages", type=int, default=1)
    parser.add_argument("--crash-marker", default=None)
    parser.add_argument("--delay-tool", default=None)
    parser.add_argument("--delay-seconds", type=float, default=0.05)
    parser.add_argument("--demand-version", default=None)
    parser.add_argument("--ttl-ms", type=int, default=60000)
    fixture = Fixture(parser.parse_args())

    reader = asyncio.StreamReader()
    loop = asyncio.get_running_loop()
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)

    tasks: set[asyncio.Task] = set()
    while line := await reader.readline():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        task = asyncio.create_task(fixture.handle(message))
        tasks.add(task)
        task.add_done_callback(tasks.discard)
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    with __import__("contextlib").suppress(KeyboardInterrupt):
        asyncio.run(main())
