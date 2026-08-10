"""The headless driver: one instruction in, an exit code and a session out.

Uploaded into the task container and run there, so it imports `luca` and the
standard library only — notably not `luca.agent.contrib.tui`, whose package
root pulls in Textual. It composes the agent through core + contrib's public
surface, exactly as any application would.

    python runner.py --model openai/gpt-5.4-mini --provider openrouter \\
        --workspace /app --session-out /logs/agent/session.json -- "<task>"

Exit codes: 0 completed, 1 aborted, 2 blocked on an approval gate, 124 timeout.
The session is written after every drive, so a crash still leaves a trajectory.

No `luca.json` is read: a benchmark run should be a pure function of its
arguments.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import traceback
from pathlib import Path

from luca.agent.contrib.memory import MemoryPlugin
from luca.agent.contrib.plugins import PluginAgentSessionRunner
from luca.agent.contrib.prompts import InstructionsPlugin, SystemPromptPlugin
from luca.agent.contrib.shell import ShellAccessPlugin
from luca.agent.contrib.simple_context_manager import SummarizingContextManager
from luca.agent.contrib.simple_tool_registry import SimpleToolRegistry
from luca.agent.contrib.subagents import SubagentsPlugin
from luca.agent.core.events import TextBlock, ToolCallReceived, ToolExecuted
from luca.agent.core.models import AgentSession, LLMConfig, RuntimeConfig, SystemPromptPart
from luca.agent.core.runner import AgentSessionRunner

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_BLOCKED = 2
EXIT_TIMEOUT = 124

ADDENDUM_PRIORITY = 110  # after the project's own instruction files (100)

TODO_STORE_KEY = "todos"
SCRATCHPAD_STORE_KEY = "scratchpad"


class PromptAddendumPlugin:
    """One extra system-prompt part, appended last. Plugin hooks are
    duck-typed, so a plain class is a complete plugin."""

    def __init__(self, text: str) -> None:
        self.text = text

    def get_system_prompt_parts(self, agent_session: AgentSession) -> list:
        return [SystemPromptPart(text=self.text, source="benchmark", priority=ADDENDUM_PRIORITY)]


def build_runner(
    session: AgentSession,
    *,
    workspace: str | Path,
    mode: str,
    provider=None,
    instructions: bool = True,
    subagents: bool = False,
    addendum: str | None = None,
    compaction: bool = True,
) -> PluginAgentSessionRunner:
    """Shell tools, memory, the system prompt, and one permission strategy
    shared by every registry so a single gate serves everything."""
    shell = ShellAccessPlugin(workspace=Path(workspace), mode=mode)
    registry = SimpleToolRegistry(tools=[], permission_policy=shell.permission_strategy)
    memory = MemoryPlugin(
        scratchpad_store=session.extras.setdefault(SCRATCHPAD_STORE_KEY, {}),
        todo_store=session.extras.setdefault(TODO_STORE_KEY, {}),
    )
    plugins: list = [SystemPromptPlugin(workspace=workspace), memory, shell]
    if instructions:
        plugins.append(InstructionsPlugin(workspace=workspace))
    if subagents:
        plugins.append(SubagentsPlugin())
    if addendum:
        plugins.append(PromptAddendumPlugin(addendum))
    # Core's default accounts but never compacts, so a long task would die on
    # context overflow rather than on its own merits.
    context_manager = SummarizingContextManager(enabled=compaction, provider=provider) if compaction else None
    return PluginAgentSessionRunner(
        session,
        tool_registry=registry,
        plugins=plugins,
        provider=provider,
        context_manager=context_manager,
    )


async def drive(runner: PluginAgentSessionRunner, on_save) -> int:
    """Advance until the turn is done, narrating to stdout. Headless, an
    approval gate is terminal."""
    final = ""
    while not runner.idle():
        async with runner.run() as run:
            async for event in run:
                match event:
                    case ToolCallReceived(execution=execution):
                        call = execution.raw_tool_call
                        print(f"  → {call.name} {_argline(call.arguments)}", flush=True)
                    case ToolExecuted(result_text=text, is_error=is_error):
                        marker = "✗" if is_error else "←"
                        print(f"  {marker} {_head(text)}", flush=True)
                    case TextBlock(text=text):
                        final = text
                        print(text, flush=True)
        on_save()
        if runner.blocked():
            pending = runner.pending_approvals()
            names = ", ".join(execution.raw_tool_call.name for execution in pending)
            print(f"blocked on approval for: {names}", file=sys.stderr, flush=True)
            return EXIT_BLOCKED
    if final:
        print(f"\n--- final ---\n{final}", flush=True)
    return EXIT_OK


def _argline(arguments: dict, limit: int = 120) -> str:
    text = " ".join(f"{key}={value!r}" for key, value in arguments.items())
    return text if len(text) <= limit else text[:limit] + "…"


def _head(text: str | None, limit: int = 200) -> str:
    if not text:
        return "(no output)"
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[:limit] + "…"


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run luca headlessly against one instruction.")
    parser.add_argument("prompt", help="The instruction. Use '-' to read it from stdin.")
    parser.add_argument("--model", required=True, help="Model id, as luca names it.")
    parser.add_argument("--provider", required=True, help="Provider host, e.g. openrouter.")
    parser.add_argument("--reasoning", default=None, help="Reasoning effort, when the model takes one.")
    parser.add_argument("--workspace", default=".", help="Tool root; match the task's WORKDIR.")
    parser.add_argument("--session-out", default=None, help="Where to write the AgentSession JSON.")
    parser.add_argument(
        "--permission-mode",
        default="yolo",
        choices=("yolo", "ask"),
        help="yolo auto-approves every tool call (the benchmark default); ask exits 2 at the first gate.",
    )
    parser.add_argument("--max-steps", type=int, default=200, help="Hard ceiling on assistant messages per turn.")
    parser.add_argument("--timeout", type=int, default=0, help="Wall-clock seconds; 0 disables.")
    parser.add_argument("--subagents", action="store_true", help="Allow parallel subagents (off by default).")
    parser.add_argument("--no-instructions", dest="instructions", action="store_false", help="Ignore AGENTS.md.")
    parser.add_argument("--no-compaction", dest="compaction", action="store_false", help="Never compact the context.")
    parser.add_argument("--append-system-prompt", default=None, help="Extra system-prompt text, appended last.")
    parser.add_argument(
        "--append-system-prompt-file",
        default=None,
        help="Read the extra system-prompt text from this file.",
    )
    return parser


def resolve_prompt(value: str) -> str:
    return sys.stdin.read() if value == "-" else value


def resolve_addendum(args: argparse.Namespace) -> str | None:
    if args.append_system_prompt_file:
        return Path(args.append_system_prompt_file).read_text()
    return args.append_system_prompt


async def run(args: argparse.Namespace, provider=None) -> int:
    session = AgentSessionRunner.new_session(
        LLMConfig(model=args.model, provider=args.provider, reasoning=args.reasoning),
        runtime_config=RuntimeConfig(
            hard_max_steps=args.max_steps,
            subagents_enabled=args.subagents,
        ),
    )
    runner = build_runner(
        session,
        workspace=args.workspace,
        mode=args.permission_mode,
        provider=provider,
        instructions=args.instructions,
        subagents=args.subagents,
        addendum=resolve_addendum(args),
        compaction=args.compaction,
    )

    def save() -> None:
        if args.session_out:
            path = Path(args.session_out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(session.model_dump_json(indent=2))

    runner.post_message(resolve_prompt(args.prompt))
    try:
        if args.timeout:
            return await asyncio.wait_for(drive(runner, save), timeout=args.timeout)
        return await drive(runner, save)
    except TimeoutError:
        print(f"timed out after {args.timeout}s", file=sys.stderr, flush=True)
        return EXIT_TIMEOUT
    except Exception:
        # Broad on purpose: the traceback plus the saved trajectory is worth
        # more than letting it crash and lose which layer failed.
        traceback.print_exc()
        return EXIT_ERROR
    finally:
        save()  # the runner mutates the session in place


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(run(arg_parser().parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
