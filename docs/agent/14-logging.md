# Logging

The runner turns exceptions into durable state. A tool body that raises becomes
a `ToolExecution` with `status=FAILED` and a `ToolExecutionError`; a failing LLM
call becomes `TurnFinish(outcome=ERRORED)`. What survives into the session is
`str(exc)` — the traceback is gone.

Logging is where that traceback goes. Every conversion point logs at `ERROR`
with `exc_info` first, so `KeyError: 'path'` in your tool has a stack you can
read.

## luca configures nothing

The library emits records and never touches handlers, levels, or
`basicConfig`. That is yours:

```python
import logging

logging.basicConfig(filename="agent.log", level="INFO")
```

That's the whole setup. Lines look like this:

```
2026-08-08 14:22:31 ERROR    luca.agent.core.runner conv=c1 tool=read_file raised
Traceback (most recent call last):
  ...
KeyError: 'path'
```

Loggers are named after their module, so the tree is the API:

```python
logging.getLogger("luca").setLevel("INFO")             # everything
logging.getLogger("luca.agent").setLevel("DEBUG")      # the agent loop
logging.getLogger("luca.client").setLevel("WARNING")   # quiet the HTTP layer
```

`luca/__init__.py` attaches a `NullHandler` to the `luca` logger. That is not
politeness: without a handler anywhere in the chain, `logging.lastResort` writes
every `WARNING` and above to **stderr**, which paints over a full-screen TUI.

## Reading a line

Records carry the conversation in the message text, as `conv=<id>`:

```
conv=c1 tool=lookup raised
conv=sub_a2f9 LLM call failed (model=anthropic/claude-sonnet-5)
```

A session runs several conversations at once — the main one plus subagents —
so that prefix is what makes an interleaved log readable. It is plain text, not
a `record` attribute, so no custom format string or filter is needed to see it.
Records from `luca.client` have no conversation and carry no prefix.

## What is logged

| Level | What |
|---|---|
| `ERROR` | Every exception converted into state, with the traceback: a raising `create_execution`, `prepare()`, or tool body; a failed LLM call; a failed compaction; an `on_event` callback that raised; a subagent that could not start |
| `WARNING` | Failed but recovered: a tool that hit its deadline, a session file that could not be read |

`INFO` and `DEBUG` are deliberately sparse for now — the failure surface came
first, and quiet levels get filled in as specific needs appear.

Two things are deliberately absent:

- **The event stream.** [`AgentEvent`](04-runner.md) already is the rendering
  API. Mirroring it into logs would be a second, worse copy of it. Logs cover
  what events do not carry: internal decisions and tracebacks.
- **Message bodies.** Prompts are large and carry user data. Nothing above
  `DEBUG` logs message content, and the client logs URL, status and latency —
  never headers or bodies.

## Under a TUI

Anything written to stderr lands on top of the interface. So a terminal app
sends luca's records to a file and takes them off the root logger:

```python
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler("agent.log", maxBytes=5_000_000, backupCount=3)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s %(message)s"))

log = logging.getLogger("luca")
log.setLevel("INFO")
log.addHandler(handler)
log.propagate = False       # keep luca's records away from a root stderr handler
```

The shipped TUI does exactly this, per session — see
[`contrib/tui/config.md`](contrib/tui/config.md#logging).

## Adding logs of your own

Your tools, middleware and registries are ordinary Python. Use the standard
module logger and follow the same two conventions, so your lines sort and grep
alongside luca's:

```python
logger = logging.getLogger(__name__)

logger.info("conv=%s indexed %d files", conversation_id, count)
```

Lazy `%` arguments, never f-strings — the arguments are not rendered when the
level is off. And `conv=<id>` first: every tool, middleware hook and registry
method already receives `conversation_id`.

Next: [`15-rewind.md`](15-rewind.md).
