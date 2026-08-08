"""luca — an AI agent framework.

`luca` EMITS log records and never configures logging: no handlers, no levels,
no `basicConfig`. Loggers are named after their module (`luca.agent.core.runner`,
`luca.client.transports.openai`), so the tree itself is the API — attach a
handler to `luca` for everything, or to a subtree to narrow it.

The `NullHandler` below is load-bearing, not boilerplate. Without a handler
anywhere in the chain, `logging.lastResort` writes every WARNING and above to
STDERR — which paints over a running Textual TUI. This makes the handler search
succeed so `lastResort` never fires.
"""

import logging

__version__ = "0.1.0"

logging.getLogger(__name__).addHandler(logging.NullHandler())
