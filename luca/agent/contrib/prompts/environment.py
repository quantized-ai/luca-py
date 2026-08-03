"""The environment block: what the model cannot work out for itself.

Pure and fully injectable — the caller supplies the date, the platform and the
git verdict, so a test asserts the whole rendered string against a literal and
nothing here reads the clock or the filesystem.
"""

from __future__ import annotations

import os
from datetime import date

ENVIRONMENT_TEMPLATE = """
### Environment
You are powered by the model {model} on the {provider} provider.
Working directory: {workspace}
Is a git repository: {git}
Platform: {platform}
Today's date: {today}
""".strip()


def environment_text(
    *,
    workspace: str | os.PathLike[str],
    model: str,
    provider: str,
    platform_name: str,
    today: date,
    is_git_repo: bool,
) -> str:
    return ENVIRONMENT_TEMPLATE.format(
        model=model,
        provider=provider,
        workspace=workspace,
        git="yes" if is_git_repo else "no",
        platform=platform_name,
        today=today.isoformat(),
    )
