"""Resolve the dataset's tasks and chunk them into a GitHub Actions matrix.

    python -m ci.shard --tasks all --shards 9

Writes `matrix={"include":[{"shard":1,"tasks":"a b c"},...]}` to $GITHUB_OUTPUT
when it is set, and always prints the JSON.

Nothing hardcodes the task count. The dataset is the source of truth, so a
version bump changes the matrix instead of silently dropping tasks off the end.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from fnmatch import fnmatch
from pathlib import Path

DATASET = "terminal-bench/terminal-bench-2-1"


async def dataset_tasks(dataset: str = DATASET) -> list[str]:
    from harbor.models.job.config import DatasetConfig

    configs = await DatasetConfig(name=dataset).get_task_configs(disable_verification=False)
    return sorted(config.name for config in configs)


def select(tasks: list[str], spec: str) -> list[str]:
    """`all`, a count, or a glob against the full namespaced name.

    The glob is matched the way harbor's own `-i` matches, so a pattern that
    works here works when passed through."""
    spec = spec.strip()
    if not spec or spec == "all":
        return tasks
    if spec.isdigit():
        return tasks[: int(spec)]
    matched = [task for task in tasks if fnmatch(task, spec)]
    if not matched:
        raise SystemExit(f"no task matched {spec!r}; the dataset has {len(tasks)}, e.g. {tasks[:3]}")
    return matched


def chunk(tasks: list[str], shards: int) -> list[list[str]]:
    """Round-robin rather than contiguous slices.

    The heavy images cluster alphabetically (`torch-*`, `qemu-*`), and a
    contiguous split would put a shard's whole disk budget in one bucket."""
    shards = max(1, min(shards, len(tasks)))
    buckets: list[list[str]] = [[] for _ in range(shards)]
    for index, task in enumerate(tasks):
        buckets[index % shards].append(task)
    return [bucket for bucket in buckets if bucket]


def matrix(buckets: list[list[str]]) -> dict:
    return {"include": [{"shard": i + 1, "tasks": " ".join(bucket)} for i, bucket in enumerate(buckets)]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--tasks", default="all", help="'all', a count, or a glob")
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()

    selected = select(asyncio.run(dataset_tasks(args.dataset)), args.tasks)
    payload = matrix(chunk(selected, args.shards))

    print(f"{len(selected)} tasks across {len(payload['include'])} shard(s)")
    print(json.dumps(payload, indent=2))

    if output := os.environ.get("GITHUB_OUTPUT"):
        with Path(output).open("a") as handle:
            handle.write(f"matrix={json.dumps(payload)}\n")
            handle.write(f"count={len(selected)}\n")


if __name__ == "__main__":
    main()
