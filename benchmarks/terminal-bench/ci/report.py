"""Merge trial results into a run summary and one scoreboard row.

    python -m ci.report --jobs-dir merged --model ... --run-url ...

Writes the markdown summary to $GITHUB_STEP_SUMMARY when set, appends the row
to `results.md` when `--results` is given, and always prints both.

Reads `.verifier_result.rewards.reward` and `.agent_result`, the paths the
README documents. Accuracy counts errored trials as 0 rather than dropping
them, matching how the leaderboard scores a run: a crash in our adapter is
indistinguishable from the agent failing, and hiding it would flatter us.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

SCOREBOARD_HEADER = "| date | model | effort | tasks | k | accuracy | errored | cost | run |"


@dataclass
class Trial:
    task: str
    reward: float | None
    exception: str | None
    cost: float | None
    input_tokens: int
    output_tokens: int
    setup_sec: float | None

    @property
    def scored(self) -> float:
        """An errored trial is a zero, not a gap."""
        return self.reward or 0.0


@dataclass
class Run:
    trials: list[Trial] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return sum(t.scored for t in self.trials) / len(self.trials) if self.trials else 0.0

    @property
    def errored(self) -> list[Trial]:
        return [t for t in self.trials if t.exception]

    @property
    def passed(self) -> list[Trial]:
        return [t for t in self.trials if (t.reward or 0) > 0]

    @property
    def cost(self) -> float:
        return sum(t.cost or 0 for t in self.trials)

    @property
    def tokens(self) -> tuple[int, int]:
        return sum(t.input_tokens for t in self.trials), sum(t.output_tokens for t in self.trials)


def _seconds(phase: dict | None) -> float | None:
    if not phase or not phase.get("started_at") or not phase.get("finished_at"):
        return None
    from datetime import datetime

    started = datetime.fromisoformat(phase["started_at"].replace("Z", "+00:00"))
    finished = datetime.fromisoformat(phase["finished_at"].replace("Z", "+00:00"))
    return (finished - started).total_seconds()


def load(jobs_dir: Path) -> Run:
    """Every per-trial `result.json` under `jobs_dir`, at any depth.

    Depth-agnostic because the shards are downloaded into per-artifact
    subdirectories, so the tree is one level deeper than a local run."""
    run = Run()
    for path in sorted(jobs_dir.rglob("result.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "task_name" not in data:  # the job-level result.json, not a trial
            continue
        agent = data.get("agent_result") or {}
        run.trials.append(
            Trial(
                task=data["task_name"].split("/")[-1],
                reward=((data.get("verifier_result") or {}).get("rewards") or {}).get("reward"),
                exception=(data.get("exception_info") or {}).get("exception_type"),
                cost=agent.get("cost_usd"),
                input_tokens=agent.get("n_input_tokens") or 0,
                output_tokens=agent.get("n_output_tokens") or 0,
                setup_sec=_seconds(data.get("agent_setup")),
            )
        )
    return run


def summary(run: Run, *, model: str, effort: str, k: int, run_url: str) -> str:
    in_tokens, out_tokens = run.tokens
    lines = [
        "## Terminal-Bench",
        "",
        f"**{run.accuracy:.1%}** over {len(run.trials)} trials "
        f"({len(run.passed)} passed, {len(run.errored)} errored) · ${run.cost:.2f}",
        "",
        f"`{model}`{f' · effort {effort}' if effort else ''} · k={k} · {in_tokens:,} in / {out_tokens:,} out",
        "",
        "| task | reward | setup | cost | note |",
        "|---|---|---|---|---|",
    ]
    for trial in sorted(run.trials, key=lambda t: (-(t.reward or 0), t.task)):
        reward = "—" if trial.reward is None else f"{trial.reward:.0f}"
        setup = f"{trial.setup_sec:.0f}s" if trial.setup_sec else "—"
        cost = f"${trial.cost:.3f}" if trial.cost else "—"
        lines.append(f"| {trial.task} | {reward} | {setup} | {cost} | {trial.exception or ''} |")
    if run_url:
        lines += ["", f"[Full trajectories]({run_url}) are attached as artifacts."]
    return "\n".join(lines)


def scoreboard_row(
    run: Run,
    *,
    date: str,
    model: str,
    effort: str,
    k: int,
    run_url: str,
) -> str:
    link = f"[run]({run_url})" if run_url else "local"
    return (
        f"| {date} | {model} | {effort or '—'} | {len(run.trials)} | {k} "
        f"| {run.accuracy:.1%} | {len(run.errored)} | ${run.cost:.2f} | {link} |"
    )


def append_row(results: Path, row: str) -> None:
    """Insert below the header and its `|---|` separator, so newest sorts first."""
    lines = results.read_text().splitlines() if results.exists() else []
    try:
        header = lines.index(SCOREBOARD_HEADER)
    except ValueError:
        raise SystemExit(f"{results} has no scoreboard table to append to") from None
    lines.insert(header + 2, row)
    results.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--effort", default="")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--date", required=True)
    parser.add_argument("--run-url", default="")
    parser.add_argument("--results", type=Path, default=None)
    args = parser.parse_args()

    run = load(args.jobs_dir)
    if not run.trials:
        raise SystemExit(f"no trial results under {args.jobs_dir}")

    text = summary(run, model=args.model, effort=args.effort, k=args.k, run_url=args.run_url)
    row = scoreboard_row(run, date=args.date, model=args.model, effort=args.effort, k=args.k, run_url=args.run_url)
    print(text)
    print(f"\n{row}")

    import os

    if step_summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        Path(step_summary).write_text(text)
    if args.results:
        append_row(args.results, row)


if __name__ == "__main__":
    main()
