# Terminal-Bench scoreboard

One row per run, newest first, appended by the `terminal-bench` workflow.

Accuracy is the mean reward over every trial, with errored trials counted as 0
rather than dropped. Rows are only comparable when model, `k` and task count
match. Full trajectories live in the linked run's artifacts for 30 days.

Runs use `--agent-setup-timeout-multiplier 3`, which a leaderboard submission
may not, so these numbers are for our own iteration rather than for publishing.

| date | model | effort | tasks | k | accuracy | errored | cost | run |
|---|---|---|---|---|---|---|---|---|

## Local runs before CI existed

Kept for context; these ran on Apple Silicon under QEMU, where setup timeouts
inflate the errored count.

| date | model | tasks | accuracy | errored | cost | note |
|---|---|---|---|---|---|---|
| 2026-08-10 | gpt-5.4-mini | 10 | 10.0% | 2 | $0.45 | 1 passed: openssl-selfsigned-cert |
