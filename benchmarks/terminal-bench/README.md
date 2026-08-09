# luca on Terminal-Bench

Scores `luca` on [Terminal-Bench](https://www.tbench.ai) 2.1 so prompt, tool and
compaction changes can be measured instead of argued about.

A separate uv project, outside the `luca` package, composing an agent from core
+ contrib's public surface like any application would. `harbor` needs Python
>= 3.12 (the root declares >= 3.11), and `luca-ai` should keep `httpx` +
`pydantic` as its only runtime deps.

## Results

**2026-08-09 · gpt-5.4-mini via OpenRouter · 10 tasks, k=1**

```
scored:    6 trials -> 2 passed, 4 failed        mean reward 0.200
errored:   4 trials (count as reward 0)
             AgentSetupTimeoutError       2   ours, fixed below
             VerifierTimeoutError         1   slow verifier, also fails for nop
             EnvironmentStartTimeoutError 1   image pull
passed:    openssl-selfsigned-cert, pypi-server
cost:      ~$0.04 per solved task, 19m wall clock at -n 4
```

Baselines on the same setup: oracle 5/5, nop 0/5. So the harness is sound and
no task passes for free.

**Fixed since:** `uv venv --python 3.11` downloaded a managed CPython on heavy
ML images and blew the 360s agent-setup budget. Now prefers an interpreter the
image already has. `torch-pipeline-parallelism` setup went from timeout to
121s.

Not yet run: all 89 tasks, or `-k 5`.

## Setup

```bash
uv sync
export OPENROUTER_API_KEY=sk-or-...   # exported, not just in .env
docker info
```

## Run it

Cheapest first, each only worth running if the last passed.

```bash
# 1. luca headless, no Docker, ~30s
uv run python -m luca_tb.runner --model openai/gpt-5.4-mini --provider openrouter \
    --workspace /tmp/scratch --max-steps 20 -- "create hello.py and run it"

# 2. harness sane?  no luca involved.  slow once, images cache
uv run harbor run -d terminal-bench/terminal-bench-2-1 -a oracle -l 5   # expect 5/5

# 3. luca installs?  a container, zero tokens
uv run harbor run -d terminal-bench/terminal-bench-2-1 \
    -a luca_tb.agent:LucaAgent -m openrouter/openai/gpt-5.4-mini \
    -i terminal-bench/fix-git --install-only

# 4. a score
uv run harbor run -d terminal-bench/terminal-bench-2-1 \
    -a luca_tb.agent:LucaAgent -m openrouter/openai/gpt-5.4-mini -l 10 -n 4
```

Task names are namespaced and `-i` is a glob against the full name:
`-i terminal-bench/fix-git` or `-i '*/fix-git'` match, `-i fix-git` matches
nothing and aborts the job. To list all 89:

```bash
uv run python -c "
import asyncio
from harbor.models.job.config import DatasetConfig
async def main():
    c = await DatasetConfig(name='terminal-bench/terminal-bench-2-1').get_task_configs(
        disable_verification=False)
    print('\n'.join(sorted(t.name for t in c)))
asyncio.run(main())"
```

Agent knobs: `--ak max_steps=300 --ak reasoning_effort=high --ak subagents=true`.

Through Docker, from the repo root:

```bash
docker compose -f docker/docker-compose.bench.yml run --rm bench \
    harbor run -d terminal-bench/terminal-bench-2-1 -a oracle -l 5
```

## Reading results

```bash
JOB=$(ls -t jobs | head -1); TRIAL=$(ls jobs/$JOB | grep -v '\.json$' | head -1)

jq '.stats' jobs/$JOB/result.json                                  # the score
jq '.verifier_result.rewards.reward' jobs/$JOB/$TRIAL/result.json  # 1.0 or 0.0
jq '.agent_result' jobs/$JOB/$TRIAL/result.json                    # tokens, cost
cat jobs/$JOB/$TRIAL/agent/luca.txt                                # what it did
cat jobs/$JOB/$TRIAL/trial.log                                     # install commands
uv run harbor view jobs                                            # or browse
```

Each task is a container plus a hidden test script; the verifier runs after the
agent and emits reward `1.0` or `0.0`. Only the filesystem is graded. Accuracy
is the mean reward over all trials, and **errored trials count as 0** rather
than being excluded, so an adapter crash costs score directly.

Three piles when triaging: *failed* is ordinary, *errored* is our bug, and a
*core gap* (context overflow, an unrepresentable tool result) feeds back into
the data model and the spec.

## Leaderboard

Rules from `terminal-bench-2-1/leaderboard/ci/static_analysis.py`, which
rejects rather than warns:

- All 89 tasks, ≥5 trials each (`-k 5`), so 445 trials.
- No timeout or resource overrides, on the job config *and* every trial config.
  This is why the driver passes `--timeout 0` and lets each task's own
  `agent.timeout_sec` govern.
- Effort is keyed on the kwarg literally named `reasoning_effort`.
- Trials must be public on Harbor Hub; CI re-derives every number from there.

```bash
harbor auth login
uv run harbor run -d terminal-bench/terminal-bench-2-1 \
    -a luca_tb.agent:LucaAgent -m <provider/model> \
    --ak reasoning_effort=<effort> -e daytona -k 5 -n <concurrency> \
    --upload --public

git clone https://github.com/harbor-framework/terminal-bench-2-1.git
cd terminal-bench-2-1/leaderboard && uv run lb submit https://hub.harborframework.com/jobs/<uuid>
```

For scale, the current top entry (Claude Code on Fable 5, 83.8%) cost $552.

## Layout

| File | Runs | Does |
|---|---|---|
| `luca_tb/agent.py` | host | harbor `BaseInstalledAgent`: installs the driver, invokes it, reads the trajectory back |
| `luca_tb/runner.py` | container | headless driver: one instruction in, exit code + `AgentSession` out |
| `luca_tb/mapping.py` | both | model-string parsing, usage rollup; no harbor import |

Driver exit codes: `0` done, `1` aborted, `2` approval gate, `124` timeout.
`2` and `124` are recorded but not raised, so the verifier still scores the task.

```bash
uv run pytest tests/
```
