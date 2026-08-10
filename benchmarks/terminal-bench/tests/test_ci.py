"""`ci/shard.py` and `ci/report.py` — the two places CI can quietly lie.

A bad chunk silently drops tasks from a run; bad accuracy maths misreports the
score. Neither needs Docker or a dataset.
"""

import json

import pytest
from ci.report import Run, Trial, append_row, load, scoreboard_row, summary
from ci.shard import chunk, matrix, select

TASKS = [f"terminal-bench/task-{i:02d}" for i in range(1, 12)]


# ── shard ────────────────────────────────────────────────────────────────────


def test_all_selects_everything():
    assert select(TASKS, "all") == TASKS
    assert select(TASKS, "") == TASKS


def test_a_count_takes_the_first_n():
    assert select(TASKS, "3") == TASKS[:3]


def test_a_glob_matches_the_full_namespaced_name():
    assert select(TASKS, "*/task-0*") == TASKS[:9]


def test_a_glob_matching_nothing_fails_loudly():
    # silently running zero tasks would look like a passing workflow
    with pytest.raises(SystemExit, match="no task matched"):
        select(TASKS, "nope")


def test_chunking_keeps_every_task_exactly_once():
    buckets = chunk(TASKS, 4)

    assert sorted(t for bucket in buckets for t in bucket) == sorted(TASKS)


def test_chunking_is_round_robin_so_heavy_neighbours_split_up():
    # the big images cluster alphabetically; contiguous slices would put a
    # whole shard's disk budget in one bucket
    assert chunk(TASKS, 3)[0] == [TASKS[0], TASKS[3], TASKS[6], TASKS[9]]


def test_more_shards_than_tasks_yields_no_empty_shard():
    buckets = chunk(TASKS[:2], 9)

    assert buckets == [[TASKS[0]], [TASKS[1]]]


def test_zero_or_negative_shards_still_produces_one():
    assert chunk(TASKS, 0) == [TASKS]


def test_the_matrix_is_json_serialisable_with_space_joined_tasks():
    payload = matrix(chunk(TASKS[:4], 2))

    assert json.loads(json.dumps(payload)) == {
        "include": [
            {"shard": 1, "tasks": f"{TASKS[0]} {TASKS[2]}"},
            {"shard": 2, "tasks": f"{TASKS[1]} {TASKS[3]}"},
        ]
    }


# ── report ───────────────────────────────────────────────────────────────────


def trial(task, reward=None, exception=None, cost=None):
    return Trial(
        task=task,
        reward=reward,
        exception=exception,
        cost=cost,
        input_tokens=100,
        output_tokens=10,
        setup_sec=30.0,
    )


def test_errored_trials_count_as_zero_rather_than_being_dropped():
    # the leaderboard scores them this way, and excluding them would flatter us
    run = Run(trials=[trial("a", reward=1.0), trial("b", exception="Boom")])

    assert run.accuracy == 0.5
    assert len(run.errored) == 1


def test_accuracy_is_the_mean_reward():
    run = Run(trials=[trial("a", reward=1.0), trial("b", reward=0.0), trial("c", reward=1.0)])

    assert run.accuracy == pytest.approx(2 / 3)


def test_an_empty_run_is_zero_not_a_crash():
    assert Run().accuracy == 0.0


def test_cost_sums_and_tolerates_unpriced_trials():
    run = Run(trials=[trial("a", reward=1.0, cost=0.25), trial("b", reward=0.0)])

    assert run.cost == 0.25


def test_load_reads_trials_at_any_depth_and_skips_the_job_level_result(tmp_path):
    # shards download into per-artifact subdirectories, so the tree is deeper
    (tmp_path / "shard-1" / "trial-a").mkdir(parents=True)
    (tmp_path / "shard-1" / "result.json").write_text(json.dumps({"stats": {}}))  # job level
    (tmp_path / "shard-1" / "trial-a" / "result.json").write_text(
        json.dumps(
            {
                "task_name": "terminal-bench/fix-git",
                "verifier_result": {"rewards": {"reward": 1.0}},
                "agent_result": {"cost_usd": 0.04, "n_input_tokens": 49162, "n_output_tokens": 1292},
                "agent_setup": {"started_at": "2026-08-10T00:00:00Z", "finished_at": "2026-08-10T00:00:22Z"},
            }
        )
    )

    run = load(tmp_path)

    assert [t.task for t in run.trials] == ["fix-git"]
    assert run.accuracy == 1.0
    assert run.trials[0].setup_sec == 22.0
    assert run.tokens == (49162, 1292)


def test_load_skips_unreadable_files_instead_of_failing_the_report(tmp_path):
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "result.json").write_text("{not json")

    assert load(tmp_path).trials == []


def test_the_summary_leads_with_accuracy_and_cost():
    run = Run(trials=[trial("a", reward=1.0, cost=0.1), trial("b", exception="Boom")])

    text = summary(run, model="m", effort="high", k=5, run_url="")

    assert "**50.0%** over 2 trials (1 passed, 1 errored) · $0.10" in text
    assert "effort high" in text
    assert "Boom" in text


def test_the_scoreboard_row_records_what_makes_runs_comparable():
    run = Run(trials=[trial("a", reward=1.0, cost=0.1)])

    row = scoreboard_row(run, date="2026-08-10", model="m", effort="", k=1, run_url="https://x/1")

    assert row == "| 2026-08-10 | m | — | 1 | 1 | 100.0% | 0 | $0.10 | [run](https://x/1) |"


def test_a_new_row_lands_directly_under_the_header_so_newest_is_first(tmp_path):
    results = tmp_path / "results.md"
    results.write_text(
        "# Results\n\n| date | model | effort | tasks | k | accuracy | errored | cost | run |\n"
        "|---|---|---|---|---|---|---|---|---|\n| old |\n"
    )

    append_row(results, "| new |")

    assert results.read_text().splitlines()[-2:] == ["| new |", "| old |"]


def test_appending_to_a_file_without_the_table_fails_loudly(tmp_path):
    results = tmp_path / "results.md"
    results.write_text("# Results\n")

    with pytest.raises(SystemExit, match="no scoreboard table"):
        append_row(results, "| new |")
