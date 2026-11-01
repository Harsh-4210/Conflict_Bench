---
title: ConflictBench Leaderboard
emoji: ⚖️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.13.0
app_file: app.py
pinned: false
license: mit
---

# ConflictBench leaderboard

Upload a predictions file for the ConflictBench test set and get a score. The answers are private; the Space scores
your file against them with the same `verifier.score` that the GitHub repository uses
(https://github.com/Harsh-4210/Conflict_Bench).

1. Get the prompts from the public dataset (`public/test.jsonl`; each row has `messages` to send to your model).
2. Write one line per scenario: `{"scenario_id": "...", "completion": "<raw model output>"}`.
3. Upload the file here with a model name. Every scenario must appear exactly once.

You see the composite score with a 95% bootstrap interval, the mean per difficulty and the mean per rubric.
Scores are self-reported and not independently verified.

## Setup (maintainer)

In the Space settings (Settings, Variables and secrets):

- `CONFLICTBENCH_REF` (variable, required): the git tag or commit SHA of the ConflictBench repository that holds
  `verifier.py`, `generator.py`, `scenario_io.py` and `scripts/evaluate.py`. On first start the app fetches exactly that
  commit (branch names are refused, so the scoring code cannot change under the leaderboard).
- `CONFLICTBENCH_PRIVATE_REPO` (variable, required): your PRIVATE dataset repo that holds `test_full.jsonl`.
- `HF_TOKEN` (secret, required): a read-only token that can read that private repo.
- `CONFLICTBENCH_LEADERBOARD` (variable, recommended): path of the leaderboard file, for example `/data/leaderboard.json`
  with persistent storage enabled. Without it the table lives in the container and is lost on rebuild.

To run it locally inside a checkout of the repository: set `CONFLICTBENCH_PRIVATE_SET=path/to/test_full.jsonl`, then
`python leaderboard_space/app.py`. Full guide: `docs/BENCHMARK.md` in the main repository.
