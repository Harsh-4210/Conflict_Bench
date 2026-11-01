# ConflictBench — Training Guide

There is one training pipeline, [`training/train_grpo.py`](../training/train_grpo.py), driven by one YAML file per setup in [`configs/`](../configs/). This guide describes what the code does. **None of these configs has been run on a GPU yet**, so there are no expected rewards or run times to quote; record what you see in `results/<run_name>/`.

---

## 1. Install

Training needs Linux and an NVIDIA GPU (Colab, Kaggle or a cloud machine). The repo's CPU environment (`.venv`, used for tests) is separate; do not install torch or Unsloth into it.

```bash
git clone https://github.com/Harsh-4210/Conflict_Bench.git
cd Conflict_Bench
pip install -r requirements-train.txt
```

`requirements-train.txt` pins every package to a released version. It leaves out torch, torchvision and xformers so that Colab and Kaggle keep their preinstalled CUDA build; its header gives the matching versions and the command for a bare machine. The pins were resolved from package metadata and are **not yet confirmed by a GPU run**. After the first successful smoke run, refresh them from `results/grpo_smoke/pip_freeze.txt` as the header explains.

## 2. The configs

| | `grpo_smoke.yaml` | `grpo_t4.yaml` | `grpo_a10g.yaml` |
|---|---|---|---|
| Purpose | plumbing test; its numbers mean nothing | free Kaggle / Colab T4 | A10G / L4 24 GB, also the training Space |
| Model | `unsloth/Qwen2.5-0.5B-Instruct` | `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` | `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` |
| 4-bit | no | yes | yes |
| LoRA r / alpha | 8 / 8 | 16 / 16 | 16 / 16 |
| Train / val scenarios | 20 / 4 | 200 / 20 | 600 / 30 |
| Train difficulty mix | level 1 | 60% L1, 40% L2 | 40% L1, 40% L2, 20% L3 |
| Val difficulty mix | level 1 | 50% L1, 50% L2 | 40% L1, 40% L2, 20% L3 |
| max_prompt_tokens + max_completion_tokens | 1792 + 384 | 2048 + 640 | 2560 + 1024 |
| max_seq_length | 2304 | 2688 | 4096 |
| num_generations | 2 | 4 | 8 |
| Batch × accumulation | 2 × 1 | 4 × 2 | 8 × 2 |
| Learning rate / beta | 5e-6 / 0.04 | 5e-6 / 0.04 | 5e-6 / 0.04 |
| Steps | max_steps 10 | 1 epoch | 1 epoch |
| eval_steps / save_steps | 5 / 5 | 25 / 25 | 50 / 50 |

Every config sets `report_to: none` and `hub_repo: null`, so by default nothing is sent to W&B or the Hugging Face Hub. All keys are checked when the file is loaded (`training/config.py`); for example the run stops before loading a model if `max_prompt_tokens + max_completion_tokens > max_seq_length`, if batch × accumulation is not divisible by `num_generations`, or if `save_steps` is not a multiple of `eval_steps`.

## 3. Run

```bash
# 1. No GPU needed: validate the config, build the datasets with the real tokenizer, print token stats, exit.
python -m training.train_grpo --config configs/grpo_t4.yaml --dry-run

# (a quicker sample: average and max prompt tokens per difficulty, and how many fit)
python scripts/diagnose_tokens.py --config configs/grpo_t4.yaml

# 2. 10-step plumbing test on a small model.
python -m training.train_grpo --config configs/grpo_smoke.yaml

# 3. A real run.
python -m training.train_grpo --config configs/grpo_t4.yaml
python -m training.train_grpo --config configs/grpo_a10g.yaml
```

Overrides without editing the YAML:

```bash
python -m training.train_grpo --config configs/grpo_t4.yaml --max-steps 10 --run-name t4_try1 \
    --output-dir /content/drive/MyDrive/cb-out --set learning_rate=3e-6 --set val_size=40
```

`--set KEY=VALUE` works for any config key and can be repeated. `python train_grpo.py` in the repo root is a deprecated shim that forwards here (with `configs/grpo_a10g.yaml` if no `--config` is given).

## 4. Datasets and token budgets

- Training scenarios come from the **train** split, in-training evaluation from the **val** split (`GeneratorConfig(split=...)`), each with its own seed. The **test** split is never generated during training; the code refuses it.
- Each scenario's prompt is built with `prompts.build_messages()`, the same text used by evaluation and the demo.
- The length filter counts **tokens** of the full chat-templated prompt (system + user + generation prompt) with the model's own tokenizer. Prompts over `max_prompt_tokens` are dropped and counted per difficulty. If any requested difficulty keeps less than half of its target, the run stops with an error telling you to raise `max_prompt_tokens` (and `max_seq_length`) or change the mix. Val scenarios with the same prompt as a train scenario are dropped.
- Per-difficulty counts (generated, dropped as too long, dropped as duplicate, kept, largest prompt seen) are printed by `--dry-run` and saved to `results/<run_name>/dataset_stats.json`.

## 5. Reward

- The reward TRL optimises is exactly `verifier.score(completion, scenario).composite` (see [REWARD_DESIGN.md](REWARD_DESIGN.md)), with no difficulty bonus or re-weighting.
- The three rubrics are passed as extra reward functions (`reward_correct_final_state`, `reward_conflict_identification`, `reward_no_contradictions`) with weight 0, so TRL logs each one separately without changing the total.
- If scoring a completion raises, the reward for it is 0, a WARNING with the first 200 characters of the completion is logged and counted, and training stops after 10 failures in a row.

## 6. Checkpoints, resume and the best checkpoint

- **Resume.** On start, the trainer looks in `output_dir` for the highest `checkpoint-N` folder that has a `trainer_state.json` and resumes from it. Re-running the same command after an interrupted session continues the run. `--no-resume` ignores existing checkpoints. On Colab, put `output_dir` on Google Drive (`--output-dir /content/drive/MyDrive/...`) so checkpoints survive a reset.
- **Best checkpoint.** Every `eval_steps` the trainer computes the mean composite reward on the val split (`eval_reward`). `load_best_model_at_end` is on with `metric_for_best_model = eval_reward` (higher is better), so at the end the checkpoint with the best val reward is reloaded. `save_total_limit` checkpoints are kept on disk.
- After training the pipeline runs one more val pass on that checkpoint, saves the adapter and tokenizer to `<output_dir>/final`, and prints the next step.

## 7. What a run writes

`results/<run_name>/`:

| File | Content |
|---|---|
| `train_config.yaml` | the resolved config, overrides included |
| `git_commit.txt` | commit of the code (or `CONFLICTBENCH_REF` if git is unavailable) |
| `pip_freeze.txt` | exact package versions |
| `environment.json` | Python version, platform, start time |
| `dataset_stats.json` | per-split, per-difficulty dataset counts |
| `final_metrics.json` | best checkpoint and its val metric, step count, final val metrics, reward-failure count |
| `log_history.json` | TRL's full log history |

If `hub_repo` is set and `HF_TOKEN` is in the environment, `<output_dir>/final` is uploaded to that Hub repo (created as private). Without a token nothing is pushed.

## 8. Evaluate on the test set (once, at the end)

```bash
python scripts/evaluate.py --model <the base model of your config> --adapter <output_dir>/final \
    --data data/test_v1.jsonl --run-name <run_name>_test
python scripts/make_leaderboard.py
```

This scores every test scenario with greedy decoding and the same prompt, and writes `results/<run_name>_test/summary.json` and `completions.jsonl`. Compare it with the baselines on the same data set in `results/leaderboard.md`. More options (sampling for pass@k, API models): [`results/README.md`](../results/README.md).

## Training Space

[`training_space/`](../training_space/) is the source of a separate Hugging Face Docker Space with a Start button.

- **Reproducible build.** The Dockerfile installs pinned torch packages, clones this repo at the build argument `CONFLICTBENCH_REF`, and installs `requirements-train.txt` from that checkout. `CONFLICTBENCH_REF` must be a git tag or commit SHA; the build stops if it is unset or a branch name (`main`, `master`, `HEAD`). In the Space, set it under Settings → Variables and secrets. To cut a tag for a run: `git tag train-run-1 && git push origin train-run-1`, then set `CONFLICTBENCH_REF=train-run-1`.
- **What it runs.** `train_script.py` loads `configs/grpo_a10g.yaml` from that checkout and calls the same `training.train_grpo.train`. The settings table in the UI comes from constants in `train_script.py`; the run, and `tests/test_training_space_lock.py`, refuse to go ahead if they differ from the YAML.
- **Cost lock.** The Start button works only with the password in the Space secret `TRAIN_PASSWORD`; without that secret, training is disabled. Also keep the Space private or paused when not in use (a manual setting on huggingface.co).
- **Hub upload.** Optional: set `HF_REPO_ID` (and the `HF_TOKEN` secret) to push the adapter.
