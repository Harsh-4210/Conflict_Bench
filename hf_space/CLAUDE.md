# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

ConflictBench GRPO Trainer — a Gradio-based HuggingFace Space that fine-tunes `Qwen2.5-3B-Instruct` using GRPO (Group Relative Policy Optimization) on the ConflictBench instruction priority resolution task.

## Environment

Requires CUDA GPU with at least 24GB VRAM (L40S 48GB recommended). Set these secrets in HuggingFace Space settings:
- `HF_TOKEN` — HuggingFace write token (required for model upload)
- `HF_REPO_ID` — target repo (defaults to `Harsh-9209/conflictbench-qwen2.5-3b-grpo-lora`)
- `WANDB_API_KEY` — Weights & Biases key (optional; training defaults to `report_to="none"`)

## Running

```bash
pip install -r requirements.txt
python app.py          # launches Gradio UI (calls train.run_training in background thread)
python train.py        # runs full pipeline directly via __main__
```

## Architecture

Two files, clean separation of concerns:

**`app.py`** — Gradio UI only. Manages global state (`LOGS`, `TRAINING_ACTIVE`, `PLOT_PATHS`), launches `train.run_training(progress_callback=...)` in a daemon thread, and exposes manual refresh buttons for logs and plots. The UI displays up to 4 plot images: Reward Curve, Loss Curve, KL Divergence, Training Dashboard.

**`train.py`** — All training logic. Entry point is `run_training(progress_callback=None)`, which runs five sequential stages:

1. **`clone_repo()`** — git-clones `github.com/Harsh-4210/Conflict_Bench` to `/tmp/conflictbench_repo` and injects it into `sys.path`. This makes `generator` and `verifier` importable (they live in the cloned repo, not this one).
2. **`detect_gpu_config()`** — auto-selects batch size, gradient accumulation, and whether to use 4-bit quantization based on VRAM (≥40GB → full precision, 2×batch; otherwise 4-bit).
3. **`build_dataset()`** — imports `generator.ScenarioGenerator` from the cloned repo, generates 400 train (80% D1 / 20% D2 curriculum) + 60 eval (30/50/20 D1/D2/D3) scenarios. Scenarios >4000 chars are skipped.
4. **Training** — `GRPOTrainer` with a single composite reward function (`verifier.score(...).composite`). A `GradioLogCallback` appends step metrics to the module-level `_training_log` list and forwards them to `progress_callback`. LoRA targets all attention + MLP projections (`q/k/v/o_proj`, `gate/up/down_proj`).
5. **`generate_plots()`** — reads `_training_log` to produce 4 matplotlib PNG files in `./grpo-out/plots/`: reward curve, loss curve, KL divergence, and a 2×2 dashboard.
6. **Upload** — uploads the latest `checkpoint-*` directory (or `final/` fallback) to HF Hub, skipping optimizer/scheduler state files.

### Key constants (`train.py`)

| Name | Value |
|------|-------|
| `MAX_SEQ_LENGTH` | 4096 |
| `MAX_NEW_TOKENS` | 768 |
| `MAX_PROMPT_LENGTH` | 3200 |
| `BETA` (KL penalty) | 0.02 |
| `LEARNING_RATE` | 3e-6 |
| `OUTPUT_DIR` | `./grpo-out` |

### External dependencies from cloned repo

`generator.ScenarioGenerator`, `generator.Scenario`, `generator.Instruction`, `generator.ConflictPair`, and `verifier.score` are **not in this repo** — they are imported after `clone_repo()` inserts the cloned path into `sys.path`. Any changes to scenario generation or scoring logic must be made in `github.com/Harsh-4210/Conflict_Bench`.
