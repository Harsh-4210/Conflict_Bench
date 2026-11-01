---
title: ConflictBench GRPO (A100)
emoji: ⚔️
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

# ConflictBench GRPO Training Space

A Gradio front end for the ConflictBench training pipeline. Pressing **Start Training** (password required) calls the repo's shared training pipeline (`training.train_grpo.train`) with `configs/grpo_a10g.yaml` (Qwen2.5-3B in 4-bit, LoRA rank 16, GRPO via Unsloth + TRL), streams the logs, and shows reward, loss and KL plots when the run ends. The config is sized for a 24 GB GPU (A10G / L4); larger GPUs also work. It has not been run end to end yet.

Setup, in the Space settings (Settings → Variables and secrets):

- `CONFLICTBENCH_REF` (variable, required): the git tag or commit SHA of the ConflictBench repo to build from. The Docker build stops if it is missing or a branch name.
- `TRAIN_PASSWORD` (secret, required): without it the Start button is disabled.
- `HF_REPO_ID` and `HF_TOKEN` (optional): push the trained adapter to that Hub repo. Without them nothing is uploaded.

Keep the Space private or paused when not in use. Full guide: `docs/TRAINING_GUIDE.md` in the main repository (section "Training Space").
