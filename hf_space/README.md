---
title: ConflictBench GRPO Trainer
emoji: ⚔️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: true
license: mit
hardware: l40s-48gb
---

# ConflictBench — GRPO Training Space

**One-click GRPO fine-tuning** for the ConflictBench instruction priority resolution task.

## What This Does

1. **Clones** the [ConflictBench repo](https://github.com/Harsh-4210/Conflict_Bench) with all production-grade bug fixes
2. **Generates** 400 adversarial training scenarios using the 6-tier authority hierarchy
3. **Trains** Qwen2.5-3B-Instruct with GRPO (Group Relative Policy Optimization)
4. **Produces** all presentation plots (reward curve, per-rubric breakdown, training metrics)
5. **Uploads** the best LoRA adapter to HuggingFace Hub

## Hardware Requirements

| GPU | VRAM | Estimated Time | Cost |
|-----|------|---------------|------|
| L40S | 48GB | ~8 hours | ~$14 |
| A10G | 24GB | ~12 hours | ~$14 |
| L4 | 24GB | ~14 hours | ~$11 |

## Environment Variables

Set these in your Space settings → **Repository Secrets**:
- `HF_TOKEN` — Your HuggingFace write token (for model upload)
- `WANDB_API_KEY` — *(Optional)* Weights & Biases key for experiment tracking

## Training Configuration

- **Model**: `unsloth/Qwen2.5-3B-Instruct-bnb-4bit`
- **Method**: GRPO with 6 reward rubrics
- **Scenarios**: 400 train / 60 eval (80% difficulty-1, 20% difficulty-2 curriculum)
- **LoRA**: r=32, targeting all attention + MLP projections
- **Epochs**: 3
