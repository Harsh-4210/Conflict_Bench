---
title: ConflictBench GRPO (L40S)
emoji: ⚔️
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

# ConflictBench GRPO Training Pipeline

One-click automated GRPO training pipeline tailored for an **NVIDIA L40S** GPU.
This pipeline automatically clones the ConflictBench environment, generates scenarios based on a defined curriculum, runs GRPO fine-tuning using Unsloth, generates loss/reward plots, and pushes the final LoRA weights to the Hugging Face Hub.
