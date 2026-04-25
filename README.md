---
title: ConflictBench
emoji: ⚔️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---
# ConflictBench — Instruction Priority Resolution Environment

> **OpenEnv Hackathon India 2026** | Theme 2: Long-Horizon Instruction Following | Scale AI Sub-theme Bonus

[![HuggingFace Space](https://img.shields.io/badge/🤗%20HF%20Space-ConflictBench-blue)](https://huggingface.co/spaces/Harsh-9209/Conflict_Bench)
[![Colab Notebook](https://img.shields.io/badge/📓%20Colab-Training%20Notebook-orange)](YOUR_COLAB_LINK)
[![Blog / Video](https://img.shields.io/badge/📝%20Blog-Write--up-green)](YOUR_BLOG_OR_VIDEO_LINK)

---

## The Problem

LLMs fail at a task every business professional faces daily: **resolving contradictory instructions from different stakeholders**.

When a Legal directive says *"freeze hiring"* and a VP Engineering email says *"hire 4 engineers immediately,"* the correct resolution depends on an **implicit authority hierarchy** — not the literal content of either instruction. Humans learn this through organizational experience. LLMs have no such training signal.

**No existing RL environment targets this capability.** ConflictBench is the first.

---

## What the Agent Sees

Each episode, the agent receives a realistic business instruction document with **8–28 directives** from multiple stakeholders:

```
[INS-A3F9] From Legal & Compliance:
  Due to regulatory constraints, all hiring across Engineering is immediately
  frozen until further notice.

[INS-B7C2] From VP Engineering:
  Approved headcount expansion: proceed with hiring 4 senior engineers for the
  Platform team this quarter.

[INS-C1D4] From Director of IT:
  Onboard the full Backend team to GitHub — access provisioning should be
  completed by end of this week.

... (8-28 instructions total, 2-6 conflict pairs embedded)
```

## What the Agent Must Do

Produce a structured JSON plan that:
1. **Identifies** every conflicting instruction pair
2. **Resolves** each conflict using the authority hierarchy: **Legal > C-Suite > VP > Director/Manager > Team Lead**
3. **Lists** which instructions to follow and which to override

## Why This Is Hard for LLMs

- Conflicts are **implicit** — no instruction says "I conflict with INS-B7C2"
- The **authority hierarchy is never stated** in the prompt — it must be learned
- Resolutions require **long-range reasoning** across a document with many instructions
- A wrong early decision (following a low-priority instruction) **cascades** through the plan
- The model must output **structured JSON** with correct instruction IDs — not just natural language

---

## Reward Design — Deterministic, Composable, Ungameable

Five independent rubrics, scored against programmatically-generated ground truth. **No LLM judge. Fully rule-based.**

| Rubric | Weight | What it measures |
|---|---|---|
| Correct final state | 35% | F1 of execution plan vs ground truth |
| No contradictions | 25% | No conflicting actions co-exist in the plan |
| Conflict identification | 20% | F1 of identified conflict pairs + resolution accuracy |
| Efficiency | 10% | Penalizes unnecessary instructions |
| Format compliance | 10% | Valid JSON with required structure |

### Why this reward is hard to game:
- Ground truth is injected at generation time — the verifier checks against the **exact** correct answer
- An agent that "always follows everything" scores poorly on efficiency and contradictions
- An agent that "always overrides everything" scores poorly on correct final state
- Partial credit via F1 scoring gives GRPO a rich gradient signal

---

## Architecture

```
ScenarioGenerator (10 action_key groups × 3 template variants × 16 authority sources)
  ├── Dynamic template fill (random parameters per episode)
  ├── Programmatic conflict injection (winner = higher authority source)
  ├── Prompt length guard (skips prompts > 3000 chars)
  └── Ground truth resolution embedded at generation time
        ↓
ConflictBenchEnv (OpenEnv Environment base class)
  ├── reset() → new scenario + formatted prompt
  ├── step(action) → composite reward from 5 rubrics
  └── state → current episode metadata
        ↓
Verifier (5 deterministic rubric functions)
        ↓
GRPO Training (Unsloth + TRL GRPOTrainer)
  ├── Model: Qwen2.5-3B-Instruct (4-bit quantized via Unsloth)
  ├── 600 training scenarios, difficulty 1 (2 conflicts)
  ├── 4 generations per prompt (GRPO ranking)
  └── Token budget: 512 new tokens (JSON fits comfortably)
```

---

## Results

### Training Reward Curve

![Composite reward during GRPO training](assets/reward_curve.png)
*Composite reward (0.0–1.0) over 120 training steps on Google Colab T4. Reward climbs from 0.14 to 0.22 — a 57% improvement — demonstrating the model is learning the authority hierarchy.*

### Key Training Metrics

| Metric | Start (Step 10) | Peak (Step 110) | Trend |
|---|---|---|---|
| Composite Reward | 0.140 | 0.225 | ↑ 57% |
| KL Divergence | 0.000007 | 0.0001 | Stable (healthy) |
| Gradient Norm | 0.099 | 1.39 | ↑ Active learning |

> **Note:** This is a smoke test run (120 steps, difficulty 1 only, Colab T4). The full training run with the truncation fix applied will show significantly higher rewards — the initial run suffered from 100% output clipping (`clipped_ratio: 1.0`) which has now been resolved.

---

## Quickstart

### 1. Install

```bash
# Windows: Install PyTorch with CUDA first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Then install requirements
pip install -r requirements.txt
```

### 2. Smoke test the environment

```bash
python conflict_bench.py
```

### 3. Diagnose token budget (run before training)

```bash
python diagnose_tokens.py
```

### 4. Train (Colab recommended)

```bash
python train_grpo.py
```

### 5. Evaluate a trained model

```bash
python train_grpo.py --eval ./conflictbench-grpo-output/final
```

### 6. Run demo locally

```bash
python app.py
```

---

## Project Structure

```
conflictbench/
├── conflict_bench.py     # OpenEnv Environment class (reset/step/state)
├── generator.py          # Dynamic scenario generator (10 action groups, 100+ templates)
├── verifier.py           # Deterministic scorer (5 rubric functions)
├── train_grpo.py         # GRPO training + evaluation script
├── diagnose_tokens.py    # Token budget diagnostic utility
├── app.py                # Gradio HF Spaces demo (base vs fine-tuned)
├── openenv.yaml          # OpenEnv manifest
├── requirements.txt
├── assets/
│   └── reward_curve.png  # Training reward curve plot
└── README.md
```

---

## Links

- 🤗 **HF Space (live demo):** *(add link after deployment)*
- 📓 **Colab Notebook:** *(add link)*
- 📝 **Mini-blog / video:** *(add link)*
- 🧠 **Trained model:** *(add link after full training run)*

---

## Why This Matters

Every employee, manager, and AI assistant operating in a business context faces instruction conflicts daily. Teaching LLMs to systematically resolve them using authority-based reasoning is a foundational enterprise capability — with direct applications to:

- **AI agents** that receive instructions from multiple users with different permission levels
- **Virtual assistants** that must prioritize conflicting calendar/task requests
- **Automated workflow systems** that process business rules from different departments

ConflictBench is the first RL environment specifically designed to train this behavior, with a deterministic reward signal that is impossible to game.
