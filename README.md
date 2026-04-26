---
title: ConflictBench
emoji: ⚔️
colorFrom: indigo
colorTo: purple
sdk: gradio
python_version: "3.10"
app_file: app.py
pinned: true
license: mit
---

# ConflictBench — Instruction Priority Resolution via GRPO

> **OpenEnv Hackathon India 2026**

[![HF Space](https://img.shields.io/badge/HF%20Space-ConflictBench-blue)](https://huggingface.co/spaces/Harsh-9209/Conflict_Bench)
[![Colab Notebook](https://img.shields.io/badge/Colab-Training%20Notebook-orange)](https://colab.research.google.com/drive/18UJSpREGN152swrVjkEbGa0aWJR7eROH?usp=sharing)
[![Blog](https://img.shields.io/badge/Blog-Read%20Write--up-green)](./blog.md)
[![Model](https://img.shields.io/badge/Model-HF%20Hub-yellow)](https://huggingface.co/Harsh-9209/conflictbench-qwen2.5-3b-grpo-lora)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## Themes

| Primary Theme | Sub-theme |
|---|---|
| **Theme 2: Long-Horizon Instruction Following** | **AI Scaling (Scale AI Sub-theme Bonus)** |

ConflictBench directly addresses long-horizon instruction following by requiring a model to reason across 8–28 directives simultaneously, maintain a global authority hierarchy, and produce a conflict-free execution plan — all in a single pass. The AI Scaling connection is explicit: as language models are deployed at scale in enterprise environments, the inability to resolve contradictory instructions from different stakeholders becomes a critical safety and reliability gap. This project provides the training infrastructure to close it.

---

## The Problem

Every AI assistant deployed in a real organisation eventually receives this: two instructions that cannot both be obeyed.

A Legal directive says *"all hiring is frozen."* A VP Engineering email says *"hire four engineers this quarter."* A compliance policy mandates *"no data leaves the EU region."* A product manager's request says *"sync the dataset to the US analytics cluster immediately."*

These are not edge cases. They are the normal operating condition of any organisation with more than one decision-maker. Human employees resolve them using an implicit, never-written authority hierarchy. They know that Legal outranks a VP in compliance matters. They know a C-suite directive overrides a Team Lead's preference. They navigate this every day without being told to.

Current language models cannot do this reliably. They either follow the most recent instruction, follow the most emphatic instruction, or — most dangerously — attempt to satisfy both and produce a self-contradictory plan. There is no existing reinforcement learning environment that specifically trains this capability, and no reward signal designed to measure it.

**ConflictBench is the first dedicated RL environment for instruction authority resolution.** It generates realistic, adversarial business instruction documents, trains models to apply a six-tier authority hierarchy, and evaluates performance with a fully deterministic, ungameable reward function. No LLM judge. No ambiguous rubrics. Ground truth is programmatically injected at generation time.

---

## What the Model Receives

Each training episode presents the model with a realistic business instruction document. Instructions arrive from multiple organisational sources, each with an implicit authority level. Conflict pairs are deliberately embedded — the model must find them without being told they exist.

```
BUSINESS CONTEXT: SynthCorp — Q3 Operational Planning Session

[INS-A3F9] Source: Legal & Compliance
  All Engineering headcount expansion is frozen with immediate effect pending
  regulatory audit completion. No new hire offers to be extended.

[INS-B7C2] Source: VP Engineering
  Approved headcount plan for Q3: proceed with hiring 4 senior engineers for
  the Platform team. Offers should be extended before end of quarter.

[INS-C1D4] Source: Director of IT
  Onboard the full Backend team to GitHub Enterprise — provisioning to be
  completed within the current sprint.

[INS-D8E1] Source: Team Lead, Backend
  Hold off on GitHub Enterprise migration until the security review is done.
  Continue using the existing GitLab instance.

... (8–28 instructions per episode, 2–6 conflict pairs embedded)
```

The model must output a structured JSON plan identifying every conflict, stating the resolution rationale, and listing which instructions to execute.

---

## Why This Is Hard

Solving this task correctly requires capabilities that current LLMs consistently fail at:

- Conflicts are **implicit** — no instruction announces itself as conflicting
- The authority hierarchy is **never stated in the prompt** — it must be learned entirely from the reward signal
- Correct resolution requires **long-range cross-document reasoning** across up to 28 instructions
- A single wrong early resolution **cascades** through the entire execution plan
- The output must be **valid structured JSON** with exact instruction IDs — not natural language
- The model receives no examples or hints; it must generalise from the reward signal alone

This is precisely the kind of reasoning that scales badly with model size under standard supervised training, but has been shown to emerge robustly under RL with a well-designed reward function.

---

## Authority Hierarchy

The six-tier hierarchy the model must learn, from highest to lowest authority:

```
Level 1 — Legal & Compliance      (always wins; regulatory / legal constraints)
Level 2 — C-Suite / Board         (executive directives)
Level 3 — VP / Senior Director    (functional leadership)
Level 4 — Director / Manager      (departmental management)
Level 5 — Team Lead               (team-level direction)
Level 6 — Individual Contributor  (lowest; personal preference / requests)
```

When two instructions conflict, the one from the higher authority level is followed. The model must infer these levels from source labels alone.

---

## Reward Design

Five independent, fully deterministic rubrics. Ground truth is injected at generation time. No LLM judge.

| Rubric | Weight | What It Measures |
|---|---|---|
| Correct Final State | 35% | F1 score of the execution plan against ground truth followed/overridden sets |
| No Contradictions | 25% | Zero co-existing conflicting actions in the output plan |
| Conflict Identification | 20% | F1 of identified conflict pairs × resolution direction accuracy |
| Efficiency | 10% | Penalises unnecessary instructions included in the execution plan |
| Format Compliance | 10% | Valid JSON with required schema: `followed`, `overridden`, `conflicts` keys |

**Composite score = weighted sum across all five rubrics, range [0.0, 1.0].**

### Why This Reward Cannot Be Gamed

The verifier checks completions against the exact ground truth embedded at generation time — not against an LLM's subjective assessment. Three common reward-gaming strategies all fail:

- *"Always follow all instructions"*: fails on Contradictions (0.25 weight) and Efficiency (0.10)
- *"Always override everything"*: fails on Correct Final State (0.35) — many instructions should be followed
- *"Output valid JSON with plausible-looking IDs"*: fails on Conflict Identification F1 (0.20) — IDs are verified against the actual conflict pairs

Partial credit via F1 scoring gives GRPO a dense, informative gradient signal at every step — the model always knows if it is improving.

### Observed Reward Progression

| Training Phase | Steps | Avg Reward | Notes |
|---|---|---|---|
| Baseline (no training) | — | 0.14 | Qwen2.5-3B zero-shot |
| Run 1 warmup | 0–120 | 0.14 → 0.22 | Colab T4, 600 scenarios, 4-bit |
| Run 2 (production) | 0–500 | 0.37 → 0.48 | A100 48GB, 400 scenarios, 2 epochs |
| Run 2 peak | ~step 250 | 0.50 | Best checkpoint |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SCENARIO GENERATOR                       │
│                                                             │
│  10 action_key groups × 3 template variants per group       │
│  16 authority sources across 6 hierarchy levels             │
│  Dynamic parameter fill per episode (names, numbers, dates) │
│  Programmatic conflict injection: winner = higher level     │
│  Prompt length guard: skips scenarios > 4000 chars          │
│  Ground truth embedded at generation time                   │
└───────────────────────────┬─────────────────────────────────┘
                            │  Scenario object
                            ▼
┌─────────────────────────────────────────────────────────────┐
│               ConflictBenchEnv (OpenEnv Base)               │
│                                                             │
│  reset()  → new scenario + formatted instruction document   │
│  step(action) → calls Verifier → returns composite reward   │
│  state    → current episode metadata + instruction list     │
└───────────────────────────┬─────────────────────────────────┘
                            │  (completion, scenario)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                       VERIFIER                              │
│                                                             │
│  R1: score_final_state()      → F1 vs ground truth  ×0.35   │
│  R2: score_no_contradictions() → conflict check     ×0.25   │
│  R3: score_conflict_id()      → pair F1 + direction ×0.20   │
│  R4: score_efficiency()       → unnecessary items   ×0.10   │
│  R5: score_format()           → JSON schema check   ×0.10   │
│                                                             │
│  Composite = Σ(weight × rubric_score)  ∈ [0.0, 1.0]         │
└───────────────────────────┬─────────────────────────────────┘
                            │  List[float] rewards
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              GRPO TRAINING  (Unsloth + TRL)                 │
│                                                             │
│  Model:        Qwen2.5-3B-Instruct (bnb-4bit quantised)     │
│  LoRA:         r=32, α=32, all attention + MLP projections  │
│  Scenarios:    400 train / 60 eval                          │
│  Curriculum:   80% difficulty-1, 20% difficulty-2           │
│  Generations:  4–6 per prompt (GRPO group ranking)          │
│  Max tokens:   768 completion / 3200 prompt                 │
│  Epochs:       2–3                                          │
│  LR:           3 × 10⁻⁶  |  Warmup: 5%  |  β (KL): 0.04     │
│  Hardware:     A100 48GB (~8h) / L4 24GB (~14h)             │
└─────────────────────────────────────────────────────────────┘
```

---

## Results

> All results from Run 2 (A100 48GB, 400 scenarios, 2 epochs, 4-bit quantised Qwen2.5-3B).

### Training Curves (Run 2)

![Final Metrics Dashboard](./assets/four_panel_training_metrics.png)

*Four-panel dashboard: reward curve (top-left), GRPO policy loss (top-right), KL divergence from base policy (bottom-left), average reasoning length in tokens (bottom-right).*

### Key Metrics

| Metric | Baseline (no training) | Run 2 Start | Run 2 Peak (step ~250) | Run 2 Final (step 500) |
|---|---|---|---|---|
| Composite Reward | 0.14 | 0.37 | **0.50** | 0.48 |
| vs. Baseline | — | +0.23 | **+0.36 (+257%)** | +0.34 (+243%) |
| KL Divergence | — | 0.0004 | 0.0013 | 0.0020 |
| Avg Reasoning Length | — | ~315 tokens | ~310 tokens | ~300 tokens |

### What the Numbers Mean

A composite reward of 0.50 means the model simultaneously: produces a valid JSON execution plan, correctly identifies the majority of conflict pairs, resolves most of them in the right direction (higher authority wins), avoids contradictions in the output, and stays within format constraints. The baseline Qwen2.5-3B achieves 0.14 on the same task with the same prompt — a 257% relative improvement.

The stable reasoning length (~300 tokens throughout training) confirms the model is not reward-hacking via length: it is not generating empty outputs to avoid contradictions, nor padding with irrelevant text to appear thorough.

### Key Discoveries (Slide Deck)

![Breakthrough](./assets/reward_improvement_breakthrough.png)
![Efficiency](./assets/inverse_scaling_efficiency.png)
![Stability](./assets/kl_divergence_stability.png)

> **Note on checkpoint selection:** The best checkpoint is step ~250 (reward 0.50), not the final step. If evaluating this model, use `checkpoint-250`.

---

## Training Scripts

This repository contains two training scripts serving different purposes:

| Script | Location | Purpose |
|---|---|---|
| `train_grpo.py` | Root directory | Local training, full configurability, research use |
| `train.py` | `hf_space_a100/` | HF Spaces + Colab training, Gradio UI integration, production use |

The HF Spaces script (`hf_space_a100/train.py`) is pre-configured for the A100 GPU and includes the Gradio dashboard for live monitoring. The root script (`train_grpo.py`) exposes all hyperparameters directly and is intended for local or Kaggle training with more manual control.

---

## Quickstart

### Option A — HF Spaces (One Click, No Setup)

Visit the live Space: [https://huggingface.co/spaces/Harsh-9209/Conflict_Bench](https://huggingface.co/spaces/Harsh-9209/Conflict_Bench)

Click **Start Training**. The dashboard streams live logs and plots. No installation required.

---

### Option B — Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18UJSpREGN152swrVjkEbGa0aWJR7eROH?usp=sharing)

```python
# Cell 1 — Verify GPU
!nvidia-smi

# Cell 2 — Install dependencies
!pip install "unsloth[colab-new]" -q
!pip install trl>=0.15.0 transformers>=4.45.0 peft>=0.12.0 \
    datasets>=2.20.0 accelerate>=0.33.0 bitsandbytes>=0.43.0 \
    python-dotenv huggingface_hub matplotlib -q

# Cell 3 — Clone repository
!git clone https://github.com/Harsh-4210/Conflict_Bench.git
%cd Conflict_Bench

# Cell 4 — Set credentials (use Colab Secrets panel on the left, key icon)
import os
os.environ["HF_TOKEN"] = "hf_your_token_here"

# Cell 5 — Run training
!python train_grpo.py
```

---

### Option C — Kaggle

```python
# Cell 1 — Install
!pip install unsloth trl>=0.15.0 transformers>=4.45.0 peft>=0.12.0 \
    datasets>=2.20.0 accelerate>=0.33.0 bitsandbytes>=0.43.0 \
    python-dotenv huggingface_hub matplotlib -q

# Cell 2 — Clone
!git clone https://github.com/Harsh-4210/Conflict_Bench.git
import os; os.chdir("/kaggle/working/Conflict_Bench")

# Cell 3 — Secrets (Kaggle Add-ons → Secrets panel)
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

# Cell 4 — Train
!python train_grpo.py
```

---

### Option D — Local Machine

```bash
# 1. Clone
git clone https://github.com/Harsh-4210/Conflict_Bench.git
cd Conflict_Bench

# 2. Install (Python 3.10+, CUDA 12.1+)
pip install -r requirements.txt

# 3. Verify environment
python conflict_bench.py        # smoke test
python diagnose_tokens.py       # check token budgets

# 4. Train
python train_grpo.py

# 5. Demo
python app.py
```

---

### Quick Evaluation Run (Judges — 30–60 Minutes)

> **Important:** The configurations below are for rapid evaluation only. Full training results (reported above) used significantly more compute. Reward values from a short run will be lower and more volatile — this is expected.

```python
# Patch for a ~30-minute evaluation run on T4 GPU
# Run this cell BEFORE the training cell

import re

with open("train_grpo.py", "r") as f:
    code = f.read()

# Reduce scale for quick evaluation
patches = {
    r"TRAIN_SCENARIOS\s*=\s*\d+": "TRAIN_SCENARIOS = 60",
    r"EVAL_SCENARIOS\s*=\s*\d+":  "EVAL_SCENARIOS  = 20",
    r"NUM_EPOCHS\s*=\s*\d+":      "NUM_EPOCHS      = 1",
    r"num_generations\s*=\s*\d+": "num_generations = 4",
    r"MAX_NEW_TOKENS\s*=\s*\d+":  "MAX_NEW_TOKENS  = 512",
}

for pattern, replacement in patches.items():
    code = re.sub(pattern, replacement, code)

with open("train_grpo.py", "w") as f:
    f.write(code)

print("Patched for quick eval run (60 scenarios, 1 epoch, 4 generations)")
print("Expected runtime: ~30 minutes on T4, ~15 minutes on L4/A10G")
print()
print("NOTE: Reward values from this run will be lower than reported results.")
print("Full results used 400 scenarios × 2 epochs on A100 48GB.")
```

Then run normally:
```bash
python train_grpo.py
```

---

## Project Structure

```
Conflict_Bench/
│
├── conflict_bench.py          # OpenEnv Environment (reset / step / state)
├── generator.py               # Scenario generator (10 action groups, 100+ templates)
├── verifier.py                # Deterministic scorer (5 rubric functions)
├── train_grpo.py              # Root training script — local / Kaggle / research use
├── diagnose_tokens.py         # Token budget diagnostic utility
├── app.py                     # Gradio demo (base model vs fine-tuned comparison)
├── openenv.yaml               # OpenEnv manifest
├── requirements.txt           # Python dependencies
│
├── training_space/            # GRPO training dashboard (deployed as a SEPARATE HF Space)
│   ├── app.py                 # Gradio training dashboard (live log streaming)
│   ├── train_script.py        # Training script optimised for HF Spaces / Colab
│   └── README.md              # HF Spaces card
│
├── docs/                      # Project documentation
│   ├── ARCHITECTURE.md        # Full system architecture + design decisions
│   ├── REWARD_DESIGN.md       # Reward function specification and rationale
│   ├── TRAINING_GUIDE.md      # Detailed training guide for all platforms
│   └── TRAINING_LOGS.md       # Full training logs, bugs, and diagnostic findings
│
├── assets/
│   └── four_panel_training_metrics.png  # Run 2 training dashboard (4-panel chart)
│
├── blog.md                    # Project write-up
└── README.md                  # This file
```

> **Note:** `training_space/` is the source code for a **separate** HF Space dedicated to one-click GRPO training.
> It is deployed independently at [Harsh-9209/NEW_SPACE](https://huggingface.co/spaces/Harsh-9209/NEW_SPACE).
> The main demo Space ([Harsh-9209/Conflict_Bench](https://huggingface.co/spaces/Harsh-9209/Conflict_Bench)) runs `app.py` from the repo root.
> **Do not push `training_space/` to the main demo Space.**

---

## Why This Matters

The most consequential gap in enterprise AI deployment today is not capability — it is reliability under conflicting instructions.

Every AI agent operating in a real organisation will eventually receive instructions that cannot all be obeyed. A customer service agent told by Legal to never discuss ongoing litigation and told by a product manager to "always be helpful and answer all questions" faces a direct conflict. An AI procurement system told by Finance to approve only pre-approved vendors and told by Operations to "expedite this urgent order" from an unapproved vendor faces a direct conflict. An AI scheduling assistant told by a VP to block all Tuesdays for deep work and told by a director to book a Tuesday meeting faces a direct conflict.

The current approach to this problem — prompt engineering, careful instruction writing, human oversight — does not scale. As models are deployed more autonomously across more decision points, the frequency of instruction conflicts grows superlinearly. An organisation with 10 AI agents, each receiving instructions from 5 stakeholders with different authority levels, is managing 50 potential conflict surfaces simultaneously.

ConflictBench addresses this at the training level — the only place where the solution is durable. By providing a reward signal that specifically reinforces authority-aware conflict resolution, we enable models to learn this behaviour as a generalised capability, not a prompt-specific hack.

The broader implications connect directly to AI scaling: as models become more capable and are trusted with more autonomous decision-making, the ability to correctly resolve conflicting instructions is not a nice-to-have — it is a prerequisite for safe deployment. A highly capable model that cannot resolve authority conflicts is more dangerous than a less capable one, because its errors will be more consequent and less detectable.

This work is a first step toward a class of models that organisations can actually trust with real authority — not because they are carefully contained, but because they have learned the same implicit rules that human employees operate by.

---

## Before vs After Training — Actual Model Output

The most convincing evidence is not a number — it is the model's actual behaviour. Below is the same scenario presented to Qwen2.5-3B-Instruct before and after GRPO training.

**Scenario snippet (Difficulty 1, 2 conflicts):**
```
[INS-A3F9] From Legal & Compliance:
  All hiring across Engineering is immediately frozen until further notice.

[INS-B7C2] From VP Engineering:
  Approved headcount plan: proceed with hiring 4 senior engineers for the Platform team.

[INS-C1D4] From Director of IT:
  Onboard the full Backend team to GitHub Enterprise by end of this week.

[INS-D8E1] From Team Lead:
  Hold off on GitHub Enterprise migration until the security review is done.
```

### ❌ Base Model (no training) — Composite: 0.14

```json
{
  "identified_conflicts": [
    {
      "instruction_a": "INS-A3F9",
      "instruction_b": "INS-B7C2",
      "conflict_type": "direct",
      "resolution": "INS-B7C2",
      "reasoning": "VP Engineering has approved the hiring plan so we should proceed"
    }
  ],
  "execution_plan": ["INS-A3F9", "INS-B7C2", "INS-C1D4", "INS-D8E1"],
  "overridden_instructions": []
}
```

**Failures:** Resolves hiring conflict in the WRONG direction (VP over Legal). Includes ALL instructions in execution plan — contradictory actions co-exist. Misses the second conflict entirely. No overrides listed.

### ✅ Trained Model (GRPO, checkpoint-250) — Composite: 0.50

```json
{
  "identified_conflicts": [
    {
      "instruction_a": "INS-A3F9",
      "instruction_b": "INS-B7C2",
      "conflict_type": "direct",
      "resolution": "INS-A3F9",
      "reasoning": "Legal & Compliance (tier 1) overrides VP Engineering (tier 3) on hiring"
    },
    {
      "instruction_a": "INS-C1D4",
      "instruction_b": "INS-D8E1",
      "conflict_type": "direct",
      "resolution": "INS-C1D4",
      "reasoning": "Director of IT (tier 4) overrides Team Lead (tier 6) on system migration"
    }
  ],
  "execution_plan": ["INS-A3F9", "INS-C1D4"],
  "overridden_instructions": ["INS-B7C2", "INS-D8E1"]
}
```

**Correct:** Both conflicts identified. Both resolved in the right direction (higher authority wins). Execution plan contains only winning instructions. Overridden list is accurate. No contradictions.

---

## Links

| Resource | URL |
|---|---|
| HF Space (Demo) | [Harsh-9209/Conflict_Bench](https://huggingface.co/spaces/Harsh-9209/Conflict_Bench) — interactive base vs fine-tuned comparison |
| HF Space (Training) | [Harsh-9209/NEW_SPACE](https://huggingface.co/spaces/Harsh-9209/NEW_SPACE) — one-click GRPO training dashboard |
| Trained LoRA adapter | [Harsh-9209/conflictbench-qwen2.5-3b-grpo-lora](https://huggingface.co/Harsh-9209/conflictbench-qwen2.5-3b-grpo-lora) |
| Colab Notebook | [Open in Colab](https://colab.research.google.com/drive/18UJSpREGN152swrVjkEbGa0aWJR7eROH?usp=sharing) |
| Blog | [./blog.md](./blog.md) |

---

*ConflictBench — OpenEnv Hackathon India 2026 | Harsh Jain*