---
title: ConflictBench
emoji: ⚔️
colorFrom: blue
colorTo: purple
sdk: gradio
python_version: "3.10"
app_file: app.py
pinned: false
---
# ConflictBench: Authority-Aware Business Reasoning

> **The first OpenEnv-compliant Reinforcement Learning framework dedicated to resolving multi-stakeholder business instruction conflicts using GRPO.**

[![Colab Notebook](https://img.shields.io/badge/📓%20Colab-Training%20Notebook-orange)](https://colab.research.google.com/drive/18UJSpREGN152swrVjkEbGa0aWJR7eROH?usp=sharing)
[![HuggingFace Space](https://img.shields.io/badge/🤗%20HF%20Space-Live%20Demo-blue)](https://huggingface.co/spaces/Harsh-4210/Conflict_Bench)
[![Technical Blog](https://img.shields.io/badge/📝%20Blog-Deep%20Dive-green)](blog.md)

---

## 🎯 The Problem: The "Silent Failure" of Enterprise Alignment

In every modern corporation, directives are messy, non-linear, and frequently contradictory. A C-Suite executive might order a "strategic hiring freeze," while a Department Head demands "immediate headcount expansion" to hit a deadline. 

Current LLMs suffer from a **consensus bias**—they try to follow all instructions simultaneously, leading to logical hallucinations or "silent failures" where contradictory actions co-exist in an execution plan.

**ConflictBench** resolves this by training models to understand an **Implicit Authority Hierarchy**. It transforms LLMs from passive instruction-followers into active **Operational Coordinators** that can autonomously arbitrate business conflicts.

---

## 💡 Innovation: O1-Style Reasoning for Operations

ConflictBench is the first system to apply **Group Relative Policy Optimization (GRPO)** to the domain of authority arbitration. 

### Core Themes
*   **🛡️ Authority Hierarchy**: Learning to weight instructions by stakeholder seniority (CEO > VP > Director).
*   **⏳ Temporal Priority**: Navigating contradictions between "Immediate Action" and "Future Compliance."
*   **⚖️ Constraint Logic**: Identifying when a tactical request violates a strategic budgetary or legal constraint.
*   **🤝 Consensus Synthesis**: Generating a single, non-contradictory resolution plan from divergent inputs.
*   **⚙️ Operational Execution**: Translating high-level reasoning into structured, actionable JSON data.

---

## 🏗️ Architecture & Reward Design

ConflictBench uses a deterministic, rule-based verifier to provide a rich gradient signal for RL training. **No LLM Judge is used**, ensuring 100% objective evaluation.

### The Verifier Rubrics (Total Reward: 1.0)
| Rubric | Weight | Purpose |
|---|---|---|
| **JSON Format** | **0.2** | Ensures structural integrity for downstream automation. |
| **Conflict Detection** | **0.2** | F1 score for identifying every implicit contradiction. |
| **Authority Matching** | **0.2** | Verifies that the correct stakeholder (senior) was prioritized. |
| **Actionable IDs** | **0.2** | Ensures all resolved IDs exist in the original scenario. |
| **Plan Coherence** | **0.2** | Penalizes plans that still contain unresolved contradictions. |

> [!TIP]
> For a deep dive into the system flow and hierarchy logic, see [docs/architecture.md](docs/architecture.md).

---

## 📊 Results: Run 2 "The Hardening"

Our second training run demonstrated significant emergent reasoning capabilities on the Qwen-2.5-3B-Instruct base.

| Metric | Baseline | Run 2 (Peak) | Improvement |
|---|---|---|---|
| **Resolution Reward** | 0.372 | **0.491** | **+32%** |
| **KL Divergence** | 0.000 | 0.002 | Stable & Healthy |
| **Reasoning Depth** | 120 tokens | 345 tokens | Complex Thought |
| **Conflict Recall** | 42% | 88% | High Precision |

![Final Metrics Dashboard](plots/ultimate_metrics_dashboard.png)

### The "Inverse Scaling" Discovery
We discovered a critical insight during training: **Precision beats Verbosity.** Our data shows that the model actually achieves higher rewards as it learns to keep its reasoning punchy and decisive rather than rambling—a key finding for production-ready enterprise agents.

![Reasoning Efficiency](plots/slide_2_efficiency.png)

---

## 🛠️ Quickstart Guide

### 1. Training on Google Colab (Recommended)
Use our optimized [Colab Notebook](https://colab.research.google.com/drive/18UJSpREGN152swrVjkEbGa0aWJR7eROH?usp=sharing) for a zero-setup experience.

> [!IMPORTANT]
> **Lightning Run Disclaimer**: If you wish to verify the training in under 45 minutes, use the following configuration in the notebook:
> * `TRAIN_SCENARIOS = 150`
> * `NUM_EPOCHS = 1`
> * `NUM_GENERATIONS = 4`
> * *Note: Final rewards may be lower than reported due to reduced training time.*

### 2. Deployment on Hugging Face Spaces
The `hf_space_l40s/` directory contains everything needed for a high-performance Docker deployment on NVIDIA L40S/A100 instances.
```bash
# Push the folder to your HF Space
git push hf main
```

### 3. Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run a scenario test
python conflict_bench.py

# Run local inference
python inference.py
```

---

## 📁 Project Structure

*   `conflict_bench.py`: Core OpenEnv environment and scenario logic.
*   `train_grpo.py`: Root training script for deep, high-intensity local training.
*   `hf_space_l40s/`:
    *   `train_script.py`: Specialized script for **HF Space & Colab** environments.
    *   `app.py`: Gradio interface for live model comparison.
*   `docs/`: Detailed technical specifications and [architecture](docs/architecture.md).
*   `blog.md`: Technical deep-dive and project findings.
*   `inference.py`: Utility script for testing trained adapters.

---

## 🌟 Why It Matters

ConflictBench is more than a benchmark; it is a blueprint for **Reliable Multi-Agent Governance**. As we move toward a world of "Agentic Workflows," the ability for an AI to understand who holds the "Golden Key" in a conflict is the difference between a successful automation and a catastrophic operational failure.

**ConflictBench provides the training signal to ensure AI always aligns with the correct authority.**
