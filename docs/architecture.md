# ConflictBench — System Architecture

## Overview

ConflictBench is composed of four independent, composable components:

1. **ScenarioGenerator** — produces synthetic training episodes
2. **ConflictBenchEnv** — OpenEnv-compatible environment wrapper
3. **Verifier** — deterministic reward computation
4. **GRPO Training Stack** — Unsloth + TRL training loop

Each component has a single, well-defined interface with the next. The boundary between generation and evaluation is strict: the verifier never sees the generation logic, and the generator never sees the reward function.

---

## Component 1: ScenarioGenerator (`generator.py`)

### Purpose
Produces an unlimited supply of novel, realistic business instruction documents with ground truth embedded at generation time.

### Design Decisions

**Programmatic generation over LLM generation.** Each scenario is built by filling templates with randomly sampled parameters. This eliminates the risk of a generative model introducing inconsistencies or biases into the training data. Every scenario is verifiably correct by construction.

**Ground truth injection at generation time.** When two instructions conflict, the winner (higher authority source) is determined at the moment of conflict creation and stored with the scenario object. The verifier checks completions against this stored ground truth — not against a post-hoc judgement.

**Prompt length guard.** Scenarios exceeding 4000 characters are discarded and regenerated. This prevents VRAM-exhausting prompts and ensures consistent context length distributions across the training batch.

### Parameters

| Parameter | Value | Notes |
|---|---|---|
| Action key groups | 10 | hiring, budget, data access, policy, vendor, etc. |
| Template variants per group | 3 | low/medium/high complexity phrasings |
| Authority sources | 16 | spanning 6 hierarchy levels |
| Instructions per episode | 8–28 | sampled uniformly |
| Conflict pairs per episode | 2–6 | embedded proportionally |
| Difficulty 1 | 2 conflicts, obvious source labels | 80% of training mix |
| Difficulty 2 | 4 conflicts, ambiguous source labels | 20% of training mix |
| Difficulty 3 | 6 conflicts, multi-hop reasoning required | eval only |

### Authority Hierarchy

```
Level 1  Legal & Compliance           (always wins)
Level 2  C-Suite / Board              (executive directives)
Level 3  VP / Senior Director         (functional leadership)
Level 4  Director / Manager           (departmental management)
Level 5  Team Lead                    (team-level direction)
Level 6  Individual Contributor       (lowest authority)
```

When instructions from Level 3 and Level 5 conflict, Level 3 wins. This is deterministic — no judgement required.

---

## Component 2: ConflictBenchEnv (`conflict_bench.py`)

### Purpose
Wraps the generator and verifier into an OpenEnv-compatible environment class.

### Interface

```python
env = ConflictBenchEnv()

obs = env.reset()          # → str: formatted instruction document
reward, done, info = env.step(action)  # action: str (model completion)
state = env.state          # → dict: current episode metadata
```

### Episode Structure

Each episode is a single turn. The model receives the full instruction document as the observation and must produce a complete JSON resolution plan as the action. There is no multi-turn interaction within an episode.

This design choice is intentional: business instruction resolution is typically a single-pass task. An employee reads the full set of instructions and produces a resolution plan. They do not interactively negotiate with the environment.

---

## Component 3: Verifier (`verifier.py`)

### Purpose
Computes the composite reward from a model completion and the corresponding scenario ground truth.

### Rubric Implementation

```python
def score(completion: str, scenario: Scenario) -> ScoreBreakdown:
    r1 = score_final_state(completion, scenario)      # F1 vs ground truth
    r2 = score_no_contradictions(completion)          # binary contradiction check
    r3 = score_conflict_identification(completion, scenario)  # pair F1 × direction
    r4 = score_efficiency(completion, scenario)       # unnecessary instruction penalty
    r5 = score_format_compliance(completion)          # JSON schema validation
    composite = 0.35*r1 + 0.25*r2 + 0.20*r3 + 0.10*r4 + 0.10*r5
    return ScoreBreakdown(r1, r2, r3, r4, r5, composite)
```

### Why Each Rubric Weight Was Chosen

- **35% Final State**: The primary task is producing a correct execution plan. This is the dominant signal.
- **25% No Contradictions**: A plan that contradicts itself is worse than useless — it is actively harmful. High weight ensures the model never learns to produce contradictory plans.
- **20% Conflict Identification**: Explicit conflict naming builds structured reasoning, not just lucky resolution. Without this, a model could get high Final State scores by memorising common resolution patterns rather than actually identifying conflicts.
- **10% Efficiency**: Prevents padding/inclusion of irrelevant instructions. Lower weight because inefficiency is less harmful than incorrect resolution.
- **10% Format**: Minimum bar. Valid JSON is required for downstream use but not the primary measure of capability.

---

## Component 4: GRPO Training Stack

### Why GRPO

GRPO (Group Relative Policy Optimisation) works by:
1. Generating N completions per prompt
2. Scoring each with the reward function
3. Computing advantages relative to the group mean
4. Updating the policy to increase probability of above-average completions

This is ideal for ConflictBench because:
- No reference completions are needed — the model learns from its own distribution
- The reward function provides a dense, informative signal at every step
- GRPO is stable under the composite reward structure (no single rubric dominates during training instability)

### Model Choice

Qwen2.5-3B-Instruct was selected for:
- Strong instruction following baseline (reduces cold-start problem)
- Sufficient capacity for structured JSON output
- 3B parameters fits comfortably in 4-bit on T4/L4/A10G for broad accessibility

### LoRA Configuration

```
Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
Rank (r): 32
Alpha: 32  (alpha = r is standard; effective scale = 1.0)
Dropout: 0.0
Bias: none
Gradient checkpointing: unsloth (memory-efficient)
```

Rank 32 was chosen to balance expressiveness (the model needs to learn a genuinely new capability) with memory efficiency (higher ranks risk OOM on L4/A10G).

### Key Hyperparameters and Their Rationale

| Hyperparameter | Value | Rationale |
|---|---|---|
| Learning rate | 3e-6 | Conservative; avoids catastrophic forgetting of instruction following |
| Warmup ratio | 0.05 | 5% warmup stabilises early GRPO steps |
| β (KL penalty) | 0.04 | Prevents excessive drift from base policy; 0.02 was too low (KL diverged) |
| Temperature | 1.0 | Full diversity in GRPO generation; lower values reduce exploration |
| num_generations | 4–6 | 4 is minimum for meaningful group comparison; 6 better on large VRAM |
| Max completion tokens | 768 | Average completion ~300 tokens; headroom without waste |
| Max prompt tokens | 3200 | Accommodates 28-instruction documents; prompt guard at generation enforces this |

---

## Data Flow Diagram

```
Episode Start
     │
     ▼
ScenarioGenerator.generate(difficulty)
     │  Returns: Scenario(instructions, conflicts, ground_truth, prompt)
     ▼
ConflictBenchEnv.reset()
     │  Returns: formatted_prompt (str)
     ▼
GRPOTrainer generates N completions
     │  N = num_generations (4–6)
     ▼
[For each completion]
Verifier.score(completion, scenario)
     │  Returns: ScoreBreakdown(r1..r5, composite)
     ▼
GRPOTrainer computes group advantages
     │  advantage_i = reward_i - mean(rewards)
     ▼
Policy gradient update (PPO-style clipped objective + KL penalty)
     │
     ▼
Next Episode
```

---

## File Reference

| File | Lines | Purpose |
|---|---|---|
| `conflict_bench.py` | ~150 | OpenEnv environment class |
| `generator.py` | ~500 | Scenario generation with all templates |
| `verifier.py` | ~300 | Five rubric functions + composite scorer |
| `train_grpo.py` | ~350 | Root training script (local/Kaggle) |
| `hf_space_a100/train.py` | ~400 | HF Spaces/Colab training with Gradio integration |
| `hf_space_a100/app.py` | ~130 | Gradio dashboard |
| `diagnose_tokens.py` | ~80 | Token budget analysis utility |
| `app.py` | ~100 | Local demo (base vs fine-tuned comparison) |
