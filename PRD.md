# ConflictBench — Product Requirements Document

**Version:** 1.0  
**Author:** Harsh Jain  
**Status:** Submitted — OpenEnv Hackathon India 2026  
**Last Updated:** April 2026

---

## 1. Executive Summary

ConflictBench is a reinforcement learning training environment that teaches language models to resolve contradictory instructions using an implicit authority hierarchy. It is the first dedicated RL environment for this capability class. The product consists of a scenario generator, a deterministic verifier, a GRPO training stack, and a HuggingFace Spaces deployment with a live training dashboard.

The primary deliverable is a fine-tuned Qwen2.5-3B-Instruct LoRA adapter that demonstrably outperforms the base model on authority-aware conflict resolution tasks, trained via a reward signal that cannot be gamed without actually solving the task.

---

## 2. Problem Statement

### 2.1 The Core Problem

Language models deployed in enterprise environments routinely receive instructions from multiple stakeholders with different authority levels. When these instructions conflict — as they frequently do in real organisations — current models exhibit one of three failure modes:

1. **Recency bias**: follows the most recently stated instruction regardless of authority
2. **Salience bias**: follows the most emphatic or detailed instruction regardless of authority
3. **Contradiction**: produces a plan that attempts to satisfy both sides of the conflict simultaneously, resulting in a logically inconsistent output

None of these failure modes are acceptable for autonomous deployment. All three stem from the same root cause: no existing training signal specifically rewards authority-aware conflict resolution.

### 2.2 Why Existing Solutions Do Not Work

**Prompt engineering** can partially address this by explicitly stating the authority hierarchy in the system prompt. This approach does not scale: it requires the hierarchy to be known in advance, stated explicitly, and maintained by the operator. It also fails when the conflict involves instruction sources not anticipated at prompt-writing time.

**Supervised fine-tuning** on labelled conflict resolution examples requires human annotation of ground truth resolutions — expensive, slow to scale, and subject to annotator disagreement on complex cases.

**RLHF with a preference model** introduces a learned judge that may itself be miscalibrated and is vulnerable to reward model exploitation.

**ConflictBench** addresses all three limitations: no explicit hierarchy in the prompt, no human annotation required, and a fully deterministic reward function that cannot be exploited without solving the underlying task.

---

## 3. Goals and Non-Goals

### Goals

- Provide a complete, runnable RL training environment for authority-aware instruction conflict resolution
- Demonstrate measurable improvement over the base model via a non-gameable reward signal
- Make the environment accessible across training platforms (HF Spaces, Colab, Kaggle, local)
- Produce a trained model artifact that can be evaluated independently
- Document the approach thoroughly enough for reproduction and extension

### Non-Goals

- Production-ready enterprise deployment (this is a research prototype)
- Support for non-English instruction documents (English only in current version)
- Real-time instruction conflict detection (inference-time application is a future extension)
- Training models larger than 7B parameters (compute scope is limited to hackathon budget)

---

## 4. User Stories

### Primary User: ML Researcher

> "I want to train a language model on instruction conflict resolution without needing human-labelled examples, so that I can study how RL reward design affects authority-hierarchy learning."

Acceptance criteria:
- Environment can be instantiated and run locally with `python conflict_bench.py`
- Training script produces a checkpoint with measurably higher reward than baseline
- Reward function components are individually inspectable via `ScoreBreakdown`

### Secondary User: Hackathon Judge

> "I want to evaluate this submission quickly to understand what was built, verify that training works, and assess the quality of the approach."

Acceptance criteria:
- A 30–60 minute evaluation run can be completed on a free Colab T4
- The README explains the approach clearly without requiring prior familiarity
- Results tables show honest numbers with appropriate caveats about run duration

### Tertiary User: Enterprise AI Team

> "I want to understand whether this approach could be applied to reduce instruction conflict errors in our AI assistant deployment."

Acceptance criteria:
- The blog and documentation explain the business motivation and technical approach
- The trained model is publicly accessible on HF Hub for evaluation
- The authority hierarchy is configurable for different organisational structures

---

## 5. Functional Requirements

### 5.1 Scenario Generator

| ID | Requirement | Priority |
|---|---|---|
| SG-01 | Generate scenarios with 8–28 instructions per episode | Must Have |
| SG-02 | Support 3 difficulty levels (2, 4, 6 conflict pairs) | Must Have |
| SG-03 | Embed ground truth resolution at generation time | Must Have |
| SG-04 | Enforce prompt length limit (4000 chars max) | Must Have |
| SG-05 | Produce reproducible scenarios given a fixed seed | Should Have |
| SG-06 | Support 10 action key groups with 3 template variants each | Must Have |
| SG-07 | Sample from 16 authority sources across 6 hierarchy levels | Must Have |

### 5.2 Verifier

| ID | Requirement | Priority |
|---|---|---|
| VR-01 | Implement all 5 rubric functions deterministically | Must Have |
| VR-02 | Return ScoreBreakdown with per-rubric and composite scores | Must Have |
| VR-03 | Handle malformed JSON completions gracefully (score = 0.0) | Must Have |
| VR-04 | Compute F1 correctly for partial matches | Must Have |
| VR-05 | Return composite score in range [0.0, 1.0] | Must Have |

### 5.3 Training Stack

| ID | Requirement | Priority |
|---|---|---|
| TR-01 | Support GRPO training via TRL GRPOTrainer | Must Have |
| TR-02 | Auto-detect GPU and set optimal configuration | Must Have |
| TR-03 | Always use 4-bit quantisation (prevents Unsloth dtype kernel bug) | Must Have |
| TR-04 | Save checkpoints every 50 steps | Must Have |
| TR-05 | Auto-upload best checkpoint to HF Hub after training | Should Have |
| TR-06 | Stream training logs to Gradio dashboard in real time | Should Have |
| TR-07 | Support checkpoint resume after session interruption | Should Have |

### 5.4 HF Spaces Deployment

| ID | Requirement | Priority |
|---|---|---|
| SP-01 | One-click training start via Gradio button | Must Have |
| SP-02 | Live log streaming with auto-refresh | Must Have |
| SP-03 | Training plots displayed after completion | Should Have |
| SP-04 | Thread-safe log buffer | Must Have |

---

## 6. Non-Functional Requirements

| Category | Requirement |
|---|---|
| Reproducibility | Same seed → same scenarios → same training trajectory |
| Accessibility | Runnable on free Colab T4 (with reduced config) |
| Correctness | Verifier outputs identical scores for identical inputs |
| Transparency | All reward components inspectable individually |
| Maintainability | Generator, verifier, and trainer are independently testable |

---

## 7. Architecture Summary

See `doc/ARCHITECTURE.md` for full detail.

```
ScenarioGenerator → ConflictBenchEnv → Verifier → GRPO Trainer
     (generator.py)   (conflict_bench.py)  (verifier.py)  (train_grpo.py)
```

---

## 8. Training Configuration (Production)

| Parameter | Value | Rationale |
|---|---|---|
| Base model | Qwen2.5-3B-Instruct (4-bit) | Balance of capability and accessibility |
| LoRA rank | 32 | Sufficient expressiveness; stable on L4/A10G |
| Training scenarios | 400 | Enough for GRPO convergence at difficulty 1–2 |
| Eval scenarios | 60 | Reliable eval metric with low variance |
| Epochs | 2 | Reward peaks around epoch 2; more risks KL drift |
| Learning rate | 3e-6 | Conservative; preserves instruction following |
| β (KL penalty) | 0.04 | Prevents excessive drift; 0.02 was insufficient |
| num_generations | 4–6 | 4 minimum; 6 preferred on L40S |

---

## 9. Success Metrics

| Metric | Baseline | Target | Achieved (Run 2) |
|---|---|---|---|
| Composite reward (peak) | 0.14 | 0.40 | **0.50** |
| Improvement over baseline | — | +100% | **+257%** |
| No contradictions rate | ~31% | >60% | **~74%** |
| Conflict identification F1 | ~8% | >25% | **~39%** |
| Format compliance | ~65% | >85% | **~88%** |

All targets exceeded.

---

## 10. Known Limitations and Future Work

### Current Limitations

- Training is limited to difficulty 1–2 scenarios; difficulty 3 is eval-only
- The authority hierarchy is fixed (6 tiers); domain-specific hierarchies require code changes
- Reward peak at ~step 250 suggests reward saturation under current configuration; longer runs require β adjustment and curriculum progression
- KL divergence rises monotonically; future runs should monitor this and halt early if drift exceeds threshold

### Planned Extensions

1. **Difficulty curriculum progression**: start with difficulty 1, transition to difficulty 2–3 as reward improves
2. **Domain-configurable hierarchies**: parameterise the authority levels for healthcare, legal, government contexts
3. **Multi-hop conflict reasoning**: conflicts where resolution of pair A depends on resolution of pair B
4. **Adversarial instruction injection**: scenarios designed to trick the model into following low-authority instructions
5. **7B and 13B model training**: with appropriate quantisation and VRAM scaling
6. **Multilingual extension**: Spanish, French, German authority resolution scenarios

---

## 11. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Reward hacking via length | Low | Medium | Efficiency rubric penalises over-inclusion |
| JSON format memorisation without semantic understanding | Medium | High | Conflict ID F1 requires correct instruction IDs |
| KL divergence runaway | Medium | Medium | β=0.04 provides sufficient penalty |
| Unsloth kernel dtype bug on full-precision path | High (L40S) | Critical | Always use 4-bit quantisation |
| Kaggle session timeout mid-training | High | Medium | Checkpoint-every-50-steps + resume support |

---

*ConflictBench PRD v1.0 — OpenEnv Hackathon India 2026*
