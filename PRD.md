# ConflictBench — Product Requirements Document

**Version:** 1.0 (with November 2026 corrections)  
**Author:** Harsh Jain  
**Status:** Submitted — OpenEnv Hackathon India 2026  
**Last Updated:** April 2026; corrections added November 2026

---

> **Update — November 2026.** This PRD was written for the hackathon submission in April 2026. An audit of the code in October 2026 found that the original reward could be gamed (an empty answer scored 0.340 and a keyword rule that ignores authority scored 0.951, `results/baselines.json`), that the original task could be solved by a lookup script (1.000, same file), and that the training results in §9 cannot be reproduced from this repository. The reward and the task have been redesigned and are tested on CPU; GPU training and model evaluation on the new task have not been run yet. Claims below that turned out false are struck through or followed by a *Correction* note; the requirements themselves are kept as the original plan. Current facts are in the [README](./README.md) (see "What we found and fixed").

---

## 1. Executive Summary

ConflictBench is a reinforcement learning training environment that teaches language models to resolve contradictory instructions using an authority hierarchy. ~~It is the first dedicated RL environment for this capability class.~~ *Correction: instruction-priority research and benchmarks already exist (for example OpenAI's 2024 Instruction Hierarchy work and IHEval); ConflictBench's focus is a multi-stakeholder business setting with a deterministic scorer.* The product consists of a scenario generator, a deterministic verifier, a GRPO training stack, and a HuggingFace Spaces deployment with a live training dashboard.

The primary deliverable is a fine-tuned Qwen2.5-3B-Instruct LoRA adapter that demonstrably outperforms the base model on authority-aware conflict resolution tasks, ~~trained via a reward signal that cannot be gamed without actually solving the task~~. *Correction: the original reward could be gamed (see the update above). No adapter has yet been shown to outperform the base model: the hackathon adapter has not been re-scored, and nothing has been trained on the redesigned task.*

---

## 2. Problem Statement

### 2.1 The Core Problem

Language models deployed in enterprise environments routinely receive instructions from multiple stakeholders with different authority levels. When these instructions conflict — as they frequently do in real organisations — current models exhibit one of three failure modes:

1. **Recency bias**: follows the most recently stated instruction regardless of authority
2. **Salience bias**: follows the most emphatic or detailed instruction regardless of authority
3. **Contradiction**: produces a plan that attempts to satisfy both sides of the conflict simultaneously, resulting in a logically inconsistent output

None of these failure modes are acceptable for autonomous deployment. ~~All three stem from the same root cause: no existing training signal specifically rewards authority-aware conflict resolution.~~ *Correction: this was stated as a hypothesis, not measured; ConflictBench has not yet measured any model's failure modes.*

### 2.2 Why Existing Solutions Do Not Work

**Prompt engineering** can partially address this by explicitly stating the authority hierarchy in the system prompt. This approach does not scale: it requires the hierarchy to be known in advance, stated explicitly, and maintained by the operator. It also fails when the conflict involves instruction sources not anticipated at prompt-writing time.

**Supervised fine-tuning** on labelled conflict resolution examples requires human annotation of ground truth resolutions — expensive, slow to scale, and subject to annotator disagreement on complex cases.

**RLHF with a preference model** introduces a learned judge that may itself be miscalibrated and is vulnerable to reward model exploitation.

**ConflictBench** addresses these limitations: ~~no explicit hierarchy in the prompt~~, no human annotation required, and a fully deterministic reward function ~~that cannot be exploited without solving the underlying task~~. *Correction: the original prompt printed the full hierarchy. Since November 2026 the default prompt shows only names and job titles plus the general seniority rule in words; a printed-hierarchy mode still exists. A deterministic reward can still be exploited; the current one is checked against a set of cheat strategies by tests, which is not a proof that no other cheat exists.*

---

## 3. Goals and Non-Goals

### Goals

- Provide a complete, runnable RL training environment for authority-aware instruction conflict resolution
- Demonstrate measurable improvement over the base model via a reward signal that is tested against known cheat strategies *(not yet demonstrated; the April 2026 wording called the reward impossible to game, which was false)*
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
- Training script produces a checkpoint with measurably higher reward than baseline *(not yet met: no run with the current pipeline)*
- Reward function components are individually inspectable via `ScoreBreakdown`

### Secondary User: Hackathon Judge

> "I want to evaluate this submission quickly to understand what was built, verify that training works, and assess the quality of the approach."

Acceptance criteria:
- A 30–60 minute evaluation run can be completed on a free Colab T4 *(now: `configs/grpo_smoke.yaml`, a 10-step run, not yet timed)*
- The README explains the approach clearly without requiring prior familiarity
- Results tables show honest numbers with appropriate caveats about run duration *(the April 2026 tables did not; see §9)*

### Tertiary User: Enterprise AI Team

> "I want to understand whether this approach could be applied to reduce instruction conflict errors in our AI assistant deployment."

Acceptance criteria:
- The blog and documentation explain the business motivation and technical approach
- The trained model is publicly accessible on HF Hub for evaluation
- The authority hierarchy is configurable for different organisational structures *(not met: the hierarchy is fixed in `generator.py`)*

---

## 5. Functional Requirements

### 5.1 Scenario Generator

| ID | Requirement | Priority |
|---|---|---|
| SG-01 | ~~Generate scenarios with 8–28 instructions per episode~~ *Correction: the original generator produced fewer; in the current test set level 1–3 scenarios have 9–37 instructions (`data/test_v1.jsonl`)* | Must Have |
| SG-02 | Support 3 difficulty levels (2, 4, 6 conflict pairs) *(now 5 levels: 2, 4, 6, 8, 10)* | Must Have |
| SG-03 | Embed ground truth resolution at generation time *(now derived by `resolver.resolve()` and cross-checked)* | Must Have |
| SG-04 | ~~Enforce prompt length limit (4000 chars max)~~ *Correction: the generator never had such a guard; training filters prompts by token count instead* | Must Have |
| SG-05 | Produce reproducible scenarios given a fixed seed *(met since the Phase 1 fixes)* | Should Have |
| SG-06 | ~~Support 10 action key groups with 3 template variants each~~ *Correction: the original generator had 10 groups and 60 template texts; the current one has 17 topics with 5 phrasings per side* | Must Have |
| SG-07 | ~~Sample from 16 authority sources across 6 hierarchy levels~~ *Correction: 18 offices on 6 tiers* | Must Have |

### 5.2 Verifier

| ID | Requirement | Priority |
|---|---|---|
| VR-01 | ~~Implement all 5 rubric functions deterministically~~ *Correction: the current composite uses 3 weighted rubrics and a format gate; efficiency and format are logged only* | Must Have |
| VR-02 | Return ScoreBreakdown with per-rubric and composite scores | Must Have |
| VR-03 | Handle malformed JSON completions gracefully (score = 0.0) | Must Have |
| VR-04 | Compute F1 correctly for partial matches | Must Have |
| VR-05 | Return composite score in range [0.0, 1.0] | Must Have |

### 5.3 Training Stack

| ID | Requirement | Priority |
|---|---|---|
| TR-01 | Support GRPO training via TRL GRPOTrainer | Must Have |
| TR-02 | ~~Auto-detect GPU and set optimal configuration~~ *Correction: replaced by one YAML config per GPU class in `configs/`* | Must Have |
| TR-03 | Always use 4-bit quantisation (prevents Unsloth dtype kernel bug) *(the 3B configs do; the 0.5B smoke config does not)* | Must Have |
| TR-04 | Save checkpoints every 50 steps *(now set per config: 5, 25 or 50)* | Must Have |
| TR-05 | Auto-upload best checkpoint to HF Hub after training *(now optional, off by default: `hub_repo: null`)* | Should Have |
| TR-06 | Stream training logs to Gradio dashboard in real time | Should Have |
| TR-07 | Support checkpoint resume after session interruption *(implemented in the current pipeline; not yet exercised on a GPU)* | Should Have |

### 5.4 HF Spaces Deployment

| ID | Requirement | Priority |
|---|---|---|
| SP-01 | One-click training start via Gradio button *(now password-locked)* | Must Have |
| SP-02 | Live log streaming with auto-refresh | Must Have |
| SP-03 | Training plots displayed after completion | Should Have |
| SP-04 | Thread-safe log buffer | Must Have |

---

## 6. Non-Functional Requirements

| Category | Requirement |
|---|---|
| Reproducibility | Same seed → same scenarios → same training trajectory *(scenarios: yes; training: not yet tested)* |
| Accessibility | Runnable on free Colab T4 (with reduced config) |
| Correctness | Verifier outputs identical scores for identical inputs |
| Transparency | All reward components inspectable individually |
| Maintainability | Generator, verifier, and trainer are independently testable |

---

## 7. Architecture Summary

See [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md) for full detail. *(Corrected path: this used to point to `doc/ARCHITECTURE.md`.)*

```
ScenarioGenerator + resolver → prompts → ConflictBenchEnv / server → Verifier → GRPO Trainer / evaluate.py
  (generator.py, resolver.py)  (prompts.py)  (conflict_bench.py, server/)  (verifier.py)  (training/, scripts/)
```

---

## 8. Training Configuration (Production)

~~The April 2026 table (LoRA rank 32, 400 training / 60 eval scenarios, 2 epochs, learning rate 3e-6, β 0.04, 4–6 generations).~~ *Correction: those values belonged to a script that is not in the repository. The current values are in `configs/grpo_t4.yaml` and `configs/grpo_a10g.yaml` (Qwen2.5-3B in 4-bit, LoRA rank 16, learning rate 5e-6, β 0.04; 200/20 or 600/30 train/val scenarios), summarised in [`docs/TRAINING_GUIDE.md`](./docs/TRAINING_GUIDE.md).*

---

## 9. Success Metrics

~~The April 2026 table of baseline, target and achieved composite reward, relative improvement and per-rubric rates, ending "All targets exceeded."~~

*Correction: those "achieved" numbers came from a training script that is not in the repository, on the old, gameable task, and some are impossible under the scorer of that time (`FIX_BLUEPRINT.md`, C.2). They are withdrawn; an archived copy, marked unverified, is in [`docs/TRAINING_LOGS.md`](./docs/TRAINING_LOGS.md). No success metric has been measured for a model yet. The measured rule-based and cheat baselines on the held-out test set are in [`results/leaderboard.md`](./results/leaderboard.md); a trained model has to be judged against them.*

---

## 10. Known Limitations and Future Work

### Current Limitations

- ~~Training is limited to difficulty 1–2 scenarios; difficulty 3 is eval-only~~ *Correction: in the old scripts difficulty 3 was removed by a character-length filter and never trained or evaluated. Now `grpo_a10g.yaml` includes level 3 and the filter counts tokens.*
- The authority hierarchy is fixed (6 tiers); domain-specific hierarchies require code changes
- ~~Reward peak at ~step 250 suggests reward saturation under current configuration; longer runs require β adjustment and curriculum progression~~ *(unverified run data, withdrawn)*
- ~~KL divergence rises monotonically; future runs should monitor this and halt early if drift exceeds threshold~~ *(unverified run data, withdrawn)*
- Each episode is a single step, so the environment is not long-horizon *(added November 2026)*

### Planned Extensions

1. **Difficulty curriculum progression**: start with difficulty 1, transition to difficulty 2–3 as reward improves
2. **Domain-configurable hierarchies**: parameterise the authority levels for healthcare, legal, government contexts
3. **Multi-hop conflict reasoning**: conflicts where resolution of pair A depends on resolution of pair B *(a first form exists: conditional and contingent instructions)*
4. **Adversarial instruction injection**: scenarios designed to trick the model into following low-authority instructions
5. **7B and 13B model training**: with appropriate quantisation and VRAM scaling
6. **Multilingual extension**: Spanish, French, German authority resolution scenarios

---

## 11. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Reward hacking via length | Low | Medium | ~~Efficiency rubric penalises over-inclusion~~ *Over-inclusion is penalised by the plan and overridden F1 and the split penalty; efficiency is no longer weighted* |
| Reward hacking via degenerate answers *(added November 2026; this risk was real: an empty answer scored 0.340)* | High (was) | High | Cheat-strategy regression tests in `tests/test_reward_hacks.py` and `tests/test_difficulty.py` |
| JSON format memorisation without semantic understanding | Medium | High | Conflict ID F1 requires correct instruction IDs |
| KL divergence runaway | Medium | Medium | β=0.04 in all current configs |
| Unsloth kernel dtype bug on full-precision path | High (A100) | Critical | 4-bit for the 3B configs |
| Kaggle session timeout mid-training | High | Medium | Checkpoints every `save_steps` + automatic resume |

---

*ConflictBench PRD v1.0 — OpenEnv Hackathon India 2026 (corrections November 2026)*
