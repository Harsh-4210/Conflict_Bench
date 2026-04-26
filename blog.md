# Teaching AI to Navigate the Chain of Command

### How ConflictBench uses reinforcement learning to train language models on the skill every enterprise employee learns on day one — and no AI has ever been explicitly taught.

---

There is a skill every new employee learns within their first week at any organisation, and it is never written in any onboarding document.

When your manager asks you to do something that contradicts what Legal said last week, you do not flip a coin. You do not try to do both. You do not ask your manager which one they meant. You know — through a mix of common sense, organisational instinct, and quiet survival — that Legal wins. You complete the task in a way that satisfies the compliance constraint, then you let your manager know and explain why.

This is authority-aware instruction resolution. Humans develop it through experience. It is one of the most fundamental soft skills in professional life.

No language model has ever been explicitly trained to do it.

---

## The Problem That Lives in Every Organisation

Walk into any enterprise using AI assistants today and ask them about instruction conflicts. They will describe the problem carefully: the AI sometimes follows the wrong instruction. Sometimes it tries to do both and produces a contradictory plan. Sometimes it freezes and refuses to act. In the worst cases — the ones nobody talks about publicly — it executes the lower-authority instruction with full confidence, because that instruction happened to be more recent, more emphatic, or simply longer.

The reason this happens is structural. Current language models are trained on human-written text that describes how decisions are made, but they are not trained with a reward signal that penalises authority misjudgements. They have read millions of documents about organisational hierarchies, but they have never experienced the consequence of getting the hierarchy wrong.

This is the gap ConflictBench is designed to close.

---

## What We Built

ConflictBench is a reinforcement learning environment — specifically, an OpenEnv-compatible training environment — that presents a language model with realistic business instruction documents and rewards it for correctly resolving the authority conflicts embedded within them.

Each training episode works like this:

The model receives a document containing between 8 and 28 instructions from multiple organisational sources. The sources span six tiers of authority: Legal and Compliance at the top, then C-Suite, then VP and Senior Director, then Director and Manager, then Team Lead, then Individual Contributor. Several pairs of instructions within the document directly contradict each other. The model is not told which pairs conflict. It is not told the authority hierarchy. It must infer both from the structure of the document and the labels on each instruction.

The model must produce a structured JSON execution plan that: identifies every conflicting pair, states which instruction wins and why, and lists what should be executed and what should be overridden.

The reward function evaluates this plan against programmatically-generated ground truth across five deterministic rubrics. No language model judge. No human scorer. The verifier checks the output against the exact correct answer that was injected at generation time.

---

## Why GRPO, and Why It Works Here

Group Relative Policy Optimisation — the same training method behind DeepSeek-R1's reasoning improvements — is particularly well-suited to this problem for a reason that is easy to overlook.

GRPO works by generating multiple completions for the same prompt, scoring each, and updating the policy based on which completions scored above or below the group mean. This means the model is always being compared to itself — not to a fixed target, not to a human preference label, but to the distribution of its own outputs on a given problem.

This is exactly right for authority resolution. The model does not need to know what the correct answer looks like. It needs to learn to generate better answers than its own average. The reward function provides the signal that defines "better," and because the reward is fully deterministic and rule-based, the gradient is clean — there is no ambiguity about whether a completion deserved its score.

We used Unsloth's optimised GRPO implementation with TRL's GRPOTrainer, applied to Qwen2.5-3B-Instruct with 4-bit quantisation and LoRA (rank 32, targeting all attention and MLP projections). The full training run on an L40S 48GB GPU takes approximately 8 hours for 400 scenarios across 2 epochs.

---

## The Reward Function: Designed to Be Ungameable

The five rubrics are worth describing in detail, because the design choices matter.

**Correct Final State (35%)** measures F1 between the model's execution plan and the ground truth followed/overridden sets. This is the primary signal. An output that correctly identifies which instructions should execute and which should be blocked scores well here, regardless of how it explains its reasoning.

**No Contradictions (25%)** checks whether the output plan contains two instructions that cannot coexist. An agent that tries to satisfy both sides of a conflict — "follow the hiring freeze AND hire the four engineers" — receives zero on this rubric regardless of how detailed its reasoning appears.

**Conflict Identification (20%)** measures F1 of identified conflict pairs multiplied by resolution direction accuracy. This rewards the model for explicitly naming the conflicts, not just producing an output that happens to be correct. It builds the skill of structured conflict reasoning, not just lucky resolution.

**Efficiency (10%)** penalises unnecessary instructions in the execution plan. A model that includes every instruction regardless of relevance — a common padding strategy — scores poorly here.

**Format Compliance (10%)** verifies valid JSON with the required schema. This is a minimum bar, but it matters: an output that cannot be parsed contributes nothing to a downstream system.

Three common reward-gaming strategies all fail against this rubric set. Always follow all instructions: the Contradictions rubric catches this. Always override everything: the Final State rubric catches this (many instructions should be followed). Produce well-formatted JSON with plausible-sounding reasoning: the Conflict Identification F1 catches this — the IDs are verified against the actual conflict pairs.

---

## What the Training Curves Show

The results from Run 2 — 400 training scenarios, 2 epochs, L40S 48GB — are instructive.

The baseline Qwen2.5-3B-Instruct achieves a composite reward of 0.14 on this task with no training. This is low, but not zero: the model can produce valid JSON and occasionally resolves conflicts correctly by chance.

After GRPO training, reward climbs to a peak of 0.50 around step 250 — a 257% improvement over baseline. The model has learned, from the reward signal alone and without any labelled examples, to apply a consistent authority hierarchy across documents it has never seen.

Several things about the training dynamics are worth noting.

The reasoning length stays stable throughout training at approximately 300 tokens per completion. This tells us the model is not reward-hacking via length — it is not generating empty outputs to avoid contradictions, and it is not padding with irrelevant text to appear thorough. The reward improvement reflects genuine task learning.

The reward curve is volatile, as is typical for GRPO on complex structured reasoning tasks. Each step sees only a small batch of generated completions, and the reward landscape for this task has many local features. The upward trend is real but noisy.

The KL divergence from the base policy rises steadily throughout training, which is expected given our low β (0.02 in early runs, corrected to 0.04 in later runs). The model is moving meaningfully away from the base distribution — which is the intended behaviour when learning a genuinely new capability.

---

## The Connection to AI Scaling

This project was built under the Scale AI sub-theme of the OpenEnv Hackathon, and the connection is not superficial.

The core hypothesis of AI scaling is that larger models, trained on more data with better objectives, develop more generalised capabilities. This has been demonstrated across many dimensions: reasoning, coding, multilingual understanding, factual knowledge. But one capability has been conspicuously absent from the scaling story: the ability to operate correctly under authority constraints in multi-stakeholder environments.

This is not because the problem is hard. It is because there has been no training signal for it.

ConflictBench provides that signal. And because the reward function is fully deterministic and the environment is infinitely generatable — each episode is a fresh scenario with new parameters, new instruction content, and new conflict pairs — the environment can scale to any amount of training data without human labelling cost.

This is the pattern that produced the breakthroughs in mathematical reasoning: not more supervised examples, but better reward functions applied to generated problems. ConflictBench applies the same pattern to a different class of capability — one that is arguably more important for real-world deployment than the ability to solve competition mathematics.

---

## Where This Goes Next

The current implementation trains on single-domain business scenarios with a fixed six-tier hierarchy. Several natural extensions suggest themselves.

**Cross-domain generalisation.** The authority hierarchy in healthcare (attending physician > specialist > resident > nurse) is different from the hierarchy in a law firm (partner > senior associate > junior associate > paralegal) but the underlying structure is the same. A model trained on one domain should be able to transfer to others if the reward signal is consistent.

**Dynamic hierarchy learning.** In real organisations, authority is not always fixed. A VP's directive outranks a Manager's except in that manager's domain of specific expertise. Training on scenarios with contextual authority — where the hierarchy shifts based on subject matter — would produce more robust models.

**Multi-agent conflict resolution.** The current environment has a single model resolving conflicts. An extension with multiple models negotiating authority claims in a shared workspace would address a different but related problem: what happens when two AI agents receive conflicting instructions from their respective principals?

**Adversarial instruction injection.** Training on scenarios where some instructions are specifically designed to appear higher-authority than they are — instruction injection attacks — would directly address an emerging security concern in enterprise AI deployment.

---

## The Honest Assessment

A composite reward of 0.50 after 2 epochs on 400 scenarios is a promising result, not a solved problem. The model resolves conflicts correctly more often than chance, and significantly better than the untrrained baseline, but it does not yet perform at the level required for autonomous deployment in a real enterprise context.

What this work demonstrates is that the capability is learnable from a reward signal. That is the important result. The path from 0.50 to 0.80 is a matter of scale — more scenarios, more epochs, larger models, harder difficulty levels. The reward function is already in place. The environment is already generating valid training data. The infrastructure works.

The first step in teaching any capability is proving it can be taught. ConflictBench does that.

---

*ConflictBench was built for the OpenEnv Hackathon India 2026. The codebase, trained model, and training infrastructure are open source.*

*Codebase: [github.com/Harsh-4210/Conflict_Bench](https://github.com/Harsh-4210/Conflict_Bench)*
*HF Space: [huggingface.co/spaces/Harsh-9209/Conflict_Bench](https://huggingface.co/spaces/Harsh-9209/Conflict_Bench)*
*Trained model: [huggingface.co/Harsh-9209/conflictbench-qwen2.5-3b-grpo-lora](https://huggingface.co/Harsh-9209/conflictbench-qwen2.5-3b-grpo-lora)*