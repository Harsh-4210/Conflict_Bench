# Teaching AI to Navigate the Chain of Command

### How ConflictBench uses reinforcement learning to train language models to resolve conflicting instructions by authority, and what an audit of its first version found.

---

> **Update — November 2026.** This post was first written for the OpenEnv Hackathon in April 2026. An audit of the code in October 2026 found that the original reward could be gamed and the original task could be solved without any learning. On the original task and scorer, an empty answer scored 0.340, a keyword rule that ignores who sent each instruction scored 0.951, and a short lookup script scored 1.000 (`results/baselines.json`). The training results this post used to report came from a script that is not in the repository and cannot be reproduced, so they have been removed. The reward and the task have since been redesigned and tested; no model has been trained or evaluated on the new version yet. The sections below are corrected to match the code. The full story, with a before/after table of the cheat strategies, is in the [README](./README.md#what-we-found-and-fixed).

---

There is a skill every new employee learns within their first week at any organisation, and it is rarely written in any onboarding document.

When your manager asks you to do something that contradicts what Legal said last week, you do not flip a coin. You do not try to do both. You know, through a mix of common sense and organisational instinct, that Legal wins. You complete the task in a way that satisfies the compliance constraint, then you tell your manager and explain why.

This is authority-aware instruction resolution. Humans develop it through experience. Researchers have started to study it for language models too: OpenAI's 2024 *Instruction Hierarchy* work trains models to rank system, user and tool instructions, and benchmarks such as IHEval measure it. ConflictBench looks at a different setting: many human stakeholders in a business, with rules that go beyond a single ranking.

---

## The Problem That Lives in Every Organisation

AI assistants inside companies receive instructions from many people. Sometimes the assistant follows the wrong one. Sometimes it tries to do both and produces a contradictory plan. Sometimes it follows the lower-authority instruction because it was more recent, more emphatic, or simply longer.

Language models have read a great deal of text about how organisations make decisions, but they are rarely trained with a reward that penalises getting the authority wrong. That is the gap ConflictBench is meant to help with.

---

## What We Built

ConflictBench is an OpenEnv-compatible reinforcement learning environment. It presents a language model with a generated business instruction document and rewards it for resolving the conflicts inside it.

Each episode works like this. *(Corrected November 2026: the original post described 8 to 28 instructions, a different six-tier list, and a prompt that did not show the hierarchy. None of that matched the code.)*

- The model receives a document of instructions from named people. In the fixed test set, a level-1 scenario has 9 to 15 instructions, level 2 has 16 to 24 and level 3 has 25 to 37 (counted from `data/test_v1.jsonl`); levels 4 and 5 are larger still.
- Authority runs over 18 offices on six tiers: legal and regulatory roles at the top, then the C-suite, vice presidents, directors, managers, and finally team leads and coordinators.
- **How much of the hierarchy is shown depends on the mode.** In the default mode the prompt states the general rule in words (legal outranks everyone, then C-suite, VPs, directors, managers, team leads), and each instruction shows only the sender's name and job title, such as "Senior Finance Director". The model has to map each title to its level. In the optional printed mode, and in the original hackathon task, the numbered tier list is printed in the prompt.
- Seniority is not the only rule. Standing notices can give one person the final say on one topic, a senior person can delegate their authority on a topic, the later date wins between equals, some instructions only apply if a stated condition holds, and some depend on another instruction going ahead.
- Not every pair on the same topic is a conflict: instructions that agree, or that cover different departments or systems, do not clash.

The model must produce a JSON plan that lists every real conflict with its winner, and splits all instructions into an execution plan and an overridden list.

The scorer checks this plan against the correct answer, which a deterministic resolver derives from the scenario's structured data. No language model judge and no human scorer.

---

## Why GRPO

Group Relative Policy Optimisation, the method used for DeepSeek-R1's reasoning training, generates several completions for the same prompt, scores each one, and pushes the policy toward the completions that scored above the group's mean. The model is compared with its own outputs on the same problem, not with a labelled example.

That suits this task: there is no need for human-written answers, only a reward that reliably says which of two answers is better. Because the reward here is rule-based, it gives the same score to the same answer every time.

The training pipeline uses Unsloth with TRL's `GRPOTrainer` on Qwen2.5 models with LoRA. *(Corrected November 2026: the original post described a rank-32 LoRA and an 8-hour A100 run. The current configs use rank 8 for a 0.5B smoke test and rank 16 for Qwen2.5-3B on a T4 or an A10G/L4, and none of them has been run yet. See [`docs/TRAINING_GUIDE.md`](./docs/TRAINING_GUIDE.md).)*

---

## The Reward Function, and How It Was Gamed

The original reward had five rubrics: correct final state (35%), no contradictions (25%), conflict identification (20%), efficiency (10%) and format (10%). The original post argued that it could not be gamed. That was wrong:

- An answer with **empty lists** scored 0.340, because "no contradictions" gave full marks to an empty plan and the format rubric paid for any JSON with the right keys.
- Putting **every instruction** in the plan scored 0.429.
- A keyword rule that always picks the strict-sounding side ("freeze", "block"), **without looking at who sent it**, scored 0.951, because the strict side won almost every conflict in the original generator.
- A short script that matched templates and looked up the printed tier scored 1.000.

(All four numbers: `results/baselines.json`.)

The current reward, in [`verifier.py`](./verifier.py) and [`docs/REWARD_DESIGN.md`](./docs/REWARD_DESIGN.md), has three weighted rubrics and a format gate:

- **Correct final state (45%)**: the average of the F1 of the execution plan and the F1 of the overridden list against the correct sets, with penalties for made-up IDs and for putting an ID in both lists.
- **Conflict identification (30%)**: the F1 of the conflict pairs found, multiplied by the share of them resolved in the right direction.
- **No contradictions (25%)**: on each real conflict, credit only if exactly one side is in the plan and the model named that side as the winner. An empty plan scores zero.
- **Format gate**: no parseable answer means a score of 0; type errors multiply the score by 0.9.

Efficiency was dropped from the score because it measured the same thing as the plan F1 a second time. On the original task, the new reward cut the empty answer to 0.000 and "follow everything" to 0.177 (`results/baselines_v2.json`). The keyword rule and the lookup script still scored 0.921 and 1.000 there, because on that task they really were giving the right answers. That is why the task itself had to change.

On the redesigned task's held-out test set, the old keyword rule and the old lookup script both score 0.189, and the best shortcut measured, a seniority lookup that is handed the hidden topic and scope fields for free, scores 0.808 (`results/leaderboard.md`). A test fails if any of these shortcuts climbs above its limit.

---

## What the Training Runs Showed

*(Replaced November 2026.)* The original post reported reward curves, a baseline, a peak and a relative improvement for two hackathon training runs. Those numbers came from a training script that is not in this repository, on the old, gameable task, and some of the per-rubric figures are impossible under the scorer of the time. They are kept only as an archive in [`docs/TRAINING_LOGS.md`](./docs/TRAINING_LOGS.md), marked unverified.

Nothing has been trained on the new task yet, and the hackathon adapter has not been re-scored. The commands to measure the base model, the old adapter and any API model on the fixed test set are in [`results/README.md`](./results/README.md). Whatever those runs show will be reported next to the rule-based baselines.

---

## The Connection to AI Scaling

This project was built under the Scale AI sub-theme of the OpenEnv Hackathon.

As language models take on more autonomous work in organisations, they will meet more instructions that cannot all be obeyed. Handling those correctly is a reliability and safety question, not only a capability question.

ConflictBench offers one kind of training signal for it. Because the scorer is rule-based and every scenario is generated, the environment can produce as many fresh training scenarios as needed without human labelling. The audit is also a reminder of the limits of that approach: a generated task and a hand-written reward can both contain shortcuts, and the only way to find them is to try to cheat on purpose and measure the result.

---

## Where This Goes Next

Some extensions the original post proposed now exist in a first form: **topic-specific authority** (standing rules and delegations) and **multi-step dependencies** (conditional and contingent instructions). Others are still open:

- **Measuring models.** Score the base model, the hackathon adapter and a newly trained model on the held-out test set.
- **Multi-turn episodes.** Instructions that arrive over several turns, with tool calls and a simulated company state. That would make the environment long-horizon; today each episode is a single step.
- **Spoofed authority.** Instructions that claim to come from Legal but do not, so the model must check the real sender.
- **Configurable organisation charts.** Other hierarchies, such as a hospital or a law firm, loaded from a file.

---

## The Honest Assessment

*(Rewritten November 2026.)* The first version of ConflictBench did not show that the capability can be learned. Its reward paid for doing nothing, its task could be solved with a lookup, and its training numbers cannot be reproduced. What the project has now is a scorer whose known cheats score near zero, a task on which every rule-based shortcut tested falls well short of a perfect score, a fixed held-out test set, and one training pipeline with configs ready to run. The next step is to run it and report what happens, whether or not training helps.

---

*ConflictBench was built for the OpenEnv Hackathon India 2026. The code is open source under the MIT license.*

*Codebase: [github.com/Harsh-4210/Conflict_Bench](https://github.com/Harsh-4210/Conflict_Bench)*
*HF Space: [huggingface.co/spaces/Harsh-9209/Conflict_Bench](https://huggingface.co/spaces/Harsh-9209/Conflict_Bench)*
*Hackathon adapter (old task, not re-scored): [huggingface.co/Harsh-9209/conflictbench-qwen2.5-3b-grpo-lora](https://huggingface.co/Harsh-9209/conflictbench-qwen2.5-3b-grpo-lora)*
