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

# ConflictBench — Resolving Conflicting Business Instructions by Authority

> **Status (November 2026):** the reward and the task were redesigned after an audit of this repo (see [What we found and fixed](#what-we-found-and-fixed)). Both are covered by tests that run on a CPU. **No GPU training run and no model evaluation has been done on the new task yet**, and the adapter trained during the hackathon has not been re-scored. The only measured numbers in this repo are rule-based and cheat baselines under [`results/`](./results/). The commands for the GPU runs are in [`results/README.md`](./results/README.md).

> **Meta × PyTorch × HuggingFace | OpenEnv Hackathon × SCALER School of Technology**

![Hackathon Banner](./assets/hackathon_banner.png)

[![HF Space](https://img.shields.io/badge/HF%20Space-ConflictBench-blue)](https://huggingface.co/spaces/Harsh-9209/Conflict_Bench)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## What it is

ConflictBench is a reinforcement learning environment plus a GRPO training pipeline. The task: a language model reads a set of business instructions sent by different people (a General Counsel, a CFO, a VP, a team lead, and so on). Some instructions clash, for example "hiring in Sales is frozen" against "open 4 new positions in Sales". The model must find each real clash, decide which instruction wins by authority, and output a JSON plan: which instructions to carry out and which to override.

Scoring is fully rule-based. Every scenario is generated from structured data, the correct answer comes from one deterministic function ([`resolver.py`](./resolver.py)), and the scorer ([`verifier.py`](./verifier.py)) compares the model's JSON with it. There is no LLM judge.

Parts of the repo:

| Part | Files |
|---|---|
| Scenario generator (task, difficulty levels, splits) | [`generator.py`](./generator.py), [`resolver.py`](./resolver.py) |
| Prompt shared by training, evaluation and the demo | [`prompts.py`](./prompts.py) |
| Scorer (reward) | [`verifier.py`](./verifier.py), spec in [`docs/REWARD_DESIGN.md`](./docs/REWARD_DESIGN.md) |
| Environment (Python) and OpenEnv HTTP/WebSocket server | [`conflict_bench.py`](./conflict_bench.py), [`server/`](./server/) |
| Baselines, evaluation harness, leaderboard | [`scripts/`](./scripts/), [`results/`](./results/) |
| GRPO training (one script, YAML configs) | [`training/`](./training/), [`configs/`](./configs/), guide in [`docs/TRAINING_GUIDE.md`](./docs/TRAINING_GUIDE.md) |
| Demo app (Gradio) and training Space | [`app.py`](./app.py), [`training_space/`](./training_space/) |
| Fixed held-out test set | [`data/test_v1.jsonl`](./data/test_v1.jsonl) and its SHA-256 [`data/test_v1.sha256`](./data/test_v1.sha256) |

How the pieces connect is in [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md).

**Related work.** Instruction priority is an active research topic (for example OpenAI's 2024 *Instruction Hierarchy* paper and later benchmarks such as IHEval). ConflictBench is not the first work on it. What it adds is a business setting with many stakeholders, rules beyond plain seniority, a deterministic scorer, and a documented record of how the first version of its reward was gamed and fixed.

**About "long-horizon".** The project was submitted under the hackathon's long-horizon instruction-following theme. One episode is a single step (one prompt, one JSON answer), so it is not long-horizon in the RL sense. A multi-turn version is future work (`FIX_BLUEPRINT.md`, P7-1).

---

## The task

Each scenario is built by `ScenarioGenerator` from a `GeneratorConfig` (both in [`generator.py`](./generator.py)). The default configuration is the redesigned task. `GeneratorConfig(legacy=True)` reproduces the original hackathon task exactly, kept only so old and new numbers can be compared.

### What the prompt contains

- **People, not tiers (hidden-hierarchy mode, the default).** Each instruction shows the sender's name and job title, for example `From Diego Sato (Manager, Financial Operations)`. The prompt states the general rules in words (legal, compliance and regulatory roles outrank everyone; then C-suite > VPs > directors > managers > team leads and coordinators) but never prints which office sits on which tier. The model has to map titles such as "Senior Finance Director" or "Group Chief Operating Officer" to a level itself. Each office has 5 job titles; the train, val and test splits use different ones.
- **Printed-hierarchy mode** (`GeneratorConfig(show_hierarchy=True)`): the prompt prints the numbered tier list with office names instead of job titles. The legacy task always prints it.
- **Context, status and notices.** A one-line company context, a `CURRENT STATUS` list of facts, and `STANDING NOTICES`.
- **Instructions**, shuffled, each with an ID such as `INS-20D8F5`, the sender and a date.
- The JSON format the answer must follow (see [Reward](#reward)).

### Rule types (all decided by [`resolver.py`](./resolver.py))

1. **Seniority.** On a clash, the higher level wins.
2. **Standing rule (scoped authority).** A notice can give one person the final say on one topic, whatever the other sender's seniority.
3. **Delegation.** A senior person can delegate their authority on one topic to someone junior, who then ranks at the delegator's level on that topic.
4. **Dated supersession.** Between instructions of equal authority, the later date wins. A later date never beats higher authority.
5. **Conditional instructions.** "If the audit is still open, ..." applies only if `CURRENT STATUS` says so; otherwise it is void (overridden, and part of no conflict).
6. **Chains.** "Contingent on [INS-...] going ahead: ..." is cancelled when the instruction it depends on is overridden.

Two instructions conflict only if they are on the same topic, give different directions, and their scopes overlap (company-wide overlaps everything; two different departments or systems do not).

### Distractors and balance

- **Same topic, no conflict.** Instructions on one topic that agree, that cover different departments or systems, or whose condition is false. A model that treats "same topic" as "conflict" loses points.
- **Decoy notices.** Standing rules and delegations on topics where they change nothing.
- **Balanced winners.** The restrictive side ("freeze", "block") wins about half of the conflicts, so "pick the strict-sounding one" no longer works. A test checks it stays between 45% and 55% on the test split (`tests/test_difficulty.py`).

### Difficulty levels

From `DIFFICULTY_LEVELS` in [`generator.py`](./generator.py). Distractor topics per scenario = conflicts × 0.75, rounded, limited by the 17 available topics.

| Level | Conflicts | Distractor topics | Extra single-instruction topics |
|---:|---:|---:|---|
| 1 | 2 | 2 | 1–2 |
| 2 | 4 | 3 | 1–2 |
| 3 | 6 | 5 | 1–2 |
| 4 | 8 | 6 | 0–1 |
| 5 | 10 | up to 7 | 0–1 |

In the fixed test set `data/test_v1.jsonl` (levels 1–3 only, 200 scenarios each) a level-1 scenario has 9–15 instructions, level 2 has 16–24 and level 3 has 25–37 (counted from that file).

### Splits

`GeneratorConfig(split="train" | "val" | "test")` draws instruction phrasings (5 per side of each topic: 3 train, 1 val, 1 test), job titles, company names, situations, sign-offs and follow-up actions from disjoint pools. The 17 topics and the rules are shared by all splits. Training uses only `train` and `val`; `test` is for the final evaluation. `data/test_v1.jsonl` is 600 test-split scenarios built once with seed 20261101 and committed with its SHA-256 so any change to it shows up.

---

## Reward

From [`verifier.py`](./verifier.py); the full spec, with the reasons for each part, is [`docs/REWARD_DESIGN.md`](./docs/REWARD_DESIGN.md).

The model must output one JSON object:

```json
{"identified_conflicts": [{"instruction_a": "<ID>", "instruction_b": "<ID>", "conflict_type": "direct|resource|temporal|conditional", "resolution": "<winning ID>", "reasoning": "<one sentence>"}],
 "execution_plan": ["<ID>", "..."],
 "overridden_instructions": ["<ID>", "..."]}
```

```
composite = (0.45 × correct_final_state + 0.30 × conflict_identification + 0.25 × no_contradictions) × format_gate

format_gate = 0    if no JSON object with all three keys is found
            = 0.9  if one is found but has type errors (e.g. a number where a list should be)
            = 1    otherwise
```

- **correct_final_state**: mean of F1(execution_plan, correct followed set) and F1(overridden_instructions, correct overridden set). A made-up ID counts as a false positive in both lists. An ID in both lists multiplies the rubric by (1 − overlap / number of instructions).
- **conflict_identification**: F1 of the conflict pairs listed (unordered, duplicates count once) × the share of correctly listed pairs whose `resolution` is the right winner. Listing a pair twice with both winners counts as wrong.
- **no_contradictions**: for each real conflict, 1 if exactly one of its two instructions is in the plan *and* the model listed that pair with that same instruction as the winner; else 0. Mean over the real conflicts. An empty plan scores 0.

`conflict_type` and `reasoning` are not scored. `efficiency` and `format_compliance` are still computed and logged but are not part of the composite.

---

## A real example

Scenario `2ec08947-df5a-32eb-7545-c4ed4bb3027d` from `data/test_v1.jsonl` (level 1). The fixed AUTHORITY and CONFLICTS rules at the top of the prompt are left out here; everything else is copied from the prompt:

```
CONTEXT: Northwind Retail is a retail chain preparing for a store expansion. Several leaders have sent instructions this month.

CURRENT STATUS:
- The security incident is still active.

STANDING NOTICES:
- On contractors, Diego Sato (Manager, Financial Operations) has the final say over every other instruction, whatever the sender's seniority.
- Wen Patel (Group Chief Operating Officer) has delegated their authority on customer discounts to Aisha Okafor (Manager, Employee Relations).

INSTRUCTIONS:
[INS-20D8F5] From Liam Walsh (EVP Human Resources), 23 Jan 2026:
  Share the customer success story for the Sales team this week.

[INS-C0EE47] From Maya Ahmed (VP, Software Engineering), 17 Mar 2026:
  Resume regular production releases in AWS production starting today.

[INS-9CA9A4] From Priya Brown (Team Leader), 9 Feb 2026:
  External communications in every department are frozen.

[INS-899218] From Carlos Brown (Senior Finance Director), 17 Feb 2026:
  Use of agency staff company-wide is suspended. Reply here if anything is unclear.

[INS-F3D9A6] From Diego Sato (Manager, Financial Operations), 18 Jan 2026:
  Engage outside specialists in every department as needed.

[INS-2812D5] From Aisha Okafor (Manager, Employee Relations), 14 Mar 2026:
  Training requests in every department can be approved by line managers. Reply here if anything is unclear.

[INS-4C3E7D] From Maya Tanaka (Group Chief Technology Officer), 8 Jan 2026:
  Do not push any release on the analytics platform until further notice.

[INS-8E8127] From Priya Brown (Team Leader), 30 Jan 2026:
  Directors within Finance may approve new vendors under $25,000.

[INS-B2EB49] From Wen Reyes (EVP Operations), 12 Mar 2026:
  Hold all pending vendor agreements in the Sales department for executive approval.
```

There are two real conflicts. External communications: the EVP (Sales) beats the Team Leader (every department), so `INS-20D8F5` wins. Contractors: the Senior Finance Director outranks the Manager, but the standing notice gives Diego Sato the final say on contractors, so `INS-F3D9A6` wins. The release instructions (AWS production vs the analytics platform) and the vendor instructions (Finance vs Sales) cover different systems or departments, so they are not conflicts. The delegation notice is a decoy: no instruction is about customer discounts.

The `resolver` answer, from `results/resolver__test_v1/completions.jsonl`, scores **1.0**:

```json
{"identified_conflicts": [{"instruction_a": "INS-20D8F5", "instruction_b": "INS-9CA9A4", "conflict_type": "direct", "resolution": "INS-20D8F5", "reasoning": "resolver"}, {"instruction_a": "INS-899218", "instruction_b": "INS-F3D9A6", "conflict_type": "direct", "resolution": "INS-F3D9A6", "reasoning": "resolver"}], "execution_plan": ["INS-20D8F5", "INS-2812D5", "INS-4C3E7D", "INS-8E8127", "INS-B2EB49", "INS-C0EE47", "INS-F3D9A6"], "overridden_instructions": ["INS-899218", "INS-9CA9A4"]}
```

The `oracle_lookup` cheat (seniority only, and it is handed the hidden topic and scope fields, so it finds both conflicts for free), from `results/oracle_lookup__test_v1/completions.jsonl`, ignores the standing notice and gives the contractor clash to the Director. It scores **0.7054** (correct_final_state 0.6786, conflict_identification 0.5, no_contradictions 1.0):

```json
{"identified_conflicts": [{"instruction_a": "INS-20D8F5", "instruction_b": "INS-9CA9A4", "conflict_type": "direct", "resolution": "INS-20D8F5", "reasoning": "baseline"}, {"instruction_a": "INS-899218", "instruction_b": "INS-F3D9A6", "conflict_type": "direct", "resolution": "INS-899218", "reasoning": "baseline"}], "execution_plan": ["INS-20D8F5", "INS-C0EE47", "INS-899218", "INS-2812D5", "INS-4C3E7D", "INS-8E8127", "INS-B2EB49"], "overridden_instructions": ["INS-9CA9A4", "INS-F3D9A6"]}
```

The `keyword_restrictive_v2` cheat (always picks the restrictive side) gets both clashes wrong and scores **0.4107** on this scenario (`results/keyword_restrictive_v2__test_v1/completions.jsonl`).

---

## What we found and fixed

The first version of this repo (the hackathon submission) reported large reward gains for a GRPO-trained model. An audit of the code in October 2026 ([`FIX_BLUEPRINT.md`](./FIX_BLUEPRINT.md)) found three problems.

1. **The reward could be gamed.** On the original task, with the original scorer, an answer with empty lists scored 0.340, listing every instruction scored 0.429, and a keyword rule that picks whichever side *sounds* strict, without looking at who sent it, scored 0.951 (`results/baselines.json`). The empty answer got credit because "no contradictions" gave full marks to an empty plan and "format" gave points for any JSON with the right keys.
2. **The task was a lookup.** Each conflict was always the two templates of one topic, the hierarchy was printed in the prompt, and the strict side won almost every time. A short template-matching script with a tier lookup scored 1.000 (`results/baselines.json`).
3. **The reported training numbers can't be reproduced.** The training script those numbers came from is not in the repo, and some reported per-rubric numbers are impossible under the repo's own scorer (FIX_BLUEPRINT C.2). They are archived, marked unverified, in [`docs/TRAINING_LOGS.md`](./docs/TRAINING_LOGS.md).

What changed:

- **Reward redesign** (`verifier.py`): the format check became a gate instead of free points, the overridden list is now scored, "no contradictions" needs an explicit, consistent choice on every real conflict, conflict identification multiplies pair F1 by resolution accuracy, the duplicate `efficiency` rubric was removed from the composite, and the parser no longer crashes on odd output. `tests/test_reward_hacks.py` keeps the cheat scores below fixed limits.
- **A harder task with splits** (`generator.py`, `resolver.py`): hidden hierarchy, standing rules, delegations, dates, conditions, chains, distractors, balanced winners, 17 topics, and train/val/test splits, with ground truth from one resolver. `tests/test_difficulty.py` checks that the shortcuts stay below fixed limits on the test split.
- **One evaluation harness and one training pipeline** (`scripts/evaluate.py`, `training/train_grpo.py`): the same prompt everywhere, every completion saved, and a leaderboard generated from the saved results.

### Cheat and rule-based baselines, before and after

Mean composite over 600 scenarios. No model is involved in any row.

| Strategy | Old task, old reward | Old task, new reward | New task (test_v1), new reward |
|---|---:|---:|---:|
| `empty` (empty lists) | 0.340 | 0.000 | 0.000 |
| `follow_all` (every ID in the plan) | 0.429 | 0.177 | 0.189 |
| `override_all` (every ID overridden) | 0.340 | 0.116 | 0.097 |
| `random_single` (one random ID in the plan) | 0.410 | 0.037 | 0.024 |
| `random_half` (random half in the plan) | 0.411 | 0.222 | 0.207 |
| `hallucinated_ids` (made-up IDs) | 0.340 | 0.000 | 0.000 |
| `keyword_restrictive` (strict-sounding side wins, old template matching) | 0.951 | 0.921 | 0.189 ¹ |
| `oracle_lookup_v1` (old template matching + tier lookup) | 1.000 | 1.000 | 0.189 ¹ |
| `keyword_restrictive_v2` (hidden topic/scope fields + strict side wins) | — | — | 0.669 |
| `latest_wins` (hidden topic/scope fields + later date wins) | — | — | 0.730 |
| `oracle_lookup` (hidden topic/scope fields + seniority, later date on ties) | — | — | 0.808 |
| `gt_one_conflict_flipped` (correct answer with one conflict reversed) | — | 0.805 | 0.841 |
| `resolver` / `ground_truth` (correct answer) | 1.000 ² | 1.000 | 1.000 |

Sources: column 1 = `results/baselines.json` (legacy generator, old verifier; the lookup row is the strategy then called `oracle_lookup`); column 2 = `results/baselines_v2.json` (legacy generator, new verifier); column 3 = the `data/test_v1.jsonl` table in [`results/leaderboard.md`](./results/leaderboard.md), built from `results/*__test_v1/summary.json`.

¹ The old template matcher finds no topics in the new phrasings, so it falls back to following every instruction and scores the same as `follow_all`.
² `ground_truth` only; `resolver` did not exist yet.

The three `v2`-style rule solvers are generous to the shortcut: they read the hidden structured fields (topic, direction, scope) that a model never sees, so conflict detection is free for them apart from conditions and chains. Even so, none of them solves the task.

---

## Results so far

| What | Status |
|---|---|
| Cheat and rule-based baselines on `data/test_v1.jsonl` | Measured; full table in [`results/leaderboard.md`](./results/leaderboard.md) |
| Qwen2.5-3B-Instruct base model on the new task | Not run yet (needs a GPU) |
| GRPO-trained model on the new task | Not trained yet |
| Hackathon adapter `Harsh-9209/conflictbench-qwen2.5-3b-grpo-lora`, re-scored | Not run yet |

The hackathon adapter was trained on the old task, with the old reward and a prompt that printed the hierarchy. The commands to score the base model, the old adapter and any API model are in [`results/README.md`](./results/README.md). Until those runs exist, there is no trained-model number to report.

---

## Running it

Python 3.10 or newer. The environment, scorer, tests, baselines and leaderboard all run on a CPU and need no torch.

### Tests

```bash
pip install -r requirements-dev.txt openenv-core==0.2.3 gradio pyyaml
python -m pytest -q
python scripts/check_links.py        # every relative markdown link resolves (case-sensitive)
```

### Use the environment from Python

```python
from conflict_bench import ConflictBenchEnv

env = ConflictBenchEnv(auto_scale_difficulty=False, fixed_difficulty=2, seed=0)
obs = env.reset()                    # obs["prompt"] is the instruction document
_, reward, done, info = env.step(env.sample_action())   # the correct answer -> reward 1.0
```

`ConflictBenchEnv` uses the default generator (new task, `train` split). With `auto_scale_difficulty=True` (the default) it serves level 1 for the first 200 episodes, level 2 up to episode 500, then level 3. `step()` returns the score breakdown and the correct answer in `info`; a second `step()` before `reset()` raises an error.

### Baselines, evaluation and the leaderboard

```bash
# score one strategy on the fixed test set -> results/<strategy>__test_v1/
python scripts/evaluate.py --strategy oracle_lookup --data data/test_v1.jsonl

# all strategies at once, generated on the fly (test split by default) -> one JSON file
python scripts/baselines.py --out results/baselines_test_v1.json

# rebuild results/leaderboard.md from every results/*/summary.json
python scripts/make_leaderboard.py
```

`scripts/evaluate.py` also takes `--model` (a Hugging Face model, optional `--adapter`) or `--api` (any OpenAI-compatible endpoint; the key is read from `CONFLICTBENCH_API_KEY`). It writes every completion and a summary with a seeded 95% bootstrap interval, per-difficulty and per-rubric means, and the parse-failure rate. Details: [`results/README.md`](./results/README.md).

### Training (GPU)

```bash
pip install -r requirements-train.txt          # Linux + NVIDIA GPU; see the header of that file for torch
python -m training.train_grpo --config configs/grpo_t4.yaml --dry-run   # build datasets, print token stats, no GPU
python -m training.train_grpo --config configs/grpo_smoke.yaml          # 10-step plumbing test, Qwen2.5-0.5B
python -m training.train_grpo --config configs/grpo_t4.yaml             # free Kaggle / Colab T4
python -m training.train_grpo --config configs/grpo_a10g.yaml           # A10G / L4 24 GB
```

| Config | Model | LoRA r | Train / val scenarios | Difficulty mix (train) |
|---|---|---:|---|---|
| `grpo_smoke.yaml` | `unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit` | 8 | 20 / 4 | level 1 only |
| `grpo_t4.yaml` | `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` | 16 | 200 / 20 | 60% level 1, 40% level 2 |
| `grpo_a10g.yaml` | `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` | 16 | 600 / 30 | 40% / 40% / 20% levels 1–3 |

The reward during training is exactly `verifier.score(...).composite`; each rubric is logged separately. Training resumes from the latest checkpoint in `output_dir`, picks the best checkpoint by validation reward, and writes its config, git commit, package versions and metrics to `results/<run_name>/`. None of these configs has been run yet; the pinned package versions in `requirements-train.txt` are also not yet confirmed by a GPU run. Full guide: [`docs/TRAINING_GUIDE.md`](./docs/TRAINING_GUIDE.md).

### OpenEnv server

```bash
python -m server.app --port 8000          # or: uvicorn server.app:app --port 8000
```

Endpoints from `openenv-core` 0.2.3: `POST /reset`, `POST /step`, `GET /state`, `/schema`, `/metadata`, `/health`, and `WS /ws`. Plain HTTP `/reset` and `/step` create a fresh environment per request, so a real reset-then-step episode needs the `/ws` session. `reset` accepts `seed`, `difficulty` (1–5) and `domain` (`HR`, `Finance`, `IT`, `Operations`). The served environment keeps `reveal_ground_truth` off, so observations never contain the correct answer.

### Demo app

```bash
pip install -r requirements.txt     # torch, transformers, peft, gradio; a GPU is strongly advised
python app.py
```

The Gradio app (also the [HF Space](https://huggingface.co/spaces/Harsh-9209/Conflict_Bench)) generates a new-task scenario at level 1–3, runs Qwen2.5-3B-Instruct with the LoRA adapter switched off ("base") and on ("trained") with greedy decoding by default, and scores both answers with the verifier. The adapter is the hackathon one, trained on the old task, so this comparison is not a measured result. Models load on the first click. Set `CONFLICTBENCH_ADAPTER` to use a different adapter, or to an empty string to run only the base model.

### Training Space

[`training_space/`](./training_space/) is the source of a separate, password-locked HF Space that runs `configs/grpo_a10g.yaml`. Its Docker build clones this repo at a fixed tag or commit (`CONFLICTBENCH_REF`); see [`docs/TRAINING_GUIDE.md`](./docs/TRAINING_GUIDE.md#training-space). Do not push `training_space/` to the main demo Space.

---

## Repository layout

```
conflict_bench.py        ConflictBenchEnv (reset / step / state)
generator.py             ScenarioGenerator, GeneratorConfig, DIFFICULTY_LEVELS, splits, legacy mode
resolver.py              resolve(scenario): the only source of ground truth
verifier.py              score(): parser, three rubrics, format gate, RUBRIC_WEIGHTS
prompts.py               SYSTEM_PROMPT and build_messages()
scenario_io.py           scenario <-> JSON
server/                  OpenEnv FastAPI app, environment wrapper, action/observation models
scripts/                 baselines.py, evaluate.py, backends.py, make_leaderboard.py, diagnose_tokens.py, check_links.py
training/                train_grpo.py (pipeline), config.py (YAML loading and checks)
configs/                 grpo_smoke.yaml, grpo_t4.yaml, grpo_a10g.yaml
data/                    test_v1.jsonl (fixed test set) and its SHA-256
results/                 baseline runs, summaries, leaderboard.md (all generated by scripts)
tests/                   pytest suite (CPU only)
app.py                   Gradio demo
inference.py             run an adapter on one generated scenario and print its score
training_space/          separate training Space (Dockerfile, Gradio UI, glue to training/)
train_grpo.py, diagnose_tokens.py   deprecated shims that forward to training/ and scripts/
docs/                    ARCHITECTURE.md, REWARD_DESIGN.md, TRAINING_GUIDE.md, TRAINING_LOGS.md
FIX_BLUEPRINT.md         the audit and the fix plan
```

---

## Links

| Resource | URL |
|---|---|
| HF Space (demo) | [Harsh-9209/Conflict_Bench](https://huggingface.co/spaces/Harsh-9209/Conflict_Bench) |
| HF Space (training) | [Harsh-9209/NEW_SPACE](https://huggingface.co/spaces/Harsh-9209/NEW_SPACE) |
| Hackathon LoRA adapter (old task, not re-scored) | [Harsh-9209/conflictbench-qwen2.5-3b-grpo-lora](https://huggingface.co/Harsh-9209/conflictbench-qwen2.5-3b-grpo-lora) |
| Hackathon Colab notebook (old training script, not updated) | [Open in Colab](https://colab.research.google.com/drive/18UJSpREGN152swrVjkEbGa0aWJR7eROH?usp=sharing) |
| Original write-up, with a November 2026 update | [blog.md](./blog.md) |

---

*ConflictBench — Meta × PyTorch × HuggingFace | OpenEnv Hackathon × SCALER School of Technology | Harsh Jain*
