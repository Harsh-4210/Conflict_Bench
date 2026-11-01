# ConflictBench — Fix Blueprint

**Written:** 31 Oct 2026, from a full read of every tracked file plus experiments run against the real code.
**Who it's for:** you (Harsh) and any coding agent (Claude, Codex, Cursor) that will do the fixes.
**How to use it:** do the phases in order. Each task has an ID, the files it touches, exact steps, and a "Done when" check. An agent should do **one task (or one small group of tasks) per branch or PR**, run the check, and stop.

---

## Part A — The short version (plain language)

### What the project is
ConflictBench gives an AI a list of office instructions from different bosses (Legal, CEO, VP, Team Lead…). Some instructions clash, like "hiring is frozen" vs "hire 4 engineers". The AI must spot each clash, pick the winner (the more senior boss), and output a JSON plan. A scoring script marks the answer from 0 to 1. The project then trained a small model (Qwen2.5-3B) with GRPO, a type of reinforcement learning, to raise that score.

### What's wrong, in one paragraph
The idea is good, but right now **the test is too easy and the scoring is easy to cheat.** An answer with **nothing in it scores 0.34**. Listing **every instruction** scores **0.43**. Picking whichever side *sounds* strict, without even looking at who sent it, scores **0.95**. A **30-line script with no AI at all scores a perfect 1.00**. The trained model's headline result (0.50) is only a little better than "list everything". Several claims in the README, blog, PRD and docs don't match the code: "the hierarchy is never stated in the prompt" (it's printed in the prompt), "8–28 instructions" (really 6–16), "ungameable" (it is gameable), and a before/after example labelled "actual model output" that the code could not have produced. Some reported per-rubric numbers are **mathematically impossible** with this scoring code. On top of that, the API server can't start with the current OpenEnv version, and the demo app's "base model" panel secretly runs the fine-tuned model too.

### Is it worth your time?
**Yes, but only a focused amount, and in a set order.**

- **As it stands, it's a risk on your résumé.** Anyone who checks it (an interviewer, a hackathon judge, a reviewer) can find the 0.34-for-nothing problem in about ten minutes. Until Phase 0 is done, don't quote "257% improvement" or "0.50 reward" anywhere.
- **With about 25–35 hours (Phases 0–2 plus part of 4), it becomes a solid portfolio project.** You can honestly say "I found reward hacking in my own RL environment, proved it, fixed it, and re-measured." That story is worth more to interviewers than the original claims.
- **With about 80–120 more hours (Phases 3–7), it could become a real public benchmark** or a workshop paper. Other teams already work on this topic ("instruction hierarchy" research, e.g. OpenAI's 2024 *Instruction Hierarchy* paper and later benchmarks such as IHEval), so drop the "first ever" claim and sell what's different about yours: deterministic scoring, a business setting, and multi-step, multi-stakeholder conflicts. Do that only if you actually want a research angle.
- **Timing:** your final-year project is paused for exams (31 Oct to about 4 Dec). Treat this the same way. If you have 2–3 spare hours before exams, do **Phase 0 only** (stop the false claims from spreading). Start Phase 1 after exams.

### Recommended time plan

| When | Phases | Hours | What you get |
|---|---|---|---|
| Before/around exams (optional) | Phase 0 | 2–4 h | No more false public claims; a safety net of tests |
| Week 1 after exams | Phase 1 | 6–8 h | Code that doesn't crash, a server that starts, a fair demo |
| Week 2 | Phase 2 | 6–8 h | Scoring that can't be cheated, proven by tests |
| Week 3 | Phase 4 (baselines only) + Phase 6 | 6–8 h | Honest numbers and an honest README → **résumé-ready here** |
| Weeks 4–6 (optional) | Phase 3 + 5 | 20–30 h | A task that's actually hard; one clean, re-runnable training run |
| Later (optional) | Phase 7 | 40–80 h | Public benchmark / leaderboard / paper |

GPU budget: Phases 0–4 need **no GPU** apart from an optional base-model baseline on free Colab/Kaggle. Phase 5 needs one real training run (one Kaggle T4 session with a small config, or about $10–15 of A10G/L4 time).

---

## Part B — Rules for any agent doing this work

1. **Repo:** `C:\Users\Harsh Jain\Desktop\PROJECTS\conflictbench` (GitHub `Harsh-4210/Conflict_Bench`, HF Space remote `hf`).
2. **Before you start:** `git fetch`, check open PRs and new commits on `main`, then branch from up-to-date `main`. Branch name: `fix/<task-id>-short-name`.
3. **Never push to the `hf` remote** (the live HF Space) unless Harsh asks for it in so many words.
4. **Never commit `.env`**. It holds `HF_TOKEN`. It's already in `.gitignore`; keep it there.
5. **Python:** the repo `.venv` is Python 3.14. Use it for CPU work (generator/verifier/tests). Training needs a separate CUDA environment (Python 3.10/3.11). Never install torch/unsloth into `.venv` just to run tests.
6. **Every task ends with tests passing:** `.venv/Scripts/python -m pytest -q` (from Phase 0 on).
7. **Don't make up numbers.** Any number written into README/blog/docs must come from a script in the repo whose output is saved under `results/`. If you haven't run it, write "not yet measured".
8. **Keep the public API names** (`ConflictBenchEnv`, `ScenarioGenerator`, `score`, `ScoreBreakdown`) unless a task says to change them.
9. **Match the existing code style:** dataclasses, type hints, short docstrings, section-comment banners.
10. One task = one focused commit. Commit message format: `fix(verifier): <what>` / `feat(generator): <what>` / `docs: <what>`.

---

## Part C — Evidence: every flaw found

Severity: **S1** = invalidates results or makes public claims false · **S2** = real bug / crash / unfair comparison · **S3** = quality, maintainability, inconsistency.

### C.1 Measured with experiments (scratch script, 600 scenarios, 200 per difficulty)

| Strategy (no AI involved) | Mean composite score |
|---|---|
| Output empty lists `{"identified_conflicts":[],"execution_plan":[],"overridden_instructions":[]}` | **0.340** |
| Put every instruction ID in `execution_plan` | **0.430** |
| Put one random ID in `execution_plan` | **0.411** |
| Keyword rule ("pick the side that sounds restrictive"), **ignores authority completely** | **0.951** (first rough version: 0.744; re-measured with `scripts/baselines.py`) |
| ~30-line template-matching script + hierarchy lookup | **1.000** |
| *For comparison: reported trained-model peak* | *0.50* |
| *Reported "Run 2 start"* | *0.37 (≈ the empty-answer score)* |

Other measured facts:
- The "restrictive" instruction (freeze/block/limit) wins **2346 of 2400** conflicts (97.75%).
- Instructions per scenario: difficulty 1 = **6–8**, difficulty 2 = **10–12**, difficulty 3 = **14–16**. The README/PRD say 8–28.
- **Every** difficulty-3 prompt is longer than 2800 characters (200/200). The training scripts drop prompts over 2800 chars, so **difficulty 3 is never trained on and never evaluated**, even though the eval mix asks for 20% of it.
- Same seed → same text but **different instruction IDs** (`uuid4` ignores the seed). Scenarios can't be reproduced exactly.
- Crash inputs (raise from `score()`): the bare output `[]` (`AttributeError`, found in Phase 0), `"execution_plan": null`, `"execution_plan": 5`, `"identified_conflicts": 5`, conflict IDs given as lists.
- `overridden_instructions` is **never scored**. Adding or removing it changes nothing.
- `server/app.py` **fails to import** with the installed `openenv-core 0.2.3`: `cannot import name 'TextAction'`.

### C.2 The "impossible numbers" proof (S1)
In `verifier.py`, efficiency = `tp / (tp + fp + fn)` (the Jaccard overlap of the plan with the right answer), and correct-final-state = F1 = `2J / (1 + J)`. For any single answer, **F1 ≥ Jaccard**, so the average of R1 (correct final state) can never be below the average of R4 (efficiency). The docs report baseline R1 = 0.11 with R4 = 0.62, and trained R1 = 0.48 with R4 = 0.71 (`docs/REWARD_DESIGN.md`, `docs/TRAINING_LOGS.md`, PRD §9). **Those numbers can't come from this verifier.** Either a different scorer produced them or they were estimated. The training script named in the logs (`detect_gpu_config()`, L40S paths) isn't in the repo either. **Treat every reported number as unverified until Phase 4 re-measures it.**

This also shows R4 adds nothing new: it's just another way of writing R1, so 45% of the reward measures the same thing twice.

### C.3 Full findings table

| ID | Sev | Where | Flaw | Fixed in |
|---|---|---|---|---|
| F-01 | S1 | `verifier.py:156-157` | Empty plan gets `no_contradictions = 1.0` (25% of reward for doing nothing) | P2-1 |
| F-02 | S1 | `verifier.py:268-292` | Format scores 0.9 for *any* JSON with the 3 keys; the `isinstance` checks are always true because the parser already turned the values into lists | P2-4 |
| F-03 | S1 | `generator.py:452-453`, template sources | Restrictive side wins 97.75% of the time, so the model can learn "pick the prohibition" without reading authority | P3-1 |
| F-04 | S1 | `generator.py:519-528` | Prompt prints the full hierarchy, yet README/app/PRD/blog say it is "never stated" | P6-1 (claims), P3-3 (hidden-hierarchy mode) |
| F-05 | S1 | whole task | Fully solvable by a lookup script (1.00) because each clash = the same template group and the source tier is printed | P3 |
| F-06 | S1 | `verifier.py` | `overridden_instructions` never scored; an ID can be both followed and overridden without penalty | P2-2 |
| F-07 | S1 | docs/README/PRD/blog | Reported numbers can't be reproduced (C.2); "actual model output" example is made up (its "trained" output would score ~1.0, not 0.50) | P4, P6 |
| F-08 | S1 | `docs/REWARD_DESIGN.md` | Describes different formulas (binary R2, multiplicative R3, different R4, different JSON schema `followed/overridden/conflicts`) from the code | P6-2 |
| F-09 | S2 | `verifier.py:80-81,185-187,209-210` | Crashes on `null`/number values and unhashable IDs; server `step()` would return 500 | P1-1 |
| F-10 | S2 | `verifier.py:42-49` | Grabs the first code fence, then a greedy `{...}` from the first `{` to the last `}`. Two objects or a `{` inside reasoning text → parse fail | P1-1 |
| F-11 | S2 | `generator.py:402-404,421` | `random.seed()` changes the *global* RNG (side effect), and `uuid4` IDs ignore the seed → not reproducible; two generators interfere | P1-2 |
| F-12 | S2 | `generator.py:456-462` | Retry loop for ties can finish without fixing the tie (20 tries); no assert. Tie odds per try reach 22% for `overtime_authorization` | P1-2 |
| F-13 | S2 | `generator.py:479`, prompt `:536` | Ground-truth `conflict_type` is `absolute/resource/time_ordered`; prompt asks for `direct/resource/temporal`. Not scored, but the shown "ground truth" disagrees with the instructions | P1-3 |
| F-14 | S2 | `server/conflictbench_environment.py:13,38,51` | Imports removed `TextAction/TextObservation`; `reset/step` signatures miss `seed`, `episode_id`, `timeout_s`; server can't start | P1-4 |
| F-15 | S2 | server + generator | `SUPPORTS_CONCURRENT_SESSIONS=True` but every session shares the global `random` | P1-2, P1-4 |
| F-16 | S2 | `conflict_bench.py:34-43,109-111` | Local fake `Rubric/RubricSet` stubs, never used for scoring; claims to "use OpenEnv's Rubric system". Real `openenv.core.rubrics.Rubric` exists | P1-5 |
| F-17 | S2 | `conflict_bench.py:210-229` | `step()` can be called again after `done`; it re-scores and bumps `episode_count` | P1-5 |
| F-18 | S2 | `app.py:54-57` | `PeftModel.from_pretrained(_base_model, …)` puts the LoRA layers into the base model object, so the "Base Model" button also runs the fine-tuned weights. The before/after demo is invalid | P1-6 |
| F-19 | S2 | `app.py:79`, `train_grpo.py:453`, `train_grpo_prod.py:333` | Demo and eval feed raw text to an *Instruct* model without the chat template, and training used the template → the base model is handicapped, which unfairly inflates "improvement" | P1-6, P4-2 |
| F-20 | S2 | `app.py:79` | `truncation=True, max_length=1800` cuts the **end** of long prompts, which is exactly where the JSON format instructions are | P1-6 |
| F-21 | S2 | `app.py:172-203` | Model output is pasted into HTML without escaping (HTML/script injection risk); `.get`/slicing crash if an entry isn't a dict/str | P1-6 |
| F-22 | S2 | `app.py:86`, all eval funcs | `do_sample=True, temperature=0.7`, one sample → results change every click/run | P1-6, P4-2 |
| F-23 | S2 | `training_space/app.py` | Public "Start Training" button: anyone visiting the Space can start an A100 job on your bill. No auth, no lock across workers | P0-3 |
| F-24 | S2 | `train_grpo.py:138`, `train_script.py` | Prompt length filter in **characters** (2800) passed as `max_prompt_length` in **tokens**; silently removes all difficulty 3 | P5-1 |
| F-25 | S2 | `train_grpo.py:277-278` | "Difficulty bonus" ×1.05. GRPO normalises rewards inside each prompt's group, so a per-prompt multiplier does almost nothing. It's noise, and it breaks the [0,1] meaning | P5-1 |
| F-26 | S2 | `train_grpo.py:282-284` | `except Exception: reward 0.0` with debug-level logging hides verifier crashes during training | P5-1 |
| F-27 | S2 | train scripts | Train and eval come from the same generator, same templates → eval measures template memorisation, not generalisation. No held-out split | P3-6, P4-1 |
| F-28 | S2 | `docs/TRAINING_GUIDE.md` | Says TRL auto-resumes and `load_best_model_at_end=True` is set. Neither is true in the code | P5-2, P6 |
| F-29 | S2 | `inference.py` | Different system prompt (asks for `<thought>` + JSON, unlike training), different model ID, made-up sample with sources not in the hierarchy ("HR Director"), hard-coded `.to("cuda")` | P1-7 |
| F-30 | S3 | `pyproject.toml` | `package-dir "conflictbench" = "."` packages the whole repo root; `verifier.py` does `from generator import …` (absolute), which breaks once installed. Version 1.0.0 vs yaml 2.0.0 | P1-8 |
| F-31 | S3 | `openenv.yaml` | Points to `hf_space_a100/app.py`, `hf_space_a100/train.py` (don't exist); lora_rank 32 (code: 16); results unverified | P1-8, P6 |
| F-32 | S3 | `requirements.txt` | Missing `trl`, `datasets`, `unsloth`, `python-dotenv`, `matplotlib`; nothing pinned; `pytest` mixed in with runtime deps | P1-8 |
| F-33 | S3 | `training_space/Dockerfile` | Installs unsloth from git HEAD (unpinned) → build breaks randomly; Space clones repo HEAD (unpinned) | P5-3 |
| F-34 | S3 | repo | 3 copies of the training code (`train_grpo.py`, untracked `train_grpo_prod.py`, `training_space/train_script.py`) with different configs; UI table says 600/30 but the script uses 300/10 | P5-1 |
| F-35 | S3 | repo | No tests, no CI, no LICENSE file (README links to one) | P0-1, P0-4 |
| F-36 | S3 | `docs/architecture.md` | Tracked in lower case, but README/PRD link to `docs/ARCHITECTURE.md` / `doc/ARCHITECTURE.md` → broken links on GitHub (case-sensitive) | P6-3 |
| F-37 | S3 | README/PRD/arch | Wrong facts: "100+ templates" (60), "16 sources" (18), "4000-char guard" (doesn't exist), hierarchy levels described differently from code, curriculum 80/20 vs 60/40, 400 vs 600 vs 300 scenarios | P6-1 |
| F-38 | S3 | theme claim | "Long-horizon": the env is one step with ≤16 short items, so it isn't long-horizon | P6-1, P7-1 |
| F-39 | S3 | `generator.py:486-506` | Fillers only come from groups not used in conflicts, so "same topic but not a conflict" never appears. Any repeated topic = conflict (trivial detection) | P3-2 |
| F-40 | S3 | `generator.py` | `instruction_type` "conditional" and the `condition` field are defined but never used | P3-4 |
| F-41 | S3 | `app.py:1103-1106` | Says "will retry at inference time" but nothing retries; model loads at import (slow startup, blocks tests) | P1-6 |
| F-42 | S3 | `train_grpo.py:62,380` | Hard-coded `HF_REPO_ID` (pushes to your repo by default); `report_to="wandb"` crashes without a wandb login | P5-1 |
| F-43 | S3 | `diagnose_tokens.py` | Checks 512 new tokens / 4096 window, but training uses 768–1024 | P5-1 |

---

## Part D — The phases

Every task has: **Goal · Files · Steps · Done when**.

---

### PHASE 0 — Stop the bleeding (2–4 h, no GPU)

Goal: no false claims stay public, a test harness exists, and nobody can spend your money.

#### P0-1 · Add a test harness and pin the current behaviour
- **Files:** new `tests/__init__.py`, `tests/conftest.py`, `tests/test_smoke.py`, `pyproject.toml` (`[tool.pytest.ini_options]`), new `requirements-dev.txt`.
- **Steps:**
  1. `requirements-dev.txt`: `pytest>=8`, `pytest-cov`. Install: `.venv/Scripts/python -m pip install -r requirements-dev.txt`.
  2. `conftest.py`: add the repo root to `sys.path` (until P1-8 fixes packaging).
  3. `test_smoke.py`: (a) ground-truth `sample_action()` scores 1.0 for difficulties 1–3 over 50 seeds; (b) `score("garbage")` is 0.0; (c) composite always in [0,1].
  4. Add `[tool.pytest.ini_options] testpaths=["tests"]`.
- **Done when:** `.venv/Scripts/python -m pytest -q` passes.

#### P0-2 · Add a "cheat baselines" script (this becomes your most important regression test)
- **Files:** new `scripts/baselines.py`, new `results/` folder, `tests/test_reward_hacks.py`.
- **Steps:**
  1. Write `scripts/baselines.py`. It generates a fixed set (seed 1234, 200 per difficulty) and scores these strategies: `empty`, `follow_all`, `override_all`, `random_single`, `random_half`, `keyword_restrictive` (picks the side with words like freeze/suspend/restrict, never looks at the source), `oracle_lookup` (template match + tier lookup), and `ground_truth`.
  2. Print a table and write it to `results/baselines.json` (strategy → mean, std, per-difficulty mean).
  3. `tests/test_reward_hacks.py` imports the strategies and **asserts the current (bad) values as `xfail`** with a clear reason, e.g. `assert mean(empty) < 0.05`. They'll start passing in Phase 2.
- **Done when:** the script reproduces roughly empty ≈ 0.34, follow_all ≈ 0.43, oracle ≈ 1.0, and the hack tests show as `xfail`.

#### P0-3 · Lock the public training Space (cost safety)
- **Files:** `training_space/app.py`.
- **Steps:**
  1. Require a secret: read `TRAIN_PASSWORD` from env; add a password `gr.Textbox(type="password")`. `start_training` refuses unless it matches (`hmac.compare_digest`). If the env var is missing, the button is disabled.
  2. Fix the header table so it shows the values the script actually uses (import them from `train_script.py` instead of hard-coding them).
  3. Tell Harsh (in the PR text) to also set the Space to **private** or **paused** on huggingface.co. That's a manual step an agent must not do.
- **Done when:** with no `TRAIN_PASSWORD` set, clicking Start does nothing and shows a message.

#### P0-4 · Add a "status" banner to the README and a LICENSE
- **Files:** `README.md`, new `LICENSE` (MIT, "Harsh Jain", 2026).
- **Steps:** add right under the title:
  > **Status (Oct 2026):** an internal audit found that the reward can be partly gamed (an empty answer scores 0.34) and that the reported training numbers can't be reproduced from this repo. Fixes are in progress (see `FIX_BLUEPRINT.md`). Treat the results below as unverified.

  Don't rewrite the rest yet (that's Phase 6).
- **Done when:** the banner and LICENSE exist. (Pushing to GitHub/HF is Harsh's call.)

---

### PHASE 1 — Make the code correct (6–8 h, no GPU)

#### P1-1 · Harden the output parser
- **Files:** `verifier.py` (`parse_agent_output`), `tests/test_parser.py`.
- **Steps:**
  1. Accept `str`, a list of message dicts (TRL), or anything else via `str()`.
  2. **Extraction order:** (a) every fenced block, tried **last first** (models often fix themselves at the end); (b) otherwise scan with `json.JSONDecoder().raw_decode` from each `{` position and keep the **last** object that contains the required keys. Remove the greedy regex.
  3. **Type-normalise** each field with a helper `_as_id_list(v)`: `None` → `[]`; `str` → `[v]`; `list` → keep items that are `str`/`int` (as `str`, stripped), drop the rest; anything else → `[]` and record a `schema_errors` entry.
  4. `identified_conflicts`: keep only `dict` entries; coerce `instruction_a/b/resolution` to `str` when they're scalars, else drop that entry and record a schema error.
  5. Add `schema_errors: List[str]` to `AgentOutput` (used by P2-4). `parsed_ok` = a JSON object was found with all 3 keys.
  6. Normalise IDs: `.strip().upper()`, so `"ins-ab12cd"` matches `INS-AB12CD`.
- **Tests:** every crash probe from C.1 returns a score and doesn't raise; two objects in text → the last valid one is used; a fenced block followed by trailing prose works; `{` inside a reasoning string works.
- **Done when:** a fuzz test passes: 2000 random JSON-ish strings (use `random` + `json.dumps` of nested random values), and none raises.

#### P1-2 · Make the generator reproducible and isolated
- **Files:** `generator.py`, `tests/test_generator.py`.
- **Steps:**
  1. `self.rng = random.Random(seed)`; replace **every** `random.` call in the class with `self.rng.` (`_fill`, `_make_instruction`, `_generate_*`, `_format_prompt`, `generate`, `generate_batch`). Never call `random.seed`.
  2. IDs: `f"INS-{self.rng.getrandbits(24):06X}"` with the same collision loop. `scenario_id`: `uuid.UUID(int=self.rng.getrandbits(128))`.
  3. Tie handling: after the retry loop, `assert instr_a.source_priority != instr_b.source_priority`. Better still, pick the second source *from the allowed sources that have a different tier* so retries aren't needed.
  4. `_format_prompt` must not shuffle the caller's list in place: `ordered = list(instructions); self.rng.shuffle(ordered)`.
  5. Add `generate(…, seed=None)` per-call override: if given, use `random.Random(seed)` for that call only (needed for fixed eval sets).
- **Tests:** same seed → identical `Scenario` (IDs, text, order, ground truth); two generators with different seeds don't affect each other; no tie across 5000 scenarios; `random.random()` sequence is unchanged by creating a generator.
- **Done when:** tests pass, and the old smoke test still scores 1.0.

#### P1-3 · Make the conflict-type labels agree
- **Files:** `generator.py`.
- **Steps:** add a mapping `{"absolute": "direct", "resource": "resource", "time_ordered": "temporal", "conditional": "conditional"}` and store the mapped value in `ConflictPair.conflict_type`. Use the same allowed list in the prompt. (Scoring it is optional: see P2-3.)
- **Done when:** every ground-truth `conflict_type` is one of the prompt's allowed values (test).

#### P1-4 · Fix the OpenEnv server so it starts
- **Files:** `server/conflictbench_environment.py`, `server/app.py`, new `server/models.py`, `tests/test_server.py`.
- **Steps:**
  1. Look at what `openenv-core 0.2.3` provides: `.venv/Scripts/python -c "import openenv.core.env_server.types as t; print(dir(t))"` and read `openenv/core/env_server/interfaces.py`. Use the library's real base classes (`Action`, `Observation`, `State`). If no text action type exists, define `ConflictBenchAction(Action)` with `text: str` and `ConflictBenchObservation(Observation)` with `text: str` in `server/models.py`. Match the upstream example envs under `site-packages/openenv` if there are any.
  2. Match signatures exactly: `reset(self, seed=None, episode_id=None, **kwargs)` and `step(self, action, timeout_s=None, **kwargs)`. Pass `seed` down to the generator (P1-2).
  3. Each environment instance owns its own `ScenarioGenerator` (no globals), so concurrent sessions are safe.
  4. `step` before `reset` → a clear error observation, not an exception. `step` after done → error observation, no re-scoring.
  5. Don't put the ground truth in the step observation unless a `reveal_ground_truth` flag is set at construction. Default **off** for the server (an agent being tested over HTTP must not see answers it could reuse).
- **Tests:** use FastAPI `TestClient(app)`: `POST /reset` → 200 with the prompt text; `POST /step` with the GT answer → reward 1.0; with garbage → reward 0, status 200.
- **Done when:** `.venv/Scripts/python -m server.app --port 8001` starts and the tests pass.

#### P1-5 · Clean up `ConflictBenchEnv`
- **Files:** `conflict_bench.py`.
- **Steps:**
  1. Delete the fake `Rubric/RubricSet` stubs and `_build_rubric_set`. Optional and better: wrap the 5 rubric functions in real `openenv.core.rubrics.Rubric` subclasses (read `openenv/core/rubrics/base.py` first) and have `score()` use them, so the "uses OpenEnv rubrics" claim becomes true.
  2. Guard `step()`: if `self._done`, raise `RuntimeError("Episode finished; call reset()")`.
  3. `sample_action()` returns `str` (`json.dumps`) to match its docstring; update the `__main__` smoke test.
  4. `reset(seed=None)`: forward to the generator.
  5. Remove the `sys.path.insert` hack once P1-8 is done (use relative imports).
- **Done when:** all tests pass; `python conflict_bench.py` still prints GT = 1.0.

#### P1-6 · Make the demo app fair and safe
- **Files:** `app.py`.
- **Steps:**
  1. **Base vs trained bug (F-18):** load the adapter with `PeftModel.from_pretrained(base, id)` *once*, then run the base model with `with _trained_model.disable_adapter(): …` and the trained model normally. Or load two separate model objects if memory allows. Add a startup self-check: same prompt, greedy, adapter on vs off → outputs should differ; log a warning if they're identical.
  2. **Chat template (F-19):** build messages exactly like training does (same system prompt; see P5-1, take it from one shared module `prompts.py`) and call `tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")`.
  3. **Truncation (F-20):** no right-side truncation. If the prompt is longer than the limit, show an error instead of silently cutting it.
  4. **Determinism (F-22):** greedy decoding (`do_sample=False`) by default, with an optional "sample" checkbox.
  5. **Escaping (F-21):** `html.escape()` every model-supplied string (IDs, type, reasoning) before inserting it into HTML; guard `isinstance(c, dict)` and `str(...)[:200]`.
  6. **Lazy loading (F-41):** load models on the first inference call (cached), not at import, so `import app` works in tests without a GPU.
  7. **False text:** replace the Methodology bullets "Authority hierarchy is never stated…" and "non-gameable gradient signal", and replace the Training Results tab numbers with values read from `results/*.json` (Phase 4). Until then show "Not yet re-measured".
- **Done when:** `python -c "import app"` works on CPU without downloading models; the GT reveal tab renders; an escaping test passes (`<script>` in reasoning comes out as text).

#### P1-7 · Fix or delete `inference.py`
- **Files:** `inference.py`.
- **Steps:** use the shared `prompts.py` system prompt (JSON only, no `<thought>`), take the model ID from one constant/env var (`CONFLICTBENCH_ADAPTER`, default the real HF repo from `openenv.yaml`), take input from a **generated** scenario (`ScenarioGenerator(seed).generate()`), print the score, and use `model.device` instead of `"cuda"`.
- **Done when:** the script's prompt matches training character for character (a test compares `prompts.build_messages(sc)` usage).

#### P1-8 · Packaging and config hygiene
- **Files:** move code into a `conflictbench/` package folder: `conflictbench/{__init__,generator,verifier,env,prompts}.py` and `conflictbench/server/…`; update `pyproject.toml`, `openenv.yaml`, `requirements.txt`, new `requirements-train.txt`.
- **Steps:**
  1. `git mv` the files (keeps history). Use relative imports (`from .generator import Scenario`).
  2. Leave thin shims at the old root paths (`conflict_bench.py`, `generator.py`, `verifier.py`, each doing `from conflictbench.X import *`) so external links and the Colab notebook keep working. Mark them deprecated.
  3. `pyproject.toml`: normal `packages = find:` with `where=["."]`, `include=["conflictbench*"]`; one version number (`2.0.0`) in one place (`conflictbench/__init__.__version__`).
  4. `requirements.txt` = runtime env/verifier/server only (`openenv-core==0.2.3`, `fastapi`, `uvicorn`). `requirements-train.txt` = pinned `torch`, `transformers`, `trl`, `peft`, `datasets`, `unsloth`, `bitsandbytes`, `accelerate`, `python-dotenv`, `matplotlib` (pin to versions that work together; record them from a working Colab/Kaggle run with `pip freeze`). Demo app: `requirements-app.txt`.
  5. `openenv.yaml`: fix the paths to files that exist; LoRA rank = whatever the training config uses; delete the `results:` block until Phase 4 fills it from `results/`.
  6. Delete `__pycache__/` and `*.egg-info/` from disk (already ignored); decide on `train_grpo_prod.py` (untracked) in P5-1.
- **Done when:** `pip install -e .` in a fresh venv works; `python -c "from conflictbench import ConflictBenchEnv"` works; the old `python conflict_bench.py` still works.

---

### PHASE 2 — Make the reward impossible to cheat (6–8 h, no GPU)

**The target:** a reward where *doing nothing* or *listing everything* scores near zero, where you can't get high marks without getting the answer right, and where every part measures something different.

#### P2-1 · Fix "no contradictions" (F-01)
- **File:** `verifier.py::rubric_no_contradictions`.
- **New rule:** only give credit for being consistent when the plan actually *covers* the conflict topics.
  - For each ground-truth conflict pair, look at which of its two IDs are in the plan: **exactly one** → consistent (1); **both** → contradiction (0); **neither** → not handled (0).
  - Score = mean over the ground-truth conflicts. Empty plan → 0.
- **Why:** this gives credit for "made a consistent choice on each clash" and nothing for skipping.

#### P2-2 · Score the overridden list and require a clean split (F-06)
- **Rule:** the plan and the overridden list must not share IDs, and every ID must be a real instruction.
  - `followed_f1` = F1(plan, GT followed). `overridden_f1` = F1(overridden, GT overridden).
  - `correct_final_state` = `(followed_f1 + overridden_f1) / 2`.
  - **Penalties:** any ID in both lists → multiply this rubric by `(1 - overlap / n_instructions)`; any made-up ID (not in the scenario) counts as a false positive in both.
- **Why:** now "override everything" and "follow everything" both lose about half of this rubric.

#### P2-3 · Make conflict identification strict about direction
- **Rule:** pair F1 × resolution accuracy (multiply, don't add with 60/40), as the docs already claim. Optional small extra: +10% of this rubric for the correct `conflict_type`.
- **Edge case:** a pair the model lists twice counts once; the resolution must be one of the two IDs in the pair or it counts as wrong.

#### P2-4 · Turn format into a gate, not free points (F-02)
- **Rule:** delete `format_compliance` as a weighted 10%. Instead: if `parsed_ok` is false, composite = 0. If `schema_errors` is non-empty, multiply the composite by 0.9 (a small, bounded penalty). Report `format_ok` separately in `ScoreBreakdown` for logging.
- **Why:** format is a requirement, not an achievement. A small model learns it anyway because nothing else scores without it.

#### P2-5 · Remove the duplicate efficiency rubric (C.2)
- Delete `rubric_efficiency` from the composite (keep the function for logging if you like). Its job is now done by P2-2's F1 plus the split penalty.

#### P2-6 · New weights and one place for them
- Composite = `0.45 × correct_final_state + 0.30 × conflict_identification + 0.25 × no_contradictions`, then the format gate.
- Keep `RUBRIC_WEIGHTS` in `verifier.py` as the **only** copy. Training scripts, `openenv.yaml`, app and docs must import or generate from it (P5-1, P6). No more copies like `RUBRIC_WEIGHTS_V2`.

#### P2-7 · Turn the hack tests on
- **File:** `tests/test_reward_hacks.py`. Remove the `xfail` marks. Required thresholds (mean over the fixed set):

| Strategy | Must be |
|---|---|
| empty | ≤ 0.05 |
| follow_all | ≤ 0.35 |
| override_all | ≤ 0.25 (≤ 0.35 already passes at 0.34, the empty-answer score) |
| random_half | ≤ 0.35 |
| hallucinated IDs only | ≤ 0.05 |
| ground_truth | = 1.0 |
| GT with one conflict flipped (diff 1) | between 0.4 and 0.85 (partial credit still works) |

- Also add a **property test**: for 500 random scenarios, a more correct answer (fewer wrong IDs) never scores lower (monotonicity), using random edits to the GT answer.
- **Done when:** all of these pass. Save the new baseline table to `results/baselines_v2.json` and note in the PR description the old → new numbers.

> Note: after Phase 2, `oracle_lookup` will still score 1.0. That's expected: the *scoring* is now fair, but the *task* is still a lookup. Phase 3 fixes the task.

---

### PHASE 3 — Make the task actually hard (12–20 h, no GPU)

**The target:** a simple script without reasoning should score clearly below a good model. Write this as a test (P3-7).

Implement each item as a **difficulty knob** in a `GeneratorConfig` dataclass, so old behaviour stays available as `legacy=True` for comparison.

#### P3-1 · Balance who wins (F-03)
- Allow both sides of each topic to come from any tier (redefine `sources` per template as tier ranges, not fixed names). Then **choose the winner side first** (50/50 restrictive vs permissive) and sample sources to match.
- **Done when:** across 5000 conflicts the restrictive side wins 45–55%, and `keyword_restrictive` drops to about chance on conflict resolution.

#### P3-2 · Distractors: same topic, no conflict (F-39)
- Add "compatible" instruction pairs on the same `action_key` (e.g. two freeze instructions that agree, or one with a narrower scope such as a different department). Ground truth: both followed, no conflict.
- Add an `action_scope` field (department/team/system). Two instructions conflict only if **key matches, values differ, and scopes overlap**.
- **Done when:** "every repeated topic is a conflict" drops below 0.5 F1 on conflict identification.

#### P3-3 · Hidden-hierarchy mode (makes the README's original idea true)
- `GeneratorConfig.show_hierarchy: bool`. When false, the prompt shows only names and job titles. Titles vary: "General Counsel", "Head of People", "Staff Engineer", "SVP Ops", from a title → tier table with synonyms. The model must know or learn that "General Counsel" outranks "SVP Ops".
- Keep a **held-out** set of titles that never appear in training (tests generalisation).

#### P3-4 · Richer rules than a single ranking
Add these one at a time, each with its own flag and tests:
1. **Scoped authority:** Legal wins on compliance topics, the CFO wins on budget topics even against the CEO's delegate, etc. Express it as a small rule table per `action_key`.
2. **Time/supersession:** two instructions from the *same* tier: the later dated one wins ("Effective 12 Mar…"). Needs dates in the text.
3. **Delegation:** "The COO delegates vendor approvals to VP Operations this quarter" raises that VP to tier 2 for one topic.
4. **Conditional instructions** (uses the unused `condition` field): "If the audit is still open, freeze hiring". The context says whether the condition holds.
5. **Chains (multi-hop):** A overrides B, and B was the condition for C, so C is cancelled too.
- Each rule needs a **deterministic resolver** in one place (`resolve(scenario) -> ground truth`), so ground truth always comes from the rules and never from the template order. Tests: every generated scenario is solved by `resolve()` exactly as stored.

#### P3-5 · More variety and scale
- Expand from 10 to 25+ topics and from 3 to 8+ phrasings each; or paraphrase with an LLM **offline**, store the results as fixtures, and check every paraphrase by having the resolver re-label it.
- Difficulty levels: 4 (2 clashes, ~8 instructions) → 12 (8+ clashes, 30–40 instructions, distractors, chains). Real token budget: measure with the tokenizer (P5-1), not characters.

#### P3-6 · Train / validation / test splits (F-27)
- Split **templates, titles, company contexts and topics** into train/val/test pools. Generate `data/test_v1.jsonl` (e.g. 600 scenarios, fixed seed) **once**, commit it, and never train on it. Store a SHA-256 hash of the file in `results/` so any change shows up.

#### P3-7 · The "not a lookup" test
- `tests/test_difficulty.py`: on the test split, `keyword_restrictive` ≤ 0.35, `oracle_lookup_v1` (the old solver) ≤ 0.6, `resolve()` = 1.0.
- **Done when:** passes. This is the proof that the task needs reasoning.

---

### PHASE 4 — Honest evaluation and baselines (4–8 h; GPU optional)

#### P4-1 · One eval harness
- **Files:** new `scripts/evaluate.py`.
- **Inputs:** `--model` (HF id or local path, optional `--adapter`), `--data data/test_v1.jsonl`, `--limit`, `--greedy` (default), `--samples k` (for pass@k).
- **Behaviour:** build messages with `prompts.build_messages()` (same as training), apply the chat template, generate, score, and save **every** completion plus its score to `results/<run_name>/completions.jsonl`, with a summary in `results/<run_name>/summary.json`: mean, 95% bootstrap CI, per-difficulty, per-rubric, parse-failure rate.
- Also support an `--api` mode (OpenAI-compatible endpoint, Groq, etc.) so frontier models can be scored. Keep the key in an env var, never in code or logs.

#### P4-2 · The baseline table that goes into the README
Run and save, all on the same test file:
1. Cheat strategies (from P0-2).
2. Rule solvers (`keyword_restrictive`, `oracle_lookup_v1`, `resolve()`).
3. Qwen2.5-3B-Instruct base: greedy, correct chat template (free Colab/Kaggle T4 is enough for 600 short prompts).
4. Optional: 1–2 bigger open models and 1 frontier API model, to show the task isn't trivial for them either.
- **Done when:** `results/leaderboard.md` is generated by a script (`scripts/make_leaderboard.py`) from the summaries, not typed by hand.

#### P4-3 · Re-score the old adapter (to replace unverifiable claims)
- Evaluate `Harsh-9209/conflictbench-qwen2.5-3b-grpo-lora` with the **old verifier** (git tag `v1-legacy`) **and** the new one, on the legacy generator set and on the new test set. Write the results to `results/legacy_adapter/`. These four numbers replace "0.50 / 257%" everywhere. If the adapter is worse than `follow_all`, say so plainly. It's a finding, not a failure.

---

### PHASE 5 — One clean, reproducible training pipeline (8–12 h + 1 GPU run)

#### P5-1 · One training script, one config file
- **Files:** `conflictbench/train/train_grpo.py`, `configs/grpo_t4.yaml`, `configs/grpo_a10g.yaml`, `conflictbench/prompts.py`. Delete `train_grpo_prod.py` (untracked) and make `training_space/train_script.py` call the shared script with a config instead of copying it.
- **Steps:**
  1. Everything comes from YAML: model, LoRA r/alpha, dataset sizes, difficulty mix, lengths, lr, beta, generations, seed, output dir, `hub_repo` (default **null**), `report_to` (default `none`).
  2. Dataset: generate from the **train split only** (P3-6); filter by **token** length with the real tokenizer applied to the full chat-templated prompt; log how many were dropped per difficulty, and fail if any requested difficulty ends up with < 50% of its target.
  3. Reward function: build the scenario from stored JSON and call `verifier.score()` directly. No re-weighting, no difficulty bonus. On exception: log at WARNING with the first 200 chars of the completion, count it in a metric, re-raise after 10 consecutive failures.
  4. Log each rubric as its own reward function (TRL supports a list of `reward_funcs`) so W&B/plots show which part is learning; weight them with `reward_weights` matching `RUBRIC_WEIGHTS`.
  5. Use `processing_class=tokenizer` (the newer TRL name) and pin the TRL version that matches.
  6. `diagnose_tokens.py` → `scripts/diagnose_tokens.py`, reading the same config.
- **Done when:** a 10-step dry run (`--max_steps 10`, tiny model like `Qwen2.5-0.5B-Instruct`) completes on Colab/Kaggle and writes logs.

#### P5-2 · Resume, checkpoint choice, and eval on held-out data
- `trainer.train(resume_from_checkpoint=True if checkpoints exist else None)`.
- Eval during training uses the **val split** (not test). Pick the best checkpoint by val reward, then run `scripts/evaluate.py` on the **test split** once, at the end.
- Write `results/<run>/train_config.yaml`, `pip_freeze.txt`, git commit hash, and the final summary.

#### P5-3 · Reproducible Space builds
- Pin `unsloth`/`unsloth_zoo` to released versions (not git HEAD) and clone the repo at a **tag**, not HEAD. Keep the password lock from P0-3.

#### P5-4 · The real experiment (one GPU run)
- Train Qwen2.5-3B on the new task (Phases 2–3), evaluate on test, compare with: base model, best rule solver, cheat baselines. Save everything under `results/`. **That table** is your headline result, whatever it shows.
- Tip: if GRPO doesn't move on the hard levels, add a curriculum (easy levels first, more hard levels as val reward rises). That's a real, explainable improvement.

---

### PHASE 6 — Rewrite the docs so every sentence is true (3–5 h)

#### P6-1 · README
- Remove or correct: "never stated in the prompt", "ungameable/non-gameable", "first dedicated RL environment", "8–28 instructions", "100+ templates", "16 sources", "4000-char guard", "long-horizon" (unless P7-1 is done), the made-up before/after example (replace it with **real** completions copied from `results/.../completions.jsonl`), and all old numbers.
- Add: a "What we found and fixed" section (reward hacking found, proof, fix, before/after cheat-baseline table). It's honest and it's a strong story.
- The results table is generated: `scripts/make_leaderboard.py` writes `results/leaderboard.md`; README links to it or includes it between marker comments that a script updates.

#### P6-2 · Docs
- `docs/REWARD_DESIGN.md`: rewrite from the code (formulas, JSON schema, the format gate, why each rubric exists, the hack table). Add a test that compares the weights in the doc to `RUBRIC_WEIGHTS` (parse the markdown table).
- `docs/TRAINING_LOGS.md`: keep the bug history (it's genuinely useful), but add a header saying the metric tables come from code not in this repo and can't be reproduced; move them to an "Archive (unverified)" section.
- `docs/TRAINING_GUIDE.md`: fix resume/best-checkpoint claims, point to the configs.
- `PRD.md`, `blog.md`: update the claims or add a dated "Update, Dec 2026" note at the top pointing to the new results.

#### P6-3 · Small fixes
- Rename `docs/architecture.md` → `docs/ARCHITECTURE.md` in git (`git mv` through a temporary name on Windows, because the rename only changes letter case). Fix the `doc/` typo in the PRD. Make the hierarchy description identical everywhere (copy it from `SOURCE_PRIORITY`).
- **Done when:** a link checker (`scripts/check_links.py`: every relative markdown link resolves, case-sensitive) passes.

---

### PHASE 7 — Future scope (pick based on your goal)

Ordered by value for effort.

1. **P7-1 · Make it truly multi-step ("long-horizon").** Instructions arrive over several turns (emails/tickets). The agent calls tools (`approve(id)`, `reject(id)`, `ask_clarification`) and the environment updates state. Reward: the final state of a small simulated company (hires, deployments, budgets) compared with the correct state. This is what OpenEnv is built for, and it makes the "long-horizon" claim true.
2. **P7-2 · Prompt-injection / spoofed authority.** A low-level instruction claims "Per Legal: proceed with hiring". The model must check the real source field, not the claim in the text. This links the project to AI-safety work, which is a strong angle for interviews and papers.
3. **P7-3 · Public dataset + leaderboard.** Publish `test_v1` on the HF Hub as a dataset with a scorer; a Space that scores uploaded JSONL; seed it with open and API models.
4. **P7-4 · Custom org charts.** Load the hierarchy from YAML (healthcare, government, a university) so teams can use it for their own rules. This also delivers the PRD promise that the "hierarchy is configurable".
5. **P7-5 · Ask when it's ambiguous.** Some clashes are true ties. The correct action is "escalate". Reward the model for escalating only on those.
6. **P7-6 · Write-up.** A 4-page workshop paper: environment design, the reward hacks you found (empty answer 0.34, keyword shortcut 0.95, lookup 1.0) and how you closed them, then baseline vs trained results. The reward-hacking analysis is the most novel part, so don't hide it.
7. **P7-7 · CI.** GitHub Actions: pytest + link check + hack thresholds on every PR (CPU only, < 2 min).

---

## Part E — Checklist summary (copy into issues)

### Status as of 1 Nov 2026

Done and tested on CPU (`python -m pytest -q`): Phases 0–2, Phase 3, the CPU parts of Phase 4 (eval harness, cheat and rule-based baselines on `data/test_v1.jsonl`, generated leaderboard), the code of Phase 5, and Phase 6 (docs rewritten from the code, link checker, doc tests). Not done: anything that needs a GPU or an API key (the base-model and frontier rows of P4-2, P4-3 re-scoring of the old adapter, the P5-1 dry run on Colab/Kaggle, P5-4 real training run), the package move of P1-8 (`openenv.yaml` paths are fixed; the `conflictbench/` package folder is not), and the full scale target of P3-5 (17 topics × 5 phrasings instead of 25+ × 8+). `[~]` = partly done.

```
PHASE 0  [x] P0-1 tests  [x] P0-2 cheat baselines  [x] P0-3 lock Space (code; Space settings still manual)  [x] P0-4 status banner + LICENSE
PHASE 1  [x] P1-1 parser  [x] P1-2 seeded generator  [x] P1-3 type labels  [x] P1-4 server
         [x] P1-5 env cleanup  [x] P1-6 demo app  [x] P1-7 inference.py  [~] P1-8 packaging (yaml paths only)
PHASE 2  [x] P2-1 contradictions  [x] P2-2 overridden + split  [x] P2-3 strict direction
         [x] P2-4 format gate  [x] P2-5 drop efficiency  [x] P2-6 weights  [x] P2-7 hack tests ON
PHASE 3  [x] P3-1 balanced winners  [x] P3-2 distractors  [x] P3-3 hidden hierarchy
         [x] P3-4 rules (scope/time/delegation/conditional/chains)  [~] P3-5 variety  [x] P3-6 splits  [x] P3-7 not-a-lookup test
PHASE 4  [x] P4-1 eval harness  [~] P4-2 baseline table (no model rows yet)  [ ] P4-3 re-score old adapter
PHASE 5  [x] P5-1 one script + YAML (GPU dry run pending)  [x] P5-2 resume/held-out  [x] P5-3 pinned Space  [ ] P5-4 real run
PHASE 6  [x] P6-1 README  [x] P6-2 docs  [x] P6-3 links/case
PHASE 7  [ ] pick 1–2 from the list
```

## Part F — What you can honestly say, at each stage

| Stage reached | Safe résumé/interview line |
|---|---|
| Today | "Built an RL environment + GRPO pipeline for instruction-conflict resolution (hackathon prototype)." No numbers. |
| After Phase 2 + P4-2 | "…found and fixed reward-hacking in my own environment (an empty answer scored 0.34; now ≤ 0.05), with regression tests." |
| After Phase 3 | "…redesigned the task so rule-based shortcuts score ≤ 0.35 while correct reasoning scores 1.0." |
| After P5-4 | "…GRPO improved Qwen2.5-3B from X to Y on a held-out test set (95% CI …), versus a rule-based baseline of Z." Use real X/Y/Z from `results/`. |
