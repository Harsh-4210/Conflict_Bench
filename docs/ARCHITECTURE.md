# ConflictBench — Architecture

How the parts of the repo fit together, as of November 2026. File names are links to the code, which is the reference when this page and the code disagree.

## Data flow

```
 GeneratorConfig ──► ScenarioGenerator ──► Scenario (structured fields + prompt text)
 (generator.py)       (generator.py)            │
                             │                  │  resolver.resolve(scenario)
                             └──── checks ◄─────┤  -> followed / overridden / conflicts
                                                │     (the only source of ground truth)
                                                ▼
                                      prompts.build_messages(scenario)
                                      = [SYSTEM_PROMPT, scenario.prompt]
                                                │
         ┌──────────────────────────────┬───────┴───────────────┬──────────────────────────┐
         ▼                              ▼                       ▼                          ▼
 ConflictBenchEnv               OpenEnv server            training/train_grpo.py     scripts/evaluate.py
 (conflict_bench.py)            (server/)                 (GRPO, train + val split)  (strategies, local models,
 reset() / step() / state       /reset /step /ws ...                                   API models; any split/file)
         │                              │                       │                          │
         └──────────────────────────────┴──────► verifier.score(completion, scenario) ◄────┘
                                                 = composite + per-rubric breakdown
                                                                │
                                                                ▼
                                      results/<run>/completions.jsonl + summary.json
                                                                │
                                                                ▼
                                       scripts/make_leaderboard.py -> results/leaderboard.md
```

## Components

### 1. Scenario generator — [`generator.py`](../generator.py)

- `Instruction`, `ConflictPair` and `Scenario` dataclasses. Besides the text, every instruction carries structured fields: `source` (office), `source_priority` (tier 1–6), `action_key` (topic), `action_value` (direction), `action_scope` (`all` or one department/system), `polarity`, `author`, `title`, `effective_date`, `condition`, `depends_on`. A scenario also holds `facts`, `authority_rules`, `delegations`, `split` and `generator_version`.
- `GeneratorConfig` (frozen dataclass): one switch per rule (`scoped_authority`, `delegation`, `time_supersession`, `conditionals`, `chains`, `distractors`, `decoy_notices`), the probabilities that control them, `restrictive_win_rate` (default 0.5), `show_hierarchy` (default off = hidden hierarchy), `split` (`train`/`val`/`test`, default `train`) and `legacy`.
- `DIFFICULTY_LEVELS`: levels 1–5 with 2/4/6/8/10 conflicts. Out-of-range levels are clamped.
- `split_pool()`: a fixed, non-random partition of each list of phrasings, job titles, companies, situations, sign-offs and follow-ups into train/val/test pools.
- The v2 builder records what it intended each instruction's outcome to be; after building, the generator calls `resolver.resolve()` and raises if the two disagree. Ground truth is therefore never taken from template order.
- `GeneratorConfig(legacy=True)` reproduces the original hackathon generator exactly (10 topic groups, printed hierarchy). A snapshot test pins its output.
- All randomness comes from a private `random.Random(seed)`; `generate(..., seed=s)` uses its own RNG for that call. The global `random` module is never touched, so concurrent environments do not interfere.

### Authority levels — `SOURCE_PRIORITY` in [`generator.py`](../generator.py)

```
1  Legal & Compliance, Regulatory Affairs
2  CEO Office, Chief Financial Officer, Chief Technology Officer, Chief Operating Officer
3  VP Engineering, VP Finance, VP Operations, VP Human Resources
4  Director of IT, Director of Finance
5  Engineering Manager, Finance Manager, HR Manager, IT Manager
6  Team Lead, Department Coordinator
```

18 offices on 6 tiers; lower number = more authority. In the default (hidden-hierarchy) prompt these office names are not shown: each person appears with one of five job titles for their office (`OFFICE_TITLES`), for example "Senior Finance Director" for Director of Finance.

### 2. Resolver — [`resolver.py`](../resolver.py)

`resolve(scenario)` reads only the structured fields, never the text:

1. Instructions whose `condition` is not true in `facts` are void (overridden, in no conflict).
2. Two in-force instructions conflict when they share `action_key`, differ in `action_value`, and their scopes overlap.
3. Rank on a topic: a standing rule naming the office → 0; otherwise the tier, lowered to the delegator's tier if the office holds a delegation for that topic.
4. Equal rank → the later `effective_date` wins.
5. Instructions that depend on an overridden instruction are cancelled; repeated until nothing changes.

It works on legacy scenarios unchanged.

### 3. Prompts — [`prompts.py`](../prompts.py)

`SYSTEM_PROMPT` gives the task and the JSON schema and contains no hierarchy. `build_messages(scenario)` returns `[system, user]` messages. The demo app, `inference.py`, the evaluation harness and the training pipeline all use it, so every component sees the same text.

### 4. Environment — [`conflict_bench.py`](../conflict_bench.py) and [`server/`](../server/)

- `ConflictBenchEnv`: `reset(seed=None)` returns a dict with the prompt, difficulty, domain, number of conflicts and scenario ID; `step(text)` returns `(observation, reward, done=True, info)` where `info` has the score breakdown and the correct answer. One step per episode; a second step raises. Optional automatic difficulty: level 1 for 200 episodes, level 2 until episode 500, then level 3.
- `server/`: an OpenEnv (`openenv-core` 0.2.3) FastAPI app. `ConflictBenchEnvironment` wraps one `ConflictBenchEnv` per session; `ConflictBenchAction` / `ConflictBenchObservation` are the pydantic models. HTTP `/reset` and `/step` are stateless (a fresh environment per request); the `/ws` WebSocket keeps a session for real episodes. The correct answer is only included if the environment is constructed with `reveal_ground_truth=True`, which the served app does not do.

### 5. Verifier — [`verifier.py`](../verifier.py)

`score(completion, scenario) -> ScoreBreakdown`: a parser that never raises, three weighted rubrics and a format gate. Full spec: [REWARD_DESIGN.md](REWARD_DESIGN.md). `RUBRIC_WEIGHTS` is the single copy of the weights.

### 6. Evaluation — [`scripts/`](../scripts/)

- `baselines.py`: the cheat and rule-based strategies (`STRATEGIES`) and a quick aggregate run.
- `evaluate.py` + `backends.py`: one harness for strategies, local Hugging Face models (optional LoRA adapter, chat template, greedy by default, prompts that are too long are skipped and counted, never truncated) and OpenAI-compatible APIs. Writes `results/<run>/completions.jsonl` and `summary.json` (mean, seeded bootstrap 95% CI, per difficulty, per rubric, parse-failure rate, data SHA-256, git commit).
- `make_leaderboard.py`: builds `results/leaderboard.md` from every `summary.json`, one table per data file hash.
- `check_links.py`: checks every relative markdown link (case-sensitive).

### 7. Training — [`training/`](../training/) and [`configs/`](../configs/)

`python -m training.train_grpo --config configs/<name>.yaml`. Datasets come from the train split (training) and the val split (in-training evaluation), filtered by the token length of the full chat-templated prompt. The reward is `verifier.score(...).composite`; each rubric is logged as its own zero-weight reward function. Unsloth + TRL `GRPOTrainer`, resume from the latest checkpoint, best checkpoint by validation reward, run artefacts in `results/<run_name>/`. See [TRAINING_GUIDE.md](TRAINING_GUIDE.md).

### 8. Apps

- [`app.py`](../app.py): Gradio demo. Generates a scenario (default generator, levels 1–3), runs the base model (adapter disabled) and the LoRA adapter on the same network, scores both, and can reveal the correct answer. Models load lazily on the first click.
- [`training_space/`](../training_space/): a separate, password-locked training Space. Its Dockerfile clones this repo at a fixed tag or commit and its `train_script.py` calls `training.train_grpo.train` with `configs/grpo_a10g.yaml`.

## Design choices

- **Programmatic generation.** Scenarios come from templates and structured fields, so every answer can be derived exactly and checked by a test.
- **One source of ground truth.** The generator builds, the resolver decides, and the generator refuses to emit a scenario on which the two disagree.
- **Single-step episodes.** One prompt, one JSON answer. This keeps scoring exact; it also means the environment is not long-horizon.
- **Fixed held-out test set.** `data/test_v1.jsonl` is generated once from the test split and committed with its SHA-256; training never reads the test split.
