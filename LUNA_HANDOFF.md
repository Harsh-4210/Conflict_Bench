# ConflictBench: complete handoff for Luna

- **Prepared:** 6 November 2026
- **Repository:** [Harsh-4210/Conflict_Bench](https://github.com/Harsh-4210/Conflict_Bench)
- **Working branch:** `feat/future-scope`
- **Starting implementation commit:** `beed2f0ffc8d39d8eb5a8d435824251c1b23b78b`
- **Local checkout:** `C:\Users\Harsh Jain\Desktop\PROJECTS\conflictbench`

## 1. Your assignment and the user's instructions

Luna, continue this existing project. Do not rebuild components that already exist. Finish the remaining engineering, experiments, documentation and release work listed here, maintaining an explicit status for every item.

The immediate milestone is an honest, reproducible comparison of the base model and a model trained on the redesigned single-step task. The remaining backlog also includes making the multi-step task require timely action, publishing the benchmark, completing packaging and task variety, and implementing the outstanding research extensions.

**Harsh will execute the tests and GPU commands.** His latest instruction was: do not run the tests yourself; provide the commands, and push the changes to GitHub. That instruction remains in force until he explicitly changes it.

- You may inspect code, implement changes, write tests, review diffs, prepare commands, and interpret results Harsh supplies.
- Do not execute pytest, test scripts, model evaluation, tokenizer dry runs, training, or a training-image build on his behalf without new authorization. Do not start downloads or restart services merely to check something.
- Installing Superpowers is requested; install or enable it in the actual host as described below. Preserve an existing installation.
- Use the current feature branch or a suitable existing checkout. Preserve `.claude/`, other agents' worktrees, unrelated edits, `.env`, datasets, historical results and adapters.
- Commit and push completed code/documentation changes to the verified GitHub `origin`. Do not push to the `hf` remote, merge to `main`, force-push, publish a Space/dataset/model, or spend money without the relevant authorization.
- Do not claim a test passed, a GPU run completed, a dependency stack works, or a model improved without the corresponding evidence.
- A blocked item must have a reason and a concrete next action. It must remain in the ledger; it must not disappear from the scope.

## 2. Install and use Superpowers before implementation

Official source: [obra/superpowers](https://github.com/obra/superpowers). Follow its current installation instructions for the host running you. **Luna is a model/agent name, not enough information to choose a host-specific installer. Do not invent a Luna installation command.**

1. Check whether the host already exposes `using-superpowers` and the other Superpowers skills. This checkout's previous agent used an installed copy; do not overwrite or duplicate it.
2. If running in Codex App, use **Plugins → Superpowers → Install**. In Codex CLI, open `/plugins`, search for Superpowers, and install the plugin.
3. If running in Claude Code, use its official marketplace command:

   ```text
   /plugin install superpowers@claude-plugins-official
   ```

4. For another host, use that host's section in the official repository README. Reopen/restart the session if the host requires it, then confirm that the skills are discoverable.
5. Read and use `using-superpowers`; apply brainstorming/design and writing-plans for new behavior, systematic-debugging for failures, the implementation/testing workflow, code review, and verification-before-completion as appropriate.
6. **The user's execution restriction overrides a skill's instruction to run tests automatically.** Write the tests and verification commands; label verification pending until Harsh supplies results. Do not fabricate a red/green test cycle. Explain this execution split in the task record.

Installation directions above were checked against the official README on 6 November 2026. Recheck that page when installing rather than following an obsolete raw installation URL.

## 3. Read these files in order

1. [README.md](README.md): current task, reward, historical audit and baseline numbers.
2. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): existing single-step data flow.
3. [docs/REWARD_DESIGN.md](docs/REWARD_DESIGN.md): exact scorer contract.
4. [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md), [requirements-train.txt](requirements-train.txt), [configs](configs): training/configuration contract.
5. [results/README.md](results/README.md), [results/leaderboard.md](results/leaderboard.md): evaluation commands and measured results.
6. [docs/MULTISTEP.md](docs/MULTISTEP.md), [multistep](multistep), [scripts/evaluate_multistep.py](scripts/evaluate_multistep.py).
7. [docs/BENCHMARK.md](docs/BENCHMARK.md), [scripts/export_hf_dataset.py](scripts/export_hf_dataset.py), [leaderboard_space](leaderboard_space).
8. [FIX_BLUEPRINT.md](FIX_BLUEPRINT.md), [PRD.md](PRD.md), [blog.md](blog.md), [docs/TRAINING_LOGS.md](docs/TRAINING_LOGS.md).
9. [Consolidation design](docs/superpowers/specs/2026-11-06-training-entrypoint-design.md) and [execution record](docs/superpowers/plans/2026-11-06-training-entrypoint.md).

The blueprint contains historical instructions, incomplete checklists and old names. Use it as an audit trail, not as proof that an item is still missing or already finished. Code and saved evidence decide current status; this handoff explicitly reconciles the known differences.

## 4. What exists already: do not redo it

| Component | Current implementation |
|---|---|
| Task generation | `generator.py`: deterministic local RNG, legacy mode, five difficulty levels, hidden hierarchy, scoped authority, delegation, dates, conditions, chains, distractors and balanced winners. |
| Ground truth | `resolver.py`: deterministic resolution of structured scenarios. |
| Shared prompt | `prompts.py`: used by training, demo, inference and single-step evaluation. |
| Scoring | `verifier.py`: parser hardening, followed/overridden scoring, strict conflict resolution, contradiction coverage and format gate. No LLM judge. |
| Single-step environment | `conflict_bench.py`, `server/`: Python and OpenEnv wrappers. Full episodes use WebSocket sessions; plain HTTP reset/step are stateless in the pinned OpenEnv version. |
| Evaluation | `scripts/evaluate.py`, `scripts/backends.py`: scripted, HF model/adapter and OpenAI-compatible API backends; raw completions, summaries, bootstrap intervals and leaderboard generation. |
| Training | `training/train_grpo.py`, `training/config.py`: one YAML-driven pipeline, explicit train/val splits, real-tokenizer filtering, shared reward, validation, resume and best-checkpoint selection. |
| Compatibility commands | Root `train_grpo.py` and `diagnose_tokens.py` are shims. `train_grpo_prod.py` now delegates to the maintained pipeline; its separate broken trainer is gone. |
| New local preset | `configs/grpo_4060.yaml`: experimental 120 train / 20 val, level 1 only, two epochs, batch 1 × accumulation 4, four generations, 3072 prompt + 1024 completion tokens. **8 GB fit is not established.** |
| Config-only mode | `--check-config` validates and prints resolved JSON without model imports, downloads or training. Mutually exclusive with `--dry-run`. |
| Local Docker source | `docker/training/Dockerfile` and its specific `.dockerignore`: Linux/Python/CUDA dependencies; only requirements enter the build context. Mount the checkout at runtime. |
| Multi-step environment | `multistep/`: messages and status updates arrive over turns; apply/reject/revert/escalate/wait/finish tools, state scoring, OpenEnv server and scripted baselines. |
| Publication tooling | Dataset export separates prompt-only and full-answer files; `leaderboard_space/` validates/scorers uploaded predictions. Tooling exists; live publication was not verified. |
| CPU CI | `.github/workflows/tests.yml`: Python 3.11/3.12, CPU suite and fixed-set checksum. Triggers on PRs and pushes to `main`, not every feature-branch push. |

### Evidence at handoff

- Before consolidation: 538 CPU tests passed and 1 skipped. After consolidation: **549 passed, 1 skipped**, with one dependency deprecation warning. These are historical results, not a result for future edits.
- Initial sandboxed tests had temporary-directory permission errors. The same suite passed outside the sandbox; do not rewrite working product code to fix sandbox ACLs.
- The previous independent code reviewer found no Critical, Important or Minor defects in the consolidation increment. Docker runtime execution was outside that review's proven scope.
- The 600-scenario `data/test_v1.jsonl` checksum was verified as `e17169bb9de6bfb760ff23e0551ec170396ce11bd39e8db3ef50090497217b9a`.
- GitHub branch `feat/future-scope` was verified to point at `beed2f0`. The authenticated account was `Harsh-4210`, with ADMIN access to the PUBLIC repository.
- Local hardware: RTX 4060 Laptop GPU, 8188 MiB VRAM. Docker Desktop's Linux engine successfully exposed it to an Ubuntu container.
- **Docker image build and separate Windows tokenizer installation were interrupted at Harsh's request.** Do not assume either completed. No training container or GPU smoke run was completed by this agent.
- No saved real-model result on the redesigned task, no redesigned-task trained adapter, and no re-scored historical adapter were established during this work. Ask Harsh for any results he generates after this handoff.

## 5. Tracking and execution order

Maintain `docs/LUNA_PROGRESS.md` when you begin work. Give each task below one of: `PENDING`, `IN_PROGRESS`, `AWAITING_HARSH_RUN`, `BLOCKED`, `DONE`.

Each record must contain: task ID, implementation commit, files changed, exact command handed to Harsh, supplied result/artifact path, outstanding blocker, and next action. A code change awaiting execution is not fully verified. Keep old results immutable; use new run names for reruns.

Recommended dependency order:

1. R0–R1: reconcile Harsh's results, stabilize runtime/configuration and collect baseline measurements.
2. R2–R3: re-score the historical adapter and complete the new single-step training experiment.
3. R4: fix multi-step deferral and add real model-agent evaluation.
4. R5–R6: packaging and expanded/versioned task coverage. Do not invalidate the original experiment while doing this.
5. R7–R9: spoofing, configurable org charts and meaningful escalation, each with its own design and versioned evaluation.
6. R10–R13: publication, truthful documentation/write-up, CI/integration and release closure. Prepare these in parallel where they do not depend on missing results.

Do not silently omit later tasks because the first GPU run takes time. Implement independent work while waiting for Harsh's evidence. If he changes priorities, retain the displaced tasks and record the explicit decision.

## 6. R0 — Reconcile current state and Harsh's manual runs

- Inspect `git status`, branch/upstream, recent commits, attached worktrees, local run directories and any logs Harsh supplies. Do not reset or overwrite his work.
- Ask for the build/dry-run/smoke output only if it has not already been supplied. Do not rerun those commands yourself.
- Preserve the fixed test file and checksum. Identify result provenance: task version, scorer version, dataset hash, model/adapter, decoding settings, package versions, code commit and hardware.
- Record which user-provided files are partial logs versus completed runs. A successful config check is not a tokenizer check; a tokenizer check is not GPU validation.

**Complete when:** the progress ledger reflects the actual checkout and supplied run status, with no duplicate work or invented measurements.

## 7. R1 — Stabilize the local Docker runtime and training plumbing

**Relevant files:** `docker/training/`, `requirements-train.txt`, `training/train_grpo.py`, `training/config.py`, `configs/grpo_*.yaml`, `tests/test_training_*.py`, `docs/TRAINING_GUIDE.md`.

- Diagnose actual build/import/runtime errors from Harsh's logs before changing pins. Read the installed libraries' real signatures where relevant. Do not randomly upgrade the stack.
- Validate the tokenizer path on smoke, 4060 and the larger difficulty-mix configs. Inspect per-difficulty generated/kept/dropped counts, real prompt-token lengths and whether completion budgets can fit complete answers.
- Preserve the minimum-retention guard, split isolation, shared system/user prompt and exact composite reward. Never restore a character-count filter.
- Verify from user-supplied smoke artifacts: model load, generated completions, scoring, nontrivial reward signal, validation at steps 5/10, checkpoint saves, best-model reload, final adapter save and final validation. Investigate a flat zero reward or truncated JSON before a full run.
- Separately validate resume after interruption and selection by `eval_reward`. Do not assume configuration flags prove the runtime behavior.
- Ensure the runtime's `pip check` succeeds. Refresh pins from a successful environment snapshot and align the training Space Dockerfile where appropriate. Record an image digest/ID and sufficient dependency information to recreate the run.
- If 8 GB is insufficient, make the smallest measured adjustment: batch/group-compatible accumulation, prompt/completion budgets, LoRA size or tiny model for plumbing. Preserve at least two generations for GRPO. Do not claim the 3B experiment ran if only the 0.5B smoke ran.
- Keep CPU test dependencies separate from CUDA dependencies. Do not install the training stack into `.venv`.

**Complete when:** Harsh supplies a successful 10-step smoke run, complete artifacts, validated resume/checkpoint behavior, and a recorded working dependency/runtime setup. Any 3B memory limitation is explicit.

### Commands for Harsh: local Docker, PowerShell

Run from the repository root, with Docker Desktop's Linux engine running:

```powershell
docker build -f docker/training/Dockerfile -t conflictbench-training:local .

$dockerArgs = @(
    "run", "--rm", "--gpus", "all", "--shm-size", "2g",
    "--mount", "type=bind,source=$((Get-Location).Path),target=/workspace",
    "--mount", "type=volume,source=conflictbench-hf-cache,target=/cache/huggingface",
    "conflictbench-training:local"
)

docker @dockerArgs python -m training.train_grpo --config configs/grpo_4060.yaml --check-config
docker @dockerArgs python -m training.train_grpo --config configs/grpo_smoke.yaml --dry-run
docker @dockerArgs python -m training.train_grpo --config configs/grpo_4060.yaml --dry-run
docker @dockerArgs python -m training.train_grpo --config configs/grpo_smoke.yaml
```

These commands are supplied for Harsh to run; their presence in this document is not execution evidence. The first build downloads several GB. The named volume preserves model downloads; bind-mounted output folders preserve checkpoints and logs.

Required run artifacts: resolved YAML, git commit, package freeze, environment metadata, dataset statistics, `log_history.json`, `final_metrics.json`, checkpoints and final adapter/tokenizer. See `docs/TRAINING_GUIDE.md` for exact locations.

## 8. R2 — Measure base models and re-score the historical adapter

**Blueprint:** P4-2 and P4-3. **Relevant files:** `scripts/evaluate.py`, `scripts/backends.py`, `scenario_io.py`, `results/README.md`, `results/leaderboard.md`.

### Current task baseline

- Evaluate `Qwen/Qwen2.5-3B-Instruct` on all 600 fixed scenarios, using the shared chat prompt and greedy decoding. Use a validation/smoke subset for debugging before the final test run; do not tune against test scores.
- Do not silently truncate prompts. Record skipped/failed cases, parse failures, completion truncation and the denominator. A high mean over a small retained subset is not a complete 600-scenario result.
- Record model revision, adapter revision and quantization/precision; keep base-versus-trained inference conditions comparable.
- The current HF evaluation loader is not a documented 4-bit inference path. If full-precision 3B evaluation exceeds 8 GB, implement and document an explicit compatible path or provide a supported alternative; do not invent nonexistent CLI switches or silently change the evaluated model.
- Add one larger open model and one OpenAI-compatible API model if access and budget are approved. Prepare the harness/commands regardless; an unavailable credential or budget stays an explicit blocker. Read `CONFLICTBENCH_API_KEY` from the environment only.

### Historical adapter: all four cells

Re-score `Harsh-9209/conflictbench-qwen2.5-3b-grpo-lora` using this matrix:

| | Legacy dataset | Redesigned test dataset |
|---|---|---|
| Legacy scorer | Required | Required |
| Current scorer | Required | Required |

- Identify and verify the historical revision (`88caba0` is documented as the candidate). Check whether `v1-legacy` exists; create a local isolated historical checkout/ref only when appropriate. Do not rewind the active checkout.
- Modern evaluation/serialization code may not run unchanged against old dataclasses or scorers. Build a small compatibility bridge if necessary and document precisely which prompt, model and scorer each cell used.
- For comparisons intended to isolate scorer changes, keep saved completions fixed and re-score them. Separate any prompt/task changes from scorer changes; do not attribute all differences to training.
- Preserve old reward-hack baseline tables and unverified training history. An adapter that fails, is unavailable, or performs poorly is a reportable outcome, not permission to substitute an invented score.

**Complete when:** full raw completions and metadata exist for the base-model run and each obtainable matrix cell, missing cells are explicitly blocked with evidence, and the leaderboard is generated from saved summaries.

### Commands for Harsh: established current-task evaluation interfaces

```powershell
docker @dockerArgs python scripts/evaluate.py --model Qwen/Qwen2.5-3B-Instruct --data data/test_v1.jsonl --run-name qwen3b_base_test_v1 --max-new-tokens 1024
docker @dockerArgs python scripts/evaluate.py --model Qwen/Qwen2.5-3B-Instruct --adapter Harsh-9209/conflictbench-qwen2.5-3b-grpo-lora --data data/test_v1.jsonl --run-name legacy_adapter_current_task --max-new-tokens 1024
docker @dockerArgs python scripts/make_leaderboard.py
```

Memory fit remains to be checked. The historical four-cell study needs the isolated compatibility setup above; these two commands alone do not complete it.

## 9. R3 — Run and report the redesigned single-step experiment

**Blueprint:** P5-4 and runtime portions of P5-1/P5-2/P5-3.

- Freeze the task/scorer/configuration used for the first experiment; do not expand the generator mid-run.
- Train the 3B model only after the tiny smoke pipeline is proven. Start with the experimental 4060 preset if it fits. It trains on level 1 only: explicitly limit any generalization claims.
- Add a documented level-1/2/3 experiment or curriculum when memory and evidence justify it. Avoid treating an L1-only run as fulfillment of the original mixed-difficulty experiment.
- Use train for optimization and val for hyperparameters/checkpoint selection. Evaluate the selected checkpoint on test at the end; keep checkpoint search away from the test set.
- Compare base, legacy adapter, new adapter and scripted baselines on the same data/scorer. Structured rule-based baselines use hidden fields; label them as privileged rather than ordinary language-model baselines.
- Report mean, 95% interval, per-difficulty, per-rubric, parse-failure and skipped-case rates. Include paired improvement uncertainty when claiming a gain; identical scenario IDs allow paired analysis.
- Inspect raw outputs for formatting-only improvements, output-length effects and shortcuts. Do not claim reasoning improvement solely from a higher composite.
- Record negative/flat results honestly. A curriculum or objective change requires a documented new experiment, not a rewritten earlier result.
- Keep binary checkpoints/adapters out of Git. Prepare a model card with base model, data/splits, reward, config, runtime, limitations and reproduction steps. Upload only when authorized.

```powershell
docker @dockerArgs python -m training.train_grpo --config configs/grpo_4060.yaml
docker @dockerArgs python scripts/evaluate.py --model Qwen/Qwen2.5-3B-Instruct --adapter conflictbench-grpo-output/grpo_4060/final --data data/test_v1.jsonl --run-name grpo4060_selected_test_v1 --max-new-tokens 1024
docker @dockerArgs python scripts/make_leaderboard.py
```

**Complete when:** a reproducible trained-versus-base table is backed by artifacts; the actual difficulty coverage, precision and statistical limits are stated. A smoke run alone does not complete this task.

## 10. R4 — Finish the multi-step task and real agent evaluation

**Blueprint:** remaining P7-1. **Relevant files:** `multistep/`, `scripts/evaluate_multistep.py`, `tests/test_multistep_*.py`, `docs/MULTISTEP.md`.

### Confirmed design gap: free deferral

An exploratory scripted strategy waited until every message/status update arrived, resolved the complete known inbox, applied only final winners, and finished. It scored **1.0 on all 600 scenarios**, averaging **25.47 steps**. This observation was not saved as a committed baseline run; preserve it as a finding and prepare a reproducible regression experiment.

This does not show that an LLM solves the task. It shows that the current mechanics let an ideal solver postpone every decision, avoiding the intended need to act and correct course.

Required work:

- Add `wait_until_end` as a named baseline, with saved trajectories and summary support.
- Design and implement meaningful action timing: explicit deadlines, intermediate company-state obligations, or irreversible consequences. The policy must sometimes need to act before later information arrives.
- Do not fix the shortcut by merely charging an arbitrary wait penalty or punishing unavoidable uncertainty. Document what the agent can know, what action is due and when, and which later corrections are justified.
- Keep the original multi-step version reproducible; version mechanics/rewards and re-run every baseline on the new version before comparing scores.
- Handle unknown facts and not-yet-arrived dependency parents explicitly. Test early/late facts, equal-rank dated supersession, overlapping scopes, dependencies, invalid calls, exhausted budgets and legitimate reverts.
- Decide whether standing notices/delegations can change over time; this is a current limitation. Implement the chosen temporal-notice behavior or record a deliberate documented fixed-notice scope with a separate outstanding enhancement.
- Add an LLM/API agent backend: send only public observations/history and tool schemas, accept one action, feed back observations, finish or exhaust the budget. Never give the model hidden `Scenario` fields, `resolve()`, ground truth or future messages.
- Save complete action/observation transcripts, tool errors, latency/steps, timing violations, corrections, final state, reward breakdown and model/runtime provenance.
- Evaluate at least the baseline model; treat multi-step training as a separate extension requiring a specific algorithm/design. Do not claim the existing single-step GRPO script already trains tool-using trajectories.
- Keep OpenEnv WebSocket episode coverage and cross-session isolation. Improve leaderboard metadata so task/reward versions are explicit; do not present a prefixed dataset hash as the actual file SHA-256.

**Complete when:** delayed-action exploits are quantified and closed by task semantics, a timely competent strategy has a clear advantage, an actual model-agent result exists, and both versions' results remain interpretable.

## 11. R5 — Complete packaging, imports and dependency separation

**Blueprint:** P1-8. **Relevant files:** `pyproject.toml`, root modules, `server/`, `training/`, `multistep/`, apps, scripts and OpenEnv metadata.

- Replace the root-directory package mapping with a conventional installable `conflictbench/` package. Include the training and multi-step components that are intended public interfaces.
- Use package-relative imports internally and remove checkout-dependent `sys.path` hacks where feasible.
- Retain thin deprecated root compatibility commands/imports for existing notebooks and external users. Do not accidentally create duplicate dataclass identities through both old and new module paths.
- Establish one package version source and document the compatibility/version change. Include necessary configs/resources in distribution artifacts with a clear lookup policy.
- Separate minimal environment/server dependencies, demo dependencies, training dependencies and development dependencies. Pin the OpenEnv version to the API actually supported; avoid forcing torch into a CPU-only environment install.
- Update console scripts, environment entry points, `openenv.yaml`, Docker/Space imports, dataset exporter, leaderboard startup and documentation together.
- Prepare clean-environment editable-install and wheel-install checks, invoked from outside the checkout so local files cannot mask missing installed modules.
- Cover `from conflictbench import ConflictBenchEnv`, the multi-step public import, CLI entry points, server startup, legacy imports and packaged config/resource access.
- Do not delete arbitrary caches, old worktrees or artifacts as part of packaging cleanup. Only remove owned generated files when necessary.

**Complete when:** Harsh verifies clean editable and wheel installations, installed entry points work without checkout path hacks, CPU installs stay lightweight, and legacy interfaces retain their documented behavior.

## 12. R6 — Complete task variety and stronger split generalization

**Blueprint:** P3-5 and the topic-held-out part of P3-6.

- Current task has 17 topics and five phrasings per side, not the blueprint's 25+ topics and 8+ phrasings target. Expand to at least those targets, preserving realistic business semantics.
- Verify that each added wording expresses its structured meaning. The resolver validates structured labels; **it cannot validate that a paraphrase's English meaning matches those labels**. Use semantic review for paraphrases, especially conditions, scope, negation and dates.
- Existing splits hold out phrasings, titles and context vocabulary, but share topics/rules. Add an explicitly topic-held-out protocol or document this as an unfulfilled generalization target until implemented.
- Retain balanced restrictive/permissive winners, valid scopes, compatible distractors, decoy notices, chains, deterministic seeds, no accidental undecidable ties in the standard task and shortcut regression thresholds.
- Measure real token budgets for difficulties 1–5, including L4/L5 coverage beyond the original fixed test set's L1–L3. Version difficulty definitions if changed.
- Add diversity/coverage statistics and checks against identity, instruction ordering, template wording and ID-position shortcuts.
- Create a new dataset/version for expanded or changed task semantics. **Never overwrite `data/test_v1.jsonl`, its checksum or the old results.**

**Complete when:** expanded pools and a clearly specified split protocol have semantic review, coverage evidence, shortcut comparisons and a versioned reproducible dataset.

## 13. R7 — Add spoofed-authority and prompt-injection scenarios

**Blueprint:** P7-2.

- Add opt-in scenarios where lower-authority text claims to speak for Legal, the CEO or another senior sender, or asks the model to ignore the task/output rules.
- Establish that authenticated sender metadata and explicit delegation notices determine authority; a claim inside message text does not.
- Keep benign quoted mentions and genuine delegation controls so the task does not reward rejecting every mention of a senior office.
- Keep ground truth deterministic, scenario provenance explicit, and injection text isolated as untrusted task content throughout prompts, demos and tool-agent histories.
- Evaluate base/trained models and adversarial baselines separately on clean and spoofed cases. Report success, over-rejection, false conflict and format-failure rates.

**Complete when:** versioned spoofing scenarios, controls, regression tests, documented trust rules and real model comparisons exist.

## 14. R8 — Add configurable organization charts and authority rules

**Blueprint:** P7-4 and the PRD's configurable-hierarchy promise.

- Define and validate a YAML schema for offices, titles, tiers, scoped authority, allowed delegations and optional domain-specific rules. Provide business, healthcare, government and university examples as scoped fixtures.
- Reject unknown offices, contradictory rules, invalid tiers and unsupported/cyclic delegations with clear errors. Define how configured authority interacts with dates and standing notices.
- Load the same configuration into generator, resolver, prompts, single-/multi-step servers, training/evaluation metadata and demos. Keep the existing hierarchy as the backward-compatible default.
- Prevent hidden-label leakage in hidden-hierarchy mode. Include the hierarchy/config fingerprint in result provenance and separate comparisons when it differs.

**Complete when:** user-supplied validated org charts change actual generation/resolution consistently, defaults remain compatible, and cross-domain generalization can be evaluated reproducibly.

## 15. R9 — Add real ambiguity and selective escalation

**Blueprint:** P7-5.

- Standard scenarios currently have a resolvable winner. True equal-authority/equal-date ties can raise `UnresolvableConflict`; multi-step `escalate` is currently bookkeeping and does not affect reward.
- Design a versioned extension representing unresolved conflicts as a valid outcome. Define the required response/schema and what remains safe to execute while awaiting clarification.
- Reward appropriate escalation on genuine ambiguity, penalize unnecessary escalation of resolvable cases, and prevent `escalate everything`/`do nothing` strategies from earning a high score.
- Make escalation meaningful in multi-step state, deadlines and tool feedback. If clarification replies are supported, specify how they arrive and resolve ambiguity without giving hidden answers away.
- Preserve legacy/current standard-scenario behavior; update generator, resolver contract, verifier, schema, apps and both evaluation protocols consistently.

**Complete when:** mixed ambiguous/resolvable cases have exact ground truth, calibrated escalation metrics and abuse baselines, and environments no longer crash on deliberately supported ambiguous scenarios.

## 16. R10 — Publish the dataset, scorer and leaderboard

**Blueprint:** remaining P7-3. Existing export and Space code must be completed and deployed, not rewritten wholesale.

- Follow `docs/BENCHMARK.md`, correcting stale statements first. In particular, `dist/` is already ignored despite a sentence claiming otherwise.
- The committed `test_v1.jsonl` contains answers in a public repository. Exporting prompt-only rows does **not** make that existing set contamination-resistant. Preserve the caveat.
- For a hidden evaluation protocol, generate a fresh private set with a nonpublic seed outside tracked paths. Keep full records/answers and seed private, and record a public dataset fingerprint/version that does not disclose the answers.
- Export prompt-only/public files, card and manifest; verify private fields are absent. Keep training and private-evaluation data separate.
- Prepare public prompts and private answers in separate Hub repositories. Confirm private visibility before upload. Obtain authorization and use fine-grained credentials; never commit tokens or place them in run summaries.
- Deploy a **separate leaderboard Space**; do not overwrite the demo Space or push `leaderboard_space/` to the existing `hf` remote.
- Use a fixed code tag/SHA, validate first-boot code fetching and private-dataset loading on the real Space, and record the scorer/task/data versions on every row.
- Configure persistent leaderboard storage or a documented backup/restore process. Verify restart/rebuild persistence and corrupted-store behavior.
- Validate perfect, empty, partial and invalid submissions; missing/duplicate/unknown IDs; size limits; malicious model names/reasoning; token/path-safe errors; and concurrent submissions/atomic persistence where needed.
- Mark model names as self-reported: the Space does not prove a prediction came from the named model. Seed the board with known reproducible model runs.
- Verify public responses never disclose private answers. Document intended feedback and the limits of repeated adaptive submissions to the private set.

**Complete when:** authorized live dataset/Space URLs work, visibility and persistence are verified, baseline submissions are present, and contamination/self-reporting limitations are explicit.

## 17. R11 — Finish documentation, demo and research write-up

**Blueprint:** follow-through on P6-1/P6-2/P6-3 and P7-6.

- Update README, architecture, repository layout and roadmap: multi-step episodes, exporter, leaderboard app and CI already exist. Do not continue labeling all of P7 as unimplemented.
- Explain the distinction between the original single-step environment and the new multi-step version; do not claim genuine timely-decision evaluation before R4 evidence supports it.
- Keep reward formulas and weights generated from/imported from the current scorer where possible. `conflict_type` and free-text reasoning are not currently scored; do not claim they are.
- Update training guide, OpenEnv metadata, PRD, blog and archived logs from actual results. Preserve historical unverified claims as clearly labeled history, never as current proof.
- Update the demo's base/adapter labels and configured adapter after real measurements. Keep lazy loading, consistent prompts, adapter-off base inference, no silent truncation and escaped model output.
- Review training Space settings with Harsh: password secret configured, private/paused status as appropriate, pinned code ref, no unauthenticated public job trigger, and constants/config alignment. Source-code locks alone do not prove live settings are correct.
- Prepare a concise workshop-style write-up, approximately four pages: task/rules, original reward hacks, redesign, protocol, baselines, trained results, multi-step deferral finding/fix, limitations, related work and reproduction pointers. Use real completions, not invented before/after examples.
- Attribute instruction-priority prior work properly; do not claim first-of-its-kind status or an ungameable reward. Do not equate a deterministic privileged resolver with model reasoning.
- Link the code, dataset/model cards, run artifacts and leaderboard. A failed training experiment must remain honestly reportable.

**Complete when:** every public claim has a code or artifact source, links and hierarchy descriptions agree, stale status banners are replaced accurately, and the write-up/model/dataset documentation is ready for authorized release.

## 18. R12 — CI, integration, evaluation provenance and robustness

**Blueprint:** P7-7 exists; finish verification and extensions rather than creating duplicate CI.

- Explain current triggers: a feature-branch push alone does not trigger the existing push-to-main job. Open an authorized PR or deliberately extend triggers if Harsh wants branch CI.
- Keep CPU tests, reward-hack limits, semantic generator/resolver checks, docs/formula checks, link validation and fixed-set integrity in CI. Add packaging checks once R5 lands.
- Cover new task versions, spoofing, org charts, escalation, timed actions, private-export separation and installed-package entry points with meaningful tests. Keep GPU/model downloads out of routine CPU CI.
- Audit previously untested config input boundaries, including malformed YAML and invalid/non-finite numeric values, when hardening configuration. They were explicitly outside the prior consolidation review; they are not proven safe.
- Record explicit task/scorer/hierarchy/dataset/model revisions rather than relying only on a dataset hash or a local package version. Preserve raw file hashes as real hashes.
- Harden long evaluation runs against network failures and interruptions: save completed work incrementally, support resumable runs without duplicate records, preserve failure counts, and avoid replacing a complete result with a partial rerun.
- Recheck OpenEnv session isolation and ground-truth hiding, parser robustness, app escaping, scoring monotonicity and baseline thresholds after contract changes.
- Prepare Python 3.11/3.12 verification commands consistent with CI. The previous local CPU venv used Python 3.14; do not assume that is the declared supported runtime for all dependencies.

**Complete when:** Harsh-supplied checks/CI runs cover the new contracts, all failures have explicit resolution or blockers, and no unvalidated GPU assumption is hidden behind a passing CPU suite.

## 19. R13 — Final integration and reproducible release

- Review the complete diff and obtain a fresh code review for substantial changes. Reconcile every important finding, with tests prepared for Harsh to execute.
- Keep commits focused; do not bundle secrets, `.claude/`, model weights, private datasets, local caches or unrelated worktrees.
- Verify `origin` still points to the intended GitHub repository before pushing. Use normal pushes; never force-push to fix a transport problem.
- Earlier transport issue: HTTP/2 push returned HTTP 408; a normal `git -c http.version=HTTP/1.1 push -u origin feat/future-scope` succeeded. Use it only if that transport issue recurs.
- Verify the remote commit SHA after pushing. A local commit or an `Everything up-to-date` line accompanying an RPC error is not sufficient proof.
- Create/update an authorized PR with an accurate description and evidence. Do not merge to `main` on your own. Release tags, HF publishing and infrastructure changes require their own applicable authorization.
- Finish the progress ledger with all task IDs, commits, commands, evidence, release URLs and unresolved limitations. No unchecked task may be omitted from the final handoff.

**Complete when:** the agreed release is traceable from source to runtime/data/model/scorer artifacts, GitHub contains the intended changes, and every task has evidence or an explicit outstanding owner/blocker.

## 20. Original blueprint coverage: nothing silently dropped

| Original IDs | Handoff treatment |
|---|---|
| P0-1, P0-2, P0-4 | Implemented; preserve test harness, cheat baselines, historical evidence, license and truthful status. Maintain through R11–R12. |
| P0-3 | Lock code implemented; live password/privacy/paused settings still require verification in R11. |
| P1-1 through P1-7 | Implemented; preserve parser, seeded RNG, labels, server, env lifecycle, fair demo and shared-prompt inference. Recheck integration after R5/R7–R9. |
| P1-8 | Partial; complete package structure, compatibility, resources and dependency separation in R5. Production-script consolidation is already done. |
| P2-1 through P2-7 | Implemented, including 500-scenario monotonicity regression; preserve scorer contracts and hack limits in R12. Do not rewrite merely to satisfy stale blueprint wording. |
| P3-1 through P3-4 | Implemented; preserve balanced winners, distractors, hidden hierarchy and richer rules. Extend only with versioned evidence. |
| P3-5 | Partial; 25+ topics / 8+ phrasings and broader difficulty coverage remain in R6. |
| P3-6 | Vocabulary/title/context splits implemented; topic-held-out coverage is still outstanding in R6. Keep test_v1 immutable. |
| P3-7 | Existing shortcut tests implemented; preserve them and distinguish privileged structured solvers from text-only models in R2/R6/R12. |
| P4-1 | Harness implemented; real model-agent support and resumable/provenance improvements in R4/R12. |
| P4-2 | Scripted rows measured; real-model rows and approved broader/API comparisons in R2. |
| P4-3 | Historical adapter four-cell re-score in R2. |
| P5-1, P5-2 | Code implemented and consolidated; real tokenizer, GPU, resume and best-checkpoint verification remain in R1/R3. |
| P5-3 | Pinning/Space code implemented; working-stack evidence and actual deployment settings remain in R1/R11. |
| P5-4 | New-task 3B training, final held-out evaluation and comparison remain in R3. |
| P6-1 through P6-3 | Earlier rewrite/link cleanup implemented; newer features and future measured results need R11 updates. |
| P7-1 | Multi-step source and scripted baselines implemented; deferral fix, temporal-design decision and model-agent evaluation remain in R4. |
| P7-2 | Spoofed authority/prompt injection in R7. |
| P7-3 | Export/scoring tooling implemented; fresh private set, live publication, persistence and model-seeded board in R10. |
| P7-4 | Configurable hierarchy/domain fixtures in R8. |
| P7-5 | Genuine ambiguity/selective escalation in R9. |
| P7-6 | Evidence-based write-up and cards in R11. |
| P7-7 | CI implemented; trigger awareness, new contract coverage and installed-package checks in R12. |

## 21. Commands to hand to Harsh after relevant edits

These are a verification menu, **not instructions for Luna to execute them**. Supply only the commands relevant to the changes made and the environment Harsh is using; include expected artifacts and how to report failures.

```powershell
# Existing CPU suite, inside the project's CPU environment
.\.venv\Scripts\python.exe -m pytest -q

# Relative documentation links
.\.venv\Scripts\python.exe scripts/check_links.py

# Windows checksum inspection: compare with data/test_v1.sha256
Get-FileHash -Algorithm SHA256 data/test_v1.jsonl
Get-Content data/test_v1.sha256

# Existing multi-step scripted baselines: v1 behavior, not the planned timed extension
.\.venv\Scripts\python.exe scripts/evaluate_multistep.py

# Existing prompt-only/full-answer export; this performs no upload
.\.venv\Scripts\python.exe scripts/export_hf_dataset.py --data data/test_v1.jsonl --out dist/conflictbench-test-v1
```

Do not publish newly generated metrics from stale code or treat these existing commands as covering features that have not yet been implemented. Add the exact new commands alongside each new interface.

## 22. What your progress/final report must say

For each R0–R13 task, state implementation status separately from verification/deployment status. Give the commit, the evidence or the precise missing evidence, and the next owner/action. Include any negative experiment result and any deliberate scope decision. Never say the entire project is finished while packaging, model evidence, timing semantics, publishing, research extensions or a user-owned verification step is still outstanding.

Start by installing/confirming Superpowers, reading this handoff and the linked project material, then reconciling Harsh's manual-run output. Proceed with independent implementation work while he runs the supplied commands.
