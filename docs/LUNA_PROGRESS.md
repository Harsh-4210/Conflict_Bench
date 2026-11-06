# ConflictBench phase progress

Updated 6 November 2026. This is the live execution ledger for [the detailed phase plan](../LUNA_HANDOFF.md).

**Execution owner:** the agent implements/reviews code; Harsh runs tests, experiments and GPU commands. No test/evaluation/training has been executed for this increment.

## Current implementation batch: evaluation integrity

Starting commit: `1efb0fa`. Implementation commit: [`6f6e651`](https://github.com/Harsh-4210/Conflict_Bench/commit/6f6e651).
Pushed to `origin/feat/future-scope`. Plan: [evaluation integrity](superpowers/plans/2026-11-06-evaluation-integrity.md).

| Item | Implementation | Verification | Files / evidence |
|---|---|---|---|
| Phase 0: source inventory | Implemented | Read-only inventory recorded | Feature branch at handoff; unrelated `.claude/` preserved |
| Phase 2: selected-scenario metadata | Implemented | AWAITING_HARSH_RUN | `scripts/evaluate.py`, `tests/test_evaluate.py` |
| Phase 4: `wait_until_end` baseline | Implemented | AWAITING_HARSH_RUN | `scripts/evaluate_multistep.py`, `tests/test_multistep_evaluate.py` |
| Phase 12: scorer/task/protocol provenance | Implemented | AWAITING_HARSH_RUN | `scripts/provenance.py`, both evaluators, leaderboard; requested and actually scored subsets have separate hashes |
| Phase 12: config boundary hardening | Implemented | AWAITING_HARSH_RUN | `training/config.py`, override parsing, `tests/test_training_config_boundaries.py` |
| Phase 11: phase-wise plan | Implemented | Document review pending | `LUNA_HANDOFF.md`: 14 phases, detailed checkboxes and all original blueprint IDs retained |
| Phase 13: current-batch source push | Implemented and pushed | Tests and independent final verdict remain pending | `6f6e651`; review findings addressed, source self-review recorded |

## Harsh: commands for this batch

Run from the repository root in the CPU environment. These are commands to run manually; they have not been executed by the agent.

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_evaluation_provenance.py tests/test_multistep_evaluate.py tests/test_leaderboard.py tests/test_evaluate.py tests/test_training_config_boundaries.py tests/test_training_config.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/check_links.py
```

Optional reference-baseline measurement, after tests pass; writes a new run under ignored `dist/` and does not rewrite committed results:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_multistep.py --strategy wait_until_end --data data/test_v1.jsonl --out-dir dist/manual-evaluation-integrity
.\.venv\Scripts\python.exe scripts/make_leaderboard.py --results-dir dist/manual-evaluation-integrity
```

Expected evidence to supply: pytest summaries/errors; `dist/manual-evaluation-integrity/multistep_wait_until_end/summary.json`; generated leaderboard. For inbox-v1 the structured deferral solver is expected to reach a perfect final state without reverts; that is a regression expectation, not a new measured result.

## Remaining phase states and owners

- Phase 0: waiting for Harsh's new runtime artifacts; do not repeat his work.
- Phase 1: awaiting Harsh's Docker build, tokenizer filtering and GPU smoke/resume/checkpoint evidence.
- Phase 2: all real-model rows and historical adapter's four-cell study remain pending.
- Phase 3: full 3B training/comparison remains pending; L1-only coverage must be labeled.
- Phase 4: timed-action semantics, full trajectories and actual model-agent backend remain pending; adding a baseline alone does not fix deferral.
- Phase 5: package migration remains pending. Keep an explicit integration boundary around changes to installed imports/resources.
- Phase 6: expanded topics/phrasings and topic-held-out protocol remain pending; preserve test_v1.
- Phase 7: spoofing/injection extension remains pending.
- Phase 8: configurable org-chart schema/domain fixtures remain pending.
- Phase 9: genuine ambiguity and meaningful escalation remain pending.
- Phase 10: fresh private set and live publication remain pending; publication requires authorization.
- Phase 11: public README/architecture/demo/write-up updates await implementation and measurement evidence.
- Phase 12: incremental/resumable evaluation, richer model/hierarchy revisions, packaging CI and further integration remain pending.
- Phase 13: implementation commit `6f6e651` is pushed; user verification and independent final approval remain open. Release, PR merge and HF publication are not authorized by a normal feature-branch push.

## Decisions and limitations

- User-owned test execution supersedes the skills' automatic test-running workflow. Tests were written before implementation but no RED/GREEN result is claimed.
- Existing task/scorer semantics and saved results remain unchanged. Provenance groups conservatively separate historical unversioned results from newly fingerprinted results.
- Scorer fingerprints hash source files; even a comment-only source edit creates a different group. This conservative separation avoids assuming semantic equivalence without evidence.
- Protocol identity includes multi-step episode seeds/settings and scenario selection; the raw dataset hash remains a real hash. It does not yet attest to every model/hierarchy revision.
- Actually scored scenario IDs have a separate fingerprint, so equal skip counts on different cases do not create a shared ranking. Multi-step scorer fingerprints include `resolver.py` as well as environment/state/scoring code.
- Files and CLI overrides share the safe duplicate-detecting YAML parser. Invalid timestamps are normalized to configuration errors. Empty/null root documents are rejected; an explicit `{}` mapping remains the way to request defaults.
- The deferral solver is privileged because it uses known structured fields. It is a diagnostic reference strategy, not a language-model result.
- Do not begin invasive packaging/task-semantic changes inside this batch while its integration checks await execution. Continue those as separately tracked phase changes.

## Commit and review record

An independent read-only review identified missing strict CLI YAML parsing, missing resolver provenance,
different skipped-case subsets sharing a ranking, and invalid timestamp/null-document boundary handling.
All four were addressed with source changes and additional unexecuted regression cases. The reviewer
reached its usage limit before a final verdict; no independent approval is claimed. The implementation
received an additional source/diff self-review, with no execution. Commit `6f6e651` was pushed successfully
to `feat/future-scope` (Git reported remote update `1efb0fa..6f6e651`). This closes source delivery for
this batch, not test verification, independent final approval or the whole release phase. Keep all
user-run verification entries open until actual results arrive.
