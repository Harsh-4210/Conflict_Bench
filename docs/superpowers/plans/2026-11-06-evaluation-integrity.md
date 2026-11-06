# Evaluation integrity implementation plan

> **For agentic workers:** Use executing-plans for inline implementation. The user owns execution of all tests and experiments; write regression tests first but do not run them.

**Goal:** Make the documented deferral baseline reproducible, separate incompatible evaluation protocols, and reject invalid training configuration before expensive work.

**Architecture:** Keep existing task/reward behavior unchanged. Add explicit evaluation provenance and protocol grouping, register a wait-until-end reference strategy, and harden config parsing at its boundary. Preserve saved historical results.

**Spec:** `LUNA_HANDOFF.md`, phases 0, 2, 4 and 12 (R0/R2/R4/R12).

## Global constraints

- No tests, model evaluation, dry runs, training or image builds are executed by the agent.
- No changes to fixed datasets, existing result files, scorer formulas, or active environment semantics.
- Implementation checkboxes and user-run verification checkboxes stay separate.
- Do not merge, publish or overwrite unrelated `.claude/` content.

## Tasks

- [x] Reorganize the full handoff into numbered phases and create the progress ledger.
- [x] Write unexecuted regression tests for deferral, provenance/grouping, and invalid configuration.
- [x] Add the wait-until-end strategy using only arrived instructions and known information; retain arrival ordering when applying winners.
- [x] Add task/scorer/protocol metadata and real dataset hashes to new evaluation summaries. Separate task versions, scorer fingerprints, selections and episode settings in leaderboards, with conservative handling of legacy records.
- [x] Reject malformed YAML, nonmapping inputs, nonstring keys, nonfinite numbers and ambiguous difficulty-map keys as ConfigError, without model imports.
- [x] Update docs and phase checkboxes to distinguish implementation from pending verification; provide exact commands.
- [x] Review changes, commit, and push the scoped implementation (`6f6e651` on `origin/feat/future-scope`). Leave package migration and task-semantic extensions pending until their own integration boundary. Independent final approval and user-run tests remain open.

## Review focus

- Waiting baseline must not inspect future instructions, and final writes must respect arrival order.
- A prefixed legacy hash must not be displayed as a real SHA-256 in a new summary.
- Different task/scorer/selected-scenario/protocol groups must never share one ranking.
- Old summaries must remain readable without inventing missing provenance.
- Numeric/YAML failures must become actionable ConfigError, not raw exceptions or silent normalization.
- Actually scored subsets must be compared, not just requested subsets or equal skip counts.
- Resolver changes must invalidate the multi-step scorer fingerprint.

## Execution record

- User requested implementation and detailed phase-wise marking; earlier user-owned test execution remains in force.
- Current source branch starts at `1efb0fa`. Only `.claude/` is untracked before this increment.
- GPU build, tokenizer validation, baseline model runs and training evidence have not been supplied; mark their phases awaiting Harsh rather than repeating them.
- Read-only reviewer findings addressed: strict YAML parser now shared by files/CLI; invalid timestamps and null/empty roots handled; resolver included in scorer sources; scored-subset hash added to ranking identity. New regression cases remain unexecuted per the user instruction.
- Reviewer reached its usage limit before a final verdict. Record review as incomplete and finish with a source/diff self-review; do not invent independent approval or test results.
