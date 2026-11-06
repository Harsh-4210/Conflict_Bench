# Training Entry Point Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route the production command through the maintained pipeline and verify configuration and dataset preparation before training.

**Architecture:** A compatibility command selects a YAML preset and delegates to the existing CLI. The CLI gains a config-only mode; dataset construction and rewards stay shared.

**Tech Stack:** Python, PyYAML, pytest; Transformers only for the real tokenizer dry run.

**Spec:** `docs/superpowers/specs/2026-11-06-training-entrypoint-design.md`

## Global Constraints

- Preserve the original untracked script outside the repository before replacement.
- Use explicit train/val splits, the shared prompt, token filtering, standard reward, checkpoint selection and local-only reporting.
- No changes to the test set, reward formula or multi-step task in this increment.
- No publication or remote GPU provisioning.

## Review Focus

- Explicit `--config=...` must not be replaced by the shim's default.
- Invalid configuration must fail before tokenizer or model imports.
- `--check-config` must not start training, download anything or write artifacts.
- Legacy `--eval` must not accidentally start training.
- The low-memory preset remains experimental until validated on a GPU.

## Task 1: Consolidate and verify training commands

**Files:** Create `configs/grpo_4060.yaml`, `tests/test_training_cli.py`; modify `train_grpo_prod.py`, `training/train_grpo.py`, `docs/TRAINING_GUIDE.md`.

**Interfaces:** Preserve `training.train_grpo.main(argv=None) -> int`; add mutually exclusive `--check-config` and `--dry-run`. Preserve `train_grpo_prod.main(argv=None) -> int` as a compatibility entry point.

- [x] Add subprocess tests: config-only prints JSON with overrides and no artifacts; invalid config returns 2; conflicting modes return 2; shim help works without GPU dependencies; shim selects its default and respects both explicit config syntaxes; legacy evaluation returns 2 with evaluation instructions. Add a missing-tokenizer dependency test at the import boundary.
- [x] Run `python -m pytest -q tests/test_training_cli.py --tb=short -p no:cacheprovider`. Expected: new behaviors fail before implementation.
- [x] Implement the preset, compatibility command and config-only CLI. Catch a missing Transformers import for dry runs and explain the tokenizer dependency.
- [x] Run the new tests and the existing training tests. Expected: all pass.
- [x] Run the full CPU suite, link checker and checksum check. Expected: no failures; distinguish environment access errors if encountered.
- [ ] Install the pinned tokenizer dependencies in a separate environment and run real-tokenizer dry runs on smoke and 4060 configs. Expected: populated train/val datasets, no test split.
- [x] Update the training guide with commands and current limitations; real tokenizer retention has not been measured. Review the complete diff. User subsequently requested a commit and GitHub push.

## Execution record

- Ruling: Reuse the current feature checkout; no worktree is attached and the task includes an untracked user script. Keep unrelated files unchanged.
- Original script backed up to the chat's visualization workspace under `training-backup-20261106/train_grpo_prod.py`.
- Initial ruling: Keep this increment uncommitted for review. Superseded by the user's explicit request to push the changes to GitHub.
- User selected local Docker on the RTX 4060 for the GPU smoke run. Docker Desktop was started and an Ubuntu container successfully reported the RTX 4060 and 8188 MiB VRAM.
- Added `docker/training/Dockerfile` plus a Dockerfile-specific context allowlist; reuse the pinned training stack, bind-mount the checkout, and retain model downloads in a Docker volume. Verify by building and running, rather than source-text tests for container configuration.
- Verification: new CLI tests failed before implementation, then 91 training tests passed; full CPU suite 549 passed / 1 skipped; relative documentation links and fixed test-set SHA-256 passed.
- Final consolidation review: independent read-only reviewer approved, no Critical/Important/Minor findings. Ruling: GPU memory fit and pinned-stack execution require the runtime validation below, not CPU claims. Ruling: unchanged malformed-YAML/exotic-value config validation is outside this increment; its existing behavior remains. Ruling: actual tokenizer retention is validated by real-tokenizer dry runs, not inferred from unit tests.
- User requested no further test execution and will run the commands themselves. Stopped the host tokenizer installation and Docker image build before completion. No tokenizer retention results, completed CUDA image or GPU smoke-run metrics are claimed. The user owns the remaining runtime validation.
