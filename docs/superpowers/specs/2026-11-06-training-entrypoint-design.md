# Consolidate the training entry point

The approved next milestone is a reproducible base-versus-trained evaluation. This first increment removes the competing production training path and verifies the dataset plumbing before a GPU run.

Use `training/train_grpo.py` for all training. Preserve the original untracked `train_grpo_prod.py` outside the repository, then replace it with a deprecated compatibility command. Move its 120/20 scenario counts, level-1 focus, two epochs, batch 1 with accumulation 4, four generations, 4096-token context and 1024-token completion budget into `configs/grpo_4060.yaml`. This is an experimental low-memory preset, not a claim of verified 8 GB compatibility. Use explicit train/val splits, the shared prompt, token filtering, standard reward, checkpoint selection and local-only reporting from the maintained pipeline.

Add `--check-config` to print the validated, resolved config without importing model libraries, downloading files or starting training. Keep it mutually exclusive with `--dry-run`. Missing tokenizer dependencies should produce a useful error and nonzero status. Preserve explicit config paths and overrides through the compatibility command. Reject its former `--eval` option with directions to the maintained evaluation harness.

The alternative, fixing the duplicate trainer in place, would preserve divergent prompts, splits and checkpoint behavior. A YAML preset plus compatibility command avoids maintaining two trainers. Deleting the old command would unnecessarily break existing commands.

No changes to the test set, reward formula or multi-step task in this increment. No publication or remote GPU provisioning. Check the real tokenizer in a separate environment, and record any runtime limitation honestly.
