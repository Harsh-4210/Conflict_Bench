"""Deprecated compatibility command for the experimental low-memory training preset.

Use ``python -m training.train_grpo --config configs/grpo_4060.yaml``.
All dataset construction, rewards and training live in the maintained pipeline.
"""

import sys
from pathlib import Path

from training.train_grpo import main as _main

DEFAULT_CONFIG = str(Path(__file__).resolve().parent / "configs" / "grpo_4060.yaml")


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    print("NOTE: train_grpo_prod.py is deprecated; use "
          "`python -m training.train_grpo --config configs/grpo_4060.yaml`", file=sys.stderr)
    if any(arg == "--eval" or arg.startswith("--eval=") for arg in args):
        print("Use `python scripts/evaluate.py --model <base-model> --adapter <adapter-path> "
              "--data data/test_v1.jsonl` for evaluation.", file=sys.stderr)
        return 2
    if not any(arg == "--config" or arg.startswith("--config=") for arg in args):
        args = ["--config", DEFAULT_CONFIG] + args
    return _main(args)


if __name__ == "__main__":
    raise SystemExit(main())
