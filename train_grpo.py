"""
DEPRECATED - kept only so old commands keep working.

The training code now lives in ``training/train_grpo.py`` and is driven by YAML files in ``configs/``:

    python -m training.train_grpo --config configs/grpo_smoke.yaml     # 10-step plumbing test
    python -m training.train_grpo --config configs/grpo_t4.yaml        # free Kaggle / Colab T4
    python -m training.train_grpo --config configs/grpo_a10g.yaml      # A10G / L4

``python train_grpo.py [args]`` still works: it forwards to the new entry point and, if you pass no
``--config``, uses configs/grpo_a10g.yaml. Importing this module has no side effects.
"""

import sys
import warnings
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training.train_grpo import main as _main  # noqa: E402  (no heavy imports at module level)

DEFAULT_CONFIG = str(_ROOT / "configs" / "grpo_a10g.yaml")


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    warnings.warn("train_grpo.py is deprecated; use `python -m training.train_grpo --config configs/<name>.yaml`",
                  DeprecationWarning, stacklevel=2)
    print("NOTE: train_grpo.py is deprecated; use `python -m training.train_grpo --config configs/<name>.yaml`",
          file=sys.stderr)
    if not any(a == "--config" or a.startswith("--config=") for a in args):
        args = ["--config", DEFAULT_CONFIG] + args
    return _main(args)


if __name__ == "__main__":
    raise SystemExit(main())
