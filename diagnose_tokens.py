"""Moved to scripts/diagnose_tokens.py (it now reads a training YAML). This shim forwards to it.

    python scripts/diagnose_tokens.py --config configs/grpo_t4.yaml
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.diagnose_tokens import main  # noqa: E402,F401

if __name__ == "__main__":
    argv = sys.argv[1:]
    if not any(a == "--config" or a.startswith("--config=") for a in argv):
        argv = ["--config", str(Path(__file__).resolve().parent / "configs" / "grpo_t4.yaml")] + argv
    raise SystemExit(main(argv))
