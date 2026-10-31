"""ConflictBench training package (one GRPO pipeline, driven by a YAML config).

Importing this package never imports torch / unsloth / trl / datasets; those are
imported inside ``training.train_grpo.train`` only.
"""

import sys
from pathlib import Path

# The environment modules (generator, verifier, prompts, scenario_io) live at the repo root and are not
# installed as a package yet (P1-8), so make them importable when running ``python -m training.train_grpo``
# from anywhere (Colab, Kaggle, the training Space).
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
