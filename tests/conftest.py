"""Shared pytest setup. Puts the repo root on sys.path until packaging is fixed (P1-8)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
