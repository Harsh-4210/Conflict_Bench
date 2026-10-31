"""ConflictBench — Instruction Priority Resolution Environment for OpenEnv."""

try:
    from .conflict_bench import ConflictBenchEnv
except ImportError:
    # Imported as a plain module, e.g. by pytest when the checkout folder isn't a valid
    # package name (worktrees like "agent-xyz"). Goes away with the package move (P1-8).
    from conflict_bench import ConflictBenchEnv

__all__ = [
    "ConflictBenchEnv",
]
