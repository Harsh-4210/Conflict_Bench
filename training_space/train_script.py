"""
ConflictBench training Space - glue between the Gradio UI (app.py) and the shared training pipeline.

There is no training code here any more: ``run_training`` loads ``configs/grpo_a10g.yaml`` from the repo checkout
that the Dockerfile cloned at the pinned ``CONFLICTBENCH_REF`` and calls ``training.train_grpo.train``.

The constants below exist only because app.py reads them with ``ast`` (importing this file must stay free of
heavy imports) to fill its configuration table. They mirror the YAML file, and ``run_training`` as well as
tests/test_training_space_lock.py refuse to run if they drift from it.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Callable, List, Optional

log = logging.getLogger("train")

# Where the Dockerfile cloned the repo (at CONFLICTBENCH_REF). Never cloned at run time: that would follow HEAD.
REPO_DIR = Path(os.environ.get("CONFLICTBENCH_DIR", str(Path.home() / "conflictbench")))
CONFIG_FILE = "configs/grpo_a10g.yaml"

# ── Mirror of configs/grpo_a10g.yaml, for the UI table in app.py (checked against the YAML) ──────────────────────
TRAIN_SCENARIOS = 600
EVAL_SCENARIOS = 30
DIFFICULTY_TRAIN_WEIGHTS = {1: 0.4, 2: 0.4, 3: 0.2}
NUM_EPOCHS = 1.0
NUM_GENERATIONS = 8
MAX_NEW_TOKENS = 1024
LEARNING_RATE = 5.0e-6
BETA = 0.04

# constant name -> TrainConfig field
CONSTANT_TO_FIELD = {
    "TRAIN_SCENARIOS": "train_size",
    "EVAL_SCENARIOS": "val_size",
    "DIFFICULTY_TRAIN_WEIGHTS": "train_difficulty_mix",
    "NUM_EPOCHS": "epochs",
    "NUM_GENERATIONS": "num_generations",
    "MAX_NEW_TOKENS": "max_completion_tokens",
    "LEARNING_RATE": "learning_rate",
    "BETA": "beta",
}


def check_constants_match(cfg, constants: Optional[dict] = None) -> List[str]:
    """Names of UI constants that differ from the loaded config (empty list = in sync)."""
    constants = constants if constants is not None else {name: globals()[name] for name in CONSTANT_TO_FIELD}
    return [name for name, field in CONSTANT_TO_FIELD.items() if constants[name] != getattr(cfg, field)]


def generate_plots(log_history: list, plots_dir: Path) -> List[str]:
    """Reward / loss / KL / dashboard PNGs from the Trainer's log history."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)

    def series(key):
        return [(e["step"], e[key]) for e in log_history if key in e and "step" in e
                and isinstance(e[key], (int, float))]

    paths: List[str] = []

    def save(fig, name):
        p = plots_dir / name
        fig.savefig(p)
        plt.close(fig)
        paths.append(str(p))

    fig, ax = plt.subplots(figsize=(10, 5))
    for key, label in (("reward", "Train reward"), ("eval_reward", "Val reward")):
        pts = series(key)
        if pts:
            ax.plot(*zip(*pts), label=label, marker="o" if key == "eval_reward" else None)
    ax.set_title("Reward (composite)"); ax.legend(); ax.grid(True)
    save(fig, "reward_curve.png")

    for key, title, name, color in (("loss", "Training loss", "loss_curve.png", "red"),
                                    ("kl", "KL divergence", "kl_divergence.png", "purple")):
        fig, ax = plt.subplots(figsize=(10, 5))
        pts = series(key)
        if pts:
            ax.plot(*zip(*pts), color=color)
        ax.set_title(title); ax.grid(True)
        save(fig, name)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, key in zip(axes.flat, ["reward", "loss", "kl", "eval_reward"]):
        pts = series(key)
        if pts:
            ax.plot(*zip(*pts))
        ax.set_title(key); ax.grid(True)
    fig.tight_layout()
    save(fig, "training_dashboard.png")
    return paths


def run_training(progress_callback: Optional[Callable[[str], None]] = None) -> List[str]:
    def emit(msg: str) -> None:
        log.info(msg)
        if progress_callback:
            progress_callback(msg)

    if not (REPO_DIR / "training" / "train_grpo.py").is_file():
        raise RuntimeError(f"ConflictBench checkout not found at {REPO_DIR}. The Dockerfile clones it at "
                           "CONFLICTBENCH_REF; this script deliberately does not clone HEAD at run time.")
    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))

    from training.config import load_config
    from training.train_grpo import git_commit_hash, train

    overrides = {}
    if os.environ.get("HF_REPO_ID"):          # optional: push the adapter (needs HF_TOKEN as a Space secret too)
        overrides["hub_repo"] = os.environ["HF_REPO_ID"]
    cfg = load_config(REPO_DIR / CONFIG_FILE, overrides)

    stale = check_constants_match(cfg)
    if stale:
        raise RuntimeError(f"training_space/train_script.py constants {stale} differ from {CONFIG_FILE}; "
                           "the UI table would show the wrong run. Update the constants.")
    emit(f"code: {REPO_DIR} @ {git_commit_hash(REPO_DIR)} (CONFLICTBENCH_REF={os.environ.get('CONFLICTBENCH_REF', '?')})")
    emit(f"config: {CONFIG_FILE} -> run '{cfg.run_name}'")

    result = train(cfg, progress_callback=progress_callback)
    emit("generating plots")
    return generate_plots(result["log_history"], Path(result["run_dir"]) / "plots")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_training()
