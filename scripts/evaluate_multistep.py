"""
ConflictBench — multi-step baselines
Plays the multi-step environment (multistep/) with scripted strategies and writes one
summary per strategy to results/multistep_<strategy>/summary.json. No model, no network.

Strategies (all but do_nothing play turn by turn; "known" = what has arrived so far):
  oracle                 sees the future: applies exactly the final winners, never reverts
  cautious_resolver      each turn re-runs resolve() on what is known; applies winners and
                         reverts anything that stopped being a winner
  apply_then_fix         applies every message on arrival, then reverts what resolve() on what
                         is known says is outranked (so it reverts things it could have skipped)
  apply_all_never_revert applies every message that is not already known to lose; never reverts
  greedy_latest          applies every message the moment it arrives; last writer wins; never reverts
  do_nothing             calls finish at once

Usage:
  python scripts/evaluate_multistep.py                       # all strategies on data/test_v1.jsonl
  python scripts/evaluate_multistep.py --strategy oracle --generate 30 --seed 5
"""

import argparse
import hashlib
import json
import random
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generator import GeneratorConfig, Scenario, ScenarioGenerator
from multistep.env import DEFAULT_BUDGET_FACTOR, DEFAULT_MAX_PER_TURN, MultiStepConflictEnv
from multistep.scoring import REVERT_PENALTY, REVERT_PENALTY_CAP, WASTE_PENALTY, WASTE_PENALTY_CAP
from scenario_io import dumps_scenario, load_scenarios

DEFAULT_DATA = ROOT / "data" / "test_v1.jsonl"
N_RESAMPLES = 1000
CI_SEED = 0

Action = Dict[str, Any]
Strategy = Callable[[MultiStepConflictEnv], Iterator[Action]]


# ---------------------------------------------------------------------------
# Strategies. Each is a generator that yields one tool call at a time and reads the
# environment (through its privileged helpers) after every step.
# ---------------------------------------------------------------------------

def _play(env: MultiStepConflictEnv, decide: Callable[[MultiStepConflictEnv], List[Action]]) -> Iterator[Action]:
    """Per turn: yield the decisions, then wait for the next messages; finish at the end."""
    while True:
        for action in decide(env):
            yield action
        if not env.more_messages:
            break
        yield {"tool": "wait"}
    yield {"tool": "finish"}


def oracle(env: MultiStepConflictEnv) -> Iterator[Action]:
    final = set(env.ground_truth()["followed"])
    return _play(env, lambda e: [{"tool": "apply", "id": i} for i in e.current_message_ids if i in final])


def greedy_latest(env: MultiStepConflictEnv) -> Iterator[Action]:
    return _play(env, lambda e: [{"tool": "apply", "id": i} for i in e.current_message_ids])


def apply_all_never_revert(env: MultiStepConflictEnv) -> Iterator[Action]:
    def decide(e: MultiStepConflictEnv) -> List[Action]:
        known = e.known_followed()
        return [{"tool": "apply", "id": i} for i in e.current_message_ids if i in known]
    return _play(env, decide)


def _reconcile(e: MultiStepConflictEnv) -> List[Action]:
    """Make the applied set equal to what resolve() says on what is known."""
    want = e.known_followed()
    applied = e.applied_ids()
    actions: List[Action] = [{"tool": "revert", "id": i} for i in applied if i not in want]
    keep = set(applied)
    actions += [{"tool": "apply", "id": i} for i in e.arrived_ids if i in want and i not in keep]
    return actions


def cautious_resolver(env: MultiStepConflictEnv) -> Iterator[Action]:
    return _play(env, _reconcile)


def apply_then_fix(env: MultiStepConflictEnv) -> Iterator[Action]:
    def decide(e: MultiStepConflictEnv) -> List[Action]:
        actions: List[Action] = [{"tool": "apply", "id": i} for i in e.current_message_ids]
        # whatever is applied now (old and new) that is no longer a winner gets reverted
        want = e.known_followed()
        applied = set(e.applied_ids()) | set(e.current_message_ids)
        actions += [{"tool": "revert", "id": i} for i in e.arrived_ids if i in applied and i not in want]
        # and a winner that was reverted earlier comes back
        actions += [{"tool": "apply", "id": i} for i in e.arrived_ids
                    if i in want and i not in applied]
        return actions
    return _play(env, decide)


def do_nothing(env: MultiStepConflictEnv) -> Iterator[Action]:
    yield {"tool": "finish"}


STRATEGIES: Dict[str, Strategy] = {
    "oracle": oracle,
    "cautious_resolver": cautious_resolver,
    "apply_then_fix": apply_then_fix,
    "apply_all_never_revert": apply_all_never_revert,
    "greedy_latest": greedy_latest,
    "do_nothing": do_nothing,
}


def run_episode(
    strategy: str, env: MultiStepConflictEnv, scenario: Scenario, seed: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Play one episode. Returns (reward breakdown, stats)."""
    env.reset(seed=seed, scenario=scenario)
    info: Dict[str, Any] = {}
    done = False
    for action in STRATEGIES[strategy](env):
        _, _, done, info = env.step(action)
        if done:
            break
    if not done:  # a strategy that ran out of moves without finishing
        _, _, done, info = env.step({"tool": "finish"})
    return info["reward_breakdown"], info["stats"]


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def bootstrap_ci(values: Sequence[float], n_resamples: int = N_RESAMPLES,
                 seed: int = CI_SEED, level: float = 0.95) -> Tuple[float, float]:
    """Seeded percentile bootstrap CI of the mean. Same inputs -> same interval."""
    vals = list(values)
    if not vals:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples))
    alpha = (1.0 - level) / 2.0
    return means[int(alpha * n_resamples)], means[min(n_resamples - 1, int((1.0 - alpha) * n_resamples))]


def _mean(xs: Sequence[float]) -> float:
    return statistics.fmean(xs) if xs else 0.0


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_info() -> Dict[str, object]:
    def run(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    status = run("status", "--porcelain", "--untracked-files=no")
    return {"git_commit": run("rev-parse", "HEAD"), "git_dirty": bool(status) if status is not None else None}


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def summarize(
    strategy: str, rows: List[Dict[str, Any]], data_label: str, data_sha: str,
    n_in_data: int, seed: int, env_settings: Dict[str, Any],
) -> Dict[str, Any]:
    rewards = [r["reward"] for r in rows]
    lo, hi = bootstrap_ci(rewards)

    def per(field: str) -> float:
        return _mean([r[field] for r in rows])

    return {
        "run_name": f"multistep_{strategy}",
        "task": "multistep",
        "backend": "strategy",
        "strategy": strategy,
        "model": None,
        "adapter": None,
        "data_path": data_label,
        # Distinct from the single-step runs' hash so scripts/make_leaderboard.py (which groups by
        # this field) never ranks multi-step rewards next to single-step composites.
        "data_sha256": f"multistep-{data_sha}",
        "data_file_sha256": data_sha,
        "n_scenarios_in_data": n_in_data,
        "episode_seed_base": seed,
        "env": env_settings,
        "reward_formula": {
            "revert_penalty_each": REVERT_PENALTY, "revert_penalty_cap": REVERT_PENALTY_CAP,
            "waste_penalty_each": WASTE_PENALTY, "waste_penalty_cap": WASTE_PENALTY_CAP,
        },
        **git_info(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n": len(rows),
        "skipped_too_long": 0,
        "parse_failure_rate": 0.0,
        "composite": {"mean": _mean(rewards), "ci95_low": lo, "ci95_high": hi,
                      "n_resamples": N_RESAMPLES, "ci_seed": CI_SEED},
        "per_difficulty": {
            str(d): {"mean": _mean([r["reward"] for r in rows if r["difficulty"] == d]),
                     "n": sum(1 for r in rows if r["difficulty"] == d)}
            for d in sorted({r["difficulty"] for r in rows})
        },
        "per_rubric": {
            "accuracy": per("accuracy"),
            "foreseeable_reverts": per("foreseeable_reverts"),
            "wasted_steps": per("wasted_steps"),
            "revert_penalty": per("revert_penalty"),
            "waste_penalty": per("waste_penalty"),
        },
        "episode_stats": {
            "mean_instructions": per("n_instructions"),
            "mean_turns": per("turns"),
            "mean_steps_used": per("steps_used"),
            "mean_reverts": per("reverts"),
            "share_episodes_with_a_revert": _mean([1.0 if r["reverts"] > 0 else 0.0 for r in rows]),
            "share_ended_by_budget": _mean([1.0 if r["ended_by"] == "budget" else 0.0 for r in rows]),
            "share_perfect_accuracy": _mean([1.0 if r["accuracy"] == 1.0 else 0.0 for r in rows]),
        },
    }


# ---------------------------------------------------------------------------
# Data and main
# ---------------------------------------------------------------------------

def _load_data(args: argparse.Namespace) -> Tuple[List[Scenario], str, str, int]:
    """(scenarios, label, sha256, size of the source before --n)."""
    if args.generate:
        gen = ScenarioGenerator(seed=args.seed, config=GeneratorConfig(split="test"))
        scenarios = [gen.generate(1 + (i % 3), seed=args.seed * 100_003 + i) for i in range(args.generate)]
        blob = "\n".join(dumps_scenario(s) for s in scenarios).encode("utf-8")
        return scenarios, f"generated:test:n{args.generate}:seed{args.seed}", sha256_bytes(blob), len(scenarios)
    path = Path(args.data)
    scenarios = load_scenarios(path)
    return scenarios[: args.n] if args.n else scenarios, _display_path(path), sha256_bytes(path.read_bytes()), len(scenarios)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, Any]]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strategy", action="append", choices=sorted(STRATEGIES) + ["all"],
                   help="repeatable; default: all")
    p.add_argument("--data", default=str(DEFAULT_DATA), help="scenario file (JSONL)")
    p.add_argument("--n", type=int, default=0, help="use only the first N scenarios of --data")
    p.add_argument("--generate", type=int, default=0, help="generate N test-split scenarios instead of reading --data")
    p.add_argument("--seed", type=int, default=0, help="episode seed base (episode i uses seed + i)")
    p.add_argument("--budget-factor", type=int, default=DEFAULT_BUDGET_FACTOR)
    p.add_argument("--max-per-turn", type=int, default=DEFAULT_MAX_PER_TURN)
    p.add_argument("--out-dir", default=str(ROOT / "results"))
    args = p.parse_args(argv)

    names = args.strategy or ["all"]
    if "all" in names:
        names = list(STRATEGIES)
    scenarios, label, sha, n_in_data = _load_data(args)
    settings = {"budget_factor": args.budget_factor, "max_per_turn": args.max_per_turn}
    env = MultiStepConflictEnv(budget_factor=args.budget_factor, max_per_turn=args.max_per_turn)

    summaries: Dict[str, Dict[str, Any]] = {}
    for name in names:
        rows = []
        for i, sc in enumerate(scenarios):
            ep_seed = args.seed + i
            bd, stats = run_episode(name, env, sc, ep_seed)
            rows.append({
                "scenario_id": sc.scenario_id, "difficulty": sc.difficulty, "episode_seed": ep_seed,
                "n_instructions": stats["n_instructions"], "turns": stats["turns"],
                "reverts": stats["reverts"], **bd,
            })
        summary = summarize(name, rows, label, sha, n_in_data, args.seed, settings)
        out = Path(args.out_dir) / f"multistep_{name}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
        with (out / "episodes.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        summaries[name] = summary
        print(f"{name:24s} reward {summary['composite']['mean']:.4f}  n={len(rows)}", file=sys.stderr)
    return summaries


if __name__ == "__main__":
    main()
