"""
ConflictBench — Cheat / non-ML baselines
Scores strategies that involve no learning at all, on a fixed scenario set.
If any of these score close to a trained model, the reward or the task is broken.

Usage: python scripts/baselines.py [--per-difficulty 200] [--seed 1234] [--out results/baselines.json]
"""

import argparse
import json
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Callable, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generator import SOURCE_PRIORITY, TEMPLATE_GROUPS, Scenario, ScenarioGenerator
from verifier import score

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _answer(conflicts: List[Dict], plan: List[str], overridden: List[str]) -> str:
    return json.dumps({
        "identified_conflicts": conflicts,
        "execution_plan": plan,
        "overridden_instructions": overridden,
    })


def _ids(sc: Scenario) -> List[str]:
    return [ins.id for ins in sc.instructions]


# Each template text becomes a regex with its {slots} as wildcards, so an
# instruction's topic can be recovered from its text alone.
_TEMPLATE_PATTERNS = [
    (re.compile("^" + re.sub(r"\\\{[a-z_]+\\\}", ".+?", re.escape(text)) + "$"), group_key)
    for group_key, group in TEMPLATE_GROUPS.items()
    for template in group["templates"]
    for text in template["texts"]
]

_RESTRICTIVE_WORDS = re.compile(
    r"frozen|suspend|paused|no production|locked|require|no purchase|limited|on-site|revoked|"
    r"in-office|prohibit|must not|no overtime|c-suite|without legal|restricted|revoke|reduced|"
    r"freeze|reduce|immediately|within \d+ hours|no exceptions|before|purge|delete|erasure",
    re.I,
)


def _group_by_topic(sc: Scenario) -> Dict[str, list]:
    groups: Dict[str, list] = {}
    for ins in sc.instructions:
        topic = next((g for p, g in _TEMPLATE_PATTERNS if p.match(ins.text)), ins.id)
        groups.setdefault(topic, []).append(ins)
    return groups


def _solve_by_topic(sc: Scenario, pick_winner: Callable) -> str:
    conflicts, plan, overridden = [], [], []
    for members in _group_by_topic(sc).values():
        if len(members) == 2:
            a, b = members
            winner = pick_winner(a, b)
            loser = b if winner is a else a
            conflicts.append({
                "instruction_a": a.id,
                "instruction_b": b.id,
                "conflict_type": "direct",
                "resolution": winner.id,
                "reasoning": "baseline",
            })
            plan.append(winner.id)
            overridden.append(loser.id)
        else:
            plan.extend(ins.id for ins in members)
    return _answer(conflicts, plan, overridden)


# ---------------------------------------------------------------------------
# Strategies — each takes (scenario, rng) and returns a raw completion string
# ---------------------------------------------------------------------------

def empty(sc, rng):
    return _answer([], [], [])


def follow_all(sc, rng):
    return _answer([], _ids(sc), [])


def override_all(sc, rng):
    return _answer([], [], _ids(sc))


def random_single(sc, rng):
    return _answer([], [rng.choice(_ids(sc))], [])


def random_half(sc, rng):
    ids = _ids(sc)
    plan = rng.sample(ids, len(ids) // 2)
    return _answer([], plan, [i for i in ids if i not in plan])


def hallucinated_ids(sc, rng):
    return _answer([], [f"INS-FAKE{n:02d}" for n in range(5)], [])


def keyword_restrictive(sc, rng):
    """Ignores authority entirely: the side that *sounds* strict wins."""
    return _solve_by_topic(sc, lambda a, b: a if _RESTRICTIVE_WORDS.search(a.text) else b)


def oracle_lookup(sc, rng):
    """Template match + tier lookup from the hierarchy printed in the prompt. No ML."""
    return _solve_by_topic(
        sc, lambda a, b: a if SOURCE_PRIORITY[a.source] < SOURCE_PRIORITY[b.source] else b
    )


def ground_truth(sc, rng):
    return _answer(
        [
            {
                "instruction_a": c.instruction_a_id,
                "instruction_b": c.instruction_b_id,
                "conflict_type": c.conflict_type,
                "resolution": c.resolution_id,
                "reasoning": c.explanation,
            }
            for c in sc.conflicts
        ],
        sc.ground_truth_followed,
        sc.ground_truth_overridden,
    )


def gt_one_conflict_flipped(sc, rng):
    """Ground truth, except one random conflict is resolved the wrong way round."""
    flipped = rng.choice(sc.conflicts)
    winner = flipped.resolution_id
    loser = flipped.instruction_b_id if winner == flipped.instruction_a_id else flipped.instruction_a_id
    swap = {winner: loser, loser: winner}
    return _answer(
        [
            {
                "instruction_a": c.instruction_a_id,
                "instruction_b": c.instruction_b_id,
                "conflict_type": c.conflict_type,
                "resolution": swap.get(c.resolution_id, c.resolution_id) if c is flipped else c.resolution_id,
                "reasoning": c.explanation,
            }
            for c in sc.conflicts
        ],
        [swap.get(i, i) for i in sc.ground_truth_followed],
        [swap.get(i, i) for i in sc.ground_truth_overridden],
    )


STRATEGIES: Dict[str, Callable] = {
    "empty": empty,
    "follow_all": follow_all,
    "override_all": override_all,
    "random_single": random_single,
    "random_half": random_half,
    "hallucinated_ids": hallucinated_ids,
    "keyword_restrictive": keyword_restrictive,
    "oracle_lookup": oracle_lookup,
    "ground_truth": ground_truth,
    "gt_one_conflict_flipped": gt_one_conflict_flipped,
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def build_scenarios(per_difficulty: int = 200, seed: int = 1234) -> List[Scenario]:
    gen = ScenarioGenerator(seed=seed)
    return [gen.generate(difficulty=d) for d in (1, 2, 3) for _ in range(per_difficulty)]


def evaluate(strategy: Callable, scenarios: List[Scenario], seed: int = 0) -> Dict:
    rng = random.Random(seed)
    by_diff: Dict[int, List[float]] = {}
    for sc in scenarios:
        by_diff.setdefault(sc.difficulty, []).append(score(strategy(sc, rng), sc).composite)
    every = [v for vals in by_diff.values() for v in vals]
    return {
        "mean": round(statistics.mean(every), 4),
        "std": round(statistics.pstdev(every), 4),
        "by_difficulty": {str(d): round(statistics.mean(v), 4) for d, v in sorted(by_diff.items())},
        "n": len(every),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-difficulty", type=int, default=200)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", type=str, default=str(ROOT / "results" / "baselines.json"))
    args = p.parse_args()

    scenarios = build_scenarios(args.per_difficulty, args.seed)
    results = {name: evaluate(fn, scenarios) for name, fn in STRATEGIES.items()}

    print(f"{'strategy':22s} {'mean':>6s} {'std':>6s}   d1     d2     d3")
    for name, r in results.items():
        d = r["by_difficulty"]
        print(f"{name:22s} {r['mean']:6.3f} {r['std']:6.3f}   {d['1']:.3f}  {d['2']:.3f}  {d['3']:.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "seed": args.seed,
        "per_difficulty": args.per_difficulty,
        "strategies": results,
    }, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
