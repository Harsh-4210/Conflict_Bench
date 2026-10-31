"""
Reward-hacking regression tests.

The thresholds are the Phase 2 targets from FIX_BLUEPRINT.md (P2-7) and are now
enforced. The keyword shortcut stays xfail(strict=True) until Phase 3 makes the
task itself harder: it needs a harder task, not a stricter reward.
"""

import json
import random

import pytest

from generator import ScenarioGenerator
from scripts.baselines import STRATEGIES, build_scenarios, evaluate
from verifier import score


@pytest.fixture(scope="module")
def scenarios():
    return build_scenarios(per_difficulty=50, seed=1234)


def _mean(name, scenarios):
    return evaluate(STRATEGIES[name], scenarios)["mean"]


def test_ground_truth_is_perfect(scenarios):
    assert _mean("ground_truth", scenarios) == 1.0


def test_empty_answer_scores_near_zero(scenarios):
    assert _mean("empty", scenarios) <= 0.05


def test_follow_all_is_penalised(scenarios):
    assert _mean("follow_all", scenarios) <= 0.35


def test_override_all_is_penalised(scenarios):
    assert _mean("override_all", scenarios) <= 0.25


def test_random_half_is_penalised(scenarios):
    assert _mean("random_half", scenarios) <= 0.35


def test_hallucinated_ids_score_near_zero(scenarios):
    assert _mean("hallucinated_ids", scenarios) <= 0.05


def test_one_flipped_conflict_gets_partial_credit(scenarios):
    d1 = evaluate(STRATEGIES["gt_one_conflict_flipped"], scenarios)["by_difficulty"]["1"]
    assert 0.4 <= d1 <= 0.85


@pytest.mark.xfail(strict=True, reason="Phase 3 (P3-1..P3-7) not done yet: task is a lookup")
def test_keyword_shortcut_does_not_solve_task(scenarios):
    assert _mean("keyword_restrictive", scenarios) <= 0.35


# ---------------------------------------------------------------------------
# Monotonicity: starting from the GT answer, every extra error never raises
# the score. Each edit touches a fresh item, so errors only accumulate.
# ---------------------------------------------------------------------------

def _loser(c):
    return c.instruction_b_id if c.resolution_id == c.instruction_a_id else c.instruction_a_id


class _Answer:
    def __init__(self, sc):
        self.conflicts = {
            id(c): {
                "instruction_a": c.instruction_a_id,
                "instruction_b": c.instruction_b_id,
                "conflict_type": "direct",
                "resolution": c.resolution_id,
                "reasoning": "r",
            }
            for c in sc.conflicts
        }
        self.plan = list(sc.ground_truth_followed)
        self.overridden = list(sc.ground_truth_overridden)
        self.extra_conflicts = []
        self.junk = []  # schema-error items appended to the plan

    def raw(self):
        return json.dumps({
            "identified_conflicts": list(self.conflicts.values()) + self.extra_conflicts,
            "execution_plan": self.plan + self.junk,
            "overridden_instructions": self.overridden,
        })


def _edits(sc, answer, rng):
    """Yield callables; each adds exactly one new error to `answer`."""
    winners = {c.resolution_id for c in sc.conflicts}
    fillers = [i for i in sc.ground_truth_followed if i not in winners]
    conflicts = list(sc.conflicts)
    rng.shuffle(conflicts)
    rng.shuffle(fillers)
    fake_ids = (f"INS-FAKE{n:02d}" for n in range(100))

    def flip(c):
        w, l = c.resolution_id, _loser(c)
        answer.plan[answer.plan.index(w)] = l
        answer.overridden[answer.overridden.index(l)] = w
        answer.conflicts[id(c)]["resolution"] = l

    def wrong_resolution(c):
        answer.conflicts[id(c)]["resolution"] = _loser(c)

    def drop_entry(c):
        del answer.conflicts[id(c)]

    def drop_filler(f):
        answer.plan.remove(f)

    def filler_to_overridden(f):
        answer.plan.remove(f)
        answer.overridden.append(f)

    def filler_also_overridden(f):
        answer.overridden.append(f)

    def hallucinate():
        (answer.plan if rng.random() < 0.5 else answer.overridden).append(next(fake_ids))

    def spurious_pair():
        a, b = rng.sample([i.id for i in sc.instructions], 2)
        answer.extra_conflicts.append({"instruction_a": a, "instruction_b": b, "conflict_type": "direct",
                                       "resolution": a, "reasoning": "r"})

    def schema_junk():
        answer.junk.append(None)

    item_edits = []
    for c in conflicts:
        item_edits.append(lambda c=c: rng.choice([flip, wrong_resolution, drop_entry])(c))
    for f in fillers:
        item_edits.append(lambda f=f: rng.choice([drop_filler, filler_to_overridden, filler_also_overridden])(f))
    free_edits = [hallucinate, spurious_pair, schema_junk]

    while True:
        if item_edits and rng.random() < 0.7:
            yield item_edits.pop(rng.randrange(len(item_edits)))
        else:
            yield rng.choice(free_edits)


def test_more_errors_never_score_higher():
    rng = random.Random(97)
    gen = ScenarioGenerator(seed=4321)
    for i in range(500):
        sc = gen.generate(difficulty=1 + i % 3)
        answer = _Answer(sc)
        previous = score(answer.raw(), sc).composite
        assert previous == 1.0
        edits = _edits(sc, answer, rng)
        for step in range(rng.randint(1, 8)):
            next(edits)()
            current = score(answer.raw(), sc).composite
            assert current <= previous, (i, step, answer.raw())
            if step == 0:
                assert current < 1.0  # any single error costs something
            previous = current
