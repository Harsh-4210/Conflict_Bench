"""
P3-7: the "not a lookup" test.

On the held-out test split with the default (Phase 3) generator, scripts with
no reasoning must fall clearly short, while applying every rule (resolve())
scores 1.0. The legacy generator is checked too, to show what changed.

Two families of shortcut are measured:
  - the original scripts (keyword_restrictive, oracle_lookup_v1), which find
    topics by matching the legacy templates and so fail at detection already;
  - "structured" shortcuts that are handed perfect topic/scope detection from
    the hidden fields and only have to pick winners (keyword_restrictive_v2,
    oracle_lookup, latest_wins). These are the honest measure of how much the
    rules (notices, dates, conditions, chains) matter.
"""

import pytest

from generator import GeneratorConfig, ScenarioGenerator
from scripts.baselines import STRATEGIES, build_scenarios, evaluate


@pytest.fixture(scope="module")
def test_split():
    return build_scenarios(per_difficulty=100, seed=1234, config=GeneratorConfig(split="test"),
                           difficulties=(1, 2, 3, 4, 5))


@pytest.fixture(scope="module")
def legacy():
    return build_scenarios(per_difficulty=50, seed=1234, config=GeneratorConfig(legacy=True))


def _mean(name, scenarios):
    return evaluate(STRATEGIES[name], scenarios)["mean"]


# ---- the P3-7 thresholds -----------------------------------------------------

def test_keyword_shortcut_fails_on_test_split(test_split):
    assert _mean("keyword_restrictive", test_split) <= 0.35


def test_old_lookup_solver_fails_on_test_split(test_split):
    assert _mean("oracle_lookup_v1", test_split) <= 0.6


def test_resolver_is_perfect_on_test_split(test_split):
    assert _mean("resolver", test_split) == 1.0
    assert _mean("ground_truth", test_split) == 1.0


# ---- shortcuts with perfect conflict detection --------------------------------

def test_perfect_keyword_detector_is_far_from_solving(test_split):
    """Even knowing every clash and which side is restrictive: about chance on who wins."""
    assert _mean("keyword_restrictive_v2", test_split) <= 0.75


def test_seniority_lookup_with_perfect_detection_is_not_enough(test_split):
    """Reads the true tier of every sender (hidden in the prompt) and still misses the rules."""
    assert _mean("oracle_lookup", test_split) <= 0.9


def test_latest_date_wins_is_not_enough(test_split):
    assert _mean("latest_wins", test_split) <= 0.8


def test_restrictive_side_wins_about_half_on_test_split(test_split):
    wins = total = 0
    for sc in test_split:
        by_id = {ins.id: ins for ins in sc.instructions}
        for c in sc.conflicts:
            wins += by_id[c.resolution_id].polarity == "restrictive"
            total += 1
    assert 0.45 <= wins / total <= 0.55


def test_trivial_strategies_stay_low(test_split):
    for name in ("empty", "follow_all", "override_all", "random_half", "hallucinated_ids"):
        assert _mean(name, test_split) <= 0.35, name


# ---- before/after: the legacy task was a lookup ------------------------------

def test_legacy_task_was_a_lookup(legacy):
    assert _mean("oracle_lookup_v1", legacy) == 1.0
    assert _mean("keyword_restrictive", legacy) >= 0.85
    assert _mean("resolver", legacy) == 1.0


def test_test_split_scenarios_are_marked():
    sc = ScenarioGenerator(seed=0, config=GeneratorConfig(split="test")).generate(2)
    assert sc.split == "test" and sc.generator_version == "v2"
