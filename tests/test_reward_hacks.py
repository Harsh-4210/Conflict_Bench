"""
Reward-hacking regression tests.

The thresholds are the Phase 2 targets from FIX_BLUEPRINT.md (P2-7). The current
verifier fails them, so they are marked xfail(strict=True): once Phase 2 fixes the
reward they will XPASS, strict mode turns that into a failure, and the xfail marks
must then be removed so the thresholds are enforced from then on.
"""

import pytest

from scripts.baselines import STRATEGIES, build_scenarios, evaluate

PHASE2 = "Phase 2 (P2-1..P2-6) not done yet: current verifier rewards this strategy"


@pytest.fixture(scope="module")
def scenarios():
    return build_scenarios(per_difficulty=50, seed=1234)


def _mean(name, scenarios):
    return evaluate(STRATEGIES[name], scenarios)["mean"]


def test_ground_truth_is_perfect(scenarios):
    assert _mean("ground_truth", scenarios) == 1.0


@pytest.mark.xfail(strict=True, reason=PHASE2)
def test_empty_answer_scores_near_zero(scenarios):
    assert _mean("empty", scenarios) <= 0.05


@pytest.mark.xfail(strict=True, reason=PHASE2)
def test_follow_all_is_penalised(scenarios):
    assert _mean("follow_all", scenarios) <= 0.35


@pytest.mark.xfail(strict=True, reason=PHASE2)
def test_override_all_is_penalised(scenarios):
    assert _mean("override_all", scenarios) <= 0.25


@pytest.mark.xfail(strict=True, reason=PHASE2)
def test_random_half_is_penalised(scenarios):
    assert _mean("random_half", scenarios) <= 0.35


@pytest.mark.xfail(strict=True, reason=PHASE2)
def test_hallucinated_ids_score_near_zero(scenarios):
    assert _mean("hallucinated_ids", scenarios) <= 0.05


@pytest.mark.xfail(strict=True, reason="Phase 3 (P3-1..P3-7) not done yet: task is a lookup")
def test_keyword_shortcut_does_not_solve_task(scenarios):
    assert _mean("keyword_restrictive", scenarios) <= 0.35
