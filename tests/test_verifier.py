"""
Unit tests for the Phase 2 scoring rules (FIX_BLUEPRINT.md P2-1..P2-6).
"""

import json
import math

import pytest

import verifier
from generator import ScenarioGenerator
from verifier import (
    RUBRIC_WEIGHTS,
    ScoreBreakdown,
    parse_agent_output,
    rubric_conflict_identification,
    rubric_correct_final_state,
    rubric_efficiency,
    rubric_format_compliance,
    rubric_no_contradictions,
    score,
)


@pytest.fixture(scope="module")
def sc():
    return ScenarioGenerator(seed=3).generate(difficulty=2)


def _conflict(c, resolution=None):
    return {
        "instruction_a": c.instruction_a_id,
        "instruction_b": c.instruction_b_id,
        "conflict_type": "direct",
        "resolution": c.resolution_id if resolution is None else resolution,
        "reasoning": "r",
    }


def _loser(c):
    return c.instruction_b_id if c.resolution_id == c.instruction_a_id else c.instruction_a_id


def _answer(conflicts, plan, overridden):
    return json.dumps({
        "identified_conflicts": conflicts,
        "execution_plan": plan,
        "overridden_instructions": overridden,
    })


def _gt(sc):
    return _answer([_conflict(c) for c in sc.conflicts], sc.ground_truth_followed, sc.ground_truth_overridden)


def _parsed(raw):
    return parse_agent_output(raw)


# ---------------------------------------------------------------------------
# Weights and breakdown shape
# ---------------------------------------------------------------------------

def test_weights_are_the_phase2_weights():
    assert RUBRIC_WEIGHTS == {
        "correct_final_state": 0.45,
        "conflict_identification": 0.30,
        "no_contradictions": 0.25,
    }
    assert math.isclose(sum(RUBRIC_WEIGHTS.values()), 1.0)


def test_breakdown_keeps_public_fields(sc):
    d = score(_gt(sc), sc).to_dict()
    for key in ("correct_final_state", "no_contradictions", "conflict_identification",
                "efficiency", "format_compliance", "composite", "format_ok"):
        assert key in d
    assert isinstance(score(_gt(sc), sc), ScoreBreakdown)


def test_ground_truth_scores_one_on_every_rubric(sc):
    r = score(_gt(sc), sc)
    assert (r.correct_final_state, r.no_contradictions, r.conflict_identification) == (1.0, 1.0, 1.0)
    assert r.composite == 1.0 and r.format_ok and r.format_compliance == 1.0 and r.efficiency == 1.0


def test_composite_is_weighted_sum_of_rubrics(sc):
    c0 = sc.conflicts[0]
    plan = [i for i in sc.ground_truth_followed if i != c0.resolution_id] + [_loser(c0)]
    raw = _answer([_conflict(c) for c in sc.conflicts[1:]], plan, sc.ground_truth_overridden)
    r = score(raw, sc)
    expected = sum(RUBRIC_WEIGHTS[k] * getattr(r, k) for k in RUBRIC_WEIGHTS)
    assert r.composite == pytest.approx(expected, abs=1e-4)


def test_efficiency_is_not_in_the_composite(sc):
    assert "efficiency" not in RUBRIC_WEIGHTS and "format_compliance" not in RUBRIC_WEIGHTS


# ---------------------------------------------------------------------------
# P2-4 format gate
# ---------------------------------------------------------------------------

def test_unparsed_output_scores_zero(sc):
    r = score("I think the CEO wins.", sc)
    assert r.composite == 0.0 and not r.format_ok and r.format_compliance == 0.0


def test_schema_errors_cost_ten_percent(sc):
    data = json.loads(_gt(sc))
    data["execution_plan"].append(None)
    r = score(json.dumps(data), sc)
    assert r.format_ok and r.schema_errors
    assert r.format_compliance == 0.9
    assert r.composite == pytest.approx(0.9)


def test_format_rubric_is_gate_multiplier(sc):
    assert rubric_format_compliance(_parsed("nope"), sc) == 0.0
    assert rubric_format_compliance(_parsed(_gt(sc)), sc) == 1.0


# ---------------------------------------------------------------------------
# P2-1 no contradictions
# ---------------------------------------------------------------------------

def test_empty_plan_has_no_consistency_credit(sc):
    assert rubric_no_contradictions(_parsed(_answer([], [], [])), sc) == 0.0


def test_both_sides_in_plan_is_a_contradiction(sc):
    plan = [i.id for i in sc.instructions]
    assert rubric_no_contradictions(_parsed(_answer([_conflict(c) for c in sc.conflicts], plan, [])), sc) == 0.0


def test_consistent_wrong_choice_still_counts_as_consistent(sc):
    conflicts = [_conflict(c, resolution=_loser(c)) for c in sc.conflicts]
    plan = [_loser(c) for c in sc.conflicts]
    assert rubric_no_contradictions(_parsed(_answer(conflicts, plan, [])), sc) == 1.0


def test_unstated_choice_gets_no_consistency_credit(sc):
    # Exactly one side in the plan, but the clash was never declared.
    assert rubric_no_contradictions(_parsed(_answer([], sc.ground_truth_followed, [])), sc) == 0.0


def test_stated_winner_must_match_plan(sc):
    conflicts = [_conflict(c, resolution=_loser(c)) for c in sc.conflicts]  # says loser wins
    r = rubric_no_contradictions(_parsed(_answer(conflicts, sc.ground_truth_followed, [])), sc)
    assert r == 0.0


# ---------------------------------------------------------------------------
# P2-2 correct final state
# ---------------------------------------------------------------------------

def test_follow_all_loses_about_half(sc):
    r = rubric_correct_final_state(_parsed(_answer([], [i.id for i in sc.instructions], [])), sc)
    assert 0.3 < r < 0.5


def test_overridden_list_is_scored(sc):
    without = rubric_correct_final_state(_parsed(_answer([], sc.ground_truth_followed, [])), sc)
    assert without == 0.5


def test_overlap_is_penalised(sc):
    extra = sc.ground_truth_followed[0]
    raw = _answer([], sc.ground_truth_followed, sc.ground_truth_overridden + [extra])
    n = len(sc.instructions)
    gt_o = len(sc.ground_truth_overridden)
    overridden_f1 = 2 * gt_o / (2 * gt_o + 1)
    expected = (1.0 + overridden_f1) / 2 * (1 - 1 / n)
    assert rubric_correct_final_state(_parsed(raw), sc) == pytest.approx(expected, abs=1e-4)


def test_hallucinated_id_is_false_positive_in_both_lists(sc):
    raw = _answer([], sc.ground_truth_followed + ["INS-ZZZZZZ"], sc.ground_truth_overridden)
    f = len(sc.ground_truth_followed)
    o = len(sc.ground_truth_overridden)
    expected = (2 * f / (2 * f + 1) + 2 * o / (2 * o + 1)) / 2
    assert rubric_correct_final_state(_parsed(raw), sc) == pytest.approx(expected, abs=1e-4)


def test_hallucinated_only_scores_zero(sc):
    raw = _answer([], ["INS-FAKE01", "INS-FAKE02"], ["INS-FAKE03"])
    assert score(raw, sc).composite == 0.0


# ---------------------------------------------------------------------------
# P2-3 conflict identification
# ---------------------------------------------------------------------------

def test_pair_order_does_not_matter(sc):
    flipped = [dict(_conflict(c), instruction_a=c.instruction_b_id, instruction_b=c.instruction_a_id)
               for c in sc.conflicts]
    assert rubric_conflict_identification(_parsed(_answer(flipped, [], [])), sc) == 1.0


def test_duplicate_pair_counts_once(sc):
    conflicts = [_conflict(c) for c in sc.conflicts] + [_conflict(sc.conflicts[0])]
    assert rubric_conflict_identification(_parsed(_answer(conflicts, [], [])), sc) == 1.0


def test_hedging_with_both_resolutions_is_wrong(sc):
    c0 = sc.conflicts[0]
    conflicts = [_conflict(c) for c in sc.conflicts] + [_conflict(c0, resolution=_loser(c0))]
    r = rubric_conflict_identification(_parsed(_answer(conflicts, [], [])), sc)
    assert r == pytest.approx(1 - 1 / len(sc.conflicts), abs=1e-4)


def test_resolution_outside_pair_is_wrong(sc):
    conflicts = [_conflict(c, resolution="INS-NOPE00") for c in sc.conflicts]
    assert rubric_conflict_identification(_parsed(_answer(conflicts, [], [])), sc) == 0.0


def test_identification_is_f1_times_accuracy(sc):
    n = len(sc.conflicts)
    c0 = sc.conflicts[0]
    # All pairs found, one resolved wrong -> F1 1 x accuracy (n-1)/n.
    conflicts = [_conflict(c0, resolution=_loser(c0))] + [_conflict(c) for c in sc.conflicts[1:]]
    r = rubric_conflict_identification(_parsed(_answer(conflicts, [], [])), sc)
    assert r == pytest.approx((n - 1) / n, abs=1e-4)
    # Half the pairs found, all right -> F1 only.
    half = [_conflict(c) for c in sc.conflicts[: n // 2]]
    r = rubric_conflict_identification(_parsed(_answer(half, [], [])), sc)
    k = n // 2
    assert r == pytest.approx(2 * k / (k + n), abs=1e-4)


# ---------------------------------------------------------------------------
# Rubric functions stay importable for conflict_bench.py
# ---------------------------------------------------------------------------

def test_rubric_functions_exist_and_return_unit_floats(sc):
    out = _parsed(_gt(sc))
    for fn in (rubric_correct_final_state, rubric_no_contradictions, rubric_conflict_identification,
               rubric_efficiency, rubric_format_compliance):
        assert 0.0 <= fn(out, sc) <= 1.0
    assert hasattr(verifier, "RUBRIC_WEIGHTS")
