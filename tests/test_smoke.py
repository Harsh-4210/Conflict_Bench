"""
Smoke tests — pin the current behaviour of the environment and verifier.
These must keep passing through every later phase.
"""

import json

import pytest

from conflict_bench import ConflictBenchEnv
from generator import ScenarioGenerator
from verifier import score


@pytest.mark.parametrize("difficulty", [1, 2, 3])
def test_ground_truth_scores_perfect(difficulty):
    env = ConflictBenchEnv(auto_scale_difficulty=False, fixed_difficulty=difficulty, seed=difficulty)
    for _ in range(50):
        env.reset()
        _, reward, done, _ = env.step(env.sample_action())
        assert done is True
        assert reward == 1.0


@pytest.mark.parametrize("raw", [
    "",
    "garbage",
    "not json at all",
    "{",
    pytest.param("[]", marks=pytest.mark.xfail(
        strict=True, raises=AttributeError,
        reason="P1-1: a top-level JSON list crashes parse_agent_output (data.keys())",
    )),
])
def test_unparseable_output_scores_zero(raw):
    sc = ScenarioGenerator(seed=0).generate(difficulty=1)
    assert score(raw, sc).composite == 0.0


def test_composite_always_in_unit_range():
    gen = ScenarioGenerator(seed=5)
    for d in (1, 2, 3):
        for _ in range(30):
            sc = gen.generate(difficulty=d)
            ids = [i.id for i in sc.instructions]
            for plan in ([], ids, ids[:3], sc.ground_truth_followed):
                raw = json.dumps({
                    "identified_conflicts": [],
                    "execution_plan": plan,
                    "overridden_instructions": [],
                })
                assert 0.0 <= score(raw, sc).composite <= 1.0


def test_step_before_reset_raises():
    env = ConflictBenchEnv(seed=0)
    with pytest.raises(RuntimeError):
        env.step("{}")
