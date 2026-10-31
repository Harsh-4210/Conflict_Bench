"""Tests for the gym-style ConflictBenchEnv (P1-5)."""

import json

import pytest

import conflict_bench
from conflict_bench import ConflictBenchEnv


def test_sample_action_is_json_string():
    env = ConflictBenchEnv(auto_scale_difficulty=False, fixed_difficulty=1, seed=0)
    env.reset()
    action = env.sample_action()
    assert isinstance(action, str)
    plan = json.loads(action)
    assert set(plan) == {"identified_conflicts", "execution_plan", "overridden_instructions"}


def test_ground_truth_action_scores_one():
    env = ConflictBenchEnv(auto_scale_difficulty=False, fixed_difficulty=2, seed=1)
    env.reset()
    _, reward, done, info = env.step(env.sample_action())
    assert reward == 1.0
    assert done is True
    assert info["episode_count"] == 1


def test_step_after_done_raises_and_does_not_rescore():
    env = ConflictBenchEnv(auto_scale_difficulty=False, fixed_difficulty=1, seed=2)
    env.reset()
    env.step(env.sample_action())
    with pytest.raises(RuntimeError, match="reset"):
        env.step(env.sample_action())
    assert env._episode_count == 1  # second step must not bump the counter

    env.reset()  # a new episode works again
    _, reward, _, _ = env.step(env.sample_action())
    assert reward == 1.0
    assert env._episode_count == 2


def test_step_before_reset_raises():
    with pytest.raises(RuntimeError):
        ConflictBenchEnv(seed=0).step("{}")


def test_reset_forwards_seed_to_a_fresh_generator(monkeypatch):
    seeds = []
    real = conflict_bench.ScenarioGenerator

    def recording_generator(seed=None):
        seeds.append(seed)
        return real(seed=seed)

    monkeypatch.setattr(conflict_bench, "ScenarioGenerator", recording_generator)
    env = ConflictBenchEnv(auto_scale_difficulty=False, fixed_difficulty=1)
    assert seeds == [None]  # constructor
    first_generator = env.generator
    env.reset(seed=123)
    assert seeds == [None, 123]
    assert env.generator is not first_generator
    env.reset()  # no seed: keep the current generator, do not recreate
    assert seeds == [None, 123]


def test_fake_rubric_stubs_are_gone():
    assert not hasattr(conflict_bench, "Rubric")
    assert not hasattr(conflict_bench, "RubricSet")
    assert not hasattr(ConflictBenchEnv, "_build_rubric_set")
