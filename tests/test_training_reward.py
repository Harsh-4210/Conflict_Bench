"""Training reward: must equal verifier.score exactly, expose per-rubric logging functions, fail loudly."""

import json
import logging

import pytest

import verifier
from generator import ScenarioGenerator
from scenario_io import dumps_scenario
from training.train_grpo import (MAX_CONSECUTIVE_REWARD_FAILURES, RewardFailure, RewardScorer,
                                 build_reward_functions, completion_text)


def perfect_answer(sc) -> str:
    return json.dumps({
        "identified_conflicts": [
            {"instruction_a": c.instruction_a_id, "instruction_b": c.instruction_b_id,
             "conflict_type": c.conflict_type, "resolution": c.resolution_id, "reasoning": "higher authority wins"}
            for c in sc.conflicts],
        "execution_plan": list(sc.ground_truth_followed),
        "overridden_instructions": list(sc.ground_truth_overridden),
    })


def wrong_answer(sc) -> str:
    return json.dumps({"identified_conflicts": [], "execution_plan": list(sc.ground_truth_overridden),
                       "overridden_instructions": list(sc.ground_truth_followed)})


@pytest.fixture
def cases():
    gen = ScenarioGenerator(seed=7)
    out = []
    for d in (1, 2, 3):
        sc = gen.generate(difficulty=d)
        for text in (perfect_answer(sc), wrong_answer(sc), "not json at all", "", '{"execution_plan": ["x"]}'):
            out.append((sc, text))
    return out


def test_composite_matches_verifier_score_exactly(cases):
    funcs, weights, _ = build_reward_functions()
    composite = funcs[0]
    completions = [t for _, t in cases]
    got = composite(completions=completions, scenario_json=[dumps_scenario(s) for s, _ in cases])
    expected = [verifier.score(t, s).composite for s, t in cases]
    assert got == expected                      # exact, not approximate: no re-weighting, no bonus
    assert max(got) == 1.0                      # sanity: a perfect answer really scores 1.0
    assert min(got) < 1.0


def test_no_difficulty_bonus_or_rounding(cases):
    funcs, _, _ = build_reward_functions()
    sc = ScenarioGenerator(seed=3).generate(difficulty=2)
    text = perfect_answer(sc)
    assert funcs[0](completions=[text], scenario_json=[dumps_scenario(sc)]) == [verifier.score(text, sc).composite]


def test_chat_style_completions_are_scored_like_strings(cases):
    funcs, _, _ = build_reward_functions()
    sc, text = cases[0]
    chat = [[{"role": "assistant", "content": text}]]
    assert funcs[0](completions=chat, scenario_json=[dumps_scenario(sc)]) == \
        funcs[0](completions=[text], scenario_json=[dumps_scenario(sc)])


def test_completion_text_variants():
    assert completion_text("abc") == "abc"
    assert completion_text([{"role": "assistant", "content": "x"}]) == "x"
    assert completion_text({"role": "assistant", "content": "y"}) == "y"
    assert completion_text([{"role": "assistant", "content": [{"type": "text", "text": "z"}]}]) == "z"
    assert completion_text(None) == ""


def test_one_function_per_rubric_plus_composite(cases):
    funcs, weights, _ = build_reward_functions()
    names = [f.__name__ for f in funcs]
    assert names == ["reward_composite"] + [f"reward_{n}" for n in verifier.RUBRIC_WEIGHTS]
    assert len(set(names)) == len(names)
    assert set(verifier.RUBRIC_WEIGHTS) == {"correct_final_state", "no_contradictions", "conflict_identification"}

    completions = [t for _, t in cases]
    sjs = [dumps_scenario(s) for s, _ in cases]
    for func, rubric in zip(funcs[1:], verifier.RUBRIC_WEIGHTS):
        assert func(completions=completions, scenario_json=sjs) == \
            [getattr(verifier.score(t, s), rubric) for s, t in cases]


def test_weights_make_total_equal_composite(cases):
    funcs, weights, _ = build_reward_functions()
    assert len(weights) == len(funcs)
    assert weights == [1.0] + [0.0] * len(verifier.RUBRIC_WEIGHTS)
    completions = [t for _, t in cases]
    sjs = [dumps_scenario(s) for s, _ in cases]
    per_func = [f(completions=completions, scenario_json=sjs) for f in funcs]
    totals = [sum(w * col[i] for w, col in zip(weights, per_func)) for i in range(len(cases))]
    assert totals == per_func[0]


def test_each_completion_is_scored_once(monkeypatch, cases):
    calls = []
    real = verifier.score
    monkeypatch.setattr(verifier, "score", lambda raw, sc: calls.append(1) or real(raw, sc))
    funcs, _, _ = build_reward_functions()
    completions = [t for _, t in cases]
    sjs = [dumps_scenario(s) for s, _ in cases]
    for f in funcs:
        f(completions=completions, scenario_json=sjs)
    assert len(calls) == len(set(zip(completions, sjs)))


def test_extra_trl_kwargs_are_accepted(cases):
    funcs, _, _ = build_reward_functions()
    sc, text = cases[0]
    metrics = []
    out = funcs[0](prompts=["p"], completions=[text], completion_ids=[[1]], scenario_json=[dumps_scenario(sc)],
                   difficulty=[sc.difficulty], trainer_state=object(), log_extra=None,
                   log_metric=lambda name, value: metrics.append((name, value)))
    assert out == [verifier.score(text, sc).composite]
    assert metrics == [("reward_errors_total", 0.0)]


def test_log_metric_hook_errors_never_break_training(cases):
    funcs, _, _ = build_reward_functions()
    sc, text = cases[0]

    def broken(*a, **k):
        raise TypeError("different signature")

    assert funcs[0](completions=[text], scenario_json=[dumps_scenario(sc)], log_metric=broken)


# ---- failure behaviour ------------------------------------------------------------------------------------------

def _boom(raw, scenario):
    raise RuntimeError("verifier exploded")


def test_failures_are_zero_reward_logged_and_counted(monkeypatch, caplog, cases):
    monkeypatch.setattr(verifier, "score", _boom)
    funcs, _, scorer = build_reward_functions()
    sc = cases[0][0]
    long_text = "x" * 500
    with caplog.at_level(logging.WARNING, logger="conflictbench.train"):
        out = funcs[0](completions=[long_text, "b"], scenario_json=[dumps_scenario(sc)] * 2)
    assert out == [0.0, 0.0]
    assert scorer.failures == 2
    warnings_ = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings_) == 2
    assert "verifier exploded" in warnings_[0].getMessage()
    assert "x" * 200 in warnings_[0].getMessage() and "x" * 201 not in warnings_[0].getMessage()


def test_raises_after_ten_consecutive_failures(monkeypatch, cases):
    monkeypatch.setattr(verifier, "score", _boom)
    funcs, _, scorer = build_reward_functions()
    sc = cases[0][0]
    sj = dumps_scenario(sc)
    assert MAX_CONSECUTIVE_REWARD_FAILURES == 10
    # nine in a row are tolerated (distinct texts so the cache does not hide them) ...
    assert funcs[0](completions=[f"t{i}" for i in range(9)], scenario_json=[sj] * 9) == [0.0] * 9
    # ... the tenth raises, chained to the original error
    with pytest.raises(RewardFailure) as exc:
        funcs[0](completions=["t9"], scenario_json=[sj])
    assert isinstance(exc.value.__cause__, RuntimeError)
    assert scorer.failures == 10


def test_a_success_resets_the_consecutive_counter(monkeypatch, cases):
    sc = cases[0][0]
    sj = dumps_scenario(sc)
    real = verifier.score
    state = {"fail": True}

    def flaky(raw, scenario):
        if state["fail"]:
            raise RuntimeError("flaky")
        return real(raw, scenario)

    monkeypatch.setattr(verifier, "score", flaky)
    funcs, _, scorer = build_reward_functions()
    for _round in range(3):                                   # 9 failures, 1 success, repeated: never 10 in a row
        state["fail"] = True
        funcs[0](completions=[f"f{_round}-{i}" for i in range(9)], scenario_json=[sj] * 9)
        state["fail"] = False
        funcs[0](completions=[f"ok{_round}"], scenario_json=[sj])
        assert scorer.consecutive_failures == 0
    assert scorer.failures == 27


def test_missing_scenario_json_is_a_counted_failure(caplog):
    funcs, _, scorer = build_reward_functions()
    with caplog.at_level(logging.WARNING, logger="conflictbench.train"):
        assert funcs[0](completions=["{}"], scenario_json=[None]) == [0.0]
    assert scorer.failures == 1


def test_custom_failure_limit():
    scorer = RewardScorer(max_consecutive_failures=2)
    scorer.score("a", None)
    with pytest.raises(RewardFailure):
        scorer.score("b", None)
