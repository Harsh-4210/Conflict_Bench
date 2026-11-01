"""
Tests for the multi-step environment core (multistep/env.py, state.py, scoring.py).

Most tests use tiny hand-built scenarios with max_per_turn=1 and distinct dates, so the
arrival order is the date order and every expected reward can be worked out by hand.
"""

import json

import pytest

from generator import SOURCE_PRIORITY, Instruction, Scenario
from multistep.env import MultiStepConflictEnv
from multistep.scoring import final_reward, state_accuracy
from multistep.state import CompanyState, build_schedule, cell_of, target_cells
from resolver import resolve
from scripts.evaluate_multistep import STRATEGIES, run_episode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ins(iid, source, key, value, date, scope="all", condition=None, depends_on=None):
    return Instruction(
        id=iid, text=f"{key}={value} ({iid})", source=source, source_priority=SOURCE_PRIORITY[source],
        instruction_type="absolute", action_key=key, action_value=value, condition=condition,
        action_scope=scope, author="Pat Doe", title=source, effective_date=date, depends_on=depends_on,
    )


def scenario(instructions, facts=None):
    return Scenario(
        scenario_id="hand", domain="HR", difficulty=1, business_context="ctx", instructions=list(instructions),
        conflicts=[], ground_truth_followed=[], ground_truth_overridden=[], prompt="", facts=facts or {},
    )


def play(env, sc, actions, seed=0):
    """Reset on `sc`, send the actions, return (last obs, last reward, done, last info)."""
    env.reset(seed=seed, scenario=sc)
    out = None
    for a in actions:
        out = env.step(a)
        if out[2]:
            break
    return out


def apply(i):
    return {"tool": "apply", "id": i}


def revert(i):
    return {"tool": "revert", "id": i}


WAIT = {"tool": "wait"}
FINISH = {"tool": "finish"}

# Low (Team Lead) vs high (CEO Office) authority on one topic. LOW arrives first.
LOW_FIRST = [ins("LOW", "Team Lead", "hiring_status", "active", "2026-01-01"),
             ins("HIGH", "CEO Office", "hiring_status", "frozen", "2026-01-02")]
# HIGH arrives first.
HIGH_FIRST = [ins("HIGH", "CEO Office", "hiring_status", "frozen", "2026-01-01"),
              ins("LOW", "Team Lead", "hiring_status", "active", "2026-01-02")]


@pytest.fixture
def env():
    return MultiStepConflictEnv(max_per_turn=1)


# ---------------------------------------------------------------------------
# Determinism and the observation
# ---------------------------------------------------------------------------

def _transcript(seed, difficulty=3):
    e = MultiStepConflictEnv()
    obs = e.reset(seed=seed, difficulty=difficulty)
    log = [obs]
    while True:
        done = False
        for a in ([apply(m["id"]) for m in obs["new_messages"]] + [WAIT if obs["more_messages"] else FINISH]):
            obs, _, done, _ = e.step(a)
            log.append(obs)
            if done:
                break
        if done:
            return log


def test_same_seed_same_episode():
    assert _transcript(21) == _transcript(21)


def test_different_seed_different_episode():
    a, b = _transcript(21), _transcript(22)
    assert a[0]["scenario_id"] != b[0]["scenario_id"] or a[0]["new_messages"] != b[0]["new_messages"]


def test_same_seed_two_instances_same_schedule_and_truth():
    e1, e2 = MultiStepConflictEnv(), MultiStepConflictEnv()
    e1.reset(seed=5, difficulty=2)
    e2.reset(seed=5, difficulty=2)
    assert e1.schedule == e2.schedule
    assert e1.ground_truth() == e2.ground_truth()


def test_reset_without_seed_follows_the_env_seed():
    a, b = MultiStepConflictEnv(seed=9), MultiStepConflictEnv(seed=9)
    assert a.reset()["scenario_id"] == b.reset()["scenario_id"]
    assert a.reset()["scenario_id"] == b.reset()["scenario_id"]


def test_schedule_covers_every_message_once_in_date_order(env):
    e = MultiStepConflictEnv()
    e.reset(seed=3, difficulty=3)
    order = list(e.schedule.order)
    assert sorted(order) == sorted(i.id for i in e.scenario.instructions)
    dates = [next(i for i in e.scenario.instructions if i.id == x).effective_date for x in order]
    assert dates == sorted(dates)
    assert all(1 <= len(t.message_ids) <= 3 for t in e.schedule.turns[:-1])
    assert 1 <= len(e.schedule.turns[-1].message_ids) <= 3
    assert all(f in e.scenario.facts for t in e.schedule.turns for f in t.fact_keys)
    announced = [f for t in e.schedule.turns for f in t.fact_keys]
    assert sorted(announced) == sorted(e.scenario.facts)


def test_build_schedule_is_a_pure_function_of_seed():
    sc = scenario(LOW_FIRST + [ins(f"X{i}", "Team Lead", f"t{i}", "v", f"2026-02-{i + 1:02d}") for i in range(8)])
    assert build_schedule(sc, 4) == build_schedule(sc, 4)
    assert any(build_schedule(sc, s) != build_schedule(sc, 4) for s in range(5, 15))


def test_observations_never_contain_ground_truth():
    e = MultiStepConflictEnv()
    obs = e.reset(seed=8, difficulty=2)
    truth = e.ground_truth()
    overridden = set(truth["overridden"])
    seen = []
    done = False
    while not done:
        seen.append(obs)
        action = apply(obs["new_messages"][0]["id"]) if obs["new_messages"] else (WAIT if obs["more_messages"] else FINISH)
        obs, _, done, info = e.step(action)
    seen.append(obs)
    message_keys = {"id", "from", "title", "date", "text", "arrived_turn"}
    for o in seen:
        assert "ground_truth" not in o and "target_state" not in o and "followed" not in o
        assert all(set(m) == message_keys for m in o["new_messages"])   # no tier, polarity, verdict
        assert all(set(m) == {"id", "status"} for m in o["inbox"])
        json.dumps(o)                                                   # plain JSON data
    assert all(o["reward_breakdown"] is None for o in seen[:-1])
    assert "ground_truth" in info  # for trusted harnesses only; the OpenEnv wrapper hides it
    assert overridden  # the scenario really has something to get wrong


def test_first_observation_shape(env):
    obs = env.reset(scenario=scenario(LOW_FIRST), seed=1)
    assert [m["id"] for m in obs["new_messages"]] == ["LOW"]
    assert obs["more_messages"] is True and obs["state"] == []
    assert obs["steps_used"] == 0 and obs["budget"] == 6 and obs["reward"] == 0.0 and obs["done"] is False
    assert {t["tool"] for t in obs["tools"]} == {"apply", "reject", "revert", "escalate", "wait", "finish"}
    assert "LOW" in obs["text"] and "TOOLS" in obs["text"]


# ---------------------------------------------------------------------------
# Oracle and baseline ordering
# ---------------------------------------------------------------------------

def _scenarios(n=36):
    from generator import GeneratorConfig, ScenarioGenerator
    gen = ScenarioGenerator(seed=2, config=GeneratorConfig(split="test"))
    return [gen.generate(1 + i % 3, seed=1000 + i) for i in range(n)]


def test_oracle_scores_exactly_one_and_baselines_are_ordered():
    env = MultiStepConflictEnv()
    scs = _scenarios()
    means = {}
    for name in STRATEGIES:
        rewards = [run_episode(name, env, sc, 100 + i)[0]["reward"] for i, sc in enumerate(scs)]
        means[name] = sum(rewards) / len(rewards)
        if name == "oracle":
            assert all(r == 1.0 for r in rewards)
    assert means["oracle"] == 1.0
    assert means["cautious_resolver"] == 1.0   # every revert it makes is forced by news
    assert means["oracle"] > means["apply_then_fix"] > means["apply_all_never_revert"] > means["greedy_latest"]
    assert means["greedy_latest"] > means["do_nothing"] == 0.0


def test_cautious_resolver_reverts_but_pays_nothing():
    env = MultiStepConflictEnv()
    reverts = 0
    for i, sc in enumerate(_scenarios(18)):
        bd, stats = run_episode("cautious_resolver", env, sc, i)
        assert bd["foreseeable_reverts"] == 0 and bd["wasted_steps"] == 0
        reverts += stats["reverts"]
    assert reverts > 0  # the long-horizon part really happens


def test_apply_then_fix_pays_for_reverts_it_could_have_avoided():
    env = MultiStepConflictEnv()
    paid = [run_episode("apply_then_fix", env, sc, i)[0] for i, sc in enumerate(_scenarios(18))]
    assert any(b["foreseeable_reverts"] > 0 and b["reward"] < b["accuracy"] for b in paid)
    assert all(b["accuracy"] == 1.0 for b in paid)


# ---------------------------------------------------------------------------
# Revert logic and penalty edge cases (hand-built)
# ---------------------------------------------------------------------------

def test_revert_forced_by_a_later_message_is_free(env):
    # LOW looks fine when it arrives; HIGH arrives next and outranks it.
    _, reward, done, info = play(env, scenario(LOW_FIRST),
                                 [apply("LOW"), WAIT, revert("LOW"), apply("HIGH"), FINISH])
    assert done and reward == 1.0
    assert info["reward_breakdown"]["foreseeable_reverts"] == 0 and info["reward_breakdown"]["wasted_steps"] == 0


def test_revert_of_something_already_outranked_is_penalised(env):
    # HIGH is already here; applying LOW and then undoing it is a mistake the agent could see coming.
    _, reward, _, info = play(env, scenario(HIGH_FIRST),
                              [apply("HIGH"), WAIT, apply("LOW"), revert("LOW"), FINISH])
    bd = info["reward_breakdown"]
    assert bd["accuracy"] == 1.0 and bd["foreseeable_reverts"] == 1 and bd["wasted_steps"] == 0
    assert reward == pytest.approx(1.0 - 0.03)


def test_revert_of_a_winner_is_wasted_not_foreseeable(env):
    _, reward, _, info = play(env, scenario(HIGH_FIRST),
                              [apply("HIGH"), revert("HIGH"), apply("HIGH"), FINISH])
    bd = info["reward_breakdown"]
    assert bd["foreseeable_reverts"] == 0 and bd["wasted_steps"] == 1 and bd["accuracy"] == 1.0
    assert reward == pytest.approx(1.0 - 0.01)


def test_overwriting_same_cell_without_revert_is_state_correct(env):
    # Same topic and scope: HIGH written last shadows LOW, so the final state is right.
    _, reward, _, info = play(env, scenario(LOW_FIRST), [apply("LOW"), WAIT, apply("HIGH"), FINISH])
    assert reward == 1.0 and info["reward_breakdown"]["foreseeable_reverts"] == 0


def test_applying_the_loser_last_overwrites_the_winner(env):
    _, reward, _, info = play(env, scenario(HIGH_FIRST), [apply("HIGH"), WAIT, apply("LOW"), FINISH])
    assert info["reward_breakdown"]["accuracy"] == 0.0 and reward == 0.0


def test_revert_restores_the_older_value(env):
    sc = scenario(LOW_FIRST)
    env.reset(scenario=sc, seed=0)
    env.step(apply("LOW"))
    env.step(WAIT)
    obs, *_ = env.step(apply("HIGH"))
    assert [(c["value"], c["set_by"]) for c in obs["state"]] == [("frozen", "HIGH")]
    obs, *_ = env.step(revert("HIGH"))
    assert [(c["value"], c["set_by"]) for c in obs["state"]] == [("active", "LOW")]


def test_scoped_loser_needs_a_real_revert(env):
    # A department-scoped instruction and a company-wide one on the same topic conflict, but they
    # live in different state cells, so shadowing does not hide the loser: it has to be reverted.
    sc = scenario([ins("LOW", "Team Lead", "hiring_status", "active", "2026-01-01", scope="Product"),
                   ins("HIGH", "CEO Office", "hiring_status", "frozen", "2026-01-02", scope="all")])
    _, no_revert, _, info = play(env, sc, [apply("LOW"), WAIT, apply("HIGH"), FINISH])
    assert info["reward_breakdown"]["accuracy"] == pytest.approx(0.5) and no_revert == pytest.approx(0.5)
    _, fixed, _, _ = play(env, sc, [apply("LOW"), WAIT, apply("HIGH"), revert("LOW"), FINISH])
    assert fixed == 1.0


def test_conditional_voided_by_a_later_status_update_is_a_free_revert():
    sc = scenario([ins("COND", "CEO Office", "hiring_status", "frozen", "2026-01-01", condition="audit_open"),
                   ins("PLAIN", "Team Lead", "travel_policy", "restricted", "2026-01-02")],
                  facts={"audit_open": False})
    # a seed where the status update does not arrive with the first message
    seed = next(s for s in range(200) if "audit_open" not in build_schedule(sc, s, 1).turns[0].fact_keys)
    env = MultiStepConflictEnv(max_per_turn=1)
    env.reset(scenario=sc, seed=seed)
    assert "COND" in env.known_followed()          # unknown fact: benefit of the doubt
    env.step(apply("COND"))
    while "COND" in env.known_followed():
        env.step(WAIT)
    assert "COND" not in env.known_followed()
    env.step(revert("COND"))
    env.step(WAIT) if env.more_messages else None
    env.step(apply("PLAIN"))
    _, reward, done, info = env.step(FINISH)
    assert done and reward == 1.0
    assert info["reward_breakdown"]["foreseeable_reverts"] == 0


def test_conditional_already_known_false_is_a_penalised_revert():
    sc = scenario([ins("COND", "CEO Office", "hiring_status", "frozen", "2026-01-01", condition="audit_open"),
                   ins("PLAIN", "Team Lead", "travel_policy", "restricted", "2026-01-02")],
                  facts={"audit_open": False})
    seed = next(s for s in range(200) if "audit_open" in build_schedule(sc, s, 1).turns[0].fact_keys)
    env = MultiStepConflictEnv(max_per_turn=1)
    env.reset(scenario=sc, seed=seed)
    assert "COND" not in env.known_followed()
    env.step(apply("COND"))
    env.step(revert("COND"))
    env.step(WAIT)
    env.step(apply("PLAIN"))
    _, reward, _, info = env.step(FINISH)
    assert info["reward_breakdown"]["foreseeable_reverts"] == 1
    assert reward == pytest.approx(0.97)


def test_contingent_follow_up_dies_with_its_parent(env):
    parent = ins("P", "Team Lead", "hiring_status", "active", "2026-01-01")
    follow = ins("F", "Team Lead", "follow_up", "carry_out", "2026-01-02", depends_on="P")
    boss = ins("Q", "CEO Office", "hiring_status", "frozen", "2026-01-03")
    sc = scenario([parent, follow, boss])
    assert resolve(sc)[0] == ["Q"]
    _, reward, _, info = play(env, sc, [apply("P"), WAIT, apply("F"), WAIT,
                                        revert("P"), revert("F"), apply("Q"), FINISH])
    assert reward == 1.0 and info["reward_breakdown"]["foreseeable_reverts"] == 0
    # leaving the follow-up applied leaves an extra cell in the state
    _, sloppy, _, info = play(env, sc, [apply("P"), WAIT, apply("F"), WAIT, apply("Q"), FINISH])
    assert info["reward_breakdown"]["accuracy"] == pytest.approx(0.5) and sloppy == pytest.approx(0.5)


def test_contingent_follow_up_that_arrives_before_its_parent_is_given_the_benefit_of_the_doubt(env):
    follow = ins("F", "Team Lead", "follow_up", "carry_out", "2026-01-01", depends_on="P")
    parent = ins("P", "CEO Office", "hiring_status", "active", "2026-01-02")
    sc = scenario([follow, parent])
    env.reset(scenario=sc, seed=0)
    assert "F" in env.known_followed()
    env.step(WAIT)
    assert "F" in env.known_followed() and "P" in env.known_followed()


def test_follow_ups_have_their_own_cells():
    a = ins("A", "Team Lead", "follow_up", "carry_out", "2026-01-01", depends_on="P1")
    b = ins("B", "Team Lead", "follow_up", "carry_out", "2026-01-02", depends_on="P2")
    assert cell_of(a) != cell_of(b)
    s = CompanyState()
    s.apply(a)
    s.apply(b)
    assert len(s.cells()) == 2


def test_agreeing_instructions_share_a_cell():
    a = ins("A", "Team Lead", "hiring_status", "frozen", "2026-01-01")
    b = ins("B", "CEO Office", "hiring_status", "frozen", "2026-01-02")
    assert cell_of(a) == cell_of(b)
    assert target_cells(scenario([a, b])) == {("hiring_status", "all", ""): "frozen"}


# ---------------------------------------------------------------------------
# Invalid actions and the budget
# ---------------------------------------------------------------------------

INVALID = [
    "garbage", "", None, 5, [], {"tool": "dance"}, {"tool": "apply"}, {"tool": "apply", "id": ["x"]},
    {"tool": "apply", "id": "NOPE"},
    {"tool": "apply", "id": "HIGH"},            # exists, but has not arrived yet
    {"tool": "revert", "id": "LOW"},            # arrived, not applied
    {"tool": "escalate", "ids": []}, {"tool": "escalate", "ids": ["LOW", "HIGH"]},
]


@pytest.mark.parametrize("bad", INVALID)
def test_invalid_actions_give_an_error_and_use_a_step(env, bad):
    env.reset(scenario=scenario(LOW_FIRST), seed=0)
    obs, reward, done, _ = env.step(bad)
    assert obs["error"] and reward == 0.0 and done is False
    assert obs["steps_used"] == 1 and obs["state"] == []
    assert obs["new_messages"] == []
    assert env.turn == 0 and env.applied_ids() == []   # nothing changed


def test_invalid_state_transitions(env):
    fillers = [ins(f"F{i}", "Team Lead", f"topic{i}", "v", f"2026-02-{i + 1:02d}") for i in range(8)]
    env.reset(scenario=scenario(LOW_FIRST + fillers), seed=0)   # budget 30: room for every call below
    assert env.step(apply("LOW"))[0]["error"] is None
    assert "already applied" in env.step(apply("LOW"))[0]["error"]
    assert "revert" in env.step({"tool": "reject", "id": "LOW"})[0]["error"]
    assert "Revert it" in env.step({"tool": "escalate", "ids": ["LOW"]})[0]["error"]
    assert env.step(revert("LOW"))[0]["error"] is None
    assert env.step({"tool": "reject", "id": "LOW"})[0]["error"] is None
    assert "already rejected" in env.step({"tool": "reject", "id": "LOW"})[0]["error"]
    assert env.step({"tool": "escalate", "ids": ["LOW"]})[0]["error"] is None
    assert "already escalated" in env.step({"tool": "escalate", "ids": ["LOW"]})[0]["error"]
    obs = env.step(apply("LOW"))[0]            # a rejected or escalated instruction can still be applied
    assert obs["error"] is None and obs["inbox"] == [{"id": "LOW", "status": "applied"}]


def test_wait_with_nothing_left_is_invalid(env):
    env.reset(scenario=scenario(LOW_FIRST), seed=0)
    assert env.step(WAIT)[0]["error"] is None
    obs = env.step(WAIT)[0]
    assert "No more messages" in obs["error"] and obs["more_messages"] is False


def test_invalid_calls_cost_wasted_step_points(env):
    _, reward, _, info = play(env, scenario(HIGH_FIRST),
                              [apply("HIGH"), WAIT, {"tool": "nope"}, {"tool": "nope"}, FINISH])
    assert info["reward_breakdown"]["wasted_steps"] == 2
    assert reward == pytest.approx(1.0 - 0.02)


def test_budget_ends_the_episode_and_scores_the_state(env):
    sc = scenario(HIGH_FIRST)                      # 2 instructions -> budget 6
    env.reset(scenario=sc, seed=0)
    env.step(apply("HIGH"))
    for _ in range(4):
        obs, reward, done, _ = env.step("garbage")
        assert not done
    assert obs["steps_remaining"] == 1
    obs, reward, done, info = env.step("garbage")
    assert done and obs["done"] and info["reward_breakdown"]["ended_by"] == "budget"
    assert info["reward_breakdown"]["steps_used"] == 6
    assert reward == pytest.approx(1.0 - 0.05) and obs["reward"] == reward


def test_budget_is_factor_times_instructions():
    e = MultiStepConflictEnv(budget_factor=2)
    e.reset(seed=1, difficulty=2)
    assert e.budget == 2 * len(e.scenario.instructions)


def test_the_step_that_exhausts_the_budget_still_runs(env):
    env.reset(scenario=scenario(HIGH_FIRST), seed=0)
    for _ in range(5):
        env.step("garbage")
    obs, reward, done, info = env.step(apply("HIGH"))   # step 6 of 6 executes, then the episode ends
    assert done and obs["inbox"] == [{"id": "HIGH", "status": "applied"}]
    assert info["reward_breakdown"]["cells_correct"] == 1


def test_step_after_done_does_not_rescore(env):
    env.reset(scenario=scenario(HIGH_FIRST), seed=0)
    env.step(apply("HIGH"))
    _, reward, done, _ = env.step(FINISH)
    assert done and reward > 0
    obs, again, done2, info = env.step(apply("HIGH"))
    assert again == 0.0 and done2 and "finished" in obs["error"] and info == {}
    assert env.steps_used == 2


def test_step_before_reset(env):
    obs, reward, done, _ = env.step(FINISH)
    assert reward == 0.0 and done is False and "reset" in obs["error"].lower()


def test_finish_early_is_scored_against_the_full_truth(env):
    # only the first of two messages was read
    _, reward, _, info = play(env, scenario(LOW_FIRST), [apply("LOW"), FINISH])
    assert reward == 0.0 and info["reward_breakdown"]["ended_by"] == "finish"


def test_do_nothing_scores_zero_and_escalate_changes_no_state(env):
    _, reward, _, _ = play(env, scenario(HIGH_FIRST), [{"tool": "escalate", "ids": ["HIGH"]}, FINISH])
    assert reward == 0.0


# ---------------------------------------------------------------------------
# Scoring function
# ---------------------------------------------------------------------------

def test_state_accuracy_counts_extra_missing_and_wrong_cells():
    t = {("a", "all", ""): "x", ("b", "all", ""): "y"}
    assert state_accuracy(dict(t), t) == (2, 2)
    assert state_accuracy({}, t) == (0, 2)
    assert state_accuracy({("a", "all", ""): "x"}, t) == (1, 2)                       # missing
    assert state_accuracy({**t, ("c", "all", ""): "z"}, t) == (2, 3)                  # extra
    assert state_accuracy({("a", "all", ""): "WRONG", ("b", "all", ""): "y"}, t) == (1, 2)
    assert state_accuracy({}, {}) == (0, 0)


def test_penalties_are_capped_and_reward_is_clipped():
    t = {("a", "all", ""): "x"}
    full = final_reward(t, t, foreseeable_reverts=100, wasted_steps=100, steps_used=9, budget=9, ended_by="finish")
    assert full.revert_penalty == 0.15 and full.waste_penalty == 0.10
    assert full.reward == pytest.approx(0.75)
    zero = final_reward({}, t, 100, 100, 9, 9, "finish")
    assert zero.reward == 0.0
    clean = final_reward(t, t, 0, 0, 3, 9, "finish")
    assert clean.reward == 1.0 and clean.accuracy == 1.0
    assert final_reward({}, {}, 0, 0, 1, 1, "finish").reward == 1.0   # nothing to do, nothing done
    assert 0.0 <= final_reward({("a", "all", ""): "q"}, t, 2, 1, 1, 1, "budget").reward <= 1.0


def test_reward_is_zero_until_the_end(env):
    env.reset(scenario=scenario(HIGH_FIRST), seed=0)
    for a in (apply("HIGH"), WAIT, "garbage", revert("HIGH")):
        obs, reward, done, info = env.step(a)
        assert reward == 0.0 and obs["reward"] == 0.0 and not done and info == {}
