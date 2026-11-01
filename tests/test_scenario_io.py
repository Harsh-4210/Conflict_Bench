"""Scenario JSON round-trips exactly and scores identically after loading."""

import json

from generator import ScenarioGenerator
from scenario_io import dumps_scenario, load_scenarios, loads_scenario, save_scenarios
from verifier import score


def _gt(sc):
    return json.dumps({
        "identified_conflicts": [
            {"instruction_a": c.instruction_a_id, "instruction_b": c.instruction_b_id,
             "conflict_type": c.conflict_type, "resolution": c.resolution_id, "reasoning": "r"}
            for c in sc.conflicts
        ],
        "execution_plan": sc.ground_truth_followed,
        "overridden_instructions": sc.ground_truth_overridden,
    })


def test_round_trip_is_exact():
    gen = ScenarioGenerator(seed=11)
    for d in (1, 2, 3):
        sc = gen.generate(difficulty=d)
        assert loads_scenario(dumps_scenario(sc)) == sc


def test_file_round_trip_and_scoring(tmp_path):
    gen = ScenarioGenerator(seed=12)
    scs = [gen.generate(difficulty=1 + i % 3) for i in range(9)]
    path = tmp_path / "set.jsonl"
    save_scenarios(path, scs)
    loaded = load_scenarios(path)
    assert loaded == scs
    for sc in loaded:
        assert score(_gt(sc), sc).composite == 1.0


def test_unknown_keys_are_ignored():
    sc = ScenarioGenerator(seed=13).generate(difficulty=1)
    data = json.loads(dumps_scenario(sc))
    data["future_field"] = 1
    data["instructions"][0]["another"] = "x"
    assert loads_scenario(json.dumps(data)) == sc
