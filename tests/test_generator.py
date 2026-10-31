"""
Generator tests (P1-2, P1-3): reproducibility, isolation, no ties, label agreement.
"""

import dataclasses
import json
import random
import re

import pytest

from generator import (
    CONFLICT_TYPE_LABELS,
    PROMPT_CONFLICT_TYPES,
    SOURCE_PRIORITY,
    ScenarioGenerator,
)
from verifier import score


def _as_dict(scenario):
    return dataclasses.asdict(scenario)


def _ground_truth_answer(sc) -> str:
    """Perfect answer built straight from the stored ground truth."""
    return json.dumps({
        "identified_conflicts": [
            {
                "instruction_a": c.instruction_a_id,
                "instruction_b": c.instruction_b_id,
                "conflict_type": c.conflict_type,
                "resolution": c.resolution_id,
                "reasoning": c.explanation,
            }
            for c in sc.conflicts
        ],
        "execution_plan": sc.ground_truth_followed,
        "overridden_instructions": sc.ground_truth_overridden,
    })


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("difficulty", [1, 2, 3])
def test_same_seed_gives_identical_scenarios(difficulty):
    a = ScenarioGenerator(seed=42)
    b = ScenarioGenerator(seed=42)
    for _ in range(25):
        sa, sb = a.generate(difficulty), b.generate(difficulty)
        assert _as_dict(sa) == _as_dict(sb)
        # Spell out the pieces the blueprint cares about.
        assert [i.id for i in sa.instructions] == [i.id for i in sb.instructions]
        assert [i.text for i in sa.instructions] == [i.text for i in sb.instructions]
        assert sa.scenario_id == sb.scenario_id
        assert sa.prompt == sb.prompt
        assert sa.ground_truth_followed == sb.ground_truth_followed
        assert sa.ground_truth_overridden == sb.ground_truth_overridden


def test_different_seeds_differ():
    s1 = ScenarioGenerator(seed=1).generate(2)
    s2 = ScenarioGenerator(seed=2).generate(2)
    assert s1.scenario_id != s2.scenario_id
    assert s1.prompt != s2.prompt


def test_successive_scenarios_differ():
    gen = ScenarioGenerator(seed=3)
    assert gen.generate(2).scenario_id != gen.generate(2).scenario_id


def test_generate_batch_reproducible():
    a = ScenarioGenerator(seed=9).generate_batch(20)
    b = ScenarioGenerator(seed=9).generate_batch(20)
    assert [_as_dict(s) for s in a] == [_as_dict(s) for s in b]


def test_unseeded_generators_differ():
    assert ScenarioGenerator().generate(2).prompt != ScenarioGenerator().generate(2).prompt


def test_ids_and_scenario_id_format():
    gen = ScenarioGenerator(seed=11)
    for _ in range(200):
        sc = gen.generate(3)
        ids = [i.id for i in sc.instructions]
        assert len(ids) == len(set(ids))
        assert all(re.fullmatch(r"INS-[0-9A-F]{6}", i) for i in ids)
        # scenario_id is a valid UUID string
        import uuid
        assert str(uuid.UUID(sc.scenario_id)) == sc.scenario_id


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------

def test_generators_do_not_interfere():
    """Interleaving a second generator must not change the first one's output."""
    solo = ScenarioGenerator(seed=100)
    expected = [_as_dict(solo.generate(2)) for _ in range(15)]

    g1 = ScenarioGenerator(seed=100)
    g2 = ScenarioGenerator(seed=200)
    got = []
    for _ in range(15):
        got.append(_as_dict(g1.generate(2)))
        g2.generate(3)
        g2.generate(1)
    assert got == expected


def test_global_random_is_untouched():
    random.seed(987)
    expected = [random.random() for _ in range(10)]

    random.seed(987)
    gen = ScenarioGenerator(seed=5)          # creating a generator must not touch it
    gen.generate(3)
    gen.generate(1, seed=77)
    gen.generate_batch(5)
    got = [random.random() for _ in range(10)]
    assert got == expected


def test_global_random_does_not_affect_generator():
    random.seed(1)
    a = ScenarioGenerator(seed=8).generate(2)
    random.seed(2)
    random.random()
    b = ScenarioGenerator(seed=8).generate(2)
    assert _as_dict(a) == _as_dict(b)


# ---------------------------------------------------------------------------
# Per-call seed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 12345])
def test_per_call_seed_identical_across_instances(seed):
    a = ScenarioGenerator(seed=1)
    b = ScenarioGenerator(seed=2)
    c = ScenarioGenerator()
    ref = _as_dict(a.generate(2, seed=seed))
    assert _as_dict(b.generate(2, seed=seed)) == ref
    assert _as_dict(c.generate(2, seed=seed)) == ref


def test_per_call_seed_does_not_advance_instance_rng():
    a = ScenarioGenerator(seed=50)
    b = ScenarioGenerator(seed=50)
    a.generate(3, seed=999)
    a.generate(1, seed=999)
    assert _as_dict(a.generate(2)) == _as_dict(b.generate(2))


def test_per_call_seed_respects_difficulty_and_domain():
    sc = ScenarioGenerator().generate(1, domain="Finance", seed=4)
    assert sc.difficulty == 1 and sc.domain == "Finance"
    assert len(sc.conflicts) == 2
    sc3 = ScenarioGenerator().generate(3, domain="IT", seed=4)
    assert len(sc3.conflicts) == 6 and sc3.domain == "IT"


# ---------------------------------------------------------------------------
# No ties
# ---------------------------------------------------------------------------

def test_no_ties_over_5000_scenarios():
    gen = ScenarioGenerator(seed=2024)
    n = 0
    expected = 0
    for i in range(5000):
        sc = gen.generate(difficulty=(i % 3) + 1)
        expected += 2 * ((i % 3) + 1)
        by_id = {ins.id: ins for ins in sc.instructions}
        for c in sc.conflicts:
            a, b = by_id[c.instruction_a_id], by_id[c.instruction_b_id]
            assert a.source_priority != b.source_priority, "tie between conflicting sources"
            winner = by_id[c.resolution_id]
            loser = b if winner is a else a
            assert winner.source_priority < loser.source_priority   # strictly higher authority
            assert a.source_priority == SOURCE_PRIORITY[a.source]
            assert b.source_priority == SOURCE_PRIORITY[b.source]
            assert c.resolution_id in sc.ground_truth_followed
            assert loser.id in sc.ground_truth_overridden
            n += 1
    assert n == expected > 0   # every conflict of every scenario was checked


def test_every_template_pair_can_avoid_a_tie():
    """For every group and every possible first source, a different-tier partner exists."""
    from generator import TEMPLATE_GROUPS
    for key, group in TEMPLATE_GROUPS.items():
        t_a, t_b = group["templates"][0], group["templates"][1]
        for s in t_a["sources"]:
            assert any(SOURCE_PRIORITY[o] != SOURCE_PRIORITY[s] for o in t_b["sources"]), (key, s)


# ---------------------------------------------------------------------------
# _format_prompt must not mutate its input
# ---------------------------------------------------------------------------

def test_format_prompt_does_not_shuffle_callers_list():
    gen = ScenarioGenerator(seed=6)
    sc = gen.generate(3)
    original = list(sc.instructions)
    snapshot_ids = [i.id for i in original]
    for _ in range(20):
        gen._format_prompt(sc.business_context, original)
        assert [i.id for i in original] == snapshot_ids


def test_format_prompt_still_randomises_order():
    gen = ScenarioGenerator(seed=6)
    sc = gen.generate(3)
    orders = set()
    for _ in range(20):
        prompt = gen._format_prompt(sc.business_context, sc.instructions)
        orders.add(tuple(re.findall(r"\[(INS-[0-9A-F]{6})\]", prompt)))
    assert len(orders) > 1


def test_scenario_instruction_order_matches_prompt_order():
    gen = ScenarioGenerator(seed=13)
    for _ in range(50):
        sc = gen.generate(2)
        in_prompt = re.findall(r"^\[(INS-[0-9A-F]{6})\]", sc.prompt, flags=re.M)
        assert in_prompt == [i.id for i in sc.instructions]


# ---------------------------------------------------------------------------
# P1-3: conflict-type labels agree with the prompt
# ---------------------------------------------------------------------------

def test_conflict_type_mapping():
    assert CONFLICT_TYPE_LABELS == {
        "absolute": "direct",
        "resource": "resource",
        "time_ordered": "temporal",
        "conditional": "conditional",
    }


def test_ground_truth_conflict_types_are_in_prompt_allowed_list():
    gen = ScenarioGenerator(seed=77)
    seen = set()
    for i in range(300):
        sc = gen.generate(difficulty=(i % 3) + 1)
        m = re.search(r'"conflict_type":"([^"]+)"', sc.prompt)
        assert m, "prompt lost its conflict_type field"
        allowed = m.group(1).split("|")
        assert allowed == list(PROMPT_CONFLICT_TYPES)
        for c in sc.conflicts:
            assert c.conflict_type in allowed
            seen.add(c.conflict_type)
    # All three label families are exercised by the current templates.
    assert seen == {"direct", "resource", "temporal"}


def test_prompt_wording_unchanged_for_current_templates():
    sc = ScenarioGenerator(seed=0).generate(1)
    assert '"conflict_type":"direct|resource|temporal"' in sc.prompt


# ---------------------------------------------------------------------------
# Existing behaviour still holds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("difficulty", [1, 2, 3])
def test_ground_truth_still_scores_perfect(difficulty):
    gen = ScenarioGenerator(seed=31)
    for _ in range(50):
        sc = gen.generate(difficulty)
        assert score(_ground_truth_answer(sc), sc).composite == 1.0


def test_conflict_counts_per_difficulty():
    gen = ScenarioGenerator(seed=21)
    for d, n in ((1, 2), (2, 4), (3, 6)):
        assert len(gen.generate(d).conflicts) == n
