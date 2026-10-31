"""
resolver.resolve() is the only source of ground truth (P3-4).

1. Every generated scenario (legacy and v2, every split, every level) stores
   exactly what resolve() derives.
2. Hand-built scenarios pin each rule: tier, scope, agreement, standing rule,
   delegation, dated supersession, conditions, chains, ties.
3. resolve() reads structured fields only: text and presentation order don't matter.
"""

import dataclasses
import random

import pytest

from generator import (
    MAX_DIFFICULTY,
    SPLITS,
    GeneratorConfig,
    Instruction,
    Scenario,
    ScenarioGenerator,
)
from resolver import (
    UnresolvableConflict,
    matches_ground_truth,
    resolution_methods,
    resolve,
    scopes_overlap,
)
from scenario_io import dumps_scenario, loads_scenario


# ---------------------------------------------------------------------------
# 1. Generated scenarios == resolve()
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("config", [
    GeneratorConfig(legacy=True),
    *[GeneratorConfig(split=s) for s in SPLITS],
    GeneratorConfig(show_hierarchy=True),
    GeneratorConfig(scoped_authority=False, delegation=False, time_supersession=False,
                    conditionals=False, chains=False),
], ids=["legacy", "train", "val", "test", "shown", "rules_off"])
def test_every_generated_scenario_matches_resolve(config):
    gen = ScenarioGenerator(seed=2718, config=config)
    levels = (1, 2, 3) if config.legacy else range(1, MAX_DIFFICULTY + 1)
    for i in range(600):
        sc = gen.generate(levels[i % len(levels)])
        assert matches_ground_truth(sc) is None, matches_ground_truth(sc)


def test_v2_stores_resolver_output_exactly():
    gen = ScenarioGenerator(seed=1)
    for i in range(200):
        sc = gen.generate(1 + i % 5)
        followed, overridden, conflicts = resolve(sc)
        assert sc.ground_truth_followed == followed
        assert sc.ground_truth_overridden == overridden
        assert sc.conflicts == conflicts


def test_resolve_after_json_round_trip():
    gen = ScenarioGenerator(seed=2)
    for i in range(100):
        sc = loads_scenario(dumps_scenario(gen.generate(1 + i % 5)))
        assert matches_ground_truth(sc) is None


def test_all_rule_kinds_are_exercised():
    gen = ScenarioGenerator(seed=3)
    kinds = set()
    for i in range(300):
        kinds.update(resolution_methods(gen.generate(1 + i % 3)).values())
    assert kinds == {"tier", "rule", "delegation", "time"}


def test_followed_and_overridden_partition_the_instructions():
    gen = ScenarioGenerator(seed=4)
    for i in range(300):
        sc = gen.generate(1 + i % 5)
        f, o = set(sc.ground_truth_followed), set(sc.ground_truth_overridden)
        assert not f & o
        assert f | o == {ins.id for ins in sc.instructions}
        for c in sc.conflicts:  # exactly one side of each conflict is followed
            assert len({c.instruction_a_id, c.instruction_b_id} & f) == 1


# ---------------------------------------------------------------------------
# 2. Hand-built scenarios
# ---------------------------------------------------------------------------

_TIER = {"Legal & Compliance": 1, "CEO Office": 2, "Chief Financial Officer": 2, "Chief Operating Officer": 2,
         "VP Finance": 3, "VP Operations": 3, "Director of Finance": 4, "Finance Manager": 5, "Team Lead": 6}


def _ins(iid, office, value, key="hiring_status", scope="all", date=None, condition=None, depends_on=None):
    return Instruction(
        id=iid, text=f"text {iid}", source=office, source_priority=_TIER[office],
        instruction_type="absolute", action_key=key, action_value=value, condition=condition,
        action_scope=scope, effective_date=date, depends_on=depends_on,
    )


def _sc(instructions, **extra):
    return Scenario("s", "HR", 1, "ctx", instructions, [], [], [], "p", **extra)


def _outcome(sc):
    followed, overridden, conflicts = resolve(sc)
    pairs = {frozenset((c.instruction_a_id, c.instruction_b_id)): c.resolution_id for c in conflicts}
    return set(followed), set(overridden), pairs


def test_higher_tier_wins():
    f, o, pairs = _outcome(_sc([_ins("A", "Team Lead", "active"), _ins("B", "VP Finance", "frozen")]))
    assert f == {"B"} and o == {"A"} and pairs == {frozenset("AB"): "B"}


def test_agreeing_instructions_do_not_conflict():
    f, o, pairs = _outcome(_sc([_ins("A", "Team Lead", "frozen"), _ins("B", "VP Finance", "frozen")]))
    assert f == {"A", "B"} and not o and not pairs


def test_different_departments_do_not_conflict():
    sc = _sc([_ins("A", "Team Lead", "active", scope="Sales"), _ins("B", "CEO Office", "frozen", scope="HR")])
    f, o, pairs = _outcome(sc)
    assert f == {"A", "B"} and not pairs


def test_company_wide_overlaps_a_department():
    sc = _sc([_ins("A", "Team Lead", "active", scope="all"), _ins("B", "CEO Office", "frozen", scope="HR")])
    assert _outcome(sc)[2] == {frozenset("AB"): "B"}
    assert scopes_overlap("all", "HR") and scopes_overlap("HR", "HR") and not scopes_overlap("HR", "Sales")
    assert scopes_overlap("", "Sales")  # old data without scopes = company-wide


def test_different_topics_do_not_conflict():
    sc = _sc([_ins("A", "Team Lead", "x", key="travel_policy"), _ins("B", "CEO Office", "y")])
    assert _outcome(sc)[2] == {}


def test_standing_rule_beats_seniority_on_its_topic_only():
    ins = [
        _ins("A", "CEO Office", "frozen", key="budget_approval_threshold"),
        _ins("B", "Finance Manager", "active", key="budget_approval_threshold"),
        _ins("C", "CEO Office", "frozen", key="travel_policy"),
        _ins("D", "Finance Manager", "active", key="travel_policy"),
    ]
    rules = [{"action_key": "budget_approval_threshold", "office": "Finance Manager"}]
    f, o, pairs = _outcome(_sc(ins, authority_rules=rules))
    assert pairs == {frozenset("AB"): "B", frozenset("CD"): "C"}
    assert f == {"B", "C"}


def test_delegation_raises_the_delegate_on_one_topic():
    ins = [
        _ins("A", "VP Operations", "executive_only", key="vendor_contract_approval"),
        _ins("B", "Finance Manager", "manager_approved", key="vendor_contract_approval"),
        _ins("C", "VP Operations", "x", key="travel_policy"),
        _ins("D", "Finance Manager", "y", key="travel_policy"),
    ]
    dels = [{"action_key": "vendor_contract_approval", "from_office": "Chief Operating Officer",
             "to_office": "Finance Manager"}]
    f, o, pairs = _outcome(_sc(ins, delegations=dels))
    assert pairs == {frozenset("AB"): "B", frozenset("CD"): "C"}


def test_delegate_still_loses_to_someone_above_the_delegator():
    ins = [_ins("A", "Legal & Compliance", "frozen"), _ins("B", "Finance Manager", "active")]
    dels = [{"action_key": "hiring_status", "from_office": "CEO Office", "to_office": "Finance Manager"}]
    assert _outcome(_sc(ins, delegations=dels))[2] == {frozenset("AB"): "A"}


def test_equal_authority_later_date_wins():
    ins = [_ins("A", "VP Finance", "frozen", date="2026-02-01"),
           _ins("B", "VP Operations", "active", date="2026-03-01")]
    sc = _sc(ins)
    assert _outcome(sc)[2] == {frozenset("AB"): "B"}
    assert resolution_methods(sc) == {frozenset("AB"): "time"}


def test_same_person_revising_an_instruction():
    ins = [_ins("A", "CEO Office", "active", date="2026-03-09"), _ins("B", "CEO Office", "frozen", date="2026-01-20")]
    assert _outcome(_sc(ins))[2] == {frozenset("AB"): "A"}


def test_later_date_never_beats_higher_authority():
    ins = [_ins("A", "Legal & Compliance", "frozen", date="2026-01-01"),
           _ins("B", "Team Lead", "active", date="2026-03-01")]
    assert _outcome(_sc(ins))[2] == {frozenset("AB"): "A"}


@pytest.mark.parametrize("dates", [(None, None), ("2026-02-01", "2026-02-01"), ("2026-02-01", None)])
def test_equal_authority_without_usable_dates_is_unresolvable(dates):
    ins = [_ins("A", "VP Finance", "frozen", date=dates[0]), _ins("B", "VP Operations", "active", date=dates[1])]
    with pytest.raises(UnresolvableConflict):
        resolve(_sc(ins))


def test_false_condition_voids_the_instruction_and_its_conflict():
    ins = [_ins("A", "CEO Office", "frozen", condition="audit_open"), _ins("B", "Team Lead", "active")]
    f, o, pairs = _outcome(_sc(ins, facts={"audit_open": False}))
    assert f == {"B"} and o == {"A"} and pairs == {}


def test_true_condition_keeps_the_conflict():
    ins = [_ins("A", "CEO Office", "frozen", condition="audit_open"), _ins("B", "Team Lead", "active")]
    f, o, pairs = _outcome(_sc(ins, facts={"audit_open": True}))
    assert pairs == {frozenset("AB"): "A"} and f == {"A"}
    assert resolve(_sc(ins, facts={"audit_open": True}))[2][0].conflict_type == "conditional"


def test_unknown_fact_counts_as_false():
    ins = [_ins("A", "CEO Office", "frozen", condition="nobody_said")]
    assert _outcome(_sc(ins))[1] == {"A"}


def test_chain_is_cancelled_when_its_parent_loses():
    ins = [
        _ins("A", "CEO Office", "frozen"),
        _ins("B", "Team Lead", "active"),
        _ins("C", "Team Lead", "carry_out", key="follow_up", depends_on="B"),
        _ins("D", "Team Lead", "carry_out", key="follow_up", depends_on="A"),
    ]
    f, o, pairs = _outcome(_sc(ins))
    assert f == {"A", "D"} and o == {"B", "C"} and pairs == {frozenset("AB"): "A"}


def test_multi_hop_chain():
    ins = [
        _ins("A", "CEO Office", "frozen", condition="audit_open"),
        _ins("B", "Team Lead", "carry_out", key="follow_up", depends_on="A"),
        _ins("C", "Team Lead", "carry_out", key="follow_up", depends_on="B"),
    ]
    assert _outcome(_sc(ins, facts={"audit_open": False}))[1] == {"A", "B", "C"}
    assert _outcome(_sc(ins, facts={"audit_open": True}))[0] == {"A", "B", "C"}


def test_legacy_shaped_data_resolves_by_tier():
    """Old JSON has no scopes/dates/notices: the resolver falls back to plain tier lookup."""
    ins = [_ins("A", "Team Lead", "active"), _ins("B", "VP Finance", "frozen"),
           _ins("C", "CEO Office", "x", key="travel_policy")]
    for i in ins:
        i.action_scope = ""
    f, o, pairs = _outcome(_sc(ins))
    assert f == {"B", "C"} and o == {"A"}


# ---------------------------------------------------------------------------
# 3. Structured fields only
# ---------------------------------------------------------------------------

def test_text_and_presentation_order_do_not_matter():
    gen = ScenarioGenerator(seed=9)
    rng = random.Random(0)
    for i in range(150):
        sc = gen.generate(1 + i % 5)
        followed, overridden, conflicts = resolve(sc)
        scrambled = [dataclasses.replace(ins, text="lorem ipsum", author="", title="") for ins in sc.instructions]
        rng.shuffle(scrambled)
        alt = dataclasses.replace(sc, instructions=scrambled, prompt="", business_context="")
        f2, o2, c2 = resolve(alt)
        assert (f2, o2) == (followed, overridden)
        assert {frozenset((c.instruction_a_id, c.instruction_b_id)): c.resolution_id for c in c2} == \
               {frozenset((c.instruction_a_id, c.instruction_b_id)): c.resolution_id for c in conflicts}


def test_conflict_pairs_follow_presentation_order():
    gen = ScenarioGenerator(seed=10)
    for i in range(100):
        sc = gen.generate(1 + i % 3)
        pos = {ins.id: n for n, ins in enumerate(sc.instructions)}
        starts = [pos[c.instruction_a_id] for c in sc.conflicts]
        assert starts == sorted(starts)
        assert all(pos[c.instruction_a_id] < pos[c.instruction_b_id] for c in sc.conflicts)
