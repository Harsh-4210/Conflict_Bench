"""
Generator tests.

P1-2, P1-3 (both generators): reproducibility, isolation, no ties, label agreement.
Phase 3 (v2, the default): config, balanced winners (P3-1), distractors (P3-2),
hidden hierarchy (P3-3), rule flags (P3-4), difficulty levels (P3-5), splits (P3-6).
The legacy generator must stay byte-for-byte what it was.
"""

import dataclasses
import hashlib
import itertools
import json
import random
import re
import uuid

import pytest

from generator import (
    CONFLICT_TYPE_LABELS,
    DIFFICULTY_LEVELS,
    MAX_DIFFICULTY,
    OFFICE_TITLES,
    PROMPT_CONFLICT_TYPES,
    SOURCE_PRIORITY,
    SPLITS,
    V2_COMPANIES,
    V2_CONFLICT_TYPES,
    V2_TOPICS,
    GeneratorConfig,
    Instruction,
    Scenario,
    ScenarioGenerator,
    clamp_difficulty,
    format_date,
    split_pool,
)
from resolver import pick_winner, scopes_overlap
from scenario_io import dumps_scenario, loads_scenario
from verifier import score

LEGACY = GeneratorConfig(legacy=True)
V2 = GeneratorConfig()
CONFIGS = pytest.mark.parametrize("config", [LEGACY, V2], ids=["legacy", "v2"])


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
# Reproducibility (both generators)
# ---------------------------------------------------------------------------

@CONFIGS
@pytest.mark.parametrize("difficulty", [1, 2, 3])
def test_same_seed_gives_identical_scenarios(config, difficulty):
    a = ScenarioGenerator(seed=42, config=config)
    b = ScenarioGenerator(seed=42, config=config)
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


@CONFIGS
def test_different_seeds_differ(config):
    s1 = ScenarioGenerator(seed=1, config=config).generate(2)
    s2 = ScenarioGenerator(seed=2, config=config).generate(2)
    assert s1.scenario_id != s2.scenario_id
    assert s1.prompt != s2.prompt


@CONFIGS
def test_successive_scenarios_differ(config):
    gen = ScenarioGenerator(seed=3, config=config)
    assert gen.generate(2).scenario_id != gen.generate(2).scenario_id


@CONFIGS
def test_generate_batch_reproducible(config):
    a = ScenarioGenerator(seed=9, config=config).generate_batch(20)
    b = ScenarioGenerator(seed=9, config=config).generate_batch(20)
    assert [_as_dict(s) for s in a] == [_as_dict(s) for s in b]


@CONFIGS
def test_unseeded_generators_differ(config):
    assert ScenarioGenerator(config=config).generate(2).prompt != ScenarioGenerator(config=config).generate(2).prompt


@CONFIGS
def test_ids_and_scenario_id_format(config):
    gen = ScenarioGenerator(seed=11, config=config)
    for _ in range(200):
        sc = gen.generate(3)
        ids = [i.id for i in sc.instructions]
        assert len(ids) == len(set(ids))
        assert all(re.fullmatch(r"INS-[0-9A-F]{6}", i) for i in ids)
        assert str(uuid.UUID(sc.scenario_id)) == sc.scenario_id


# The legacy generator's output for these calls, hashed over the pre-Phase-3
# fields, before any Phase 3 code existed. Any change to legacy output fails here.
_LEGACY_SNAPSHOT = "bad92cfdb00705e60ffd29fe37ab8a22a9224a1c0d0823698047304a5caf9389"
_OLD_INS = ["id", "text", "source", "source_priority", "instruction_type", "action_key", "action_value", "condition"]
_OLD_CP = ["instruction_a_id", "instruction_b_id", "conflict_type", "resolution_id", "explanation"]


def _old_fields(sc):
    return {
        "scenario_id": sc.scenario_id, "domain": sc.domain, "difficulty": sc.difficulty,
        "business_context": sc.business_context,
        "instructions": [{k: getattr(i, k) for k in _OLD_INS} for i in sc.instructions],
        "conflicts": [{k: getattr(c, k) for k in _OLD_CP} for c in sc.conflicts],
        "f": sc.ground_truth_followed, "o": sc.ground_truth_overridden, "p": sc.prompt,
    }


def test_legacy_output_is_unchanged():
    gen = ScenarioGenerator(seed=2024, config=LEGACY)
    scs = [gen.generate(difficulty=1 + i % 3) for i in range(60)]
    scs += [gen.generate(2, domain="IT", seed=s) for s in range(5)]
    scs += gen.generate_batch(5)
    digest = hashlib.sha256(json.dumps([_old_fields(s) for s in scs]).encode()).hexdigest()
    assert digest == _LEGACY_SNAPSHOT


def test_legacy_config_helper():
    assert GeneratorConfig.legacy_config() == LEGACY
    assert ScenarioGenerator(seed=1, config=LEGACY).generate(1).generator_version == ""


# ---------------------------------------------------------------------------
# Isolation (both generators)
# ---------------------------------------------------------------------------

@CONFIGS
def test_generators_do_not_interfere(config):
    """Interleaving a second generator must not change the first one's output."""
    solo = ScenarioGenerator(seed=100, config=config)
    expected = [_as_dict(solo.generate(2)) for _ in range(15)]

    g1 = ScenarioGenerator(seed=100, config=config)
    g2 = ScenarioGenerator(seed=200, config=config)
    got = []
    for _ in range(15):
        got.append(_as_dict(g1.generate(2)))
        g2.generate(3)
        g2.generate(1)
    assert got == expected


@CONFIGS
def test_global_random_is_untouched(config):
    random.seed(987)
    expected = [random.random() for _ in range(10)]

    random.seed(987)
    gen = ScenarioGenerator(seed=5, config=config)  # creating a generator must not touch it
    gen.generate(3)
    gen.generate(1, seed=77)
    gen.generate_batch(5)
    got = [random.random() for _ in range(10)]
    assert got == expected


@CONFIGS
def test_global_random_does_not_affect_generator(config):
    random.seed(1)
    a = ScenarioGenerator(seed=8, config=config).generate(2)
    random.seed(2)
    random.random()
    b = ScenarioGenerator(seed=8, config=config).generate(2)
    assert _as_dict(a) == _as_dict(b)


# ---------------------------------------------------------------------------
# Per-call seed (both generators)
# ---------------------------------------------------------------------------

@CONFIGS
@pytest.mark.parametrize("seed", [0, 1, 12345])
def test_per_call_seed_identical_across_instances(config, seed):
    a = ScenarioGenerator(seed=1, config=config)
    b = ScenarioGenerator(seed=2, config=config)
    c = ScenarioGenerator(config=config)
    ref = _as_dict(a.generate(2, seed=seed))
    assert _as_dict(b.generate(2, seed=seed)) == ref
    assert _as_dict(c.generate(2, seed=seed)) == ref


@CONFIGS
def test_per_call_seed_does_not_advance_instance_rng(config):
    a = ScenarioGenerator(seed=50, config=config)
    b = ScenarioGenerator(seed=50, config=config)
    a.generate(3, seed=999)
    a.generate(1, seed=999)
    assert _as_dict(a.generate(2)) == _as_dict(b.generate(2))


@CONFIGS
def test_per_call_seed_respects_difficulty_and_domain(config):
    sc = ScenarioGenerator(config=config).generate(1, domain="Finance", seed=4)
    assert sc.difficulty == 1 and sc.domain == "Finance"
    assert len(sc.conflicts) == 2
    sc3 = ScenarioGenerator(config=config).generate(3, domain="IT", seed=4)
    assert len(sc3.conflicts) == 6 and sc3.domain == "IT"


# ---------------------------------------------------------------------------
# No ties
# ---------------------------------------------------------------------------

def test_no_ties_over_5000_scenarios():
    """Legacy: every conflict is between different tiers, and the higher tier wins."""
    gen = ScenarioGenerator(seed=2024, config=LEGACY)
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


def test_v2_every_conflict_is_decidable_over_5000_scenarios():
    """v2: equal authority is allowed, but then the dates always differ (never a real tie)."""
    gen = ScenarioGenerator(seed=2024)
    n = 0
    for i in range(5000):
        sc = gen.generate(difficulty=(i % 3) + 1)
        assert len(sc.conflicts) == 2 * ((i % 3) + 1)
        by_id = {ins.id: ins for ins in sc.instructions}
        for c in sc.conflicts:
            a, b = by_id[c.instruction_a_id], by_id[c.instruction_b_id]
            winner, how = pick_winner(a, b, sc)   # raises UnresolvableConflict on a tie
            assert winner.id == c.resolution_id
            if how == "time":
                assert a.effective_date and b.effective_date and a.effective_date != b.effective_date
            assert a.source_priority == SOURCE_PRIORITY[a.source]
            assert c.resolution_id in sc.ground_truth_followed
            n += 1
    assert n > 0


def test_every_template_pair_can_avoid_a_tie():
    """For every group and every possible first source, a different-tier partner exists."""
    from generator import TEMPLATE_GROUPS
    for key, group in TEMPLATE_GROUPS.items():
        t_a, t_b = group["templates"][0], group["templates"][1]
        for s in t_a["sources"]:
            assert any(SOURCE_PRIORITY[o] != SOURCE_PRIORITY[s] for o in t_b["sources"]), (key, s)


# ---------------------------------------------------------------------------
# Prompt order
# ---------------------------------------------------------------------------

def test_format_prompt_does_not_shuffle_callers_list():
    gen = ScenarioGenerator(seed=6, config=LEGACY)
    sc = gen.generate(3)
    original = list(sc.instructions)
    snapshot_ids = [i.id for i in original]
    for _ in range(20):
        gen._format_prompt(sc.business_context, original)
        assert [i.id for i in original] == snapshot_ids


def test_format_prompt_still_randomises_order():
    gen = ScenarioGenerator(seed=6, config=LEGACY)
    sc = gen.generate(3)
    orders = set()
    for _ in range(20):
        prompt = gen._format_prompt(sc.business_context, sc.instructions)
        orders.add(tuple(re.findall(r"\[(INS-[0-9A-F]{6})\]", prompt)))
    assert len(orders) > 1


@CONFIGS
def test_scenario_instruction_order_matches_prompt_order(config):
    gen = ScenarioGenerator(seed=13, config=config)
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


@pytest.mark.parametrize("config,allowed_list,expected_seen", [
    (LEGACY, list(PROMPT_CONFLICT_TYPES), {"direct", "resource", "temporal"}),
    (V2, list(V2_CONFLICT_TYPES), {"direct", "resource", "temporal", "conditional"}),
], ids=["legacy", "v2"])
def test_ground_truth_conflict_types_are_in_prompt_allowed_list(config, allowed_list, expected_seen):
    gen = ScenarioGenerator(seed=77, config=config)
    seen = set()
    for i in range(300):
        sc = gen.generate(difficulty=(i % 3) + 1)
        m = re.search(r'"conflict_type":"([^"]+)"', sc.prompt)
        assert m, "prompt lost its conflict_type field"
        allowed = m.group(1).split("|")
        assert allowed == allowed_list
        for c in sc.conflicts:
            assert c.conflict_type in allowed
            seen.add(c.conflict_type)
    assert seen == expected_seen


def test_prompt_wording_unchanged_for_current_templates():
    sc = ScenarioGenerator(seed=0, config=LEGACY).generate(1)
    assert '"conflict_type":"direct|resource|temporal"' in sc.prompt


# ---------------------------------------------------------------------------
# Existing behaviour still holds
# ---------------------------------------------------------------------------

@CONFIGS
@pytest.mark.parametrize("difficulty", [1, 2, 3])
def test_ground_truth_still_scores_perfect(config, difficulty):
    gen = ScenarioGenerator(seed=31, config=config)
    for _ in range(50):
        sc = gen.generate(difficulty)
        assert score(_ground_truth_answer(sc), sc).composite == 1.0


@CONFIGS
def test_conflict_counts_per_difficulty(config):
    gen = ScenarioGenerator(seed=21, config=config)
    for d, n in ((1, 2), (2, 4), (3, 6)):
        assert len(gen.generate(d).conflicts) == n


# ===========================================================================
# Phase 3 (v2)
# ===========================================================================

def test_default_config_is_the_v2_task():
    gen = ScenarioGenerator(seed=0)
    assert gen.config == GeneratorConfig() and not gen.config.legacy
    sc = gen.generate(2)
    assert sc.generator_version == "v2" and sc.split == "train"


@pytest.mark.parametrize("bad", [
    {"split": "dev"}, {"restrictive_win_rate": 1.5}, {"p_time": -0.1},
    {"p_scoped": 0.5, "p_delegation": 0.4, "p_time": 0.3}, {"distractor_ratio": -1},
])
def test_config_validation(bad):
    with pytest.raises(ValueError):
        GeneratorConfig(**bad)


def test_new_fields_have_defaults_so_old_data_loads():
    ins = Instruction("INS-000001", "t", "CEO Office", 2, "absolute", "k", "v")
    assert ins.action_scope == "all" and ins.effective_date is None and ins.depends_on is None
    sc = Scenario("id", "HR", 1, "c", [ins], [], ["INS-000001"], [], "p")
    assert sc.facts == {} and sc.authority_rules == [] and sc.delegations == [] and sc.split == ""


def test_v2_scenarios_round_trip_through_scenario_io():
    gen = ScenarioGenerator(seed=12)
    for d in range(1, MAX_DIFFICULTY + 1):
        sc = gen.generate(d)
        assert loads_scenario(dumps_scenario(sc)) == sc


# ---- P3-1 balanced winners -------------------------------------------------

def test_restrictive_side_wins_about_half_of_5000_conflicts():
    gen = ScenarioGenerator(seed=314)
    wins = total = 0
    while total < 5000:
        sc = gen.generate(difficulty=1 + total % 3)
        by_id = {ins.id: ins for ins in sc.instructions}
        for c in sc.conflicts:
            a, b = by_id[c.instruction_a_id], by_id[c.instruction_b_id]
            assert {a.polarity, b.polarity} == {"restrictive", "permissive"}
            wins += by_id[c.resolution_id].polarity == "restrictive"
            total += 1
    assert 0.45 <= wins / total <= 0.55, wins / total


def test_both_sides_of_every_topic_come_from_every_tier():
    """Sources are tier ranges, not fixed names: each side appears from senior and junior offices."""
    gen = ScenarioGenerator(seed=5)
    tiers = {}
    for i in range(1500):
        for ins in gen.generate(1 + i % 3).instructions:
            if ins.polarity:
                tiers.setdefault((ins.action_key, ins.polarity), set()).add(ins.source_priority)
    for key in V2_TOPICS:
        for side in ("restrictive", "permissive"):
            assert {1, 2} <= tiers[(key, side)] and max(tiers[(key, side)]) >= 5, (key, side)


# ---- P3-2 distractors -------------------------------------------------------

def _pair_f1(pred, truth):
    if not pred and not truth:
        return 1.0
    tp = len(pred & truth)
    return 0.0 if tp == 0 else 2 * tp / (len(pred) + len(truth))


def test_every_repeated_topic_is_not_a_conflict():
    """'Same topic => conflict' scores below 0.5 F1 on conflict identification."""
    gen = ScenarioGenerator(seed=99)
    f1s = []
    for i in range(600):
        sc = gen.generate(1 + i % 3)
        truth = {frozenset((c.instruction_a_id, c.instruction_b_id)) for c in sc.conflicts}
        groups = {}
        for ins in sc.instructions:
            groups.setdefault(ins.action_key, []).append(ins.id)
        guess = {frozenset(p) for ids in groups.values() for p in itertools.combinations(ids, 2)}
        f1s.append(_pair_f1(guess, truth))
    assert sum(f1s) / len(f1s) < 0.5


def test_conflicts_need_key_value_and_overlapping_scope():
    gen = ScenarioGenerator(seed=17)
    saw_disjoint = saw_agree = False
    for i in range(500):
        sc = gen.generate(1 + i % 3)
        by_id = {ins.id: ins for ins in sc.instructions}
        pairs = {frozenset((c.instruction_a_id, c.instruction_b_id)) for c in sc.conflicts}
        for p in pairs:
            a, b = (by_id[x] for x in p)
            assert a.action_key == b.action_key and a.action_value != b.action_value
            assert scopes_overlap(a.action_scope, b.action_scope)
        for a, b in itertools.combinations(sc.instructions, 2):
            if a.action_key != b.action_key or a.condition or b.condition:
                continue
            if frozenset((a.id, b.id)) in pairs:
                continue
            if a.action_value != b.action_value:
                assert not scopes_overlap(a.action_scope, b.action_scope)
                saw_disjoint = True
            else:
                saw_agree = True
    assert saw_disjoint and saw_agree


def test_scope_is_written_in_the_text():
    gen = ScenarioGenerator(seed=23)
    for _ in range(200):
        for ins in gen.generate(2).instructions:
            if ins.action_scope != "all":
                assert ins.action_scope in ins.text, ins


# ---- P3-3 hidden hierarchy ------------------------------------------------

_TIER_LINE = re.compile(r"^\s*[1-6]\. ", re.M)


def test_hidden_hierarchy_prompt_has_no_tier_list():
    gen = ScenarioGenerator(seed=3)
    for i in range(100):
        sc = gen.generate(1 + i % 3)
        assert not _TIER_LINE.search(sc.prompt)
        assert "Regulatory Affairs, " not in sc.prompt and "lower number" not in sc.prompt.lower()


def test_shown_hierarchy_prompt_prints_the_tiers():
    sc = ScenarioGenerator(seed=3, config=GeneratorConfig(show_hierarchy=True)).generate(2)
    assert len(_TIER_LINE.findall(sc.prompt)) == 6
    for ins in sc.instructions:
        assert ins.title == ins.source  # titles are the office names the tier list uses


def test_titles_are_shown_and_come_from_the_split_pool():
    for split in SPLITS:
        gen = ScenarioGenerator(seed=8, config=GeneratorConfig(split=split))
        for _ in range(100):
            sc = gen.generate(2)
            for ins in sc.instructions:
                assert ins.title in split_pool(OFFICE_TITLES[ins.source], split)
                assert f"From {ins.author} ({ins.title})" in sc.prompt


def test_test_split_titles_never_appear_in_train_prompts():
    test_titles = {t for titles in OFFICE_TITLES.values() for t in split_pool(titles, "test")}
    gen = ScenarioGenerator(seed=4, config=GeneratorConfig(split="train"))
    for i in range(300):
        sc = gen.generate(1 + i % 3)
        shown = {ins.title for ins in sc.instructions}
        assert not shown & test_titles


def test_every_title_belongs_to_one_office():
    titles = [t for ts in OFFICE_TITLES.values() for t in ts]
    assert len(titles) == len(set(titles))
    assert set(OFFICE_TITLES) == set(SOURCE_PRIORITY)


# ---- P3-4 rule flags ------------------------------------------------------

def _all_rules_off(**kw):
    return GeneratorConfig(scoped_authority=False, delegation=False, time_supersession=False,
                           conditionals=False, chains=False, **kw)


def test_each_rule_appears_with_its_flag_on():
    gen = ScenarioGenerator(seed=40)
    seen = {"rule": 0, "delegation": 0, "dates": 0, "conditional": 0, "chain": 0}
    for i in range(300):
        sc = gen.generate(1 + i % 3)
        seen["rule"] += bool(sc.authority_rules)
        seen["delegation"] += bool(sc.delegations)
        seen["dates"] += all(ins.effective_date for ins in sc.instructions)
        seen["conditional"] += any(ins.condition for ins in sc.instructions)
        seen["chain"] += any(ins.depends_on for ins in sc.instructions)
    assert all(v > 0 for v in seen.values()), seen
    assert seen["dates"] == 300


@pytest.mark.parametrize("flag,check", [
    ("scoped_authority", lambda sc: not sc.authority_rules),
    ("delegation", lambda sc: not sc.delegations),
    ("time_supersession", lambda sc: all(i.effective_date is None for i in sc.instructions)),
    ("conditionals", lambda sc: not sc.facts and all(i.condition is None for i in sc.instructions)),
    ("chains", lambda sc: all(i.depends_on is None for i in sc.instructions)),
    ("distractors", lambda sc: len({i.action_key for i in sc.instructions if i.polarity})
                               == len([i for i in sc.instructions if i.polarity]) - len(sc.conflicts)),
])
def test_each_flag_off_removes_its_feature(flag, check):
    cfg = dataclasses.replace(GeneratorConfig(), **{flag: False})
    gen = ScenarioGenerator(seed=41, config=cfg)
    for i in range(150):
        sc = gen.generate(1 + i % 3)
        assert check(sc), flag
        assert score(_ground_truth_answer(sc), sc).composite == 1.0


def test_all_rules_off_is_a_plain_tier_task():
    gen = ScenarioGenerator(seed=42, config=_all_rules_off(show_hierarchy=True))
    for i in range(200):
        sc = gen.generate(1 + i % 3)
        by_id = {ins.id: ins for ins in sc.instructions}
        for c in sc.conflicts:
            a, b = by_id[c.instruction_a_id], by_id[c.instruction_b_id]
            assert by_id[c.resolution_id].source_priority == min(a.source_priority, b.source_priority)
            assert a.source_priority != b.source_priority
        assert score(_ground_truth_answer(sc), sc).composite == 1.0


def test_conditions_and_notices_are_stated_in_the_prompt():
    gen = ScenarioGenerator(seed=43)
    for i in range(200):
        sc = gen.generate(1 + i % 3)
        if sc.facts:
            assert "CURRENT STATUS:" in sc.prompt
        if sc.authority_rules or sc.delegations:
            assert "STANDING NOTICES:" in sc.prompt
        for ins in sc.instructions:
            if ins.condition:
                assert ins.text.startswith("If ")
                assert ins.condition in sc.facts
            if ins.depends_on:
                assert f"[{ins.depends_on}]" in ins.text
            if ins.effective_date:
                assert format_date(ins.effective_date) in sc.prompt


def test_restrictive_win_rate_knob():
    gen = ScenarioGenerator(seed=44, config=GeneratorConfig(restrictive_win_rate=1.0))
    for _ in range(100):
        sc = gen.generate(2)
        by_id = {ins.id: ins for ins in sc.instructions}
        assert all(by_id[c.resolution_id].polarity == "restrictive" for c in sc.conflicts)


# ---- P3-5 difficulty levels and length -------------------------------------

def test_difficulty_is_clamped():
    assert clamp_difficulty(0) == 1 and clamp_difficulty(-3) == 1
    assert clamp_difficulty(99) == MAX_DIFFICULTY
    assert clamp_difficulty("hard") == 2 and clamp_difficulty(None) == 2
    gen = ScenarioGenerator(seed=50)
    assert gen.generate(0).difficulty == 1
    assert gen.generate(99).difficulty == MAX_DIFFICULTY
    assert gen.generate("x").difficulty == 2
    assert ScenarioGenerator(seed=50, config=LEGACY).generate(9).difficulty == 3


@pytest.mark.parametrize("difficulty", range(1, MAX_DIFFICULTY + 1))
def test_every_level_generates_and_scores(difficulty):
    gen = ScenarioGenerator(seed=60 + difficulty)
    for _ in range(40):
        sc = gen.generate(difficulty)
        assert len(sc.conflicts) == DIFFICULTY_LEVELS[difficulty]["conflicts"]
        assert score(_ground_truth_answer(sc), sc).composite == 1.0


def _approx_tokens(text: str) -> int:
    """No tokenizer in the test env: ~4 characters per token is close for English + IDs."""
    return len(text) // 4


# Budget per level in approximate tokens (prompt only; the system prompt adds ~150).
_TOKEN_BUDGET = {1: 1400, 2: 1800, 3: 2400, 4: 2800, 5: 3200}


@pytest.mark.parametrize("difficulty", range(1, MAX_DIFFICULTY + 1))
def test_prompt_length_fits_token_budget(difficulty):
    gen = ScenarioGenerator(seed=70)
    lengths = [_approx_tokens(gen.generate(difficulty).prompt) for _ in range(100)]
    assert max(lengths) <= _TOKEN_BUDGET[difficulty], max(lengths)


def test_harder_levels_are_longer():
    gen = ScenarioGenerator(seed=71)
    means = [sum(len(gen.generate(d).instructions) for _ in range(50)) / 50 for d in range(1, MAX_DIFFICULTY + 1)]
    assert means == sorted(means)


# ---- P3-6 splits -----------------------------------------------------------

def test_split_pool_is_a_disjoint_partition():
    for items in (list(range(5)), list(range(10)), list(range(15)), list(range(3))):
        parts = [split_pool(items, s) for s in SPLITS]
        assert sorted(sum(parts, [])) == items
        assert all(parts)
    with pytest.raises(ValueError):
        split_pool([1, 2, 3], "dev")


def _phrasing_patterns(split):
    pats = []
    for topic in V2_TOPICS.values():
        for side in ("restrictive", "permissive"):
            for text in split_pool(topic[side][1], split):
                body = re.escape(text)
                body = re.sub(r"\\\{[a-z_]+\\\}", ".+?", body)
                pats.append(re.compile(body, re.I))  # re.I: 'If ..., ' lower-cases the first letter
    return pats


def test_phrasings_and_contexts_never_cross_splits():
    patterns = {s: _phrasing_patterns(s) for s in SPLITS}
    for split in SPLITS:
        others = [p for s in SPLITS if s != split for p in patterns[s]]
        companies = split_pool(V2_COMPANIES, split)
        gen = ScenarioGenerator(seed=80, config=GeneratorConfig(split=split))
        for i in range(60):
            sc = gen.generate(1 + i % 3)
            assert sc.split == split
            assert any(sc.business_context.startswith(c) for c in companies)
            for ins in sc.instructions:
                if not ins.polarity:
                    continue  # chain follow-ups
                assert any(p.search(ins.text) for p in patterns[split]), ins.text
                assert not any(p.search(ins.text) for p in others), ins.text
