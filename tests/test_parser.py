"""
Parser hardening tests (FIX_BLUEPRINT.md P1-1).
parse_agent_output() and score() must never raise, whatever the model writes.
"""

import json
import random

import pytest

from generator import ScenarioGenerator
from verifier import REQUIRED_KEYS, parse_agent_output, score


@pytest.fixture(scope="module")
def sc():
    return ScenarioGenerator(seed=0).generate(difficulty=1)


def _gt(sc) -> dict:
    return {
        "identified_conflicts": [
            {
                "instruction_a": c.instruction_a_id,
                "instruction_b": c.instruction_b_id,
                "conflict_type": c.conflict_type,
                "resolution": c.resolution_id,
                "reasoning": "higher authority wins",
            }
            for c in sc.conflicts
        ],
        "execution_plan": list(sc.ground_truth_followed),
        "overridden_instructions": list(sc.ground_truth_overridden),
    }


# ---------------------------------------------------------------------------
# Crash probes from FIX_BLUEPRINT.md C.1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw", [
    "[]",
    "[1, 2, 3]",
    "null",
    "5",
    '"just a string"',
    '{"identified_conflicts": [], "execution_plan": null, "overridden_instructions": []}',
    '{"identified_conflicts": [], "execution_plan": 5, "overridden_instructions": []}',
    '{"identified_conflicts": 5, "execution_plan": [], "overridden_instructions": []}',
    '{"identified_conflicts": [{"instruction_a": ["X"], "instruction_b": {"y": 1}, '
    '"resolution": [1]}], "execution_plan": [[1], {"a": 2}, null, true, 1.5], '
    '"overridden_instructions": "INS-1"}',
    '{"identified_conflicts": ["a", 3, null], "execution_plan": {}, "overridden_instructions": 7.5}',
    '{"execution_plan": []}',
    "{" * 5000,
    "[" * 100000,
    '{"a": ' * 3000 + "1" + "}" * 3000,
])
def test_crash_probes_return_a_score(raw, sc):
    result = score(raw, sc)
    assert 0.0 <= result.composite <= 1.0


@pytest.mark.parametrize("raw", [None, 5, 3.2, {"a": 1}, [], [{"role": "assistant"}], [None], [{"content": 5}],
                                 [{"content": [{"type": "text", "text": "{}"}]}], object()])
def test_non_string_inputs_do_not_raise(raw, sc):
    assert 0.0 <= score(raw, sc).composite <= 1.0


def test_top_level_list_is_not_an_answer(sc):
    out = parse_agent_output("[]")
    assert not out.parsed_ok
    assert score("[]", sc).composite == 0.0


# ---------------------------------------------------------------------------
# Extraction order
# ---------------------------------------------------------------------------

def test_two_objects_last_valid_one_wins(sc):
    wrong = {"identified_conflicts": [], "execution_plan": [], "overridden_instructions": []}
    raw = f"First try: {json.dumps(wrong)}\nCorrected: {json.dumps(_gt(sc))}"
    assert score(raw, sc).composite == 1.0


def test_last_object_without_required_keys_is_ignored(sc):
    raw = json.dumps(_gt(sc)) + '\nNote: {"confidence": 0.9}'
    assert score(raw, sc).composite == 1.0


def test_fenced_block_followed_by_prose(sc):
    raw = "Here is my plan:\n```json\n" + json.dumps(_gt(sc), indent=2) + "\n```\nHope that helps {smile}."
    assert score(raw, sc).composite == 1.0


def test_last_fenced_block_wins(sc):
    wrong = {"identified_conflicts": [], "execution_plan": [], "overridden_instructions": []}
    raw = f"```json\n{json.dumps(wrong)}\n```\nOops, fixed:\n```\n{json.dumps(_gt(sc))}\n```"
    assert score(raw, sc).composite == 1.0


def test_brace_inside_reasoning_string(sc):
    answer = _gt(sc)
    answer["identified_conflicts"][0]["reasoning"] = "rule {A} beats {B} because } and {"
    raw = "Thinking about {the hierarchy}... " + json.dumps(answer) + " done }"
    assert score(raw, sc).composite == 1.0


def test_answer_nested_in_wrapper_object(sc):
    raw = json.dumps({"answer": _gt(sc), "meta": {"model": "x"}})
    assert score(raw, sc).composite == 1.0


def test_unclosed_fence_still_parses(sc):
    raw = "```json\n" + json.dumps(_gt(sc))
    assert score(raw, sc).composite == 1.0


def test_trl_message_list(sc):
    raw = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": json.dumps(_gt(sc))}]
    assert score(raw, sc).composite == 1.0


def test_missing_key_is_not_parsed():
    out = parse_agent_output('{"identified_conflicts": [], "execution_plan": []}')
    assert not out.parsed_ok
    assert "required keys" in out.parse_error


# ---------------------------------------------------------------------------
# Normalisation and schema errors
# ---------------------------------------------------------------------------

def test_ids_are_stripped_and_upper_cased(sc):
    answer = _gt(sc)
    answer["execution_plan"] = [f"  {i.lower()} " for i in answer["execution_plan"]]
    answer["overridden_instructions"] = [i.lower() for i in answer["overridden_instructions"]]
    for c in answer["identified_conflicts"]:
        c["instruction_a"] = c["instruction_a"].lower()
        c["resolution"] = " " + c["resolution"].lower()
    result = score(json.dumps(answer), sc)
    assert result.composite == 1.0
    assert result.schema_errors == []


def test_clean_answer_has_no_schema_errors(sc):
    out = parse_agent_output(json.dumps(_gt(sc)))
    assert out.parsed_ok and out.schema_errors == []


def test_bad_types_become_schema_errors():
    out = parse_agent_output(json.dumps({
        "identified_conflicts": [{"instruction_a": ["X"], "instruction_b": "B", "resolution": "B"}, "junk"],
        "execution_plan": ["INS-1", 2, None, True, {"id": 3}],
        "overridden_instructions": 7.5,
    }))
    assert out.parsed_ok
    assert out.execution_plan == ["INS-1", "2"]
    assert out.overridden_instructions == []
    assert out.identified_conflicts == []
    assert len(out.schema_errors) == 6


def test_none_and_string_values_are_accepted():
    out = parse_agent_output(json.dumps({
        "identified_conflicts": None, "execution_plan": "ins-1", "overridden_instructions": None,
    }))
    assert out.parsed_ok
    assert out.execution_plan == ["INS-1"]
    assert out.identified_conflicts == [] and out.overridden_instructions == []
    assert out.schema_errors == []


def test_scalar_conflict_fields_are_coerced():
    out = parse_agent_output(json.dumps({
        "identified_conflicts": [{"instruction_a": 1, "instruction_b": "b", "resolution": 1,
                                  "conflict_type": 3, "reasoning": None}],
        "execution_plan": [], "overridden_instructions": [],
    }))
    entry = out.identified_conflicts[0]
    assert (entry["instruction_a"], entry["instruction_b"], entry["resolution"]) == ("1", "B", "1")
    assert entry["conflict_type"] == "3" and entry["reasoning"] == "None"


# ---------------------------------------------------------------------------
# Fuzz test: 2000 random JSON-ish inputs, none may raise
# ---------------------------------------------------------------------------

def _random_value(rng: random.Random, depth: int = 0):
    kinds = ["str", "int", "float", "none", "bool"]
    if depth < 4:
        kinds += ["list", "dict", "list", "dict"]
    kind = rng.choice(kinds)
    if kind == "str":
        alphabet = "INS-abc0123{}[]\":, \n`"
        return "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 12)))
    if kind == "int":
        return rng.randint(-10**6, 10**6)
    if kind == "float":
        return rng.uniform(-1e6, 1e6)
    if kind == "none":
        return None
    if kind == "bool":
        return rng.random() < 0.5
    if kind == "list":
        return [_random_value(rng, depth + 1) for _ in range(rng.randint(0, 5))]
    keys = list(REQUIRED_KEYS) + ["instruction_a", "instruction_b", "resolution", "x", ""]
    return {rng.choice(keys): _random_value(rng, depth + 1) for _ in range(rng.randint(0, 5))}


def _random_input(rng: random.Random, ids):
    if rng.random() < 0.5:
        # Answer-shaped object whose fields are random (sometimes real IDs).
        pool = ids + ["INS-FAKE", "ins-" + ids[0][4:].lower()]
        obj = {k: _random_value(rng) for k in REQUIRED_KEYS}
        if rng.random() < 0.5:
            obj["execution_plan"] = rng.sample(pool, rng.randint(0, len(pool)))
        if rng.random() < 0.3:
            del obj[rng.choice(list(REQUIRED_KEYS))]
    else:
        obj = _random_value(rng)
    text = json.dumps(obj)

    mutation = rng.random()
    if mutation < 0.15:
        text = text[: rng.randint(0, len(text))]                       # truncated
    elif mutation < 0.30:
        i = rng.randint(0, len(text))
        text = text[:i] + rng.choice("{}[]\",:`") + text[i:]            # stray character
    elif mutation < 0.45:
        text = f"```json\n{text}\n```\ntrailing {{prose}}"              # fenced
    elif mutation < 0.60:
        text = f"Reasoning {{x}}: {json.dumps(_random_value(rng))} then {text} {text[:5]}"
    return text


def test_fuzz_2000_random_inputs_never_raise():
    rng = random.Random(20261031)
    gen = ScenarioGenerator(seed=7)
    scenarios = [gen.generate(difficulty=d) for d in (1, 2, 3)]
    for i in range(2000):
        sc = scenarios[i % 3]
        raw = _random_input(rng, [ins.id for ins in sc.instructions])
        out = parse_agent_output(raw)
        assert isinstance(out.execution_plan, list)
        assert all(isinstance(x, str) for x in out.execution_plan + out.overridden_instructions)
        assert all(isinstance(c, dict) for c in out.identified_conflicts)
        result = score(raw, sc)
        assert 0.0 <= result.composite <= 1.0, raw
