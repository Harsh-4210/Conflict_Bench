"""Comparability depends on task, scorer, selection and episode protocol, not just data."""

from scripts.provenance import comparison_key, evaluation_provenance, real_data_sha256


def test_provenance_changes_when_scorer_selection_or_protocol_changes(tmp_path):
    scorer = tmp_path / "score.py"
    scorer.write_text("reward = 1\n", encoding="utf-8")

    def build(ids=("A", "B"), budget=3):
        return evaluation_provenance("multistep", "v1", {"score.py": scorer},
                                     {"budget_factor": budget}, ids)

    original = build()
    assert original == build()
    assert len(original["scorer_sha256"]) == 64
    assert original["selection_sha256"] != build(("A",))["selection_sha256"]
    assert original["selection_sha256"] != build(("B", "A"))["selection_sha256"]
    assert original["protocol_sha256"] != build(budget=2)["protocol_sha256"]
    scorer.write_text("reward = 0\n", encoding="utf-8")
    assert original["scorer_sha256"] != build()["scorer_sha256"]


def test_same_data_different_task_or_scorer_cannot_share_a_ranking():
    base = {"data_sha256": "a" * 64, "task": "single_step", "task_version": "v2",
            "scorer_sha256": "b" * 64, "protocol_sha256": "c" * 64,
            "selection_sha256": "d" * 64, "scored_selection_sha256": "d" * 64}
    for field, value in (("task", "multistep"), ("task_version", "v3"),
                         ("scorer_sha256", "e" * 64), ("protocol_sha256", "f" * 64),
                         ("selection_sha256", "0" * 64), ("scored_selection_sha256", "1" * 64)):
        assert comparison_key(base) != comparison_key({**base, field: value})
    assert comparison_key(base) != comparison_key({"data_sha256": "a" * 64})


def test_scored_selection_distinguishes_different_skipped_cases(tmp_path):
    scorer = tmp_path / "scorer.py"
    scorer.write_text("reward = 1", encoding="utf-8")
    left = evaluation_provenance("single_step", "v2", {"scorer.py": scorer}, {},
                                 ["A", "B"], scored_scenario_ids=["A"])
    right = evaluation_provenance("single_step", "v2", {"scorer.py": scorer}, {},
                                  ["A", "B"], scored_scenario_ids=["B"])
    assert left["selection_sha256"] == right["selection_sha256"]
    assert left["scored_selection_sha256"] != right["scored_selection_sha256"]
    assert comparison_key(left) != comparison_key(right)


def test_legacy_multistep_hash_is_read_without_claiming_it_is_a_sha256():
    old = {"data_sha256": "multistep-" + "a" * 64}
    assert real_data_sha256(old) == "a" * 64
    assert comparison_key(old)[1] == "multistep"
    assert comparison_key(old) != comparison_key({"data_sha256": "a" * 64})
