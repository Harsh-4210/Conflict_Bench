"""Tests for scripts/export_hf_dataset.py: determinism, no answer leakage, manifest, source-hash check."""

import dataclasses
import hashlib
import json
import re
import shutil
from pathlib import Path

import pytest

from generator import ConflictPair, Instruction, Scenario
from prompts import SYSTEM_PROMPT, build_messages
from scenario_io import load_scenarios
from scripts import export_hf_dataset as exp
from verifier import RUBRIC_WEIGHTS

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "test_v1.jsonl"
ID_RE = re.compile(r"INS-[0-9A-F]{6}")

# Every field name of the scenario dataclasses that is not a public field is an answer or a hint.
SECRET_KEYS = (
    {f.name for f in dataclasses.fields(Scenario)}
    | {f.name for f in dataclasses.fields(Instruction)}
    | {f.name for f in dataclasses.fields(ConflictPair)}
) - set(exp.PUBLIC_FIELDS)


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _walk_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _walk_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_keys(v)


def leak_findings(public: dict, scenario: Scenario) -> list:
    """Everything a cheater could read in one public record that gives the answer away. [] = clean."""
    found = []
    extra = set(public) - set(exp.PUBLIC_FIELDS)
    if extra:
        found.append(f"extra top-level keys {sorted(extra)}")
    bad = set(_walk_keys(public)) & SECRET_KEYS
    if bad:
        found.append(f"answer-field keys {sorted(bad)}")
    texts = [m["content"] for m in public.get("messages", [])]
    blob = json.dumps(public, ensure_ascii=False)
    # No ground-truth explanation text, and no list of instruction IDs (the answer format is lists of IDs).
    for c in scenario.conflicts:
        if c.explanation and c.explanation in blob:
            found.append("conflict explanation text")
    for t in texts:
        if re.search(r'"INS-[0-9A-F]{6}"', t) or re.search(r"INS-[0-9A-F]{6}\s*,\s*INS-[0-9A-F]{6}", t):
            found.append("quoted or comma-separated ID list in a message")
    # The system message names no instruction at all.
    if public.get("messages") and ID_RE.search(public["messages"][0]["content"]):
        found.append("instruction ID in the system message")
    # No ID list outside the prompt, e.g. in a metadata value.
    meta = {k: v for k, v in public.items() if k != "messages"}
    if ID_RE.search(json.dumps(meta)):
        found.append("instruction ID in metadata")
    return found


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("export") / "ds"
    manifest = exp.export(DATA, out)
    return out, manifest


@pytest.fixture(scope="module")
def scenarios():
    return load_scenarios(DATA)


def test_layout_and_counts(built, scenarios):
    out, manifest = built
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()) == [
        "README.md", "manifest.json", "private/test_full.jsonl", "public/test.jsonl"]
    assert manifest["n_scenarios"] == len(scenarios) == 600
    public = _read_jsonl(out / "public" / "test.jsonl")
    private = _read_jsonl(out / "private" / "test_full.jsonl")
    assert [r["scenario_id"] for r in public] == [s.scenario_id for s in scenarios]
    assert [r["scenario_id"] for r in private] == [s.scenario_id for s in scenarios]


def test_public_records_are_exactly_the_model_input(built, scenarios):
    out, _ = built
    for rec, sc in zip(_read_jsonl(out / "public" / "test.jsonl"), scenarios):
        assert list(rec) == list(exp.PUBLIC_FIELDS)
        assert rec["messages"] == build_messages(sc)
        assert (rec["difficulty"], rec["domain"], rec["split"], rec["generator_version"]) == (
            sc.difficulty, sc.domain, sc.split, sc.generator_version)


def test_private_file_round_trips_the_full_scenarios(built, scenarios):
    out, _ = built
    again = load_scenarios(out / "private" / "test_full.jsonl")
    assert again == scenarios
    assert all(s.ground_truth_followed and s.conflicts for s in again)  # the answers really are there


def test_no_answer_leaks_into_the_public_file(built, scenarios):
    out, _ = built
    public_text = (out / "public" / "test.jsonl").read_text(encoding="utf-8")
    for rec, sc in zip(_read_jsonl(out / "public" / "test.jsonl"), scenarios):
        assert leak_findings(rec, sc) == [], sc.scenario_id
    # The raw bytes contain no secret field name used as a JSON key.
    for key in SECRET_KEYS:
        assert f'"{key}":' not in public_text, key


def test_ids_to_roles_cannot_be_read_out(built, scenarios):
    """A cheater with only public/test.jsonl: can position or ID order reveal followed vs overridden?"""
    out, _ = built
    n_sorted = 0
    first_half_followed, overall_followed = [], []
    for rec, sc in zip(_read_jsonl(out / "public" / "test.jsonl"), scenarios):
        user = rec["messages"][1]["content"]
        order = re.findall(r"^\[(INS-[0-9A-F]{6})\] From", user, re.M)
        assert sorted(order) == sorted(i.id for i in sc.instructions)  # IDs are only the labels of the task
        n_sorted += order == sorted(order)
        followed = set(sc.ground_truth_followed)
        half = len(order) // 2
        first_half_followed.append(sum(i in followed for i in order[:half]) / half)
        overall_followed.append(sum(i in followed for i in order) / len(order))
    assert n_sorted < len(scenarios) * 0.05  # the listing order is not the (sorted) ground-truth order
    gap = abs(sum(first_half_followed) / len(scenarios) - sum(overall_followed) / len(scenarios))
    assert gap < 0.03  # being listed early says nothing about being followed


def test_leak_checker_catches_leaks(scenarios):
    """Negative control: the checker above is not vacuous."""
    sc = scenarios[0]
    clean = exp.public_record(sc)
    assert leak_findings(clean, sc) == []
    ids = [i.id for i in sc.instructions]
    leaky_variants = [
        {**clean, "ground_truth_followed": sc.ground_truth_followed},
        {**clean, "conflicts": [dataclasses.asdict(c) for c in sc.conflicts]},
        {**clean, "hint": ids[0]},
        {**clean, "messages": clean["messages"] + [{"role": "assistant", "content": json.dumps(sc.ground_truth_followed)}]},
        {**clean, "messages": [clean["messages"][0], {"role": "user", "content": clean["messages"][1]["content"]
                                                       + "\n" + sc.conflicts[0].explanation}]},
        {**clean, "messages": [{"role": "system", "content": SYSTEM_PROMPT + " " + ids[0]}, clean["messages"][1]]},
    ]
    for leaky in leaky_variants:
        assert leak_findings(leaky, sc), list(leaky)


def test_export_is_byte_identical(built, tmp_path):
    out1, _ = built
    out2 = tmp_path / "again"
    exp.export(DATA, out2)
    for rel in ("README.md", "manifest.json", "private/test_full.jsonl", "public/test.jsonl"):
        assert (out1 / rel).read_bytes() == (out2 / rel).read_bytes(), rel


def test_cli_matches_function(built, tmp_path, capsys):
    out1, manifest = built
    out2 = tmp_path / "cli"
    assert exp.main(["--data", str(DATA), "--out", str(out2)]) == manifest
    assert (out2 / "public" / "test.jsonl").read_bytes() == (out1 / "public" / "test.jsonl").read_bytes()
    assert "Nothing was uploaded" in capsys.readouterr().out


def test_manifest_hashes_match_the_files(built):
    out, manifest = built
    assert set(manifest["files"]) == {"README.md", "private/test_full.jsonl", "public/test.jsonl"}
    for rel, info in manifest["files"].items():
        data = (out / rel).read_bytes()
        assert info["sha256"] == hashlib.sha256(data).hexdigest(), rel
        assert info["bytes"] == len(data), rel
    on_disk = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk == manifest
    expected = (ROOT / "data" / "test_v1.sha256").read_text().split()[0]
    assert manifest["source"]["sha256"] == expected
    assert manifest["source"]["hash_file_checked"] is True


def test_source_hash_mismatch_is_refused(tmp_path):
    data = tmp_path / "set.jsonl"
    shutil.copy(DATA, data)
    (tmp_path / "set.sha256").write_text("0" * 64 + "  set.jsonl\n")
    with pytest.raises(SystemExit, match="mismatch"):
        exp.export(data, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_no_hash_file_is_allowed_and_noted(tmp_path):
    data = tmp_path / "set.jsonl"
    data.write_text("".join(DATA.read_text(encoding="utf-8").splitlines(keepends=True)[:6]), encoding="utf-8")
    manifest = exp.export(data, tmp_path / "out")
    assert manifest["source"]["hash_file_checked"] is False
    assert "not checked" in (tmp_path / "out" / "README.md").read_text(encoding="utf-8")


def test_reexport_clears_stale_files_only_in_owned_paths(tmp_path):
    out = tmp_path / "out"
    exp.export(DATA, out)
    (out / "public" / "stale.txt").write_text("x")
    (out / "notes.txt").write_text("keep me")
    exp.export(DATA, out)
    assert not (out / "public" / "stale.txt").exists()
    assert (out / "notes.txt").read_text() == "keep me"


def test_dataset_card(built):
    out, manifest = built
    card = (out / "README.md").read_text(encoding="utf-8")
    assert card.startswith("---\n")
    front = card.split("---\n")[1]
    assert "license: mit" in front
    assert "task_categories:" in front and "size_categories:" in front
    assert "path: public/test.jsonl" in front
    assert manifest["source"]["sha256"] in card
    # Scoring formula comes from verifier.py, not typed by hand.
    for name, weight in RUBRIC_WEIGHTS.items():
        assert f"{weight:.2f} x {name}" in card
    # The answer schema is the one in the system prompt.
    assert exp.ANSWER_SCHEMA in SYSTEM_PROMPT
    assert exp.ANSWER_SCHEMA in card
    assert "How to submit" in card and "Caveats" in card
    # Counts come from the data.
    assert "| 1 | 200 |" in card and "600 synthetic" in card
    # No result figures are typed into the card.
    assert "leaderboard" in card.lower()
    assert not re.search(r"\b0\.\d{2,}\b(?! x)", card.replace("0.45 x", "").replace("0.30 x", "").replace("0.25 x", ""))
