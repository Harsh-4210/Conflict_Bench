"""Tests for scripts/make_leaderboard.py."""

import json

from scripts import make_leaderboard


def _write(root, name, sha, mean, ts="2026-01-01T00:00:00+00:00"):
    d = root / name
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps({
        "run_name": name, "backend": "strategy", "strategy": name, "model": None, "adapter": None,
        "data_path": f"results/data/{sha[:4]}.jsonl", "data_sha256": sha, "n_scenarios_in_data": 30,
        "git_commit": "abcdef1234567", "git_dirty": False, "timestamp_utc": ts,
        "composite": {"mean": mean, "ci95_low": mean - 0.01, "ci95_high": mean + 0.01},
        "per_difficulty": {"1": {"mean": mean, "n": 10}, "2": {"mean": mean, "n": 10}, "3": {"mean": mean, "n": 10}},
        "per_rubric": {}, "parse_failure_rate": 0.0, "n": 30, "skipped_too_long": 0,
    }))


def test_groups_by_hash_and_sorts(tmp_path):
    _write(tmp_path, "low", "a" * 64, 0.2)
    _write(tmp_path, "high", "a" * 64, 0.9)
    _write(tmp_path, "other", "b" * 64, 0.5, ts="2026-02-01T00:00:00+00:00")
    text = make_leaderboard.main(["--results-dir", str(tmp_path)])
    assert (tmp_path / "leaderboard.md").read_text(encoding="utf-8") == text
    assert text.count("## Data set") == 2
    assert text.index("bbbbbbbbbbbb") < text.index("aaaaaaaaaaaa")  # newest data set first
    a_section = text.split("## Data set `aaaaaaaaaaaa`")[1]
    assert a_section.index("strategy: high") < a_section.index("strategy: low")
    assert "strategy: other" not in a_section
    assert "| 1 | strategy: high" in a_section and "0.890 - 0.910" in a_section
    assert "abcdef1" in text


def test_empty_results(tmp_path):
    assert "No results yet" in make_leaderboard.main(["--results-dir", str(tmp_path)])


def test_regenerating_is_stable(tmp_path):
    _write(tmp_path, "r", "c" * 64, 0.5)
    make_leaderboard.main(["--results-dir", str(tmp_path)])
    first = (tmp_path / "leaderboard.md").read_text(encoding="utf-8")
    make_leaderboard.main(["--results-dir", str(tmp_path)])
    assert (tmp_path / "leaderboard.md").read_text(encoding="utf-8") == first
