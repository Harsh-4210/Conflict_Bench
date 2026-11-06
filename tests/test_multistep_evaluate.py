"""Tests for scripts/evaluate_multistep.py. No model, no network."""

import json

import pytest

from scripts import evaluate_multistep as ev
from scripts import make_leaderboard


def _run(tmp_path, *argv):
    return ev.main(["--out-dir", str(tmp_path), *argv])


def test_all_strategies_write_summaries(tmp_path):
    out = _run(tmp_path, "--generate", "9", "--seed", "4")
    assert set(out) == set(ev.STRATEGIES)
    for name, summary in out.items():
        path = tmp_path / f"multistep_{name}" / "summary.json"
        assert json.loads(path.read_text(encoding="utf-8")) == json.loads(json.dumps(summary))
        rows = (tmp_path / f"multistep_{name}" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(rows) == 9 and summary["n"] == 9
        assert summary["task"] == "multistep" and summary["strategy"] == name
    assert out["oracle"]["composite"]["mean"] == 1.0
    assert out["do_nothing"]["composite"]["mean"] == 0.0
    assert out["oracle"]["data_file_sha256"] == out["greedy_latest"]["data_file_sha256"]


def test_runs_are_reproducible(tmp_path):
    a = _run(tmp_path / "a", "--strategy", "apply_then_fix", "--generate", "9", "--seed", "4")
    b = _run(tmp_path / "b", "--strategy", "apply_then_fix", "--generate", "9", "--seed", "4")
    for key in ("composite", "per_difficulty", "per_rubric", "episode_stats", "data_file_sha256"):
        assert a["apply_then_fix"][key] == b["apply_then_fix"][key]
    rows_a = (tmp_path / "a" / "multistep_apply_then_fix" / "episodes.jsonl").read_text(encoding="utf-8")
    rows_b = (tmp_path / "b" / "multistep_apply_then_fix" / "episodes.jsonl").read_text(encoding="utf-8")
    assert rows_a == rows_b


def test_seed_changes_the_episodes(tmp_path):
    a = _run(tmp_path / "a", "--strategy", "greedy_latest", "--generate", "9", "--seed", "1")
    b = _run(tmp_path / "b", "--strategy", "greedy_latest", "--generate", "9", "--seed", "2")
    assert a["greedy_latest"]["data_file_sha256"] != b["greedy_latest"]["data_file_sha256"]


def test_runs_on_a_data_file_and_limits_n(tmp_path):
    out = _run(tmp_path, "--strategy", "oracle", "--n", "5")
    s = out["oracle"]
    assert s["n"] == 5 and s["n_scenarios_in_data"] >= 5
    assert s["data_path"] == "data/test_v1.jsonl"
    assert s["composite"]["mean"] == 1.0


def test_summary_keeps_the_leaderboard_script_working_and_apart(tmp_path):
    _run(tmp_path, "--strategy", "oracle", "--generate", "6", "--seed", "1")
    text = make_leaderboard.main(["--results-dir", str(tmp_path)])
    assert "multistep_oracle" in text
    assert "Task: `multistep`" in text


def test_wait_until_end_applies_nothing_until_the_final_turn(monkeypatch):
    env = ev.MultiStepConflictEnv(max_per_turn=1)
    env.reset(seed=11, difficulty=2)
    monkeypatch.setattr(env, "ground_truth", lambda: pytest.fail("must not see future ground truth"))
    actions = ev.wait_until_end(env)
    while env.more_messages:
        action = next(actions)
        assert action == {"tool": "wait"}
        assert env.applied_ids() == []
        env.step(action)
    expected = [iid for iid in env.arrived_ids if iid in env.known_followed()]
    applied = []
    for action in actions:
        if action["tool"] == "apply":
            applied.append(action["id"])
        env.step(action)
    assert applied == expected


def test_wait_until_end_records_v1_deferral_and_real_hash(tmp_path):
    summary = _run(tmp_path, "--strategy", "wait_until_end", "--generate", "9")["wait_until_end"]
    assert summary["composite"]["mean"] == 1.0
    assert summary["episode_stats"]["mean_reverts"] == 0.0
    assert summary["task_version"] == "inbox-v1"
    assert "resolver.py" in summary["scorer_files"]
    assert summary["data_sha256"] == summary["data_file_sha256"]
    assert len(summary["data_sha256"]) == 64
    assert len(summary["protocol_sha256"]) == 64


def test_step_budget_is_never_the_reason_a_baseline_stops(tmp_path):
    out = _run(tmp_path, "--generate", "12", "--seed", "3")
    for name, s in out.items():
        assert s["episode_stats"]["share_ended_by_budget"] == 0.0, name


@pytest.mark.parametrize("name", sorted(ev.STRATEGIES))
def test_strategies_never_make_invalid_calls(name, tmp_path):
    out = _run(tmp_path, "--strategy", name, "--generate", "9", "--seed", "6")
    assert out[name]["per_rubric"]["wasted_steps"] == 0.0
