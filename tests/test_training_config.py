"""Training config: YAML loading, validation, derived values, GRPOConfig kwargs, legacy shims. No GPU needed."""

import importlib
import sys
import warnings
from pathlib import Path

import pytest
import yaml

from training.config import ConfigError, TrainConfig, allocate_counts, config_from_dict, load_config
from training.train_grpo import BEST_METRIC, build_grpo_kwargs, build_reward_functions, find_resume_checkpoint
from training.train_grpo import filter_supported_kwargs

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = sorted((ROOT / "configs").glob("grpo_*.yaml"))


def test_shipped_configs_exist():
    assert {p.name for p in CONFIGS} >= {"grpo_t4.yaml", "grpo_a10g.yaml", "grpo_smoke.yaml"}


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_shipped_configs_load_and_are_safe_by_default(path):
    cfg = load_config(path)
    assert cfg.hub_repo is None                      # never pushes unless the user asks
    assert cfg.report_to == "none"                   # no wandb login needed
    assert cfg.effective_batch_size % cfg.num_generations == 0
    assert cfg.per_device_eval_batch_size == cfg.num_generations
    assert cfg.max_prompt_tokens + cfg.max_completion_tokens <= cfg.max_seq_length
    assert cfg.save_steps % cfg.eval_steps == 0
    assert cfg.run_name == path.stem


def test_smoke_config_is_tiny():
    cfg = load_config(ROOT / "configs" / "grpo_smoke.yaml")
    assert "0.5B" in cfg.model
    assert cfg.max_steps == 10
    assert cfg.train_size <= 50


def test_defaults_are_valid():
    TrainConfig().validate()


def test_string_floats_from_yaml_are_coerced():
    # PyYAML reads 3e-6 (no dot) as a string; the loader must turn it into a number.
    assert isinstance(yaml.safe_load("lr: 3e-6")["lr"], str)
    cfg = config_from_dict({"learning_rate": "3e-6", "beta": 0})
    assert cfg.learning_rate == pytest.approx(3e-6)
    assert cfg.beta == 0.0 and isinstance(cfg.beta, float)


def test_difficulty_mix_keys_become_ints():
    cfg = config_from_dict({"train_difficulty_mix": {"1": 0.5, "2": 0.5}})
    assert cfg.train_difficulty_mix == {1: 0.5, 2: 0.5}


def test_unknown_key_rejected():
    with pytest.raises(ConfigError, match="Unknown config key"):
        config_from_dict({"learning_rte": 1e-5})


def test_missing_file_rejected(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_batch_times_accum_must_divide_by_num_generations():
    with pytest.raises(ConfigError, match="divisible by num_generations"):
        config_from_dict({"num_generations": 4, "per_device_train_batch_size": 3, "gradient_accumulation_steps": 1})
    # 2 x 2 = 4 is fine even though the batch alone is not a multiple of 4
    config_from_dict({"num_generations": 4, "per_device_train_batch_size": 2, "gradient_accumulation_steps": 2})


@pytest.mark.parametrize("bad", [
    {"num_generations": 1},
    {"learning_rate": 0},
    {"learning_rate": -1e-5},
    {"top_p": 0},
    {"top_p": 1.5},
    {"temperature": 0},
    {"beta": -0.1},
    {"max_steps": 0},
    {"epochs": 0, "max_steps": -1},
    {"train_size": 0},
    {"val_size": 0},
    {"max_prompt_tokens": 2000, "max_completion_tokens": 2000, "max_seq_length": 2048},
    {"eval_steps": 25, "save_steps": 30},
    {"hub_repo": "not-a-repo-id"},
    {"train_difficulty_mix": {}},
    {"train_difficulty_mix": {1: 0, 2: 0}},
    {"train_difficulty_mix": {0: 1.0}},
    {"val_difficulty_mix": {1: -1.0}},
    {"lora_r": 0},
    {"lora_target_modules": []},
    {"load_in_4bit": "yes"},
    {"run_name": "has spaces"},
])
def test_invalid_values_rejected(bad):
    with pytest.raises(ConfigError):
        config_from_dict(bad)


def test_all_problems_are_reported_together():
    with pytest.raises(ConfigError) as exc:
        config_from_dict({"learning_rate": -1, "num_generations": 1, "top_p": 2})
    msg = str(exc.value)
    assert "learning_rate" in msg and "num_generations" in msg and "top_p" in msg


def test_overrides_apply_and_are_validated():
    base = {"max_steps": -1}
    assert config_from_dict(base, {"max_steps": 10}).max_steps == 10
    with pytest.raises(ConfigError):
        config_from_dict(base, {"max_steps": 0})


def test_derived_values():
    cfg = config_from_dict({"num_generations": 4, "per_device_train_batch_size": 4, "gradient_accumulation_steps": 3})
    assert cfg.effective_batch_size == 12
    assert cfg.prompts_per_step == 3
    assert cfg.per_device_eval_batch_size == 4


def test_yaml_roundtrip(tmp_path):
    cfg = load_config(ROOT / "configs" / "grpo_t4.yaml")
    out = tmp_path / "train_config.yaml"
    out.write_text(cfg.to_yaml(), encoding="utf-8")
    assert load_config(out) == cfg


def test_allocate_counts_sums_to_size():
    for size in (1, 7, 20, 200, 601):
        counts = allocate_counts(size, {1: 0.6, 2: 0.3, 3: 0.1})
        assert sum(counts.values()) == size
    assert allocate_counts(10, {1: 0.5, 2: 0.5}) == {1: 5, 2: 5}
    assert allocate_counts(5, {1: 1.0}) == {1: 5}
    assert 3 not in allocate_counts(4, {1: 1.0, 3: 0.0})


# ---- GRPOConfig kwargs ----------------------------------------------------------------------------------------

def test_grpo_kwargs_follow_the_config():
    cfg = load_config(ROOT / "configs" / "grpo_a10g.yaml")
    _, weights, _ = build_reward_functions()
    kw = build_grpo_kwargs(cfg, bf16=True, reward_weights=weights)
    assert kw["per_device_eval_batch_size"] == cfg.num_generations
    assert (kw["per_device_train_batch_size"] * kw["gradient_accumulation_steps"]) % kw["num_generations"] == 0
    assert kw["max_completion_length"] == cfg.max_completion_tokens
    assert kw["load_best_model_at_end"] is True
    assert kw["metric_for_best_model"] == BEST_METRIC == "eval_reward"
    assert kw["greater_is_better"] is True
    assert kw["eval_strategy"] == kw["save_strategy"] == "steps"
    assert kw["bf16"] is True and kw["fp16"] is False
    assert kw["report_to"] == "none"
    assert kw["remove_unused_columns"] is False
    assert kw["reward_weights"] == [1.0, 0.0, 0.0, 0.0]
    assert kw["beta"] == cfg.beta and kw["seed"] == cfg.seed


def test_grpo_kwargs_fp16_when_no_bf16():
    cfg = load_config(ROOT / "configs" / "grpo_t4.yaml")
    kw = build_grpo_kwargs(cfg, bf16=False, reward_weights=[1.0])
    assert kw["fp16"] is True and kw["bf16"] is False


def test_unsupported_optional_kwargs_dropped_but_critical_ones_fail():
    import dataclasses

    @dataclasses.dataclass
    class OldConfig:
        output_dir: str = ""
        learning_rate: float = 0.0

    with pytest.raises(RuntimeError, match="reward_weights"):
        filter_supported_kwargs(OldConfig, {"output_dir": "x", "learning_rate": 1.0, "reward_weights": [1.0]})

    @dataclasses.dataclass
    class NewConfig(OldConfig):
        reward_weights: list = None

    out = filter_supported_kwargs(NewConfig, {"output_dir": "x", "reward_weights": [1.0], "max_prompt_length": 5})
    assert out == {"output_dir": "x", "reward_weights": [1.0]}


# ---- resume -----------------------------------------------------------------------------------------------------

def test_find_resume_checkpoint(tmp_path):
    assert find_resume_checkpoint(str(tmp_path / "missing")) is None
    assert find_resume_checkpoint(str(tmp_path)) is None
    for step, complete in ((25, True), (50, True), (75, False)):
        d = tmp_path / f"checkpoint-{step}"
        d.mkdir()
        if complete:
            (d / "trainer_state.json").write_text("{}")
    (tmp_path / "final").mkdir()
    assert Path(find_resume_checkpoint(str(tmp_path))).name == "checkpoint-50"   # 75 has no trainer_state.json


# ---- legacy entry points -----------------------------------------------------------------------------------------

def test_root_train_grpo_shim_is_importable_without_gpu_packages():
    sys.modules.pop("train_grpo", None)
    shim = importlib.import_module("train_grpo")
    assert callable(shim.main)
    assert Path(shim.DEFAULT_CONFIG).is_file()


def test_root_shim_forwards_and_injects_default_config(monkeypatch, capsys):
    import train_grpo as shim

    seen = []
    monkeypatch.setattr(shim, "_main", lambda argv: seen.append(list(argv)) or 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert shim.main(["--max-steps", "10"]) == 0
        assert shim.main(["--config", "configs/grpo_smoke.yaml"]) == 0
    assert seen[0][:2] == ["--config", shim.DEFAULT_CONFIG] and seen[0][2:] == ["--max-steps", "10"]
    assert seen[1] == ["--config", "configs/grpo_smoke.yaml"]
    assert "deprecated" in capsys.readouterr().err


def test_root_diagnose_tokens_shim_imports():
    sys.modules.pop("diagnose_tokens", None)
    shim = importlib.import_module("diagnose_tokens")
    assert callable(shim.main)


def test_training_requirements_are_all_pinned():
    lines = [ln.strip() for ln in (ROOT / "requirements-train.txt").read_text(encoding="utf-8").splitlines()]
    reqs = [ln for ln in lines if ln and not ln.startswith("#")]
    assert reqs, "requirements-train.txt is empty"
    assert all("==" in ln and "@" not in ln and "git+" not in ln for ln in reqs), reqs
    names = {ln.split("==")[0].lower() for ln in reqs}
    assert {"unsloth", "unsloth_zoo", "trl", "transformers", "peft", "datasets", "pyyaml"} <= names


def test_diagnose_tokens_report_flags_unusable_budgets():
    from scripts.diagnose_tokens import report

    cfg = config_from_dict({"max_prompt_tokens": 100, "max_completion_tokens": 50, "max_seq_length": 256})
    lines = report(cfg, {1: [50] * 10, 3: [300] * 10})
    assert not any(line.startswith("PROBLEM") and "difficulty 1" in line for line in lines)
    assert any(line.startswith("PROBLEM") and "difficulty 3" in line for line in lines)
