"""Training commands must be usable and inspectable without GPU dependencies."""

import builtins
import json
import subprocess
import sys
from pathlib import Path

import pytest

from training.train_grpo import main

ROOT = Path(__file__).resolve().parent.parent


def run_cli(*args):
    return subprocess.run(
        [sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=30,
    )


def test_check_config_prints_resolved_overrides_without_starting_training(tmp_path):
    output = tmp_path / "checkpoints"
    result = run_cli(
        "-m", "training.train_grpo", "--config", "configs/grpo_smoke.yaml",
        "--check-config", "--max-steps", "3", "--output-dir", str(output),
        "--set", "learning_rate=3e-6",
    )
    assert result.returncode == 0, result.stderr
    cfg = json.loads(result.stdout)
    assert cfg["max_steps"] == 3
    assert cfg["learning_rate"] == 3e-6
    assert cfg["output_dir"] == str(output)
    assert not output.exists()


def test_check_config_rejects_invalid_values_before_model_imports():
    result = run_cli(
        "-m", "training.train_grpo", "--config", "configs/grpo_smoke.yaml",
        "--check-config", "--set", "learning_rate=-1",
    )
    assert result.returncode == 2
    assert "learning_rate" in result.stderr
    assert "Invalid training config" in result.stderr


def test_check_config_and_dry_run_cannot_be_combined():
    result = run_cli(
        "-m", "training.train_grpo", "--config", "configs/grpo_smoke.yaml",
        "--check-config", "--dry-run",
    )
    assert result.returncode == 2
    assert "not allowed with argument" in result.stderr


def test_production_command_help_needs_no_gpu_packages():
    result = run_cli("train_grpo_prod.py", "--help")
    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout


@pytest.mark.parametrize("config_args,model", [
    ([], "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"),
    (["--config", "configs/grpo_smoke.yaml"], "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"),
    (["--config=configs/grpo_smoke.yaml"], "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"),
])
def test_production_command_checks_default_or_explicit_config(config_args, model):
    result = run_cli("train_grpo_prod.py", *config_args, "--check-config", "--max-steps", "2")
    assert result.returncode == 0, result.stderr
    cfg = json.loads(result.stdout)
    assert cfg["model"] == model
    assert cfg["max_steps"] == 2
    assert cfg["report_to"] == "none"
    assert cfg["hub_repo"] is None
    if not config_args:
        assert cfg["train_size"] == 120 and cfg["val_size"] == 20
        assert cfg["train_difficulty_mix"] == {"1": 1.0}
        assert cfg["val_difficulty_mix"] == {"1": 1.0}
        assert cfg["per_device_train_batch_size"] == 1
        assert cfg["gradient_accumulation_steps"] == 4


@pytest.mark.parametrize("args", [["--eval", "adapter"], ["--eval=adapter"]])
def test_production_command_redirects_legacy_evaluation(args):
    result = run_cli("train_grpo_prod.py", *args)
    assert result.returncode == 2
    assert "scripts/evaluate.py" in result.stderr
    assert "--adapter" in result.stderr


def test_dry_run_explains_missing_tokenizer_dependency(monkeypatch, caplog):
    original_import = builtins.__import__

    def without_transformers(name, *args, **kwargs):
        if name == "transformers":
            raise ModuleNotFoundError("No module named 'transformers'", name="transformers")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_transformers)
    result = main(["--config", str(ROOT / "configs/grpo_smoke.yaml"), "--dry-run"])
    assert result == 2
    assert "transformers" in caplog.text
    assert "--check-config" in caplog.text
